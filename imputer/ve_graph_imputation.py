import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import time
import json
from typing import List, Tuple, Dict, Optional
import warnings
import math
warnings.filterwarnings('ignore')

# Graph imports
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Domain-specific model imports
from domain_specific_model import (
    convert_training_data_for_pgmpy,
    create_bn_structure_from_adjacency,
    learn_domain_specific_model,
    evaluate_domain_specific_model,
    extract_adjacency_from_embeddings
)

# CONFIGURATION
GRAPH_SIZES = [5, 10, 15]
TRAINING_SIZES = [100, 500, 750, 1000, 1500, 2000]
TEST_SIZE = 200
OBS_RATIO = 0.5  # Fixed 30% observation
QUERY_RATIO = 0.4  # 30% of unobserved nodes designated as "query"
EDGE_PROB = 0.35
N_STATES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Create directories - use relative paths for local development
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, 'outputs', 'GRAPH_EXP'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Base directory: {BASE_DIR}")

# ================================= ARCHITECTURE =================================

class NormLayer(nn.Module):
    """Layer normalization."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    """Feed forward network."""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.layer_1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.layer_2(self.dropout(F.relu(self.layer_1(x))))

class Positional_Encoder(nn.Module):
    """Positional encoder adapted for graph imputation - 2 streams only."""
    
    def __init__(self, n_nodes, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Node embeddings (analogous to annotator embeddings)
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        # Dummy question embedding (all nodes same "type")  
        self.question_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
        torch.nn.init.kaiming_normal_(self.node_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, embeddings, dimensions):
        """
        Create encoded representations for feature and parameter streams.
        
        Args:
            inputs: [batch_size, n_nodes, 3] - mask + state info
            embeddings: [batch_size, n_nodes, 9] - graph structure  
            dimensions: [batch_size, n_nodes] - node indices
        
        Returns:
            feature_x: Combined contextual representation
            param_x: Parameter representation
        """
        batch_size = inputs.shape[0]
        
        # Get node embeddings for all positions
        node_embeds = self.node_embedding[dimensions]  # [batch, n_nodes, hidden_dim]
        # All nodes have same "question type" 
        question_embeds = self.question_embedding.expand(batch_size, self.n_nodes, -1)
        
        # Combine embeddings
        combined_embeds = question_embeds + node_embeds
        
        # Feature stream: combined embeddings + graph structure + input features
        feature_x = torch.cat([combined_embeds, embeddings, inputs[:,:,1:]], dim=-1)
        
        # Parameter stream: just the input states (excluding mask bit)
        param_x = inputs[:,:,1:].clone()
        
        return feature_x, param_x

class EncoderLayer(nn.Module):
    """Transformer encoder layer adapted for graph imputation - no query stream or smoothing."""

    def __init__(self, feature_dim, param_dim, attention_heads, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim 
        self.param_dim = param_dim  
        self.attention_heads = attention_heads
        
        # Feature stream attention (keep original implementation)
        self.Q = nn.Linear(feature_dim, feature_dim)
        self.K = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)
        
        # Normalization and dropout
        self.norm_1 = NormLayer(feature_dim)
        self.norm_2 = NormLayer(feature_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        # Feed forward and parameter update
        self.ff = FeedForward(feature_dim, dropout=dropout)
        self.param_update = nn.Linear(feature_dim + param_dim, param_dim)

    def multihead_attention(self, feature_x, batch_size):
        """Apply multi-head attention to the features (original implementation)."""
        Q = self.Q(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        K = self.K(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        V = self.V(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.feature_dim // self.attention_heads)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        scores = torch.matmul(scores, V)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        output = self.out(scores)
        return output

    def forward(self, feature_x, param_x, mask):
        """Process features through attention and feed-forward (no query stream or smoothing)."""
        batch_size = feature_x.shape[0]
        
        # Feature stream processing (original)
        feature_x = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))
        
        # Parameter update (original)
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        # No smoothing, no query stream - just return the two streams
        return feature_x, param_x
    
class Encoder(nn.Module):
    """Full encoder consisting of multiple encoder layers - adapted for graph imputation."""

    def __init__(self, n_nodes, input_dim, embedding_dim, encoder_num, attention_heads, 
                 hidden_dim=64, dropout=0.1):
        """Initialize encoder with multiple layers."""
        super().__init__()
        
        # Calculate dimensions
        self.feature_dim = hidden_dim + embedding_dim + (input_dim - 1)  # hidden + embeddings + states (no mask)
        self.param_dim = input_dim - 1  # just the state bits (no mask)
        self.hidden_dim = hidden_dim
        
        print(f"Encoder dimensions: feature_dim={self.feature_dim}, param_dim={self.param_dim}")
        
        # Positional encoder
        self.position_encoder = Positional_Encoder(n_nodes, input_dim, embedding_dim, hidden_dim)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(self.feature_dim, self.param_dim, attention_heads, dropout)
            for _ in range(encoder_num)
        ])

        # Final normalization
        self.norm = NormLayer(self.feature_dim)

    def forward(self, inputs, embeddings, dimensions):
        """Process input through all encoder layers."""
        # Get initial representations
        feature_x, param_x = self.position_encoder(inputs, embeddings, dimensions)
        
        # Extract mask for attention
        mask = inputs[:, :, 0]  # First bit is mask (0=observed, 1=unobserved)
        
        # Process through all layers
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x, mask)
        
        # Final normalization
        feature_x = self.norm(feature_x)
        
        return feature_x, param_x

class GraphImputer(nn.Module):
    """Main model for graph imputation using 2-stream transformer."""
    
    def __init__(self, 
                 n_nodes=5, 
                 input_dim=3,  # mask + 2 binary states
                 embedding_dim=9,  # 5 adjacency + 4 CPD
                 encoder_layers_num=4, 
                 attention_heads=4, 
                 hidden_dim=64,
                 n_states=2,  # Binary nodes
                 dropout=0.1):
        """Initialize Graph Imputer model."""
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        # Main encoder
        self.encoder = Encoder(
            n_nodes=n_nodes,
            input_dim=input_dim, 
            embedding_dim=embedding_dim,
            encoder_num=encoder_layers_num, 
            attention_heads=attention_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output heads - one per node for predicting state probabilities
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_states)
            )
            for _ in range(n_nodes)
        ])
        
        print(f"GraphImputer initialized: {n_nodes} nodes, {n_states} states per node")
    
    def forward(self, inputs, embeddings, dimensions):
        """
        Forward pass for graph imputation.
        
        Args:
            inputs: [batch_size, n_nodes, 3] - mask + state info
            embeddings: [batch_size, n_nodes, 9] - graph structure
            dimensions: [batch_size, n_nodes] - node indices
        
        Returns:
            predictions: [batch_size, n_nodes, n_states] - predicted probabilities
        """
        # Process through encoder
        feature_x, param_x = self.encoder(inputs, embeddings, dimensions)
        
        # Apply output heads for each node
        predictions = []
        for i in range(self.n_nodes):
            node_logits = self.output_heads[i](feature_x[:, i, :])  # [batch, n_states]
            node_probs = F.softmax(node_logits, dim=-1)
            predictions.append(node_probs)
        
        # Stack predictions: [batch_size, n_nodes, n_states]
        predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
        if len(data) > 0:
            self.n_nodes = data[0][0].shape[0]
            self.input_dim = data[0][0].shape[1]
            self.embedding_dim = data[0][1].shape[1]
            self.n_states = data[0][4].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, embeddings, dimensions, mask, targets = self.data[idx]
        return {
            'inputs': inputs,
            'embeddings': embeddings,
            'dimensions': dimensions,
            'mask': mask,
            'targets': targets
        }

def collate_fn(batch):
    return {
        'inputs': torch.stack([sample['inputs'] for sample in batch]),
        'embeddings': torch.stack([sample['embeddings'] for sample in batch]),
        'dimensions': torch.stack([sample['dimensions'] for sample in batch]),
        'mask': torch.stack([sample['mask'] for sample in batch]),
        'targets': torch.stack([sample['targets'] for sample in batch])
    }

# ================================= TRAINING =================================

def compute_kl_loss(predictions, targets, mask):
    """KL divergence: KL(true || pred)"""
    unobserved_mask = mask.bool()
    
    if unobserved_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    pred_unobserved = predictions[unobserved_mask]
    targets_unobserved = targets[unobserved_mask]
    
    # KL(true || pred) = sum(true * log(true/pred))
    kl_loss = F.kl_div(
        torch.log(pred_unobserved + 1e-10),  # log predictions
        targets_unobserved,                   # true probabilities
        reduction='batchmean'
    )
    
    return kl_loss

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        inputs = batch['inputs'].to(DEVICE)
        embeddings = batch['embeddings'].to(DEVICE)
        dimensions = batch['dimensions'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        targets = batch['targets'].to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(inputs, embeddings, dimensions)
        loss = compute_kl_loss(predictions, targets, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(DEVICE)
            embeddings = batch['embeddings'].to(DEVICE)
            dimensions = batch['dimensions'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            predictions = model(inputs, embeddings, dimensions)
            loss = compute_kl_loss(predictions, targets, mask)
            
            # RMSE
            unobserved_mask = mask.bool()
            if unobserved_mask.sum() > 0:
                pred_unobserved = predictions[unobserved_mask]
                targets_unobserved = targets[unobserved_mask]
                rmse = torch.sqrt(((pred_unobserved - targets_unobserved) ** 2).mean())
                total_rmse += rmse.item()
            
            total_loss += loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'rmse': total_rmse / num_batches
    }

def create_model(n_nodes, input_dim, embedding_dim):
    """Create model with architecture scaled based on graph size."""
    
    # Scale architecture based on graph complexity
    if n_nodes <= 10:
        hidden_dim_base = 64
        attention_heads = 4
        encoder_layers = 4
    else:  # 15+ nodes
        hidden_dim_base = 128
        attention_heads = 8
        encoder_layers = 6
    
    # Ensure divisibility by attention heads
    base_dim = embedding_dim + (input_dim - 1)
    remainder = (hidden_dim_base + base_dim) % attention_heads
    hidden_dim = hidden_dim_base - remainder
    
    print(f"Architecture: hidden_dim={hidden_dim}, heads={attention_heads}, layers={encoder_layers}")
    
    model = GraphImputer(
        n_nodes=n_nodes,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        encoder_layers_num=encoder_layers,
        attention_heads=attention_heads,
        hidden_dim=hidden_dim,
        n_states=N_STATES,
        dropout=0.1
    ).to(DEVICE)
    
    return model

def train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_metrics = validate_epoch(model, test_loader)
        val_loss = val_metrics['loss']
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', 'temp_best_model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'temp_best_model.pth')))
    return model

# ================================= DATA GENERATION =================================

def generate_bayesian_network(n_nodes, edge_prob):
    """Generate random Bayesian Network with CPDs."""
    bn = BayesianNetwork.get_random(n_nodes=n_nodes, edge_prob=edge_prob, n_states=None, latents=False)
    
    node_list = sorted(list(bn.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Generate CPDs with Dirichlet priors
    cardinalities = {i: N_STATES for i in range(n_nodes)}
    cpd_list = []
    
    for node in range(n_nodes):
        parents = list(bn.get_parents(node))
        n_node_states = cardinalities[node]
        
        if parents:
            n_parent_configs = np.prod([cardinalities[p] for p in parents])
            values = np.random.dirichlet(np.ones(n_node_states), size=int(n_parent_configs))
            values = values.T
        else:
            values = np.random.dirichlet(np.ones(n_node_states)).reshape(n_node_states, 1)
        
        cpd = TabularCPD(
            variable=node, variable_card=n_node_states, values=values,
            evidence=parents, evidence_card=[cardinalities[p] for p in parents] if parents else []
        )
        cpd_list.append(cpd)
    
    bn.add_cpds(*cpd_list)
    
    if not bn.check_model():
        raise ValueError("Invalid Bayesian Network")
    
    # Create adjacency matrix and CPD parameters
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for edge in bn.edges():
        from_idx = node_to_idx[edge[0]]
        to_idx = node_to_idx[edge[1]]
        adj_matrix[from_idx, to_idx] = 1.0
    
    max_cpd_size = max([len(bn.get_cpds(node).get_values().flatten()) for node in node_list])
    cpd_data = np.zeros((n_nodes, max_cpd_size), dtype=np.float32)
    for node in node_list:
        idx = node_to_idx[node]
        cpd = bn.get_cpds(node)
        cpd_values = cpd.get_values().flatten()
        cpd_data[idx, :len(cpd_values)] = cpd_values
    
    param_embeddings = np.concatenate([adj_matrix, cpd_data], axis=1)
    
    return bn, param_embeddings, node_to_idx

def generate_sample(bn, param_embeddings, node_to_idx, n_nodes, obs_ratio, seed):
    """Generate single training sample."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Select observed nodes (30% of total)
    n_observed = max(1, int(obs_ratio * n_nodes))
    node_list = list(range(n_nodes))
    observed_nodes = random.sample(node_list, k=n_observed)
    unobserved_nodes = [node for node in node_list if node not in observed_nodes]
    
    # Generate random states for observed nodes
    evidence = {node: np.random.randint(0, N_STATES) for node in observed_nodes}
    
    try:
        infer = VariableElimination(bn)
        
        # Create inputs: [mask_bit, state_0_bit, state_1_bit]
        inputs = np.zeros((n_nodes, 3), dtype=np.float32)
        
        for node in node_list:
            if node in observed_nodes:
                inputs[node, 0] = 0.0  # Not masked
                state = evidence[node]
                inputs[node, 1 + state] = 1.0  # One-hot encoding
            else:
                inputs[node, 0] = 1.0  # Masked
        
        # Compute ground truth posteriors for ALL unobserved nodes
        targets = np.zeros((n_nodes, N_STATES), dtype=np.float32)
        for node in unobserved_nodes:
            posterior = infer.query(variables=[node], evidence=evidence)
            targets[node, :] = posterior.values
        
        # Create mask: 0 for observed, 1 for unobserved
        mask = np.zeros(n_nodes, dtype=np.float32)
        for node in unobserved_nodes:
            mask[node] = 1.0
        
        dimensions = np.arange(n_nodes, dtype=np.int64)
        
        return (
            torch.FloatTensor(inputs),
            torch.FloatTensor(param_embeddings),
            torch.LongTensor(dimensions),
            torch.FloatTensor(mask),
            torch.FloatTensor(targets)
        )
        
    except Exception as e:
        return None

def generate_dataset(bn, param_embeddings, node_to_idx, n_nodes, n_samples, obs_ratio):
    """Generate full dataset."""
    samples = []
    failed_count = 0
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        sample = generate_sample(bn, param_embeddings, node_to_idx, n_nodes, obs_ratio, i)
        if sample is not None:
            samples.append(sample)
        else:
            failed_count += 1
    
    print(f"Generated {len(samples)} samples, {failed_count} failed")
    return samples

def create_experiment_data(n_nodes, train_size, test_size):
    """Create complete experiment data for given configuration."""
    print(f"Creating data: {n_nodes} nodes, {train_size} train, {test_size} test")
    
    bn, param_embeddings, node_to_idx = generate_bayesian_network(n_nodes, EDGE_PROB)
    
    train_data = generate_dataset(bn, param_embeddings, node_to_idx, n_nodes, train_size, OBS_RATIO)
    test_data = generate_dataset(bn, param_embeddings, node_to_idx, n_nodes, test_size, OBS_RATIO)
    
    return bn, param_embeddings, train_data, test_data

# ================================= EVALUATION =================================

def designate_query_latent(unobserved_indices, query_ratio=0.3):
    """Randomly split unobserved nodes into query vs latent for evaluation."""
    n_unobserved = len(unobserved_indices)
    n_query = max(1, int(query_ratio * n_unobserved))
    
    query_indices = random.sample(unobserved_indices.tolist(), k=n_query)
    latent_indices = [idx for idx in unobserved_indices if idx not in query_indices]
    
    return query_indices, latent_indices

def comprehensive_evaluation(model, test_loader, bn):
    """Complete evaluation with all metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    
    # Collect all predictions
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(DEVICE)
            embeddings = batch['embeddings'].to(DEVICE)
            dimensions = batch['dimensions'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            predictions = model(inputs, embeddings, dimensions)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Separate metrics for all, query, and latent
    results = {}
    
    # Overall metrics
    unobserved_mask = all_masks.bool()
    if unobserved_mask.sum() > 0:
        pred_unobs = all_predictions[unobserved_mask]
        true_unobs = all_targets[unobserved_mask]
        
        results['rmse_all'] = torch.sqrt(((pred_unobs - true_unobs) ** 2).mean()).item()
        results['accuracy_all'] = ((pred_unobs.argmax(dim=-1) == true_unobs.argmax(dim=-1)).float().mean()).item()
        
        # Correlations
        pred_flat = pred_unobs.flatten().numpy()
        true_flat = true_unobs.flatten().numpy()
        results['pearson_all'], _ = stats.pearsonr(pred_flat, true_flat)
        results['spearman_all'], _ = stats.spearmanr(pred_flat, true_flat)
    
    # Query vs Latent metrics
    query_preds, query_trues = [], []
    latent_preds, latent_trues = [], []
    
    for i in range(all_predictions.shape[0]):
        sample_mask = all_masks[i]
        unobserved_indices = torch.where(sample_mask == 1)[0]
        
        if len(unobserved_indices) > 0:
            query_indices, latent_indices = designate_query_latent(unobserved_indices)
            
            if query_indices:
                query_preds.append(all_predictions[i, query_indices])
                query_trues.append(all_targets[i, query_indices])
            
            if latent_indices:
                latent_preds.append(all_predictions[i, latent_indices])
                latent_trues.append(all_targets[i, latent_indices])
    
    # Query metrics
    if query_preds:
        query_preds = torch.cat(query_preds, dim=0)
        query_trues = torch.cat(query_trues, dim=0)
        
        results['rmse_query'] = torch.sqrt(((query_preds - query_trues) ** 2).mean()).item()
        results['accuracy_query'] = ((query_preds.argmax(dim=-1) == query_trues.argmax(dim=-1)).float().mean()).item()
        
        pred_flat = query_preds.flatten().numpy()
        true_flat = query_trues.flatten().numpy()
        results['pearson_query'], _ = stats.pearsonr(pred_flat, true_flat)
        results['spearman_query'], _ = stats.spearmanr(pred_flat, true_flat)
    
    # Latent metrics
    if latent_preds:
        latent_preds = torch.cat(latent_preds, dim=0)
        latent_trues = torch.cat(latent_trues, dim=0)
        
        results['rmse_latent'] = torch.sqrt(((latent_preds - latent_trues) ** 2).mean()).item()
        results['accuracy_latent'] = ((latent_preds.argmax(dim=-1) == latent_trues.argmax(dim=-1)).float().mean()).item()
        
        pred_flat = latent_preds.flatten().numpy()
        true_flat = latent_trues.flatten().numpy()
        results['pearson_latent'], _ = stats.pearsonr(pred_flat, true_flat)
        results['spearman_latent'], _ = stats.spearmanr(pred_flat, true_flat)
    
    return results

def consistency_check(model, test_loader):
    """Check H(p,q) - H(p) consistency."""
    model.eval()
    
    kl_divergences = []
    entropy_ratios = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(DEVICE)
            embeddings = batch['embeddings'].to(DEVICE)
            dimensions = batch['dimensions'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            predictions = model(inputs, embeddings, dimensions)
            
            unobserved_mask = mask.bool()
            if unobserved_mask.sum() > 0:
                pred_unobs = predictions[unobserved_mask]
                true_unobs = targets[unobserved_mask]
                
                # H(p) = -sum(p * log(p))
                h_p = -(true_unobs * torch.log(true_unobs + 1e-10)).sum(dim=-1).mean()
                
                # H(p,q) = -sum(p * log(q))
                h_pq = -(true_unobs * torch.log(pred_unobs + 1e-10)).sum(dim=-1).mean()
                
                # KL = H(p,q) - H(p)
                kl_div = h_pq - h_p
                
                kl_divergences.append(kl_div.item())
                entropy_ratios.append((h_pq / (h_p + 1e-10)).item())
    
    return {
        'mean_kl_divergence': np.mean(kl_divergences),
        'std_kl_divergence': np.std(kl_divergences),
        'mean_entropy_ratio': np.mean(entropy_ratios),
        'kl_distribution': kl_divergences
    }

def time_inference(model, test_loader, bn, n_samples=50):
    """Time model inference vs Variable Elimination."""
    model.eval()
    
    # Time model inference
    model_times = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i * batch['inputs'].shape[0] >= n_samples:
                break
                
            inputs = batch['inputs'].to(DEVICE)
            embeddings = batch['embeddings'].to(DEVICE)
            dimensions = batch['dimensions'].to(DEVICE)
            
            start_time = time.time()
            predictions = model(inputs, embeddings, dimensions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_size = inputs.shape[0]
            model_times.extend([(end_time - start_time) / batch_size] * batch_size)
    
    # Time Variable Elimination
    ve_times = []
    infer = VariableElimination(bn)
    
    for i, batch in enumerate(test_loader):
        if len(ve_times) >= n_samples:
            break
            
        inputs = batch['inputs']
        mask = batch['mask']
        
        for j in range(min(inputs.shape[0], n_samples - len(ve_times))):
            sample_mask = mask[j]
            observed_indices = torch.where(sample_mask == 0)[0]
            unobserved_indices = torch.where(sample_mask == 1)[0]
            
            if len(unobserved_indices) == 0:
                continue
                
            # Create evidence
            evidence = {}
            for obs_idx in observed_indices:
                state = torch.argmax(inputs[j, obs_idx, 1:]).item()
                evidence[int(obs_idx)] = state
            
            start_time = time.time()
            try:
                for unobs_idx in unobserved_indices[:3]:  # Limit for speed
                    infer.query(variables=[int(unobs_idx)], evidence=evidence)
                ve_time = time.time() - start_time
                ve_times.append(ve_time)
            except:
                continue
    
    return {
        'model_time_mean': np.mean(model_times),
        'model_time_std': np.std(model_times),
        've_time_mean': np.mean(ve_times) if ve_times else float('inf'),
        've_time_std': np.std(ve_times) if ve_times else 0,
        'speedup': np.mean(ve_times) / np.mean(model_times) if ve_times and model_times else 1.0
    }

def convert_to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    else:
        return obj

def clear_memory():
    """Clear GPU memory between experiments."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ================================= MAIN EXPERIMENT =================================

def run_single_experiment(n_nodes, train_size):
    """Run single experiment configuration."""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {n_nodes} nodes, {train_size} training samples")
    print(f"{'='*50}")

    clear_memory()
    
    # Generate data
    bn, param_embeddings, train_data, test_data = create_experiment_data(n_nodes, train_size, TEST_SIZE)
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Failed to generate data")
        return None
    
    # Create datasets and loaders
    train_dataset = GraphDataset(train_data)
    test_dataset = GraphDataset(test_data)
    
    batch_size = min(32, len(train_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    input_dim = train_data[0][0].shape[1]
    embedding_dim = train_data[0][1].shape[1]
    model = create_model(n_nodes, input_dim, embedding_dim)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, test_loader, epochs=100, lr=1e-4, patience=15)
    
    # Comprehensive evaluation
    print("Evaluating neural imputer...")
    eval_results = comprehensive_evaluation(model, test_loader, bn)
    consistency_results = consistency_check(model, test_loader)
    timing_results = time_inference(model, test_loader, bn, n_samples=50)
    
    # Domain-specific model evaluation
    print("Training and evaluating domain-specific model...")
    try:
        # Convert training data for pgmpy
        pgmpy_train_data = convert_training_data_for_pgmpy(train_data, n_nodes)
        
        # Extract adjacency matrix and create BN structure
        adj_matrix = extract_adjacency_from_embeddings(param_embeddings, n_nodes)
        bn_structure = create_bn_structure_from_adjacency(adj_matrix)
        
        # Learn domain-specific model
        domain_model = learn_domain_specific_model(bn_structure, pgmpy_train_data, N_STATES)
        
        # Evaluate domain-specific model
        domain_results = evaluate_domain_specific_model(domain_model, test_data, n_nodes, N_STATES)
        
    except Exception as e:
        print(f"Domain-specific model failed: {e}")
        domain_results = {
            'mean_kl': float('inf'),
            'std_kl': 0.0,
            'mean_error': float('inf'),
            'failed_rate': 1.0,
            'n_evaluations': 0,
            'kl_distribution': []
        }
    
    # Combine results
    results = {
        'config': {'n_nodes': n_nodes, 'train_size': train_size},
        'neural_imputer': {
            'evaluation': eval_results,
            'consistency': consistency_results,
            'timing': timing_results,
            'model_params': sum(p.numel() for p in model.parameters())
        },
        'domain_specific': domain_results,
        'comparison': {
            'neural_kl': consistency_results.get('mean_kl_divergence', float('inf')),
            'domain_kl': domain_results.get('mean_kl', float('inf')),
            'kl_ratio': (consistency_results.get('mean_kl_divergence', float('inf')) / 
                        (domain_results.get('mean_kl', float('inf')) + 1e-10))
        }
    }
    
    # Convert to JSON serializable format
    results = convert_to_json_serializable(results)
    
    # Save model and results
    model_path = os.path.join(BASE_DIR, 'models', f'model_{n_nodes}nodes_{train_size}samples.pth')
    torch.save(model.state_dict(), model_path)
    
    results_path = os.path.join(BASE_DIR, 'results', f'results_{n_nodes}nodes_{train_size}samples.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results: Neural KL={consistency_results.get('mean_kl_divergence', 0):.4f}, "
          f"Domain KL={domain_results.get('mean_kl', 0):.4f}, "
          f"KL Ratio={results['comparison']['kl_ratio']:.2f}")
    
    del model, train_loader, test_loader, train_dataset, test_dataset
    del train_data, test_data, bn, param_embeddings
    del eval_results, consistency_results, timing_results
    
    # Clear GPU memory
    clear_memory()
    
    return results

def generate_plots(all_results):
    """Generate all plots separated by graph size."""
    
    # Group results by graph size
    results_by_size = {}
    for key, result in all_results.items():
        n_nodes, train_size = key
        if n_nodes not in results_by_size:
            results_by_size[n_nodes] = {}
        results_by_size[n_nodes][train_size] = result
    
    # Create plots for each graph size
    for n_nodes in GRAPH_SIZES:
        if n_nodes not in results_by_size:
            continue
            
        size_results = results_by_size[n_nodes]
        train_sizes = sorted(size_results.keys())
        
        # Extract neural imputer metrics
        rmse_all = [size_results[ts]['neural_imputer']['evaluation'].get('rmse_all', 0) for ts in train_sizes]
        rmse_query = [size_results[ts]['neural_imputer']['evaluation'].get('rmse_query', 0) for ts in train_sizes]
        rmse_latent = [size_results[ts]['neural_imputer']['evaluation'].get('rmse_latent', 0) for ts in train_sizes]
        
        # Extract KL metrics for comparison
        neural_kl = [size_results[ts]['neural_imputer']['consistency'].get('mean_kl_divergence', 0) for ts in train_sizes]
        domain_kl = [size_results[ts]['domain_specific'].get('mean_kl', float('inf')) for ts in train_sizes]
        
        # Pearson correlation
        pearson_all = [size_results[ts]['neural_imputer']['evaluation'].get('pearson_all', 0) for ts in train_sizes]
        pearson_query = [size_results[ts]['neural_imputer']['evaluation'].get('pearson_query', 0) for ts in train_sizes]
        pearson_latent = [size_results[ts]['neural_imputer']['evaluation'].get('pearson_latent', 0) for ts in train_sizes]
        
        # Create plots (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Graph Imputation Results - {n_nodes} Nodes', fontsize=16)
        
        # RMSE plot (Neural Imputer only)
        axes[0, 0].plot(train_sizes, rmse_all, 'o-', label='All Unobserved', linewidth=2, color='blue')
        axes[0, 0].plot(train_sizes, rmse_query, 's-', label='Query Nodes', linewidth=2, color='orange')
        axes[0, 0].plot(train_sizes, rmse_latent, '^-', label='Latent Nodes', linewidth=2, color='green')
        axes[0, 0].set_xlabel('Training Samples')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Neural Imputer: RMSE vs Training Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL Divergence Comparison (Neural vs Domain-specific)
        # Filter out infinite values for plotting
        neural_kl_clean = [kl if kl != float('inf') else np.nan for kl in neural_kl]
        domain_kl_clean = [kl if kl != float('inf') else np.nan for kl in domain_kl]
        
        axes[0, 1].plot(train_sizes, neural_kl_clean, 'o-', label='Neural Imputer', linewidth=2, color='red')
        axes[0, 1].plot(train_sizes, domain_kl_clean, 's-', label='Domain-specific BN', linewidth=2, color='blue')
        axes[0, 1].set_xlabel('Training Samples')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].set_title('KL Divergence vs Training Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')  # Log scale for better visualization
        
        # KL Distribution Histogram (both models)
        neural_kl_dists = []
        domain_kl_dists = []
        for ts in train_sizes:
            # Neural imputer KL distribution
            neural_dist = size_results[ts]['neural_imputer']['consistency'].get('kl_distribution', [])
            neural_kl_dists.extend(neural_dist)
            
            # Domain-specific KL distribution
            domain_dist = size_results[ts]['domain_specific'].get('kl_distribution', [])
            if domain_dist:  # Only if we have valid results
                domain_kl_dists.extend(domain_dist)
        
        if neural_kl_dists:
            axes[1, 0].hist(neural_kl_dists, bins=30, alpha=0.6, color='red', 
                           label=f'Neural (μ={np.mean(neural_kl_dists):.3f})', edgecolor='black')
        if domain_kl_dists:
            axes[1, 0].hist(domain_kl_dists, bins=30, alpha=0.6, color='blue',
                           label=f'Domain-specific (μ={np.mean(domain_kl_dists):.3f})', edgecolor='black')
        
        axes[1, 0].set_xlabel('KL Divergence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('KL Divergence Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pearson correlation (Neural Imputer)
        axes[1, 1].plot(train_sizes, pearson_all, 'o-', label='All Unobserved', linewidth=2, color='blue')
        axes[1, 1].plot(train_sizes, pearson_query, 's-', label='Query Nodes', linewidth=2, color='orange')
        axes[1, 1].plot(train_sizes, pearson_latent, '^-', label='Latent Nodes', linewidth=2, color='green')
        axes[1, 1].set_xlabel('Training Samples')
        axes[1, 1].set_ylabel('Pearson Correlation')
        axes[1, 1].set_title('Neural Imputer: Pearson Correlation vs Training Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'results', f'results_{n_nodes}_nodes.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main experiment loop."""
    print("Starting Graph Imputation Experiments")
    print(f"Graph sizes: {GRAPH_SIZES}")
    print(f"Training sizes: {TRAINING_SIZES}")
    
    all_results = {}
    
    for n_nodes in GRAPH_SIZES:
        for train_size in TRAINING_SIZES:
            try:
                results = run_single_experiment(n_nodes, train_size)
                if results:
                    all_results[(n_nodes, train_size)] = results
                    
                    with open(os.path.join(BASE_DIR, 'results', 'all_results_intermediate.json'), 'w') as f:
                        json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) for k, v in all_results.items()}
                        json.dump(json_results, f, indent=2)
                        
            except Exception as e:
                print(f"Experiment failed for {n_nodes} nodes, {train_size} samples: {e}")
                clear_memory()
                continue
    
    # Generate final plots
    print("\nGenerating plots...")
    generate_plots(all_results)
    
    # Save final results
    with open(os.path.join(BASE_DIR, 'results', 'all_results_final.json'), 'w') as f:
        json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) for k, v in all_results.items()}
        json.dump(json_results, f, indent=2)
    
    print("All experiments completed!")
    print(f"Results saved in ./results/")
    print(f"Models saved in ./models/")

if __name__ == "__main__":
    main()