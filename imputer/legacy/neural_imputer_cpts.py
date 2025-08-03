"""
Neural imputer implementation for graph imputation (structure + observed CPTs version).

This module contains the transformer-based neural imputer model architecture 
that uses graph structure, observed node states, and CPTs of observed nodes.

Author: Prabhav Singh
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    print("Warning: pyAgrum not available for CPT extraction")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================= CPT EXTRACTION =================================

def extract_cpts_for_observed_nodes(bn, observed_nodes, n_nodes, max_cpt_size=None):
    """Extract CPTs for observed nodes from the true BN."""
    if not PYAGRUM_AVAILABLE:
        return np.zeros((n_nodes, 8))
    
    cpt_data = []
    cpt_sizes = []
    
    for node in range(n_nodes):
        if node in observed_nodes:
            # Extract CPT for this observed node
            node_str = str(node)
            if node_str in [bn.variable(i).name() for i in bn.nodes()]:
                cpt = bn.cpt(node_str)
                cpt_values = np.array(cpt.tolist()).flatten()
                cpt_data.append(cpt_values)
                cpt_sizes.append(len(cpt_values))
            else:
                # Node not in BN, use uniform
                cpt_data.append(np.array([0.5, 0.5]))
                cpt_sizes.append(2)
        else:
            # Unobserved node - use zeros as placeholder
            cpt_data.append(np.array([0.0, 0.0]))
            cpt_sizes.append(2)
    
    # Determine max size for padding
    if max_cpt_size is None:
        max_cpt_size = max(cpt_sizes) if cpt_sizes else 2
    
    # Pad all CPTs to same size
    padded_cpts = np.zeros((n_nodes, max_cpt_size))
    for i, cpt in enumerate(cpt_data):
        padded_cpts[i, :len(cpt)] = cpt
    
    return padded_cpts

# ================================= NEURAL ARCHITECTURE =================================

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
    """Positional encoder for graph imputation with CPT information."""
    
    def __init__(self, n_nodes, input_dim, structure_dim, cpt_dim, hidden_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.structure_dim = structure_dim
        self.cpt_dim = cpt_dim
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        # Question embedding
        self.question_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
        # CPT processing layer
        self.cpt_processor = nn.Linear(cpt_dim, hidden_dim)
        
        torch.nn.init.kaiming_normal_(self.node_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, structure_info, cpt_info, dimensions):
        batch_size = inputs.shape[0]
        
        # Get node embeddings for all positions
        node_embeds = self.node_embedding[dimensions]
        # All nodes have same "question type" 
        question_embeds = self.question_embedding.expand(batch_size, self.n_nodes, -1)
        
        # Process CPT information
        cpt_embeds = self.cpt_processor(cpt_info)
        
        # Combine embeddings
        combined_embeds = question_embeds + node_embeds + cpt_embeds
        
        # Feature stream: combined embeddings + structural info + input features
        feature_x = torch.cat([combined_embeds, structure_info, inputs[:,:,1:]], dim=-1)
        
        # Parameter stream: just the input states (excluding mask bit)
        param_x = inputs[:,:,1:].clone()
        
        return feature_x, param_x

class EncoderLayer(nn.Module):
    """Transformer encoder layer for graph imputation."""

    def __init__(self, feature_dim, param_dim, attention_heads, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim 
        self.param_dim = param_dim  
        self.attention_heads = attention_heads
        
        # Feature stream attention
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
        """Apply multi-head attention to the features."""
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
        """Process features through attention and feed-forward."""
        batch_size = feature_x.shape[0]
        
        # Feature stream processing
        feature_x = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))
        
        # Parameter update
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        return feature_x, param_x

class Encoder(nn.Module):
    """Full encoder consisting of multiple encoder layers."""

    def __init__(self, n_nodes, input_dim, structure_dim, cpt_dim, encoder_num, attention_heads, 
                 hidden_dim=64, dropout=0.1):
        super().__init__()
        
        # Calculate dimensions
        self.feature_dim = hidden_dim + structure_dim + (input_dim - 1)
        self.param_dim = input_dim - 1
        self.hidden_dim = hidden_dim
        self.cpt_dim = cpt_dim
        
        print(f"Encoder dimensions: feature_dim={self.feature_dim}, param_dim={self.param_dim}, cpt_dim={cpt_dim}")
        
        # Positional encoder
        self.position_encoder = Positional_Encoder(n_nodes, input_dim, structure_dim, cpt_dim, hidden_dim)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(self.feature_dim, self.param_dim, attention_heads, dropout)
            for _ in range(encoder_num)
        ])

        # Final normalization
        self.norm = NormLayer(self.feature_dim)

    def forward(self, inputs, structure_info, cpt_info, dimensions):
        """Process input through all encoder layers."""
        # Get initial representations
        feature_x, param_x = self.position_encoder(inputs, structure_info, cpt_info, dimensions)
        
        # Extract mask for attention
        mask = inputs[:, :, 0]  # First bit is mask
        
        # Process through all layers
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x, mask)
        
        # Final normalization
        feature_x = self.norm(feature_x)
        
        return feature_x, param_x

class GraphImputerWithCPTs(nn.Module):
    """Main model for graph imputation using transformer with CPT information."""
    
    def __init__(self, 
                 n_nodes=5, 
                 input_dim=3,
                 structure_dim=None,
                 cpt_dim=None,
                 encoder_layers_num=4, 
                 attention_heads=4, 
                 hidden_dim=64,
                 n_states=2,
                 dropout=0.1):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        # Structure dimension is n_nodes x n_nodes for adjacency matrix
        if structure_dim is None:
            structure_dim = n_nodes
        
        # CPT dimension - default to reasonable size
        if cpt_dim is None:
            cpt_dim = 8  # Reasonable default for small graphs
        
        self.cpt_dim = cpt_dim
        
        # Main encoder
        self.encoder = Encoder(
            n_nodes=n_nodes,
            input_dim=input_dim, 
            structure_dim=structure_dim,
            cpt_dim=cpt_dim,
            encoder_num=encoder_layers_num, 
            attention_heads=attention_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output heads - one per node
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_states)
            )
            for _ in range(n_nodes)
        ])
        
        print(f"GraphImputer (with CPTs) initialized: {n_nodes} nodes, {n_states} states per node, cpt_dim={cpt_dim}")
    
    def forward(self, inputs, structure_info, cpt_info, dimensions):
        # Process through encoder
        feature_x, param_x = self.encoder(inputs, structure_info, cpt_info, dimensions)
        
        # Apply output heads for each node
        predictions = []
        for i in range(self.n_nodes):
            node_logits = self.output_heads[i](feature_x[:, i, :])
            node_probs = F.softmax(node_logits, dim=-1)
            predictions.append(node_probs)
        
        # Stack predictions: [batch_size, n_nodes, n_states]
        predictions = torch.stack(predictions, dim=1)
        
        return predictions

# ================================= DATASET AND TRAINING =================================

class GraphDatasetWithCPTs(Dataset):
    def __init__(self, data, bn=None):
        self.data = data
        self.bn = bn
        if len(data) > 0:
            self.n_nodes = data[0][0].shape[0]
            self.input_dim = data[0][0].shape[1]
            self.embedding_dim = data[0][1].shape[1]
            self.n_states = data[0][4].shape[1]
            
            # Pre-compute max CPT size for this dataset to ensure consistency
            self.max_cpt_size = self._compute_max_cpt_size()
    
    def _compute_max_cpt_size(self):
        """Compute the maximum CPT size for this BN to ensure consistent tensor shapes."""
        if not self.bn or not PYAGRUM_AVAILABLE:
            return 8  # Safe default
        
        max_size = 2  # Minimum size for nodes with no parents
        for node_id in self.bn.nodes():
            node_str = str(node_id)
            cpt = self.bn.cpt(node_str)
            cpt_values = np.array(cpt.tolist()).flatten()
            max_size = max(max_size, len(cpt_values))
        
        return max_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, structure_info, dimensions, mask, targets = self.data[idx]
        
        # Extract observed nodes for this sample
        observed_nodes = []
        for node in range(self.n_nodes):
            if mask[node] == 0:  # Observed
                observed_nodes.append(node)
        
        # Extract CPTs for observed nodes with consistent size
        if self.bn is not None:
            cpt_info = extract_cpts_for_observed_nodes(self.bn, observed_nodes, self.n_nodes, self.max_cpt_size)
        else:
            # Fallback - use zeros with consistent size
            cpt_info = np.zeros((self.n_nodes, self.max_cpt_size))
        
        return {
            'inputs': inputs,
            'structure_info': structure_info,
            'cpt_info': torch.FloatTensor(cpt_info),
            'dimensions': dimensions,
            'mask': mask,
            'targets': targets
        }

def collate_fn_cpts(batch):
    return {
        'inputs': torch.stack([sample['inputs'] for sample in batch]),
        'structure_info': torch.stack([sample['structure_info'] for sample in batch]),
        'cpt_info': torch.stack([sample['cpt_info'] for sample in batch]),
        'dimensions': torch.stack([sample['dimensions'] for sample in batch]),
        'mask': torch.stack([sample['mask'] for sample in batch]),
        'targets': torch.stack([sample['targets'] for sample in batch])
    }

def compute_kl_loss_cpts(predictions, targets, mask):
    """KL divergence: KL(true || pred)"""
    unobserved_mask = mask.bool()
    
    if unobserved_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    pred_unobserved = predictions[unobserved_mask]
    targets_unobserved = targets[unobserved_mask]
    
    # KL(true || pred) = sum(true * log(true/pred))
    kl_loss = F.kl_div(
        torch.log(pred_unobserved + 1e-10),
        targets_unobserved,
        reduction='batchmean'
    )
    
    return kl_loss

def train_epoch_cpts(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        inputs = batch['inputs'].to(DEVICE)
        structure_info = batch['structure_info'].to(DEVICE)
        cpt_info = batch['cpt_info'].to(DEVICE)
        dimensions = batch['dimensions'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        targets = batch['targets'].to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(inputs, structure_info, cpt_info, dimensions)
        loss = compute_kl_loss_cpts(predictions, targets, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch_cpts(model, test_loader):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(DEVICE)
            structure_info = batch['structure_info'].to(DEVICE)
            cpt_info = batch['cpt_info'].to(DEVICE)
            dimensions = batch['dimensions'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            predictions = model(inputs, structure_info, cpt_info, dimensions)
            loss = compute_kl_loss_cpts(predictions, targets, mask)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def create_model_cpts(n_nodes, input_dim, structure_dim, cpt_dim=None):
    """Create model with CPT support and architecture scaled based on graph size."""
    
    if n_nodes <= 10:
        hidden_dim_base = 64
        attention_heads = 4
        encoder_layers = 4
    else:
        hidden_dim_base = 128
        attention_heads = 8
        encoder_layers = 6
    
    # Default CPT dimension based on graph size - should be 2^(max_parents + 1)
    if cpt_dim is None:
        # For target_parents=1.0, max parents is typically 2, so max CPT size is 2^3 = 8
        cpt_dim = 8  # Conservative default for graphs with O(1) parents
    
    # Ensure divisibility by attention heads
    base_dim = structure_dim + (input_dim - 1)
    remainder = (hidden_dim_base + base_dim) % attention_heads
    hidden_dim = hidden_dim_base - remainder
    
    print(f"Architecture: hidden_dim={hidden_dim}, heads={attention_heads}, layers={encoder_layers}, cpt_dim={cpt_dim}")
    
    model = GraphImputerWithCPTs(
        n_nodes=n_nodes,
        input_dim=input_dim,
        structure_dim=structure_dim,
        cpt_dim=cpt_dim,
        encoder_layers_num=encoder_layers,
        attention_heads=attention_heads,
        hidden_dim=hidden_dim,
        n_states=2,
        dropout=0.1
    ).to(DEVICE)
    
    return model

def train_model_cpts(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch_cpts(model, train_loader, optimizer)
        val_loss = validate_epoch_cpts(model, test_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    return model

def evaluate_neural_model_cpts(model, test_data, bn, n_nodes, n_states=2):
    """Evaluate neural model with CPTs and compute KL divergence."""
    print(f"Evaluating neural model (with CPTs) on {len(test_data)} test samples...")
    
    model.eval()
    kl_divergences = []
    prediction_errors = []
    failed_inferences = 0
    
    for inputs, structure_info, dimensions, mask, targets in test_data:
        # Get unobserved nodes for this sample
        unobserved_nodes = []
        observed_nodes = []
        
        for node in range(n_nodes):
            if mask[node] == 1:  # Unobserved
                unobserved_nodes.append(node)
            else:  # Observed
                observed_nodes.append(node)
        
        if not unobserved_nodes:
            continue
        
        # Extract CPTs for observed nodes - use same max size as training
        # Need to get max_cpt_size from somewhere - let's use the model's expected size
        expected_cpt_dim = model.encoder.cpt_dim
        cpt_info = extract_cpts_for_observed_nodes(bn, observed_nodes, n_nodes, expected_cpt_dim)
            
        # Get predictions for unobserved nodes
        for node in unobserved_nodes:
            try:
                with torch.no_grad():
                    # Add batch dimension
                    inputs_batch = inputs.unsqueeze(0).to(DEVICE)
                    structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
                    cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
                    dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)
                    
                    predictions = model(inputs_batch, structure_info_batch, cpt_info_batch, dimensions_batch)
                    pred_probs = predictions[0, node, :].cpu().numpy()
                
                # Get ground truth
                true_probs = targets[node].numpy()
                
                # Ensure probabilities are valid
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    pred_probs = np.ones(n_states) / n_states
                else:
                    pred_probs = pred_probs / np.sum(pred_probs)
                
                if np.any(np.isnan(true_probs)) or np.sum(true_probs) == 0:
                    if failed_inferences < 5:  # Only print first few failures
                        print(f"Invalid true_probs for node {node}: {true_probs}")
                    failed_inferences += 1
                    continue
                
                # Compute KL divergence: KL(true || pred)
                kl = 0.0
                for state in range(n_states):
                    if true_probs[state] > 1e-10:
                        kl += true_probs[state] * np.log(
                            (true_probs[state] + 1e-10) / (pred_probs[state] + 1e-10)
                        )
                
                if np.isnan(kl) or np.isinf(kl) or kl < 0:
                    if failed_inferences < 5:  # Only print first few failures
                        print(f"Invalid KL for node {node}: kl={kl}, true_probs={true_probs}, pred_probs={pred_probs}")
                    failed_inferences += 1
                    continue
                
                kl_divergences.append(kl)
                
                # Prediction error
                error = np.linalg.norm(pred_probs - true_probs)
                prediction_errors.append(error)
                
            except Exception as e:
                if len(kl_divergences) < 5:
                    print(f"Neural evaluation failed for node {node}: {str(e)[:100]}")
                failed_inferences += 1
                continue
    
    if not kl_divergences:
        return {
            'mean_kl': float('inf'),
            'std_kl': 0.0,
            'mean_error': float('inf'),
            'failed_rate': 1.0,
            'n_evaluations': 0
        }
    
    results = {
        'mean_kl': np.mean(kl_divergences),
        'std_kl': np.std(kl_divergences),
        'mean_error': np.mean(prediction_errors),
        'failed_rate': failed_inferences / (len(kl_divergences) + failed_inferences),
        'n_evaluations': len(kl_divergences),
        'kl_distribution': kl_divergences
    }
    
    print(f"Neural evaluation: Mean KL = {results['mean_kl']:.4f}, "
          f"Failed rate = {results['failed_rate']:.2%}")
    
    return results