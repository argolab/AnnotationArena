"""
Graph imputation using two-stream transformer architecture.

This module implements a transformer-based imputer that uses:
1. Embedding stream: Node embeddings + structure + observed evidence
2. Parameter stream: True CPTs for observed nodes, zeros for unobserved nodes

The model learns to impute missing CPT values through multi-layer processing
while never seeing true CPT values for unobserved nodes in the input.

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
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    logger.warning("pyAgrum not available for CPT extraction")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================= CPT PROCESSING =================================

def extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size=None):
    """
    Extract CPTs for observed nodes from BN, zeros for unobserved nodes.
    
    Args:
        bn: BayesNet object
        observed_nodes: List of observed node indices
        n_nodes: Total number of nodes
        max_cpt_size: Maximum CPT size for padding
        
    Returns:
        np.ndarray: (n_nodes, max_cpt_size) with CPTs for observed, zeros for unobserved
    """
    if not PYAGRUM_AVAILABLE:
        return np.zeros((n_nodes, 8))
    
    cpt_data = []
    cpt_sizes = []
    
    for node in range(n_nodes):
        if node in observed_nodes:
            # Extract true CPT for observed node
            node_str = str(node)
            if node_str in [bn.variable(i).name() for i in bn.nodes()]:
                cpt = bn.cpt(node_str)
                cpt_values = np.array(cpt.tolist()).flatten()
                cpt_data.append(cpt_values)
                cpt_sizes.append(len(cpt_values))
            else:
                # Fallback uniform
                cpt_data.append(np.array([0.5, 0.5]))
                cpt_sizes.append(2)
        else:
            # ZEROS for unobserved nodes - no information leakage
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

def compute_max_cpt_size(bn):
    """Compute maximum CPT size for consistent tensor shapes."""
    if not bn or not PYAGRUM_AVAILABLE:
        return 8
    
    max_size = 2
    for node_id in bn.nodes():
        node_str = str(node_id)
        cpt = bn.cpt(node_str)
        cpt_values = np.array(cpt.tolist()).flatten()
        max_size = max(max_size, len(cpt_values))
    
    return max_size

# ================================= NEURAL ARCHITECTURE =================================

class LayerNorm(nn.Module):
    """Layer normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    """Feed forward network."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 2)
        self.layer2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.layer2(self.dropout(F.relu(self.layer1(x))))

class PositionalEncoder(nn.Module):
    """Initialize the two-stream architecture."""
    
    def __init__(self, n_nodes, input_dim, structure_dim, cpt_dim, hidden_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.structure_dim = structure_dim
        self.cpt_dim = cpt_dim
        self.hidden_dim = hidden_dim
        
        # Node embeddings for positional encoding
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        self.question_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Initialize embeddings
        torch.nn.init.kaiming_normal_(self.node_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, structure_info, cpt_info, dimensions):
        batch_size = inputs.shape[0]
        
        # Create embedding stream
        node_embeds = self.node_embedding[dimensions]
        question_embeds = self.question_embedding.expand(batch_size, self.n_nodes, -1)
        combined_embeds = question_embeds + node_embeds
        
        # Embedding stream: embeddings + structure + observed states
        embedding_stream = torch.cat([
            combined_embeds, 
            structure_info, 
            inputs[:, :, 1:]  # Remove mask bit
        ], dim=-1)
        
        # Parameter stream: CPT data (true for observed, zeros for unobserved)
        parameter_stream = cpt_info
        
        return embedding_stream, parameter_stream

class TransformerLayer(nn.Module):
    """Two-stream transformer layer."""

    def __init__(self, embedding_dim, parameter_dim, attention_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.parameter_dim = parameter_dim
        self.attention_heads = attention_heads
        
        # Multi-head attention for embedding stream
        self.attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, dropout=dropout, batch_first=True
        )
        
        # Feed forward for embedding stream
        self.ff_embedding = FeedForward(embedding_dim, dropout)
        
        # Cross-stream parameter update
        self.parameter_update = nn.Linear(embedding_dim + parameter_dim, parameter_dim)
        
        # Normalization layers
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(parameter_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_stream, parameter_stream, mask=None):
        """
        Process both streams.
        
        Args:
            embedding_stream: (batch, n_nodes, embedding_dim)
            parameter_stream: (batch, n_nodes, parameter_dim)
            mask: Optional attention mask
            
        Returns:
            Updated embedding_stream and parameter_stream
        """
        # Embedding stream processing
        # 1. Multi-head attention
        embedding_norm = self.norm1(embedding_stream)
        attn_out, _ = self.attention(embedding_norm, embedding_norm, embedding_norm, 
                                   key_padding_mask=mask)
        embedding_stream = embedding_stream + self.dropout(attn_out)
        
        # 2. Feed forward
        embedding_norm = self.norm2(embedding_stream)
        ff_out = self.ff_embedding(embedding_norm)
        embedding_stream = embedding_stream + self.dropout(ff_out)
        
        # Parameter stream update using embedding context
        parameter_norm = self.norm3(parameter_stream)
        combined = torch.cat([embedding_stream, parameter_norm], dim=-1)
        parameter_stream = self.parameter_update(combined)
        
        return embedding_stream, parameter_stream

class TwoStreamTransformer(nn.Module):
    """Main transformer with embedding and parameter streams."""

    def __init__(self, n_nodes, input_dim, structure_dim, cpt_dim, 
                 num_layers=4, attention_heads=4, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.cpt_dim = cpt_dim
        
        # Calculate embedding dimension
        self.embedding_dim = hidden_dim + structure_dim + (input_dim - 1)
        
        logger.debug(f"Transformer dimensions: embedding_dim={self.embedding_dim}, "
                    f"parameter_dim={cpt_dim}, layers={num_layers}")
        
        # Positional encoder
        self.pos_encoder = PositionalEncoder(n_nodes, input_dim, structure_dim, 
                                           cpt_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.embedding_dim, cpt_dim, attention_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm_embedding = LayerNorm(self.embedding_dim)
        self.final_norm_parameter = LayerNorm(cpt_dim)

    def forward(self, inputs, structure_info, cpt_info, dimensions):
        """
        Forward pass through two-stream transformer.
        
        Returns:
            parameter_stream: Final CPT predictions (observed + imputed)
        """
        # Initialize streams
        embedding_stream, parameter_stream = self.pos_encoder(
            inputs, structure_info, cpt_info, dimensions
        )
        
        # Extract mask for attention (1 for unobserved = should be masked)
        mask = inputs[:, :, 0].bool()  # First bit is mask
        
        # Process through transformer layers
        for layer in self.layers:
            embedding_stream, parameter_stream = layer(
                embedding_stream, parameter_stream, mask
            )
        
        # Final normalization
        embedding_stream = self.final_norm_embedding(embedding_stream)
        parameter_stream = self.final_norm_parameter(parameter_stream)
        
        return parameter_stream

class GraphImputer(nn.Module):
    """Complete graph imputation model."""
    
    def __init__(self, n_nodes=5, input_dim=3, structure_dim=None, cpt_dim=None,
                 num_layers=4, attention_heads=4, hidden_dim=64, 
                 n_states=2, dropout=0.1):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_states = n_states
        
        # Default structure dimension (adjacency matrix columns)
        if structure_dim is None:
            structure_dim = n_nodes
        
        # Default CPT dimension
        if cpt_dim is None:
            cpt_dim = 8
        
        self.cpt_dim = cpt_dim
        
        # Main transformer
        self.transformer = TwoStreamTransformer(
            n_nodes=n_nodes,
            input_dim=input_dim,
            structure_dim=structure_dim,
            cpt_dim=cpt_dim,
            num_layers=num_layers,
            attention_heads=attention_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output heads - convert CPT parameters to state probabilities
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cpt_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_states),
                nn.Softmax(dim=-1)
            )
            for _ in range(n_nodes)
        ])
        
        logger.info(f"GraphImputer initialized: {n_nodes} nodes, {n_states} states, "
                   f"cpt_dim={cpt_dim}, layers={num_layers}")
    
    def forward(self, inputs, structure_info, cpt_info, dimensions):
        """
        Forward pass.
        
        Args:
            inputs: (batch, n_nodes, input_dim) - observed states with mask
            structure_info: (batch, n_nodes, structure_dim) - adjacency matrix
            cpt_info: (batch, n_nodes, cpt_dim) - CPTs (true for observed, zeros for unobserved)
            dimensions: (batch, n_nodes) - node indices
            
        Returns:
            predictions: (batch, n_nodes, n_states) - probability distributions
        """
        # Get updated parameter stream (imputed CPTs)
        parameter_stream = self.transformer(inputs, structure_info, cpt_info, dimensions)
        
        # Convert CPT parameters to state probabilities for each node
        predictions = []
        for i in range(self.n_nodes):
            node_probs = self.output_heads[i](parameter_stream[:, i, :])
            predictions.append(node_probs)
        
        # Stack: (batch, n_nodes, n_states)
        predictions = torch.stack(predictions, dim=1)
        
        return predictions

# ================================= DATASET AND TRAINING =================================

class ImputationDataset(Dataset):
    """Dataset for graph imputation."""
    
    def __init__(self, data, bn=None):
        self.data = data
        self.bn = bn
        
        if len(data) > 0:
            self.n_nodes = data[0][0].shape[0]
            self.input_dim = data[0][0].shape[1]
            self.structure_dim = data[0][1].shape[1]
            self.n_states = data[0][4].shape[1]
            
            # Compute max CPT size for consistency
            self.max_cpt_size = compute_max_cpt_size(bn) if bn else 8
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, structure_info, dimensions, mask, targets = self.data[idx]
        
        # Extract observed nodes
        observed_nodes = [i for i in range(self.n_nodes) if mask[i] == 0]
        
        # Extract CPTs (true for observed, zeros for unobserved)
        if self.bn is not None:
            cpt_info = extract_cpts_for_nodes(self.bn, observed_nodes, 
                                            self.n_nodes, self.max_cpt_size)
        else:
            cpt_info = np.zeros((self.n_nodes, self.max_cpt_size))
        
        return {
            'inputs': inputs,
            'structure_info': structure_info,
            'cpt_info': torch.FloatTensor(cpt_info),
            'dimensions': dimensions,
            'mask': mask,
            'targets': targets
        }

def collate_batch(batch):
    """Collate function for DataLoader."""
    return {
        'inputs': torch.stack([sample['inputs'] for sample in batch]),
        'structure_info': torch.stack([sample['structure_info'] for sample in batch]),
        'cpt_info': torch.stack([sample['cpt_info'] for sample in batch]),
        'dimensions': torch.stack([sample['dimensions'] for sample in batch]),
        'mask': torch.stack([sample['mask'] for sample in batch]),
        'targets': torch.stack([sample['targets'] for sample in batch])
    }

def compute_kl_loss(predictions, targets, mask):
    """Compute KL divergence loss: KL(true || pred) for unobserved nodes only."""
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

def train_epoch(model, train_loader, optimizer):
    """Train for one epoch."""
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
        loss = compute_kl_loss(predictions, targets, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, test_loader):
    """Validate for one epoch."""
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
            loss = compute_kl_loss(predictions, targets, mask)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# ================================= MODEL CREATION AND TRAINING =================================

def create_model(n_nodes, input_dim, structure_dim, cpt_dim=None):
    """Create imputation model with architecture scaled to graph size."""
    
    # Scale architecture based on graph size
    if n_nodes <= 10:
        hidden_dim_base = 64
        attention_heads = 4
        num_layers = 4
    else:
        hidden_dim_base = 128
        attention_heads = 8
        num_layers = 6
    
    # Default CPT dimension
    if cpt_dim is None:
        cpt_dim = 8  # Conservative for O(1) parents
    
    # Ensure divisibility by attention heads
    base_dim = structure_dim + (input_dim - 1)
    remainder = (hidden_dim_base + base_dim) % attention_heads
    hidden_dim = hidden_dim_base - remainder
    
    logger.info(f"Model architecture: hidden_dim={hidden_dim}, heads={attention_heads}, "
               f"layers={num_layers}, cpt_dim={cpt_dim}")
    
    model = GraphImputer(
        n_nodes=n_nodes,
        input_dim=input_dim,
        structure_dim=structure_dim,
        cpt_dim=cpt_dim,
        num_layers=num_layers,
        attention_heads=attention_heads,
        hidden_dim=hidden_dim,
        n_states=2,
        dropout=0.1
    ).to(DEVICE)
    
    return model

def train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=10):
    """Train the imputation model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate_epoch(model, test_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model

def evaluate_model(model, test_data, bn, n_nodes, n_states=2):
    """Evaluate imputation model and compute KL divergence."""
    logger.debug(f"Evaluating imputation model on {len(test_data)} test samples")
    
    model.eval()
    kl_divergences = []
    prediction_errors = []
    failed_inferences = 0
    
    # Get max CPT size for consistency
    max_cpt_size = compute_max_cpt_size(bn) if bn else 8
    
    for inputs, structure_info, dimensions, mask, targets in test_data:
        # Get unobserved nodes
        unobserved_nodes = [i for i in range(n_nodes) if mask[i] == 1]
        observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
        
        if not unobserved_nodes:
            continue
        
        # Extract CPTs for observed nodes
        cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            
        # Get predictions
        for node in unobserved_nodes:
            try:
                with torch.no_grad():
                    # Add batch dimension
                    inputs_batch = inputs.unsqueeze(0).to(DEVICE)
                    structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
                    cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
                    dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)
                    
                    predictions = model(inputs_batch, structure_info_batch, 
                                      cpt_info_batch, dimensions_batch)
                    pred_probs = predictions[0, node, :].cpu().numpy()
                
                # Get ground truth
                true_probs = targets[node].numpy()
                
                # Validate predictions
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    pred_probs = np.ones(n_states) / n_states
                else:
                    pred_probs = pred_probs / np.sum(pred_probs)
                
                if np.any(np.isnan(true_probs)) or np.sum(true_probs) == 0:
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
                    failed_inferences += 1
                    continue
                
                kl_divergences.append(kl)
                
                # Prediction error
                error = np.linalg.norm(pred_probs - true_probs)
                prediction_errors.append(error)
                
            except Exception as e:
                if len(kl_divergences) < 5:
                    logger.debug(f"Evaluation failed for node {node}: {str(e)[:100]}")
                failed_inferences += 1
                continue
    
    if not kl_divergences:
        logger.warning("No successful evaluations!")
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
    
    logger.info(f"Imputation evaluation: Mean KL = {results['mean_kl']:.4f}, "
               f"Failed rate = {results['failed_rate']:.2%}")
    
    return results