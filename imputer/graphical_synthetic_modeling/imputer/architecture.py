"""
Two-stream transformer architecture for progressive imputation.

Implements a transformer-based neural imputer that uses:
1. Embedding stream: Node embeddings + structure + observed evidence
2. Parameter stream: True CPTs for observed nodes, zeros for unobserved nodes

The model learns to impute missing CPT values through multi-layer processing
while maintaining privacy by never seeing true CPT values for unobserved nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for CPT extraction in neural imputer. "
        "Please install pyAgrum: pip install pyagrum"
    )

logger = logging.getLogger(__name__)

# Determine device for computation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Type alias for sample tuple
SampleTuple = Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]


# ================================= CPT PROCESSING =================================

def extract_cpts_for_nodes(bn: gum.BayesNet, observed_nodes: List[int], 
                          n_nodes: int, max_cpt_size: Optional[int] = None) -> np.ndarray:
    """
    Extract CPTs for observed nodes from BayesNet, zeros for unobserved nodes.
    
    This implements privacy-preserving CPT extraction where the neural model only
    sees true CPT values for observed nodes, maintaining the privacy constraint.
    
    Args:
        bn: pyAgrum BayesNet object
        observed_nodes: List of observed node indices  
        n_nodes: Total number of nodes
        max_cpt_size: Maximum CPT size for padding consistency
        
    Returns:
        np.ndarray: (n_nodes, max_cpt_size) with CPTs for observed, zeros for unobserved
    """
    if not PYAGRUM_AVAILABLE:
        logger.warning("pyAgrum not available, using default CPT size")
        return np.zeros((n_nodes, 8))
    
    logger.debug(f"Extracting CPTs for {len(observed_nodes)} observed nodes out of {n_nodes}")
    
    cpt_data = []
    cpt_sizes = []
    
    # Extract CPTs node by node
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
                # Fallback uniform distribution
                cpt_data.append(np.array([0.5, 0.5]))
                cpt_sizes.append(2)
        else:
            # ZEROS for unobserved nodes - critical for privacy preservation
            cpt_data.append(np.array([0.0, 0.0]))
            cpt_sizes.append(2)
    
    # Determine maximum CPT size for consistent padding
    if max_cpt_size is None:
        max_cpt_size = max(cpt_sizes) if cpt_sizes else 2
    
    logger.debug(f"Using max_cpt_size = {max_cpt_size} for padding")
    
    # Pad all CPTs to same size with zeros
    padded_cpts = np.zeros((n_nodes, max_cpt_size), dtype=np.float32)
    for i, cpt in enumerate(cpt_data):
        padded_cpts[i, :len(cpt)] = cpt
    
    return padded_cpts


def compute_max_cpt_size(bn: gum.BayesNet) -> int:
    """
    Compute maximum CPT size across all nodes for consistent tensor shapes.
    
    Args:
        bn: pyAgrum BayesNet object
        
    Returns:
        Maximum CPT size across all nodes
    """
    if not bn or not PYAGRUM_AVAILABLE:
        return 8  # Conservative default for O(1) parents
    
    max_size = 2  # Minimum size for binary variables
    
    for node_id in bn.nodes():
        node_str = str(node_id)
        cpt = bn.cpt(node_str)
        cpt_values = np.array(cpt.tolist()).flatten()
        max_size = max(max_size, len(cpt_values))
    
    logger.debug(f"Maximum CPT size in BN: {max_size}")
    return max_size


# ================================= NEURAL ARCHITECTURE =================================

class LayerNorm(nn.Module):
    """Layer normalization with learnable parameters."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    """Feed-forward network with ReLU activation and dropout."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 2)
        self.layer2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.dropout(F.relu(self.layer1(x))))


class PositionalEncoder(nn.Module):
    """
    Positional encoder that initializes the two-stream architecture.
    
    Creates embedding stream (node positions + structure + states) and
    parameter stream (CPT data with privacy masking).
    """
    
    def __init__(self, n_nodes: int, input_dim: int, structure_dim: int, 
                 cpt_dim: int, hidden_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.structure_dim = structure_dim
        self.cpt_dim = cpt_dim
        self.hidden_dim = hidden_dim
        
        # Learnable node embeddings for positional encoding
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        self.question_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Initialize embeddings with proper scaling
        torch.nn.init.kaiming_normal_(self.node_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')
        
        logger.debug(f"PositionalEncoder: n_nodes={n_nodes}, hidden_dim={hidden_dim}")

    def forward(self, inputs: torch.Tensor, structure_info: torch.Tensor, 
                cpt_info: torch.Tensor, dimensions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create embedding and parameter streams.
        
        Args:
            inputs: [batch, n_nodes, input_dim] - observed states with mask
            structure_info: [batch, n_nodes, structure_dim] - adjacency matrix
            cpt_info: [batch, n_nodes, cpt_dim] - CPTs (true for observed, zeros for unobserved)
            dimensions: [batch, n_nodes] - node indices
            
        Returns:
            Tuple of (embedding_stream, parameter_stream)
        """
        batch_size = inputs.shape[0]
        
        # Create positional embeddings
        node_embeds = self.node_embedding[dimensions]
        question_embeds = self.question_embedding.expand(batch_size, self.n_nodes, -1)
        combined_embeds = question_embeds + node_embeds
        
        # Embedding stream: node positions + structure + observed states (no mask bit)
        embedding_stream = torch.cat([
            combined_embeds,
            structure_info,
            inputs[:, :, 1:]  # Remove mask bit from inputs
        ], dim=-1)
        
        # Parameter stream: CPT data (true for observed, zeros for unobserved)
        parameter_stream = cpt_info
        
        return embedding_stream, parameter_stream


class TransformerLayer(nn.Module):
    """
    Two-stream transformer layer with cross-stream interactions.
    
    Processes embedding stream with self-attention and updates parameter stream
    using embedding context through cross-stream connections.
    """

    def __init__(self, embedding_dim: int, parameter_dim: int, 
                 attention_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.parameter_dim = parameter_dim
        self.attention_heads = attention_heads
        
        # Multi-head self-attention for embedding stream
        self.attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward for embedding stream
        self.ff_embedding = FeedForward(embedding_dim, dropout)
        
        # Cross-stream parameter update (embedding -> parameter)
        self.parameter_update = nn.Linear(embedding_dim + parameter_dim, parameter_dim)
        
        # Normalization layers
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(parameter_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_stream: torch.Tensor, parameter_stream: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both streams with attention and cross-stream updates.
        
        Args:
            embedding_stream: [batch, n_nodes, embedding_dim]
            parameter_stream: [batch, n_nodes, parameter_dim]
            mask: Optional attention mask for unobserved nodes
            
        Returns:
            Tuple of (updated_embedding_stream, updated_parameter_stream)
        """
        # Embedding stream processing
        # 1. Multi-head self-attention with residual connection
        embedding_norm = self.norm1(embedding_stream)
        attn_out, _ = self.attention(embedding_norm, embedding_norm, embedding_norm, 
                                   key_padding_mask=mask)
        embedding_stream = embedding_stream + self.dropout(attn_out)
        
        # 2. Feed-forward with residual connection
        embedding_norm = self.norm2(embedding_stream)
        ff_out = self.ff_embedding(embedding_norm)
        embedding_stream = embedding_stream + self.dropout(ff_out)
        
        # Parameter stream update using embedding context
        parameter_norm = self.norm3(parameter_stream)
        combined = torch.cat([embedding_stream, parameter_norm], dim=-1)
        parameter_stream = self.parameter_update(combined)
        
        return embedding_stream, parameter_stream


class TwoStreamTransformer(nn.Module):
    """
    Main two-stream transformer with embedding and parameter streams.
    
    The core architecture that processes both streams through multiple layers
    with self-attention and cross-stream parameter updates.
    """

    def __init__(self, n_nodes: int, input_dim: int, structure_dim: int, cpt_dim: int,
                 num_layers: int = 4, attention_heads: int = 4, hidden_dim: int = 64, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.cpt_dim = cpt_dim
        
        # Calculate embedding dimension (positions + structure + states)
        self.embedding_dim = hidden_dim + structure_dim + (input_dim - 1)
        
        logger.debug(f"TwoStreamTransformer: embedding_dim={self.embedding_dim}, "
                    f"parameter_dim={cpt_dim}, layers={num_layers}")
        
        # Positional encoder initializes both streams
        self.pos_encoder = PositionalEncoder(n_nodes, input_dim, structure_dim, 
                                           cpt_dim, hidden_dim)

        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.embedding_dim, cpt_dim, attention_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization layers
        self.final_norm_embedding = LayerNorm(self.embedding_dim)
        self.final_norm_parameter = LayerNorm(cpt_dim)

    def forward(self, inputs: torch.Tensor, structure_info: torch.Tensor, 
                cpt_info: torch.Tensor, dimensions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through two-stream transformer.
        
        Args:
            inputs: [batch, n_nodes, input_dim] - observed states with mask
            structure_info: [batch, n_nodes, structure_dim] - adjacency matrix
            cpt_info: [batch, n_nodes, cpt_dim] - CPTs with privacy masking
            dimensions: [batch, n_nodes] - node indices
            
        Returns:
            parameter_stream: [batch, n_nodes, cpt_dim] - Final CPT predictions
        """
        # Initialize both streams
        embedding_stream, parameter_stream = self.pos_encoder(
            inputs, structure_info, cpt_info, dimensions
        )
        
        # Extract attention mask (mask bit = 1 for unobserved nodes)
        mask = inputs[:, :, 0].bool()  # First bit is mask
        
        # Process through all transformer layers
        for layer in self.layers:
            embedding_stream, parameter_stream = layer(
                embedding_stream, parameter_stream, mask
            )
        
        # Apply final normalization
        embedding_stream = self.final_norm_embedding(embedding_stream)
        parameter_stream = self.final_norm_parameter(parameter_stream)
        
        return parameter_stream


class GraphImputer(nn.Module):
    """
    Complete graph imputation model with two-stream transformer.
    
    Converts imputed CPT parameters to final probability distributions
    through node-specific output heads.
    """
    
    def __init__(self, n_nodes: int = 5, input_dim: int = 3, structure_dim: Optional[int] = None, 
                 cpt_dim: Optional[int] = None, num_layers: int = 4, attention_heads: int = 4, 
                 hidden_dim: int = 64, n_states: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_states = n_states
        
        # Default structure dimension (adjacency matrix columns)
        if structure_dim is None:
            structure_dim = n_nodes
        
        # Default CPT dimension (conservative for O(1) parents)
        if cpt_dim is None:
            cpt_dim = 8
        
        self.cpt_dim = cpt_dim
        
        # Main two-stream transformer
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
        
        # Node-specific output heads: CPT parameters -> probability distributions
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
    
    def forward(self, inputs: torch.Tensor, structure_info: torch.Tensor, 
                cpt_info: torch.Tensor, dimensions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete imputation model.
        
        Args:
            inputs: [batch, n_nodes, input_dim] - observed states with mask
            structure_info: [batch, n_nodes, structure_dim] - adjacency matrix
            cpt_info: [batch, n_nodes, cpt_dim] - CPTs (true for observed, zeros for unobserved)
            dimensions: [batch, n_nodes] - node indices
            
        Returns:
            predictions: [batch, n_nodes, n_states] - probability distributions for all nodes
        """
        # Get updated parameter stream (imputed CPTs)
        parameter_stream = self.transformer(inputs, structure_info, cpt_info, dimensions)
        
        # Convert CPT parameters to state probabilities for each node
        predictions = []
        for i in range(self.n_nodes):
            node_probs = self.output_heads[i](parameter_stream[:, i, :])
            predictions.append(node_probs)
        
        # Stack predictions: [batch, n_nodes, n_states]
        predictions = torch.stack(predictions, dim=1)
        
        return predictions


class ImputationDataset(Dataset):
    """
    Dataset wrapper for imputation training data.
    
    Handles batching and CPT extraction for training samples.
    """
    
    def __init__(self, samples: List[SampleTuple], bn: gum.BayesNet):
        self.samples = samples
        self.bn = bn
        self.n_nodes = len(samples[0][2])  # Get from dimensions tensor
        self.max_cpt_size = compute_max_cpt_size(bn)
        
        logger.debug(f"ImputationDataset: {len(samples)} samples, "
                    f"{self.n_nodes} nodes, max_cpt_size={self.max_cpt_size}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, structure_info, dimensions, mask, targets = self.samples[idx]
        
        # Extract observed nodes (mask == 0)
        observed_nodes = [i for i in range(self.n_nodes) if mask[i] == 0]
        
        # Extract CPTs with privacy masking
        cpt_info = extract_cpts_for_nodes(self.bn, observed_nodes, self.n_nodes, self.max_cpt_size)
        cpt_tensor = torch.FloatTensor(cpt_info)
        
        return inputs, structure_info, dimensions, mask, targets, cpt_tensor


def collate_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for DataLoader to handle batching.
    
    Args:
        batch: List of sample tuples from ImputationDataset
        
    Returns:
        Batched tensors ready for model forward pass
    """
    inputs, structure_info, dimensions, masks, targets, cpt_info = zip(*batch)
    
    return (
        torch.stack(inputs),
        torch.stack(structure_info), 
        torch.stack(dimensions),
        torch.stack(masks),
        torch.stack(targets),
        torch.stack(cpt_info)
    )


# ================================= MODEL CONFIGURATION =================================

def create_model(n_nodes: int, input_dim: int, structure_dim: int, 
                cpt_dim: Optional[int] = None, model_size: str = "Large") -> GraphImputer:
    """
    Create imputation model with configurable architecture size.
    
    Args:
        n_nodes: Number of nodes in the graph
        input_dim: Input dimension (typically 3 for [mask, state_0, state_1])
        structure_dim: Structure dimension (typically n_nodes for adjacency)
        cpt_dim: CPT dimension (auto-computed if None)
        model_size: Architecture size ("Tiny", "Small", "Large")
        
    Returns:
        Configured GraphImputer model
    """
    # Define architecture variants
    model_configs = {
        "Tiny": {
            "hidden_dim_base": 32,
            "attention_heads": 2,
            "num_layers": 2
        },
        "Small": {
            "hidden_dim_base": 64,
            "attention_heads": 4,
            "num_layers": 3
        },
        "Large": {
            "hidden_dim_base": 128,
            "attention_heads": 8,
            "num_layers": 4
        }
    }
    
    # Validate and get configuration
    if model_size not in model_configs:
        logger.warning(f"Unknown model size '{model_size}', using 'Large'")
        model_size = "Large"
    
    config = model_configs[model_size]
    hidden_dim_base = config["hidden_dim_base"]
    attention_heads = config["attention_heads"]
    num_layers = config["num_layers"]
    
    # Default CPT dimension
    if cpt_dim is None:
        cpt_dim = 8  # Conservative default for O(1) parents
    
    # Ensure embedding dimension is divisible by attention heads
    base_dim = structure_dim + (input_dim - 1)
    remainder = (hidden_dim_base + base_dim) % attention_heads
    hidden_dim = hidden_dim_base - remainder
    
    logger.info(f"Model architecture: hidden_dim={hidden_dim}, heads={attention_heads}, "
               f"layers={num_layers}, cpt_dim={cpt_dim}")
    
    # Create model and move to device
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