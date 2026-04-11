"""
Attention pattern analysis for progressive imputation experiments.

Captures and analyzes attention weights from the two-stream transformer to understand
how the model learns structural dependencies and respects privacy constraints.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AttentionCapture:
    """
    Context manager for capturing attention weights from transformer layers.
    
    Registers forward hooks to capture attention weights during inference,
    then averages across attention heads for interpretable analysis.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
    
    def __enter__(self):
        """Register forward hooks for attention capture."""
        self._register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _register_hooks(self):
        """Register hooks on MultiheadAttention layers to capture attention weights."""
        for i, layer in enumerate(self.model.transformer.layers):
            hook = layer.attention.register_forward_hook(
                self._create_attention_hook(f'layer_{i}')
            )
            self.hooks.append(hook)
    
    def _create_attention_hook(self, layer_name: str):
        """Create hook function for specific layer."""
        def hook_fn(module, input, output):
            # MultiheadAttention returns (output, attention_weights)
            if len(output) == 2:
                attn_output, attn_weights = output
                if attn_weights is not None:
                    # Store attention weights: [batch_size, n_heads, n_nodes, n_nodes]
                    self.attention_weights[layer_name] = attn_weights.detach().cpu()
        return hook_fn
    
    def get_averaged_attention(self) -> Dict[str, torch.Tensor]:
        """
        Get attention weights averaged across heads for each layer.
        
        Returns:
            Dict mapping layer_name to averaged attention matrix [batch_size, n_nodes, n_nodes]
        """
        averaged_weights = {}
        for layer_name, weights in self.attention_weights.items():
            # Average across attention heads (dim=1)
            # Input: [batch_size, n_heads, n_nodes, n_nodes]
            # Output: [batch_size, n_nodes, n_nodes]
            averaged_weights[layer_name] = weights.mean(dim=1)
        return averaged_weights
    
    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights.clear()


def analyze_attention_patterns(attention_weights: Dict[str, torch.Tensor], 
                              adjacency_matrix: np.ndarray,
                              observed_mask: torch.Tensor,
                              sample_idx: int = 0) -> Dict[str, Any]:
    """
    Analyze attention patterns for structural discovery and privacy compliance.
    
    Args:
        attention_weights: Dict of layer_name -> averaged attention [batch, n_nodes, n_nodes]
        adjacency_matrix: True BN adjacency matrix [n_nodes, n_nodes]
        observed_mask: Mask indicating observed nodes [batch, n_nodes] (1=observed, 0=unobserved)  
        sample_idx: Which sample to analyze
        
    Returns:
        Dictionary with attention pattern statistics
    """
    n_nodes = adjacency_matrix.shape[0]
    observed_nodes = observed_mask[sample_idx].bool()
    
    analysis = {
        'n_nodes': n_nodes,
        'n_observed': observed_nodes.sum().item(),
        'layer_patterns': {},
        'privacy_compliance': {},
        'structural_discovery': {}
    }
    
    # Analyze each layer
    for layer_name, attention in attention_weights.items():
        layer_attn = attention[sample_idx]  # [n_nodes, n_nodes]
        
        # Basic attention statistics
        layer_analysis = {
            'attention_entropy': compute_attention_entropy(layer_attn),
            'attention_focus': compute_attention_focus(layer_attn),
            'max_attention': layer_attn.max().item(),
            'min_attention': layer_attn.min().item()
        }
        
        # Structural discovery: attention to true parents
        parent_attention_ratio = compute_parent_attention_ratio(
            layer_attn, adjacency_matrix
        )
        layer_analysis['parent_attention_ratio'] = parent_attention_ratio
        
        # Privacy compliance: unobserved nodes shouldn't attend to "forbidden" information
        privacy_score = compute_privacy_compliance_score(
            layer_attn, observed_nodes
        )
        layer_analysis['privacy_compliance_score'] = privacy_score
        
        analysis['layer_patterns'][layer_name] = layer_analysis
    
    # Cross-layer analysis
    analysis['layer_progression'] = analyze_layer_progression(attention_weights, sample_idx)
    
    return analysis


def compute_attention_entropy(attention_matrix: torch.Tensor) -> float:
    """
    Compute average attention entropy across all query nodes.
    Higher entropy = more distributed attention, Lower entropy = more focused attention.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    attention_probs = attention_matrix + eps
    
    # Compute entropy for each row (query node)
    log_probs = torch.log(attention_probs)
    entropy_per_node = -(attention_probs * log_probs).sum(dim=1)  # [n_nodes]
    
    return entropy_per_node.mean().item()


def compute_attention_focus(attention_matrix: torch.Tensor) -> float:
    """
    Compute attention focus as 1 - normalized_entropy.
    Higher focus = attention concentrated on few nodes.
    """
    n_nodes = attention_matrix.shape[1]
    max_entropy = np.log(n_nodes)  # Maximum possible entropy for uniform distribution
    
    entropy = compute_attention_entropy(attention_matrix)
    focus = 1.0 - (entropy / max_entropy)  # Normalized to [0, 1]
    
    return focus


def compute_parent_attention_ratio(attention_matrix: torch.Tensor, 
                                 adjacency_matrix: np.ndarray) -> float:
    """
    Compute fraction of attention that goes to true parent nodes.
    
    For each node, compute how much of its attention goes to its true parents
    versus non-parent nodes.
    """
    n_nodes = attention_matrix.shape[0]
    parent_ratios = []
    
    for child_idx in range(n_nodes):
        # Find true parents of this child node
        true_parents = np.where(adjacency_matrix[:, child_idx] == 1)[0]
        
        if len(true_parents) > 0:
            # Sum attention to true parents
            parent_attention = attention_matrix[child_idx, true_parents].sum().item()
            total_attention = attention_matrix[child_idx].sum().item()
            
            if total_attention > 0:
                ratio = parent_attention / total_attention
                parent_ratios.append(ratio)
    
    return np.mean(parent_ratios) if parent_ratios else 0.0


def compute_privacy_compliance_score(attention_matrix: torch.Tensor,
                                   observed_nodes: torch.Tensor) -> float:
    """
    Compute privacy compliance score.
    
    Unobserved nodes should not attend strongly to other unobserved nodes
    since they don't have access to their true state information.
    
    Returns score in [0, 1] where 1 = perfect compliance.
    """
    unobserved_mask = ~observed_nodes  # [n_nodes]
    
    if unobserved_mask.sum() <= 1:
        return 1.0  # Perfect compliance if ≤1 unobserved node
    
    # Get attention from unobserved nodes to other unobserved nodes
    unobserved_indices = torch.where(unobserved_mask)[0]
    
    forbidden_attention = 0.0
    total_pairs = 0
    
    for i in unobserved_indices:
        for j in unobserved_indices:
            if i != j:  # Don't count self-attention
                forbidden_attention += attention_matrix[i, j].item()
                total_pairs += 1
    
    if total_pairs == 0:
        return 1.0
    
    avg_forbidden_attention = forbidden_attention / total_pairs
    
    # Score is 1 - average_forbidden_attention (higher is better)
    return max(0.0, 1.0 - avg_forbidden_attention)


def analyze_layer_progression(attention_weights: Dict[str, torch.Tensor], 
                            sample_idx: int = 0) -> Dict[str, Any]:
    """
    Analyze how attention patterns evolve across layers.
    """
    layer_names = sorted(attention_weights.keys())  # ['layer_0', 'layer_1', ...]
    
    progression = {
        'entropy_progression': [],
        'focus_progression': [],
        'attention_similarity': {}
    }
    
    prev_attention = None
    
    for layer_name in layer_names:
        attention = attention_weights[layer_name][sample_idx]
        
        # Track entropy and focus progression
        entropy = compute_attention_entropy(attention)
        focus = compute_attention_focus(attention)
        
        progression['entropy_progression'].append(entropy)
        progression['focus_progression'].append(focus)
        
        # Compute similarity to previous layer
        if prev_attention is not None:
            similarity = torch.cosine_similarity(
                attention.flatten(), 
                prev_attention.flatten(), 
                dim=0
            ).item()
            progression['attention_similarity'][layer_name] = similarity
        
        prev_attention = attention
    
    return progression


def create_attention_heatmaps(attention_weights: Dict[str, torch.Tensor],
                            adjacency_matrix: np.ndarray,
                            observed_mask: torch.Tensor,
                            sample_idx: int = 0,
                            output_dir: str = "attention_analysis") -> None:
    """
    Create clean layer-by-layer attention heatmaps.
    
    Args:
        attention_weights: Dict of layer_name -> averaged attention matrices
        adjacency_matrix: True BN structure for comparison
        observed_mask: Node observation status
        sample_idx: Which sample to visualize
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(attention_weights.keys())
    n_layers = len(layer_names)
    
    # Get number of nodes from first attention matrix
    first_attention = attention_weights[layer_names[0]][sample_idx]
    n_nodes = first_attention.shape[0]
    
    # Create subplot grid based on number of layers
    if n_layers <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        cols = 3
        rows = (n_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()
    
    # Set title
    observed_nodes = observed_mask[sample_idx].bool()
    n_observed = observed_nodes.sum().item()
    fig.suptitle('MHA Pattern By Layer', 
                 fontsize=16, fontweight='bold')
    
    for i, layer_name in enumerate(layer_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        attention = attention_weights[layer_name][sample_idx]  # [n_nodes, n_nodes]
        attention_np = attention.numpy()
        
        # Clean heatmap with viridis colormap
        im = ax.imshow(attention_np, cmap='viridis', aspect='equal')
        
        # Simple layer title
        ax.set_title(f'Layer {i}', fontsize=12)
        
        # Clean axis labels
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels([f'N{j}' for j in range(n_nodes)])
        ax.set_yticklabels([f'N{j}' for j in range(n_nodes)])
    
    # Hide unused subplots
    for i in range(len(layer_names), len(axes)):
        axes[i].set_visible(False)
    
    # Add single colorbar on the right side
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    plt.savefig(f"{output_dir}/attention_heatmaps_sample_{sample_idx}.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Clean attention heatmaps saved to {output_dir}/attention_heatmaps_sample_{sample_idx}.png")
    plt.close()


def save_attention_analysis(analysis: Dict[str, Any], 
                          output_dir: str = "attention_analysis",
                          sample_idx: int = 0) -> None:
    """Save attention analysis results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert any torch tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_data = convert_for_json(analysis)
    
    output_file = f"{output_dir}/attention_analysis_sample_{sample_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Attention analysis saved to {output_file}")