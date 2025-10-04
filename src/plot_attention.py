#!/usr/bin/env python3
"""
Standalone script to plot attention patterns from a trained neural model.

Usage:
    python plot_attention.py --model_path models/neural_model_100_prefix_dev_10_obs50.pth --data_file prefix_dev_10_obs50_new.json
    python plot_attention.py --model_path models/neural_model_100_prefix_dev_10_obs50.pth --data_file prefix_dev_10_obs50_new.json --sample_idx 5 --output_dir attention_plots
    python plot_attention.py --model_path models/neural_model_100_prefix_dev_10_obs50.pth --data_file prefix_dev_10_obs50_new.json --average_all
"""

import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class AttentionCapture:
    """Context manager for capturing attention weights from transformer layers."""
    
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
        """Register hooks on attention layers."""
        logger.info("Exploring model structure:")
        self._print_model_structure(self.model, max_depth=4)
        
        attention_layers = self._find_attention_layers(self.model)
        
        if attention_layers:
            logger.info(f"Found {len(attention_layers)} attention layers")
            for layer_name, layer_module in attention_layers:
                if hasattr(layer_module, 'Q') and hasattr(layer_module, 'K') and hasattr(layer_module, 'V'):
                    hook = layer_module.register_forward_hook(
                        self._create_custom_attention_hook(layer_name, layer_module)
                    )
                    logger.info(f"Registered custom attention hook for {layer_name}")
                else:
                    hook = layer_module.register_forward_hook(
                        self._create_attention_hook(layer_name)
                    )
                    logger.info(f"Registered standard attention hook for {layer_name}")
                self.hooks.append(hook)
        else:
            logger.warning("Could not find any attention layers in model")
    
    def _print_model_structure(self, module, prefix="", max_depth=3, current_depth=0):
        """Print the structure of the model."""
        if current_depth > max_depth:
            return
            
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            module_type = type(child).__name__
            
            print(f"{'  ' * current_depth}{full_name}: {module_type}")
            
            attention_attrs = []
            if hasattr(child, 'attention'):
                attention_attrs.append('attention')
            if hasattr(child, 'self_attn'):
                attention_attrs.append('self_attn')
            if hasattr(child, 'Q') and hasattr(child, 'K') and hasattr(child, 'V'):
                attention_attrs.append('Q,K,V')
            
            if attention_attrs:
                print(f"{'  ' * current_depth}  -> Has: {', '.join(attention_attrs)}")
            
            self._print_model_structure(child, full_name, max_depth, current_depth + 1)
    
    def _find_attention_layers(self, model):
        """Find all attention layers in the model."""
        attention_layers = []
        
        # Strategy 1: Look for encoder.layers pattern (custom architecture)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            logger.info("Found encoder.layers - checking for Q,K,V components")
            for i, layer in enumerate(model.encoder.layers):
                if hasattr(layer, 'Q') and hasattr(layer, 'K') and hasattr(layer, 'V'):
                    attention_layers.append((f'encoder_layer_{i}', layer))
        
        # Strategy 2: Look for transformer.layers pattern
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            logger.info("Found transformer.layers")
            for i, layer in enumerate(model.transformer.layers):
                if hasattr(layer, 'attention'):
                    attention_layers.append((f'transformer_layer_{i}', layer.attention))
                elif hasattr(layer, 'self_attn'):
                    attention_layers.append((f'transformer_layer_{i}', layer.self_attn))
        
        return attention_layers
    
    def _create_attention_hook(self, layer_name: str):
        """Create hook for standard attention layers."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn_output, attn_weights = output
                if attn_weights is not None:
                    self.attention_weights[layer_name] = attn_weights.detach().cpu()
        return hook_fn
    
    def _create_custom_attention_hook(self, layer_name: str, layer_module):
        """Create custom hook for layers with separate Q, K, V components."""
        def hook_fn(module, input, output):
            try:
                x = input[0] if isinstance(input, tuple) else input
                batch_size, seq_len, hidden_dim = x.shape
                
                if not (hasattr(module, 'Q') and hasattr(module, 'K') and hasattr(module, 'V')):
                    return
                
                with torch.no_grad():
                    Q = module.Q(x)
                    K = module.K(x)
                    V = module.V(x)
                    
                    attention_scores = torch.bmm(Q, K.transpose(-2, -1))
                    d_k = Q.size(-1)
                    attention_scores = attention_scores / np.sqrt(d_k)
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    
                    self.attention_weights[layer_name] = attention_weights.unsqueeze(1).detach().cpu()
            except Exception as e:
                logger.warning(f"Failed to capture attention for {layer_name}: {e}")
        return hook_fn
    
    def get_averaged_attention(self) -> Dict[str, torch.Tensor]:
        """Get attention weights averaged across heads for each layer."""
        averaged_weights = {}
        for layer_name, weights in self.attention_weights.items():
            averaged_weights[layer_name] = weights.mean(dim=1)
        return averaged_weights
    
    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights.clear()


class GaussianDataset:
    """Dataset wrapper for loading Gaussian data."""
    
    def __init__(self, data_path: str):
        """Initialize dataset from JSON file."""
        with open(data_path, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                self.data = content['data'][:100]
                self.metadata = content.get('metadata', {})
            else:
                self.data = content
                self.metadata = {}
        
        logger.info(f"Loaded dataset with {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def get_neural_input(self, idx: int):
        """Get data in format expected by neural model."""
        entry = self.data[idx]
        
        known_questions = torch.tensor(entry['known_questions'], dtype=torch.float32)
        inputs = torch.tensor(entry['input'], dtype=torch.float32)
        answers = torch.tensor(entry['answers'], dtype=torch.float32)
        annotators = torch.tensor(entry['annotators'], dtype=torch.long)
        questions = torch.tensor(entry['questions'], dtype=torch.long)
        
        return known_questions, inputs, answers, annotators, questions
    
    def get_omega_matrix(self) -> Optional[np.ndarray]:
        """Get Omega precision matrix from metadata."""
        if self.metadata and 'Omega' in self.metadata:
            omega = np.array(self.metadata['Omega'])
            logger.info(f"Loaded Omega precision matrix: {omega.shape} with range [{omega.min():.3f}, {omega.max():.3f}]")
            return np.linalg.inv(omega)
        elif self.metadata and 'true_parameters' in self.metadata and 'Omega' in self.metadata['true_parameters']:
            omega = np.array(self.metadata['true_parameters']['Omega'])
            logger.info(f"Loaded Omega from true_parameters: {omega.shape} with range [{omega.min():.3f}, {omega.max():.3f}]")
            return np.linalg.inv(omega)
        else:
            logger.warning("No Omega precision matrix found in metadata")
            return None


class NeuralModelLoader:
    """Handles loading neural models from .pth files."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.model_scale = None
    
    def load_model(self, model_path: str):
        """Load neural model from .pth file."""
        try:
            from imputer_gaussian import ImputerEmbedding
        except ImportError:
            logger.error("Could not import ImputerEmbedding. Make sure imputer_gaussian.py is available.")
            raise
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['model_config']
        
        self.model_scale = self._determine_model_scale(model_config, model_path)
        
        if self.model_scale == "large":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=6,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=128,
                dropout=0.1,
            ).to(self.device)
        elif self.model_scale == "small":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=4,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=64,
                dropout=0.1,
            ).to(self.device)
        elif self.model_scale == "tiny":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=2,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=32,
                dropout=0.1,
            ).to(self.device)
        else:
            self.model = ImputerEmbedding(**model_config).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Neural model ({self.model_scale}) loaded successfully")
        return self.model
    
    def _determine_model_scale(self, config: dict, model_path: str) -> str:
        """Determine model scale from config or filename."""
        filename = os.path.basename(model_path).lower()
        if 'large' in filename:
            return 'large'
        elif 'small' in filename:
            return 'small'
        elif 'tiny' in filename:
            return 'tiny'
        
        if 'encoder_layers_num' in config:
            layers = config['encoder_layers_num']
            if layers >= 6:
                return 'large'
            elif layers >= 4:
                return 'small'
            else:
                return 'tiny'
        
        return 'medium'


def create_attention_heatmaps(attention_weights: Dict[str, torch.Tensor],
                            observed_mask: torch.Tensor,
                            omega_matrix: Optional[np.ndarray] = None,
                            sample_idx: int = 0,
                            output_dir: str = "attention_plots",
                            suffix: str = "") -> None:
    """Create attention heatmaps with optional Omega matrix comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(attention_weights.keys())
    n_layers = len(layer_names)
    
    if n_layers == 0:
        logger.warning("No attention weights captured")
        return
    
    first_attention = attention_weights[layer_names[0]][sample_idx]
    n_nodes = first_attention.shape[0]
    
    has_omega = omega_matrix is not None
    total_plots = n_layers + (1 if has_omega else 0)
    
    # Create subplot grid
    if total_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        cols = min(3, total_plots)
        rows = (total_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
    
    # Set title
    observed_nodes = observed_mask[sample_idx].bool()
    n_observed = observed_nodes.sum().item()
    
    if "_averaged" in suffix:
        title = f'Attention Patterns vs Sigma Matrix'
    else:
        sample_num = suffix.split("_sample_")[-1] if "_sample_" in suffix else str(sample_idx)
        title = f'Attention Patterns vs Sigma Matrix'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    # Plot Omega matrix first if available
    if has_omega:
        ax = axes[plot_idx]
        im_omega = ax.imshow(omega_matrix, cmap='RdBu_r', aspect='equal', 
                           vmin=-np.abs(omega_matrix).max(), vmax=np.abs(omega_matrix).max())
        ax.set_title('Omega Precision Matrix', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels([f'N{j}' for j in range(n_nodes)])
        ax.set_yticklabels([f'N{j}' for j in range(n_nodes)])
        
        # Add value annotations for significant entries
        for i in range(n_nodes):
            for j in range(n_nodes):
                if abs(omega_matrix[i, j]) > 0.1:
                    color = 'white' if abs(omega_matrix[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{omega_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8, color=color)
        
        plot_idx += 1
    
    # Plot attention layers
    for i, layer_name in enumerate(layer_names):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        attention = attention_weights[layer_name][sample_idx]
        attention_np = attention.numpy()
        
        im = ax.imshow(attention_np, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        ax.set_title(f'Layer {i}', fontsize=12)
        
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels([f'N{j}' for j in range(n_nodes)])
        ax.set_yticklabels([f'N{j}' for j in range(n_nodes)])
        
        # Highlight observed nodes for single samples
        if not "_averaged" in suffix:
            observed_indices = torch.where(observed_nodes)[0].numpy()
            for obs_idx in observed_indices:
                ax.axhline(y=obs_idx-0.5, color='red', linewidth=2, alpha=0.7)
                ax.axhline(y=obs_idx+0.5, color='red', linewidth=2, alpha=0.7)
                ax.axvline(x=obs_idx-0.5, color='red', linewidth=2, alpha=0.7)
                ax.axvline(x=obs_idx+0.5, color='red', linewidth=2, alpha=0.7)
        
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbars
    plt.tight_layout()
    if has_omega:
        cbar_attention = fig.colorbar(im, ax=axes[1:plot_idx], shrink=0.6, aspect=20, pad=0.02)
        cbar_attention.set_label('Attention Weight', rotation=270, labelpad=15)
        
        cbar_omega = fig.colorbar(im_omega, ax=axes[0], shrink=0.6, aspect=10, pad=0.02)
        cbar_omega.set_label('Omega Value', rotation=270, labelpad=15)
    else:
        cbar = fig.colorbar(im, ax=axes[:plot_idx], shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Save plot
    output_file = f"{output_dir}/attention_heatmaps{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Attention heatmaps saved to {output_file}")


def plot_model_attention(model_path: str, data_file: str, sample_idx: int = 0, 
                        output_dir: str = "attention_plots", average_all: bool = False,
                        max_samples: int = 100) -> None:
    """Main function to plot attention patterns from a neural model."""
    
    # Load dataset and model
    dataset = GaussianDataset(data_file)
    model_loader = NeuralModelLoader(device)
    model = model_loader.load_model(model_path)
    
    # Get Omega matrix
    omega_matrix = dataset.get_omega_matrix()
    
    if average_all:
        logger.info(f"Averaging attention across {min(max_samples, len(dataset))} samples")
        attention_weights_all = {}
        observed_mask_all = []
        
        num_samples = min(max_samples, len(dataset))
        
        with AttentionCapture(model) as attention_capture:
            for idx in range(num_samples):
                if idx % 10 == 0:
                    logger.info(f"Processing sample {idx}/{num_samples}")
                
                known_questions, inputs, answers, annotators, questions = dataset.get_neural_input(idx)
                observed_mask = known_questions.unsqueeze(0)
                observed_mask_all.append(observed_mask)
                
                inputs = inputs.unsqueeze(0).to(device)
                annotators = annotators.unsqueeze(0).to(device)
                questions = questions.unsqueeze(0).to(device)
                
                attention_capture.clear()
                with torch.no_grad():
                    logits, _ = model(inputs, annotators, questions)
                
                sample_attention = attention_capture.get_averaged_attention()
                
                for layer_name, attention in sample_attention.items():
                    if layer_name not in attention_weights_all:
                        attention_weights_all[layer_name] = []
                    attention_weights_all[layer_name].append(attention[0])
        
        # Average across samples
        averaged_attention = {}
        for layer_name, attention_list in attention_weights_all.items():
            if attention_list:
                stacked_attention = torch.stack(attention_list, dim=0)
                averaged_attention[layer_name] = stacked_attention.mean(dim=0).unsqueeze(0)
        
        observed_mask_stacked = torch.cat(observed_mask_all, dim=0)
        avg_observed_mask = (observed_mask_stacked.mean(dim=0) > 0.5).float().unsqueeze(0)
        
        create_attention_heatmaps(
            attention_weights=averaged_attention,
            observed_mask=avg_observed_mask,
            omega_matrix=omega_matrix,
            sample_idx=0,
            output_dir=output_dir,
            suffix="_averaged"
        )
        
    else:
        if sample_idx >= len(dataset):
            raise ValueError(f"Sample index {sample_idx} is out of range. Dataset has {len(dataset)} samples.")
        
        known_questions, inputs, answers, annotators, questions = dataset.get_neural_input(sample_idx)
        observed_mask = known_questions.unsqueeze(0)
        
        inputs = inputs.unsqueeze(0).to(device)
        annotators = annotators.unsqueeze(0).to(device)
        questions = questions.unsqueeze(0).to(device)
        
        logger.info(f"Processing sample {sample_idx}")
        logger.info(f"Observed nodes: {observed_mask.sum().item()}/{observed_mask.shape[1]}")
        
        with AttentionCapture(model) as attention_capture:
            with torch.no_grad():
                logits, _ = model(inputs, annotators, questions)
                
            attention_weights = attention_capture.get_averaged_attention()
        
        if not attention_weights:
            logger.error("No attention weights captured")
            return
        
        create_attention_heatmaps(
            attention_weights=attention_weights,
            observed_mask=observed_mask,
            omega_matrix=omega_matrix,
            sample_idx=0,
            output_dir=output_dir,
            suffix=f"_sample_{sample_idx}"
        )
    
    logger.info(f"Attention plots saved to {output_dir}/")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Plot attention patterns from a neural model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the .pth model file")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to the data JSON file")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Index of the sample to analyze (ignored if --average_all is used)")
    parser.add_argument("--output_dir", type=str, default="attention_plots",
                       help="Directory to save plots")
    parser.add_argument("--average_all", action="store_true",
                       help="Average attention across all samples")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to process when averaging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    plot_model_attention(
        model_path=args.model_path,
        data_file=args.data_file,
        sample_idx=args.sample_idx,
        output_dir=args.output_dir,
        average_all=args.average_all,
        max_samples=args.max_samples
    )
    
    print(f"\nAttention plotting completed!")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    if args.average_all:
        print(f"Mode: Averaged across {args.max_samples} samples")
    else:
        print(f"Sample: {args.sample_idx}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()