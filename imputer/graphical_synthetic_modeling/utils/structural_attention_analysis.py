"""
Structural attention analysis for transformer imputation models.

Analyzes whether the model learns to attend along graph structure (parents/children)
by averaging attention patterns across diverse missing patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})


class AttentionCapture:
    """Capture attention weights during forward pass."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def __enter__(self):
        # Register hooks on all MultiheadAttention modules
        for name, module in self.model.named_modules():
            if hasattr(module, 'in_proj_weight'):  # MultiheadAttention module
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()

    def _create_hook(self, name):
        def hook_fn(module, input, output):
            # MultiheadAttention returns (output, attention_weights)
            if isinstance(output, tuple) and len(output) == 2:
                attention = output[1]
                if attention is not None:
                    self.attention_weights[name] = attention.detach().cpu()
        return hook_fn

    def get_attention_weights(self):
        return self.attention_weights


def load_model_and_data(model_dir: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load saved model and associated data."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from utils.model_saving import load_model_for_attention_analysis, reconstruct_model_from_saved

    saved_data = load_model_for_attention_analysis(model_dir)
    model = reconstruct_model_from_saved(saved_data)
    model.eval()

    return model, saved_data


def find_common_missing_pattern(test_samples: List[Dict], n_nodes: int, min_count: int = 20) -> Tuple[List[int], List[int]]:
    """Find a missing pattern that occurs frequently across samples."""

    # Count missing patterns
    pattern_counts = {}

    for sample in test_samples:
        mask = sample['mask']
        missing_nodes = tuple(sorted([i for i in range(n_nodes) if mask[i] == 1]))
        observed_nodes = tuple(sorted([i for i in range(n_nodes) if mask[i] == 0]))
        pattern = (observed_nodes, missing_nodes)

        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Find most common pattern with enough samples
    best_pattern = None
    best_count = 0

    for pattern, count in pattern_counts.items():
        if count >= min_count and count > best_count:
            best_pattern = pattern
            best_count = count

    # Fallback to most common pattern if none meet minimum
    if best_pattern is None:
        best_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        best_count = pattern_counts[best_pattern]

    observed_nodes, missing_nodes = best_pattern
    logger.info(f"Selected missing pattern: observed={list(observed_nodes)}, missing={list(missing_nodes)} ({best_count} samples)")

    return list(observed_nodes), list(missing_nodes)


def analyze_structural_attention(model: torch.nn.Module,
                               saved_data: Dict[str, Any],
                               n_samples: int = 200) -> Dict[str, Any]:
    """
    Analyze attention patterns for a consistent missing pattern.

    Args:
        model: Trained GraphImputer model
        saved_data: Data from model loading
        n_samples: Number of test samples to analyze

    Returns:
        Dictionary with structural attention analysis results
    """
    adjacency_matrix = saved_data['adjacency_matrix']
    test_samples = saved_data['test_samples']
    n_nodes = adjacency_matrix.shape[0]

    # Limit samples to what's available
    n_samples = min(n_samples, len(test_samples))

    logger.info(f"Analyzing structural attention across {n_samples} samples with {n_nodes} nodes")

    # Find a common missing pattern
    target_observed, target_missing = find_common_missing_pattern(test_samples, n_nodes)

    # Filter samples to only those with the target missing pattern
    matching_samples = []
    for sample in test_samples:
        mask = sample['mask']
        observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
        missing_nodes = [i for i in range(n_nodes) if mask[i] == 1]

        if sorted(observed_nodes) == sorted(target_observed) and sorted(missing_nodes) == sorted(target_missing):
            matching_samples.append(sample)

    logger.info(f"Found {len(matching_samples)} samples with consistent missing pattern")

    # Use first N matching samples
    use_samples = min(n_samples, len(matching_samples))
    matching_samples = matching_samples[:use_samples]

    # Accumulate attention patterns
    layer_attention_sums = {}  # layer_name -> accumulated attention matrix [n_nodes, n_nodes]
    layer_counts = {}

    for sample_idx, sample in enumerate(matching_samples):
        if sample_idx % 20 == 0:
            logger.info(f"Processing sample {sample_idx}/{len(matching_samples)}")

        # Convert to tensors
        inputs = torch.FloatTensor(sample['inputs']).unsqueeze(0)
        structure_info = torch.FloatTensor(sample['structure_info']).unsqueeze(0)
        dimensions = torch.LongTensor(sample['dimensions']).unsqueeze(0)
        mask = torch.FloatTensor(sample['mask']).unsqueeze(0)

        # For CPT info, reconstruct properly for observed nodes
        observed_nodes = [i for i in range(n_nodes) if mask[0, i] == 0]

        # Extract CPTs using the same method as during training
        from imputer.architecture import extract_cpts_for_nodes
        bn_structure = saved_data['bn_structure']

        # Compute max CPT size from adjacency matrix
        max_parents = adjacency_matrix.sum(axis=0).max()
        max_cpt_size = 2 ** (max_parents + 1)

        cpt_info_array = extract_cpts_for_nodes(bn_structure, observed_nodes, n_nodes, int(max_cpt_size))
        cpt_info = torch.FloatTensor(cpt_info_array).unsqueeze(0)

        # Capture attention during forward pass
        with AttentionCapture(model) as attention_capture:
            _ = model(inputs, structure_info, cpt_info, dimensions)
            attention_weights = attention_capture.get_attention_weights()

        # Process attention weights from each layer
        for layer_name, attention in attention_weights.items():
            # attention shape: [batch=1, n_heads, n_nodes, n_nodes]
            # Average across heads to get [n_nodes, n_nodes]
            avg_attention = attention[0].mean(dim=0)  # Average across heads

            if layer_name not in layer_attention_sums:
                layer_attention_sums[layer_name] = torch.zeros(n_nodes, n_nodes)
                layer_counts[layer_name] = 0

            layer_attention_sums[layer_name] += avg_attention
            layer_counts[layer_name] += 1

    # Compute average attention matrices
    layer_avg_attention = {}
    for layer_name in layer_attention_sums:
        layer_avg_attention[layer_name] = layer_attention_sums[layer_name] / layer_counts[layer_name]

    # Analyze structural patterns
    analysis = {
        'n_samples_analyzed': len(matching_samples),
        'n_nodes': n_nodes,
        'adjacency_matrix': adjacency_matrix.tolist(),
        'layer_avg_attention': {name: attn.numpy().tolist() for name, attn in layer_avg_attention.items()},
        'target_observed_nodes': target_observed,
        'target_missing_nodes': target_missing
    }

    # Compute structural metrics
    analysis['structural_metrics'] = compute_structural_metrics(
        layer_avg_attention, adjacency_matrix, target_observed, target_missing
    )

    return analysis


def compute_structural_metrics(layer_avg_attention: Dict[str, torch.Tensor],
                             adjacency_matrix: np.ndarray,
                             target_observed: List[int],
                             target_missing: List[int]) -> Dict[str, Any]:
    """Compute metrics about how well attention follows graph structure."""

    metrics = {}
    n_nodes = adjacency_matrix.shape[0]

    # Create parent-child relationship maps
    parent_child_pairs = []
    non_parent_child_pairs = []

    for child in range(n_nodes):
        for parent in range(n_nodes):
            if adjacency_matrix[parent, child] == 1:
                parent_child_pairs.append((parent, child))
            elif parent != child:  # Exclude self-attention
                non_parent_child_pairs.append((parent, child))

    logger.info(f"Found {len(parent_child_pairs)} parent-child relationships")
    logger.info(f"Parent-child pairs: {parent_child_pairs}")

    # Analyze each layer
    for layer_name, attention in layer_avg_attention.items():
        attention_np = attention.numpy()

        # Parent-to-child attention strength
        parent_child_attentions = []
        for parent, child in parent_child_pairs:
            parent_child_attentions.append(attention_np[child, parent])  # child attends to parent

        # Non-parent-to-child attention strength (baseline)
        non_parent_child_attentions = []
        for parent, child in non_parent_child_pairs:
            non_parent_child_attentions.append(attention_np[child, parent])

        # Compute metrics
        if parent_child_attentions and non_parent_child_attentions:
            avg_parent_child_attn = np.mean(parent_child_attentions)
            avg_non_parent_child_attn = np.mean(non_parent_child_attentions)
            structural_bias = avg_parent_child_attn - avg_non_parent_child_attn
            structural_ratio = avg_parent_child_attn / avg_non_parent_child_attn if avg_non_parent_child_attn > 0 else float('inf')
        else:
            avg_parent_child_attn = 0
            avg_non_parent_child_attn = 0
            structural_bias = 0
            structural_ratio = 1

        # Attention entropy (how focused vs diffuse)
        attention_entropy = -np.sum(attention_np * np.log(attention_np + 1e-10), axis=1).mean()

        metrics[layer_name] = {
            'avg_parent_child_attention': float(avg_parent_child_attn),
            'avg_non_parent_child_attention': float(avg_non_parent_child_attn),
            'structural_bias': float(structural_bias),
            'structural_ratio': float(structural_ratio),
            'attention_entropy': float(attention_entropy),
            'max_attention': float(attention_np.max()),
            'min_attention': float(attention_np.min()),
            'attention_std': float(attention_np.std())
        }

    # Missing pattern information
    metrics['missing_pattern'] = {
        'observed_nodes': target_observed,
        'missing_nodes': target_missing,
        'n_observed': len(target_observed),
        'n_missing': len(target_missing)
    }

    return metrics


def create_structural_attention_plots(analysis: Dict[str, Any], output_dir: str) -> None:
    """Create visualizations of structural attention patterns."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    adjacency_matrix = np.array(analysis['adjacency_matrix'])
    layer_attention = {name: np.array(attn) for name, attn in analysis['layer_avg_attention'].items()}
    observed_nodes = analysis['target_observed_nodes']
    missing_nodes = analysis['target_missing_nodes']
    n_nodes = analysis['n_nodes']

    # 1. Create separate BN structure plot
    create_bn_structure_plot(adjacency_matrix, output_dir)

    # 2. Create individual layer attention plots
    create_individual_layer_plots(adjacency_matrix, layer_attention, observed_nodes, missing_nodes, output_dir)

    # 3. Create attention heatmaps by layer (backup visualization)
    create_layer_attention_heatmaps(layer_attention, output_dir)

    # 4. Create structural metrics plot
    create_structural_metrics_plot(analysis['structural_metrics'], output_dir)


def create_bn_structure_plot(adjacency_matrix: np.ndarray, output_dir: str) -> None:
    """Create clean BN structure plot showing only the graph connections."""

    import networkx as nx
    import matplotlib.pyplot as plt

    n_nodes = adjacency_matrix.shape[0]

    # Create directed graph
    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)

    # Add edges from adjacency matrix
    for parent in range(n_nodes):
        for child in range(n_nodes):
            if adjacency_matrix[parent, child] == 1:
                G.add_edge(parent, child)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Use consistent circular layout
    pos = nx.circular_layout(G)

    # Draw nodes (neutral color)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=2000, ax=ax, edgecolors='black', linewidths=2)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax)

    # Draw BN structure edges manually with clear arrows
    for parent in range(n_nodes):
        for child in range(n_nodes):
            if adjacency_matrix[parent, child] == 1:
                # Calculate arrow positions
                from_pos = np.array(pos[parent])
                to_pos = np.array(pos[child])

                # Vector from source to target
                direction = to_pos - from_pos
                direction_norm = direction / np.linalg.norm(direction)

                # Node radius for positioning
                node_radius = 0.15

                # Adjust start and end points
                arrow_start = from_pos + direction_norm * node_radius
                arrow_end = to_pos - direction_norm * node_radius

                # Draw thick arrow
                ax.annotate('', xy=arrow_end, xytext=arrow_start,
                           arrowprops=dict(
                               arrowstyle='->',
                               color='black',
                               lw=4,
                               alpha=1.0
                           ))

    ax.set_title('Bayesian Network Structure', fontsize=18, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bn_structure.png", dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"BN structure plot saved to {output_dir}/bn_structure.png")
    plt.close()


def create_individual_layer_plots(adjacency_matrix: np.ndarray,
                                 layer_attention: Dict[str, np.ndarray],
                                 observed_nodes: List[int],
                                 missing_nodes: List[int],
                                 output_dir: str) -> None:
    """Create individual attention plots for each layer."""

    import networkx as nx
    import matplotlib.pyplot as plt

    n_nodes = adjacency_matrix.shape[0]
    layer_names = sorted(layer_attention.keys())

    # Create directed graph
    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)

    # Add edges from adjacency matrix
    for parent in range(n_nodes):
        for child in range(n_nodes):
            if adjacency_matrix[parent, child] == 1:
                G.add_edge(parent, child)

    # Use consistent circular layout
    pos = nx.circular_layout(G)

    # Find global attention range for consistent color scaling
    all_attentions = []
    for attention in layer_attention.values():
        all_attentions.extend(attention.flatten())

    global_max = max(all_attentions)
    global_min = min(all_attentions)
    attention_threshold = 0.05

    # Create viridis colormap
    viridis = plt.cm.viridis

    # Process each layer
    for layer_idx, layer_name in enumerate(layer_names[:4]):
        attention = layer_attention[layer_name]

        # Create individual figure for this layer
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Node colors: Green = observed, Red = missing
        node_colors = []
        for node in range(n_nodes):
            if node in observed_nodes:
                node_colors.append('lightgreen')
            else:  # missing nodes
                node_colors.append('lightcoral')

        # Draw larger nodes with clear colors
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=2000, ax=ax, edgecolors='black', linewidths=3)

        # Draw node labels clearly
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax)

        # Draw original BN structure as thin gray edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              arrowsize=20, arrowstyle='->', width=2, alpha=0.4, ax=ax)

        # Calculate node radius for arrow positioning
        node_radius = 0.12  # Approximate radius based on node size

        # Draw attention as colored arrows that stop outside nodes
        for from_node in range(n_nodes):
            for to_node in range(n_nodes):
                if from_node != to_node:  # No self-loops
                    attention_weight = attention[to_node, from_node]  # to_node attends to from_node

                    if attention_weight > attention_threshold:
                        # Normalize attention for color mapping
                        if global_max > global_min:
                            color_intensity = (attention_weight - global_min) / (global_max - global_min)
                        else:
                            color_intensity = 0.5

                        # Get color from viridis colormap
                        color = viridis(color_intensity)

                        # Calculate arrow start and end points outside node circles
                        from_pos = np.array(pos[from_node])
                        to_pos = np.array(pos[to_node])

                        # Vector from source to target
                        direction = to_pos - from_pos
                        direction_norm = direction / np.linalg.norm(direction)

                        # Adjust start and end points to be outside node circles
                        arrow_start = from_pos + direction_norm * node_radius
                        arrow_end = to_pos - direction_norm * node_radius

                        # Draw attention arrow
                        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                                   arrowprops=dict(
                                       arrowstyle='->',
                                       color=color,
                                       lw=4,
                                       alpha=0.9,
                                       connectionstyle="arc3,rad=0.15"
                                   ))

        # Add colorbar for this layer
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=14)

        # Layer title (not bold, professional)
        layer_title = layer_name.replace("transformer.layers.", "Layer ").replace(".attention", "")
        missing_str = ', '.join([f'N{n}' for n in missing_nodes])
        observed_str = ', '.join([f'N{n}' for n in observed_nodes])

        ax.set_title(f'{layer_title} Attention Patterns\nObserved: {observed_str} | Missing: {missing_str}',
                    fontsize=16, pad=20)
        ax.axis('off')

        plt.tight_layout()

        # Save individual layer plot
        layer_filename = layer_name.replace("transformer.layers.", "layer_").replace(".attention", "")
        plt.savefig(f"{output_dir}/{layer_filename}_attention.png", dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"{layer_title} attention plot saved to {output_dir}/{layer_filename}_attention.png")
        plt.close()


def create_layer_attention_heatmaps(layer_attention: Dict[str, np.ndarray], output_dir: str) -> None:
    """Create attention heatmaps for each layer."""

    n_layers = len(layer_attention)
    layer_names = sorted(layer_attention.keys())

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, layer_name in enumerate(layer_names):
        attention = layer_attention[layer_name]

        im = axes[i].imshow(attention, cmap='viridis', aspect='equal')
        axes[i].set_title(f'{layer_name.replace("transformer.layers.", "Layer ").replace(".attention", "")}',
                         fontsize=12, fontweight='bold')

        # Add node labels
        n_nodes = attention.shape[0]
        axes[i].set_xticks(range(n_nodes))
        axes[i].set_yticks(range(n_nodes))
        axes[i].set_xticklabels([f'N{i}' for i in range(n_nodes)])
        axes[i].set_yticklabels([f'N{i}' for i in range(n_nodes)])
        axes[i].set_xlabel('Attended To (Source)')
        axes[i].set_ylabel('Attending From (Query)')

        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/attention_heatmaps_by_layer.png", dpi=300, bbox_inches='tight')
    logger.info(f"Layer attention heatmaps saved to {output_dir}/attention_heatmaps_by_layer.png")
    plt.close()


def create_structural_metrics_plot(structural_metrics: Dict[str, Any], output_dir: str) -> None:
    """Create plot of structural attention metrics across layers."""

    layer_names = [name for name in structural_metrics.keys() if name not in ['missing_pattern']]
    layer_names = sorted(layer_names)

    if not layer_names:
        logger.warning("No layer metrics found for structural metrics plot")
        return

    # Check what metrics are actually available
    if not layer_names:
        return

    first_layer = layer_names[0]
    available_metrics = list(structural_metrics[first_layer].keys())

    # Only plot metrics that exist
    metrics_to_plot = []
    for metric in ['structural_bias', 'structural_ratio', 'attention_entropy']:
        if metric in available_metrics:
            metrics_to_plot.append(metric)

    if not metrics_to_plot:
        logger.warning("No valid metrics found for plotting")
        return

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))

    # Handle single subplot case
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        values = []
        layer_labels = []

        for layer_name in layer_names:
            if metric in structural_metrics[layer_name]:
                values.append(structural_metrics[layer_name][metric])
                layer_labels.append(layer_name.replace("transformer.layers.", "L").replace(".attention", ""))

        if values:  # Only plot if we have values
            axes[i].bar(layer_labels, values, color='steelblue', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/structural_metrics_by_layer.png", dpi=300, bbox_inches='tight')
    logger.info(f"Structural metrics plot saved to {output_dir}/structural_metrics_by_layer.png")
    plt.close()




def run_structural_attention_analysis(model_dir: str, output_dir: str = None, n_samples: int = 200) -> None:
    """
    Main function to run structural attention analysis.

    Args:
        model_dir: Directory containing saved model
        output_dir: Directory to save analysis results
        n_samples: Number of test samples to analyze
    """
    if output_dir is None:
        output_dir = str(Path(model_dir).parent / "structural_attention_analysis")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Running structural attention analysis on {model_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load model and data
    model, saved_data = load_model_and_data(model_dir)

    # Analyze structural attention patterns
    analysis = analyze_structural_attention(model, saved_data, n_samples)

    # Save analysis results
    analysis_file = Path(output_dir) / "structural_attention_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Analysis results saved to {analysis_file}")

    # Create visualizations
    create_structural_attention_plots(analysis, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("STRUCTURAL ATTENTION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Analyzed {analysis['n_samples_analyzed']} samples with {analysis['n_nodes']} nodes")

    # Find parent-child relationships
    adj_matrix = np.array(analysis['adjacency_matrix'])
    parent_child_pairs = []
    for child in range(analysis['n_nodes']):
        for parent in range(analysis['n_nodes']):
            if adj_matrix[parent, child] == 1:
                parent_child_pairs.append((parent, child))

    print(f"Graph structure: {len(parent_child_pairs)} parent-child relationships")
    print(f"Parent → Child edges: {parent_child_pairs}")

    # Print layer-wise structural metrics
    for layer_name, metrics in analysis['structural_metrics'].items():
        if layer_name != 'overall':
            print(f"\n{layer_name}:")
            print(f"  Structural bias: {metrics['structural_bias']:.4f}")
            print(f"  Structural ratio: {metrics['structural_ratio']:.4f}")
            print(f"  Parent→Child attention: {metrics['avg_parent_child_attention']:.4f}")
            print(f"  Non-structural attention: {metrics['avg_non_parent_child_attention']:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    import sys
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Command line interface
    parser = argparse.ArgumentParser(description="Analyze structural attention patterns")
    parser.add_argument('model_dir', help='Directory containing saved model')
    parser.add_argument('--output-dir', help='Directory to save analysis results')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of test samples to analyze')

    args = parser.parse_args()

    try:
        run_structural_attention_analysis(args.model_dir, args.output_dir, args.n_samples)
    except Exception as e:
        logger.error(f"Structural attention analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)