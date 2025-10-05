"""
Mechanistic Interpretability Plotting for MARFORMER.

Visualization functions for analyzing layer-wise representations and
progressive refinement in transformer-based Bayesian network inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import pickle

logger = logging.getLogger(__name__)

# Professional publication-quality matplotlib settings (matching reference paper)
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.linewidth': 1.0,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.5,
    'grid.color': '#cccccc',
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

# Color palette for budgets
BUDGET_COLORS = {
    50: '#e74c3c',      # Red
    500: '#3498db',     # Blue
    2000: '#2ecc71',    # Green
    5000: '#9b59b6',    # Purple
    10000: '#f39c12'    # Orange
}

# Markers for budgets
BUDGET_MARKERS = {
    50: 'o',
    500: 's',
    2000: '^',
    5000: 'd',
    10000: 'v'
}


def plot_layer_kl_decay_curves(
    experiment_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot 1A: Layer KL Decay Curves.

    Clean, simple plot showing KL divergence progression through transformer layers.
    Matches reference paper style: no error bars, clean lines, minimal styling.

    Args:
        experiment_dir: Directory containing budget subdirectories
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating layer KL decay curves for {experiment_dir}")

    # Collect data from all budget directories
    budget_data = {}

    for budget_dir in sorted(experiment_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "layer_analysis.json"

        if not analysis_file.exists():
            logger.warning(f"No analysis file found at {analysis_file}")
            continue

        with open(analysis_file, 'r') as f:
            data = json.load(f)

        budget_data[budget] = {
            'layer_descriptions': data['layer_descriptions'],
            'layer_kl_means': np.array(data['layer_kl_means']),
            'layer_kl_stds': np.array(data['layer_kl_stds']),
            'n_layers': data['n_layers']
        }

    if not budget_data:
        logger.warning("No budget data found, skipping plot")
        return

    # Create figure (matching reference dimensions)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each budget - simple lines, no error bars
    for budget in sorted(budget_data.keys()):
        data = budget_data[budget]
        layer_indices = np.arange(len(data['layer_kl_means']))

        color = BUDGET_COLORS.get(budget, '#95a5a6')
        marker = BUDGET_MARKERS.get(budget, 'o')

        # Simple line plot
        ax.plot(
            layer_indices,
            data['layer_kl_means'],
            label=f'Budget {budget}',
            color=color,
            marker=marker,
            linestyle='-',
            linewidth=2.5,
            markersize=8
        )

    # Formatting (matching reference style)
    layer_labels = budget_data[list(budget_data.keys())[0]]['layer_descriptions']
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=0, ha='center')

    ax.set_xlabel('Layer')
    ax.set_ylabel('KL divergence per marginal prediction (nats)')
    ax.set_title(f'BN Domain: 10 nodes, ≈5 Parents per Node', fontsize=15, pad=12)

    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.15, linewidth=0.5, color='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved layer KL decay curves to {output_path}")
    plt.close()


def plot_layer_kl_by_observed_neighbors(
    experiment_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot 4A: Layer KL by Number of Observed Neighbors.

    Clean plot with simple lines, no error bars or shading.

    Args:
        experiment_dir: Directory containing budget subdirectories
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating layer KL by observed neighbors plot for {experiment_dir}")

    # Load graph structure
    bn_file = experiment_dir / "bn_structure" / "bn_structure.pkl"
    adj_matrix_file = experiment_dir / "bn_structure" / "adjacency_matrix.npy"

    if not bn_file.exists() or not adj_matrix_file.exists():
        logger.warning(f"Graph structure files not found in {experiment_dir}")
        return

    with open(bn_file, 'rb') as f:
        bn = pickle.load(f)
    adj_matrix = np.load(adj_matrix_file)

    n_nodes = adj_matrix.shape[0]

    # Collect data from all budgets
    budget_data = {}

    for budget_dir in sorted(experiment_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])

        analysis_file = budget_dir / imputer_size.lower() / "layer_analysis.json"
        raw_data_file = budget_dir / imputer_size.lower() / "layer_kl_raw_data.pkl"
        sample_metadata_file = budget_dir / imputer_size.lower() / "sample_metadata.pkl"

        if not all([analysis_file.exists(), raw_data_file.exists(), sample_metadata_file.exists()]):
            logger.warning(f"Missing files for budget {budget}")
            continue

        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        with open(raw_data_file, 'rb') as f:
            layer_kl_raw = pickle.load(f)

        with open(sample_metadata_file, 'rb') as f:
            sample_metadata = pickle.load(f)

        budget_data[budget] = {
            'layer_descriptions': analysis['layer_descriptions'],
            'n_layers': analysis['n_layers'],
            'layer_kl_raw': layer_kl_raw,
            'sample_metadata': sample_metadata
        }

    if not budget_data:
        logger.warning("No budget data found, skipping plot")
        return

    # Create subplots: one per budget
    n_budgets = len(budget_data)
    fig, axes = plt.subplots(1, n_budgets, figsize=(6 * n_budgets, 5), squeeze=False)
    axes = axes.flatten()

    for ax_idx, budget in enumerate(sorted(budget_data.keys())):
        ax = axes[ax_idx]
        data = budget_data[budget]

        # Categorize nodes by number of observed neighbors
        # We need to analyze per-sample: for each unobserved node, count observed neighbors

        neighbor_category_kls = {
            '0 observed neighbors': {layer_idx: [] for layer_idx in range(data['n_layers'] + 2)},
            '1-2 observed neighbors': {layer_idx: [] for layer_idx in range(data['n_layers'] + 2)},
            '3+ observed neighbors': {layer_idx: [] for layer_idx in range(data['n_layers'] + 2)}
        }

        # Process each sample
        for sample_meta in data['sample_metadata']:
            sample_idx = sample_meta['sample_idx']
            unobserved_nodes = sample_meta['unobserved_nodes']
            observed_nodes = set(sample_meta['observed_nodes'])

            # For each unobserved node, count observed neighbors
            for node_idx in unobserved_nodes:
                # Get neighbors (parents and children in the graph)
                parents = np.where(adj_matrix[:, node_idx] == 1)[0]
                children = np.where(adj_matrix[node_idx, :] == 1)[0]
                neighbors = set(parents) | set(children)

                # Count how many neighbors are observed
                n_observed_neighbors = len(neighbors & observed_nodes)

                # Categorize
                if n_observed_neighbors == 0:
                    category = '0 observed neighbors'
                elif n_observed_neighbors <= 2:
                    category = '1-2 observed neighbors'
                else:
                    category = '3+ observed neighbors'

                # Get KL values for this node across all layers
                for layer_idx in range(data['n_layers'] + 2):
                    if sample_idx in data['layer_kl_raw'][layer_idx]:
                        if node_idx in data['layer_kl_raw'][layer_idx][sample_idx]:
                            kl = data['layer_kl_raw'][layer_idx][sample_idx][node_idx]
                            neighbor_category_kls[category][layer_idx].append(kl)

        # Compute means for each category
        layer_indices = np.arange(data['n_layers'] + 2)

        colors = {
            '0 observed neighbors': '#e74c3c',      # Red (hardest)
            '1-2 observed neighbors': '#f39c12',    # Orange (medium)
            '3+ observed neighbors': '#2ecc71'      # Green (easiest)
        }

        markers = {
            '0 observed neighbors': 'o',
            '1-2 observed neighbors': 's',
            '3+ observed neighbors': '^'
        }

        for category in ['0 observed neighbors', '1-2 observed neighbors', '3+ observed neighbors']:
            means = []
            stds = []

            for layer_idx in layer_indices:
                kls = neighbor_category_kls[category][layer_idx]
                if kls:
                    means.append(np.mean(kls))
                    stds.append(np.std(kls))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)

            # Only plot if we have data - SIMPLE LINES ONLY, NO ERROR BARS
            if not np.all(np.isnan(means)):
                ax.plot(
                    layer_indices,
                    means,
                    label=category,
                    color=colors[category],
                    marker=markers[category],
                    markersize=8,
                    linewidth=2.5,
                    linestyle='-'
                )

        # Formatting
        layer_labels = data['layer_descriptions']
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=0, ha='center')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean KL Divergence')
        ax.set_title(f'Budget {budget}', fontsize=14)

        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, which='both', alpha=0.15, linewidth=0.5, color='#cccccc')

    # Overall title
    fig.suptitle(f'Layer KL by Observed Neighbors ({imputer_size} MARFORMER)',
                 fontsize=15, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved layer KL by observed neighbors plot to {output_path}")
    plt.close()


def create_mi_plots_for_experiment(
    experiment_dir: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Create all MI plots for a given experiment directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., nodes_5_missing_0.5_graph_0)
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating MI plots for {experiment_dir}")

    # Create plots subdirectory
    plots_dir = experiment_dir / "mi_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1A: Layer KL Decay Curves
    plot1a_path = plots_dir / f"layer_kl_decay_{imputer_size.lower()}.png"
    try:
        plot_layer_kl_decay_curves(experiment_dir, plot1a_path, imputer_size)
    except Exception as e:
        logger.error(f"Failed to create layer KL decay plot: {e}", exc_info=True)

    # Plot 4A: Layer KL by Observed Neighbors
    plot4a_path = plots_dir / f"layer_kl_by_neighbors_{imputer_size.lower()}.png"
    try:
        plot_layer_kl_by_observed_neighbors(experiment_dir, plot4a_path, imputer_size)
    except Exception as e:
        logger.error(f"Failed to create layer KL by neighbors plot: {e}", exc_info=True)

    logger.info(f"MI plots saved to {plots_dir}")


def create_all_mi_plots(output_dir: Path, imputer_sizes: Optional[List[str]] = None) -> None:
    """
    Create MI plots for all experiments in output directory.

    Args:
        output_dir: Root output directory (e.g., OUTPUT_MI)
        imputer_sizes: List of imputer sizes to analyze (default: ["Large"])
    """
    if imputer_sizes is None:
        imputer_sizes = ["Large"]

    logger.info(f"Creating MI plots for all experiments in {output_dir}")

    # Find all experiment directories
    experiment_dirs = [d for d in output_dir.iterdir()
                      if d.is_dir() and d.name.startswith('nodes_')]

    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {output_dir}")
        return

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    for exp_dir in experiment_dirs:
        logger.info(f"Processing {exp_dir.name}")

        for imputer_size in imputer_sizes:
            try:
                create_mi_plots_for_experiment(exp_dir, imputer_size)
            except Exception as e:
                logger.error(f"Failed to create plots for {exp_dir.name}, {imputer_size}: {e}")

    logger.info("All MI plots created successfully")
