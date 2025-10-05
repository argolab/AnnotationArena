"""
Tuned Lens Plotting for MARFORMER mechanistic interpretability.

Visualization functions for comparing logit lens vs tuned lens approaches
and analyzing the improvement from learned probe transformations.

Extends mi_plotting.py with comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
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

# Color palette
BUDGET_COLORS = {
    50: '#e74c3c',      # Red
    500: '#3498db',     # Blue
    2000: '#2ecc71',    # Green
    5000: '#9b59b6',    # Purple
    10000: '#f39c12'    # Orange
}

BUDGET_MARKERS = {
    50: 'o',
    500: 's',
    2000: '^',
    5000: 'd',
    10000: 'v'
}


def plot_tuned_vs_logit_lens_comparison(
    logit_lens_dir: Path,
    tuned_lens_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot 1: Comparison of Tuned Lens vs Logit Lens layer-wise KL divergence.

    Shows two curves per budget:
    - Logit lens (baseline): Direct application of output heads
    - Tuned lens (learned): With trained affine transformations

    Args:
        logit_lens_dir: Directory with logit lens results (OUTPUT_MI)
        tuned_lens_dir: Directory with tuned lens results (OUTPUT_TUNED_LENS)
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating tuned vs logit lens comparison for {imputer_size}")

    # Collect data from both approaches
    logit_data = {}
    tuned_data = {}

    # Load logit lens data
    for budget_dir in sorted(logit_lens_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "layer_analysis.json"

        if not analysis_file.exists():
            logger.warning(f"No logit lens analysis file at {analysis_file}")
            continue

        with open(analysis_file, 'r') as f:
            data = json.load(f)

        logit_data[budget] = {
            'layer_descriptions': data['layer_descriptions'],
            'layer_kl_means': np.array(data['layer_kl_means']),
            'layer_kl_stds': np.array(data['layer_kl_stds']),
            'n_layers': data['n_layers']
        }

    # Load tuned lens data
    for budget_dir in sorted(tuned_lens_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "tuned_lens" / "layer_analysis.json"

        if not analysis_file.exists():
            logger.warning(f"No tuned lens analysis file at {analysis_file}")
            continue

        with open(analysis_file, 'r') as f:
            data = json.load(f)

        tuned_data[budget] = {
            'layer_descriptions': data['layer_descriptions'],
            'layer_kl_means': np.array(data['layer_kl_means']),
            'layer_kl_stds': np.array(data['layer_kl_stds']),
            'n_layers': data['n_layers']
        }

    if not logit_data or not tuned_data:
        logger.warning("Insufficient data for comparison plot, skipping")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each budget
    for budget in sorted(set(logit_data.keys()) & set(tuned_data.keys())):
        logit = logit_data[budget]
        tuned = tuned_data[budget]

        layer_indices = np.arange(len(logit['layer_kl_means']))
        color = BUDGET_COLORS.get(budget, '#95a5a6')
        marker = BUDGET_MARKERS.get(budget, 'o')

        # Logit lens (baseline) - dashed line
        ax.plot(
            layer_indices,
            logit['layer_kl_means'],
            label=f'Budget {budget} (Logit Lens)',
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            linestyle='--',
            alpha=0.7
        )

        # Tuned lens (learned) - solid line
        ax.plot(
            layer_indices,
            tuned['layer_kl_means'],
            label=f'Budget {budget} (Tuned Lens)',
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            linestyle='-',
            alpha=0.95
        )

    # Formatting
    layer_labels = logit_data[list(logit_data.keys())[0]]['layer_descriptions']
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=0, ha='center')

    ax.set_xlabel('Layer')
    ax.set_ylabel('KL divergence per marginal prediction (nats)')
    ax.set_title(f'Tuned Lens vs Logit Lens Comparison', fontsize=15, pad=12)

    ax.set_yscale('log')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, which='both', alpha=0.15, linewidth=0.5, color='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved tuned vs logit lens comparison to {output_path}")
    plt.close()


def plot_probe_improvement_by_layer(
    logit_lens_dir: Path,
    tuned_lens_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot 2: Probe improvement (KL reduction) by layer.

    Shows how much each layer's predictions improve with learned probes:
    Improvement = (Logit KL - Tuned KL) / Logit KL (percent reduction)

    Args:
        logit_lens_dir: Directory with logit lens results
        tuned_lens_dir: Directory with tuned lens results
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating probe improvement analysis for {imputer_size}")

    # Collect data from both approaches (same as above)
    logit_data = {}
    tuned_data = {}

    # Load logit lens data
    for budget_dir in sorted(logit_lens_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "layer_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            logit_data[budget] = {'layer_kl_means': np.array(data['layer_kl_means'])}

    # Load tuned lens data
    for budget_dir in sorted(tuned_lens_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "tuned_lens" / "layer_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            tuned_data[budget] = {
                'layer_kl_means': np.array(data['layer_kl_means']),
                'layer_descriptions': data['layer_descriptions']
            }

    if not logit_data or not tuned_data:
        logger.warning("Insufficient data for improvement plot, skipping")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot improvement for each budget
    for budget in sorted(set(logit_data.keys()) & set(tuned_data.keys())):
        logit_kl = logit_data[budget]['layer_kl_means']
        tuned_kl = tuned_data[budget]['layer_kl_means']

        # Compute percent improvement
        improvement = (logit_kl - tuned_kl) / (logit_kl + 1e-10) * 100

        layer_indices = np.arange(len(improvement))
        color = BUDGET_COLORS.get(budget, '#95a5a6')
        marker = BUDGET_MARKERS.get(budget, 'o')

        ax.plot(
            layer_indices,
            improvement,
            label=f'Budget {budget}',
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            alpha=0.85
        )

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Formatting
    layer_labels = tuned_data[list(tuned_data.keys())[0]]['layer_descriptions']
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=45, ha='right')

    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('KL Reduction (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Tuned Lens Improvement by Layer ({imputer_size} MARFORMER)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved probe improvement plot to {output_path}")
    plt.close()


def plot_tuned_lens_layer_kl_decay(
    experiment_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot tuned lens layer-wise KL divergence progression.

    Clean, simple plot matching reference paper style.

    Args:
        experiment_dir: Directory containing tuned lens results
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating tuned lens layer KL decay plot for {experiment_dir}")

    # Collect data from all budget directories
    budget_data = {}

    for budget_dir in sorted(experiment_dir.glob("budget_*")):
        budget = int(budget_dir.name.split('_')[1])
        analysis_file = budget_dir / imputer_size.lower() / "tuned_lens" / "layer_analysis.json"

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
    ax.set_title(f'BN Domain: 10 nodes, ≈5 Parents per Node (Tuned Lens)', fontsize=15, pad=12)

    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.15, linewidth=0.5, color='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved tuned lens layer KL decay to {output_path}")
    plt.close()


def plot_tuned_lens_layer_kl_by_neighbors(
    experiment_dir: Path,
    output_path: Path,
    imputer_size: str = "Large"
) -> None:
    """
    Plot tuned lens layer KL by number of observed neighbors.

    Tests belief propagation hypothesis with tuned lens probes.

    Args:
        experiment_dir: Directory containing tuned lens results
        output_path: Where to save the plot
        imputer_size: Model size to analyze
    """
    logger.info(f"Creating tuned lens layer KL by neighbors plot for {experiment_dir}")

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

        analysis_file = budget_dir / imputer_size.lower() / "tuned_lens" / "layer_analysis.json"
        raw_data_file = budget_dir / imputer_size.lower() / "tuned_lens" / "layer_kl_raw_data.pkl"
        sample_metadata_file = budget_dir / imputer_size.lower() / "tuned_lens" / "sample_metadata.pkl"

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
        neighbor_category_kls = {
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
                if n_observed_neighbors <= 2:
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
            '1-2 observed neighbors': '#f39c12',    # Orange (medium)
            '3+ observed neighbors': '#2ecc71'      # Green (easiest)
        }

        markers = {
            '1-2 observed neighbors': 's',
            '3+ observed neighbors': '^'
        }

        for category in ['1-2 observed neighbors', '3+ observed neighbors']:
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
            stds = np.array(stds)

            # Only plot if we have data - SIMPLE LINES ONLY, NO SHADING
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
    fig.suptitle(f'Tuned Lens Layer KL by Observed Neighbors ({imputer_size} MARFORMER)',
                 fontsize=15, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved tuned lens layer KL by neighbors to {output_path}")
    plt.close()


def create_tuned_lens_plots_for_experiment(
    experiment_dir: Path,
    imputer_size: str = "Large",
    logit_lens_base_dir: Optional[Path] = None
) -> None:
    """
    Create all tuned lens plots for a given experiment directory.

    Args:
        experiment_dir: Path to tuned lens experiment directory
        imputer_size: Model size to analyze
        logit_lens_base_dir: Path to corresponding logit lens results for comparison
    """
    logger.info(f"Creating tuned lens plots for {experiment_dir}")

    # Create plots subdirectory
    plots_dir = experiment_dir / "tuned_lens_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ALWAYS create standalone tuned lens plots
    # Plot 1: Layer KL Decay
    decay_path = plots_dir / f"layer_kl_decay_{imputer_size.lower()}.png"
    try:
        plot_tuned_lens_layer_kl_decay(experiment_dir, decay_path, imputer_size)
    except Exception as e:
        logger.error(f"Failed to create tuned lens layer KL decay plot: {e}", exc_info=True)

    # Plot 2: Layer KL by Neighbors
    neighbors_path = plots_dir / f"layer_kl_by_neighbors_{imputer_size.lower()}.png"
    try:
        plot_tuned_lens_layer_kl_by_neighbors(experiment_dir, neighbors_path, imputer_size)
    except Exception as e:
        logger.error(f"Failed to create tuned lens layer KL by neighbors plot: {e}", exc_info=True)

    # If logit lens directory provided, create comparison plots
    if logit_lens_base_dir:
        # Find matching logit lens experiment
        exp_name = experiment_dir.name  # e.g., "nodes_5_missing_0.5_graph_0"
        logit_lens_dir = logit_lens_base_dir / exp_name

        if logit_lens_dir.exists():
            # Plot 1: Tuned vs Logit Lens Comparison
            comparison_path = plots_dir / f"tuned_vs_logit_comparison_{imputer_size.lower()}.png"
            try:
                plot_tuned_vs_logit_lens_comparison(
                    logit_lens_dir, experiment_dir, comparison_path, imputer_size
                )
            except Exception as e:
                logger.error(f"Failed to create comparison plot: {e}", exc_info=True)

            # Plot 2: Probe Improvement by Layer
            improvement_path = plots_dir / f"probe_improvement_{imputer_size.lower()}.png"
            try:
                plot_probe_improvement_by_layer(
                    logit_lens_dir, experiment_dir, improvement_path, imputer_size
                )
            except Exception as e:
                logger.error(f"Failed to create improvement plot: {e}", exc_info=True)

        else:
            logger.warning(f"Logit lens directory not found at {logit_lens_dir}")

    logger.info(f"Tuned lens plots saved to {plots_dir}")


def create_all_tuned_lens_plots(
    tuned_lens_dir: Path,
    logit_lens_dir: Path,
    imputer_sizes: Optional[List[str]] = None
) -> None:
    """
    Create tuned lens plots for all experiments in output directory.

    Args:
        tuned_lens_dir: Root tuned lens output directory (e.g., OUTPUT_TUNED_LENS)
        logit_lens_dir: Root logit lens output directory (e.g., OUTPUT_MI)
        imputer_sizes: List of imputer sizes to analyze (default: ["Large"])
    """
    if imputer_sizes is None:
        imputer_sizes = ["Large"]

    logger.info(f"Creating tuned lens plots for all experiments in {tuned_lens_dir}")

    # Find all experiment directories
    experiment_dirs = [d for d in tuned_lens_dir.iterdir()
                      if d.is_dir() and d.name.startswith('nodes_')]

    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {tuned_lens_dir}")
        return

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    for exp_dir in experiment_dirs:
        logger.info(f"Processing {exp_dir.name}")

        for imputer_size in imputer_sizes:
            try:
                create_tuned_lens_plots_for_experiment(
                    exp_dir, imputer_size, logit_lens_dir
                )
            except Exception as e:
                logger.error(f"Failed to create plots for {exp_dir.name}, {imputer_size}: {e}")

    logger.info("All tuned lens plots created successfully")
