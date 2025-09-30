"""
Additional visualization functions for progressive imputation results.

Creates log-loss convergence curves and 5-step model vs true scatter plots
from saved experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Enable LaTeX rendering with fallback
LATEX_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(['latex', '--version'], capture_output=True, timeout=5)
    if result.returncode == 0:
        plt.rcParams['text.usetex'] = True
        LATEX_AVAILABLE = True
        logger.info("LaTeX rendering enabled for additional plots")
    else:
        logger.warning("LaTeX command found but not working, using fallback formatting")
except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
    logger.warning(f"LaTeX not available ({e}), using fallback formatting")

# Professional publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'figure.dpi': 100
})

# Professional color palette
COLORS = {
    'em_1_restart': '#2E86AB',      # Blue for EM (1 restart)
    'em_10_restart': '#0F4C75',     # Darker blue for EM (10 restarts)
    'marformer_tiny': '#E63946',    # Darker red for Tiny
    'marformer_small': '#A4161A',   # Medium dark red for Small
    'marformer_large': '#660708',   # Very dark red for Large
    'true_model': '#228B22'         # Green for true model
}


def format_model_name(model_type: str, size: str = None) -> str:
    """
    Format model names with consistent LaTeX styling.

    Args:
        model_type: 'marformer', 'em', or 'true'
        size: For marformer: 'Tiny', 'Small', 'Large'. For EM: '1' or '10'

    Returns:
        Formatted name string
    """
    if model_type.lower() == 'marformer':
        if LATEX_AVAILABLE:
            return rf'\textsc{{Marformer}} {size}'
        else:
            return f'MARFORMER {size}'
    elif model_type.lower() == 'em':
        restart_word = 'Restart' if size == '1' else 'Restarts'
        return f'EM ({size} Random {restart_word})'
    elif model_type.lower() == 'true':
        return 'True Model (LazyPropagation)'
    else:
        return f'{model_type} {size}'


def load_results_from_directory(results_dir: str) -> Dict[str, Any]:
    """
    Load experimental results from a results directory.

    Args:
        results_dir: Path to directory containing result pickle files

    Returns:
        Dictionary of experimental results
    """
    results_path = Path(results_dir)

    # Find the most recent results pickle file
    pickle_files = list(results_path.glob("results_*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(f"No results pickle files found in {results_dir}")

    # Get the most recent file
    latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading results from {latest_file}")

    with open(latest_file, 'rb') as f:
        results = pickle.load(f)

    return results


def get_five_step_budgets(results: Dict[str, Any]) -> List[int]:
    """
    Get 5 evenly spaced budget levels for step progression plots.

    Args:
        results: Results dictionary from experiments

    Returns:
        List of 5 budget values evenly distributed across the progression
    """
    # Get budget sequence from any experiment
    for key, experiment_data in results.items():
        if experiment_data['results']:
            budgets = [step['budget'] for step in experiment_data['results']]
            if len(budgets) >= 5:
                # Select 5 evenly spaced budgets
                indices = np.linspace(0, len(budgets) - 1, 5, dtype=int)
                selected_budgets = [budgets[i] for i in indices]
                logger.debug(f"Five-step progression budgets: {selected_budgets}")
                return selected_budgets

    # Fallback if not enough steps
    logger.warning("Could not determine 5 step budgets, using fewer steps")
    for key, experiment_data in results.items():
        if experiment_data['results']:
            budgets = [step['budget'] for step in experiment_data['results']]
            return budgets[:5] if len(budgets) >= 5 else budgets

    return [10, 500, 1000, 1500, 2000]  # Final fallback


def plot_log_loss_convergence_curves(results: Dict[str, Any], output_dir: str = "plots",
                                   missing_rate: Optional[float] = None) -> None:
    """
    Plot log-loss convergence curves with bootstrap confidence intervals.

    Creates log-loss vs training set size plots with proper model naming
    and shaded confidence intervals instead of error bars.

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Group results by node size
    results_by_nodes = {}
    for key, policy_results in results.items():
        if isinstance(key, tuple) and len(key) >= 2:
            n_nodes = key[0]
        else:
            # Fallback parsing for string keys
            continue

        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][key] = policy_results

    # Create separate plot for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get first experiment for EM and True model plotting
        first_experiment = next(iter(node_results.values()))
        experiment_results = first_experiment['results']

        # Extract budget progression
        budgets = [r['budget'] for r in experiment_results]

        # True model log-loss (reference baseline)
        true_log_losses = [r.get('true_model_log_loss', float('inf')) for r in experiment_results]
        valid_true_losses = [x for x in true_log_losses if not np.isinf(x)]
        if valid_true_losses:
            true_baseline = np.mean(valid_true_losses)
            ax.axhline(y=true_baseline, color=COLORS['true_model'], linestyle='-',
                      linewidth=2, alpha=0.8, label=format_model_name('true'))

        # EM data with bootstrap CIs
        domain_log_losses = [r.get('domain_log_loss', float('inf')) for r in experiment_results]
        domain_log_loss_lowers = [r.get('domain_log_loss_lower', r.get('domain_log_loss', float('inf'))) for r in experiment_results]
        domain_log_loss_uppers = [r.get('domain_log_loss_upper', r.get('domain_log_loss', float('inf'))) for r in experiment_results]

        domain_1_log_losses = [r.get('domain_1_log_loss', float('inf')) for r in experiment_results]
        domain_1_log_loss_lowers = [r.get('domain_1_log_loss_lower', r.get('domain_1_log_loss', float('inf'))) for r in experiment_results]
        domain_1_log_loss_uppers = [r.get('domain_1_log_loss_upper', r.get('domain_1_log_loss', float('inf'))) for r in experiment_results]

        # Plot EM variants first (background)
        # EM (1 restart) - dotted line
        em_1_label = format_model_name('em', '1')
        ax.plot(budgets, domain_1_log_losses, 's:', label=em_1_label,
               color=COLORS['em_1_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(budgets, domain_1_log_loss_lowers, domain_1_log_loss_uppers,
                       color=COLORS['em_1_restart'], alpha=0.2)

        # EM (10 restarts) - solid line
        em_10_label = format_model_name('em', '10')
        ax.plot(budgets, domain_log_losses, 's-', label=em_10_label,
               color=COLORS['em_10_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(budgets, domain_log_loss_lowers, domain_log_loss_uppers,
                       color=COLORS['em_10_restart'], alpha=0.2)

        # Plot all Marformer variants
        for key, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')

            # Extract progression data
            budgets = [r['budget'] for r in experiment_results]

            # Neural imputer log-loss data with bootstrap CIs
            neural_log_losses = [r.get('neural_log_loss', float('inf')) for r in experiment_results]
            neural_log_loss_lowers = [r.get('neural_log_loss_lower', r.get('neural_log_loss', float('inf'))) for r in experiment_results]
            neural_log_loss_uppers = [r.get('neural_log_loss_upper', r.get('neural_log_loss', float('inf'))) for r in experiment_results]

            # Plot Marformer with shaded confidence intervals
            color_key = f'marformer_{imputer_size.lower()}'
            if color_key in COLORS:
                marformer_label = format_model_name('marformer', imputer_size)
                ax.plot(budgets, neural_log_losses, 'o-', label=marformer_label,
                       color=COLORS[color_key], linewidth=2, markersize=6, alpha=0.8)
                ax.fill_between(budgets, neural_log_loss_lowers, neural_log_loss_uppers,
                               color=COLORS[color_key], alpha=0.2)

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Log-Loss per marginal prediction (nats)', fontsize=12)
        ax.set_title(f'BN Domain: {n_nodes} nodes, Log-Loss Convergence', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with missing rate suffix if provided
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/log_loss_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Log-loss curves for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_five_step_model_vs_true_scatter(results: Dict[str, Any], output_dir: str = "plots",
                                        missing_rate: Optional[float] = None) -> None:
    """
    Plot 5-step model vs true scatter plots progression.

    Creates 4 rows × 5 columns grid showing evolution of cross-entropy
    predictions against true model across 5 training stages.

    Rows: EM (10), Marformer Large, Marformer Small, Marformer Tiny
    Columns: 5 evenly spaced budget levels

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get 5 step progression budgets
    step_budgets = get_five_step_budgets(results)

    # Extract data for each model and budget combination
    model_data = {}  # {model_name: {budget: [(true_entropy, model_cross_entropy)]}}

    # Collect Marformer data
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Large')

        model_name = format_model_name('marformer', imputer_size)
        if model_name not in model_data:
            model_data[model_name] = {}

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets:
                true_entropy_values = step_result.get('true_entropy_values', [])
                neural_cross_entropy_values = step_result.get('neural_cross_entropy_values', [])

                # Pair up values
                pairs = []
                min_len = min(len(true_entropy_values), len(neural_cross_entropy_values))
                for i in range(min_len):
                    true_val = true_entropy_values[i]
                    pred_val = neural_cross_entropy_values[i]
                    if not (np.isnan(true_val) or np.isinf(true_val) or
                           np.isnan(pred_val) or np.isinf(pred_val)):
                        pairs.append((true_val, pred_val))

                if budget not in model_data[model_name]:
                    model_data[model_name][budget] = []
                model_data[model_name][budget].extend(pairs)

    # Add EM (10 restarts) data
    em_model_name = format_model_name('em', '10')
    model_data[em_model_name] = {}

    for key, policy_results in results.items():
        experiment_results = policy_results['results']

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets:
                true_entropy_values = step_result.get('true_entropy_values', [])
                domain_cross_entropy_values = step_result.get('domain_cross_entropy_values', [])

                pairs = []
                min_len = min(len(true_entropy_values), len(domain_cross_entropy_values))
                for i in range(min_len):
                    true_val = true_entropy_values[i]
                    pred_val = domain_cross_entropy_values[i]
                    if not (np.isnan(true_val) or np.isinf(true_val) or
                           np.isnan(pred_val) or np.isinf(pred_val)):
                        pairs.append((true_val, pred_val))

                if budget not in model_data[em_model_name]:
                    model_data[em_model_name][budget] = []
                model_data[em_model_name][budget].extend(pairs)
        break  # Only need one policy for EM data

    # Create 4×5 subplot grid
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    # Model order for rows
    model_order = [
        format_model_name('em', '10'),
        format_model_name('marformer', 'Large'),
        format_model_name('marformer', 'Small'),
        format_model_name('marformer', 'Tiny')
    ]

    # Budget labels for columns
    budget_labels = [f'Step {i+1}' for i in range(5)]

    # Get consistent axis limits
    all_true_vals = []
    all_pred_vals = []
    for model_name, budget_data in model_data.items():
        for budget, pairs in budget_data.items():
            if pairs:
                true_vals = [p[0] for p in pairs]
                pred_vals = [p[1] for p in pairs]
                all_true_vals.extend(true_vals)
                all_pred_vals.extend(pred_vals)

    if all_true_vals and all_pred_vals:
        min_val = min(min(all_true_vals), min(all_pred_vals))
        max_val = max(max(all_true_vals), max(all_pred_vals))
        axis_margin = (max_val - min_val) * 0.05
        axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
    else:
        axis_min, axis_max = 0, 1

    # Plot each subplot
    for row, model_name in enumerate(model_order):
        for col, (budget, budget_label) in enumerate(zip(step_budgets, budget_labels)):
            ax = axes[row, col]

            if model_name in model_data and budget in model_data[model_name]:
                pairs = model_data[model_name][budget]
                if pairs:
                    true_vals = [p[0] for p in pairs]
                    pred_vals = [p[1] for p in pairs]

                    # Get model-specific color
                    if 'EM' in model_name:
                        color = COLORS['em_10_restart']
                    elif 'Marformer' in model_name or 'MARFORMER' in model_name:
                        if 'Large' in model_name:
                            color = COLORS['marformer_large']
                        elif 'Small' in model_name:
                            color = COLORS['marformer_small']
                        else:  # Tiny
                            color = COLORS['marformer_tiny']
                    else:
                        color = 'gray'

                    ax.scatter(true_vals, pred_vals, c=color, alpha=0.4, s=20)

            # Perfect agreement line
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, linewidth=1)

            # Formatting
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True, alpha=0.3)

            # Labels
            if row == 3:  # Bottom row
                ax.set_xlabel('True Entropy', fontsize=10)
            if col == 0:  # Left column
                ax.set_ylabel('Model Cross-Entropy', fontsize=10)

            # Column titles
            if row == 0:  # Top row
                ax.set_title(f'Training Size: {budget}', fontsize=10)

            # Row labels on left
            if col == 0:
                ax.text(-0.15, 0.5, model_name, rotation=90, ha='right', va='center',
                       transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    # Save plot
    node_sizes_in_data = []
    for key in results.keys():
        if isinstance(key, tuple) and len(key) >= 1:
            node_sizes_in_data.append(key[0])

    if node_sizes_in_data:
        node_sizes_str = "_".join(map(str, sorted(set(node_sizes_in_data))))
    else:
        node_sizes_str = "unknown"

    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/five_step_model_vs_true_nodes_{node_sizes_str}{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Five-step model vs true scatter plot saved to {save_path}")
    plt.close()


def create_additional_plots(results_dir: str, output_dir: str = None,
                          missing_rate: Optional[float] = None) -> None:
    """
    Create additional plots from saved experimental results.

    Args:
        results_dir: Directory containing experimental results
        output_dir: Directory to save plots (defaults to results_dir/../plots/additional)
        missing_rate: Missing rate for filename suffix
    """
    # Load results
    results = load_results_from_directory(results_dir)

    # Default output directory
    if output_dir is None:
        output_dir = str(Path(results_dir).parent / "plots" / "additional")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating additional plots from {len(results)} experimental configurations")
    logger.info(f"Output directory: {output_dir}")

    # Create both additional plot types
    plot_log_loss_convergence_curves(results, output_dir, missing_rate)
    plot_five_step_model_vs_true_scatter(results, output_dir, missing_rate)

    logger.info(f"Additional plots completed and saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Command line interface
    parser = argparse.ArgumentParser(description="Generate additional plots from experimental results")
    parser.add_argument('results_dir', help='Directory containing experimental results')
    parser.add_argument('--output-dir', help='Directory to save plots')
    parser.add_argument('--missing-rate', type=float, help='Missing rate for filename suffix')

    args = parser.parse_args()

    try:
        create_additional_plots(args.results_dir, args.output_dir, args.missing_rate)
    except Exception as e:
        logger.error(f"Failed to create additional plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)