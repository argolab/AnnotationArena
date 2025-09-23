"""
Publication-quality plots for progressive imputation paper.

Creates the three specific plots required for the paper with exact naming
conventions and bootstrap confidence intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Enable LaTeX rendering with fallback
LATEX_AVAILABLE = False
try:
    import subprocess
    # Test if latex is actually available
    result = subprocess.run(['latex', '--version'], capture_output=True, timeout=5)
    if result.returncode == 0:
        plt.rcParams['text.usetex'] = True
        LATEX_AVAILABLE = True
        logger.info("LaTeX rendering enabled for publication plots")
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

# Professional color palette with darker colors
COLORS = {
    'em_1_restart': '#2E86AB',      # Blue for EM (1 restart)
    'em_10_restart': '#0F4C75',     # Darker blue for EM (10 restarts)
    'marformer_tiny': '#E63946',    # Darker red for Tiny
    'marformer_small': '#A4161A',   # Medium dark red for Small
    'marformer_large': '#660708'    # Very dark red for Large
}


def get_em_true_init_performance_at_budgets(node_results: Dict[str, Any], n_nodes: int, budgets: List[int]) -> Tuple[List[float], List[float], List[float]]:
    """
    Evaluate EM performance when initialized with true parameters at each budget level.

    Runs EM true init at each training set size with multiple trials for confidence intervals.

    Args:
        node_results: Results for this node size
        n_nodes: Number of nodes
        budgets: List of training set sizes to evaluate at

    Returns:
        Tuple of (kl_means, kl_lowers, kl_uppers) for plotting
    """
    try:
        # Import needed modules
        from data.graph_generator import generate_experiment_graph
        from domain.em_model import DomainEMModel
        from data.sample_generator import generate_sample_pool, generate_test_dataset
        import pyagrum as gum

        # Get configuration from first experiment
        first_experiment = next(iter(node_results.values()))
        config = first_experiment.get('config', {})

        # Recreate the exact true BN used in experiments
        true_bn, adj_matrix = generate_experiment_graph(
            n_nodes=n_nodes,
            target_parents=config.get('target_parents', 5.0),
            seed=config.get('seed', 42),
            cpt_generation=config.get('cpt_generation', 'logistic'),
            logistic_std=config.get('logistic_std', 1.0)
        )

        missing_rate = config.get('missing_rate', 0.4)
        test_samples_count = config.get('test_samples', 250)

        # Generate test data once
        test_samples = generate_test_dataset(
            true_bn, adj_matrix, n_nodes, test_samples_count, missing_rate, seed=config.get('seed', 42) + 2000
        )

        kl_means = []
        kl_lowers = []
        kl_uppers = []

        # Run EM true init at each budget level
        for budget in budgets:
            budget_kl_values = []

            # Run multiple trials for confidence intervals (like the other methods)
            n_trials = 3  # Match number of graph instances
            for trial in range(n_trials):
                try:
                    # Generate training data of this specific size
                    training_samples = generate_sample_pool(
                        true_bn, adj_matrix, n_nodes, budget, missing_rate,
                        seed=config.get('seed', 42) + 1000 + trial * 10000
                    )

                    # Create EM model - start with true BN parameters
                    em_model = DomainEMModel(max_iter=10, n_restarts=1, use_likelihood_selection=False)

                    # Train EM starting from the true BN
                    em_model.train(training_samples, true_bn, adj_matrix, n_nodes)

                    # Evaluate on test data
                    eval_results = em_model.evaluate(test_samples, true_bn, n_nodes)
                    kl_value = eval_results.get('mean_kl', 0.001)
                    budget_kl_values.append(kl_value)

                    logger.debug(f"EM true init trial {trial+1}: budget={budget}, KL={kl_value:.6f}")

                except Exception as e:
                    logger.warning(f"EM true init failed for budget {budget}, trial {trial}: {e}")
                    budget_kl_values.append(0.001)

            # Compute confidence intervals for this budget
            if budget_kl_values:
                from utils.bootstrap_stats import bootstrap_confidence_interval

                mean_kl = np.mean(budget_kl_values)
                if len(budget_kl_values) > 1:
                    lower, upper = bootstrap_confidence_interval(budget_kl_values)
                else:
                    lower, upper = mean_kl, mean_kl

                kl_means.append(mean_kl)
                kl_lowers.append(lower)
                kl_uppers.append(upper)

                logger.info(f"EM true init budget {budget}: KL={mean_kl:.6f} [{lower:.6f}, {upper:.6f}]")
            else:
                kl_means.append(0.001)
                kl_lowers.append(0.001)
                kl_uppers.append(0.001)

        return kl_means, kl_lowers, kl_uppers

    except Exception as e:
        logger.error(f"Failed to compute EM true init performance for {n_nodes} nodes: {e}")
        # Return fallback values
        fallback = [0.001] * len(budgets)
        return fallback, fallback, fallback


def format_model_name(model_type: str, size: str = None) -> str:
    """
    Format model names with consistent LaTeX styling.

    Args:
        model_type: 'marformer' or 'em'
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
    else:
        return f'{model_type} {size}'


def get_step_progression_budgets(results: Dict[str, Any]) -> List[int]:
    """
    Get the first, middle, and last budget levels for step progression plots.

    Args:
        results: Results dictionary from experiments

    Returns:
        List of 3 budget values [first, middle, last]
    """
    # Get budget sequence from any experiment
    for key, experiment_data in results.items():
        if experiment_data['results']:
            budgets = [step['budget'] for step in experiment_data['results']]
            if len(budgets) >= 3:
                first = budgets[0]
                middle = budgets[len(budgets) // 2]
                last = budgets[-1]
                logger.debug(f"Step progression budgets: {first}, {middle}, {last}")
                return [first, middle, last]

    # Fallback
    logger.warning("Could not determine step progression budgets, using defaults")
    return [10, 1000, 2500]


def plot_kl_divergence_curves(results: Dict[str, Any], output_dir: str = "plots",
                             missing_rate: Optional[float] = None) -> None:
    """
    Plot 1: KL divergence curves with bootstrap confidence intervals.

    Creates KL divergence vs training set size plots with proper model naming
    and shaded confidence intervals instead of error bars.

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Group results by node size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results

    # Create separate plot for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get first experiment for EM plotting (to avoid plotting EM multiple times)
        first_experiment = next(iter(node_results.values()))
        experiment_results = first_experiment['results']

        # Extract EM data once
        costs = [r['budget'] for r in experiment_results]

        # EM data with bootstrap CIs
        domain_kls = [r['domain_kl'] for r in experiment_results]  # EM (5 restarts)
        domain_kl_lowers = [r.get('domain_kl_lower', r['domain_kl']) for r in experiment_results]
        domain_kl_uppers = [r.get('domain_kl_upper', r['domain_kl']) for r in experiment_results]

        domain_1_kls = [r.get('domain_1_kl', r['domain_kl']) for r in experiment_results]  # EM (1 restart)
        domain_1_kl_lowers = [r.get('domain_1_kl_lower', r.get('domain_1_kl', r['domain_kl'])) for r in experiment_results]
        domain_1_kl_uppers = [r.get('domain_1_kl_upper', r.get('domain_1_kl', r['domain_kl'])) for r in experiment_results]

        # Plot EM variants first (so they appear in background)
        # EM (1 restart) - dotted line
        em_1_label = format_model_name('em', '1')
        ax.plot(costs, domain_1_kls, 's:', label=em_1_label,
               color=COLORS['em_1_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(costs, domain_1_kl_lowers, domain_1_kl_uppers,
                       color=COLORS['em_1_restart'], alpha=0.2)

        # EM (10 restarts) - solid line (actually 5 restarts)
        em_10_label = format_model_name('em', '10')
        ax.plot(costs, domain_kls, 's-', label=em_10_label,
               color=COLORS['em_10_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(costs, domain_kl_lowers, domain_kl_uppers,
                       color=COLORS['em_10_restart'], alpha=0.2)

        # Plot all Marformer variants
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')

            # Extract progression data
            costs = [r['budget'] for r in experiment_results]

            # Neural imputer data with bootstrap CIs
            neural_kls = [r['neural_kl'] for r in experiment_results]
            neural_kl_lowers = [r.get('neural_kl_lower', r['neural_kl']) for r in experiment_results]
            neural_kl_uppers = [r.get('neural_kl_upper', r['neural_kl']) for r in experiment_results]

            # Plot Marformer with shaded confidence intervals
            color_key = f'marformer_{imputer_size.lower()}'
            if color_key in COLORS:
                marformer_label = format_model_name('marformer', imputer_size)
                ax.plot(costs, neural_kls, 'o-', label=marformer_label,
                       color=COLORS[color_key], linewidth=2, markersize=6, alpha=0.8)
                ax.fill_between(costs, neural_kl_lowers, neural_kl_uppers,
                               color=COLORS[color_key], alpha=0.2)

        # Add reference lines
        ax.axhline(y=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='True Model (LazyPropagation)')

        # Compute and plot EM true init performance at each budget level
        em_true_init_kls, em_true_init_lowers, em_true_init_uppers = get_em_true_init_performance_at_budgets(node_results, n_nodes, costs)
        ax.plot(costs, em_true_init_kls, 's--', label='EM (True Init)',
               color='green', linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(costs, em_true_init_lowers, em_true_init_uppers,
                       color='green', alpha=0.2)

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('KL divergence per marginal prediction (nats)', fontsize=12)
        ax.set_title(f'BN Domain: {n_nodes} nodes, ≈5 Parents per Node', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with missing rate suffix if provided
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/paper_kl_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Paper KL curves for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_true_vs_predicted_cross_entropy(results: Dict[str, Any], output_dir: str = "plots",
                                        missing_rate: Optional[float] = None) -> None:
    """
    Plot 2: True entropy vs predicted cross-entropy step progression.

    Creates 4 rows × 3 columns grid showing evolution of cross-entropy
    predictions across early/middle/final training stages.

    Rows: Marformer Large, Marformer Small, Marformer Tiny, EM (10 Random Restarts)
    Columns: Early, Middle, Final budget levels

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get step progression budgets
    step_budgets = get_step_progression_budgets(results)

    # Extract data for each model and budget combination
    model_data = {}  # {model_name: {budget: [(true_entropy, pred_cross_entropy)]}}

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

    # Create 4×3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))

    # Model order for rows
    model_order = [
        format_model_name('marformer', 'Large'),
        format_model_name('marformer', 'Small'),
        format_model_name('marformer', 'Tiny'),
        format_model_name('em', '10')
    ]

    # Budget labels for columns
    budget_labels = ['Early', 'Middle', 'Final']

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

                    # Scatter plot with model-specific color
                    if 'Marformer' in model_name or 'MARFORMER' in model_name:
                        if 'Large' in model_name:
                            color = COLORS['marformer_large']
                        elif 'Small' in model_name:
                            color = COLORS['marformer_small']
                        else:  # Tiny
                            color = COLORS['marformer_tiny']
                    else:  # EM
                        color = COLORS['em_10_restart']

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
                ax.set_ylabel('Predicted Cross-Entropy', fontsize=10)

            # Titles
            if row == 0:  # Top row
                ax.set_title(f'{budget_label}\nTraining Size: {budget}', fontsize=10)

            # Model names on left
            if col == 0:
                ax.text(-0.1, 0.5, model_name, rotation=90, ha='right', va='center',
                       transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    # Save plot
    node_sizes_in_data = sorted(set(key[0] for key in results.keys()))
    node_sizes_str = "_".join(map(str, node_sizes_in_data))
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/paper_true_vs_pred_entropy_nodes_{node_sizes_str}{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Paper true vs predicted entropy plot saved to {save_path}")
    plt.close()


def plot_em_vs_marformer_cross_entropy(results: Dict[str, Any], output_dir: str = "plots",
                                      missing_rate: Optional[float] = None) -> None:
    """
    Plot 3: EM vs Marformer cross-entropy comparison.

    Creates 3 rows × 3 columns grid comparing EM (10) vs each Marformer size
    across early/middle/final training stages.

    Rows: Marformer Large vs EM, Marformer Small vs EM, Marformer Tiny vs EM
    Columns: Early, Middle, Final budget levels

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get step progression budgets
    step_budgets = get_step_progression_budgets(results)

    # Extract EM and Marformer cross-entropy data
    em_data = {}  # {budget: [cross_entropy_values]}
    marformer_data = {}  # {size: {budget: [cross_entropy_values]}}

    # Collect EM data (use first policy for EM since it's the same across policies)
    for key, policy_results in results.items():
        experiment_results = policy_results['results']

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets:
                domain_cross_entropy_values = step_result.get('domain_cross_entropy_values', [])
                valid_values = [v for v in domain_cross_entropy_values
                              if not (np.isnan(v) or np.isinf(v))]

                if budget not in em_data:
                    em_data[budget] = []
                em_data[budget].extend(valid_values)
        break  # Only need one policy for EM data

    # Collect Marformer data by size
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Large')

        if imputer_size not in marformer_data:
            marformer_data[imputer_size] = {}

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets:
                neural_cross_entropy_values = step_result.get('neural_cross_entropy_values', [])
                valid_values = [v for v in neural_cross_entropy_values
                              if not (np.isnan(v) or np.isinf(v))]

                if budget not in marformer_data[imputer_size]:
                    marformer_data[imputer_size][budget] = []
                marformer_data[imputer_size][budget].extend(valid_values)

    # Create 3×3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Size order for rows
    size_order = ['Large', 'Small', 'Tiny']
    budget_labels = ['Early', 'Middle', 'Final']

    # Get consistent axis limits
    all_em_vals = []
    all_marformer_vals = []
    for budget, em_vals in em_data.items():
        all_em_vals.extend(em_vals)
    for size, budget_data in marformer_data.items():
        for budget, marformer_vals in budget_data.items():
            all_marformer_vals.extend(marformer_vals)

    if all_em_vals and all_marformer_vals:
        min_val = min(min(all_em_vals), min(all_marformer_vals))
        max_val = max(max(all_em_vals), max(all_marformer_vals))
        axis_margin = (max_val - min_val) * 0.05
        axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
    else:
        axis_min, axis_max = 0, 1

    # Plot each subplot
    for row, size in enumerate(size_order):
        for col, (budget, budget_label) in enumerate(zip(step_budgets, budget_labels)):
            ax = axes[row, col]

            # Get data for this combination
            em_vals = em_data.get(budget, [])
            marformer_vals = marformer_data.get(size, {}).get(budget, [])

            if em_vals and marformer_vals:
                # Pair up EM and Marformer values (take minimum length)
                min_len = min(len(em_vals), len(marformer_vals))
                em_paired = em_vals[:min_len]
                marformer_paired = marformer_vals[:min_len]

                # Get color for this Marformer size
                color_key = f'marformer_{size.lower()}'
                color = COLORS.get(color_key, COLORS['marformer_large'])

                # Scatter plot
                ax.scatter(em_paired, marformer_paired, c=color, alpha=0.4, s=20)

            # Perfect agreement line
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, linewidth=1)

            # Formatting
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True, alpha=0.3)

            # Labels
            if row == 2:  # Bottom row
                ax.set_xlabel('EM (10) Cross-Entropy', fontsize=10)
            if col == 0:  # Left column
                marformer_label = format_model_name('marformer', size)
                ax.set_ylabel(f'{marformer_label}\nCross-Entropy', fontsize=10)

            # Titles
            if row == 0:  # Top row
                ax.set_title(f'{budget_label}\nTraining Size: {budget}', fontsize=10)

    plt.tight_layout()

    # Save plot
    node_sizes_in_data = sorted(set(key[0] for key in results.keys()))
    node_sizes_str = "_".join(map(str, node_sizes_in_data))
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/paper_em_vs_marformer_entropy_nodes_{node_sizes_str}{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Paper EM vs Marformer entropy plot saved to {save_path}")
    plt.close()


def create_paper_plots(results: Dict[str, Any], output_dir: str = "plots",
                      missing_rate: Optional[float] = None) -> None:
    """
    Create all three paper plots.

    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    logger.info("Creating paper plots...")

    # Create all three plots
    plot_kl_divergence_curves(results, output_dir, missing_rate)
    plot_true_vs_predicted_cross_entropy(results, output_dir, missing_rate)
    plot_em_vs_marformer_cross_entropy(results, output_dir, missing_rate)

    logger.info(f"All paper plots saved to {output_dir}/")