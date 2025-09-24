#!/usr/bin/env python3
"""
Generate improved plots from saved experiment results.

Creates:
1. KL divergence curves with log scale and improved legend
2. Scatter plots with true entropy heatmap coloring
3. Separate Large Marformer vs EM (10) scatter plot
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Use the existing color scheme and formatting
from utils.paper_plots import COLORS, format_model_name, get_step_progression_budgets

def load_experiment_results(results_dir: str = "outputs/results") -> Tuple[Dict, Dict]:
    """Load the latest experiment results and config."""
    results_path = Path(results_dir)

    # Find latest results pickle
    pickle_files = list(results_path.glob("results_*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(f"No results pickle files found in {results_dir}")

    latest_pickle = max(pickle_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_pickle}")

    with open(latest_pickle, 'rb') as f:
        results = pickle.load(f)

    # Load config
    with open(results_path / "experiment_config.json", 'r') as f:
        config = json.load(f)

    return results, config


def plot_improved_kl_curves(results: Dict[str, Any], output_dir: str = "plots",
                           missing_rate: Optional[float] = None) -> None:
    """
    Plot KL divergence curves with log scale and improved legend.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Group results by node size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results

    # Create plot for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get experiment data
        first_experiment = next(iter(node_results.values()))
        experiment_results = first_experiment['results']
        costs = [r['budget'] for r in experiment_results]

        # EM data with bootstrap CIs
        domain_kls = [r['domain_kl'] for r in experiment_results]
        domain_kl_lowers = [r.get('domain_kl_lower', r['domain_kl']) for r in experiment_results]
        domain_kl_uppers = [r.get('domain_kl_upper', r['domain_kl']) for r in experiment_results]

        domain_1_kls = [r.get('domain_1_kl', r['domain_kl']) for r in experiment_results]
        domain_1_kl_lowers = [r.get('domain_1_kl_lower', r.get('domain_1_kl', r['domain_kl'])) for r in experiment_results]
        domain_1_kl_uppers = [r.get('domain_1_kl_upper', r.get('domain_1_kl', r['domain_kl'])) for r in experiment_results]

        # Plot EM variants
        em_1_label = format_model_name('em', '1')
        ax.plot(costs, domain_1_kls, 's:', label=em_1_label,
               color=COLORS['em_1_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(costs, domain_1_kl_lowers, domain_1_kl_uppers,
                       color=COLORS['em_1_restart'], alpha=0.2)

        em_10_label = format_model_name('em', '10')
        ax.plot(costs, domain_kls, 's-', label=em_10_label,
               color=COLORS['em_10_restart'], linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(costs, domain_kl_lowers, domain_kl_uppers,
                       color=COLORS['em_10_restart'], alpha=0.2)

        # Plot all Marformer variants
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')

            costs = [r['budget'] for r in experiment_results]
            neural_kls = [r['neural_kl'] for r in experiment_results]
            neural_kl_lowers = [r.get('neural_kl_lower', r['neural_kl']) for r in experiment_results]
            neural_kl_uppers = [r.get('neural_kl_upper', r['neural_kl']) for r in experiment_results]

            color_key = f'marformer_{imputer_size.lower()}'
            if color_key in COLORS:
                marformer_label = format_model_name('marformer', imputer_size)
                ax.plot(costs, neural_kls, 'o-', label=marformer_label,
                       color=COLORS[color_key], linewidth=2, markersize=6, alpha=0.8)
                ax.fill_between(costs, neural_kl_lowers, neural_kl_uppers,
                               color=COLORS[color_key], alpha=0.2)

        # Set log scale
        ax.set_yscale('log')

        # Add True Model to legend only (no line)
        ax.plot([], [], color='green', linestyle='-', linewidth=2, alpha=0.8,
               label='True Model (KL = 0.0)')

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('KL divergence per marginal prediction (nats)', fontsize=12)
        ax.set_title(f'BN Domain: {n_nodes} nodes, ≈5 Parents per Node', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/improved_kl_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Improved KL curves saved to {save_path}")
        plt.close()


def plot_improved_scatter_plots(results: Dict[str, Any], output_dir: str = "plots",
                               missing_rate: Optional[float] = None) -> None:
    """
    Plot cross-entropy scatter plots: EM vs Marformer with true entropy heatmap coloring.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get step progression budgets
    step_budgets = get_step_progression_budgets(results)

    # Extract cross-entropy data for comparison
    model_data = {}  # {model_name: {budget: [(em_cross_entropy, marformer_cross_entropy, true_entropy)]}}

    # Get EM cross-entropy data first
    em_cross_entropy_data = {}  # {budget: [(em_cross_entropy, true_entropy)]}

    for key, policy_results in results.items():
        experiment_results = policy_results['results']

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets:
                true_entropy_values = step_result.get('true_entropy_values', [])
                domain_cross_entropy_values = step_result.get('domain_cross_entropy_values', [])

                if budget not in em_cross_entropy_data:
                    em_cross_entropy_data[budget] = []

                min_len = min(len(true_entropy_values), len(domain_cross_entropy_values))
                for i in range(min_len):
                    true_val = true_entropy_values[i]
                    em_val = domain_cross_entropy_values[i]
                    if not (np.isnan(true_val) or np.isinf(true_val) or
                           np.isnan(em_val) or np.isinf(em_val)):
                        em_cross_entropy_data[budget].append((em_val, true_val))
        break  # Only need one policy for EM data

    # Now get Marformer data and pair with EM
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Large')

        model_name = format_model_name('marformer', imputer_size)
        if model_name not in model_data:
            model_data[model_name] = {}

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets and budget in em_cross_entropy_data:
                true_entropy_values = step_result.get('true_entropy_values', [])
                neural_cross_entropy_values = step_result.get('neural_cross_entropy_values', [])

                if budget not in model_data[model_name]:
                    model_data[model_name][budget] = []

                # Get Marformer cross-entropy values
                marformer_vals = []
                min_len = min(len(true_entropy_values), len(neural_cross_entropy_values))
                for i in range(min_len):
                    true_val = true_entropy_values[i]
                    marformer_val = neural_cross_entropy_values[i]
                    if not (np.isnan(true_val) or np.isinf(true_val) or
                           np.isnan(marformer_val) or np.isinf(marformer_val)):
                        marformer_vals.append(marformer_val)

                # Pair with EM data
                em_vals = em_cross_entropy_data[budget]
                min_pairs = min(len(marformer_vals), len(em_vals))

                for i in range(min_pairs):
                    em_cross_entropy, true_entropy_color = em_vals[i]
                    marformer_cross_entropy = marformer_vals[i]
                    model_data[model_name][budget].append((em_cross_entropy, marformer_cross_entropy, true_entropy_color))

    # Create 3×3 subplot grid (only Marformer sizes vs EM)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Model order for rows (only Marformer sizes)
    model_order = [
        format_model_name('marformer', 'Large'),
        format_model_name('marformer', 'Small'),
        format_model_name('marformer', 'Tiny')
    ]

    # Get consistent axis limits and color range
    all_em_vals = []
    all_marformer_vals = []
    all_color_vals = []
    for model_name, budget_data in model_data.items():
        for budget, pairs in budget_data.items():
            if pairs:
                em_vals = [p[0] for p in pairs]
                marformer_vals = [p[1] for p in pairs]
                color_vals = [p[2] for p in pairs]
                all_em_vals.extend(em_vals)
                all_marformer_vals.extend(marformer_vals)
                all_color_vals.extend(color_vals)

    if all_em_vals and all_marformer_vals:
        min_val = min(min(all_em_vals), min(all_marformer_vals))
        max_val = max(max(all_em_vals), max(all_marformer_vals))
        axis_margin = (max_val - min_val) * 0.05
        axis_min, axis_max = min_val - axis_margin, max_val + axis_margin

        # Color range for heatmap
        color_min, color_max = min(all_color_vals), max(all_color_vals)
    else:
        axis_min, axis_max = 0, 1
        color_min, color_max = 0, 1

    # Plot each subplot
    for row, model_name in enumerate(model_order):
        for col, budget in enumerate(step_budgets):
            ax = axes[row, col]

            if model_name in model_data and budget in model_data[model_name]:
                pairs = model_data[model_name][budget]
                if pairs:
                    em_vals = [p[0] for p in pairs]
                    marformer_vals = [p[1] for p in pairs]
                    color_vals = [p[2] for p in pairs]

                    # Scatter plot with true entropy coloring
                    scatter = ax.scatter(em_vals, marformer_vals, c=color_vals,
                                       cmap='viridis', alpha=0.6, s=15,
                                       vmin=color_min, vmax=color_max)

            # Perfect agreement line (more visible)
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.8, linewidth=1.5)

            # Formatting
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True, alpha=0.3)

            # Labels
            if row == 2:  # Bottom row (now row 2 instead of 3)
                ax.set_xlabel('EM (10) Cross-Entropy', fontsize=10)
            if col == 0:  # Left column
                ax.set_ylabel('Model Cross-Entropy', fontsize=10)

            # Titles (only training size)
            if row == 0:  # Top row
                ax.set_title(f'Training Size: {budget}', fontsize=10)

            # Model names on left
            if col == 0:
                ax.text(-0.13, 0.5, model_name, rotation=90, ha='right', va='center',
                       transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    # Add colorbar with proper spacing
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('True Entropy', fontsize=12)

    # Save plot
    node_sizes_in_data = sorted(set(key[0] for key in results.keys()))
    node_sizes_str = "_".join(map(str, node_sizes_in_data))
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/improved_scatter_entropy_heatmap_nodes_{node_sizes_str}{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Improved scatter plots with heatmap saved to {save_path}")
    plt.close()


def plot_large_vs_em_scatter(results: Dict[str, Any], output_dir: str = "plots",
                           missing_rate: Optional[float] = None) -> None:
    """
    Plot separate Large Marformer vs EM (10) cross-entropy scatter with heatmap coloring.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get step progression budgets
    step_budgets = get_step_progression_budgets(results)

    # Extract cross-entropy data for pairing
    budget_data = {}  # {budget: [(em_cross_entropy, large_cross_entropy, true_entropy_color)]}

    # Get Large Marformer data
    large_cross_entropy_data = {}
    for key, policy_results in results.items():
        imputer_size = policy_results.get('imputer_size', 'Large')
        if imputer_size == 'Large':
            experiment_results = policy_results['results']

            for step_result in experiment_results:
                budget = step_result['budget']
                if budget in step_budgets:
                    true_entropy_values = step_result.get('true_entropy_values', [])
                    neural_cross_entropy_values = step_result.get('neural_cross_entropy_values', [])

                    if budget not in large_cross_entropy_data:
                        large_cross_entropy_data[budget] = []

                    min_len = min(len(true_entropy_values), len(neural_cross_entropy_values))
                    for i in range(min_len):
                        true_val = true_entropy_values[i]
                        neural_val = neural_cross_entropy_values[i]
                        if not (np.isnan(true_val) or np.isinf(true_val) or
                               np.isnan(neural_val) or np.isinf(neural_val)):
                            large_cross_entropy_data[budget].append((neural_val, true_val))

    # Get EM data and pair with Large Marformer
    for key, policy_results in results.items():
        experiment_results = policy_results['results']

        for step_result in experiment_results:
            budget = step_result['budget']
            if budget in step_budgets and budget in large_cross_entropy_data:
                true_entropy_values = step_result.get('true_entropy_values', [])
                domain_cross_entropy_values = step_result.get('domain_cross_entropy_values', [])

                if budget not in budget_data:
                    budget_data[budget] = []

                # Get EM cross-entropy values
                em_vals = []
                min_len = min(len(true_entropy_values), len(domain_cross_entropy_values))
                for i in range(min_len):
                    true_val = true_entropy_values[i]
                    em_val = domain_cross_entropy_values[i]
                    if not (np.isnan(true_val) or np.isinf(true_val) or
                           np.isnan(em_val) or np.isinf(em_val)):
                        em_vals.append(em_val)

                # Pair with Large Marformer data
                large_vals = large_cross_entropy_data[budget]
                min_pairs = min(len(em_vals), len(large_vals))

                for i in range(min_pairs):
                    em_cross_entropy = em_vals[i]
                    large_cross_entropy, true_entropy_color = large_vals[i]
                    budget_data[budget].append((em_cross_entropy, large_cross_entropy, true_entropy_color))

        break  # Only need one policy for EM data

    # Create 1×3 subplot for the 3 budgets
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Get consistent axis limits and color range
    all_em_vals = []
    all_large_vals = []
    all_color_vals = []
    for budget, pairs in budget_data.items():
        for em_val, large_val, color_val in pairs:
            all_em_vals.append(em_val)
            all_large_vals.append(large_val)
            all_color_vals.append(color_val)

    if all_em_vals and all_large_vals:
        all_vals = all_em_vals + all_large_vals
        axis_margin = (max(all_vals) - min(all_vals)) * 0.05
        axis_min, axis_max = min(all_vals) - axis_margin, max(all_vals) + axis_margin
        color_min, color_max = min(all_color_vals), max(all_color_vals)
    else:
        axis_min, axis_max = 0, 1
        color_min, color_max = 0, 1

    # Plot each budget
    scatter = None
    for col, budget in enumerate(step_budgets):
        ax = axes[col]

        if budget in budget_data and budget_data[budget]:
            pairs = budget_data[budget]
            em_vals = [p[0] for p in pairs]
            large_vals = [p[1] for p in pairs]
            color_vals = [p[2] for p in pairs]

            scatter = ax.scatter(em_vals, large_vals, c=color_vals,
                               cmap='viridis', alpha=0.6, s=20,
                               vmin=color_min, vmax=color_max)

        # Perfect agreement line
        ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.8, linewidth=1.5)

        # Formatting
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(True, alpha=0.3)

        # Labels
        ax.set_xlabel('EM (10) Cross-Entropy', fontsize=12)
        if col == 0:
            ax.set_ylabel('MARFORMER Large Cross-Entropy', fontsize=12)

        # Title
        ax.set_title(f'Training Size: {budget}', fontsize=12)

    plt.tight_layout()

    # Add colorbar with proper spacing
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=axes, shrink=0.8, aspect=15, pad=0.02)
        cbar.set_label('True Entropy', fontsize=12)

    # Save plot
    node_sizes_in_data = sorted(set(key[0] for key in results.keys()))
    node_sizes_str = "_".join(map(str, node_sizes_in_data))
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/large_vs_em_scatter_heatmap_nodes_{node_sizes_str}{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Large vs EM scatter plot with heatmap saved to {save_path}")
    plt.close()


def main():
    """Generate all improved plots."""
    # Load results
    results, config = load_experiment_results()

    print(f"Loaded {len(results)} experiment configurations")
    print(f"Config: {config['node_sizes']} nodes, {config['missing_rates']} missing rates")

    # Create output directory
    output_dir = Path("outputs/plots/improved")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each missing rate
    for missing_rate in config['missing_rates']:
        print(f"\nGenerating improved plots for missing rate: {missing_rate}")

        # Filter results for this missing rate
        filtered_results = {}
        for key, value in results.items():
            if len(key) == 3 and key[2] == missing_rate:
                # Convert back to original key format (n_nodes, policy_imputer)
                original_key = (key[0], key[1])
                filtered_results[original_key] = value

        if filtered_results:
            print(f"Found {len(filtered_results)} configurations for missing rate {missing_rate}")

            # Generate all three improved plots
            plot_improved_kl_curves(filtered_results, str(output_dir), missing_rate)
            plot_improved_scatter_plots(filtered_results, str(output_dir), missing_rate)
            plot_large_vs_em_scatter(filtered_results, str(output_dir), missing_rate)

        else:
            print(f"No results found for missing rate: {missing_rate}")

    print(f"\nAll improved plots saved to: {output_dir}")


if __name__ == "__main__":
    main()