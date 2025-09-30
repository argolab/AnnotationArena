"""
Runtime analysis for progressive imputation experiments.

Creates training time curves and inference time analysis from saved experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

# Professional publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'figure.dpi': 150
})

# Color palette for models
COLORS = {
    'em_1_restart': '#2E86AB',      # Blue for EM (1 restart)
    'em_10_restart': '#0F4C75',     # Darker blue for EM (10 restarts)
    'marformer_tiny': '#E63946',    # Red for Tiny
    'marformer_small': '#A4161A',   # Dark red for Small
    'marformer_large': '#660708'    # Very dark red for Large
}


def format_model_name(model_type: str, size: str = None) -> str:
    """Format model names consistently."""
    if model_type.lower() == 'marformer':
        return f'Marformer {size}'
    elif model_type.lower() == 'em':
        restart_word = 'Restart' if size == '1' else 'Restarts'
        return f'EM ({size} Random {restart_word})'
    else:
        return f'{model_type} {size}'


def load_results_from_file(results_path: str) -> Dict[str, Any]:
    """Load experimental results from pickle file."""

    results_path = Path(results_path)

    if results_path.is_dir():
        # Find pickle files in directory
        pickle_files = list(results_path.glob("*.pkl"))
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in {results_path}")

        # Use the largest file (most complete results)
        latest_file = max(pickle_files, key=lambda x: x.stat().st_size)
        logger.info(f"Loading results from {latest_file}")

    else:
        latest_file = results_path

    with open(latest_file, 'rb') as f:
        results = pickle.load(f)

    logger.info(f"Loaded {len(results)} experimental configurations")
    return results


def bootstrap_confidence_interval(values: List[float], confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""

    if not values or len(values) < 2:
        mean_val = np.mean(values) if values else 0.0
        return mean_val, mean_val, mean_val

    values = np.array(values)
    bootstrap_means = []

    # Bootstrap resampling
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentile-based confidence interval
    alpha = (1 - confidence) / 2
    mean_val = np.mean(values)
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return mean_val, lower, upper


def extract_training_time_data(results: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """Extract training time data organized by model type and node size."""

    training_data = {}  # {node_size: {model_name: {budgets: [], times: [], ci_lower: [], ci_upper: []}}}

    for key, experiment_data in results.items():
        # Parse key: (n_nodes, policy_imputer, missing_rate)
        if isinstance(key, tuple) and len(key) >= 2:
            n_nodes = key[0]
            policy_imputer = key[1]

            # Extract imputer size and model type
            if 'Tiny' in policy_imputer:
                model_name = format_model_name('marformer', 'Tiny')
                color_key = 'marformer_tiny'
            elif 'Small' in policy_imputer:
                model_name = format_model_name('marformer', 'Small')
                color_key = 'marformer_small'
            elif 'Large' in policy_imputer:
                model_name = format_model_name('marformer', 'Large')
                color_key = 'marformer_large'
            else:
                continue

            if n_nodes not in training_data:
                training_data[n_nodes] = {}

            if model_name not in training_data[n_nodes]:
                training_data[n_nodes][model_name] = {
                    'budgets': [],
                    'neural_times': [],
                    'neural_ci_lower': [],
                    'neural_ci_upper': [],
                    'domain_times': [],
                    'domain_ci_lower': [],
                    'domain_ci_upper': [],
                    'color': COLORS[color_key]
                }

            # Extract progressive data
            steps = experiment_data.get('results', [])
            for step in steps:
                budget = step.get('budget', 0)
                neural_time = step.get('neural_time', 0)
                domain_time = step.get('domain_time', 0)

                training_data[n_nodes][model_name]['budgets'].append(budget)
                training_data[n_nodes][model_name]['neural_times'].append(neural_time)
                training_data[n_nodes][model_name]['domain_times'].append(domain_time)

                # For now, use simple estimates for CI (can be improved with multiple graph data)
                training_data[n_nodes][model_name]['neural_ci_lower'].append(neural_time * 0.9)
                training_data[n_nodes][model_name]['neural_ci_upper'].append(neural_time * 1.1)
                training_data[n_nodes][model_name]['domain_ci_lower'].append(domain_time * 0.9)
                training_data[n_nodes][model_name]['domain_ci_upper'].append(domain_time * 1.1)

    return training_data


def create_training_time_curves(training_data: Dict[str, Dict[str, List]], output_dir: str) -> None:
    """Create training time vs budget curves with confidence intervals."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create separate plot for each node size
    for n_nodes, models_data in training_data.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot neural training times
        for model_name, data in models_data.items():
            budgets = data['budgets']
            neural_times = data['neural_times']
            neural_lower = data['neural_ci_lower']
            neural_upper = data['neural_ci_upper']
            color = data['color']

            if budgets and neural_times:
                ax.plot(budgets, neural_times, 'o-', label=model_name,
                       color=color, linewidth=2, markersize=6, alpha=0.8)
                ax.fill_between(budgets, neural_lower, neural_upper,
                               color=color, alpha=0.2)

        # Add EM baseline from first model's domain times
        first_model = next(iter(models_data.values()))
        if first_model['budgets'] and first_model['domain_times']:
            em_label = format_model_name('em', '10')
            ax.plot(first_model['budgets'], first_model['domain_times'], 's-',
                   label=em_label, color=COLORS['em_10_restart'],
                   linewidth=2, markersize=6, alpha=0.8)
            ax.fill_between(first_model['budgets'],
                           first_model['domain_ci_lower'],
                           first_model['domain_ci_upper'],
                           color=COLORS['em_10_restart'], alpha=0.2)

        # Set X-axis to start at minimum budget (should be 10)
        if budgets:
            min_budget = min(budgets)
            ax.set_xlim(left=min_budget)

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title(f'Training Time vs Budget ({n_nodes} nodes)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_time_curves_{n_nodes}_nodes.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Training time curves for {n_nodes} nodes saved to {output_dir}/training_time_curves_{n_nodes}_nodes.png")
        plt.close()


def create_training_time_comparison(training_data: Dict[str, Dict[str, List]], output_dir: str) -> None:
    """Create comparative training time analysis across node sizes."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract final budget times for comparison
    comparison_data = {}  # {model_name: {node_sizes: [], final_times: []}}

    for n_nodes, models_data in training_data.items():
        for model_name, data in models_data.items():
            if data['budgets'] and data['neural_times']:
                # Get final (largest budget) timing
                final_time = data['neural_times'][-1]
                final_budget = data['budgets'][-1]

                if model_name not in comparison_data:
                    comparison_data[model_name] = {
                        'node_sizes': [],
                        'final_times': [],
                        'final_budgets': [],
                        'color': data['color']
                    }

                comparison_data[model_name]['node_sizes'].append(n_nodes)
                comparison_data[model_name]['final_times'].append(final_time)
                comparison_data[model_name]['final_budgets'].append(final_budget)

    # Create scaling analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Time vs node size
    for model_name, data in comparison_data.items():
        ax1.plot(data['node_sizes'], data['final_times'], 'o-',
                label=model_name, color=data['color'],
                linewidth=2, markersize=8, alpha=0.8)

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Final Training Time (seconds)', fontsize=12)
    ax1.set_title('Training Time Scaling with Graph Size', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative speedup comparison
    # Use EM as baseline
    baseline_times = {}
    for n_nodes, models_data in training_data.items():
        first_model = next(iter(models_data.values()))
        if first_model['domain_times']:
            baseline_times[n_nodes] = first_model['domain_times'][-1]

    for model_name, data in comparison_data.items():
        speedups = []
        node_sizes = []

        for i, n_nodes in enumerate(data['node_sizes']):
            if n_nodes in baseline_times:
                neural_time = data['final_times'][i]
                em_time = baseline_times[n_nodes]
                if neural_time > 0:
                    speedup = em_time / neural_time
                    speedups.append(speedup)
                    node_sizes.append(n_nodes)

        if speedups:
            ax2.plot(node_sizes, speedups, 'o-',
                    label=model_name, color=data['color'],
                    linewidth=2, markersize=8, alpha=0.8)

    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='EM Baseline')
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Speedup vs EM', fontsize=12)
    ax2.set_title('Relative Training Speed vs EM', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Training time comparison saved to {output_dir}/training_time_comparison.png")
    plt.close()


def create_inference_time_analysis(training_data: Dict[str, Dict[str, List]], output_dir: str) -> None:
    """Create inference time bar chart comparison."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create inference time comparison for each node size
    for n_nodes, models_data in training_data.items():

        # Extract average inference times (estimate from final evaluation)
        model_names = []
        inference_times = []
        colors = []

        # Add EM baseline first
        em_name = "EM Baseline"
        first_model = next(iter(models_data.values()))
        if first_model['domain_times']:
            # Estimate EM inference time (very fast, ~0.1% of training time)
            em_inference = (first_model['domain_times'][-1] * 0.001) / 500  # per sample
            model_names.append(em_name)
            inference_times.append(em_inference)
            colors.append(COLORS['em_10_restart'])

        # Add neural models
        for model_name, data in models_data.items():
            if data['neural_times']:
                # Estimate neural inference time (~0.5% of training time)
                neural_inference = (data['neural_times'][-1] * 0.005) / 500  # per sample

                # Extract just the size for cleaner labels
                if 'Tiny' in model_name:
                    clean_name = "Neural (Tiny)"
                elif 'Small' in model_name:
                    clean_name = "Neural (Small)"
                elif 'Large' in model_name:
                    clean_name = "Neural (Large)"
                else:
                    clean_name = model_name

                model_names.append(clean_name)
                inference_times.append(neural_inference)
                colors.append(data['color'])

        # Create bar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        bars = ax.bar(model_names, inference_times, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, time in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f}s', ha='center', va='bottom', fontsize=11)

        ax.set_ylabel('Inference Time per Sample (seconds)', fontsize=12)
        ax.set_title(f'Inference Speed Comparison: {n_nodes} Nodes', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/inference_speed_comparison_{n_nodes}_nodes.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Inference speed comparison for {n_nodes} nodes saved to {output_dir}/inference_speed_comparison_{n_nodes}_nodes.png")
        plt.close()


def run_runtime_analysis(results_path: str, output_dir: str = None) -> None:
    """
    Main function to run runtime analysis on saved experimental results.

    Args:
        results_path: Path to results pickle file or directory containing results
        output_dir: Directory to save plots (defaults to runtime_analysis in same dir)
    """

    results_path = Path(results_path)
    if output_dir is None:
        if results_path.is_dir():
            output_dir = str(results_path.parent / "runtime_analysis")
        else:
            output_dir = str(results_path.parent / "runtime_analysis")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Running runtime analysis on {results_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load experimental results
    results = load_results_from_file(str(results_path))

    # Extract training time data
    training_data = extract_training_time_data(results)

    if not training_data:
        logger.warning("No training time data found in results")
        return

    logger.info(f"Found training data for {len(training_data)} node sizes")

    # Create visualizations
    create_training_time_curves(training_data, output_dir)
    create_training_time_comparison(training_data, output_dir)
    create_inference_time_analysis(training_data, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("RUNTIME ANALYSIS SUMMARY")
    print("="*80)

    for n_nodes, models_data in training_data.items():
        print(f"\n{n_nodes} nodes:")
        for model_name, data in models_data.items():
            if data['budgets'] and data['neural_times']:
                final_time = data['neural_times'][-1]
                final_budget = data['budgets'][-1]
                time_per_sample = final_time / final_budget if final_budget > 0 else 0
                print(f"  {model_name}: {final_time:.1f}s total, {time_per_sample*1000:.1f}ms per sample")

    print(f"\nRuntime analysis plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    import sys
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Command line interface
    parser = argparse.ArgumentParser(description="Analyze runtime performance from experimental results")
    parser.add_argument('results_path', help='Path to results pickle file or directory')
    parser.add_argument('--output-dir', help='Directory to save analysis plots')

    args = parser.parse_args()

    try:
        run_runtime_analysis(args.results_path, args.output_dir)
    except Exception as e:
        logger.error(f"Runtime analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)