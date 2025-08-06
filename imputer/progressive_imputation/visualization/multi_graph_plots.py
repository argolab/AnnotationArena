"""
Visualization utilities for multi-graph progressive imputation experiments.

This module provides plotting functions for multi-graph experiments with
statistical error bars and separate plots for different node sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Set clean matplotlib style for publication
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

# Clean color palette for different methods
COLORS = {
    'domain_em': '#CD853F',     # Peru
    'true_model': '#4169E1',    # Royal blue
    'Tiny': '#FF6B6B',          # Light red
    'Small': '#4ECDC4',         # Teal
    'Large': '#45B7D1'          # Blue
}

def plot_cost_vs_kl_curves_separate_nodes(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Plot separate cost vs KL curves for each node size with error bars.
    Now handles multiple imputer variants in the same plot.
    
    Args:
        results: Results dictionary from multi-graph experiment
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group results by node size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results
    
    # Create separate plot for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Track domain EM (only plot once since it's the same for all imputer variants)
        domain_em_plotted = False
        n_graphs = 1
        
        # Plot all imputer variants
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            n_graphs = policy_results.get('n_graphs', 1)
            
            # Extract data with error bars
            costs = [r['budget'] for r in experiment_results]
            neural_kl_means = [r['neural_kl_mean'] for r in experiment_results]
            neural_kl_stds = [r['neural_kl_std'] for r in experiment_results]
            domain_kl_means = [r['domain_kl_mean'] for r in experiment_results]
            domain_kl_stds = [r['domain_kl_std'] for r in experiment_results]
            
            # Get imputer size from combined name (e.g., "RandomExample_Large")
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Plot Imputer variant with error bars
            ax.errorbar(costs, neural_kl_means, yerr=neural_kl_stds, 
                       fmt='o-', label=f'Imputer ({imputer_size})', color=COLORS[imputer_size],
                       linewidth=2, markersize=6, alpha=0.8, capsize=5)
            
            # Plot Domain EM only once (same across all variants)
            if not domain_em_plotted:
                ax.errorbar(costs, domain_kl_means, yerr=domain_kl_stds, 
                           fmt='s--', label='Domain EM', color=COLORS['domain_em'],
                           linewidth=2, markersize=6, alpha=0.8, capsize=5)
                domain_em_plotted = True
        
        ax.set_xlabel('Cost (Number of Training Samples)', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'Convergence Analysis: {n_nodes} Nodes (±1 std, {n_graphs} graphs)', 
                    fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = f"{output_dir}/cost_vs_kl_{n_nodes}_nodes.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cost vs KL plot for {n_nodes} nodes saved to {save_path}")
        plt.close()

def plot_final_performance_comparison(results: Dict[str, Any], save_path: Optional[str] = None,
                                    show_plot: bool = True) -> None:
    """
    Plot final performance comparison across node sizes with error bars.
    
    Args:
        results: Results dictionary from multi-graph experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data organized by node size
    node_sizes = []
    neural_means = []
    neural_stds = []
    domain_means = []
    domain_stds = []
    neural_time_means = []
    neural_time_stds = []  
    domain_time_means = []
    domain_time_stds = []
    
    for (n_nodes, policy_name), policy_results in results.items():
        if len(policy_results['results']) > 0:
            node_sizes.append(n_nodes)
            
            # Final performance
            final_result = policy_results['results'][-1]
            neural_means.append(final_result['neural_kl_mean'])
            neural_stds.append(final_result['neural_kl_std'])
            domain_means.append(final_result['domain_kl_mean'])
            domain_stds.append(final_result['domain_kl_std'])
            
            # Total training times
            neural_time_means.append(policy_results['total_time_mean'])
            neural_time_stds.append(policy_results['total_time_std'])
            domain_time_means.append(policy_results['total_time_mean'])  # Same experiment time
            domain_time_stds.append(policy_results['total_time_std'])
    
    if not node_sizes:
        logger.warning("No data to plot")
        return
        
    x = np.arange(len(node_sizes))
    width = 0.35
    
    # Final KL comparison with error bars
    ax1.bar(x - width/2, neural_means, width, yerr=neural_stds, 
           label='Neural Imputer', alpha=0.8, color='steelblue', capsize=5)
    ax1.bar(x + width/2, domain_means, width, yerr=domain_stds,
           label='Domain EM', alpha=0.8, color='orange', capsize=5)
    ax1.set_xlabel('Graph Size (nodes)', fontsize=11)
    ax1.set_ylabel('Final KL Divergence', fontsize=11)
    ax1.set_title('Final Performance Comparison (±1 std)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(node_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Relative performance (Neural/Domain ratio) with error propagation
    relative_means = []
    relative_stds = []
    for i in range(len(neural_means)):
        # Ratio of means
        ratio_mean = neural_means[i] / domain_means[i]
        relative_means.append(ratio_mean)
        
        # Error propagation for ratio: sqrt((std_a/a)^2 + (std_b/b)^2) * |a/b|
        rel_error_neural = neural_stds[i] / neural_means[i] if neural_means[i] > 0 else 0
        rel_error_domain = domain_stds[i] / domain_means[i] if domain_means[i] > 0 else 0
        ratio_std = ratio_mean * np.sqrt(rel_error_neural**2 + rel_error_domain**2)
        relative_stds.append(ratio_std)
    
    colors = ['green' if r < 1 else 'red' for r in relative_means]
    ax2.bar(x, relative_means, width*2, yerr=relative_stds, alpha=0.8, 
           color=colors, capsize=5)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
    ax2.set_xlabel('Graph Size (nodes)', fontsize=11)
    ax2.set_ylabel('Neural KL / Domain KL', fontsize=11)
    ax2.set_title('Relative Performance (< 1 = Neural Better, ±1 std)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(node_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (v, std) in enumerate(zip(relative_means, relative_stds)):
        ax2.text(i, v + std + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Learning improvement comparison
    neural_improvements = []
    neural_improvement_stds = []
    domain_improvements = []
    domain_improvement_stds = []
    
    for (n_nodes, policy_name), policy_results in results.items():
        if len(policy_results['results']) >= 2:
            first_result = policy_results['results'][0]
            final_result = policy_results['results'][-1]
            
            # Neural improvement
            neural_first_values = first_result['neural_kl_values']
            neural_final_values = final_result['neural_kl_values']
            neural_improvement_values = [f/l for f, l in zip(neural_first_values, neural_final_values)]
            neural_improvements.append(np.mean(neural_improvement_values))
            neural_improvement_stds.append(np.std(neural_improvement_values))
            
            # Domain improvement  
            domain_first_values = first_result['domain_kl_values']
            domain_final_values = final_result['domain_kl_values']
            domain_improvement_values = [f/l for f, l in zip(domain_first_values, domain_final_values)]
            domain_improvements.append(np.mean(domain_improvement_values))
            domain_improvement_stds.append(np.std(domain_improvement_values))
    
    if neural_improvements and domain_improvements:
        ax3.bar(x - width/2, neural_improvements, width, yerr=neural_improvement_stds,
               label='Neural Imputer', alpha=0.8, color='steelblue', capsize=5)
        ax3.bar(x + width/2, domain_improvements, width, yerr=domain_improvement_stds,
               label='Domain EM', alpha=0.8, color='orange', capsize=5)
        ax3.set_xlabel('Graph Size (nodes)', fontsize=11)
        ax3.set_ylabel('Improvement Factor', fontsize=11)
        ax3.set_title('Learning Improvement (First/Final KL, ±1 std)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(node_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Training time comparison with error bars
    ax4.bar(x - width/2, neural_time_means, width, yerr=neural_time_stds,
           label='Neural Imputer', alpha=0.8, color='steelblue', capsize=5)
    ax4.bar(x + width/2, domain_time_means, width, yerr=domain_time_stds,
           label='Domain EM', alpha=0.8, color='orange', capsize=5)
    ax4.set_xlabel('Graph Size (nodes)', fontsize=11)
    ax4.set_ylabel('Total Training Time (seconds)', fontsize=11)
    ax4.set_title('Training Time Comparison (±1 std)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(node_sizes)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle('Multi-Graph Experiment Results with Statistical Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance comparison plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_convergence_analysis(results: Dict[str, Any], save_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
    """
    Plot convergence analysis showing how performance varies across budget steps.
    
    Args:
        results: Results dictionary from multi-graph experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by node size
    results_by_nodes = {}
    for (n_nodes, policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][policy_name] = policy_results
    
    for idx, (n_nodes, node_results) in enumerate(results_by_nodes.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        for policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            
            # Extract budget steps and performance
            budgets = [r['budget'] for r in experiment_results]
            neural_means = [r['neural_kl_mean'] for r in experiment_results]
            neural_stds = [r['neural_kl_std'] for r in experiment_results]
            domain_means = [r['domain_kl_mean'] for r in experiment_results]
            domain_stds = [r['domain_kl_std'] for r in experiment_results]
            
            # Plot with confidence intervals
            ax.fill_between(budgets, 
                          np.array(neural_means) - np.array(neural_stds),
                          np.array(neural_means) + np.array(neural_stds),
                          alpha=0.2, color=COLORS['imputer'])
            # Get imputer size from results (if available)
            config = policy_results.get('config', {})
            imputer_size = config.get('imputer_size', 'Large')
            
            ax.plot(budgets, neural_means, 'o-', label=f'Imputer ({imputer_size})', 
                   color=COLORS['imputer'], linewidth=2, markersize=6)
            
            ax.fill_between(budgets,
                          np.array(domain_means) - np.array(domain_stds), 
                          np.array(domain_means) + np.array(domain_stds),
                          alpha=0.2, color=COLORS['domain_em'])
            ax.plot(budgets, domain_means, 's--', label='Domain EM',
                   color=COLORS['domain_em'], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.set_title(f'{n_nodes} Nodes (±1 std, {policy_results["n_graphs"]} graphs)', 
                    fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Hide unused subplots
    for idx in range(len(results_by_nodes), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Convergence Analysis Across Node Sizes', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence analysis plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_kl_frequency_distributions(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Generate KL frequency distribution histograms for each budget step.
    Now handles multiple imputer variants - creates separate frequency plots for each variant.
    
    Args:
        results: Results dictionary from multi-graph experiment
        output_dir: Directory to save plots
    """
    # Create kl_frequency subdirectory
    kl_freq_dir = Path(output_dir) / "kl_frequency"
    kl_freq_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating KL frequency distribution plots...")
    
    # Group results by node size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results
    
    # Generate frequency plots for each step and each imputer variant
    for n_nodes, node_results in results_by_nodes.items():
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            
            # Get imputer size
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            for step_idx, step_result in enumerate(experiment_results):
                budget = step_result['budget']
                
                # Get individual KL values (from all graphs)
                neural_kl_values = step_result.get('neural_kl_values', [])
                domain_kl_values = step_result.get('domain_kl_values', [])
                
                if not neural_kl_values or not domain_kl_values:
                    logger.warning(f"No individual KL values for {imputer_size} budget {budget}, skipping frequency plot")
                    continue
                
                # Create side-by-side histograms
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Neural imputer histogram
                ax1.hist(neural_kl_values, bins=50, color=COLORS[imputer_size], alpha=0.7, edgecolor='black')
                neural_mean = np.mean(neural_kl_values)
                neural_median = np.median(neural_kl_values)
                ax1.axvline(neural_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {neural_mean:.3f}')
                ax1.axvline(neural_median, color='darkblue', linestyle='--', linewidth=2, label=f'Median: {neural_median:.3f}')
                ax1.set_xlabel('KL Divergence')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Imputer ({imputer_size}) KL Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Domain EM histogram
                ax2.hist(domain_kl_values, bins=50, color=COLORS['domain_em'], alpha=0.7, edgecolor='black')
                domain_mean = np.mean(domain_kl_values)
                domain_median = np.median(domain_kl_values)
                ax2.axvline(domain_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {domain_mean:.3f}')
                ax2.axvline(domain_median, color='darkblue', linestyle='--', linewidth=2, label=f'Median: {domain_median:.3f}')
                ax2.set_xlabel('KL Divergence')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Domain EM KL Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Overall title
                fig.suptitle(f'{n_nodes} Nodes - Budget {budget} Samples - KL Distributions on Test Data', fontsize=14)
                
                plt.tight_layout()
                
                save_path = kl_freq_dir / f"kl_freq_{imputer_size}_budget_{budget:04d}_nodes_{n_nodes}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.debug(f"KL frequency plot saved: {save_path}")
    
    logger.info(f"KL frequency distribution plots saved to {kl_freq_dir}/")

def create_multi_graph_experiment_report(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Create comprehensive plots for multi-graph experiment analysis.
    
    Args:
        results: Results dictionary from multi-graph experiment
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info("Generating multi-graph experiment plots...")
    
    # 1. Separate plots for each node size
    plot_cost_vs_kl_curves_separate_nodes(results, output_dir)
    
    # 2. KL frequency distribution plots
    plot_kl_frequency_distributions(results, output_dir)
    
    # 3. Convergence analysis (only if multiple node sizes)
    results_by_nodes = {}
    for (n_nodes, policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][policy_name] = policy_results
    
    if len(results_by_nodes) > 1:
        plot_convergence_analysis(results,
                                save_path=f"{output_dir}/convergence_analysis.png", 
                                show_plot=False)
    
    logger.info(f"Multi-graph plots saved to {output_dir}/ directory")
    
    # Print detailed summary statistics
    print("\n" + "="*80)
    print("MULTI-GRAPH EXPERIMENT SUMMARY")
    print("="*80)
    
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        n_graphs = policy_results.get('n_graphs', 1)
        
        if len(experiment_results) >= 2:
            first_result = experiment_results[0]
            final_result = experiment_results[-1]
            
            print(f"\nGraph Size: {n_nodes} nodes | Policy: {policy_name} | Graphs: {n_graphs}")
            print(f"Budget Range: {first_result['budget']} → {final_result['budget']} samples")
            
            # Neural results
            print(f"Neural KL: {first_result['neural_kl_mean']:.4f}±{first_result['neural_kl_std']:.4f} → "
                  f"{final_result['neural_kl_mean']:.4f}±{final_result['neural_kl_std']:.4f}")
            neural_improvement = first_result['neural_kl_mean'] / final_result['neural_kl_mean']
            print(f"Neural Improvement: {neural_improvement:.1f}x")
            
            # Domain results
            print(f"Domain KL: {first_result['domain_kl_mean']:.4f}±{first_result['domain_kl_std']:.4f} → "
                  f"{final_result['domain_kl_mean']:.4f}±{final_result['domain_kl_std']:.4f}")
            domain_improvement = first_result['domain_kl_mean'] / final_result['domain_kl_mean']
            print(f"Domain Improvement: {domain_improvement:.1f}x")
            
            # Comparison
            ratio = final_result['neural_kl_mean'] / final_result['domain_kl_mean']
            winner = 'Neural' if ratio < 1 else 'Domain'
            print(f"Winner: {winner} (ratio: {ratio:.2f})")
            
            # Time
            print(f"Total Time: {policy_results['total_time_mean']:.1f}±{policy_results['total_time_std']:.1f}s")
            
            # Statistical significance test (simple)
            neural_final_values = final_result['neural_kl_values']
            domain_final_values = final_result['domain_kl_values']
            
            if len(neural_final_values) > 1 and len(domain_final_values) > 1:
                from scipy import stats
                try:
                    t_stat, p_value = stats.ttest_ind(neural_final_values, domain_final_values)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"Statistical Test: t={t_stat:.2f}, p={p_value:.4f} {significance}")
                except ImportError:
                    print("Statistical Test: scipy not available")
            
    print("\n" + "="*80)