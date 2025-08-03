"""
Visualization utilities for progressive imputation experiments.

This module provides plotting functions to visualize:
1. Cost vs KL reduction curves comparing methods
2. KL divergence distributions with histograms
3. Variable-level analysis for detailed insights

Author: Prabhav Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_cost_vs_kl_curves(results: Dict[str, Any], save_path: Optional[str] = None, 
                          show_plot: bool = True) -> None:
    """
    Plot cost vs KL divergence curves comparing Imputer and Domain EM methods.
    
    Args:
        results: Results dictionary from progressive experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        # Extract data
        costs = [r['budget'] for r in experiment_results]
        neural_kls = [r['neural_kl'] for r in experiment_results]
        domain_kls = [r['domain_kl'] for r in experiment_results]
        
        # Plot curves with updated names
        ax.plot(costs, neural_kls, 'o-', label=f'Imputer (N={n_nodes})', 
                linewidth=2, markersize=6, alpha=0.8)
        ax.plot(costs, domain_kls, 's--', label=f'Domain EM (N={n_nodes})', 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Cost (Number of Training Samples)', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('Progressive Imputation: Cost vs Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for KL divergence
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cost vs KL plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_kl_histograms(results: Dict[str, Any], save_path: Optional[str] = None,
                      show_plot: bool = True) -> None:
    """
    Plot KL divergence histograms with frequency (not density) and statistics.
    
    Args:
        results: Results dictionary from progressive experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect all KL values across all experiments
    all_neural_kls = []
    all_domain_kls = []
    
    for (n_nodes, policy_name), policy_results in results.items():
        neural_kls = [r['neural_kl'] for r in policy_results['results']]
        domain_kls = [r['domain_kl'] for r in policy_results['results']]
        all_neural_kls.extend(neural_kls)
        all_domain_kls.extend(domain_kls)
    
    # Plot Imputer KL distribution
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(all_neural_kls, bins=20, alpha=0.7, 
                                    color='steelblue', edgecolor='black')
    
    # Add statistics
    neural_mean = np.mean(all_neural_kls)
    neural_median = np.median(all_neural_kls)
    
    ax1.axvline(neural_mean, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {neural_mean:.3f}')
    ax1.axvline(neural_median, color='green', linestyle='--', linewidth=2,
               label=f'Median: {neural_median:.3f}')
    
    ax1.set_xlabel('KL Divergence', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Imputer KL Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Domain EM KL distribution
    ax2 = axes[1]
    counts, bins, patches = ax2.hist(all_domain_kls, bins=20, alpha=0.7, 
                                    color='orange', edgecolor='black')
    
    # Add statistics
    domain_mean = np.mean(all_domain_kls)
    domain_median = np.median(all_domain_kls)
    
    ax2.axvline(domain_mean, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {domain_mean:.3f}')
    ax2.axvline(domain_median, color='green', linestyle='--', linewidth=2,
               label=f'Median: {domain_median:.3f}')
    
    ax2.set_xlabel('KL Divergence', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Domain EM KL Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"KL histogram plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_learning_curves(results: Dict[str, Any], save_path: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Plot learning curves showing training progress over budget steps.
    
    Args:
        results: Results dictionary from progressive experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        # Extract data
        costs = [r['budget'] for r in experiment_results]
        neural_kls = [r['neural_kl'] for r in experiment_results]
        domain_kls = [r['domain_kl'] for r in experiment_results]
        neural_times = [r['neural_time'] for r in experiment_results]
        domain_times = [r['domain_time'] for r in experiment_results]
        
        # Plot KL curves
        ax1.plot(costs, neural_kls, 'o-', label=f'Neural (N={n_nodes})', 
                linewidth=2, markersize=6)
        ax1.plot(costs, domain_kls, 's--', label=f'Domain (N={n_nodes})', 
                linewidth=2, markersize=6)
        
        # Plot training times
        ax2.plot(costs, neural_times, 'o-', label=f'Neural (N={n_nodes})', 
                linewidth=2, markersize=6)
        ax2.plot(costs, domain_times, 's--', label=f'Domain (N={n_nodes})', 
                linewidth=2, markersize=6)
    
    # Configure KL plot
    ax1.set_xlabel('Training Samples', fontsize=12)
    ax1.set_ylabel('KL Divergence', fontsize=12)
    ax1.set_title('Learning Curves: Performance vs Training Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Configure time plot
    ax2.set_xlabel('Training Samples', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time vs Dataset Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_method_comparison(results: Dict[str, Any], save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Plot side-by-side comparison of final performance for both methods.
    
    Args:
        results: Results dictionary from progressive experiment
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # Extract final results
    graph_sizes = []
    neural_final_kls = []
    domain_final_kls = []
    neural_improvements = []
    domain_improvements = []
    
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        if len(experiment_results) >= 2:
            graph_sizes.append(n_nodes)
            
            # Final KL values
            neural_final_kls.append(experiment_results[-1]['neural_kl'])
            domain_final_kls.append(experiment_results[-1]['domain_kl'])
            
            # Improvements (first/final)
            neural_improvements.append(
                experiment_results[0]['neural_kl'] / experiment_results[-1]['neural_kl']
            )
            domain_improvements.append(
                experiment_results[0]['domain_kl'] / experiment_results[-1]['domain_kl']
            )
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar width
    width = 0.35
    x = np.arange(len(graph_sizes))
    
    # Final KL comparison
    ax1.bar(x - width/2, neural_final_kls, width, label='Neural', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, domain_final_kls, width, label='Domain', alpha=0.8, color='orange')
    ax1.set_xlabel('Graph Size (nodes)', fontsize=11)
    ax1.set_ylabel('Final KL Divergence', fontsize=11)
    ax1.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(graph_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Improvement factors
    ax2.bar(x - width/2, neural_improvements, width, label='Neural', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, domain_improvements, width, label='Domain', alpha=0.8, color='orange')
    ax2.set_xlabel('Graph Size (nodes)', fontsize=11)
    ax2.set_ylabel('Improvement Factor', fontsize=11)
    ax2.set_title('Learning Improvement (First/Final KL)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(graph_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Relative performance (Neural/Domain ratio)
    if len(neural_final_kls) > 0 and len(domain_final_kls) > 0:
        relative_performance = [n/d for n, d in zip(neural_final_kls, domain_final_kls)]
        colors = ['green' if r < 1 else 'red' for r in relative_performance]
        
        ax3.bar(x, relative_performance, width*2, alpha=0.8, color=colors)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
        ax3.set_xlabel('Graph Size (nodes)', fontsize=11)
        ax3.set_ylabel('Neural KL / Domain KL', fontsize=11)
        ax3.set_title('Relative Performance (< 1 = Neural Better)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(graph_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(relative_performance):
            ax3.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    neural_times = []
    domain_times = []
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        total_neural_time = sum(r['neural_time'] for r in experiment_results)
        total_domain_time = sum(r['domain_time'] for r in experiment_results)
        neural_times.append(total_neural_time)
        domain_times.append(total_domain_time)
    
    if neural_times and domain_times:
        ax4.bar(x - width/2, neural_times, width, label='Neural', alpha=0.8, color='steelblue')
        ax4.bar(x + width/2, domain_times, width, label='Domain', alpha=0.8, color='orange')
        ax4.set_xlabel('Graph Size (nodes)', fontsize=11)
        ax4.set_ylabel('Total Training Time (seconds)', fontsize=11)
        ax4.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(graph_sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    plt.suptitle('Comprehensive Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Method comparison plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_experiment_report(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Create essential plots for experiment analysis.
    
    Args:
        results: Results dictionary from progressive experiment
        output_dir: Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info("Generating experiment plots...")
    
    # Generate only the plots we want
    plot_cost_vs_kl_curves(results, 
                          save_path=f"{output_dir}/cost_vs_kl_curves.png", 
                          show_plot=False)
    
    plot_kl_histograms(results, 
                      save_path=f"{output_dir}/kl_histograms.png", 
                      show_plot=False)
    
    logger.info(f"Plots saved to {output_dir}/ directory")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for (n_nodes, policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        if len(experiment_results) >= 2:
            first_result = experiment_results[0]
            final_result = experiment_results[-1]
            
            print(f"\nGraph Size: {n_nodes} nodes | Policy: {policy_name}")
            print(f"Budget Range: {first_result['budget']} → {final_result['budget']} samples")
            print(f"Neural KL: {first_result['neural_kl']:.4f} → {final_result['neural_kl']:.4f} "
                  f"({first_result['neural_kl']/final_result['neural_kl']:.1f}x improvement)")
            print(f"Domain KL: {first_result['domain_kl']:.4f} → {final_result['domain_kl']:.4f} "
                  f"({first_result['domain_kl']/final_result['domain_kl']:.1f}x improvement)")
            print(f"Winner: {'Neural' if final_result['neural_kl'] < final_result['domain_kl'] else 'Domain'} "
                  f"(ratio: {final_result['neural_kl']/final_result['domain_kl']:.2f})")
            print(f"Total Time: {policy_results['total_time']:.1f}s")

def plot_variable_analysis(results: Dict[str, Any], test_data: List = None, 
                          save_path: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Plot variable-level KL divergence analysis (if detailed data available).
    
    Args:
        results: Results dictionary from progressive experiment
        test_data: Test dataset for variable analysis
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # This would require modification to store per-variable KL divergences
    # For now, just show a placeholder structure
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example: if we had per-variable KL data
    # variables = ['Node_0', 'Node_1', 'Node_2', 'Node_3', 'Node_4']
    # neural_var_kls = [0.1, 0.2, 0.15, 0.3, 0.25]
    # domain_var_kls = [0.15, 0.25, 0.2, 0.35, 0.3]
    
    ax.text(0.5, 0.5, 'Variable-level analysis requires\nmodified evaluation storage\n(per-variable KL divergences)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Variable-Level KL Analysis (Future Feature)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Variable analysis plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()