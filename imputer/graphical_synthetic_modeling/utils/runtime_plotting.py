"""
Runtime performance visualization for progressive imputation experiments.

Provides comprehensive runtime analysis comparing neural imputers vs EM baselines
across different training budgets and model sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Use same color scheme as main visualization
COLORS = {
    'domain_em': '#1f77b4',     # Blue for EM baseline
    'Tiny': '#ff9999',          # Light red for tiny model
    'Small': '#cc4444',         # Medium red for small model  
    'Large': '#990000'          # Dark red for large model
}


def plot_training_time_curves(results: Dict[str, Any], output_dir: str = "plots", 
                             missing_rate: Optional[float] = None) -> None:
    """
    Plot training time vs budget curves for each node size.
    
    Creates separate plots comparing neural imputer variants against 
    the domain EM baseline for training time performance.
    
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
        
        # Track whether domain EM has been plotted (same for all variants)
        domain_em_plotted = False
        
        # Plot all imputer variants for this node size
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            
            # Extract imputer size from policy results
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Extract progression data
            costs = [r['budget'] for r in experiment_results]
            neural_times = [r.get('neural_time', 0.0) for r in experiment_results]
            neural_time_stds = [r.get('neural_time_std', 0.0) for r in experiment_results]
            domain_times = [r.get('domain_time', 0.0) for r in experiment_results]
            domain_time_stds = [r.get('domain_time_std', 0.0) for r in experiment_results]
            
            # Add small horizontal offset to prevent error bar overlap
            offset = 0.02 * (max(costs) - min(costs))
            if imputer_size == 'Tiny':
                neural_costs = [c - offset for c in costs]
            elif imputer_size == 'Small':
                neural_costs = costs
            else:  # Large
                neural_costs = [c + offset for c in costs]
            
            # Plot neural imputer variant with error bars
            ax.errorbar(neural_costs, neural_times, yerr=neural_time_stds,
                       fmt='o-', label=f'Imputer ({imputer_size})', 
                       color=COLORS[imputer_size], linewidth=2, markersize=6, 
                       alpha=0.8, capsize=5)
            
            # Plot domain EM only once (same across all variants)
            if not domain_em_plotted:
                ax.errorbar(costs, domain_times, yerr=domain_time_stds,
                           fmt='s--', label='Domain EM', color=COLORS['domain_em'],
                           linewidth=2, markersize=6, alpha=0.8, capsize=5)
                domain_em_plotted = True
        
        ax.set_xlabel('Budget (Number of Training Samples)', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title(f'Training Time Performance: {n_nodes} Nodes', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with missing rate suffix if provided
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/training_time_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training time curves for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_inference_time_comparison(results: Dict[str, Any], output_dir: str = "plots",
                                  missing_rate: Optional[float] = None) -> None:
    """
    Plot inference time comparison as bar charts.
    
    Compares inference speed between neural imputers and EM baseline
    using the final budget step for each configuration.
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect inference time data from final step of each configuration
    inference_data = {}  # {n_nodes: {imputer_size: time, 'EM': time}}
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in inference_data:
            inference_data[n_nodes] = {}
            
        experiment_results = policy_results['results']
        if not experiment_results:
            continue
            
        # Use final step for inference timing
        final_result = experiment_results[-1]
        imputer_size = policy_results.get('imputer_size', 'Large')
        
        # Estimate inference time per sample (training time / training samples)
        neural_time = final_result.get('neural_time', 0.0)
        domain_time = final_result.get('domain_time', 0.0)
        n_training_samples = final_result.get('n_training_samples', 1)
        
        # Store per-sample inference time estimates
        inference_data[n_nodes][imputer_size] = neural_time / max(n_training_samples, 1)
        inference_data[n_nodes]['EM'] = domain_time / max(n_training_samples, 1)
    
    # Create bar chart for each node size
    for n_nodes, node_data in inference_data.items():
        if not node_data:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for bar chart
        methods = []
        times = []
        colors = []
        
        # Add EM baseline
        if 'EM' in node_data:
            methods.append('EM Baseline')
            times.append(node_data['EM'])
            colors.append(COLORS['domain_em'])
        
        # Add neural imputer variants
        for imputer_size in ['Tiny', 'Small', 'Large']:
            if imputer_size in node_data:
                methods.append(f'Neural ({imputer_size})')
                times.append(node_data[imputer_size])
                colors.append(COLORS[imputer_size])
        
        if methods:
            bars = ax.bar(methods, times, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Inference Time per Sample (seconds)', fontsize=12)
        ax.set_title(f'Inference Speed Comparison: {n_nodes} Nodes', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/inference_time_comparison_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Inference time comparison for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_runtime_efficiency_analysis(results: Dict[str, Any], output_dir: str = "plots",
                                    missing_rate: Optional[float] = None) -> None:
    """
    Plot runtime efficiency: performance per second of training.
    
    Creates scatter plot of KL divergence vs training time to show
    which methods achieve better performance per unit of computation.
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group by node size for separate plots
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results
    
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Extract final performance and total training time
            if not experiment_results:
                continue
                
            final_result = experiment_results[-1]
            neural_kl = final_result.get('neural_kl', float('inf'))
            domain_kl = final_result.get('domain_kl', float('inf'))
            
            total_neural_time = sum(r.get('neural_time', 0.0) for r in experiment_results)
            total_domain_time = sum(r.get('domain_time', 0.0) for r in experiment_results)
            
            # Plot neural imputer
            if not np.isinf(neural_kl) and total_neural_time > 0:
                ax.scatter(total_neural_time, neural_kl, 
                          color=COLORS[imputer_size], s=100, alpha=0.7,
                          label=f'Neural ({imputer_size})', marker='o')
            
            # Plot domain EM (only once per node size)
            if imputer_size == 'Large':  # Plot EM only once
                if not np.isinf(domain_kl) and total_domain_time > 0:
                    ax.scatter(total_domain_time, domain_kl,
                              color=COLORS['domain_em'], s=100, alpha=0.7,
                              label='Domain EM', marker='s')
        
        ax.set_xlabel('Total Training Time (seconds)', fontsize=12)
        ax.set_ylabel('Final KL Divergence', fontsize=12)
        ax.set_title(f'Runtime Efficiency Analysis: {n_nodes} Nodes', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/runtime_efficiency_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Runtime efficiency analysis for {n_nodes} nodes saved to {save_path}")
        plt.close()


def create_runtime_analysis_report(results: Dict[str, Any], output_dir: str = "plots",
                                  missing_rate: Optional[float] = None) -> None:
    """
    Create runtime analysis report.
    
    Generates runtime-related visualizations (training time curves and inference comparison)
    and prints summary statistics.
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save all plots
        missing_rate: Missing rate for filename suffixes
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Generating runtime analysis report...")
    
    # Generate runtime visualizations (removed efficiency plot)
    plot_training_time_curves(results, output_dir, missing_rate)
    plot_inference_time_comparison(results, output_dir, missing_rate)
    
    # Print runtime summary statistics
    print("\n" + "="*60)
    print("RUNTIME PERFORMANCE SUMMARY")
    print("="*60)
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        if not experiment_results:
            continue
            
        imputer_size = policy_results.get('imputer_size', 'Large')
        final_result = experiment_results[-1]
        
        neural_time = final_result.get('neural_time', 0.0)
        domain_time = final_result.get('domain_time', 0.0)
        total_time = policy_results.get('total_time', 0.0)
        
        print(f"\nGraph: {n_nodes} nodes | Imputer: {imputer_size}")
        print(f"  Final step training time: Neural={neural_time:.2f}s, EM={domain_time:.2f}s")
        print(f"  Speed ratio (EM/Neural): {domain_time/max(neural_time, 1e-6):.2f}x")
        print(f"  Total experiment time: {total_time:.2f}s")
    
    print("\n" + "="*60)
    logger.info("Runtime analysis report completed")