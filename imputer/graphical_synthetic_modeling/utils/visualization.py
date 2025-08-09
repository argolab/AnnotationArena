"""
Visualization utilities for progressive imputation experiments.

Provides clean plotting functions for analyzing experimental results with
professional publication-quality formatting and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Professional publication-quality matplotlib settings
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

# Professional color palette for consistent visualization
COLORS = {
    'domain_em': '#1f77b4',     # Blue for EM baseline
    'true_model': '#2ca02c',    # Green for true model
    'Tiny': '#ff9999',          # Light red for tiny model
    'Small': '#cc4444',         # Medium red for small model  
    'Large': '#990000'          # Dark red for large model
}


def plot_convergence_curves(results: Dict[str, Any], output_dir: str = "plots", 
                          missing_rate: Optional[float] = None) -> None:
    """
    Plot cost vs KL divergence convergence curves for each node size.
    
    Creates separate plots for each graph size comparing neural imputer
    variants against the domain EM baseline.
    
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
            neural_kls = [r['neural_kl'] for r in experiment_results]
            neural_kl_stds = [r.get('neural_kl_std', 0.0) for r in experiment_results]
            domain_kls = [r['domain_kl'] for r in experiment_results]
            domain_kl_stds = [r.get('domain_kl_std', 0.0) for r in experiment_results]
            
            # Plot neural imputer variant with error bars
            ax.errorbar(costs, neural_kls, yerr=neural_kl_stds,
                       fmt='o-', label=f'Imputer ({imputer_size})', 
                       color=COLORS[imputer_size], linewidth=2, markersize=6, 
                       alpha=0.8, capsize=5)
            
            # Plot domain EM only once (same across all variants)
            if not domain_em_plotted:
                ax.errorbar(costs, domain_kls, yerr=domain_kl_stds,
                           fmt='s--', label='Domain EM', color=COLORS['domain_em'],
                           linewidth=2, markersize=6, alpha=0.8, capsize=5)
                domain_em_plotted = True
        
        ax.set_xlabel('Cost (Number of Training Samples)', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'Progressive Imputation: {n_nodes} Nodes', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for KL divergence
        
        plt.tight_layout()
        
        # Save with missing rate suffix if provided
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/convergence_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence curves for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_log_loss_comparison(results: Dict[str, Any], output_dir: str = "plots",
                           missing_rate: Optional[float] = None) -> None:
    """
    Plot 3-curve log-loss comparison: True model vs EM vs Neural imputer.
    
    Shows the progression of log-loss values across training budgets,
    comparing ground truth model, EM baseline, and neural imputer variants.
    
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
        
        # Plot true model and EM once per node size
        baseline_plotted = False
        
        # Plot each imputer variant
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Extract log-loss progression data
            costs = [r['budget'] for r in experiment_results]
            true_log_loss = [r.get('true_model_log_loss', float('inf')) for r in experiment_results]
            em_log_loss = [r.get('domain_log_loss', float('inf')) for r in experiment_results]
            neural_log_loss = [r.get('neural_log_loss', float('inf')) for r in experiment_results]
            
            # Plot baseline models only once per node size
            if not baseline_plotted:
                ax.plot(costs, true_log_loss, 'o-', label='True Model + True Params', 
                       color=COLORS['true_model'], linewidth=2, markersize=6, alpha=0.8)
                ax.plot(costs, em_log_loss, 's--', label='True Model + EM Params', 
                       color=COLORS['domain_em'], linewidth=2, markersize=6, alpha=0.8)
                baseline_plotted = True
            
            # Plot neural imputer variant
            ax.plot(costs, neural_log_loss, '^-', label=f'Imputer ({imputer_size})', 
                   color=COLORS[imputer_size], linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Cost (Number of Training Samples)', fontsize=12)
        ax.set_ylabel('Log-Loss', fontsize=12)
        ax.set_title(f'Log-Loss Comparison: {n_nodes} Nodes', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/log_loss_curves_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Log-loss curves for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_neural_vs_true_scatterplots(results: Dict[str, Any], output_dir: str = "plots",
                                    missing_rate: Optional[float] = None) -> None:
    """
    Create Neural Imputer vs True Model scatterplots with two color schemes.
    
    Plot 1: Neural vs True with 2 subplots:
    - Left: Color-coded by budget progression  
    - Right: Color-coded by imputer size
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract neural vs true log-loss data - individual sample level
    neural_true_data = []  # [(true_loss, neural_loss, budget, imputer_size)]
    
    budgets = set()
    imputer_sizes = set()
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Unknown')
        
        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)
            imputer_sizes.add(imputer_size)
            
            # Get individual log-loss values for each test sample
            true_values = step_result.get('true_model_log_loss_values', [])
            neural_values = step_result.get('neural_log_loss_values', [])
            
            # Pair up true and neural values for each individual test sample
            min_len = min(len(true_values), len(neural_values))
            for i in range(min_len):
                true_val = true_values[i]
                neural_val = neural_values[i]
                
                if not (np.isnan(true_val) or np.isinf(true_val) or 
                       np.isnan(neural_val) or np.isinf(neural_val)):
                    neural_true_data.append((true_val, neural_val, budget, imputer_size))
    
    # Create scatterplots if we have data
    if neural_true_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Color by Budget Progression with continuous gradient
        budget_list = sorted(budgets)
        budget_min, budget_max = min(budget_list), max(budget_list)
        
        # Create arrays for scatter plot with continuous color mapping
        true_vals = [x[0] for x in neural_true_data]
        neural_vals = [x[1] for x in neural_true_data]
        budget_vals = [x[2] for x in neural_true_data]
        
        # Use scatter with continuous color mapping
        scatter1 = ax1.scatter(true_vals, neural_vals, c=budget_vals, 
                              cmap='viridis', alpha=0.6, s=20, vmin=budget_min, vmax=budget_max)
        
        # Add colorbar for budget
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Budget (Training Samples)', fontsize=10)
        
        # Perfect agreement line
        min_val = min(min(true_vals), min(neural_vals))
        max_val = max(max(true_vals), max(neural_vals))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
                label='Perfect Agreement', linewidth=2)
        
        ax1.set_xlabel('True Model Log-Loss', fontsize=12)
        ax1.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
        ax1.set_title('Neural vs True (Budget Progression)', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Color by Imputer Size with discrete colors but continuous feel
        size_list = sorted(imputer_sizes)
        size_color_values = {'Tiny': 0.2, 'Small': 0.5, 'Large': 0.8}
        
        # Create color array based on imputer size
        size_vals = [size_color_values.get(x[3], 0.5) for x in neural_true_data]
        
        scatter2 = ax2.scatter(true_vals, neural_vals, c=size_vals, 
                              cmap='Reds', alpha=0.6, s=20, vmin=0, vmax=1)
        
        # Add discrete legend for imputer sizes
        for size in size_list:
            if size in size_color_values:
                color_val = size_color_values[size]
                color = plt.cm.Reds(color_val)
                ax2.scatter([], [], c=[color], label=f'{size} Imputer', s=50)
        
        # Perfect agreement line
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7,
                label='Perfect Agreement', linewidth=2)
        
        ax2.set_xlabel('True Model Log-Loss', fontsize=12)
        ax2.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
        ax2.set_title('Neural vs True (Imputer Size)', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/neural_vs_true_scatterplots{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Neural vs True scatterplots saved to {save_path}")
        plt.close()
    else:
        logger.warning("No valid neural vs true log-loss data available for scatterplots")


def plot_em_vs_true_scatterplots(results: Dict[str, Any], output_dir: str = "plots",
                                missing_rate: Optional[float] = None) -> None:
    """
    Create EM vs True Model scatterplot with budget color coding.
    
    Plot 2: EM vs True with 1 subplot:
    - Color-coded by budget progression
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract EM vs true log-loss data - individual sample level
    em_true_data = []  # [(true_loss, em_loss, budget)]
    
    budgets = set()
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)
            
            # Get individual log-loss values for each test sample
            true_values = step_result.get('true_model_log_loss_values', [])
            em_values = step_result.get('domain_log_loss_values', [])
            
            # Pair up true and EM values for each individual test sample
            min_len = min(len(true_values), len(em_values))
            for i in range(min_len):
                true_val = true_values[i]
                em_val = em_values[i]
                
                if not (np.isnan(true_val) or np.isinf(true_val) or 
                       np.isnan(em_val) or np.isinf(em_val)):
                    em_true_data.append((true_val, em_val, budget))
    
    # Create scatterplot if we have data
    if em_true_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Color by Budget Progression with continuous gradient
        budget_list = sorted(budgets)
        budget_min, budget_max = min(budget_list), max(budget_list)
        
        # Create arrays for scatter plot with continuous color mapping
        true_vals = [x[0] for x in em_true_data]
        em_vals = [x[1] for x in em_true_data]
        budget_vals = [x[2] for x in em_true_data]
        
        # Use scatter with continuous color mapping
        scatter = ax.scatter(true_vals, em_vals, c=budget_vals, 
                            cmap='plasma', alpha=0.6, s=20, vmin=budget_min, vmax=budget_max)
        
        # Add colorbar for budget
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Budget (Training Samples)', fontsize=10)
        
        # Perfect agreement line
        min_val = min(min(true_vals), min(em_vals))
        max_val = max(max(true_vals), max(em_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
               label='Perfect Agreement', linewidth=2)
        
        ax.set_xlabel('True Model Log-Loss', fontsize=12)
        ax.set_ylabel('EM Model Log-Loss', fontsize=12)
        ax.set_title('EM vs True Model (Budget Progression)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/em_vs_true_scatterplots{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"EM vs True scatterplots saved to {save_path}")
        plt.close()
    else:
        logger.warning("No valid EM vs true log-loss data available for scatterplots")

def create_unified_scatterplots(results, output_dir="improved_plots", missing_rate=None):
    """
    Create unified 3-subplot scatterplot: Neural vs True, EM vs True, all on same axes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract all data for consistent axis scaling
    neural_true_data = []
    em_true_data = []
    
    budgets = set()
    imputer_sizes = set()
    
    for key, policy_results in results.items():
        experiment_results = policy_results['results']
        imputer_size = policy_results.get('imputer_size', 'Unknown')
        
        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)
            imputer_sizes.add(imputer_size)
            
            # Get individual sample values
            true_values = step_result.get('true_model_log_loss_values', [])
            neural_values = step_result.get('neural_log_loss_values', [])
            em_values = step_result.get('domain_log_loss_values', [])
            
            # Collect neural vs true pairs
            min_len_neural = min(len(true_values), len(neural_values))
            for i in range(min_len_neural):
                if not (np.isnan(true_values[i]) or np.isinf(true_values[i]) or 
                       np.isnan(neural_values[i]) or np.isinf(neural_values[i])):
                    neural_true_data.append((true_values[i], neural_values[i], budget, imputer_size))
            
            # Collect EM vs true pairs
            min_len_em = min(len(true_values), len(em_values))
            for i in range(min_len_em):
                if not (np.isnan(true_values[i]) or np.isinf(true_values[i]) or 
                       np.isnan(em_values[i]) or np.isinf(em_values[i])):
                    em_true_data.append((true_values[i], em_values[i], budget))
    
    if not neural_true_data and not em_true_data:
        print("No valid data for scatterplots")
        return
    
    # Get consistent axis limits across all subplots
    all_true = ([x[0] for x in neural_true_data] + [x[0] for x in em_true_data])
    all_pred = ([x[1] for x in neural_true_data] + [x[1] for x in em_true_data])
    
    min_val = min(min(all_true), min(all_pred))
    max_val = max(max(all_true), max(all_pred))
    axis_margin = (max_val - min_val) * 0.05
    axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
    
    # Create 3-subplot figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Consistent scatter parameters
    scatter_size = 20
    scatter_alpha = 0.3
    
    budgets_list = sorted(budgets)
    budget_min, budget_max = min(budgets_list), max(budgets_list)
    
    # Subplot 1: Neural vs True (Budget Progression)
    if neural_true_data:
        true_vals = [x[0] for x in neural_true_data]
        neural_vals = [x[1] for x in neural_true_data]
        budget_vals = [x[2] for x in neural_true_data]
        
        scatter1 = ax1.scatter(true_vals, neural_vals, c=budget_vals, 
                              cmap='viridis', alpha=scatter_alpha, s=scatter_size, 
                              vmin=budget_min, vmax=budget_max)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Budget (Training Samples)', fontsize=10)
    
    # Perfect agreement line (all subplots)
    ax1.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, 
            label='Perfect Agreement', linewidth=2)
    
    ax1.set_xlim(axis_min, axis_max)
    ax1.set_ylim(axis_min, axis_max)
    ax1.set_xlabel('True Model Log-Loss', fontsize=12)
    ax1.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax1.set_title('Neural vs True (Budget Progression)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Neural vs True (Imputer Size)
    if neural_true_data:
        size_color_values = {'Tiny': 0.2, 'Small': 0.5, 'Large': 0.8}
        size_vals = [size_color_values.get(x[3], 0.5) for x in neural_true_data]
        
        scatter2 = ax2.scatter(true_vals, neural_vals, c=size_vals, 
                              cmap='Reds', alpha=scatter_alpha, s=scatter_size, vmin=0, vmax=1)
        
        # Add discrete legend
        for size in sorted(set([x[3] for x in neural_true_data])):
            if size in size_color_values:
                color_val = size_color_values[size]
                color = plt.cm.Reds(color_val)
                ax2.scatter([], [], c=[color], label=f'{size} Imputer', s=50)
    
    ax2.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7,
            label='Perfect Agreement', linewidth=2)
    
    ax2.set_xlim(axis_min, axis_max)
    ax2.set_ylim(axis_min, axis_max)
    ax2.set_xlabel('True Model Log-Loss', fontsize=12)
    ax2.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
    ax2.set_title('Neural vs True (Imputer Size)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: EM vs True (Budget Progression)
    if em_true_data:
        true_vals_em = [x[0] for x in em_true_data]
        em_vals = [x[1] for x in em_true_data]
        budget_vals_em = [x[2] for x in em_true_data]
        
        scatter3 = ax3.scatter(true_vals_em, em_vals, c=budget_vals_em, 
                              cmap='viridis', alpha=scatter_alpha, s=scatter_size,
                              vmin=budget_min, vmax=budget_max)
        
        # Add colorbar
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Budget (Training Samples)', fontsize=10)
    
    ax3.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7,
            label='Perfect Agreement', linewidth=2)
    
    ax3.set_xlim(axis_min, axis_max)
    ax3.set_ylim(axis_min, axis_max)
    ax3.set_xlabel('True Model Log-Loss', fontsize=12)
    ax3.set_ylabel('EM Model Log-Loss', fontsize=12)
    ax3.set_title('EM vs True (Budget Progression)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
    save_path = f"{output_dir}/unified_scatterplots{missing_suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Unified scatterplots saved to {save_path}")


def create_experiment_report(results: Dict[str, Any], output_dir: str = "plots",
                           missing_rate: Optional[float] = None) -> None:
    """
    Create comprehensive visualization report for experimental results.
    
    Generates all key plots and prints detailed statistical summary of results
    comparing neural imputer variants against domain EM baseline.
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save all plots  
        missing_rate: Missing rate for filename suffixes
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating comprehensive experiment report...")
    
    # Generate all visualization plots
    plot_convergence_curves(results, output_dir, missing_rate)
    plot_log_loss_comparison(results, output_dir, missing_rate)
    plot_neural_vs_true_scatterplots(results, output_dir, missing_rate)
    plot_em_vs_true_scatterplots(results, output_dir, missing_rate)

    try:
        create_unified_scatterplots(results, output_dir, missing_rate)
    except:
        pass
    
    logger.info(f"All plots saved to {output_dir}/ directory")
    
    # Print comprehensive statistical summary
    print("\\n" + "="*80)
    print("PROGRESSIVE IMPUTATION EXPERIMENT SUMMARY")
    print("="*80)
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        if len(experiment_results) >= 2:
            first_result = experiment_results[0]
            final_result = experiment_results[-1]
            
            # Extract policy and imputer info
            policy_name = policy_results.get('policy_name', 'Unknown')
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            print(f"\\nGraph Size: {n_nodes} nodes | Policy: {policy_name} | Imputer: {imputer_size}")
            print(f"Budget Range: {first_result['budget']} → {final_result['budget']} samples")
            
            # KL divergence progression
            neural_first_kl = first_result['neural_kl']
            neural_final_kl = final_result['neural_kl'] 
            neural_improvement = neural_first_kl / neural_final_kl
            
            domain_first_kl = first_result['domain_kl']
            domain_final_kl = final_result['domain_kl']
            domain_improvement = domain_first_kl / domain_final_kl
            
            print(f"Neural KL: {neural_first_kl:.4f} → {neural_final_kl:.4f} "
                  f"({neural_improvement:.1f}x improvement)")
            print(f"Domain KL: {domain_first_kl:.4f} → {domain_final_kl:.4f} "
                  f"({domain_improvement:.1f}x improvement)")
            
            # Final comparison
            kl_ratio = neural_final_kl / domain_final_kl
            winner = 'Neural' if kl_ratio < 1 else 'Domain'
            print(f"Final Winner: {winner} (Neural/Domain ratio: {kl_ratio:.2f})")
            
            # Log-loss comparison
            neural_log_loss = final_result.get('neural_log_loss', float('inf'))
            domain_log_loss = final_result.get('domain_log_loss', float('inf'))
            true_log_loss = final_result.get('true_model_log_loss', float('inf'))
            
            print(f"Final Log-Loss: Neural={neural_log_loss:.3f}, "
                  f"Domain={domain_log_loss:.3f}, True={true_log_loss:.3f}")
            
            # Training time summary
            total_time = policy_results.get('total_time', 0.0)
            print(f"Total Training Time: {total_time:.1f}s")
            
    print("\\n" + "="*80)