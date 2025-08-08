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
    
    # Extract neural vs true log-loss data 
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
            
            # Get individual log-loss values (from graph aggregation)
            true_values = step_result.get('true_model_log_loss_values', [])
            neural_values = step_result.get('neural_log_loss_values', [])
            
            # Pair up true and neural values
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
        
        # Subplot 1: Color by Budget Progression
        budget_list = sorted(budgets)
        budget_colors = cm.viridis(np.linspace(0, 1, len(budget_list)))
        budget_to_color = {budget: budget_colors[i] for i, budget in enumerate(budget_list)}
        
        for i, (true_val, neural_val, budget, _) in enumerate(neural_true_data):
            ax1.scatter(true_val, neural_val, c=[budget_to_color[budget]], 
                       alpha=0.6, s=30)
        
        # Add budget legend
        for i, budget in enumerate(budget_list):
            ax1.scatter([], [], c=[budget_colors[i]], label=f'Budget {budget}', s=50)
        
        # Perfect agreement line
        all_true = [x[0] for x in neural_true_data]
        all_neural = [x[1] for x in neural_true_data]
        min_val = min(min(all_true), min(all_neural))
        max_val = max(max(all_true), max(all_neural))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
                label='Perfect Agreement')
        
        ax1.set_xlabel('True Model Log-Loss', fontsize=12)
        ax1.set_ylabel('Neural Imputer Log-Loss', fontsize=12)
        ax1.set_title('Neural vs True (Budget Progression)', fontsize=12)
        ax1.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Color by Imputer Size
        size_list = sorted(imputer_sizes)
        size_color_map = {'Tiny': '#ff9999', 'Small': '#cc4444', 'Large': '#990000'}
        
        for i, (true_val, neural_val, _, imputer_size) in enumerate(neural_true_data):
            color = size_color_map.get(imputer_size, '#666666')
            ax2.scatter(true_val, neural_val, c=color, alpha=0.6, s=30)
        
        # Add imputer size legend
        for size in size_list:
            if size in size_color_map:
                ax2.scatter([], [], c=size_color_map[size], label=f'{size} Imputer', s=50)
        
        # Perfect agreement line
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                label='Perfect Agreement')
        
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
    
    # Extract EM vs true log-loss data
    em_true_data = []  # [(true_loss, em_loss, budget)]
    
    budgets = set()
    
    for (n_nodes, combined_policy_name), policy_results in results.items():
        experiment_results = policy_results['results']
        
        for step_result in experiment_results:
            budget = step_result['budget']
            budgets.add(budget)
            
            # Get individual log-loss values (from graph aggregation)
            true_values = step_result.get('true_model_log_loss_values', [])
            em_values = step_result.get('domain_log_loss_values', [])
            
            # Pair up true and EM values
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
        
        # Color by Budget Progression
        budget_list = sorted(budgets)
        budget_colors = cm.viridis(np.linspace(0, 1, len(budget_list)))
        budget_to_color = {budget: budget_colors[i] for i, budget in enumerate(budget_list)}
        
        for i, (true_val, em_val, budget) in enumerate(em_true_data):
            ax.scatter(true_val, em_val, c=[budget_to_color[budget]], 
                      alpha=0.6, s=30)
        
        # Add budget legend
        for i, budget in enumerate(budget_list):
            ax.scatter([], [], c=[budget_colors[i]], label=f'Budget {budget}', s=50)
        
        # Perfect agreement line
        all_true = [x[0] for x in em_true_data]
        all_em = [x[1] for x in em_true_data]
        min_val = min(min(all_true), min(all_em))
        max_val = max(max(all_true), max(all_em))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
               label='Perfect Agreement')
        
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