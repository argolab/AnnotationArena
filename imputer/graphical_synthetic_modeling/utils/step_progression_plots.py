"""
Step-by-step progression visualization for progressive imputation experiments.

Creates multi-panel plots showing how predictions evolve through training steps,
providing clear insight into learning dynamics and convergence patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Color schemes for imputer sizes and progression
IMPUTER_COLORS = {
    'Tiny': '#ff9999',   
    'Small': '#cc4444',  
    'Large': '#990000'   
}


def select_progression_steps(experiment_results: List[Dict], n_steps: int = 5) -> List[int]:
    """
    Select evenly spaced steps from experiment progression.
    
    Args:
        experiment_results: List of step results from experiment
        n_steps: Number of steps to select for visualization
        
    Returns:
        List of indices into experiment_results for selected steps
    """
    if len(experiment_results) <= n_steps:
        return list(range(len(experiment_results)))
    
    # Select evenly spaced indices
    step_indices = np.linspace(0, len(experiment_results) - 1, n_steps, dtype=int)
    return list(step_indices)


def plot_step_progression_scatterplots(results: Dict[str, Any], output_dir: str = "plots",
                                     missing_rate: Optional[float] = None, 
                                     n_steps: int = 5) -> None:
    """
    Create multi-panel step progression scatterplots.
    
    Layout: 2 rows × n_steps columns
    Top row: Neural vs True for each selected step
    Bottom row: EM vs True for each selected step
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
        n_steps: Number of progression steps to show
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group results by node size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results
    
    # Create progression plots for each node size
    for n_nodes, node_results in results_by_nodes.items():
        
        # Extract data from all variants for consistent axis limits
        all_true_vals = []
        all_neural_vals = []
        all_em_vals = []
        progression_data = {}  # {imputer_size: {step_idx: (true_vals, neural_vals, em_vals, budget)}}
        
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            if imputer_size not in progression_data:
                progression_data[imputer_size] = {}
            
            # Select steps for this variant
            step_indices = select_progression_steps(experiment_results, n_steps)
            
            for i, step_idx in enumerate(step_indices):
                step_result = experiment_results[step_idx]
                budget = step_result['budget']
                
                # Get individual sample values
                true_values = step_result.get('true_model_log_loss_values', [])
                neural_values = step_result.get('neural_log_loss_values', [])
                em_values = step_result.get('domain_log_loss_values', [])
                
                # Filter out invalid values
                valid_pairs = []
                for j in range(min(len(true_values), len(neural_values), len(em_values))):
                    tv, nv, ev = true_values[j], neural_values[j], em_values[j]
                    if not any(np.isnan(x) or np.isinf(x) for x in [tv, nv, ev]):
                        valid_pairs.append((tv, nv, ev))
                
                if valid_pairs:
                    true_vals, neural_vals, em_vals = zip(*valid_pairs)
                    progression_data[imputer_size][i] = (true_vals, neural_vals, em_vals, budget)
                    
                    # Collect for global axis limits
                    all_true_vals.extend(true_vals)
                    all_neural_vals.extend(neural_vals)
                    all_em_vals.extend(em_vals)
        
        if not all_true_vals:
            logger.warning(f"No valid data for step progression plots for {n_nodes} nodes")
            continue
        
        # Compute consistent axis limits
        min_val = min(min(all_true_vals), min(all_neural_vals), min(all_em_vals))
        max_val = max(max(all_true_vals), max(all_neural_vals), max(all_em_vals))
        axis_margin = (max_val - min_val) * 0.05
        axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
        
        # Create the multi-panel plot
        fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8))
        if n_steps == 1:
            axes = axes.reshape(2, 1)
        
        # Plot each step
        for step_i in range(n_steps):
            ax_neural = axes[0, step_i]  # Top row: Neural vs True
            ax_em = axes[1, step_i]      # Bottom row: EM vs True
            
            # Plot data from all imputer variants for this step
            step_budget = None
            neural_plotted = False
            em_plotted = False
            
            for imputer_size in ['Tiny', 'Small', 'Large']:
                if imputer_size in progression_data and step_i in progression_data[imputer_size]:
                    true_vals, neural_vals, em_vals, budget = progression_data[imputer_size][step_i]
                    step_budget = budget
                    
                    # Neural vs True (top row)
                    ax_neural.scatter(true_vals, neural_vals, 
                                    color=IMPUTER_COLORS[imputer_size], 
                                    alpha=0.6, s=20, 
                                    label=f'{imputer_size}' if not neural_plotted else "")
                    neural_plotted = True
                    
                    # EM vs True (bottom row) - only plot once per step
                    if not em_plotted:
                        ax_em.scatter(true_vals, em_vals, 
                                    color='#1f77b4', alpha=0.6, s=20,
                                    label='EM' if step_i == 0 else "")
                        em_plotted = True
            
            # Configure neural plot (top row)
            ax_neural.plot([axis_min, axis_max], [axis_min, axis_max], 
                          'k--', alpha=0.7, linewidth=1)
            ax_neural.set_xlim(axis_min, axis_max)
            ax_neural.set_ylim(axis_min, axis_max)
            ax_neural.grid(True, alpha=0.3)
            
            # Configure EM plot (bottom row)  
            ax_em.plot([axis_min, axis_max], [axis_min, axis_max], 
                      'k--', alpha=0.7, linewidth=1)
            ax_em.set_xlim(axis_min, axis_max)
            ax_em.set_ylim(axis_min, axis_max)
            ax_em.grid(True, alpha=0.3)
            
            # Labels and titles
            if step_i == 0:
                ax_neural.set_ylabel('Neural Imputer Log-Loss', fontsize=11)
                ax_em.set_ylabel('EM Model Log-Loss', fontsize=11)
                ax_neural.legend(fontsize=9)
                if em_plotted:
                    ax_em.legend(fontsize=9)
            
            ax_em.set_xlabel('True Model Log-Loss', fontsize=11)
            
            budget_label = f"Budget: {step_budget}" if step_budget else f"Step {step_i+1}"
            ax_neural.set_title(f'{budget_label}', fontsize=11)
        
        # Main title
        fig.suptitle(f'Step Progression Analysis: {n_steps} Steps, {n_nodes} Nodes', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/step_progression_{n_steps}steps_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Step progression plot for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_convergence_progression_summary(results: Dict[str, Any], output_dir: str = "plots",
                                        missing_rate: Optional[float] = None) -> None:
    """
    Create convergence progression summary showing KL divergence evolution.
    
    Single plot with multiple panels showing how KL divergence changes
    across the selected progression steps for different imputer sizes.
    
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
    
    # Create summary for each node size
    for n_nodes, node_results in results_by_nodes.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            imputer_size = policy_results.get('imputer_size', 'Large')
            
            # Extract step progression data
            step_indices = select_progression_steps(experiment_results, 5)
            
            budgets = []
            neural_kls = []
            domain_kls = []
            
            for step_idx in step_indices:
                step_result = experiment_results[step_idx]
                budgets.append(step_result['budget'])
                neural_kls.append(step_result.get('neural_kl', float('inf')))
                domain_kls.append(step_result.get('domain_kl', float('inf')))
            
            # Plot neural imputer progression
            ax.plot(budgets, neural_kls, 'o-', 
                   color=IMPUTER_COLORS[imputer_size], 
                   label=f'Neural ({imputer_size})',
                   linewidth=2, markersize=8, alpha=0.8)
            
            # Plot EM progression (only once)
            if imputer_size == 'Large':  
                ax.plot(budgets, domain_kls, 's--',
                       color='#1f77b4', label='Domain EM',
                       linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Training Budget (Samples)', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'Convergence Progression: {n_nodes} Nodes', fontsize=13)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/convergence_progression_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence progression summary for {n_nodes} nodes saved to {save_path}")
        plt.close()


def plot_step_progression_for_size(results: Dict[str, Any], output_dir: str = "plots",
                                  missing_rate: Optional[float] = None,
                                  target_imputer_size: str = 'Large', n_steps: int = 5) -> None:
    """
    Create step progression plot for a specific imputer size vs EM.
    
    Layout: 2 rows × n_steps columns
    Top row: Imputer vs True for each selected step
    Bottom row: EM vs True for each selected step
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save plots
        missing_rate: Missing rate for filename suffix
        target_imputer_size: Which imputer size to plot (Tiny, Small, Large)
        n_steps: Number of progression steps to show
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group results by node size and filter for target imputer size
    results_by_nodes = {}
    for (n_nodes, combined_policy_name), policy_results in results.items():
        imputer_size = policy_results.get('imputer_size', 'Large')
        if imputer_size != target_imputer_size:
            continue
            
        if n_nodes not in results_by_nodes:
            results_by_nodes[n_nodes] = {}
        results_by_nodes[n_nodes][combined_policy_name] = policy_results
    
    # Create progression plots for each node size
    for n_nodes, node_results in results_by_nodes.items():
        
        # Extract data for this imputer size
        all_true_vals = []
        all_imputer_vals = []
        all_em_vals = []
        progression_data = {}  # {step_idx: (true_vals, imputer_vals, em_vals, budget)}
        
        for combined_policy_name, policy_results in node_results.items():
            experiment_results = policy_results['results']
            
            # Select steps for this variant
            step_indices = select_progression_steps(experiment_results, n_steps)
            
            for i, step_idx in enumerate(step_indices):
                step_result = experiment_results[step_idx]
                budget = step_result['budget']
                
                # Get individual sample values
                true_values = step_result.get('true_model_log_loss_values', [])
                imputer_values = step_result.get('neural_log_loss_values', [])
                em_values = step_result.get('domain_log_loss_values', [])
                
                # Filter out invalid values
                valid_pairs = []
                for j in range(min(len(true_values), len(imputer_values), len(em_values))):
                    tv, iv, ev = true_values[j], imputer_values[j], em_values[j]
                    if not any(np.isnan(x) or np.isinf(x) for x in [tv, iv, ev]):
                        valid_pairs.append((tv, iv, ev))
                
                if valid_pairs:
                    true_vals, imputer_vals, em_vals = zip(*valid_pairs)
                    progression_data[i] = (true_vals, imputer_vals, em_vals, budget)
                    
                    # Collect for global axis limits
                    all_true_vals.extend(true_vals)
                    all_imputer_vals.extend(imputer_vals)
                    all_em_vals.extend(em_vals)
        
        if not all_true_vals:
            logger.warning(f"No valid data for {target_imputer_size} step progression plots for {n_nodes} nodes")
            continue
        
        # Compute consistent axis limits
        min_val = min(min(all_true_vals), min(all_imputer_vals), min(all_em_vals))
        max_val = max(max(all_true_vals), max(all_imputer_vals), max(all_em_vals))
        axis_margin = (max_val - min_val) * 0.05
        axis_min, axis_max = min_val - axis_margin, max_val + axis_margin
        
        # Create the multi-panel plot
        fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8))
        if n_steps == 1:
            axes = axes.reshape(2, 1)
        
        # Plot each step
        for step_i in range(n_steps):
            ax_imputer = axes[0, step_i]  # Top row: Imputer vs True
            ax_em = axes[1, step_i]       # Bottom row: EM vs True
            
            if step_i in progression_data:
                true_vals, imputer_vals, em_vals, budget = progression_data[step_i]
                
                # Imputer vs True (top row) - single color
                ax_imputer.scatter(true_vals, imputer_vals, 
                                 color=IMPUTER_COLORS[target_imputer_size], 
                                 alpha=0.6, s=20, label=target_imputer_size if step_i == 0 else "")
                
                # EM vs True (bottom row) - single color  
                ax_em.scatter(true_vals, em_vals, 
                             color='#1f77b4', alpha=0.6, s=20,
                             label='EM' if step_i == 0 else "")
            
            # Configure imputer plot (top row)
            ax_imputer.plot([axis_min, axis_max], [axis_min, axis_max], 
                           'k--', alpha=0.7, linewidth=1)
            ax_imputer.set_xlim(axis_min, axis_max)
            ax_imputer.set_ylim(axis_min, axis_max)
            ax_imputer.grid(True, alpha=0.3)
            
            # Configure EM plot (bottom row)  
            ax_em.plot([axis_min, axis_max], [axis_min, axis_max], 
                       'k--', alpha=0.7, linewidth=1)
            ax_em.set_xlim(axis_min, axis_max)
            ax_em.set_ylim(axis_min, axis_max)
            ax_em.grid(True, alpha=0.3)
            
            # Labels and titles
            if step_i == 0:
                ax_imputer.set_ylabel(f'{target_imputer_size} Imputer Log-Loss', fontsize=11)
                ax_em.set_ylabel('EM Model Log-Loss', fontsize=11)
                ax_imputer.legend(fontsize=9)
                ax_em.legend(fontsize=9)
            
            ax_em.set_xlabel('True Model Log-Loss', fontsize=11)
            
            budget_label = f"Budget: {budget}" if step_i in progression_data else f"Step {step_i+1}"
            ax_imputer.set_title(f'{budget_label}', fontsize=11)
        
        # Main title
        fig.suptitle(f'{target_imputer_size} vs EM Step Progression: {n_steps} Steps, {n_nodes} Nodes', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        missing_suffix = f"_missing_{missing_rate}" if missing_rate is not None else ""
        save_path = f"{output_dir}/step_progression_{target_imputer_size.lower()}_vs_em_{n_steps}steps_{n_nodes}_nodes{missing_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Step progression plot for {target_imputer_size} vs EM, {n_nodes} nodes saved to {save_path}")
        plt.close()


def create_step_progression_report(results: Dict[str, Any], output_dir: str = "plots",
                                  missing_rate: Optional[float] = None) -> None:
    """
    Create comprehensive step progression report.
    
    Generates multi-panel step progression plots to show learning dynamics 
    across training budgets. Creates separate plots for each imputer size.
    
    Args:
        results: Results dictionary from experiment_runner
        output_dir: Directory to save all plots
        missing_rate: Missing rate for filename suffixes
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Generating step progression analysis report...")
    
    # Generate step progression visualizations for each imputer size separately
    for imputer_size in ['Tiny', 'Small', 'Large']:
        plot_step_progression_for_size(results, output_dir, missing_rate, imputer_size, n_steps=5)
    
    logger.info("Step progression analysis report completed")