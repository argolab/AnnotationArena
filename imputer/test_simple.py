"""
Simple graph imputation experiment framework.

This module provides a simplified testing interface for quick experiments
with direct hyperparameter specification.

Author: Prabhav Singh
"""

import os
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import domain specific model
from domain_specific_model import (
    learn_domain_specific_model,
    learn_domain_specific_model_complete,
    evaluate_domain_specific_model,
    convert_training_data_for_pyagrum
)

# Import data generation
from data_generation import create_experiment_data, create_complete_training_data

# Import both neural imputer versions
from neural_imputer_structure import (
    create_model as create_model_structure,
    train_model as train_model_structure,
    GraphDataset as GraphDatasetStructure,
    collate_fn as collate_fn_structure,
    evaluate_neural_model as evaluate_neural_model_structure,
    DEVICE
)

from neural_imputer_cpts import (
    create_model_cpts,
    train_model_cpts,
    GraphDatasetWithCPTs,
    collate_fn_cpts,
    evaluate_neural_model_cpts,
)

def clear_memory():
    """Clear GPU memory between experiments."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_simple_experiment(n_nodes=5, train_size=500, target_parents=1.5, missing_rate=0.4, 
                         neural_type="structure", seed=42):
    """
    Run simple graph imputation experiment with specified hyperparameters.
    Can accept single values or lists for n_nodes and train_size to run multiple experiments.
    
    Args:
        n_nodes: Number of nodes in the graph (int or list)
        train_size: Number of training samples (int or list)
        target_parents: Target number of parents per node (O(1) parents)
        missing_rate: Fraction of nodes that are missing
        neural_type: "structure" or "cpts" - which neural imputer to use
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with results (single experiment) or list of results (multiple experiments)
    """
    # Handle single values vs lists
    if isinstance(n_nodes, (int, float)):
        n_nodes = [n_nodes]
    if isinstance(train_size, (int, float)):
        train_size = [train_size]
    
    all_experiments = []
    total_experiments = len(n_nodes) * len(train_size)
    experiment_count = 0
    
    print(f"\n{'='*80}")
    print(f"SIMPLE EXPERIMENT BATCH")
    print(f"Node counts: {n_nodes}")
    print(f"Train sizes: {train_size}")
    print(f"Target Parents: {target_parents}, Missing Rate: {missing_rate}")
    print(f"Neural Type: {neural_type}, Total experiments: {total_experiments}")
    print(f"{'='*80}")
    
    obs_ratio = 1.0 - missing_rate
    test_size = 250
    
    clear_memory()
    
    # Run all combinations
    for n_node in n_nodes:
        for train_sz in train_size:
            experiment_count += 1
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {experiment_count}/{total_experiments}")
            print(f"Nodes: {n_node}, Train Size: {train_sz}")
            print(f"{'='*60}")
            
            # Use different seed for each experiment
            exp_seed = seed + experiment_count * 1000
            
            try:
                # Generate data
                bn, adj_matrix, train_data, test_data = create_experiment_data(
                    n_node, train_sz, test_size, target_parents=target_parents, 
                    obs_ratio=obs_ratio, seed=exp_seed
                )
                
                # Generate complete training data for domain baseline
                complete_train_data = create_complete_training_data(bn, adj_matrix, n_node, train_sz)
                
                if len(train_data) == 0 or len(test_data) == 0 or len(complete_train_data) == 0:
                    print(f"Failed to generate data for experiment {experiment_count}")
                    continue
                
                # METHOD 1: Train neural imputer
                print(f"Training Neural Imputer ({neural_type})...")
                
                if neural_type == "structure":
                    # Structure-only neural model
                    train_dataset = GraphDatasetStructure(train_data)
                    test_dataset = GraphDatasetStructure(test_data)
                    
                    batch_size = min(32, len(train_data))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_structure)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_structure)
                    
                    input_dim = train_data[0][0].shape[1]
                    structure_dim = train_data[0][1].shape[1]
                    model = create_model_structure(n_node, input_dim, structure_dim)
                    
                    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
                    model = train_model_structure(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
                    
                    # Evaluate
                    neural_results = evaluate_neural_model_structure(model, test_data, n_node, 2)
                    
                elif neural_type == "cpts":
                    # Structure + CPTs neural model
                    train_dataset = GraphDatasetWithCPTs(train_data, bn)
                    test_dataset = GraphDatasetWithCPTs(test_data, bn)
                    
                    batch_size = min(32, len(train_data))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cpts)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cpts)
                    
                    input_dim = train_data[0][0].shape[1]
                    structure_dim = train_data[0][1].shape[1]
                    cpt_dim = train_dataset.max_cpt_size  # Get actual CPT dimension from dataset
                    print(f"Using actual CPT dimension: {cpt_dim}")
                    model = create_model_cpts(n_node, input_dim, structure_dim, cpt_dim)
                    
                    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
                    model = train_model_cpts(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
                    
                    # Evaluate
                    neural_results = evaluate_neural_model_cpts(model, test_data, bn, n_node, 2)
                
                else:
                    raise ValueError(f"Unknown neural_type: {neural_type}")
            
                # METHOD 2: Train domain-specific model with EM
                print(f"Training Domain EM...")
                pyagrum_incomplete_data = convert_training_data_for_pyagrum(train_data, n_node)
                learned_bn_em = learn_domain_specific_model(
                    adj_matrix, pyagrum_incomplete_data, n_states=2, max_iter=100, epsilon=1e-3
                )
                domain_em_results = evaluate_domain_specific_model(learned_bn_em, test_data, n_node, 2)
                
                # METHOD 3: Train domain-specific model on complete data
                print(f"Training Domain Complete...")
                learned_bn_complete = learn_domain_specific_model_complete(
                    adj_matrix, complete_train_data, n_node, n_states=2
                )
                domain_complete_results = evaluate_domain_specific_model(learned_bn_complete, test_data, n_node, 2)
            
                # Store results
                neural_kl = neural_results.get('mean_kl', float('inf'))
                domain_em_kl = domain_em_results.get('mean_kl', float('inf'))
                domain_complete_kl = domain_complete_results.get('mean_kl', float('inf'))
                
                if neural_kl != float('inf') and domain_em_kl != float('inf') and domain_complete_kl != float('inf'):
                    em_ratio = neural_kl / domain_em_kl if domain_em_kl > 0 else float('inf')
                    complete_ratio = neural_kl / domain_complete_kl if domain_complete_kl > 0 else float('inf')
                    
                    experiment_result = {
                        'config': {
                            'n_nodes': n_node,
                            'train_size': train_sz,
                            'target_parents': target_parents,
                            'missing_rate': missing_rate,
                            'neural_type': neural_type
                        },
                        'neural_kl': neural_kl,
                        'domain_em_kl': domain_em_kl,
                        'domain_complete_kl': domain_complete_kl,
                        'em_ratio': em_ratio,
                        'complete_ratio': complete_ratio,
                        'status': 'SUCCESS'
                    }
                    
                    print(f"Experiment {experiment_count} Results:")
                    print(f"  Neural: {neural_kl:.4f}")
                    print(f"  Domain EM: {domain_em_kl:.4f} (ratio = {em_ratio:.2f})")
                    print(f"  Domain Complete: {domain_complete_kl:.4f} (ratio = {complete_ratio:.2f})")
                    
                    # Determine status
                    if em_ratio < 0.5:
                        status = "EXCELLENT: Neural much better than Domain EM"
                    elif em_ratio < 1.0:
                        status = "GOOD: Neural better than Domain EM"
                    else:
                        status = "POOR: Domain EM better than Neural"
                    
                    print(f"  Status: {status}")
                    
                else:
                    experiment_result = {
                        'config': {
                            'n_nodes': n_node,
                            'train_size': train_sz,
                            'target_parents': target_parents,
                            'missing_rate': missing_rate,
                            'neural_type': neural_type
                        },
                        'neural_kl': neural_kl,
                        'domain_em_kl': domain_em_kl,
                        'domain_complete_kl': domain_complete_kl,
                        'status': 'FAILED'
                    }
                    print(f"Experiment {experiment_count} failed evaluation")
                
                all_experiments.append(experiment_result)
                clear_memory()
                
            except Exception as e:
                print(f"Experiment {experiment_count} FAILED: {e}")
                experiment_result = {
                    'config': {
                        'n_nodes': n_node,
                        'train_size': train_sz,
                        'target_parents': target_parents,
                        'missing_rate': missing_rate,
                        'neural_type': neural_type
                    },
                    'status': 'ERROR',
                    'error': str(e)
                }
                all_experiments.append(experiment_result)
                clear_memory()
                continue
    
    # Print summary
    successful_experiments = [exp for exp in all_experiments if exp['status'] == 'SUCCESS']
    failed_experiments = [exp for exp in all_experiments if exp['status'] != 'SUCCESS']
    
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n{'='*60}")
        print(f"SUCCESSFUL EXPERIMENTS")
        print(f"{'='*60}")
        print(f"{'Nodes':<6} {'Train Size':<10} {'Neural KL':<12} {'EM KL':<12} {'Complete KL':<12} {'EM Ratio':<10} {'Status'}")
        print("-" * 80)
        
        for exp in successful_experiments:
            config = exp['config']
            print(f"{config['n_nodes']:<6} {config['train_size']:<10} {exp['neural_kl']:<12.4f} "
                  f"{exp['domain_em_kl']:<12.4f} {exp['domain_complete_kl']:<12.4f} "
                  f"{exp['em_ratio']:<10.2f} {'GOOD' if exp['em_ratio'] < 1.0 else 'POOR'}")
    
    if failed_experiments:
        print(f"\n{'='*60}")
        print(f"FAILED EXPERIMENTS")
        print(f"{'='*60}")
        for exp in failed_experiments:
            config = exp['config']
            error_msg = exp.get('error', 'Evaluation failed')
            print(f"Nodes: {config['n_nodes']}, Train Size: {config['train_size']} - {error_msg}")
    
    # Return appropriate result
    if len(all_experiments) == 1:
        return all_experiments[0]  # Single experiment
    else:
        return all_experiments  # Multiple experiments

def compare_neural_types(n_nodes=5, train_size=500, target_parents=1.5, missing_rate=0.4, num_trials=3, seed=42):
    """Compare structure-only vs structure+CPTs neural imputers."""
    print(f"\n{'='*80}")
    print(f"NEURAL TYPE COMPARISON")
    print(f"{'='*80}")
    
    results_structure = run_simple_experiment(
        n_nodes, train_size, target_parents, missing_rate, 
        neural_type="structure", num_trials=num_trials, seed=seed
    )
    
    results_cpts = run_simple_experiment(
        n_nodes, train_size, target_parents, missing_rate, 
        neural_type="cpts", num_trials=num_trials, seed=seed + 10000
    )
    
    if results_structure and results_cpts:
        print(f"\n{'='*60}")
        print(f"NEURAL TYPE COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        structure_kl = results_structure['neural']['mean_kl']
        structure_std = results_structure['neural']['std_kl']
        cpts_kl = results_cpts['neural']['mean_kl']
        cpts_std = results_cpts['neural']['std_kl']
        
        print(f"Structure-only: {structure_kl:.4f} ± {structure_std:.4f}")
        print(f"Structure+CPTs: {cpts_kl:.4f} ± {cpts_std:.4f}")
        
        improvement_ratio = structure_kl / cpts_kl if cpts_kl > 0 else float('inf')
        print(f"Improvement ratio: {improvement_ratio:.2f}")
        
        if improvement_ratio > 1.1:
            print("CPTs provide significant improvement")
        elif improvement_ratio > 1.05:
            print("CPTs provide modest improvement")
        elif improvement_ratio < 0.95:
            print("CPTs hurt performance")
        else:
            print("CPTs have minimal impact")
    
    return results_structure, results_cpts

def run_scaling_test(neural_type="structure", num_trials=3):
    """Test performance across different graph sizes."""
    print(f"\n{'='*80}")
    print(f"SCALING TEST - Neural Type: {neural_type}")
    print(f"{'='*80}")
    
    graph_sizes = [5, 7, 10]
    train_sizes = [100, 500, 1000]
    
    all_results = {}
    
    for n_nodes in graph_sizes:
        for train_size in train_sizes:
            print(f"\n--- Testing {n_nodes} nodes, {train_size} train size ---")
            
            result = run_simple_experiment(
                n_nodes=n_nodes,
                train_size=train_size,
                target_parents=1.5,
                missing_rate=0.4,
                neural_type=neural_type,
                num_trials=num_trials,
                seed=42
            )
            
            if result:
                all_results[(n_nodes, train_size)] = result
                
                # Save intermediate results
                base_dir = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
                with open(os.path.join(base_dir, 'results', f'scaling_test_{neural_type}.json'), 'w') as f:
                    json_results = {f"{k[0]}_{k[1]}": result for k, result in all_results.items()}
                    json.dump(json_results, f, indent=2)
    
    # Create summary plot
    if all_results:
        create_scaling_plot(all_results, neural_type)
    
    return all_results

def create_scaling_plot(results, neural_type):
    """Create plot showing scaling behavior."""
    if not results:
        return
    
    graph_sizes = sorted(set(k[0] for k in results.keys()))
    
    fig, axes = plt.subplots(1, len(graph_sizes), figsize=(6*len(graph_sizes), 6))
    if len(graph_sizes) == 1:
        axes = [axes]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for idx, n_nodes in enumerate(graph_sizes):
        ax = axes[idx]
        
        # Get data for this graph size
        train_sizes = sorted(set(k[1] for k in results.keys() if k[0] == n_nodes))
        neural_means = []
        neural_stds = []
        domain_em_means = []
        domain_em_stds = []
        domain_complete_means = []
        domain_complete_stds = []
        
        for train_size in train_sizes:
            key = (n_nodes, train_size)
            if key in results:
                result = results[key]
                neural_means.append(result['neural']['mean_kl'])
                neural_stds.append(result['neural']['std_kl'])
                domain_em_means.append(result['domain_em']['mean_kl'])
                domain_em_stds.append(result['domain_em']['std_kl'])
                domain_complete_means.append(result['domain_complete']['mean_kl'])
                domain_complete_stds.append(result['domain_complete']['std_kl'])
        
        # Plot
        ax.errorbar(train_sizes, neural_means, yerr=neural_stds, fmt='o-', 
                   label=f'Neural ({neural_type})', linewidth=2, markersize=6, capsize=4, color='blue')
        ax.errorbar(train_sizes, domain_em_means, yerr=domain_em_stds, fmt='s-', 
                   label='Domain EM', linewidth=2, markersize=6, capsize=4, color='orange')
        ax.errorbar(train_sizes, domain_complete_means, yerr=domain_complete_stds, fmt='^-', 
                   label='Domain Complete', linewidth=2, markersize=6, capsize=4, color='green')
        
        ax.set_xlabel('Training Samples', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'{n_nodes} nodes', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results', f'scaling_plot_{neural_type}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Scaling plot saved: {os.path.join(base_dir, 'results', f'scaling_plot_{neural_type}.png')}")

if __name__ == "__main__":
    print("Simple Graph Imputation Testing")
    
    # Example usage - small batch for quick testing
    result = run_simple_experiment(
        n_nodes=[5, 7],
        train_size=[50, 200, 500],
        target_parents=1.0,
        missing_rate=0.4,
        neural_type="structure"
    )
    
    print("\nExperiment completed!")