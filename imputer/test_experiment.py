"""
Graph Imputation Experiment

Compares neural transformer-based imputer vs domain-specific Bayesian Network
approaches for graph imputation tasks.

Author: Prabhav Singh
"""

import os
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import implementations
from domain_specific_model import (
    learn_domain_specific_model,
    learn_domain_specific_model_complete,
    evaluate_domain_specific_model,
    convert_training_data_for_pyagrum,
    extract_adjacency_from_embeddings
)

from data_generation import create_experiment_data, create_complete_training_data

from neural_imputer_structure import (
    create_model,
    train_model,
    GraphDataset,
    collate_fn,
    evaluate_neural_model,
    DEVICE
)

# Test configuration - Simplified framework
TEST_GRAPH_SIZES = [5, 7, 10]  # Number of nodes
TEST_TRAINING_SIZES = [10, 50, 250, 500, 1000, 1500, 2000]  # Training set sizes
TEST_SIZE = 250  # Test set size

# Fixed parameters
TARGET_PARENTS = 1.0  # Single target parent count
MISSING_RATE = 0.4   # Single missing data rate

# Experimental design parameters
NUM_GRAPHS_PER_CONDITION = 3  # Number of different graph structures per (n_nodes, train_size) combination
BASE_SEED = 42  # Base seed for reproducible experiments

def convert_to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def clear_memory():
    """Clear GPU memory between experiments."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_single_condition_experiment(n_nodes, train_size):
    """Run experiment for a single condition with multiple graphs."""
    print(f"\n{'='*80}")
    print(f"CONDITION: n_nodes={n_nodes}, train_size={train_size}")
    print(f"Fixed: target_parents={TARGET_PARENTS}, missing_rate={MISSING_RATE}")
    print(f"Design: {NUM_GRAPHS_PER_CONDITION} different graphs")
    print(f"{'='*80}")
    
    # Storage for results across graphs
    graph_results = []
    all_kl_values = []
    failed_experiments = 0
    obs_ratio = 1.0 - MISSING_RATE
    
    # Create condition hash for reproducible seeding
    condition_hash = hash((n_nodes, train_size)) % 10000
    
    clear_memory()
    
    # Loop over different graph structures for this condition
    for graph_idx in range(NUM_GRAPHS_PER_CONDITION):
        print(f"\n{'='*40}")
        print(f"GRAPH {graph_idx + 1}/{NUM_GRAPHS_PER_CONDITION}")
        print(f"{'='*40}")
        
        # Generate unique graph structure for this condition and graph index
        graph_seed = BASE_SEED + condition_hash * 1000 + graph_idx * 100
        print(f"Graph seed: {graph_seed}")
        
        try:
            # Generate complete experiment data for this graph
            bn, adj_matrix, train_data, test_data = create_experiment_data(
                n_nodes, train_size, TEST_SIZE,
                target_parents=TARGET_PARENTS, obs_ratio=obs_ratio, seed=graph_seed
            )
            
            # Generate complete training data for domain baseline
            complete_train_data = create_complete_training_data(bn, adj_matrix, n_nodes, train_size)
            
            if len(train_data) == 0 or len(test_data) == 0 or len(complete_train_data) == 0:
                print(f"Failed to generate data for graph {graph_idx + 1}")
                failed_experiments += 1
                continue
            
            print(f"Generated data: {len(train_data)} train, {len(test_data)} test, {len(complete_train_data)} complete")
            
            # METHOD 1: Train neural imputer
            print(f"Training Neural Imputer...")
            train_dataset = GraphDataset(train_data)
            test_dataset = GraphDataset(test_data)
            
            batch_size = min(32, len(train_data))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            input_dim = train_data[0][0].shape[1]
            structure_dim = train_data[0][1].shape[1]
            model = create_model(n_nodes, input_dim, structure_dim)
            model = train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
            neural_results = evaluate_neural_model(model, test_data, n_nodes, 2)
            
            # METHOD 2: Train domain EM
            print(f"Training Domain EM...")
            pyagrum_data = convert_training_data_for_pyagrum(train_data, n_nodes)
            learned_bn_em = learn_domain_specific_model(
                adj_matrix, pyagrum_data, n_states=2, max_iter=100, epsilon=1e-3
            )
            domain_em_results = evaluate_domain_specific_model(learned_bn_em, test_data, n_nodes, 2)
            
            # METHOD 3: Train domain complete
            print(f"Training Domain Complete...")
            learned_bn_complete = learn_domain_specific_model_complete(
                adj_matrix, complete_train_data, n_nodes, n_states=2
            )
            domain_complete_results = evaluate_domain_specific_model(learned_bn_complete, test_data, n_nodes, 2)
            
            # Store results for this graph
            neural_kl = neural_results.get('mean_kl', float('inf'))
            domain_em_kl = domain_em_results.get('mean_kl', float('inf'))
            domain_complete_kl = domain_complete_results.get('mean_kl', float('inf'))
            
            if neural_kl != float('inf') and domain_em_kl != float('inf') and domain_complete_kl != float('inf'):
                graph_result = {
                    'graph_idx': graph_idx,
                    'neural_kl': neural_kl,
                    'domain_em_kl': domain_em_kl,
                    'domain_complete_kl': domain_complete_kl,
                    'status': 'SUCCESS'
                }
                
                # Store individual KL distributions for histograms
                condition_id = f"{n_nodes}_{train_size}"
                if 'kl_distribution' in neural_results:
                    all_kl_values.extend([(kl, 'neural', condition_id, graph_idx, 0) for kl in neural_results['kl_distribution']])
                if 'kl_distribution' in domain_em_results:
                    all_kl_values.extend([(kl, 'domain_em', condition_id, graph_idx, 0) for kl in domain_em_results['kl_distribution']])
                if 'kl_distribution' in domain_complete_results:
                    all_kl_values.extend([(kl, 'domain_complete', condition_id, graph_idx, 0) for kl in domain_complete_results['kl_distribution']])
                
                print(f"Graph {graph_idx + 1}: Neural={neural_kl:.4f}, EM={domain_em_kl:.4f}, Complete={domain_complete_kl:.4f}")
            else:
                graph_result = {
                    'graph_idx': graph_idx,
                    'neural_kl': neural_kl,
                    'domain_em_kl': domain_em_kl,
                    'domain_complete_kl': domain_complete_kl,
                    'status': 'FAILED'
                }
                print(f"Graph {graph_idx + 1} failed evaluation")
            
            graph_results.append(graph_result)
            clear_memory()
            
        except Exception as e:
            print(f"Graph {graph_idx + 1} FAILED: {e}")
            graph_result = {
                'graph_idx': graph_idx,
                'status': 'ERROR',
                'error': str(e)
            }
            graph_results.append(graph_result)
            failed_experiments += 1
            clear_memory()
            continue
        
        except Exception as e:
            print(f"Graph {graph_idx + 1} structure generation FAILED: {e}")
            failed_experiments += 1
            continue
    
    # Final aggregation across all graphs for this condition
    if len(graph_results) == 0:
        print("All graphs failed for this condition!")
        return None
    
    # Filter successful experiments and collect results
    successful_graphs = [g for g in graph_results if g.get('status') == 'SUCCESS']
    
    if len(successful_graphs) == 0:
        print("All graphs failed for this condition!")
        return None
    
    # Compute overall means and between-graph variation
    all_neural_kls = [g['neural_kl'] for g in successful_graphs]
    all_em_kls = [g['domain_em_kl'] for g in successful_graphs]
    all_complete_kls = [g['domain_complete_kl'] for g in successful_graphs]
    
    # Overall statistics
    overall_neural_mean = np.mean(all_neural_kls)
    overall_neural_std = np.std(all_neural_kls)  # Between-graph variation
    overall_em_mean = np.mean(all_em_kls)
    overall_em_std = np.std(all_em_kls)
    overall_complete_mean = np.mean(all_complete_kls)
    overall_complete_std = np.std(all_complete_kls)
    
    results = {
        'config': {
            'target_parents': TARGET_PARENTS,
            'missing_rate': MISSING_RATE,
            'n_nodes': n_nodes, 
            'train_size': train_size, 
            'num_graphs': NUM_GRAPHS_PER_CONDITION
        },
        'overall': {
            'neural_mean': overall_neural_mean,
            'neural_std': overall_neural_std,
            'domain_em_mean': overall_em_mean,
            'domain_em_std': overall_em_std,
            'domain_complete_mean': overall_complete_mean,
            'domain_complete_std': overall_complete_std
        },
        'individual_graphs': successful_graphs,
        'kl_distribution': all_kl_values,
        'successful_graphs': len(successful_graphs),
        'failed_experiments': failed_experiments
    }
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"CONDITION RESULTS ({len(successful_graphs)}/{NUM_GRAPHS_PER_CONDITION} successful graphs)")
    print(f"{'='*60}")
    print(f"Neural Imputer:")
    print(f"  Overall: {overall_neural_mean:.4f} ± {overall_neural_std:.4f}")
    print(f"Domain EM:")
    print(f"  Overall: {overall_em_mean:.4f} ± {overall_em_std:.4f}")  
    print(f"Domain Complete:")
    print(f"  Overall: {overall_complete_mean:.4f} ± {overall_complete_std:.4f}")
    
    # Ratios and diagnostics
    em_ratio = overall_neural_mean / overall_em_mean if overall_em_mean > 0 else float('inf')
    complete_ratio = overall_neural_mean / overall_complete_mean if overall_complete_mean > 0 else float('inf')
    
    print(f"\nPerformance Ratios:")
    print(f"  Neural/EM ratio: {em_ratio:.2f}")
    print(f"  Neural/Complete ratio: {complete_ratio:.2f}")
    
    return results

def run_test():
    """Run graph imputation comparison test."""
    print("="*60)
    print("SIMPLIFIED GRAPH IMPUTATION TEST")
    print("="*60)
    print(f"Graph sizes: {TEST_GRAPH_SIZES}")
    print(f"Training sizes: {TEST_TRAINING_SIZES}")
    print(f"Target parents: {TARGET_PARENTS}")
    print(f"Missing rate: {MISSING_RATE}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Using device: {DEVICE}")
    print(f"Design: {NUM_GRAPHS_PER_CONDITION} graphs per condition")
    
    # Calculate total experiments
    total_conditions = len(TEST_GRAPH_SIZES) * len(TEST_TRAINING_SIZES)
    total_model_trainings = total_conditions * NUM_GRAPHS_PER_CONDITION * 3
    print(f"Total conditions: {total_conditions}")
    print(f"Total model trainings: {total_model_trainings}")
    print()
    
    all_results = {}
    completed_conditions = 0
    
    try:
        # Simplified loops - only n_nodes and train_size
        for n_nodes in TEST_GRAPH_SIZES:
            for train_size in TEST_TRAINING_SIZES:
                print(f"\n{'='*100}")
                print(f"CONDITION {completed_conditions + 1}/{total_conditions}")
                print(f"n_nodes={n_nodes}, train_size={train_size}")
                print(f"{'='*100}")
                
                # Run experiment for this condition
                result = run_single_condition_experiment(n_nodes, train_size)
        
                if result:
                    condition_key = (n_nodes, train_size)
                    all_results[condition_key] = result
                    completed_conditions += 1
                    
                    # Save intermediate results
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
                    with open(os.path.join(base_dir, 'results', 'simplified_results.json'), 'w') as f:
                        json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) 
                                      for k, v in all_results.items()}
                        json.dump(json_results, f, indent=2)
                    
                    print(f"\n✅ Condition {completed_conditions}/{total_conditions} completed successfully")
                else:
                    print(f"\n❌ Condition failed")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Parameter sweep interrupted by user after {completed_conditions} conditions")
    
    # Print comprehensive summary
    if all_results:
        print("\n" + "="*100)
        print("PARAMETER SWEEP SUMMARY")
        print("="*100)
        print(f"Completed {len(all_results)} out of {total_conditions} conditions")
        
        # Create summary table for the simplified framework
        print(f"\n--- TARGET_PARENTS={TARGET_PARENTS}, MISSING_RATE={MISSING_RATE} ---")
        for n_nodes in TEST_GRAPH_SIZES:
            print(f"\n--- N_NODES={n_nodes} ---")
            print(f"{'Train Size':<12} {'Neural KL':<16} {'Domain EM KL':<16} {'Domain Complete KL':<18} {'EM Ratio':<10} {'Status'}")
            print("-" * 88)
            
            # Get results for this graph size
            node_results = [(k, v) for k, v in all_results.items() if k[0] == n_nodes]
            node_results.sort(key=lambda x: x[0][1])  # Sort by training size
            
            for (n_nodes_key, train_size), result in node_results:
                overall = result.get('overall', {})
                neural_mean = overall.get('neural_mean', float('inf'))
                domain_em_mean = overall.get('domain_em_mean', float('inf'))
                domain_complete_mean = overall.get('domain_complete_mean', float('inf'))
                neural_std = overall.get('neural_std', 0.0)
                domain_em_std = overall.get('domain_em_std', 0.0)
                domain_complete_std = overall.get('domain_complete_std', 0.0)
                
                em_ratio = neural_mean / domain_em_mean if domain_em_mean > 0 else float('inf')
                
                # Determine status
                if neural_mean == float('inf') or domain_em_mean == float('inf') or domain_complete_mean == float('inf'):
                    status = "FAILED"
                elif em_ratio < 0.5:
                    status = "EXCELLENT"
                elif em_ratio < 1.0:
                    status = "GOOD"
                else:
                    status = "POOR"
                
                neural_str = f"{neural_mean:.4f}±{neural_std:.3f}"
                domain_em_str = f"{domain_em_mean:.4f}±{domain_em_std:.3f}"
                domain_complete_str = f"{domain_complete_mean:.4f}±{domain_complete_std:.3f}"
                print(f"{train_size:<12} {neural_str:<16} {domain_em_str:<16} {domain_complete_str:<18} {em_ratio:<10.2f} {status}")
        
        print("\nInterpretation:")
        print("- EM Ratio < 0.5: Neural imputer is much better than Domain EM")
        print("- EM Ratio 0.5-1.0: Neural imputer is better than Domain EM") 
        print("- EM Ratio > 1.0: Domain EM is better than Neural imputer")
        
        # Create plots for the parameter sweep
        print("\nGenerating plots...")
        create_comparison_plot(all_results)
        create_kl_histograms(all_results)
        
        print("\nParameter sweep completed successfully!")
        
    else:
        print("\nNo successful experiments!")
    
    return all_results

def create_comparison_plot(all_results):
    """Create comparison plot for experiment results."""
    if not all_results:
        return
    
    # Create subplots for each graph size
    fig, axes = plt.subplots(1, len(TEST_GRAPH_SIZES), figsize=(8*len(TEST_GRAPH_SIZES), 6))
    if len(TEST_GRAPH_SIZES) == 1:
        axes = [axes]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for node_idx, n_nodes in enumerate(TEST_GRAPH_SIZES):
        ax = axes[node_idx]
        
        # Get training sizes for this graph size
        train_sizes = sorted(set(k[1] for k in all_results.keys() if k[0] == n_nodes))
        neural_kls = []
        domain_em_kls = []
        domain_complete_kls = []
        neural_stds = []
        domain_em_stds = []
        domain_complete_stds = []
        
        for train_size in train_sizes:
            key = (n_nodes, train_size)
            if key in all_results:
                result = all_results[key]
                overall = result.get('overall', {})
                
                neural_mean = overall.get('neural_mean', float('inf'))
                domain_em_mean = overall.get('domain_em_mean', float('inf'))
                domain_complete_mean = overall.get('domain_complete_mean', float('inf'))
                neural_std = overall.get('neural_std', 0.0)
                domain_em_std = overall.get('domain_em_std', 0.0)
                domain_complete_std = overall.get('domain_complete_std', 0.0)
                
                neural_kls.append(neural_mean if neural_mean != float('inf') else np.nan)
                domain_em_kls.append(domain_em_mean if domain_em_mean != float('inf') else np.nan)
                domain_complete_kls.append(domain_complete_mean if domain_complete_mean != float('inf') else np.nan)
                neural_stds.append(neural_std)
                domain_em_stds.append(domain_em_std)
                domain_complete_stds.append(domain_complete_std)
        
        # Plot with error bars
        if len(train_sizes) > 0:
            ax.errorbar(train_sizes, neural_kls, yerr=neural_stds, fmt='o-', label='Neural Imputer', 
                       linewidth=2, markersize=6, capsize=4, color='blue')
            ax.errorbar(train_sizes, domain_em_kls, yerr=domain_em_stds, fmt='s-', label='Domain EM', 
                       linewidth=2, markersize=6, capsize=4, color='orange')
            ax.errorbar(train_sizes, domain_complete_kls, yerr=domain_complete_stds, fmt='^-', label='Domain Complete', 
                       linewidth=2, markersize=6, capsize=4, color='green')
        
        ax.set_xlabel('Training Samples', fontsize=10)
        ax.set_ylabel('KL Divergence', fontsize=10)
        ax.set_title(f'parents={TARGET_PARENTS}, miss={MISSING_RATE}, nodes={n_nodes}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results', 'experiment_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {os.path.join(base_dir, 'results', 'experiment_plot.png')}")


def create_kl_histograms(all_results):
    """Create KL distribution histograms for detailed analysis."""
    if not all_results:
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collect all KL distributions
    all_neural_kls = []
    all_domain_em_kls = []
    all_domain_complete_kls = []
    
    for result in all_results.values():
        if 'kl_distribution' in result:
            for kl_data in result['kl_distribution']:
                if len(kl_data) >= 2:  # Handle both old and new format
                    kl_val, model_type = kl_data[0], kl_data[1]
                    if not np.isnan(kl_val) and not np.isinf(kl_val):
                        if model_type == 'neural':
                            all_neural_kls.append(kl_val)
                        elif model_type == 'domain_em':
                            all_domain_em_kls.append(kl_val)
                        elif model_type == 'domain_complete':
                            all_domain_complete_kls.append(kl_val)
    
    if len(all_neural_kls) == 0 and len(all_domain_em_kls) == 0 and len(all_domain_complete_kls) == 0:
        print("No KL distribution data available for histograms")
        return
    
    # Create histogram plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Neural model histogram
    if len(all_neural_kls) > 0:
        ax1.hist(all_neural_kls, bins=50, alpha=0.7, color='blue', density=False)
        ax1.set_xlabel('KL Divergence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Neural Imputer KL Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(all_neural_kls)
        median_kl = np.median(all_neural_kls)
        ax1.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.3f}')
        ax1.axvline(median_kl, color='green', linestyle='--', label=f'Median: {median_kl:.3f}')
        ax1.legend()
    
    # Domain EM model histogram
    if len(all_domain_em_kls) > 0:
        ax2.hist(all_domain_em_kls, bins=50, alpha=0.7, color='orange', density=False)
        ax2.set_xlabel('KL Divergence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Domain EM KL Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(all_domain_em_kls)
        median_kl = np.median(all_domain_em_kls)
        ax2.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.3f}')
        ax2.axvline(median_kl, color='green', linestyle='--', label=f'Median: {median_kl:.3f}')
        ax2.legend()
    
    # Domain Complete model histogram
    if len(all_domain_complete_kls) > 0:
        ax3.hist(all_domain_complete_kls, bins=50, alpha=0.7, color='green', density=False)
        ax3.set_xlabel('KL Divergence', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Domain Complete KL Distribution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(all_domain_complete_kls)
        median_kl = np.median(all_domain_complete_kls)
        ax3.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.3f}')
        ax3.axvline(median_kl, color='green', linestyle='--', label=f'Median: {median_kl:.3f}')
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results', 'kl_histograms.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"KL histograms saved: {os.path.join(base_dir, 'results', 'kl_histograms.png')}")
    
    # Print summary statistics
    if len(all_neural_kls) > 0:
        print(f"\nNeural KL Statistics:")
        print(f"  Mean: {np.mean(all_neural_kls):.4f}")
        print(f"  Median: {np.median(all_neural_kls):.4f}")
        print(f"  Std: {np.std(all_neural_kls):.4f}")
        print(f"  Min: {np.min(all_neural_kls):.4f}")
        print(f"  Max: {np.max(all_neural_kls):.4f}")
        print(f"  Count: {len(all_neural_kls)}")
    
    if len(all_domain_em_kls) > 0:
        print(f"\nDomain EM KL Statistics:")
        print(f"  Mean: {np.mean(all_domain_em_kls):.4f}")
        print(f"  Median: {np.median(all_domain_em_kls):.4f}")
        print(f"  Std: {np.std(all_domain_em_kls):.4f}")
        print(f"  Min: {np.min(all_domain_em_kls):.4f}")
        print(f"  Max: {np.max(all_domain_em_kls):.4f}")
        print(f"  Count: {len(all_domain_em_kls)}")
    
    if len(all_domain_complete_kls) > 0:
        print(f"\nDomain Complete KL Statistics:")
        print(f"  Mean: {np.mean(all_domain_complete_kls):.4f}")
        print(f"  Median: {np.median(all_domain_complete_kls):.4f}")
        print(f"  Std: {np.std(all_domain_complete_kls):.4f}")
        print(f"  Min: {np.min(all_domain_complete_kls):.4f}")
        print(f"  Max: {np.max(all_domain_complete_kls):.4f}")
        print(f"  Count: {len(all_domain_complete_kls)}")

if __name__ == "__main__":
    print("Starting graph imputation experiment...")
    
    # Run the test
    results = run_test()
    
    print("\nExperiment finished!")