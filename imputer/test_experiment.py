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

from neural_imputer import (
    create_model,
    train_model,
    GraphDataset,
    collate_fn,
    evaluate_neural_model,
    DEVICE
)

# Test configuration
TEST_GRAPH_SIZE = [5, 7, 10]
TEST_TRAINING_SIZES = [10, 50, 100, 250, 500, 750, 1000, 1500, 1750, 2000]
TEST_SIZE = 250

# Experimental design parameters
NUM_GRAPHS = 12  # Number of different graph structures per experiment
NUM_TRAINING_SETS = 5  # Number of different training sets per graph structure
FIXED_EDGE_PROB = 0.5  # Fixed edge probability for all graphs
FIXED_MISSING_RATE = 0.4  # Fixed missing rate for all experiments
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

def run_experiment(n_nodes, train_size):
    """Run graph imputation experiment with proper nested experimental design."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {n_nodes} nodes, {train_size} training samples")
    print(f"Design: {NUM_GRAPHS} graphs × {NUM_TRAINING_SETS} training sets per graph")
    print(f"Fixed: edge_prob={FIXED_EDGE_PROB}, missing_rate={FIXED_MISSING_RATE}")
    print(f"{'='*80}")
    
    # Storage for nested results
    graph_results = []  # List of results for each graph
    all_kl_values = []  # For histogram analysis
    failed_experiments = 0
    
    clear_memory()
    
    # Loop over different graph structures
    for graph_idx in range(NUM_GRAPHS):
        print(f"\n{'='*40}")
        print(f"GRAPH {graph_idx + 1}/{NUM_GRAPHS}")
        print(f"{'='*40}")
        
        # Generate graph structure with fixed parameters
        graph_seed = BASE_SEED + graph_idx * 1000  # Large separation for graph seeds
        obs_ratio = 1.0 - FIXED_MISSING_RATE
        
        print(f"Graph seed: {graph_seed}, edge_prob: {FIXED_EDGE_PROB}, missing_rate: {FIXED_MISSING_RATE}")
        
        try:
            # Generate base graph structure
            bn, adj_matrix, _, test_data = create_experiment_data(
                n_nodes, train_size, TEST_SIZE, 
                edge_prob=FIXED_EDGE_PROB, obs_ratio=obs_ratio, seed=graph_seed
            )
            
            if len(test_data) == 0:
                print(f"Failed to generate test data for graph {graph_idx + 1}")
                failed_experiments += 1
                continue
            
            # Storage for this graph's training set results
            graph_neural_results = []
            graph_domain_em_results = []
            graph_domain_complete_results = []
            
            # Loop over different training sets for this graph structure
            for train_set_idx in range(NUM_TRAINING_SETS):
                print(f"\n--- Training Set {train_set_idx + 1}/{NUM_TRAINING_SETS} (Graph {graph_idx + 1}) ---")
                
                # Generate training data with different seed but same graph structure
                train_seed = graph_seed + train_set_idx + 1  # Different training data
                
                try:
                    # Generate training data for this specific training set
                    _, _, incomplete_train_data, _ = create_experiment_data(
                        n_nodes, train_size, TEST_SIZE,
                        edge_prob=FIXED_EDGE_PROB, obs_ratio=obs_ratio, seed=train_seed
                    )
                    
                    # Generate complete training data
                    complete_train_data = create_complete_training_data(bn, adj_matrix, n_nodes, train_size)
                    
                    if len(incomplete_train_data) == 0 or len(complete_train_data) == 0:
                        print(f"Failed to generate training data for training set {train_set_idx + 1}")
                        continue
                    
                    # METHOD 1: Train neural imputer
                    print(f"Training Neural Imputer...")
                    train_dataset = GraphDataset(incomplete_train_data)
                    test_dataset = GraphDataset(test_data)
                    
                    batch_size = min(32, len(incomplete_train_data))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                    
                    input_dim = incomplete_train_data[0][0].shape[1]
                    structure_dim = incomplete_train_data[0][1].shape[1]
                    model = create_model(n_nodes, input_dim, structure_dim)
                    model = train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
                    neural_results = evaluate_neural_model(model, test_data, n_nodes, 2)
                    
                    # METHOD 2: Train domain EM
                    print(f"Training Domain EM...")
                    pyagrum_incomplete_data = convert_training_data_for_pyagrum(incomplete_train_data, n_nodes)
                    learned_bn_em = learn_domain_specific_model(
                        adj_matrix, pyagrum_incomplete_data, n_states=2, max_iter=100, epsilon=1e-3
                    )
                    domain_em_results = evaluate_domain_specific_model(learned_bn_em, test_data, n_nodes, 2)
                    
                    # METHOD 3: Train domain complete
                    print(f"Training Domain Complete...")
                    learned_bn_complete = learn_domain_specific_model_complete(
                        adj_matrix, complete_train_data, n_nodes, n_states=2
                    )
                    domain_complete_results = evaluate_domain_specific_model(learned_bn_complete, test_data, n_nodes, 2)
                    
                    # Store results for this training set
                    neural_kl = neural_results.get('mean_kl', float('inf'))
                    domain_em_kl = domain_em_results.get('mean_kl', float('inf'))
                    domain_complete_kl = domain_complete_results.get('mean_kl', float('inf'))
                    
                    if neural_kl != float('inf') and domain_em_kl != float('inf') and domain_complete_kl != float('inf'):
                        graph_neural_results.append(neural_kl)
                        graph_domain_em_results.append(domain_em_kl)
                        graph_domain_complete_results.append(domain_complete_kl)
                        
                        # Store individual KL values for histograms
                        if 'kl_distribution' in neural_results:
                            all_kl_values.extend([(kl, 'neural', graph_idx, train_set_idx) for kl in neural_results['kl_distribution']])
                        if 'kl_distribution' in domain_em_results:
                            all_kl_values.extend([(kl, 'domain_em', graph_idx, train_set_idx) for kl in domain_em_results['kl_distribution']])
                        if 'kl_distribution' in domain_complete_results:
                            all_kl_values.extend([(kl, 'domain_complete', graph_idx, train_set_idx) for kl in domain_complete_results['kl_distribution']])
                        
                        print(f"Set {train_set_idx + 1}: Neural={neural_kl:.4f}, EM={domain_em_kl:.4f}, Complete={domain_complete_kl:.4f}")
                    else:
                        print(f"Training set {train_set_idx + 1} failed evaluation")
                    
                    clear_memory()
                    
                except Exception as e:
                    print(f"Training set {train_set_idx + 1} FAILED: {e}")
                    clear_memory()
                    continue
            
            # Aggregate results for this graph
            if len(graph_neural_results) > 0:
                graph_result = {
                    'graph_idx': graph_idx,
                    'neural_mean': np.mean(graph_neural_results),
                    'neural_std': np.std(graph_neural_results),
                    'neural_all': graph_neural_results,
                    'domain_em_mean': np.mean(graph_domain_em_results),
                    'domain_em_std': np.std(graph_domain_em_results),
                    'domain_em_all': graph_domain_em_results,
                    'domain_complete_mean': np.mean(graph_domain_complete_results),
                    'domain_complete_std': np.std(graph_domain_complete_results),
                    'domain_complete_all': graph_domain_complete_results,
                    'n_training_sets': len(graph_neural_results)
                }
                graph_results.append(graph_result)
                
                print(f"\nGraph {graph_idx + 1} Summary ({len(graph_neural_results)}/{NUM_TRAINING_SETS} successful):")
                print(f"  Neural: {graph_result['neural_mean']:.4f} ± {graph_result['neural_std']:.4f}")
                print(f"  Domain EM: {graph_result['domain_em_mean']:.4f} ± {graph_result['domain_em_std']:.4f}")
                print(f"  Domain Complete: {graph_result['domain_complete_mean']:.4f} ± {graph_result['domain_complete_std']:.4f}")
            else:
                print(f"Graph {graph_idx + 1} completely failed")
                failed_experiments += 1
        
        except Exception as e:
            print(f"Graph {graph_idx + 1} structure generation FAILED: {e}")
            failed_experiments += 1
            continue
    
    # Final aggregation across all graphs
    if len(graph_results) == 0:
        print("All graphs failed!")
        return None
    
    # Compute overall means and between-graph variation
    all_graph_neural_means = [g['neural_mean'] for g in graph_results]
    all_graph_em_means = [g['domain_em_mean'] for g in graph_results]
    all_graph_complete_means = [g['domain_complete_mean'] for g in graph_results]
    
    # Overall statistics
    overall_neural_mean = np.mean(all_graph_neural_means)
    overall_neural_std = np.std(all_graph_neural_means)  # Between-graph variation
    overall_em_mean = np.mean(all_graph_em_means)
    overall_em_std = np.std(all_graph_em_means)
    overall_complete_mean = np.mean(all_graph_complete_means)
    overall_complete_std = np.std(all_graph_complete_means)
    
    # Average within-graph variation
    avg_within_neural_std = np.mean([g['neural_std'] for g in graph_results])
    avg_within_em_std = np.mean([g['domain_em_std'] for g in graph_results])
    avg_within_complete_std = np.mean([g['domain_complete_std'] for g in graph_results])
    
    results = {
        'config': {
            'n_nodes': n_nodes, 
            'train_size': train_size, 
            'num_graphs': NUM_GRAPHS,
            'num_training_sets': NUM_TRAINING_SETS,
            'edge_prob': FIXED_EDGE_PROB,
            'missing_rate': FIXED_MISSING_RATE
        },
        'overall': {
            'neural_mean': overall_neural_mean,
            'neural_between_std': overall_neural_std,
            'neural_within_std': avg_within_neural_std,
            'domain_em_mean': overall_em_mean,
            'domain_em_between_std': overall_em_std,
            'domain_em_within_std': avg_within_em_std,
            'domain_complete_mean': overall_complete_mean,
            'domain_complete_between_std': overall_complete_std,
            'domain_complete_within_std': avg_within_complete_std
        },
        'graph_results': graph_results,
        'kl_distribution': all_kl_values,
        'successful_graphs': len(graph_results),
        'failed_experiments': failed_experiments
    }
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(graph_results)}/{NUM_GRAPHS} successful graphs)")
    print(f"{'='*60}")
    print(f"Neural Imputer:")
    print(f"  Overall: {overall_neural_mean:.4f} ± {overall_neural_std:.4f} (between-graph)")
    print(f"  Avg within-graph variation: ± {avg_within_neural_std:.4f}")
    print(f"Domain EM:")
    print(f"  Overall: {overall_em_mean:.4f} ± {overall_em_std:.4f} (between-graph)")  
    print(f"  Avg within-graph variation: ± {avg_within_em_std:.4f}")
    print(f"Domain Complete:")
    print(f"  Overall: {overall_complete_mean:.4f} ± {overall_complete_std:.4f} (between-graph)")
    print(f"  Avg within-graph variation: ± {avg_within_complete_std:.4f}")
    
    # Ratios and diagnostics
    em_ratio = overall_neural_mean / overall_em_mean if overall_em_mean > 0 else float('inf')
    complete_ratio = overall_neural_mean / overall_complete_mean if overall_complete_mean > 0 else float('inf')
    
    print(f"\nPerformance Ratios:")
    print(f"  Neural/EM ratio: {em_ratio:.2f}")
    print(f"  Neural/Complete ratio: {complete_ratio:.2f}")
    
    if complete_ratio < em_ratio:
        print("\nDIAGNOSTIC: Complete data model performs better than EM - EM may be getting stuck")
    else:
        print("\nDIAGNOSTIC: EM performs similarly to complete data model - missing data is the main challenge")
    
    return results

def run_test():
    """Run graph imputation comparison test."""
    print("="*60)
    print("GRAPH IMPUTATION COMPARISON TEST")
    print("="*60)
    print(f"Graph sizes: {TEST_GRAPH_SIZE}")
    print(f"Training sizes: {TEST_TRAINING_SIZES}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Using device: {DEVICE}")
    print()
    
    all_results = {}
    
    try:
        for n_nodes in TEST_GRAPH_SIZE:
            for train_size in TEST_TRAINING_SIZES:
                result = run_experiment(n_nodes, train_size)
                
                if result:
                    all_results[(n_nodes, train_size)] = result
                    
                    # Save intermediate results
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
                    with open(os.path.join(base_dir, 'results', 'experiment_results.json'), 'w') as f:
                        json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) 
                                      for k, v in all_results.items()}
                        json.dump(json_results, f, indent=2)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    # Print final summary
    if all_results:
        print("\n" + "="*80)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*80)
        
        # Group results by graph size
        graph_sizes = sorted(set(k[0] for k in all_results.keys()))
        
        for n_nodes in graph_sizes:
            print(f"\n--- {n_nodes} NODES ---")
            print(f"{'Train Size':<12} {'Neural KL':<16} {'Domain EM KL':<16} {'Domain Complete KL':<18} {'EM Ratio':<10} {'Status'}")
            print("-" * 88)
            
            # Get results for this graph size
            node_results = [(k, v) for k, v in all_results.items() if k[0] == n_nodes]
            node_results.sort(key=lambda x: x[0][1])  # Sort by training size
            
            for (_, train_size), result in node_results:
                comparison = result.get('comparison', {})
                neural_kl = comparison.get('neural_kl', float('inf'))
                domain_em_kl = comparison.get('domain_em_kl', float('inf'))
                domain_complete_kl = comparison.get('domain_complete_kl', float('inf'))
                em_ratio = comparison.get('em_ratio', float('inf'))
                neural_std = comparison.get('neural_std', 0.0)
                domain_em_std = comparison.get('domain_em_std', 0.0)
                domain_complete_std = comparison.get('domain_complete_std', 0.0)
                
                # Determine status
                if neural_kl == float('inf') or domain_em_kl == float('inf') or domain_complete_kl == float('inf'):
                    status = "FAILED"
                elif em_ratio < 0.5:
                    status = "EXCELLENT"
                elif em_ratio < 1.0:
                    status = "GOOD"
                else:
                    status = "POOR"
                
                neural_str = f"{neural_kl:.4f}±{neural_std:.3f}"
                domain_em_str = f"{domain_em_kl:.4f}±{domain_em_std:.3f}"
                domain_complete_str = f"{domain_complete_kl:.4f}±{domain_complete_std:.3f}"
                print(f"{train_size:<12} {neural_str:<16} {domain_em_str:<16} {domain_complete_str:<18} {em_ratio:<10.2f} {status}")
        
        print("\nInterpretation:")
        print("- KL Ratio < 0.5: Neural imputer is much better")
        print("- KL Ratio 0.5-1.0: Neural imputer is better") 
        print("- KL Ratio > 1.0: Domain model is better")
        
        # Create plots
        create_comparison_plot(all_results)
        create_kl_histograms(all_results)
        
        print("\nExperiment completed successfully!")
        
    else:
        print("\nNo successful experiments!")
    
    return all_results

def create_comparison_plot(all_results):
    """Create comparison plot for experiment results."""
    if not all_results:
        return
    
    # Get all tested graph sizes
    graph_sizes = sorted(set(k[0] for k in all_results.keys()))
    
    # Create subplots for each graph size
    fig, axes = plt.subplots(1, len(graph_sizes), figsize=(6*len(graph_sizes), 6))
    if len(graph_sizes) == 1:
        axes = [axes]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for idx, n_nodes in enumerate(graph_sizes):
        ax = axes[idx]
        
        # Get training sizes for this graph size
        train_sizes = sorted(set(k[1] for k in all_results.keys() if k[0] == n_nodes))
        neural_kls = []
        domain_em_kls = []
        domain_complete_kls = []
        
        for train_size in train_sizes:
            key = (n_nodes, train_size)
            if key in all_results:
                result = all_results[key]
                comparison = result.get('comparison', {})
                neural_kl = comparison.get('neural_kl', float('inf'))
                domain_em_kl = comparison.get('domain_em_kl', float('inf'))
                domain_complete_kl = comparison.get('domain_complete_kl', float('inf'))
                
                neural_kls.append(neural_kl if neural_kl != float('inf') else np.nan)
                domain_em_kls.append(domain_em_kl if domain_em_kl != float('inf') else np.nan)
                domain_complete_kls.append(domain_complete_kl if domain_complete_kl != float('inf') else np.nan)
        
        # Get error bars
        neural_stds = []
        domain_em_stds = []
        domain_complete_stds = []
        
        for train_size in train_sizes:
            key = (n_nodes, train_size)
            if key in all_results:
                result = all_results[key]
                comparison = result.get('comparison', {})
                neural_std = comparison.get('neural_std', 0.0)
                domain_em_std = comparison.get('domain_em_std', 0.0)
                domain_complete_std = comparison.get('domain_complete_std', 0.0)
                
                neural_stds.append(neural_std if neural_std != 0.0 else 0.0)
                domain_em_stds.append(domain_em_std if domain_em_std != 0.0 else 0.0)
                domain_complete_stds.append(domain_complete_std if domain_complete_std != 0.0 else 0.0)
        
        # Plot with error bars
        ax.errorbar(train_sizes, neural_kls, yerr=neural_stds, fmt='o-', label='Neural Imputer', 
                   linewidth=3, markersize=8, capsize=5, color='blue')
        ax.errorbar(train_sizes, domain_em_kls, yerr=domain_em_stds, fmt='s-', label='Domain EM', 
                   linewidth=3, markersize=8, capsize=5, color='orange')
        ax.errorbar(train_sizes, domain_complete_kls, yerr=domain_complete_stds, fmt='^-', label='Domain Complete', 
                   linewidth=3, markersize=8, capsize=5, color='green')
        ax.set_xlabel('Training Samples', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'KL Divergence Comparison ({n_nodes} nodes)', fontsize=14)
        ax.legend(fontsize=10)
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
            for kl_val, model_type in result['kl_distribution']:
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