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
NUM_GRAPHS = 10  # Number of random graphs per experiment
EDGE_PROBS = [0.3, 0.5, 0.7, 0.9]  # Different edge probabilities to sample from
MISSING_RATES = [0.4, 0.6, 0.3]  # Different missing rates to sample from
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
    """Run graph imputation experiment across multiple random graphs."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {n_nodes} nodes, {train_size} training samples, {NUM_GRAPHS} graphs")
    print(f"{'='*60}")
    
    all_neural_results = []
    all_domain_em_results = []
    all_domain_complete_results = []
    all_kl_values = []  # For histogram analysis
    failed_graphs = 0
    
    clear_memory()
    
    for graph_idx in range(NUM_GRAPHS):
        print(f"\n--- Graph {graph_idx + 1}/{NUM_GRAPHS} ---")
        
        try:
            # Fair seeding strategy
            graph_seed = BASE_SEED + graph_idx
            
            # Sample random experimental parameters
            np.random.seed(graph_seed)  # Ensure reproducible parameter sampling
            edge_prob = np.random.choice(EDGE_PROBS)
            obs_ratio = 1.0 - np.random.choice(MISSING_RATES)  # Convert missing rate to obs ratio
            
            print(f"Graph {graph_idx + 1}: seed={graph_seed}, edge_prob={edge_prob}, obs_ratio={obs_ratio:.1f}")
            
            # Generate incomplete data using graph structure seed
            bn, adj_matrix, incomplete_train_data, test_data = create_experiment_data(
                n_nodes, train_size, TEST_SIZE, edge_prob=edge_prob, obs_ratio=obs_ratio, seed=graph_seed
            )
            
            # Generate complete training data for domain baseline
            complete_train_data = create_complete_training_data(bn, adj_matrix, n_nodes, train_size)
        
            if len(incomplete_train_data) == 0 or len(test_data) == 0 or len(complete_train_data) == 0:
                print(f"Failed to generate data for graph {graph_idx + 1}")
                failed_graphs += 1
                continue
            
            # METHOD 1: Train neural imputer on incomplete data
            print(f"\n=== TRAINING NEURAL IMPUTER (Graph {graph_idx + 1}) ===")
            train_dataset = GraphDataset(incomplete_train_data)
            test_dataset = GraphDataset(test_data)
            
            batch_size = min(32, len(incomplete_train_data))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            input_dim = incomplete_train_data[0][0].shape[1]
            structure_dim = incomplete_train_data[0][1].shape[1]
            model = create_model(n_nodes, input_dim, structure_dim)
            
            print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
            model = train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
            
            # Evaluate neural imputer
            print(f"\n=== EVALUATING NEURAL IMPUTER (Graph {graph_idx + 1}) ===")
            neural_results = evaluate_neural_model(model, test_data, n_nodes, 2)
            
            # METHOD 2: Train domain-specific model with EM on incomplete data
            print(f"\n=== TRAINING DOMAIN-SPECIFIC MODEL EM (Graph {graph_idx + 1}) ===")
            
            # Convert incomplete training data to pyAgrum format
            pyagrum_incomplete_data = convert_training_data_for_pyagrum(incomplete_train_data, n_nodes)
            
            # Learn with pyAgrum EM
            learned_bn_em = learn_domain_specific_model(
                adj_matrix, pyagrum_incomplete_data, n_states=2, max_iter=100, epsilon=1e-3
            )
            
            # Evaluate domain-specific EM model
            print(f"\n=== EVALUATING DOMAIN-SPECIFIC MODEL EM (Graph {graph_idx + 1}) ===")
            domain_em_results = evaluate_domain_specific_model(learned_bn_em, test_data, n_nodes, 2)
            
            # METHOD 3: Train domain-specific model on complete data
            print(f"\n=== TRAINING DOMAIN-SPECIFIC MODEL COMPLETE (Graph {graph_idx + 1}) ===")
            
            # Learn from complete data (no EM needed)
            learned_bn_complete = learn_domain_specific_model_complete(
                adj_matrix, complete_train_data, n_nodes, n_states=2
            )
            
            # Evaluate domain-specific complete model
            print(f"\n=== EVALUATING DOMAIN-SPECIFIC MODEL COMPLETE (Graph {graph_idx + 1}) ===")
            domain_complete_results = evaluate_domain_specific_model(learned_bn_complete, test_data, n_nodes, 2)
        
            # Store results for this graph
            neural_kl = neural_results.get('mean_kl', float('inf'))
            domain_em_kl = domain_em_results.get('mean_kl', float('inf'))
            domain_complete_kl = domain_complete_results.get('mean_kl', float('inf'))
            
            if neural_kl != float('inf') and domain_em_kl != float('inf') and domain_complete_kl != float('inf'):
                all_neural_results.append(neural_kl)
                all_domain_em_results.append(domain_em_kl)
                all_domain_complete_results.append(domain_complete_kl)
                
                # Store individual KL values for histogram
                if 'kl_distribution' in neural_results:
                    all_kl_values.extend([(kl, 'neural') for kl in neural_results['kl_distribution']])
                if 'kl_distribution' in domain_em_results:
                    all_kl_values.extend([(kl, 'domain_em') for kl in domain_em_results['kl_distribution']])
                if 'kl_distribution' in domain_complete_results:
                    all_kl_values.extend([(kl, 'domain_complete') for kl in domain_complete_results['kl_distribution']])
                
                em_ratio = neural_kl / domain_em_kl if domain_em_kl > 0 else float('inf')
                complete_ratio = neural_kl / domain_complete_kl if domain_complete_kl > 0 else float('inf')
                print(f"\nGraph {graph_idx + 1} Results:")
                print(f"  Neural KL = {neural_kl:.4f}")
                print(f"  Domain EM KL = {domain_em_kl:.4f} (ratio = {em_ratio:.2f})")
                print(f"  Domain Complete KL = {domain_complete_kl:.4f} (ratio = {complete_ratio:.2f})")
            else:
                failed_graphs += 1
                print(f"Graph {graph_idx + 1} failed evaluation")
            
            clear_memory()
            
        except Exception as e:
            print(f"Graph {graph_idx + 1} FAILED: {e}")
            failed_graphs += 1
            clear_memory()
            continue
    
    # Aggregate results across all graphs
    if len(all_neural_results) == 0:
        print("All graphs failed!")
        return None
    
    neural_mean = np.mean(all_neural_results)
    neural_std = np.std(all_neural_results)
    domain_em_mean = np.mean(all_domain_em_results)
    domain_em_std = np.std(all_domain_em_results)
    domain_complete_mean = np.mean(all_domain_complete_results)
    domain_complete_std = np.std(all_domain_complete_results)
    
    em_ratio = neural_mean / domain_em_mean if domain_em_mean > 0 else float('inf')
    complete_ratio = neural_mean / domain_complete_mean if domain_complete_mean > 0 else float('inf')
    
    results = {
        'config': {'n_nodes': n_nodes, 'train_size': train_size, 'num_graphs': NUM_GRAPHS},
        'neural': {
            'mean_kl': neural_mean,
            'std_kl': neural_std,
            'all_kl': all_neural_results
        },
        'domain_em': {
            'mean_kl': domain_em_mean,
            'std_kl': domain_em_std,
            'all_kl': all_domain_em_results
        },
        'domain_complete': {
            'mean_kl': domain_complete_mean,
            'std_kl': domain_complete_std,
            'all_kl': all_domain_complete_results
        },
        'comparison': {
            'neural_kl': neural_mean,
            'domain_em_kl': domain_em_mean,
            'domain_complete_kl': domain_complete_mean,
            'em_ratio': em_ratio,
            'complete_ratio': complete_ratio,
            'neural_std': neural_std,
            'domain_em_std': domain_em_std,
            'domain_complete_std': domain_complete_std
        },
        'kl_distribution': all_kl_values,
        'failed_graphs': failed_graphs
    }
    
    print(f"\n=== AGGREGATED RESULTS ({NUM_GRAPHS - failed_graphs}/{NUM_GRAPHS} successful graphs) ===")
    print(f"Neural KL: {neural_mean:.4f} ± {neural_std:.4f}")
    print(f"Domain EM KL: {domain_em_mean:.4f} ± {domain_em_std:.4f} (ratio = {em_ratio:.2f})")
    print(f"Domain Complete KL: {domain_complete_mean:.4f} ± {domain_complete_std:.4f} (ratio = {complete_ratio:.2f})")
    
    # Status based on EM comparison (primary comparison)
    if em_ratio < 0.5:
        status = "EXCELLENT: Neural imputer much better than Domain EM"
    elif em_ratio < 1.0:
        status = "GOOD: Neural imputer better than Domain EM"
    else:
        status = "NEEDS WORK: Domain EM better than Neural"
    
    print(f"\nPrimary Status: {status}")
    
    # Additional diagnostic info
    if complete_ratio < em_ratio:
        print("DIAGNOSTIC: Complete data model performs better than EM - EM may be getting stuck")
    else:
        print("DIAGNOSTIC: EM performs similarly to complete data model - missing data is the main challenge")
    
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
        
        # Add text box with summary for this graph size
        if len(neural_kls) > 0 and len(domain_em_kls) > 0 and len(domain_complete_kls) > 0:
            neural_mean = np.nanmean(neural_kls)
            domain_em_mean = np.nanmean(domain_em_kls)
            domain_complete_mean = np.nanmean(domain_complete_kls)
            em_ratio = neural_mean / domain_em_mean if domain_em_mean > 0 else float('inf')
            complete_ratio = neural_mean / domain_complete_mean if domain_complete_mean > 0 else float('inf')
            
            textstr = f'Neural: {neural_mean:.4f}\nDomain EM: {domain_em_mean:.4f} (ratio: {em_ratio:.2f})\nDomain Complete: {domain_complete_mean:.4f} (ratio: {complete_ratio:.2f})'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
    
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
        ax1.hist(all_neural_kls, bins=50, alpha=0.7, color='blue', density=True)
        ax1.set_xlabel('KL Divergence', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Neural Imputer KL Distribution', fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(all_neural_kls)
        median_kl = np.median(all_neural_kls)
        ax1.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.3f}')
        ax1.axvline(median_kl, color='green', linestyle='--', label=f'Median: {median_kl:.3f}')
        ax1.legend()
    
    # Domain EM model histogram
    if len(all_domain_em_kls) > 0:
        ax2.hist(all_domain_em_kls, bins=50, alpha=0.7, color='orange', density=True)
        ax2.set_xlabel('KL Divergence', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Domain EM KL Distribution', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(all_domain_em_kls)
        median_kl = np.median(all_domain_em_kls)
        ax2.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.3f}')
        ax2.axvline(median_kl, color='green', linestyle='--', label=f'Median: {median_kl:.3f}')
        ax2.legend()
    
    # Domain Complete model histogram
    if len(all_domain_complete_kls) > 0:
        ax3.hist(all_domain_complete_kls, bins=50, alpha=0.7, color='green', density=True)
        ax3.set_xlabel('KL Divergence', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Domain Complete KL Distribution', fontsize=14)
        ax3.set_yscale('log')
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