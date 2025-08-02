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
    evaluate_domain_specific_model,
    convert_training_data_for_pyagrum,
    extract_adjacency_from_embeddings
)

from data_generation import create_experiment_data

from neural_imputer import (
    create_model,
    train_model,
    GraphDataset,
    collate_fn,
    evaluate_neural_model,
    DEVICE
)

# Test configuration
TEST_GRAPH_SIZE = [5]
TEST_TRAINING_SIZES = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750]
TEST_SIZE = 250

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
    """Run single graph imputation experiment."""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {n_nodes} nodes, {train_size} training samples")
    print(f"{'='*50}")
    
    clear_memory()
    
    try:
        # Generate data using pyAgrum
        print("Generating data with pyAgrum...")
        bn, param_embeddings, train_data, test_data = create_experiment_data(
            n_nodes, train_size, TEST_SIZE, edge_prob=0.35, obs_ratio=0.5
        )
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("Failed to generate data")
            return None
        
        # Train neural imputer
        print("\n=== TRAINING NEURAL IMPUTER ===")
        train_dataset = GraphDataset(train_data)
        test_dataset = GraphDataset(test_data)
        
        batch_size = min(32, len(train_data))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        input_dim = train_data[0][0].shape[1]
        embedding_dim = train_data[0][1].shape[1]
        model = create_model(n_nodes, input_dim, embedding_dim)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        model = train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
        
        # Evaluate neural imputer
        print("\n=== EVALUATING NEURAL IMPUTER ===")
        neural_results = evaluate_neural_model(model, test_data, n_nodes, 2)
        
        # Train domain-specific model using pyAgrum
        print("\n=== TRAINING DOMAIN-SPECIFIC MODEL (PYAGRUM) ===")
        
        # Convert training data to pyAgrum format
        pyagrum_train_data = convert_training_data_for_pyagrum(train_data, n_nodes)
        
        # Extract adjacency matrix
        adj_matrix = extract_adjacency_from_embeddings(param_embeddings, n_nodes)
        
        # Learn with pyAgrum EM
        learned_bn = learn_domain_specific_model(
            adj_matrix, pyagrum_train_data, n_states=2, max_iter=100, epsilon=1e-3
        )
        
        # Evaluate domain-specific model
        print("\n=== EVALUATING DOMAIN-SPECIFIC MODEL ===")
        domain_results = evaluate_domain_specific_model(learned_bn, test_data, n_nodes, 2)
        
        # Compare results
        neural_kl = neural_results.get('mean_kl', float('inf'))
        domain_kl = domain_results.get('mean_kl', float('inf'))
        kl_ratio = domain_kl / neural_kl if neural_kl > 0 else float('inf')
        
        results = {
            'config': {'n_nodes': n_nodes, 'train_size': train_size},
            'neural': neural_results,
            'domain': domain_results,
            'comparison': {
                'neural_kl': neural_kl,
                'domain_kl': domain_kl,
                'kl_ratio': kl_ratio
            }
        }
        
        print(f"\n=== RESULTS ===")
        print(f"Neural KL: {neural_kl:.4f}")
        print(f"Domain KL (pyAgrum): {domain_kl:.4f}")
        print(f"KL Ratio: {kl_ratio:.2f}")
        
        if kl_ratio < 2.0:
            status = "🎉 EXCELLENT: Neural imputer is competitive!"
        elif kl_ratio < 5.0:
            status = "👍 GOOD: Neural imputer is reasonable"
        else:
            status = "⚠️  NEEDS WORK: Neural imputer needs improvement"
        
        print(status)
        
        return results
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None

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
        print("\n" + "="*60)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"{'Train Size':<12} {'Neural KL':<12} {'Domain KL':<12} {'KL Ratio':<12} {'Status'}")
        print("-" * 60)
        
        for (n_nodes, train_size), result in all_results.items():
            comparison = result.get('comparison', {})
            neural_kl = comparison.get('neural_kl', float('inf'))
            domain_kl = comparison.get('domain_kl', float('inf'))
            kl_ratio = comparison.get('kl_ratio', float('inf'))
            
            # Determine status
            if neural_kl == float('inf') or domain_kl == float('inf'):
                status = "FAILED"
            elif kl_ratio < 2.0:
                status = "EXCELLENT"
            elif kl_ratio < 5.0:
                status = "GOOD"
            else:
                status = "POOR"
            
            print(f"{train_size:<12} {neural_kl:<12.4f} {domain_kl:<12.4f} {kl_ratio:<12.2f} {status}")
        
        print("\nInterpretation:")
        print("- KL Ratio < 2.0: Neural imputer is competitive")
        print("- KL Ratio 2.0-5.0: Neural imputer is reasonable") 
        print("- KL Ratio > 5.0: Neural imputer needs improvement")
        
        # Create plot
        create_comparison_plot(all_results)
        
        print("\n✅ Experiment completed successfully!")
        
    else:
        print("\n❌ No successful experiments!")
    
    return all_results

def create_comparison_plot(all_results):
    """Create comparison plot for experiment results."""
    if not all_results:
        return
    
    train_sizes = sorted(set(k[1] for k in all_results.keys()))
    neural_kls = []
    domain_kls = []
    
    for train_size in train_sizes:
        key = (5, train_size)  # Assuming 5-node graph
        if key in all_results:
            result = all_results[key]
            comparison = result.get('comparison', {})
            neural_kl = comparison.get('neural_kl', float('inf'))
            domain_kl = comparison.get('domain_kl', float('inf'))
            
            neural_kls.append(neural_kl if neural_kl != float('inf') else np.nan)
            domain_kls.append(domain_kl if domain_kl != float('inf') else np.nan)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, neural_kls, 'o-', label='Neural Imputer', linewidth=3, markersize=10)
    plt.plot(train_sizes, domain_kls, 's-', label='Domain-specific BN (pyAgrum)', linewidth=3, markersize=10)
    plt.xlabel('Training Samples', fontsize=14)
    plt.ylabel('KL Divergence', fontsize=14)
    plt.title('Graph Imputation: Neural vs Domain-Specific BN (5 nodes)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add text box with summary
    if len(neural_kls) > 0 and len(domain_kls) > 0:
        neural_mean = np.nanmean(neural_kls)
        domain_mean = np.nanmean(domain_kls)
        ratio_mean = domain_mean / neural_mean if neural_mean > 0 else float('inf')
        
        textstr = f'Average KL Ratio: {ratio_mean:.2f}\nNeural Mean: {neural_mean:.4f}\nDomain Mean: {domain_mean:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(base_dir, 'results', 'experiment_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {os.path.join(base_dir, 'results', 'experiment_plot.png')}")

if __name__ == "__main__":
    print("🚀 Starting graph imputation experiment...")
    
    # Run the test
    results = run_test()
    
    print("\n🎉 Experiment finished!")