"""
Small test script for quick validation of the graph imputation framework.

Tests a 5-node graph with reduced training sizes for faster iteration.
Useful for debugging and quick validation before running full experiments.

Author: Prabhav Singh
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Import main implementation
from ve_graph_imputation import (
    run_single_experiment,
    generate_plots,
    convert_to_json_serializable,
    clear_memory
)

# Test configuration
TEST_GRAPH_SIZE = [5]  # Single graph size for quick testing
TEST_TRAINING_SIZES = [50, 100, 500, 1000]  # Reduced training sizes
TEST_SIZE = 100  # Smaller test set for speed

def run_quick_test():
    """Run quick test with small configuration."""
    print("="*60)
    print("QUICK TEST: Graph Imputation Framework")
    print("="*60)
    print(f"Graph size: {TEST_GRAPH_SIZE}")
    print(f"Training sizes: {TEST_TRAINING_SIZES}")
    print(f"Test size: {TEST_SIZE}")
    print()
    
    # Override global test size temporarily
    import ve_graph_imputation
    original_test_size = ve_graph_imputation.TEST_SIZE
    ve_graph_imputation.TEST_SIZE = TEST_SIZE
    
    all_results = {}
    
    try:
        for n_nodes in TEST_GRAPH_SIZE:
            for train_size in TEST_TRAINING_SIZES:
                print(f"\n{'='*50}")
                print(f"TEST: {n_nodes} nodes, {train_size} training samples")
                print(f"{'='*50}")
                
                try:
                    results = run_single_experiment(n_nodes, train_size)
                    if results:
                        all_results[(n_nodes, train_size)] = results
                        
                        # Print summary
                        neural_kl = results.get('comparison', {}).get('neural_kl', float('inf'))
                        domain_kl = results.get('comparison', {}).get('domain_kl', float('inf'))
                        kl_ratio = results.get('comparison', {}).get('kl_ratio', float('inf'))
                        
                        print(f"✅ SUCCESS:")
                        print(f"   Neural KL: {neural_kl:.4f}")
                        print(f"   Domain KL: {domain_kl:.4f}")
                        print(f"   KL Ratio: {kl_ratio:.2f}")
                        
                        # Save intermediate results
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        with open(os.path.join(base_dir, 'results', 'test_results_intermediate.json'), 'w') as f:
                            json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) 
                                          for k, v in all_results.items()}
                            json.dump(json_results, f, indent=2)
                            
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    clear_memory()
                    continue
    
    finally:
        # Restore original test size
        ve_graph_imputation.TEST_SIZE = original_test_size
    
    if all_results:
        print("\n" + "="*60)
        print("GENERATING TEST PLOTS...")
        print("="*60)
        
        try:
            # Generate plots
            generate_plots(all_results)
            
            # Save final results
            base_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base_dir, 'results', 'test_results_final.json'), 'w') as f:
                json_results = {f"{k[0]}_{k[1]}": convert_to_json_serializable(v) 
                              for k, v in all_results.items()}
                json.dump(json_results, f, indent=2)
            
            print("✅ Test completed successfully!")
            print(f"Results saved in: {os.path.join(base_dir, 'results')}")
            
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    else:
        print("\n❌ No successful experiments to plot!")
    
    return all_results

def print_test_summary(all_results):
    """Print summary of test results."""
    if not all_results:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    train_sizes = sorted(set(k[1] for k in all_results.keys()))
    
    print(f"{'Train Size':<12} {'Neural KL':<12} {'Domain KL':<12} {'KL Ratio':<12} {'Status'}")
    print("-" * 60)
    
    for train_size in train_sizes:
        key = (5, train_size)  # 5-node graph
        if key in all_results:
            result = all_results[key]
            neural_kl = result.get('comparison', {}).get('neural_kl', float('inf'))
            domain_kl = result.get('comparison', {}).get('domain_kl', float('inf'))
            kl_ratio = result.get('comparison', {}).get('kl_ratio', float('inf'))
            
            # Determine status
            if neural_kl == float('inf') or domain_kl == float('inf'):
                status = "FAILED"
            elif kl_ratio < 2.0:
                status = "GOOD"
            elif kl_ratio < 5.0:
                status = "OK"
            else:
                status = "POOR"
            
            print(f"{train_size:<12} {neural_kl:<12.4f} {domain_kl:<12.4f} {kl_ratio:<12.2f} {status}")
    
    print("\nInterpretation:")
    print("- KL Ratio < 2.0: Neural imputer is competitive")
    print("- KL Ratio 2.0-5.0: Neural imputer is reasonable")
    print("- KL Ratio > 5.0: Neural imputer needs improvement")

def create_simple_plot(all_results):
    """Create a simple plot for quick visualization."""
    if not all_results:
        return
    
    train_sizes = sorted(set(k[1] for k in all_results.keys()))
    neural_kls = []
    domain_kls = []
    
    for train_size in train_sizes:
        key = (5, train_size)
        if key in all_results:
            result = all_results[key]
            neural_kl = result.get('comparison', {}).get('neural_kl', float('inf'))
            domain_kl = result.get('comparison', {}).get('domain_kl', float('inf'))
            
            # Filter out infinite values
            if neural_kl != float('inf'):
                neural_kls.append(neural_kl)
            else:
                neural_kls.append(np.nan)
                
            if domain_kl != float('inf'):
                domain_kls.append(domain_kl)
            else:
                domain_kls.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, neural_kls, 'o-', label='Neural Imputer', linewidth=2, markersize=8)
    plt.plot(train_sizes, domain_kls, 's-', label='Domain-specific BN', linewidth=2, markersize=8)
    plt.xlabel('Training Samples')
    plt.ylabel('KL Divergence')
    plt.title('Quick Test: KL Divergence Comparison (5 nodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(base_dir, 'results', 'quick_test_plot.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Quick plot saved: {os.path.join(base_dir, 'results', 'quick_test_plot.png')}")

if __name__ == "__main__":
    print("Starting quick test...")
    
    # Run the test
    results = run_quick_test()
    
    # Print summary
    print_test_summary(results)
    
    # Create simple plot
    create_simple_plot(results)
    
    print("\nQuick test completed!")