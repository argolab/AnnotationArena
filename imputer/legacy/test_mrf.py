"""
MRF to Bayesian Network conversion and exact inference comparison.

This module implements the professor's suggestion to generate MRFs,
convert them to equivalent Bayesian Networks, and compare neural methods
against exact inference that knows the true model parameters.

Author: Prabhav Singh
"""

import os
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError("pyAgrum is required for MRF experiments")

# Import neural imputers
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

# Import domain models
from domain_specific_model import (
    learn_domain_specific_model,
    learn_domain_specific_model_complete,
    evaluate_domain_specific_model,
    convert_training_data_for_pyagrum
)

# Import data generation
from data_generation import generate_dataset_fair, create_complete_training_data

def clear_memory():
    """Clear GPU memory between experiments."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def generate_random_undirected_graph(n_nodes, edge_prob=0.3, seed=42):
    """Generate random undirected graph (Erdos-Renyi)."""
    np.random.seed(seed)
    
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < edge_prob:
                edges.append((i, j))
    
    print(f"Generated undirected graph: {n_nodes} nodes, {len(edges)} edges")
    return edges

def create_adjacency_from_edges(edges, n_nodes):
    """Create adjacency matrix from edge list."""
    adj = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1  # Undirected
    return adj

def convert_mrf_to_bn(edges, n_nodes, seed=42):
    """
    Convert undirected MRF to equivalent directed Bayesian Network.
    Uses simple ordering-based approach: for each node i, 
    make all higher-numbered neighbors into parents.
    """
    np.random.seed(seed)
    
    print(f"Converting MRF to BN...")
    
    # Create BN structure
    bn = gum.BayesNet("MRF_to_BN")
    
    # Add binary variables
    for i in range(n_nodes):
        bn.add(gum.LabelizedVariable(str(i), str(i), ["0", "1"]))
    
    # Add directed edges based on ordering
    directed_edges = []
    for i, j in edges:
        if i < j:
            # Add edge i -> j (lower index -> higher index)
            bn.addArc(str(i), str(j))
            directed_edges.append((i, j))
        else:
            # Add edge j -> i (lower index -> higher index)
            bn.addArc(str(j), str(i))
            directed_edges.append((j, i))
    
    print(f"Created directed BN: {len(directed_edges)} arcs")
    
    # Generate CPTs for the BN
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    # Create adjacency matrix for directed graph
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for i, j in directed_edges:
        adj_matrix[i, j] = 1
    
    return bn, adj_matrix

def evaluate_exact_inference(bn, test_data, n_nodes, n_states=2):
    """
    Evaluate using exact inference with the true BN (oracle baseline).
    This represents the best possible performance for domain-specific methods.
    """
    print(f"Evaluating exact inference oracle on {len(test_data)} test samples...")
    
    infer = gum.LazyPropagation(bn)
    kl_divergences = []
    prediction_errors = []
    failed_inferences = 0
    
    for inputs, structure_info, dimensions, mask, targets in test_data:
        # Get observed and unobserved nodes
        evidence = {}
        unobserved_nodes = []
        
        for node in range(n_nodes):
            if mask[node] == 0:  # Observed
                state = torch.argmax(inputs[node, 1:]).item()
                evidence[str(node)] = str(state)
            else:  # Unobserved
                unobserved_nodes.append(node)
        
        if not unobserved_nodes:
            continue
        
        # Use exact inference for each unobserved node
        for node in unobserved_nodes:
            try:
                node_str = str(node)
                
                # Set evidence and run exact inference
                if evidence:
                    infer.setEvidence(evidence)
                    infer.makeInference()
                    posterior = infer.posterior(node_str)
                    # Extract probabilities
                    pred_probs = np.array([posterior[{node_str: "0"}], posterior[{node_str: "1"}]])
                else:
                    # No evidence, use marginal
                    posterior = infer.posterior(node_str)
                    pred_probs = np.array([posterior[{node_str: "0"}], posterior[{node_str: "1"}]])
                
                # Ensure probabilities are valid
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    pred_probs = np.ones(n_states) / n_states
                else:
                    pred_probs = pred_probs / np.sum(pred_probs)
                
                # Get ground truth
                true_probs = targets[node].numpy()
                
                if np.any(np.isnan(true_probs)) or np.sum(true_probs) == 0:
                    failed_inferences += 1
                    continue
                
                # Compute KL divergence: KL(true || pred)
                kl = 0.0
                for state in range(n_states):
                    if true_probs[state] > 1e-10:
                        kl += true_probs[state] * np.log(
                            (true_probs[state] + 1e-10) / (pred_probs[state] + 1e-10)
                        )
                
                if np.isnan(kl) or np.isinf(kl) or kl < 0:
                    failed_inferences += 1
                    continue
                
                kl_divergences.append(kl)
                
                # Prediction error
                error = np.linalg.norm(pred_probs - true_probs)
                prediction_errors.append(error)
                
                # Clear evidence for next inference
                infer.eraseAllEvidence()
                
            except Exception as e:
                if len(kl_divergences) < 5:
                    print(f"Exact inference failed for node {node}: {str(e)[:100]}")
                failed_inferences += 1
                continue
    
    if not kl_divergences:
        return {
            'mean_kl': float('inf'),
            'std_kl': 0.0,
            'mean_error': float('inf'),
            'failed_rate': 1.0,
            'n_evaluations': 0
        }
    
    results = {
        'mean_kl': np.mean(kl_divergences),
        'std_kl': np.std(kl_divergences),
        'mean_error': np.mean(prediction_errors),
        'failed_rate': failed_inferences / (len(kl_divergences) + failed_inferences),
        'n_evaluations': len(kl_divergences),
        'kl_distribution': kl_divergences
    }
    
    print(f"Exact inference: Mean KL = {results['mean_kl']:.4f}, "
          f"Failed rate = {results['failed_rate']:.2%}")
    
    return results

def run_mrf_experiment(n_nodes=5, train_size=500, edge_prob=0.3, missing_rate=0.4, 
                      neural_type="structure", num_trials=3, seed=42):
    """
    Run MRF experiment comparing neural methods vs exact inference oracle.
    
    Args:
        n_nodes: Number of nodes
        train_size: Training set size
        edge_prob: Edge probability for undirected graph
        missing_rate: Missing data rate
        neural_type: "structure", "cpts", or "both"
        num_trials: Number of trials
        seed: Random seed
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"MRF EXPERIMENT")
    print(f"Nodes: {n_nodes}, Train Size: {train_size}, Edge Prob: {edge_prob}")
    print(f"Missing Rate: {missing_rate}, Neural Type: {neural_type}, Trials: {num_trials}")
    print(f"{'='*80}")
    
    obs_ratio = 1.0 - missing_rate
    test_size = 250
    
    # Storage for results
    results_by_method = {}
    if neural_type in ["structure", "both"]:
        results_by_method["neural_structure"] = []
    if neural_type in ["cpts", "both"]:
        results_by_method["neural_cpts"] = []
    results_by_method["domain_em"] = []
    results_by_method["domain_complete"] = []
    results_by_method["exact_inference"] = []
    
    clear_memory()
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        trial_seed = seed + trial * 1000
        
        try:
            # Generate undirected graph (MRF)
            edges = generate_random_undirected_graph(n_nodes, edge_prob, trial_seed)
            
            # Convert to equivalent BN
            bn, adj_matrix = convert_mrf_to_bn(edges, n_nodes, trial_seed)
            
            # Generate training and test data from BN
            train_data = generate_dataset_fair(bn, adj_matrix, n_nodes, train_size, obs_ratio)
            test_data = generate_dataset_fair(bn, adj_matrix, n_nodes, test_size, obs_ratio)
            complete_train_data = create_complete_training_data(bn, adj_matrix, n_nodes, train_size)
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"Failed to generate data for trial {trial + 1}")
                continue
            
            # METHOD 1: Neural Structure (if requested)
            if neural_type in ["structure", "both"]:
                print(f"Training Neural Imputer (structure)...")
                train_dataset = GraphDatasetStructure(train_data)
                test_dataset = GraphDatasetStructure(test_data)
                
                batch_size = min(32, len(train_data))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_structure)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_structure)
                
                input_dim = train_data[0][0].shape[1]
                structure_dim = train_data[0][1].shape[1]
                model = create_model_structure(n_nodes, input_dim, structure_dim)
                model = train_model_structure(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
                
                neural_structure_results = evaluate_neural_model_structure(model, test_data, n_nodes, 2)
                results_by_method["neural_structure"].append(neural_structure_results.get('mean_kl', float('inf')))
            
            # METHOD 2: Neural CPTs (if requested)
            if neural_type in ["cpts", "both"]:
                print(f"Training Neural Imputer (cpts)...")
                train_dataset = GraphDatasetWithCPTs(train_data, bn)
                test_dataset = GraphDatasetWithCPTs(test_data, bn)
                
                batch_size = min(32, len(train_data))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cpts)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cpts)
                
                input_dim = train_data[0][0].shape[1]
                structure_dim = train_data[0][1].shape[1]
                model = create_model_cpts(n_nodes, input_dim, structure_dim)
                model = train_model_cpts(model, train_loader, test_loader, epochs=50, lr=1e-4, patience=15)
                
                neural_cpts_results = evaluate_neural_model_cpts(model, test_data, bn, n_nodes, 2)
                results_by_method["neural_cpts"].append(neural_cpts_results.get('mean_kl', float('inf')))
            
            # METHOD 3: Domain EM
            print(f"Training Domain EM...")
            pyagrum_data = convert_training_data_for_pyagrum(train_data, n_nodes)
            learned_bn_em = learn_domain_specific_model(adj_matrix, pyagrum_data, n_states=2, max_iter=100, epsilon=1e-3)
            domain_em_results = evaluate_domain_specific_model(learned_bn_em, test_data, n_nodes, 2)
            results_by_method["domain_em"].append(domain_em_results.get('mean_kl', float('inf')))
            
            # METHOD 4: Domain Complete
            print(f"Training Domain Complete...")
            learned_bn_complete = learn_domain_specific_model_complete(adj_matrix, complete_train_data, n_nodes, n_states=2)
            domain_complete_results = evaluate_domain_specific_model(learned_bn_complete, test_data, n_nodes, 2)
            results_by_method["domain_complete"].append(domain_complete_results.get('mean_kl', float('inf')))
            
            # METHOD 5: Exact Inference Oracle
            print(f"Evaluating Exact Inference Oracle...")
            exact_results = evaluate_exact_inference(bn, test_data, n_nodes, 2)
            results_by_method["exact_inference"].append(exact_results.get('mean_kl', float('inf')))
            
            # Print trial summary
            print(f"Trial {trial + 1} Results:")
            for method, values in results_by_method.items():
                if values:  # Only print if method was run
                    print(f"  {method}: {values[-1]:.4f}")
            
            clear_memory()
            
        except Exception as e:
            print(f"Trial {trial + 1} FAILED: {e}")
            clear_memory()
            continue
    
    # Aggregate results
    final_results = {}
    for method, values in results_by_method.items():
        if values:
            valid_values = [v for v in values if v != float('inf')]
            if valid_values:
                final_results[method] = {
                    'mean_kl': np.mean(valid_values),
                    'std_kl': np.std(valid_values),
                    'all_kl': valid_values,
                    'successful_trials': len(valid_values)
                }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MRF EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for method, stats in final_results.items():
        print(f"{method}: {stats['mean_kl']:.4f} ± {stats['std_kl']:.4f} ({stats['successful_trials']} trials)")
    
    # Compare against exact inference
    if 'exact_inference' in final_results:
        exact_kl = final_results['exact_inference']['mean_kl']
        print(f"\nComparison vs Exact Inference (KL = {exact_kl:.4f}):")
        
        for method, stats in final_results.items():
            if method != 'exact_inference':
                ratio = stats['mean_kl'] / exact_kl if exact_kl > 0 else float('inf')
                print(f"  {method}: {ratio:.2f}x worse than oracle")
    
    return final_results

def run_mrf_scaling_test(neural_type="both", num_trials=3):
    """Test MRF experiment across different scales."""
    print(f"\n{'='*80}")
    print(f"MRF SCALING TEST")
    print(f"{'='*80}")
    
    configs = [
        (5, 500, 0.3, 0.4),    # Small graph
        (7, 750, 0.3, 0.4),    # Medium graph  
        (10, 1000, 0.3, 0.4),  # Large graph
    ]
    
    all_results = {}
    
    for n_nodes, train_size, edge_prob, missing_rate in configs:
        print(f"\n--- Testing {n_nodes} nodes, {train_size} train size ---")
        
        result = run_mrf_experiment(
            n_nodes=n_nodes,
            train_size=train_size,
            edge_prob=edge_prob,
            missing_rate=missing_rate,
            neural_type=neural_type,
            num_trials=num_trials,
            seed=42
        )
        
        if result:
            all_results[(n_nodes, train_size)] = result
            
            # Save results
            base_dir = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
            with open(os.path.join(base_dir, 'results', 'mrf_scaling_results.json'), 'w') as f:
                # Convert to JSON-serializable format
                json_results = {}
                for key, methods in all_results.items():
                    json_results[f"{key[0]}_{key[1]}"] = methods
                json.dump(json_results, f, indent=2)
    
    # Create comparison plot
    if all_results:
        create_mrf_comparison_plot(all_results, neural_type)
    
    return all_results

def create_mrf_comparison_plot(results, neural_type):
    """Create plot comparing methods against exact inference."""
    if not results:
        return
    
    configs = sorted(results.keys())
    config_labels = [f"{n} nodes" for n, _ in configs]
    
    # Extract data for plotting
    methods = []
    method_data = {}
    
    # Determine which methods we have
    for config, result in results.items():
        for method in result.keys():
            if method not in method_data:
                method_data[method] = []
                methods.append(method)
    
    # Fill data for all configs
    for config in configs:
        result = results[config]
        for method in methods:
            if method in result:
                method_data[method].append(result[method]['mean_kl'])
            else:
                method_data[method].append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'neural_structure': 'blue', 'neural_cpts': 'cyan', 'domain_em': 'orange', 
              'domain_complete': 'green', 'exact_inference': 'red'}
    markers = {'neural_structure': 'o', 'neural_cpts': 's', 'domain_em': '^', 
               'domain_complete': 'v', 'exact_inference': '*'}
    
    x_pos = np.arange(len(configs))
    
    for method in methods:
        color = colors.get(method, 'black')
        marker = markers.get(method, 'o')
        label = method.replace('_', ' ').title()
        
        ax.plot(x_pos, method_data[method], marker=marker, color=color, 
               linewidth=2, markersize=8, label=label)
    
    ax.set_xlabel('Graph Configuration', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('MRF Experiment: Neural vs Domain vs Exact Inference', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(base_dir, 'results', 'mrf_comparison_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"MRF comparison plot saved: {os.path.join(base_dir, 'results', 'mrf_comparison_plot.png')}")

if __name__ == "__main__":
    print("MRF to Bayesian Network Experiment")
    
    # Example usage
    result = run_mrf_experiment(
        n_nodes=5,
        train_size=500,
        edge_prob=0.3,
        missing_rate=0.4,
        neural_type="both",
        num_trials=3
    )
    
    print("\nMRF experiment completed!")