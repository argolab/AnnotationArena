"""
Data generation for graph imputation experiments using pyAgrum.

This module generates Bayesian Networks and samples for training and testing
graph imputation algorithms.

Author: Prabhav Singh
"""

import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError("pyAgrum is required for this implementation")


def generate_direct_bn_structure(n_nodes: int, target_parents: float = 1.5, seed: int = 42) -> gum.BayesNet:
    """Generate BN directly with O(1) parents using min(1, c/(i-1)) method."""
    print(f"Generating direct BN with {n_nodes} nodes, target_parents={target_parents}, seed={seed}")
    
    np.random.seed(seed)
    
    # Create BN
    bn = gum.BayesNet("DirectBN")
    
    # Add binary variables
    for i in range(n_nodes):
        bn.add(gum.LabelizedVariable(str(i), str(i), ["0", "1"]))
    
    # Add edges using min(1, c/(i-1)) probability for O(1) parents
    c = target_parents
    edges_added = 0
    
    for child in range(1, n_nodes):  # Start from 1 since node 0 has no potential parents
        num_candidates = child  # Nodes 0, 1, ..., child-1 are potential parents
        edge_prob = min(1.0, c / num_candidates)
        
        for potential_parent in range(child):
            if np.random.random() < edge_prob:
                bn.addArc(str(potential_parent), str(child))
                edges_added += 1
                print(f"  Added arc {potential_parent} -> {child} (prob={edge_prob:.3f})")
    
    # Generate random CPTs
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    print(f"Generated BN: {bn.size()} nodes, {edges_added} arcs")
    
    # Print parent count statistics
    parent_counts = []
    for child in range(n_nodes):
        parents = list(bn.parents(str(child)))
        parent_counts.append(len(parents))
        if len(parents) > 0:
            print(f"  Node {child}: {len(parents)} parents {parents}")
    
    avg_parents = np.mean(parent_counts)
    print(f"Average parents per node: {avg_parents:.2f}")
    
    return bn


def create_adjacency_matrix(bn: gum.BayesNet, n_nodes: int) -> np.ndarray:
    """Create adjacency matrix from pyAgrum BN."""
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    print("Creating adjacency matrix from BN structure:")
    for arc in bn.arcs():
        parent_id, child_id = arc
        parent_name = bn.variable(parent_id).name()
        child_name = bn.variable(child_id).name()
        i, j = int(parent_name), int(child_name)
        adj_matrix[i, j] = 1.0
        print(f"  Arc {parent_name} -> {child_name}: adj_matrix[{i}, {j}] = 1")
    
    print(f"Adjacency matrix:\n{adj_matrix}")
    return adj_matrix


def create_parameter_embeddings(bn: gum.BayesNet, adj_matrix: np.ndarray) -> np.ndarray:
    """Create parameter embeddings from BN (adjacency + CPD data)."""
    n_nodes = adj_matrix.shape[0]
    
    # Get CPD data
    max_cpd_size = 0
    cpd_data_list = []
    
    for node_id in bn.nodes():
        cpt = bn.cpt(node_id)
        cpd_values = np.array(cpt.tolist()).flatten()
        cpd_data_list.append(cpd_values)
        max_cpd_size = max(max_cpd_size, len(cpd_values))
    
    # Create CPD matrix
    cpd_data = np.zeros((n_nodes, max_cpd_size), dtype=np.float32)
    for i, cpd_values in enumerate(cpd_data_list):
        cpd_data[i, :len(cpd_values)] = cpd_values
    
    # Combine adjacency and CPD data
    param_embeddings = np.concatenate([adj_matrix, cpd_data], axis=1)
    print(f"Parameter embeddings shape: {param_embeddings.shape}")
    
    return param_embeddings


def generate_sample_from_bn_with_embeddings(bn: gum.BayesNet, 
                                           param_embeddings: np.ndarray,
                                           n_nodes: int, 
                                           obs_ratio: float, 
                                           seed: int) -> Tuple:
    """Generate single training sample from pyAgrum BN with parameter embeddings."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate a complete sample from the BN
    samples_df, _ = gum.generateSample(bn, n=1, with_labels=False)
    complete_sample = samples_df.iloc[0].to_dict()
    
    # Debug: print the sample structure
    if seed == 0:  # Only print for first sample
        print(f"Sample columns: {list(samples_df.columns)}")
        print(f"Sample values: {complete_sample}")
    
    # Convert to our format
    node_states = {}
    for node_id in bn.nodes():
        node_name = bn.variable(node_id).name()
        # Try different key formats
        if node_name in complete_sample:
            state = complete_sample[node_name]
        elif node_id in complete_sample:
            state = complete_sample[node_id]
        elif str(node_id) in complete_sample:
            state = complete_sample[str(node_id)]
        else:
            # Fallback: use column index
            state = samples_df.iloc[0, node_id]
        
        node_states[int(node_name)] = int(state)
    
    # Select observed nodes
    n_observed = max(1, int(obs_ratio * n_nodes))
    node_list = list(range(n_nodes))
    observed_nodes = random.sample(node_list, k=n_observed)
    unobserved_nodes = [node for node in node_list if node not in observed_nodes]
    
    # Create inputs: [mask_bit, state_0_bit, state_1_bit]
    inputs = np.zeros((n_nodes, 3), dtype=np.float32)
    
    for node in node_list:
        if node in observed_nodes:
            inputs[node, 0] = 0.0  # Not masked
            state = node_states[node]
            inputs[node, 1 + state] = 1.0  # One-hot encoding
        else:
            inputs[node, 0] = 1.0  # Masked
    
    # Create evidence and compute ground truth using pyAgrum inference
    evidence = {str(node): str(node_states[node]) for node in observed_nodes}
    
    targets = np.zeros((n_nodes, 2), dtype=np.float32)
    
    if unobserved_nodes:
        infer = gum.LazyPropagation(bn)
        
        for node in unobserved_nodes:
            try:
                if evidence:
                    infer.setEvidence(evidence)
                    infer.makeInference()
                    posterior = infer.posterior(str(node))
                else:
                    # No evidence - use marginal
                    posterior = infer.posterior(str(node))
                
                # Get probabilities for states "0" and "1"
                # posterior is a Potential object, use indexing with variable values
                node_name = str(node)
                targets[node, 0] = posterior[{node_name: "0"}]  # P(node="0"|evidence)
                targets[node, 1] = posterior[{node_name: "1"}]  # P(node="1"|evidence)
                
                infer.eraseAllEvidence()
                
            except Exception as e:
                print(f"Inference failed for node {node}: {e}")
                return None
    
    # Create mask: 0 for observed, 1 for unobserved
    mask = np.zeros(n_nodes, dtype=np.float32)
    for node in unobserved_nodes:
        mask[node] = 1.0
    
    dimensions = np.arange(n_nodes, dtype=np.int64)
    
    return (
        torch.FloatTensor(inputs),
        torch.FloatTensor(param_embeddings),
        torch.LongTensor(dimensions),
        torch.FloatTensor(mask),
        torch.FloatTensor(targets)
    )


def generate_sample_from_bn_fair(bn: gum.BayesNet, 
                                adj_matrix: np.ndarray,
                                n_nodes: int, 
                                obs_ratio: float, 
                                seed: int) -> Tuple:
    """Generate single training sample from pyAgrum BN."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate a complete sample from the BN
    samples_df, _ = gum.generateSample(bn, n=1, with_labels=False)
    complete_sample = samples_df.iloc[0].to_dict()
    
    # Debug: print the sample structure
    if seed == 0:  # Only print for first sample
        print(f"Sample columns: {list(samples_df.columns)}")
        print(f"Sample values: {complete_sample}")
    
    # Convert to our format
    node_states = {}
    for node_id in bn.nodes():
        node_name = bn.variable(node_id).name()
        # Try different key formats
        if node_name in complete_sample:
            state = complete_sample[node_name]
        elif node_id in complete_sample:
            state = complete_sample[node_id]
        elif str(node_id) in complete_sample:
            state = complete_sample[str(node_id)]
        else:
            # Fallback: use column index
            state = samples_df.iloc[0, node_id]
        
        node_states[int(node_name)] = int(state)
    
    # Select observed nodes
    n_observed = max(1, int(obs_ratio * n_nodes))
    node_list = list(range(n_nodes))
    observed_nodes = random.sample(node_list, k=n_observed)
    unobserved_nodes = [node for node in node_list if node not in observed_nodes]
    
    # Create inputs: [mask_bit, state_0_bit, state_1_bit]
    inputs = np.zeros((n_nodes, 3), dtype=np.float32)
    
    for node in node_list:
        if node in observed_nodes:
            inputs[node, 0] = 0.0  # Not masked
            state = node_states[node]
            inputs[node, 1 + state] = 1.0  # One-hot encoding
        else:
            inputs[node, 0] = 1.0  # Masked
    
    # Create evidence and compute ground truth using pyAgrum inference
    evidence = {str(node): str(node_states[node]) for node in observed_nodes}
    
    targets = np.zeros((n_nodes, 2), dtype=np.float32)
    
    if unobserved_nodes:
        infer = gum.LazyPropagation(bn)
        
        for node in unobserved_nodes:
            try:
                if evidence:
                    infer.setEvidence(evidence)
                    infer.makeInference()
                    posterior = infer.posterior(str(node))
                else:
                    # No evidence - use marginal
                    posterior = infer.posterior(str(node))
                
                # Get probabilities for states "0" and "1"
                # posterior is a Potential object, use indexing with variable values
                node_name = str(node)
                targets[node, 0] = posterior[{node_name: "0"}]  # P(node="0"|evidence)
                targets[node, 1] = posterior[{node_name: "1"}]  # P(node="1"|evidence)
                
                infer.eraseAllEvidence()
                
            except Exception as e:
                print(f"Inference failed for node {node}: {e}")
                return None
    
    # Create mask: 0 for observed, 1 for unobserved
    mask = np.zeros(n_nodes, dtype=np.float32)
    for node in unobserved_nodes:
        mask[node] = 1.0
    
    dimensions = np.arange(n_nodes, dtype=np.int64)
    
    # Create structural embeddings (just adjacency matrix)
    structural_embeddings = np.tile(adj_matrix, (1, 1)).astype(np.float32)
    
    return (
        torch.FloatTensor(inputs),
        torch.FloatTensor(structural_embeddings),
        torch.LongTensor(dimensions),
        torch.FloatTensor(mask),
        torch.FloatTensor(targets)
    )


def generate_dataset_fair(bn: gum.BayesNet, 
                         adj_matrix: np.ndarray,
                         n_nodes: int, 
                         n_samples: int, 
                         obs_ratio: float) -> List[Tuple]:
    """Generate full dataset from pyAgrum BN without parameter embeddings."""
    samples = []
    failed_count = 0
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        sample = generate_sample_from_bn_fair(bn, adj_matrix, n_nodes, obs_ratio, i)
        if sample is not None:
            samples.append(sample)
        else:
            failed_count += 1
    
    print(f"Generated {len(samples)} samples, {failed_count} failed")
    return samples


def generate_dataset(bn: gum.BayesNet, 
                    param_embeddings: np.ndarray,
                    n_nodes: int, 
                    n_samples: int, 
                    obs_ratio: float) -> List[Tuple]:
    """Generate full dataset from pyAgrum BN."""
    samples = []
    failed_count = 0
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        sample = generate_sample_from_bn_with_embeddings(bn, param_embeddings, n_nodes, obs_ratio, i)
        if sample is not None:
            samples.append(sample)
        else:
            failed_count += 1
    
    print(f"Generated {len(samples)} samples, {failed_count} failed")
    return samples


def create_experiment_data(n_nodes: int, 
                          train_size: int, 
                          test_size: int,
                          target_parents: float = 1.5,
                          obs_ratio: float = 0.5,
                          seed: int = 42) -> Tuple:
    """Create complete experiment data using direct BN generation."""
    print(f"Creating data: {n_nodes} nodes, {train_size} train, {test_size} test, seed={seed}")
    
    # Generate BN structure and parameters using direct method
    bn = generate_direct_bn_structure(n_nodes, target_parents, seed)
    adj_matrix = create_adjacency_matrix(bn, n_nodes)
    
    # Generate training and test data
    train_data = generate_dataset_fair(bn, adj_matrix, n_nodes, train_size, obs_ratio)
    test_data = generate_dataset_fair(bn, adj_matrix, n_nodes, test_size, obs_ratio)
    
    return bn, adj_matrix, train_data, test_data


def create_complete_training_data(bn: gum.BayesNet, 
                                 adj_matrix: np.ndarray,
                                 n_nodes: int, 
                                 train_size: int) -> List[Tuple]:
    """Create complete training data (no missing values) for domain baseline."""
    print(f"Creating complete training data: {train_size} samples, no missing values")
    
    # Generate training data with obs_ratio=1.0 (no missing values)
    complete_train_data = generate_dataset_fair(bn, adj_matrix, n_nodes, train_size, obs_ratio=1.0)
    
    return complete_train_data