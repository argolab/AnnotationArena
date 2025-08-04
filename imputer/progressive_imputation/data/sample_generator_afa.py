"""
Sample generation for Pure AFA progressive imputation experiments.

This module generates samples that start completely empty (all nodes unobserved)
for Pure Active Feature Acquisition experiments. Ground truth is preserved 
for evaluation but not used during training until nodes are actively selected.

Author: Prabhav Singh
"""

import logging
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from typing import List, Tuple

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError("pyAgrum is required for this implementation")

logger = logging.getLogger(__name__)

def generate_empty_sample_from_bn(bn: gum.BayesNet, 
                                 adj_matrix: np.ndarray,
                                 n_nodes: int, 
                                 seed: int) -> Tuple:
    """Generate single sample with ALL nodes unobserved (for Pure AFA)."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate a complete sample from the BN (for ground truth)
    samples_df, _ = gum.generateSample(bn, n=1, with_labels=False)
    complete_sample = samples_df.iloc[0].to_dict()
    
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
    
    # Create inputs: ALL nodes start as unobserved (mask=1)
    inputs = np.zeros((n_nodes, 3), dtype=np.float32)
    
    for node in range(n_nodes):
        inputs[node, 0] = 1.0  # ALL nodes masked initially
        # No state information - will be filled in when nodes are observed
    
    # Compute ground truth probabilities for ALL nodes using inference
    # This gives us the true posterior P(node | structure) without any evidence
    targets = np.zeros((n_nodes, 2), dtype=np.float32)
    
    try:
        infer = gum.LazyPropagation(bn)
        
        for node in range(n_nodes):
            # Get marginal probability (no evidence since all nodes unobserved)
            posterior = infer.posterior(str(node))
            
            # Get probabilities for states "0" and "1"
            node_name = str(node)
            targets[node, 0] = posterior[{node_name: "0"}]  # P(node="0")
            targets[node, 1] = posterior[{node_name: "1"}]  # P(node="1")
                
    except Exception as e:
        logger.debug(f"Inference failed for empty sample: {e}")
        return None
    
    # Create mask: ALL nodes unobserved initially (mask=1)
    mask = np.ones(n_nodes, dtype=np.float32)
    
    dimensions = np.arange(n_nodes, dtype=np.int64)
    
    # Create structural embeddings (adjacency matrix)
    structural_embeddings = np.tile(adj_matrix, (1, 1)).astype(np.float32)
    
    # Store ground truth states for AFA to use when applying observations
    ground_truth_states = np.array([node_states[i] for i in range(n_nodes)], dtype=np.int32)
    
    return (
        torch.FloatTensor(inputs),
        torch.FloatTensor(structural_embeddings),
        torch.LongTensor(dimensions),
        torch.FloatTensor(mask),
        torch.FloatTensor(targets),
        torch.IntTensor(ground_truth_states)  # Extra: true states for AFA
    )

def generate_empty_dataset(bn: gum.BayesNet, 
                          adj_matrix: np.ndarray,
                          n_nodes: int, 
                          n_samples: int) -> List[Tuple]:
    """Generate dataset with ALL nodes unobserved initially (Pure AFA approach)."""
    samples = []
    failed_count = 0
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} empty samples"):
        sample = generate_empty_sample_from_bn(bn, adj_matrix, n_nodes, i)
        if sample is not None:
            samples.append(sample)
        else:
            failed_count += 1
    
    logger.debug(f"Generated {len(samples)} empty samples, {failed_count} failed")
    return samples

def generate_afa_sample_pool(bn, adj_matrix, n_nodes, n_samples, seed=42):
    """
    Generate a pool of completely empty samples for Pure AFA experiments.
    
    Args:
        bn: BayesNet object
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        List of samples with ALL nodes unobserved (AFA will select which to observe)
    """
    logger.debug(f"Generating {n_samples} completely empty samples for Pure AFA")
    
    # Generate samples with ALL nodes unobserved
    sample_pool = generate_empty_dataset(bn, adj_matrix, n_nodes, n_samples)
    
    logger.info(f"Generated {len(sample_pool)} empty samples for Pure AFA experiments")
    
    return sample_pool

def generate_afa_test_dataset(bn, adj_matrix, n_nodes, n_test, missing_rate=0.4, seed=42):
    """
    Generate test dataset with missing values (traditional approach for evaluation).
    
    Args:
        bn: BayesNet object  
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_test: Number of test samples
        missing_rate: Fraction of nodes that are missing
        seed: Random seed
        
    Returns:
        List of test samples with missing data (for evaluation)
    """
    logger.debug(f"Generating {n_test} test samples with missing_rate={missing_rate}")
    
    # For test data, we use the traditional approach with some nodes observed
    from .sample_generator import generate_dataset_fair
    
    obs_ratio = 1.0 - missing_rate
    test_data = generate_dataset_fair(bn, adj_matrix, n_nodes, n_test, obs_ratio)
    
    logger.info(f"Generated {len(test_data)} test samples")
    
    return test_data

def apply_afa_observation(sample, node_idx, true_value):
    """
    Apply a single AFA observation to a sample.
    
    Args:
        sample: Sample tuple (inputs, structure, dims, mask, targets, ground_truth)
        node_idx: Index of node to observe
        true_value: True value (0 or 1) to set for this node
        
    Returns:
        Updated sample tuple
    """
    inputs, structure_info, dims, mask, targets, ground_truth = sample
    
    # Update inputs
    new_inputs = inputs.clone()
    new_inputs[node_idx, 0] = 0  # Remove mask bit
    new_inputs[node_idx, 1:] = 0  # Clear old values
    new_inputs[node_idx, 1 + true_value] = 1  # Set true value
    
    # Update mask
    new_mask = mask.clone()
    new_mask[node_idx] = 0  # Mark as observed
    
    return (new_inputs, structure_info, dims, new_mask, targets, ground_truth)

def get_ground_truth_value(sample, node_idx):
    """
    Get the ground truth value for a node from the sample.
    
    Args:
        sample: Sample tuple with ground truth states
        node_idx: Node index
        
    Returns:
        int: Ground truth value (0 or 1)
    """
    inputs, structure_info, dims, mask, targets, ground_truth = sample
    return ground_truth[node_idx].item()

def count_observed_nodes(sample):
    """
    Count how many nodes are currently observed in a sample.
    
    Args:
        sample: Sample tuple
        
    Returns:
        int: Number of observed nodes (mask == 0)
    """
    inputs, structure_info, dims, mask, targets, ground_truth = sample
    return int(torch.sum(mask == 0).item())

def get_samples_with_observations(sample_pool):
    """
    Filter sample pool to only include samples that have at least one observation.
    
    Args:
        sample_pool: List of samples
        
    Returns:
        List of samples with at least one observed node
    """
    training_samples = []
    for sample in sample_pool:
        if count_observed_nodes(sample) > 0:
            # Convert to original format (remove ground truth for training)
            inputs, structure_info, dims, mask, targets, ground_truth = sample
            training_samples.append((inputs, structure_info, dims, mask, targets))
    
    return training_samples