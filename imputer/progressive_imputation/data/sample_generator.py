"""
Sample generation for progressive imputation experiments.

Copied and adapted from the main codebase.
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
        logger.debug(f"Sample columns: {list(samples_df.columns)}")
        logger.debug(f"Sample values: {complete_sample}")
    
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
                logger.debug(f"Inference failed for node {node}: {e}")
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
    
    logger.debug(f"Generated {len(samples)} samples, {failed_count} failed")
    return samples

def generate_sample_pool(bn, adj_matrix, n_nodes, n_samples, missing_rate=0.4, seed=42):
    """
    Generate a pool of samples with missing data for progressive experiments.
    
    Args:
        bn: BayesNet object
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_samples: Number of samples to generate
        missing_rate: Fraction of nodes that are missing in each sample
        seed: Random seed
        
    Returns:
        List of samples with missing data (same format as training data)
    """
    logger.debug(f"Generating {n_samples} samples with missing_rate={missing_rate}")
    
    # Generate samples with missing data
    obs_ratio = 1.0 - missing_rate
    sample_pool = generate_dataset_fair(bn, adj_matrix, n_nodes, n_samples, obs_ratio)
    
    logger.info(f"Generated {len(sample_pool)} samples with missing data for progressive experiments")
    
    return sample_pool

def generate_test_dataset(bn, adj_matrix, n_nodes, n_test, missing_rate=0.4, seed=42):
    """
    Generate test dataset with missing values.
    
    Args:
        bn: BayesNet object  
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_test: Number of test samples
        missing_rate: Fraction of nodes that are missing
        seed: Random seed
        
    Returns:
        List of test samples with missing data
    """
    logger.debug(f"Generating {n_test} test samples with missing_rate={missing_rate}")
    
    obs_ratio = 1.0 - missing_rate
    test_data = generate_dataset_fair(bn, adj_matrix, n_nodes, n_test, obs_ratio)
    
    logger.info(f"Generated {len(test_data)} test samples")
    
    return test_data

def create_training_subset(sample_pool, n_samples, seed=42):
    """
    Create a training subset from sample pool.
    
    Args:
        sample_pool: Full sample pool with missing data
        n_samples: Number of samples to select
        seed: Random seed for selection
        
    Returns:
        List of selected training samples
    """
    if n_samples >= len(sample_pool):
        return sample_pool
    
    np.random.seed(seed)
    indices = np.random.choice(len(sample_pool), n_samples, replace=False)
    subset = [sample_pool[i] for i in sorted(indices)]
    
    logger.debug(f"Created training subset: {len(subset)} samples")
    
    return subset