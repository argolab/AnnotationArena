"""
Training and test sample generation for progressive imputation experiments.

Generates samples from Bayesian Networks with controlled missing data patterns,
performs exact Bayesian inference for ground truth targets, and creates proper
input representations for the two-stream transformer architecture.
"""

import logging
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for Bayesian inference in sample generation. "
        "Please install pyAgrum: pip install pyagrum"
    )

logger = logging.getLogger(__name__)

# Type alias for sample tuple
SampleTuple = Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]


def generate_sample_from_bn_fair(bn: gum.BayesNet, 
                                adj_matrix: np.ndarray,
                                n_nodes: int, 
                                obs_ratio: float, 
                                seed: int) -> SampleTuple:
    """
    Generate single training sample from pyAgrum BayesNet with exact inference.
    
    This function creates a sample with realistic missing data patterns and computes
    exact posterior probabilities for unobserved nodes using Bayesian inference.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix representation  
        n_nodes: Number of nodes in the network
        obs_ratio: Fraction of nodes that are observed (1.0 - missing_rate)
        seed: Random seed for reproducible generation
        
    Returns:
        Tuple containing:
        - inputs: [n_nodes, 3] tensor with [mask_bit, state_0_bit, state_1_bit]
        - structure: [n_nodes, n_nodes] adjacency matrix
        - dimensions: [n_nodes] node indices  
        - mask: [n_nodes] binary mask (0=observed, 1=unobserved)
        - targets: [n_nodes, 2] posterior probabilities for unobserved nodes
        - true_states: [n_nodes] actual sampled states for all nodes
        
    Raises:
        Exception: Any inference failures will bubble up with full traceback for debugging
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Input validation
    if not (0.0 < obs_ratio <= 1.0):
        raise ValueError(f"obs_ratio must be in (0.0, 1.0], got {obs_ratio}")
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    
    logger.debug(f"Generating sample with obs_ratio={obs_ratio}, seed={seed}")
    
    # Generate a complete sample from the BN
    samples_df, _ = gum.generateSample(bn, n=1, with_labels=False)
    complete_sample = samples_df.iloc[0].to_dict()
    
    # Convert pyAgrum sample to our node state format
    node_states: Dict[int, int] = {}
    for node_id in bn.nodes():
        node_name = bn.variable(node_id).name()
        
        # Try different key formats that pyAgrum might use
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
    
    logger.debug(f"Node states: {node_states}")
    
    # Select observed nodes randomly based on obs_ratio
    n_observed = max(1, int(obs_ratio * n_nodes))
    node_list = list(range(n_nodes))
    observed_nodes = random.sample(node_list, k=n_observed)
    unobserved_nodes = [node for node in node_list if node not in observed_nodes]
    
    logger.debug(f"Observed nodes: {observed_nodes}")
    logger.debug(f"Unobserved nodes: {unobserved_nodes}")
    
    # Create inputs: [mask_bit, state_0_bit, state_1_bit]
    inputs = np.zeros((n_nodes, 3), dtype=np.float32)
    
    for node in node_list:
        if node in observed_nodes:
            # Observed node: not masked, include state information
            inputs[node, 0] = 0.0  # Not masked
            state = node_states[node]
            inputs[node, 1 + state] = 1.0  # One-hot encoding for state
        else:
            # Unobserved node: masked, no state information
            inputs[node, 0] = 1.0  # Masked
            # inputs[node, 1:] remain zeros (no state information)
    
    # Create evidence dictionary for Bayesian inference
    evidence = {str(node): str(node_states[node]) for node in observed_nodes}
    logger.debug(f"Evidence for inference: {evidence}")
    
    # Compute exact posterior probabilities for unobserved nodes
    targets = np.zeros((n_nodes, 2), dtype=np.float32)
    
    if unobserved_nodes:
        infer = gum.LazyPropagation(bn)
        
        for node in unobserved_nodes:

            if evidence:
                # Set evidence and perform inference
                infer.setEvidence(evidence)
                infer.makeInference()
                posterior = infer.posterior(str(node))
            else:
                # No evidence - use marginal distribution
                posterior = infer.posterior(str(node))
            
            # Extract probabilities for states "0" and "1"
            node_name = str(node)
            prob_0 = posterior[{node_name: "0"}]  # P(node="0"|evidence)
            prob_1 = posterior[{node_name: "1"}]  # P(node="1"|evidence)
            
            targets[node, 0] = prob_0
            targets[node, 1] = prob_1
                        
            # Validate probabilities
            prob_sum = prob_0 + prob_1
            if not (0.99 < prob_sum < 1.01):  # Allow small numerical errors
                logger.warning(f"Probabilities don't sum to 1 for node {node}: sum={prob_sum}")
            
            infer.eraseAllEvidence()
    
    # Create mask: 0 for observed, 1 for unobserved
    mask = np.zeros(n_nodes, dtype=np.float32)
    for node in unobserved_nodes:
        mask[node] = 1.0
    
    logger.debug(f"Mask: {mask}")
    
    # Create dimensions (node indices)
    dimensions = np.arange(n_nodes, dtype=np.int64)
    
    # Create structural embeddings (adjacency matrix)
    structural_embeddings = np.tile(adj_matrix, (1, 1)).astype(np.float32)
    
    # Create true_states tensor with actual sampled states for all nodes
    true_states = np.array([node_states[i] for i in range(n_nodes)], dtype=np.int64)
    
    # Convert to torch tensors
    sample_tuple = (
        torch.FloatTensor(inputs),
        torch.FloatTensor(structural_embeddings),
        torch.LongTensor(dimensions),
        torch.FloatTensor(mask),
        torch.FloatTensor(targets),
        torch.LongTensor(true_states)
    )
    
    logger.debug(f"Sample generated successfully: {len(observed_nodes)} observed, {len(unobserved_nodes)} unobserved")
    
    return sample_tuple


def generate_dataset_fair(bn: gum.BayesNet, 
                         adj_matrix: np.ndarray,
                         n_nodes: int, 
                         n_samples: int, 
                         obs_ratio: float) -> List[SampleTuple]:
    """
    Generate full dataset from pyAgrum BayesNet with progress tracking.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_samples: Number of samples to generate
        obs_ratio: Fraction of observed nodes per sample
        
    Returns:
        List of sample tuples
        
    Raises:
        Exception: Any sample generation failures will bubble up immediately
    """
    logger.info(f"Generating {n_samples} samples with obs_ratio={obs_ratio}")
    
    samples = []
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        # NO TRY-EXCEPT - Let failures bubble up immediately
        sample = generate_sample_from_bn_fair(bn, adj_matrix, n_nodes, obs_ratio, i)
        samples.append(sample)
    
    logger.info(f"Successfully generated {len(samples)} samples")
    return samples


def generate_sample_pool(bn: gum.BayesNet, adj_matrix: np.ndarray, n_nodes: int, 
                        n_samples: int, missing_rate: float = 0.4, seed: int = 42) -> List[SampleTuple]:
    """
    Generate a pool of samples with missing data for progressive experiments.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_samples: Number of samples to generate
        missing_rate: Fraction of nodes that are missing in each sample
        seed: Random seed for reproducibility
        
    Returns:
        List of samples with missing data (same format as training data)
        
    Raises:
        ValueError: Invalid parameter values
        Exception: Any generation failures will bubble up
    """
    if not (0.0 < missing_rate < 1.0):
        raise ValueError(f"missing_rate must be in (0.0, 1.0), got {missing_rate}")
    
    logger.info(f"Generating sample pool: {n_samples} samples with missing_rate={missing_rate}, seed={seed}")
    
    # Convert missing rate to observation ratio
    obs_ratio = 1.0 - missing_rate
    
    # Set global seed for sample pool generation
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate samples with missing data
    sample_pool = generate_dataset_fair(bn, adj_matrix, n_nodes, n_samples, obs_ratio)
    
    logger.info(f"Generated sample pool: {len(sample_pool)} samples for progressive experiments")
    
    return sample_pool


def generate_test_dataset(bn: gum.BayesNet, adj_matrix: np.ndarray, n_nodes: int, 
                         n_test: int, missing_rate: float = 0.4, seed: int = 42) -> List[SampleTuple]:
    """
    Generate test dataset with missing values.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix
        n_nodes: Number of nodes
        n_test: Number of test samples
        missing_rate: Fraction of nodes that are missing
        seed: Random seed for reproducibility
        
    Returns:
        List of test samples with missing data
        
    Raises:
        ValueError: Invalid parameter values
        Exception: Any generation failures will bubble up
    """
    if not (0.0 < missing_rate < 1.0):
        raise ValueError(f"missing_rate must be in (0.0, 1.0), got {missing_rate}")
    
    logger.info(f"Generating test dataset: {n_test} samples with missing_rate={missing_rate}, seed={seed}")
    
    # Convert missing rate to observation ratio  
    obs_ratio = 1.0 - missing_rate
    
    # Set global seed for test data generation
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate test data
    test_data = generate_dataset_fair(bn, adj_matrix, n_nodes, n_test, obs_ratio)
    
    logger.info(f"Generated test dataset: {len(test_data)} samples")
    
    return test_data


def create_training_subset(sample_pool: List[SampleTuple], n_samples: int, seed: int = 42) -> List[SampleTuple]:
    """
    Create a training subset from sample pool for progressive experiments.
    
    Args:
        sample_pool: Full sample pool with missing data
        n_samples: Number of samples to select
        seed: Random seed for reproducible selection
        
    Returns:
        List of selected training samples
        
    Raises:
        ValueError: Invalid parameter values
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    
    if n_samples >= len(sample_pool):
        logger.debug(f"Requested {n_samples} samples >= pool size {len(sample_pool)}, returning full pool")
        return sample_pool
    
    logger.debug(f"Creating training subset: {n_samples} samples from pool of {len(sample_pool)}, seed={seed}")
    
    # Set seed for reproducible selection
    np.random.seed(seed)
    
    # Random selection without replacement
    indices = np.random.choice(len(sample_pool), n_samples, replace=False)
    subset = [sample_pool[i] for i in sorted(indices)]
    
    logger.debug(f"Created training subset: {len(subset)} samples selected")
    
    return subset