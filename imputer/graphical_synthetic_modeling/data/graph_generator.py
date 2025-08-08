"""
Bayesian Network graph generation for progressive imputation experiments.

Generates synthetic Bayesian Networks with controlled structure using O(1) parents
per node on average, extracts adjacency matrices, and creates parameter embeddings
with privacy-preserving masking for the two-stream architecture.
"""

import logging
from typing import Tuple, List, Optional, Set
import numpy as np

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for Bayesian Network generation. "
        "Please install pyAgrum: pip install pyagrum"
    )

logger = logging.getLogger(__name__)


def generate_direct_bn_structure(n_nodes: int, target_parents: float = 1.5, seed: int = 42) -> gum.BayesNet:
    """
    Generate Bayesian Network directly with O(1) parents using min(1, c/(i-1)) method.
    
    This method ensures that each node has O(1) parents on average by using a decreasing
    probability of adding edges as the number of potential parents increases.
    
    Args:
        n_nodes: Number of nodes in the network
        target_parents: Target average number of parents per node (c parameter)
        seed: Random seed for reproducible generation
        
    Returns:
        gum.BayesNet: Generated Bayesian Network with random CPDs
    """
    logger.debug(f"Generating direct BN with {n_nodes} nodes, target_parents={target_parents}, seed={seed}")
    
    np.random.seed(seed)
    
    # Create BayesNet
    bn = gum.BayesNet("DirectBN")
    
    # Add binary variables with labels "0" and "1"
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
    
    # Generate random CPDs for all nodes
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    logger.debug(f"Generated BN: {bn.size()} nodes, {edges_added} arcs")
    
    # Log parent count statistics for verification
    parent_counts = []
    for child in range(n_nodes):
        parents = list(bn.parents(str(child)))
        parent_counts.append(len(parents))
    
    avg_parents = np.mean(parent_counts)
    logger.debug(f"Average parents per node: {avg_parents:.2f}")
    
    return bn


def create_adjacency_matrix(bn: gum.BayesNet, n_nodes: int) -> np.ndarray:
    """
    Create adjacency matrix representation from pyAgrum BayesNet.
    
    Args:
        bn: pyAgrum BayesNet object
        n_nodes: Number of nodes (for matrix dimensions)
        
    Returns:
        np.ndarray: Adjacency matrix where adj_matrix[i,j] = 1 if edge i->j exists
    """
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    logger.debug("Creating adjacency matrix from BN structure:")
    for arc in bn.arcs():
        parent_id, child_id = arc
        parent_name = bn.variable(parent_id).name()
        child_name = bn.variable(child_id).name()
        i, j = int(parent_name), int(child_name)
        adj_matrix[i, j] = 1.0
    
    logger.debug(f"Adjacency matrix shape: {adj_matrix.shape}")
    logger.debug(f"Total edges: {int(adj_matrix.sum())}")
    
    return adj_matrix


def create_parameter_embeddings(bn: gum.BayesNet, adj_matrix: np.ndarray, 
                              observed_nodes: Optional[Set[int]] = None) -> np.ndarray:
    """
    Create parameter embeddings from BN combining adjacency and CPD data.
    
    Supports privacy-preserving masking where CPD data is only included for
    observed nodes, while unobserved nodes get zeros. This is crucial for
    the two-stream transformer architecture.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix 
        observed_nodes: Set of observed node indices (if None, all nodes are observed)
        
    Returns:
        np.ndarray: Parameter embeddings (adjacency + CPD data)
    """
    n_nodes = adj_matrix.shape[0]
    
    # Extract CPD data from all nodes
    max_cpd_size = 0
    cpd_data_list = []
    
    for node_id in bn.nodes():
        cpt = bn.cpt(node_id)
        cpd_values = np.array(cpt.tolist()).flatten()
        cpd_data_list.append(cpd_values)
        max_cpd_size = max(max_cpd_size, len(cpd_values))
    
    # Create CPD matrix with masking
    cpd_data = np.zeros((n_nodes, max_cpd_size), dtype=np.float32)
    
    for i, cpd_values in enumerate(cpd_data_list):
        if observed_nodes is None or i in observed_nodes:
            # Include CPD data for observed nodes (or all nodes if no mask)
            cpd_data[i, :len(cpd_values)] = cpd_values
        else:
            # Zero out CPD data for unobserved nodes (privacy-preserving)
            cpd_data[i, :] = 0.0
    
    # Combine adjacency and CPD data
    param_embeddings = np.concatenate([adj_matrix, cpd_data], axis=1)
    
    logger.debug(f"Parameter embeddings shape: {param_embeddings.shape}")
    logger.debug(f"  Adjacency component: {adj_matrix.shape}")
    logger.debug(f"  CPD component: {cpd_data.shape}")
    
    return param_embeddings


def create_parameter_embeddings_with_masking(bn: gum.BayesNet, adj_matrix: np.ndarray, 
                                           observed_nodes: List[int]) -> np.ndarray:
    """
    Create parameter embeddings with CPD masking for unobserved nodes.
    
    This is a convenience wrapper around create_parameter_embeddings that
    converts the observed_nodes list to a set for efficient lookup.
    
    Args:
        bn: pyAgrum BayesNet object
        adj_matrix: Adjacency matrix
        observed_nodes: List of observed node indices
        
    Returns:
        np.ndarray: Parameter embeddings with privacy masking applied
    """
    logger.debug(f"Creating parameter embeddings for {len(observed_nodes)} observed nodes")
    
    # Convert list to set for efficient membership testing
    observed_set = set(observed_nodes)
    
    param_embeddings = create_parameter_embeddings(bn, adj_matrix, observed_set)
    
    return param_embeddings


def generate_experiment_graph(n_nodes: int, target_parents: float = 1.0, 
                            seed: int = 42) -> Tuple[gum.BayesNet, np.ndarray]:
    """
    Generate a complete Bayesian Network for progressive imputation experiments.
    
    This is the main public interface that combines structure generation and
    adjacency matrix creation in a single call.
    
    Args:
        n_nodes: Number of nodes in the graph
        target_parents: Target number of parents per node (O(1) parents)
        seed: Random seed for reproducible generation
        
    Returns:
        Tuple[gum.BayesNet, np.ndarray]: BayesNet object and adjacency matrix
    """
    logger.info(f"Generating experiment graph: {n_nodes} nodes, target_parents={target_parents}, seed={seed}")
    
    # Generate BN structure using direct method
    bn = generate_direct_bn_structure(n_nodes, target_parents, seed)
    adj_matrix = create_adjacency_matrix(bn, n_nodes)
    
    # Log final statistics
    n_edges = len(bn.arcs())
    avg_parents = n_edges / n_nodes if n_nodes > 0 else 0
    
    logger.info(f"Generated graph with {bn.size()} nodes, {n_edges} edges")
    logger.info(f"Average parents per node: {avg_parents:.2f}")
    
    return bn, adj_matrix