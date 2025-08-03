"""
Graph generation for progressive imputation experiments.

Copied and adapted from the main codebase.
"""

import logging
import numpy as np

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError("pyAgrum is required for this implementation")

logger = logging.getLogger(__name__)

def generate_direct_bn_structure(n_nodes: int, target_parents: float = 1.5, seed: int = 42) -> gum.BayesNet:
    """Generate BN directly with O(1) parents using min(1, c/(i-1)) method."""
    logger.debug(f"Generating direct BN with {n_nodes} nodes, target_parents={target_parents}, seed={seed}")
    
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
                logger.debug(f"  Added arc {potential_parent} -> {child} (prob={edge_prob:.3f})")
    
    # Generate random CPTs
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    logger.debug(f"Generated BN: {bn.size()} nodes, {edges_added} arcs")
    
    # Print parent count statistics
    parent_counts = []
    for child in range(n_nodes):
        parents = list(bn.parents(str(child)))
        parent_counts.append(len(parents))
        if len(parents) > 0:
            logger.debug(f"  Node {child}: {len(parents)} parents {parents}")
    
    avg_parents = np.mean(parent_counts)
    logger.debug(f"Average parents per node: {avg_parents:.2f}")
    
    return bn

def create_adjacency_matrix(bn: gum.BayesNet, n_nodes: int) -> np.ndarray:
    """Create adjacency matrix from pyAgrum BN."""
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    logger.debug("Creating adjacency matrix from BN structure:")
    for arc in bn.arcs():
        parent_id, child_id = arc
        parent_name = bn.variable(parent_id).name()
        child_name = bn.variable(child_id).name()
        i, j = int(parent_name), int(child_name)
        adj_matrix[i, j] = 1.0
        logger.debug(f"  Arc {parent_name} -> {child_name}: adj_matrix[{i}, {j}] = 1")
    
    logger.debug(f"Adjacency matrix:\n{adj_matrix}")
    return adj_matrix

def create_parameter_embeddings(bn: gum.BayesNet, adj_matrix: np.ndarray, observed_nodes=None) -> np.ndarray:
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
        if observed_nodes is None or i in observed_nodes:
            # Include CPD data for observed nodes (or all nodes if no mask)
            cpd_data[i, :len(cpd_values)] = cpd_values
        else:
            # Zero out CPD data for unobserved nodes
            cpd_data[i, :] = 0.0
    
    # Combine adjacency and CPD data
    param_embeddings = np.concatenate([adj_matrix, cpd_data], axis=1)
    
    return param_embeddings

def generate_experiment_graph(n_nodes, target_parents=1.0, seed=42):
    """
    Generate a Bayesian Network for progressive imputation experiments.
    
    Args:
        n_nodes: Number of nodes in the graph
        target_parents: Target number of parents per node (O(1) parents)
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (bn, adj_matrix) - BayesNet object and adjacency matrix
    """
    logger.debug(f"Generating graph: {n_nodes} nodes, target_parents={target_parents}, seed={seed}")
    
    # Generate BN structure using direct method
    bn = generate_direct_bn_structure(n_nodes, target_parents, seed)
    adj_matrix = create_adjacency_matrix(bn, n_nodes)
    
    logger.info(f"Generated graph with {bn.size()} nodes, {len(bn.arcs())} edges")
    
    return bn, adj_matrix

def create_parameter_embeddings_with_masking(bn, adj_matrix, observed_nodes):
    """
    Create parameter embeddings with CPD masking for unobserved nodes.
    
    Args:
        bn: BayesNet object
        adj_matrix: Adjacency matrix
        observed_nodes: List of observed node indices
        
    Returns:
        np.ndarray: Parameter embeddings (adjacency + CPD data)
    """
    logger.debug(f"Creating parameter embeddings for {len(observed_nodes)} observed nodes")
    
    # Use the modified function that supports masking
    param_embeddings = create_parameter_embeddings(bn, adj_matrix, observed_nodes)
    
    return param_embeddings