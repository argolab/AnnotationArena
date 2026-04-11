"""
Cross-entropy evaluation utilities for progressive imputation experiments.

Computes cross-entropy H(p_true, q_model) by saving full posterior distributions
over all possible states, enabling Gibbs inequality visualization.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from imputer.training_eval import extract_cpts_for_nodes, DEVICE, SampleTuple

logger = logging.getLogger(__name__)

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False


def compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy H(p) = -Σ p(i) log p(i)"""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def compute_cross_entropy(true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    """Compute cross-entropy H(p,q) = -Σ p(i) log q(i)"""
    eps = 1e-12
    pred_probs = np.clip(pred_probs, eps, 1.0)
    return -np.sum(true_probs * np.log(pred_probs))


def evaluate_neural_cross_entropy(model: torch.nn.Module, test_data: List[SampleTuple],
                                 bn: gum.BayesNet, n_nodes: int, 
                                 max_cpt_size: int) -> Dict[str, Any]:
    """
    Evaluate neural model using cross-entropy H(p_true, q_neural).
    
    Saves full posterior distributions for cross-entropy computation.
    
    Args:
        model: Trained neural imputation model
        test_data: List of test samples with missing values  
        bn: Ground truth BayesNet for computing true posteriors
        n_nodes: Number of nodes
        max_cpt_size: Maximum CPT size for model input
        
    Returns:
        dict: Cross-entropy results with full posterior data
    """
    if not PYAGRUM_AVAILABLE:
        logger.error("PyAgrum required for cross-entropy evaluation")
        return {'mean_cross_entropy': float('inf'), 'cross_entropy_values': []}
    
    logger.debug(f"Computing neural cross-entropy on {len(test_data)} samples")
    
    model.eval()
    cross_entropies = []
    true_entropies = []
    
    # Create true model inference engine
    true_infer = gum.LazyPropagation(bn)
    
    with torch.no_grad():
        for sample_idx, (inputs, structure_info, dimensions, mask, targets, true_states) in enumerate(test_data):
            
            # Create evidence from observed nodes
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
                
            # Extract CPTs for neural model
            observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            
            # Get neural predictions
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)
            structure_info_batch = structure_info.unsqueeze(0).to(DEVICE) 
            cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
            dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)
            
            predictions = model(inputs_batch, structure_info_batch, 
                              cpt_info_batch, dimensions_batch)
            predictions = predictions.squeeze(0).cpu()
            
            # Set evidence for true model inference
            if evidence:
                true_infer.setEvidence(evidence)
                true_infer.makeInference()
            
            # Compute cross-entropy for each unobserved node
            sample_cross_entropy = 0.0
            sample_true_entropy = 0.0
            
            for node in unobserved_nodes:
                node_str = str(node)
                
                # Get true posterior distribution
                if evidence:
                    true_posterior = true_infer.posterior(node_str)
                    true_probs = np.array([
                        true_posterior[{node_str: str(state)}] 
                        for state in range(2)  # Binary nodes
                    ])
                else:
                    # No evidence - use marginal
                    true_marginal = bn.cpt(node_str)
                    true_probs = np.array([
                        true_marginal[{node_str: str(state)}]
                        for state in range(2)
                    ])
                
                # Get neural posterior distribution
                neural_probs = predictions[node, :2].numpy()  # First 2 elements for binary
                
                # Ensure normalization
                true_probs = true_probs / np.sum(true_probs)
                neural_probs = neural_probs / np.sum(neural_probs)
                
                # Compute entropy and cross-entropy for this node
                node_true_entropy = compute_entropy(true_probs)
                node_cross_entropy = compute_cross_entropy(true_probs, neural_probs)
                
                sample_true_entropy += node_true_entropy
                sample_cross_entropy += node_cross_entropy
            
            true_entropies.append(sample_true_entropy)
            cross_entropies.append(sample_cross_entropy)
            true_infer.eraseAllEvidence()
    
    # Clean up inference engine
    try:
        if true_infer:
            true_infer.eraseAllEvidence()
            del true_infer
    except:
        pass
    
    if not cross_entropies:
        logger.warning("No cross-entropy values computed!")
        return {
            'mean_cross_entropy': float('inf'),
            'mean_true_entropy': float('inf'), 
            'cross_entropy_values': [],
            'true_entropy_values': []
        }
    
    results = {
        'mean_cross_entropy': np.mean(cross_entropies),
        'std_cross_entropy': np.std(cross_entropies),
        'mean_true_entropy': np.mean(true_entropies),
        'std_true_entropy': np.std(true_entropies),
        'cross_entropy_values': cross_entropies,
        'true_entropy_values': true_entropies
    }
    
    logger.info(f"Neural cross-entropy: Mean={results['mean_cross_entropy']:.4f}, "
               f"True entropy: {results['mean_true_entropy']:.4f}")
    
    return results


def evaluate_em_cross_entropy(learned_bn: gum.BayesNet, test_data: List[SampleTuple],
                             true_bn: gum.BayesNet, n_nodes: int) -> Dict[str, Any]:
    """
    Evaluate EM model using cross-entropy H(p_true, q_em).
    
    Args:
        learned_bn: EM learned BayesNet
        test_data: List of test samples
        true_bn: Ground truth BayesNet for computing true posteriors  
        n_nodes: Number of nodes
        
    Returns:
        dict: Cross-entropy results with full posterior data
    """
    if not PYAGRUM_AVAILABLE:
        logger.error("PyAgrum required for cross-entropy evaluation")
        return {'mean_cross_entropy': float('inf'), 'cross_entropy_values': []}
    
    logger.debug(f"Computing EM cross-entropy on {len(test_data)} samples")
    
    cross_entropies = []
    true_entropies = []
    
    # Create inference engines
    em_infer = gum.LazyPropagation(learned_bn)
    true_infer = gum.LazyPropagation(true_bn)
    
    for sample_idx, (inputs, embeddings, dimensions, mask, targets, true_states) in enumerate(test_data):
        
        # Create evidence from observed nodes
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
            
        # Set evidence for both models
        if evidence:
            em_infer.setEvidence(evidence)
            em_infer.makeInference()
            true_infer.setEvidence(evidence)
            true_infer.makeInference()
        
        # Compute cross-entropy for each unobserved node
        sample_cross_entropy = 0.0
        sample_true_entropy = 0.0
        
        for node in unobserved_nodes:
            node_str = str(node)
            
            # Get true posterior distribution
            if evidence:
                true_posterior = true_infer.posterior(node_str)
                true_probs = np.array([
                    true_posterior[{node_str: str(state)}] 
                    for state in range(2)  # Binary nodes
                ])
            else:
                # No evidence - use marginal
                true_marginal = true_bn.cpt(node_str)
                true_probs = np.array([
                    true_marginal[{node_str: str(state)}]
                    for state in range(2)
                ])
            
            # Get EM posterior distribution
            if evidence:
                em_posterior = em_infer.posterior(node_str)
                em_probs = np.array([
                    em_posterior[{node_str: str(state)}] 
                    for state in range(2)
                ])
            else:
                # No evidence - use EM marginal
                em_marginal = learned_bn.cpt(node_str)
                em_probs = np.array([
                    em_marginal[{node_str: str(state)}]
                    for state in range(2)
                ])
            
            # Ensure normalization
            true_probs = true_probs / np.sum(true_probs)
            em_probs = em_probs / np.sum(em_probs)
            
            # Compute entropy and cross-entropy for this node
            node_true_entropy = compute_entropy(true_probs)
            node_cross_entropy = compute_cross_entropy(true_probs, em_probs)
            
            sample_true_entropy += node_true_entropy
            sample_cross_entropy += node_cross_entropy
        
        true_entropies.append(sample_true_entropy)
        cross_entropies.append(sample_cross_entropy)
        
        em_infer.eraseAllEvidence()
        true_infer.eraseAllEvidence()
    
    # Clean up inference engines
    try:
        if em_infer:
            em_infer.eraseAllEvidence()
            del em_infer
        if true_infer:
            true_infer.eraseAllEvidence()
            del true_infer
    except:
        pass
    
    if not cross_entropies:
        logger.warning("No EM cross-entropy values computed!")
        return {
            'mean_cross_entropy': float('inf'),
            'mean_true_entropy': float('inf'),
            'cross_entropy_values': [],
            'true_entropy_values': []
        }
    
    results = {
        'mean_cross_entropy': np.mean(cross_entropies),
        'std_cross_entropy': np.std(cross_entropies),
        'mean_true_entropy': np.mean(true_entropies),
        'std_true_entropy': np.std(true_entropies),
        'cross_entropy_values': cross_entropies,
        'true_entropy_values': true_entropies
    }
    
    logger.info(f"EM cross-entropy: Mean={results['mean_cross_entropy']:.4f}, "
               f"True entropy: {results['mean_true_entropy']:.4f}")
    
    return results