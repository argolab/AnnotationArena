"""
Domain-specific Bayesian Network model using pyAgrum.

This module provides functionality to learn BN parameters from incomplete data
using pyAgrum's built-in EM algorithm for graph imputation tasks.

Author: Prabhav Singh
"""

import numpy as np
import pandas as pd
import warnings
import torch
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings('ignore')

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
    print("pyAgrum available - using for all BN operations")
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError("pyAgrum is required for this implementation")


def convert_training_data_for_pyagrum(train_data: List[Tuple], n_nodes: int) -> pd.DataFrame:
    """
    Convert masked training data to pyAgrum format with NaN for missing values.
    
    Args:
        train_data: List of (inputs, embeddings, dimensions, mask, targets)
        n_nodes: Number of nodes in the graph
        
    Returns:
        pandas DataFrame with columns for each node, NaN for unobserved values
    """
    print(f"Converting {len(train_data)} samples for pyAgrum...")
    samples = []
    
    for i, (inputs, embeddings, dimensions, mask, targets) in enumerate(train_data):
        sample = {}
        observed_count = 0
        
        for node in range(n_nodes):
            if mask[node] == 0:  # Observed node
                # Extract state from one-hot encoding in inputs
                state = torch.argmax(inputs[node, 1:]).item()
                sample[str(node)] = str(state)  # String values for pyAgrum
                observed_count += 1
                
                # Debug first few samples
                if i < 3:
                    print(f"  Sample {i}, Node {node}: observed state = {state}")
            else:  # Unobserved node
                sample[str(node)] = np.nan
                
        samples.append(sample)
        
        # Debug observation ratio
        if i < 3:
            print(f"  Sample {i}: {observed_count}/{n_nodes} nodes observed ({observed_count/n_nodes:.1%})")
    
    df = pd.DataFrame(samples)
    
    # Convert to categorical for pyAgrum
    for col in df.columns:
        df[col] = df[col].astype('category')
    
    print(f"Created DataFrame with columns: {list(df.columns)}")
    print(f"Sample data shape: {df.shape}")
    print(f"Missing values per variable: {df.isnull().sum().to_dict()}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    return df


def create_pyagrum_bn_from_adjacency(adj_matrix: np.ndarray) -> gum.BayesNet:
    """
    Create pyAgrum BayesNet structure from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix where adj_matrix[i,j] = 1 means edge i -> j
        
    Returns:
        pyAgrum BayesNet with structure
    """
    n_nodes = adj_matrix.shape[0]
    print(f"Creating pyAgrum BN from {n_nodes}x{n_nodes} adjacency matrix")
    
    # Create pyAgrum BN
    bn = gum.BayesNet("DomainSpecificBN")
    
    # Add binary variables with string states "0", "1"
    for i in range(n_nodes):
        bn.add(gum.LabelizedVariable(str(i), str(i), ["0", "1"]))
        print(f"  Added variable {i}")
    
    # Add arcs from adjacency matrix
    edges_added = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                bn.addArc(str(i), str(j))
                edges_added += 1
                print(f"  Added arc {i} -> {j}")
    
    # Generate initial CPTs
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    print(f"Created pyAgrum BN: {bn.size()} nodes, {edges_added} arcs")
    return bn


def learn_with_pyagrum_em(bn: gum.BayesNet, 
                         training_data: pd.DataFrame,
                         max_iter: int = 100,
                         epsilon: float = 1e-3) -> gum.BayesNet:
    """Learn BN parameters using pyAgrum EM."""
    print(f"=== LEARNING WITH PYAGRUM EM ===")
    print(f"Training data: {len(training_data)} samples, {training_data.isnull().sum().sum()} missing values")
    print(f"EM config: max_iter={max_iter}, epsilon={epsilon}")
    
    # Create learner and configure EM
    learner = gum.BNLearner(training_data)
    learner.useEM(epsilon)
    learner.setMaxIter(max_iter)
    
    print(f"EM enabled: {learner.isUsingEM()}")
    print(f"Max iterations: {learner.EMMaxIter()}")
    print(f"Epsilon: {learner.EMEpsilon()}")
    
    # Learn parameters
    print("Running pyAgrum EM...")
    learned_bn = learner.learnParameters(bn)
    
    print(f"EM completed in {learner.EMnbrIterations()} iterations")
    print(f"Final EM state: {learner.EMStateMessage()}")
    
    return learned_bn


def learn_domain_specific_model(adj_matrix: np.ndarray, 
                              training_data: pd.DataFrame,
                              n_states: int = 2,
                              max_iter: int = 100,
                              epsilon: float = 1e-3) -> gum.BayesNet:
    """
    Main interface - learn BN using pyAgrum EM.
    
    Args:
        adj_matrix: Adjacency matrix defining BN structure
        training_data: DataFrame with missing values (pyAgrum format)
        n_states: Number of states per variable (must be 2)
        max_iter: Maximum EM iterations
        epsilon: Convergence threshold
        
    Returns:
        Learned pyAgrum BayesNet
    """
    print("🔥 USING CLEAN PYAGRUM IMPLEMENTATION! 🔥")
    
    # Create BN structure
    bn = create_pyagrum_bn_from_adjacency(adj_matrix)
    
    # Learn with EM
    learned_bn = learn_with_pyagrum_em(bn, training_data, max_iter, epsilon)
    
    return learned_bn


def evaluate_domain_specific_model(learned_bn: gum.BayesNet, 
                                 test_data: List[Tuple],
                                 n_nodes: int,
                                 n_states: int = 2) -> Dict:
    """
    Evaluate learned pyAgrum BN on test set and compute KL divergence.
    
    Args:
        learned_bn: Trained pyAgrum BayesNet
        test_data: List of (inputs, embeddings, dimensions, mask, targets)
        n_nodes: Number of nodes
        n_states: Number of states per node
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating pyAgrum domain-specific model...")
    
    # Create inference engine
    infer = gum.LazyPropagation(learned_bn)
    
    kl_divergences = []
    prediction_errors = []
    failed_inferences = 0
    
    for inputs, embeddings, dimensions, mask, targets in test_data:
        # Create evidence from observed nodes
        evidence = {}
        unobserved_nodes = []
        
        for node in range(n_nodes):
            if mask[node] == 0:  # Observed
                state = torch.argmax(inputs[node, 1:]).item()
                evidence[str(node)] = str(state)  # String format for pyAgrum
            else:  # Unobserved
                unobserved_nodes.append(node)
        
        if not unobserved_nodes:
            continue
            
        # Get predictions for unobserved nodes
        for node in unobserved_nodes:
            try:
                node_str = str(node)
                
                # Set evidence and run inference
                if evidence:
                    infer.setEvidence(evidence)
                    infer.makeInference()
                    posterior = infer.posterior(node_str)
                    # Extract probabilities in order ["0", "1"]
                    pred_probs = np.array([posterior[{node_str: "0"}], posterior[{node_str: "1"}]])
                else:
                    # No evidence, use uniform
                    pred_probs = np.ones(n_states) / n_states
                
                # Debug first few samples
                if len(kl_divergences) < 3:
                    print(f"DEBUG: Node {node_str}, evidence: {evidence}")
                    print(f"DEBUG: Predicted probs: {pred_probs}")
                
                # Ensure probabilities are valid
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    pred_probs = np.ones(n_states) / n_states
                else:
                    pred_probs = pred_probs / np.sum(pred_probs)
                
                # Get ground truth
                true_probs = targets[node].numpy()
                
                # Ensure ground truth is valid
                if np.any(np.isnan(true_probs)) or np.sum(true_probs) == 0:
                    failed_inferences += 1
                    continue
                
                # Compute KL divergence: KL(true || pred)
                kl = 0.0
                for state in range(n_states):
                    if true_probs[state] > 1e-10:  # Avoid log(0)
                        kl += true_probs[state] * np.log(
                            (true_probs[state] + 1e-10) / (pred_probs[state] + 1e-10)
                        )
                
                # Sanity check on KL
                if np.isnan(kl) or np.isinf(kl) or kl < 0:
                    failed_inferences += 1
                    continue
                
                kl_divergences.append(kl)
                
                # Prediction error (L2 norm)
                error = np.linalg.norm(pred_probs - true_probs)
                prediction_errors.append(error)
                
                # Clear evidence for next inference
                infer.eraseAllEvidence()
                
            except Exception as e:
                if len(kl_divergences) < 5:
                    print(f"Inference failed for node {node}: {str(e)[:100]}")
                failed_inferences += 1
                continue
    
    if not kl_divergences:
        print("No successful inferences!")
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
    
    print(f"pyAgrum evaluation: Mean KL = {results['mean_kl']:.4f}, "
          f"Failed rate = {results['failed_rate']:.2%}")
    
    return results


def extract_adjacency_from_embeddings(param_embeddings: np.ndarray, n_nodes: int) -> np.ndarray:
    """Extract adjacency matrix from parameter embeddings."""
    print(f"Extracting adjacency from embeddings shape: {param_embeddings.shape}")
    print(f"Expecting adjacency matrix in first {n_nodes} columns")
    
    adj_matrix = param_embeddings[:, :n_nodes]
    print(f"Extracted adjacency matrix shape: {adj_matrix.shape}")
    print(f"Extracted adjacency matrix:\n{adj_matrix}")
    
    return adj_matrix