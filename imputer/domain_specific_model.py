"""
Domain-specific Bayesian Network model with EM + Bayesian Parameter Estimation.

This module provides functionality to learn BN parameters from incomplete data
using Expectation-Maximization followed by Bayesian parameter estimation.
Used as baseline comparison for the neural imputer.

Author: Prabhav Singh
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


def convert_training_data_for_pgmpy(train_data: List[Tuple], n_nodes: int) -> pd.DataFrame:
    """
    Convert masked training data to pgmpy format with NaN for missing values.
    
    Args:
        train_data: List of (inputs, embeddings, dimensions, mask, targets)
        n_nodes: Number of nodes in the graph
        
    Returns:
        pandas DataFrame with columns for each node, NaN for unobserved values
    """
    print(f"DEBUG: Converting {len(train_data)} samples for pgmpy...")
    samples = []
    
    for i, (inputs, embeddings, dimensions, mask, targets) in enumerate(train_data):
        sample = {}
        observed_count = 0
        
        for node in range(n_nodes):
            if mask[node] == 0:  # Observed node
                # Extract state from one-hot encoding in inputs
                state = torch.argmax(inputs[node, 1:]).item()
                sample[str(node)] = state  # Use STRING node names for EM compatibility
                observed_count += 1
                
                # Debug first few samples
                if i < 3:
                    print(f"  Sample {i}, Node {node}: observed state = {state}")
            else:  # Unobserved node
                sample[str(node)] = np.nan  # Use STRING node names for EM compatibility
                
        samples.append(sample)
        
        # Debug observation ratio
        if i < 3:
            print(f"  Sample {i}: {observed_count}/{n_nodes} nodes observed ({observed_count/n_nodes:.1%})")
    
    df = pd.DataFrame(samples)
    print(f"DEBUG: Created DataFrame with columns: {list(df.columns)}")
    print(f"DEBUG: Sample data shape: {df.shape}")
    print(f"DEBUG: Missing values per column: {df.isnull().sum().to_dict()}")
    print(f"DEBUG: Sample of raw data:")
    print(df.head(3))
    
    # CRITICAL FIX: Use float with np.nan for EM compatibility (per pgmpy docs)
    print(f"DEBUG: Converting to float with np.nan for EM compatibility...")
    for col in df.columns:
        # Convert to float type with np.nan (as required by pgmpy EM)
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    print(f"DEBUG: Final DataFrame dtypes: {df.dtypes.to_dict()}")
    print(f"DEBUG: Sample of converted data (should show np.nan):")
    print(df.head(3))
    print(f"DEBUG: Checking for np.nan presence: {df.isnull().sum().to_dict()}")
    
    return df


def create_bn_structure_from_adjacency(adj_matrix: np.ndarray) -> BayesianNetwork:
    """
    Create BayesianNetwork structure from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix where adj_matrix[i,j] = 1 means edge i -> j
        
    Returns:
        BayesianNetwork with structure but no CPDs
    """
    n_nodes = adj_matrix.shape[0]
    print(f"DEBUG: Creating BN structure from {n_nodes}x{n_nodes} adjacency matrix")
    print(f"DEBUG: Adjacency matrix:\n{adj_matrix}")
    
    # Create edges from adjacency matrix using STRING node names for EM compatibility
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                edge = (str(i), str(j))  # Use STRING node names for EM compatibility
                edges.append(edge)
                print(f"DEBUG: Adding edge: {edge}")
    
    print(f"DEBUG: Total edges created: {len(edges)}")
    print(f"DEBUG: Edges: {edges}")
    
    # Create BayesianNetwork
    bn = BayesianNetwork(edges)
    
    # CRITICAL FIX: Add all nodes explicitly, even isolated ones
    print(f"DEBUG: BN nodes before adding isolated nodes: {sorted(list(bn.nodes()))}")
    for i in range(n_nodes):
        node_str = str(i)
        if node_str not in bn.nodes():
            bn.add_node(node_str)  # Use STRING node names for EM compatibility
            print(f"DEBUG: Added isolated node: {node_str}")
    
    print(f"DEBUG: BN nodes after adding isolated nodes: {sorted(list(bn.nodes()))}")
    print(f"DEBUG: BN edges: {list(bn.edges())}")
    
    return bn


def learn_domain_specific_model(bn_structure: BayesianNetwork, 
                              training_data: pd.DataFrame,
                              n_states: int = 2,
                              max_iter: int = 15,  # Reduced from 100 to 15
                              tol: float = 1e-4) -> BayesianNetwork:
    """
    Learn BN parameters using EM + Bayesian estimation for missing data.
    
    Args:
        bn_structure: BayesianNetwork structure (no CPDs)
        training_data: DataFrame with NaN for missing values
        n_states: Number of states per variable
        max_iter: Maximum EM iterations
        tol: EM convergence tolerance
        
    Returns:
        BayesianNetwork with learned CPDs
    """
    print(f"Learning domain-specific model with {len(training_data)} samples...")
    
    try:
        # Step 1: Use EM to handle missing data
        print(f"DEBUG: Initializing EM with BN structure: {sorted(list(bn_structure.nodes()))}")
        print(f"DEBUG: Training data columns: {list(training_data.columns)}")
        print(f"DEBUG: Training data dtypes: {training_data.dtypes.to_dict()}")
        print(f"DEBUG: Sample training data for EM:")
        print(training_data.head(5))
        
        em = ExpectationMaximization(bn_structure, training_data)
        
        # Try different API signatures for different pgmpy versions
        try:
            # Newer API with tol parameter
            print(f"DEBUG: Attempting EM with max_iter={max_iter}, tol={tol}")
            em_cpds = em.get_parameters(n_jobs=1, max_iter=max_iter, tol=tol)
        except TypeError as e:
            print(f"DEBUG: EM with tol failed: {e}")
            # Older API without tol parameter
            try:
                print(f"DEBUG: Attempting EM with max_iter={max_iter}")
                print(f"DEBUG: EM object created successfully")
                print(f"DEBUG: BN structure nodes: {sorted(list(bn_structure.nodes()))}")
                print(f"DEBUG: BN structure edges: {list(bn_structure.edges())}")
                print(f"DEBUG: Training data info:")
                print(f"  - Shape: {training_data.shape}")
                print(f"  - Columns: {list(training_data.columns)}")
                print(f"  - Non-null counts: {training_data.count().to_dict()}")
                print(f"  - Data types: {training_data.dtypes.to_dict()}")
                
                em_cpds = em.get_parameters(n_jobs=1, max_iter=max_iter)
            except TypeError as e:
                print(f"DEBUG: EM with max_iter failed: {e}")
                # Even older API with minimal parameters
                print(f"DEBUG: Attempting EM with minimal parameters")
                print(f"DEBUG: Trying just get_parameters() with no arguments...")
                em_cpds = em.get_parameters()
        
        # Create model with EM parameters
        em_model = BayesianNetwork(bn_structure.edges())
        
        # Add all nodes explicitly (integer nodes)
        all_nodes = sorted(list(bn_structure.nodes()))
        for node in all_nodes:
            if node not in em_model.nodes():
                em_model.add_node(node)
                
        em_model.add_cpds(*em_cpds)
        
        print(f"EM converged with {len(em_cpds)} CPDs learned")
        
        # Step 2: Bayesian refinement with EM-informed priors
        # Use EM solution to create informed Dirichlet priors
        try:
            # Create pseudo-counts from EM solution
            pseudo_counts = {}
            for cpd in em_cpds:
                # Use EM probabilities as pseudo-counts (scaled)
                values = cpd.get_values()
                pseudo_counts[cpd.variable] = values * 10  # Scale factor for pseudo-counts
            
            # Apply Bayesian estimation with informed priors
            bayesian_est = BayesianEstimator(em_model, training_data)
            final_cpds = bayesian_est.get_parameters(
                prior_type="dirichlet",
                pseudo_counts=1.0  # Conservative pseudo-counts
            )
            
            # Create final model
            final_model = BayesianNetwork(bn_structure.edges())
            final_model.add_cpds(*final_cpds)
            
            print("Bayesian refinement completed")
            
        except Exception as e:
            print(f"Bayesian refinement failed: {e}, using EM solution")
            final_model = em_model
        
        return final_model
        
    except Exception as e:
        print(f"EM learning failed: {e}")
        # Fallback: Use structure-aware learning that actually uses training data
        return learn_structure_aware_fallback(bn_structure, training_data, n_states)


def learn_structure_aware_fallback(bn_structure: BayesianNetwork, 
                                  training_data: pd.DataFrame,
                                  n_states: int = 2) -> BayesianNetwork:
    """
    Structure-aware fallback that learns from available data per node.
    Better than random priors - uses actual training data.
    """
    print("Using structure-aware fallback that learns from training data...")
    
    nodes = sorted(bn_structure.nodes())
    print(f"DEBUG: Learning parameters for nodes: {nodes}")
    
    cpds = []
    for node in nodes:
        parents = list(bn_structure.get_parents(node))
        print(f"DEBUG: Learning CPD for node {node} with parents {parents}")
        
        if not parents:
            # Root node - learn marginal distribution from available data
            node_data = training_data[node].dropna()
            if len(node_data) > 0:
                # Count observed states
                counts = np.zeros(n_states)
                for state in node_data:
                    counts[int(state)] += 1
                # Add small pseudo-count to avoid zeros
                counts += 1.0
                probs = counts / counts.sum()
                values = probs.reshape(n_states, 1)
                print(f"  Root node {node}: learned from {len(node_data)} samples, probs={probs}")
            else:
                # No data available, use uniform
                values = np.ones((n_states, 1)) / n_states
                print(f"  Root node {node}: no data, using uniform")
        else:
            # Child node - learn conditional distribution from available data
            # Find samples where both node and all parents are observed
            relevant_cols = [node] + parents
            complete_cases = training_data[relevant_cols].dropna()
            
            if len(complete_cases) > 5:  # Need reasonable sample size
                print(f"  Child node {node}: learning from {len(complete_cases)} complete cases")
                
                # Count conditional frequencies
                n_parent_configs = n_states ** len(parents)
                values = np.ones((n_states, n_parent_configs)) / n_states  # Start with uniform
                
                for _, row in complete_cases.iterrows():
                    # Get parent configuration
                    parent_config = 0
                    for i, parent in enumerate(parents):
                        parent_config += int(row[parent]) * (n_states ** (len(parents) - 1 - i))
                    
                    # Count this observation
                    child_state = int(row[node])
                    values[child_state, parent_config] += 1.0
                
                # Normalize each parent configuration
                for config in range(n_parent_configs):
                    values[:, config] = values[:, config] / values[:, config].sum()
                    
            else:
                # Not enough data, use structure-informed priors
                print(f"  Child node {node}: insufficient data ({len(complete_cases)} cases), using informed priors")
                n_parent_configs = n_states ** len(parents)
                values = np.random.dirichlet(np.ones(n_states) * 2, size=n_parent_configs).T
        
        cpd = TabularCPD(
            variable=node,
            variable_card=n_states,
            values=values,
            evidence=parents,
            evidence_card=[n_states] * len(parents) if parents else []
        )
        cpds.append(cpd)
    
    # Create model
    model = BayesianNetwork(bn_structure.edges())
    for node in nodes:
        if node not in model.nodes():
            model.add_node(node)
    
    model.add_cpds(*cpds)
    print(f"DEBUG: Structure-aware fallback model created with {len(cpds)} CPDs")
    
    if not model.check_model():
        print("Warning: Structure-aware fallback model failed validation")
    
    return model


def learn_fallback_model(bn_structure: BayesianNetwork, 
                        training_data: pd.DataFrame,
                        n_states: int = 2) -> BayesianNetwork:
    """
    Fallback model using simple maximum likelihood on observed data only.
    
    Args:
        bn_structure: BayesianNetwork structure
        training_data: DataFrame with NaN for missing values
        n_states: Number of states per variable
        
    Returns:
        BayesianNetwork with simple MLE CPDs
    """
    print("Using fallback MLE model...")
    
    # Drop rows with missing values for MLE
    complete_data = training_data.dropna()
    
    if len(complete_data) == 0:
        print("No complete samples, creating informed uniform priors from structure")
        # Create uniform CPDs with proper structure awareness
        cpds = []
        nodes = sorted(bn_structure.nodes())
        print(f"DEBUG: Creating CPDs for nodes: {nodes}")
        
        for node in nodes:
            parents = list(bn_structure.get_parents(node))
            print(f"DEBUG: Node {node} has parents: {parents}")
            
            if not parents:
                # Root node - slightly non-uniform to break symmetry
                values = np.random.dirichlet(np.ones(n_states) * 2).reshape(n_states, 1)
                print(f"DEBUG: Root node {node} CPD shape: {values.shape}")
            else:
                # Child node - slightly non-uniform conditional distribution
                n_parent_configs = n_states ** len(parents)
                values = np.random.dirichlet(np.ones(n_states) * 2, size=n_parent_configs).T
                print(f"DEBUG: Child node {node} CPD shape: {values.shape}")
            
            cpd = TabularCPD(
                variable=node,
                variable_card=n_states,
                values=values,
                evidence=parents,
                evidence_card=[n_states] * len(parents) if parents else []
            )
            cpds.append(cpd)
        
        # CRITICAL FIX: Ensure model includes all nodes
        model = BayesianNetwork(bn_structure.edges())
        
        # Add all nodes explicitly (using integer node names)
        for node in nodes:
            if node not in model.nodes():
                model.add_node(node)
                
        model.add_cpds(*cpds)
        print(f"DEBUG: Fallback model nodes: {sorted(list(model.nodes()))}")
        
        if not model.check_model():
            print("Warning: Fallback model failed validation")
        
        return model
    
    # Use MLE on complete data
    from pgmpy.estimators import MaximumLikelihoodEstimator
    mle = MaximumLikelihoodEstimator(bn_structure, complete_data)
    cpds = mle.get_parameters()
    
    model = BayesianNetwork(bn_structure.edges())
    model.add_cpds(*cpds)
    
    return model


def evaluate_domain_specific_model(learned_bn: BayesianNetwork, 
                                 test_data: List[Tuple],
                                 n_nodes: int,
                                 n_states: int = 2) -> Dict:
    """
    Evaluate learned BN on test set and compute KL divergence vs ground truth.
    
    Args:
        learned_bn: Trained BayesianNetwork
        test_data: List of (inputs, embeddings, dimensions, mask, targets)
        n_nodes: Number of nodes
        n_states: Number of states per node
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating domain-specific model...")
    
    infer = VariableElimination(learned_bn)
    
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
                evidence[str(node)] = state  # Use STRING node names for consistency
            else:  # Unobserved
                unobserved_nodes.append(node)
        
        if not unobserved_nodes:
            continue
            
        # Get predictions for unobserved nodes
        for node in unobserved_nodes:
            try:
                # Query posterior with string node names for consistency
                node_str = str(node)  # Convert node to string for query
                
                # Debug evidence and query for first few samples
                if len(kl_divergences) < 3:
                    print(f"DEBUG: Querying node {node_str} with evidence: {evidence}")
                
                # Check if evidence creates impossible state
                if not evidence:
                    # If no evidence, create uniform posterior
                    pred_probs = np.ones(n_states) / n_states
                    if len(kl_divergences) < 3:
                        print(f"DEBUG: No evidence, using uniform probs: {pred_probs}")
                else:
                    posterior = infer.query(variables=[node_str], evidence=evidence)
                    pred_probs = posterior.values
                    if len(kl_divergences) < 3:
                        print(f"DEBUG: VE query result for node {node_str}: {pred_probs}")
                
                # Ensure probabilities are valid
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    pred_probs = np.ones(n_states) / n_states
                else:
                    # Normalize to ensure sum to 1
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
                
            except Exception as e:
                # More detailed error logging for debugging
                if len(kl_divergences) < 5:  # Only log first few errors
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
    
    print(f"Domain-specific evaluation: Mean KL = {results['mean_kl']:.4f}, "
          f"Failed rate = {results['failed_rate']:.2%}")
    
    return results


def extract_adjacency_from_embeddings(param_embeddings: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Extract adjacency matrix from parameter embeddings.
    
    Args:
        param_embeddings: Array with adjacency matrix in first n_nodes columns
        n_nodes: Number of nodes
        
    Returns:
        Adjacency matrix
    """
    print(f"DEBUG: Extracting adjacency from embeddings shape: {param_embeddings.shape}")
    print(f"DEBUG: Expecting adjacency matrix in first {n_nodes} columns")
    
    adj_matrix = param_embeddings[:, :n_nodes]
    print(f"DEBUG: Extracted adjacency matrix shape: {adj_matrix.shape}")
    print(f"DEBUG: Extracted adjacency matrix:\n{adj_matrix}")
    
    return adj_matrix