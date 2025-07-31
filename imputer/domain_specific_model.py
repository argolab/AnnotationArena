"""
Domain-specific Bayesian Network model with Gibbs Sampling + MLE.

This module provides functionality to learn BN parameters from incomplete data
using Gibbs sampling for missing value imputation followed by MLE.
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
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
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
                              n_samples: int = 500) -> BayesianNetwork:
    """
    Learn BN parameters using Gibbs sampling + MLE for missing data.
    
    Args:
        bn_structure: BayesianNetwork structure (no CPDs)
        training_data: DataFrame with NaN for missing values
        n_states: Number of states per variable
        n_samples: Number of Gibbs samples for imputation
        
    Returns:
        BayesianNetwork with learned CPDs
    """
    print(f"Learning domain-specific model with Gibbs sampling + MLE...")
    print(f"Training data: {len(training_data)} samples, {training_data.isnull().sum().sum()} missing values")
    
    # Step 1: Create initial model with uniform priors for Gibbs sampling
    initial_model = create_uniform_model(bn_structure, n_states)
    
    # Step 2: Use Gibbs sampling to complete missing data
    print("Running Gibbs sampling for missing data imputation...")
    completed_data = impute_missing_with_gibbs(initial_model, training_data, n_samples)
    
    # Step 3: Learn final parameters with MLE on completed data
    print("Learning final parameters with MLE...")
    mle = MaximumLikelihoodEstimator(bn_structure, completed_data)
    final_cpds = mle.get_parameters()
    
    # Create final model
    final_model = BayesianNetwork(bn_structure.edges())
    all_nodes = sorted(list(bn_structure.nodes()))
    for node in all_nodes:
        if node not in final_model.nodes():
            final_model.add_node(node)
    
    final_model.add_cpds(*final_cpds)
    
    print(f"Model learned with {len(final_cpds)} CPDs")
    return final_model


def create_uniform_model(bn_structure: BayesianNetwork, n_states: int) -> BayesianNetwork:
    """Create initial model with uniform CPDs for Gibbs sampling."""
    cpds = []
    nodes = sorted(bn_structure.nodes())
    
    for node in nodes:
        parents = list(bn_structure.get_parents(node))
        
        if not parents:
            # Root node - uniform distribution
            values = np.ones((n_states, 1)) / n_states
        else:
            # Child node - uniform conditional distribution
            n_parent_configs = n_states ** len(parents)
            values = np.ones((n_states, n_parent_configs)) / n_states
        
        cpd = TabularCPD(
            variable=node,
            variable_card=n_states,
            values=values,
            evidence=parents,
            evidence_card=[n_states] * len(parents) if parents else []
        )
        cpds.append(cpd)
    
    model = BayesianNetwork(bn_structure.edges())
    for node in nodes:
        if node not in model.nodes():
            model.add_node(node)
    
    model.add_cpds(*cpds)
    return model


def impute_missing_with_gibbs(model: BayesianNetwork, 
                             data: pd.DataFrame, 
                             n_samples: int) -> pd.DataFrame:
    """Use Gibbs sampling to impute missing values."""
    print(f"Imputing {data.isnull().sum().sum()} missing values with {n_samples} Gibbs samples...")
    
    # Create Gibbs sampler
    gibbs = GibbsSampling(model)
    
    completed_data = data.copy()
    
    # For each row with missing values, use Gibbs sampling
    for idx, row in data.iterrows():
        if row.isnull().any():
            # Create evidence from observed values
            evidence = {}
            for col in data.columns:
                if not pd.isna(row[col]):
                    evidence[col] = int(row[col])
            
            if evidence:  # If we have some evidence
                # Generate samples for missing variables
                missing_vars = [col for col in data.columns if pd.isna(row[col])]
                
                try:
                    # Generate samples
                    samples = gibbs.sample(size=n_samples, evidence=evidence)
                    
                    # Take mean of samples for each missing variable
                    for var in missing_vars:
                        if var in samples.columns:
                            # Use mode (most frequent value) for discrete variables
                            imputed_value = samples[var].mode().iloc[0]
                            completed_data.loc[idx, var] = imputed_value
                        
                except Exception as e:
                    print(f"Gibbs sampling failed for row {idx}: {e}")
                    # Fallback: use prior probabilities
                    for var in missing_vars:
                        completed_data.loc[idx, var] = np.random.randint(0, 2)  # Random binary
    
    print(f"Imputation complete. Missing values after: {completed_data.isnull().sum().sum()}")
    return completed_data




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