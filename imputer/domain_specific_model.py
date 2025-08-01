"""
Domain-specific Bayesian Network model with Gibbs Sampling + MLE.

This module provides functionality to learn BN parameters from incomplete data
using Gibbs sampling for missing value imputation followed by MLE.
Used as baseline comparison for the neural imputer.

Author: Prabhav Singh
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
from typing import List, Tuple, Dict, Optional
import warnings
import traceback
warnings.filterwarnings('ignore')

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
from pgmpy.factors.discrete import TabularCPD

# Your pgmpy code that generates the warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Probability values don't exactly sum to 1")
    # Your pgmpy operations here, e.g., model learning, inference, etc.

import logging

logging.getLogger('pgmpy').setLevel(logging.ERROR)


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
    
    # CRITICAL FIX: Force pure integer data (no floats) for pgmpy 1.0.0 EM
    print(f"DEBUG: Converting to pure integer data for pgmpy 1.0.0 EM...")
    for col in df.columns:
        # First convert to object type to allow mixed int/NaN
        df[col] = df[col].astype('object')
        # Then convert only non-NaN values to pure integers
        observed_mask = df[col].notna()
        if observed_mask.any():
            # Force conversion to Python int (not numpy float or int64)
            df.loc[observed_mask, col] = [int(x) for x in df.loc[observed_mask, col]]
    
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





def learn_domain_specific_model_simple(bn_structure: BayesianNetwork,
                                      training_data: pd.DataFrame,
                                      n_states: int = 2,
                                      max_iter: int = 50) -> BayesianNetwork:
    """
    Custom EM implementation with multiple imputation for missing data.
    
    Uses principled approach:
    1. E-step: Multiple imputation via Gibbs sampling
    2. M-step: Parameter learning via MLE on completed datasets
    3. Parameter pooling: Rubin's rules for combining estimates
    4. Convergence: Log-likelihood based stopping criterion
    """
    print(f"=== CUSTOM EM WITH MULTIPLE IMPUTATION ===")
    print(f"Training data: {len(training_data)} samples, {training_data.isnull().sum().sum()} missing values")
    print(f"EM config: {max_iter} iterations, {n_states} states per variable")
    
    all_nodes = sorted(bn_structure.nodes())
    n_missing = training_data.isnull().sum().sum()
    
    if n_missing == 0:
        print("No missing data - using direct MLE")
        return _learn_with_mle(bn_structure, training_data)
    
    # Initialize with uniform priors for Gibbs sampling
    current_model = _initialize_uniform_model(bn_structure, n_states)
    prev_log_likelihood = float('-inf')
    
    print(f"Starting EM iterations...")
    
    for iteration in range(max_iter):
        print(f"\n--- EM Iteration {iteration + 1}/{max_iter} ---")
        
        # E-STEP: Multiple imputation via Gibbs sampling
        print("E-step: Generating multiple imputations...")
        completed_datasets = _multiple_imputation_gibbs(
            training_data, current_model, 
            n_imputations=5, n_samples=100, burn_in=20
        )
        
        # M-STEP: Parameter learning on completed datasets
        print("M-step: Learning parameters via MLE...")
        new_model = _parameter_learning_mle(bn_structure, completed_datasets)
        
        # CONVERGENCE CHECK: Log-likelihood improvement
        current_ll = _compute_log_likelihood(training_data, new_model)
        ll_improvement = current_ll - prev_log_likelihood
        
        print(f"Log-likelihood: {current_ll:.4f} (improvement: {ll_improvement:.4f})")
        
        if ll_improvement < 0.01:
            print(f"Converged after {iteration + 1} iterations!")
            break
            
        current_model = new_model
        prev_log_likelihood = current_ll
    
    print(f"EM completed. Final model has {len(current_model.get_cpds())} CPDs")
    return current_model


def _learn_with_mle(bn_structure: BayesianNetwork, complete_data: pd.DataFrame) -> BayesianNetwork:
    """Learn parameters using MLE on complete data."""
    from pgmpy.estimators import MaximumLikelihoodEstimator
    
    mle = MaximumLikelihoodEstimator(bn_structure, complete_data)
    learned_cpds = mle.get_parameters()
    
    # Create final model
    model = BayesianNetwork(bn_structure.edges())
    for node in sorted(bn_structure.nodes()):
        if node not in model.nodes():
            model.add_node(node)
    model.add_cpds(*learned_cpds)
    
    return model


def _initialize_uniform_model(bn_structure: BayesianNetwork, n_states: int) -> BayesianNetwork:
    """Create initial model with uniform CPDs."""
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


def _multiple_imputation_gibbs(training_data: pd.DataFrame, 
                              current_model: BayesianNetwork,
                              n_imputations: int = 5,
                              n_samples: int = 100,
                              burn_in: int = 20) -> List[pd.DataFrame]:
    """Generate multiple completed datasets using Gibbs sampling."""
    completed_datasets = []
    
    for m in range(n_imputations):
        completed_data = _gibbs_imputation(training_data, current_model, n_samples, burn_in)
        completed_datasets.append(completed_data)
    
    print(f"Generated {len(completed_datasets)} completed datasets")
    return completed_datasets


def _gibbs_imputation(data: pd.DataFrame, 
                     model: BayesianNetwork,
                     n_samples: int,
                     burn_in: int) -> pd.DataFrame:
    """Complete missing values using Gibbs sampling."""
    from pgmpy.sampling import GibbsSampling
    
    completed_data = data.copy()
    gibbs = GibbsSampling(model)
    
    # Find rows with missing values
    incomplete_rows = data.isnull().any(axis=1)
    
    for idx in data.index[incomplete_rows]:
        row = data.loc[idx]
        missing_vars = row.isnull()
        
        if missing_vars.any():
            # Create evidence from observed values
            evidence = {col: int(val) for col, val in row.items() if not pd.isna(val)}
            
            try:
                # Generate samples for missing variables
                samples = gibbs.sample(size=n_samples + burn_in)
                
                # Use samples after burn-in to fill missing values
                for col in row.index[missing_vars]:
                    if col in samples.columns:
                        post_burnin = samples[col].iloc[burn_in:]
                        # Use mode (most frequent value)
                        imputed_value = post_burnin.mode().iloc[0] if len(post_burnin) > 0 else 0
                        completed_data.loc[idx, col] = int(imputed_value)
                    else:
                        # Fallback: random binary
                        completed_data.loc[idx, col] = np.random.randint(0, 2)
                        
            except Exception as e:
                # Fallback: random imputation
                for col in row.index[missing_vars]:
                    completed_data.loc[idx, col] = np.random.randint(0, 2)
    
    return completed_data


def _parameter_learning_mle(bn_structure: BayesianNetwork, 
                           completed_datasets: List[pd.DataFrame]) -> BayesianNetwork:
    """Learn parameters from multiple completed datasets using MLE + pooling."""
    from pgmpy.estimators import MaximumLikelihoodEstimator
    
    # Learn parameters from each completed dataset
    all_cpd_estimates = []
    for i, completed_data in enumerate(completed_datasets):
        try:
            mle = MaximumLikelihoodEstimator(bn_structure, completed_data)
            cpds = mle.get_parameters()
            all_cpd_estimates.append(cpds)
        except Exception as e:
            print(f"  Warning: Failed to learn from completion {i + 1}: {e}")
    
    if not all_cpd_estimates:
        print("  No successful parameter learning, using uniform")
        return _initialize_uniform_model(bn_structure, 2)
    
    # Pool estimates using simple averaging (Rubin's rules approximation)
    pooled_cpds = _pool_cpd_estimates(all_cpd_estimates)
    
    # Create final model
    model = BayesianNetwork(bn_structure.edges())
    nodes = sorted(bn_structure.nodes())
    for node in nodes:
        if node not in model.nodes():
            model.add_node(node)
    model.add_cpds(*pooled_cpds)
    
    return model


def _pool_cpd_estimates(all_cpd_estimates: List[List]) -> List[TabularCPD]:
    """Pool CPD estimates using simple averaging."""
    n_completions = len(all_cpd_estimates)
    n_cpds = len(all_cpd_estimates[0])
    
    pooled_cpds = []
    
    for cpd_idx in range(n_cpds):
        # Get all estimates for this CPD
        cpd_values_list = []
        reference_cpd = all_cpd_estimates[0][cpd_idx]
        
        for completion_idx in range(n_completions):
            cpd = all_cpd_estimates[completion_idx][cpd_idx]
            cpd_values_list.append(cpd.get_values())
        
        # Average the values
        averaged_values = np.mean(cpd_values_list, axis=0)
        
        # Create pooled CPD
        pooled_cpd = TabularCPD(
            variable=reference_cpd.variable,
            variable_card=reference_cpd.variable_card,
            values=averaged_values,
            evidence=reference_cpd.variables[1:] if len(reference_cpd.variables) > 1 else [],
            evidence_card=reference_cpd.cardinality[1:] if len(reference_cpd.cardinality) > 1 else []
        )
        pooled_cpds.append(pooled_cpd)
    
    return pooled_cpds


def _compute_log_likelihood(data: pd.DataFrame, model: BayesianNetwork) -> float:
    """Compute approximate log-likelihood using available data."""
    try:
        # Use all data, compute likelihood for observed values only
        log_likelihood = 0.0
        total_observations = 0
        
        for _, row in data.iterrows():
            for node in model.nodes():
                # Skip missing values
                if pd.isna(row[node]):
                    continue
                    
                # Check if all parents are observed
                parents = list(model.get_parents(node))
                if any(pd.isna(row[p]) for p in parents):
                    continue  # Skip if any parent is missing
                
                # Compute probability for this observed value
                cpd = model.get_cpds(node)
                
                if parents:
                    parent_values = [int(row[p]) for p in parents]
                    prob = cpd.get_value(**{node: int(row[node])}, **{p: v for p, v in zip(parents, parent_values)})
                else:
                    prob = cpd.get_value(**{node: int(row[node])})
                
                log_likelihood += np.log(max(prob, 1e-10))
                total_observations += 1
        
        if total_observations == 0:
            return float('-inf')
            
        return log_likelihood / total_observations
        
    except Exception as e:
        print(f"  Log-likelihood computation failed: {e}")
        return float('-inf')


def learn_domain_specific_model(bn_structure: BayesianNetwork, 
                              training_data: pd.DataFrame,
                              n_states: int = 2,
                              n_samples: int = 500) -> BayesianNetwork:
    """Main interface - uses custom EM implementation."""
    return learn_domain_specific_model_simple(bn_structure, training_data, n_states, max_iter=10)


# Legacy function - replaced by clean custom EM implementation above




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