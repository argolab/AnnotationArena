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
import traceback
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


def learn_domain_specific_model_proper(bn_structure: BayesianNetwork,
                                      training_data: pd.DataFrame,
                                      n_states: int = 2,
                                      max_em_iters: int = 7,
                                      n_imputations: int = 5,
                                      chain_length: int = 1500,
                                      burn_in: int = 300,
                                      thin: int = 3) -> BayesianNetwork:
    """
    Principled EM with Multiple Imputation using Gibbs Sampling.
    
    Args:
        bn_structure: BayesianNetwork structure (no CPDs)
        training_data: DataFrame with NaN for missing values  
        n_states: Number of states per variable
        max_em_iters: Maximum EM iterations
        n_imputations: Number of imputed datasets per E-step
        chain_length: Length of Gibbs chain
        burn_in: Burn-in samples to discard
        thin: Thinning interval
        
    Returns:
        BayesianNetwork with learned CPDs
    """
    print(f"=== PRINCIPLED EM + MULTIPLE IMPUTATION ===")
    print(f"Training data: {len(training_data)} samples, {training_data.isnull().sum().sum()} missing values")
    print(f"EM config: {max_em_iters} iters, {n_imputations} imputations")
    print(f"Gibbs config: {chain_length} length, {burn_in} burn-in, thin={thin}")
    
    # Initialize with informed priors from complete cases
    current_model = initialize_with_informed_priors(bn_structure, training_data, n_states)
    prev_log_likelihood = float('-inf')
    
    for em_iter in range(max_em_iters):
        print(f"\n--- EM Iteration {em_iter + 1}/{max_em_iters} ---")
        
        # E-STEP: Multiple Imputation with Gibbs Sampling
        print("E-step: Generating multiple imputations...")
        completed_datasets = multiple_imputation_step(
            training_data, current_model, n_imputations, 
            chain_length, burn_in, thin
        )
        
        # M-STEP: Parameter Learning from Multiple Completions
        print("M-step: Learning parameters from completions...")
        new_model = parameter_learning_step(bn_structure, completed_datasets)
        
        # Check convergence
        current_log_likelihood = compute_log_likelihood_approx(training_data, new_model)
        ll_change = abs(current_log_likelihood - prev_log_likelihood)
        
        print(f"Log-likelihood: {current_log_likelihood:.4f} (change: {ll_change:.4f})")
        
        if ll_change < 0.01:
            print(f"Converged after {em_iter + 1} iterations!")
            break
            
        current_model = new_model
        prev_log_likelihood = current_log_likelihood
    
    print(f"EM completed. Final model has {len(current_model.get_cpds())} CPDs")
    return current_model


def initialize_with_informed_priors(bn_structure: BayesianNetwork, 
                                   training_data: pd.DataFrame,
                                   n_states: int) -> BayesianNetwork:
    """Initialize model with priors learned from complete cases."""
    print("Initializing with informed priors from complete cases...")
    
    # Find complete cases for each variable
    nodes = sorted(bn_structure.nodes())
    cpds = []
    
    for node in nodes:
        parents = list(bn_structure.get_parents(node))
        
        if not parents:
            # Root node - use marginal from complete cases
            complete_data = training_data[node].dropna()
            if len(complete_data) > 0:
                # Empirical distribution + small pseudocount
                counts = np.zeros(n_states)
                for val in complete_data:
                    counts[int(val)] += 1
                counts += 0.5  # Small pseudocount
                probs = counts / counts.sum()
                values = probs.reshape(n_states, 1)
                print(f"  Root {node}: learned from {len(complete_data)} samples, probs={probs}")
            else:
                # Fallback to uniform
                values = np.ones((n_states, 1)) / n_states
                print(f"  Root {node}: no data, using uniform")
        else:
            # Child node - use complete cases
            relevant_cols = [node] + parents
            complete_cases = training_data[relevant_cols].dropna()
            
            n_parent_configs = n_states ** len(parents)
            
            if len(complete_cases) > 3:  # Need some data
                # Learn from complete cases
                values = np.ones((n_states, n_parent_configs)) * 0.5  # Small pseudocount
                
                for _, row in complete_cases.iterrows():
                    # Get parent configuration
                    parent_config = 0
                    for i, parent in enumerate(parents):
                        parent_config += int(row[parent]) * (n_states ** (len(parents) - 1 - i))
                    
                    child_state = int(row[node])
                    values[child_state, parent_config] += 1.0
                
                # Normalize
                for config in range(n_parent_configs):
                    values[:, config] = values[:, config] / values[:, config].sum()
                    
                print(f"  Child {node}: learned from {len(complete_cases)} complete cases")
            else:
                # Not enough data - use informed random
                values = np.random.dirichlet(np.ones(n_states) * 2, size=n_parent_configs).T
                print(f"  Child {node}: insufficient data, using informed random")
        
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
    print(f"Initialized model with {len(cpds)} informed CPDs")
    return model


def multiple_imputation_step(training_data: pd.DataFrame,
                           current_model: BayesianNetwork,
                           n_imputations: int,
                           chain_length: int,
                           burn_in: int,
                           thin: int) -> List[pd.DataFrame]:
    """Generate multiple completed datasets using Gibbs sampling."""
    completed_datasets = []
    
    for m in range(n_imputations):
        print(f"  Generating imputation {m + 1}/{n_imputations}...")
        
        # Generate one completed dataset
        completed_data = gibbs_complete_dataset(
            training_data, current_model, chain_length, burn_in, thin
        )
        completed_datasets.append(completed_data)
    
    print(f"Generated {len(completed_datasets)} completed datasets")
    return completed_datasets


def gibbs_complete_dataset(data: pd.DataFrame,
                          model: BayesianNetwork,
                          chain_length: int,
                          burn_in: int,
                          thin: int) -> pd.DataFrame:
    """Complete dataset using single long Gibbs chain."""
    
    # Initialize missing values randomly
    current_data = data.copy()
    missing_positions = []
    
    for idx, row in data.iterrows():
        for col in data.columns:
            if pd.isna(row[col]):
                # Initialize missing value randomly
                current_data.loc[idx, col] = np.random.randint(0, 2)
                missing_positions.append((idx, col))
    
    print(f"    Gibbs chain: {len(missing_positions)} missing positions")
    
    # Create Gibbs sampler
    gibbs = GibbsSampling(model)
    
    # Store completions (after burn-in with thinning)
    stored_completions = []
    
    try:
        for step in range(chain_length):
            # Sample all variables jointly
            current_state = {}
            for col in data.columns:
                # Get current values for evidence
                current_state[col] = int(current_data[col].mode().iloc[0])  # Use mode as representative
            
            # Generate one Gibbs sample
            samples = gibbs.sample(start_state=list(current_state.items()), size=1)
            
            # Update missing positions with new sample
            for idx, col in missing_positions:
                if col in samples.columns:
                    current_data.loc[idx, col] = samples[col].iloc[0]
            
            # Store after burn-in with thinning
            if step >= burn_in and step % thin == 0:
                stored_completions.append(current_data.copy())
        
        # Return final completion (or random one from stored)
        if stored_completions:
            return stored_completions[-1]
        else:
            return current_data
            
    except Exception as e:
        print(f"    Gibbs sampling failed: {e}, using random completion")
        # Fallback: random completion
        for idx, col in missing_positions:
            current_data.loc[idx, col] = np.random.randint(0, 2)
        return current_data


def parameter_learning_step(bn_structure: BayesianNetwork,
                           completed_datasets: List[pd.DataFrame]) -> BayesianNetwork:
    """Learn parameters from multiple completed datasets (Rubin's rules)."""
    
    # Learn parameters from each completion
    all_cpd_estimates = []
    for i, completed_data in enumerate(completed_datasets):
        try:
            mle = MaximumLikelihoodEstimator(bn_structure, completed_data)
            cpds = mle.get_parameters()
            all_cpd_estimates.append(cpds)
            print(f"  Learned from completion {i + 1}: {len(cpds)} CPDs")
        except Exception as e:
            print(f"  Failed to learn from completion {i + 1}: {e}")
    
    if not all_cpd_estimates:
        print("  No successful parameter learning, using uniform")
        return create_uniform_model(bn_structure, 2)
    
    # Pool estimates (simple averaging for now)
    pooled_cpds = pool_cpd_estimates(all_cpd_estimates)
    
    # Create final model
    model = BayesianNetwork(bn_structure.edges())
    nodes = sorted(bn_structure.nodes())
    for node in nodes:
        if node not in model.nodes():
            model.add_node(node)
    
    model.add_cpds(*pooled_cpds)
    print(f"  Pooled {len(pooled_cpds)} CPDs from {len(all_cpd_estimates)} completions")
    return model


def pool_cpd_estimates(all_cpd_estimates: List[List]) -> List[TabularCPD]:
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


def compute_log_likelihood_approx(data: pd.DataFrame, model: BayesianNetwork) -> float:
    """Approximate log-likelihood for convergence checking."""
    try:
        # Use complete cases only for quick approximation
        complete_data = data.dropna()
        if len(complete_data) == 0:
            return float('-inf')
        
        # Compute log-likelihood on complete cases
        log_likelihood = 0.0
        for _, row in complete_data.iterrows():
            sample_ll = 0.0
            for node in model.nodes():
                cpd = model.get_cpds(node)
                parents = list(model.get_parents(node))
                
                if parents:
                    # Get parent values
                    parent_values = [int(row[p]) for p in parents]
                    # Get probability
                    prob = cpd.get_value(**{node: int(row[node])}, **{p: v for p, v in zip(parents, parent_values)})
                else:
                    prob = cpd.get_value(**{node: int(row[node])})
                
                sample_ll += np.log(max(prob, 1e-10))  # Avoid log(0)
            
            log_likelihood += sample_ll
        
        return log_likelihood / len(complete_data)  # Normalized
    except:
        return float('-inf')


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


def learn_domain_specific_model_simple(bn_structure: BayesianNetwork,
                                      training_data: pd.DataFrame,
                                      n_states: int = 2,
                                      max_iter: int = 50) -> BayesianNetwork:
    """
    Simple EM using pgmpy's built-in ExpectationMaximization.
    Much more robust than custom Gibbs implementation.
    """
    print(f"=== USING PGMPY'S BUILT-IN EM ===")
    print(f"Training data: {len(training_data)} samples, {training_data.isnull().sum().sum()} missing values")
    print(f"Max EM iterations: {max_iter}")
    
    try:
        # Use regular BayesianNetwork with ExpectationMaximization
        from pgmpy.estimators import ExpectationMaximization
        
        # Use the existing BN structure directly
        all_nodes = sorted(bn_structure.nodes())
        print(f"Using BayesianNetwork with nodes: {all_nodes}")
        print(f"Edges: {list(bn_structure.edges())}")
        
        # Method 1: Try direct EM
        try:
            print("Trying direct ExpectationMaximization...")
            em_estimator = ExpectationMaximization(bn_structure, training_data)
            learned_cpds = em_estimator.get_parameters(max_iter=max_iter)
            
            # Create final model with learned CPDs
            final_model = BayesianNetwork(bn_structure.edges())
            for node in all_nodes:
                if node not in final_model.nodes():
                    final_model.add_node(node)
            
            final_model.add_cpds(*learned_cpds)
            
            print(f"Direct EM completed successfully!")
            print(f"Learned model has {len(learned_cpds)} CPDs")
            return final_model
            
        except Exception as e1:
            print(f"Direct EM failed: {e1}")
            
            # Method 2: Try using fit method
            print("Trying BayesianNetwork.fit() with EM...")
            temp_model = BayesianNetwork(bn_structure.edges())
            for node in all_nodes:
                if node not in temp_model.nodes():
                    temp_model.add_node(node)
            
            # Use fit method with EM estimator
            temp_model.fit(training_data, estimator=ExpectationMaximization, complete_samples_only=False)
            
            print(f"Fit method completed successfully!")
            print(f"Learned model has {len(temp_model.get_cpds())} CPDs")
            return temp_model
        
    except Exception as e:
        print(f"Built-in EM failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to informed uniform model
        print("Falling back to informed uniform model...")
        return initialize_with_informed_priors(bn_structure, training_data, n_states)


def learn_domain_specific_model(bn_structure: BayesianNetwork, 
                              training_data: pd.DataFrame,
                              n_states: int = 2,
                              n_samples: int = 500) -> BayesianNetwork:
    """
    Main interface - tries simple EM first, falls back to complex if needed.
    """
    print("Trying pgmpy's built-in EM first...")
    try:
        return learn_domain_specific_model_simple(bn_structure, training_data, n_states, max_iter=20)
    except Exception as e:
        print(f"Built-in EM failed: {e}")
        print("Falling back to complex implementation...")
        return learn_domain_specific_model_proper(bn_structure, training_data, n_states)


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
            # Create initial state with observed values and random missing values
            start_state = {}
            missing_vars = []
            
            for col in data.columns:
                if not pd.isna(row[col]):
                    # Observed value - fix in start state
                    start_state[col] = int(row[col])
                else:
                    # Missing value - initialize randomly and track
                    start_state[col] = np.random.randint(0, 2)
                    missing_vars.append(col)
            
            if missing_vars:  # If we have missing variables to impute
                try:
                    # Generate Gibbs samples starting from observed state
                    # samples = gibbs.sample(start_state=start_state, size=n_samples)

                    start_state_tuples = list(start_state.items())
                    samples = gibbs.sample(start_state=start_state_tuples, size=n_samples)

                    
                    # For missing variables, use mode (most frequent value) from samples
                    c = 0
                    for var in missing_vars:
                        if var in samples.columns and len(samples) > 0:
                            # Skip burn-in samples (first 10%) and use mode
                            burn_in = max(1, n_samples // 10)
                            var_samples = samples[var].iloc[burn_in:]
                            if not var_samples.empty:
                                imputed_value = var_samples.mode().iloc[0]
                                completed_data.loc[idx, var] = imputed_value
                            else:
                                completed_data.loc[idx, var] = np.random.randint(0, 2)
                        else:
                            # Fallback: random binary
                            c = c + 1
                            completed_data.loc[idx, var] = np.random.randint(0, 2)

                    print(f"Total length is {len(missing_vars)}. Random Binary for {c}")
                        
                except Exception as e:
                    print(f"Gibbs sampling failed for row {idx}: {e}")
                    traceback.print_exc()
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