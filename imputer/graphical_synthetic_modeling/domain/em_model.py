"""
Domain-specific EM baseline model for progressive imputation experiments.

Implements Expectation-Maximization algorithm for learning Bayesian Network parameters
from incomplete data using pyAgrum. Includes multiple random restarts for robust learning
and exact Bayesian inference for evaluation metrics.
"""

import logging
import numpy as np
import pandas as pd
import torch
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

# Suppress pandas warnings
warnings.filterwarnings('ignore')

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for EM learning and Bayesian inference. "
        "Please install pyAgrum: pip install pyagrum"
    )

logger = logging.getLogger(__name__)

# Type alias for sample tuple
SampleTuple = Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]



class BaseImputationModel(ABC):
    """
    Abstract base class for all imputation models.
    
    Defines the standard interface that all imputation models must implement
    for training, evaluation, and state management.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def train(self, training_data: List[SampleTuple], bn: gum.BayesNet, 
              adj_matrix: np.ndarray, n_nodes: int, **kwargs) -> None:
        """
        Train the model on the given training data.
        
        Args:
            training_data: List of training samples
            bn: BayesNet object (for structure info)
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[SampleTuple], bn: gum.BayesNet, 
                n_nodes: int, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on test data using KL divergence.
        
        Args:
            test_data: List of test samples with missing values
            bn: BayesNet object
            n_nodes: Number of nodes
            **kwargs: Additional evaluation parameters
            
        Returns:
            dict: Evaluation results containing 'mean_kl' and other metrics
        """
        pass
    
    def reset(self) -> None:
        """Reset model parameters for retraining from scratch."""
        self.is_trained = False
        logger.debug(f"Reset {self.name} model")
        
    def __str__(self) -> str:
        return f"{self.name}({'trained' if self.is_trained else 'untrained'})"


def convert_training_data_for_pyagrum(train_data: List[SampleTuple], n_nodes: int) -> pd.DataFrame:
    """
    Convert masked training data to pyAgrum format with missing value markers.
    
    PyAgrum expects missing values to be marked with "?" string in a pandas DataFrame.
    This function converts the torch tensor format to the expected pyAgrum format.
    
    Args:
        train_data: List of (inputs, embeddings, dimensions, mask, targets)
        n_nodes: Number of nodes in the graph
        
    Returns:
        pandas DataFrame with columns for each node, "?" for missing values
    """
    logger.debug(f"Converting {len(train_data)} samples for pyAgrum EM learning...")
    samples = []
    
    for i, (inputs, embeddings, dimensions, mask, targets, true_states) in enumerate(train_data):
        sample = {}
        observed_count = 0
        
        for node in range(n_nodes):
            if mask[node] == 0:  # Observed node
                # Extract state from one-hot encoding in inputs[node, 1:3]
                state = torch.argmax(inputs[node, 1:]).item()
                sample[str(node)] = str(state)  # String values for pyAgrum
                observed_count += 1
            else:  # Unobserved node
                sample[str(node)] = "?"  # pyAgrum missing value marker
                
        samples.append(sample)
        
        # Debug observation patterns
        if i < 3:
            obs_ratio = observed_count / n_nodes
            logger.debug(f"  Sample {i}: {observed_count}/{n_nodes} nodes observed ({obs_ratio:.1%})")
    
    # Create DataFrame with categorical columns
    df = pd.DataFrame(samples)
    
    # Convert to categorical for pyAgrum compatibility
    for col in df.columns:
        df[col] = df[col].astype('category')
    
    # Log data statistics
    total_missing = (df == '?').sum().sum()
    total_values = df.shape[0] * df.shape[1]
    missing_rate = total_missing / total_values
    
    logger.debug(f"Created DataFrame: columns={list(df.columns)}")
    logger.info(f"Training data: {df.shape[0]} samples, {total_missing}/{total_values} missing ({missing_rate:.1%})")
    logger.debug(f"Missing values per node: {(df == '?').sum().to_dict()}")
    
    # Check for problematic data patterns that might cause zero log-likelihood
    if missing_rate > 0.95:
        logger.warning(f"Very high missing rate: {missing_rate:.1%} - may cause pyAgrum issues")
    
    # Check if any column is completely missing
    completely_missing_cols = []
    for col in df.columns:
        if (df[col] == '?').all():
            completely_missing_cols.append(col)
    
    if completely_missing_cols:
        logger.warning(f"Completely missing columns: {completely_missing_cols} - may cause pyAgrum logLikelihood=0")
    
    # Check for columns with only one unique value (excluding '?')
    low_variance_cols = []
    for col in df.columns:
        observed_values = df[col][df[col] != '?']
        if len(observed_values) > 0 and len(observed_values.unique()) == 1:
            low_variance_cols.append(col)
    
    if low_variance_cols:
        logger.warning(f"Columns with no variance: {low_variance_cols} - may cause numerical issues")
    
    return df


def create_pyagrum_bn_from_adjacency(adj_matrix: np.ndarray) -> gum.BayesNet:
    """
    Create pyAgrum BayesNet structure from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix where adj_matrix[i,j] = 1 means edge i -> j
        
    Returns:
        pyAgrum BayesNet with structure and initial random CPDs
    """
    n_nodes = adj_matrix.shape[0]
    logger.debug(f"Creating pyAgrum BN from {n_nodes}x{n_nodes} adjacency matrix")
    
    # Create pyAgrum BayesNet
    bn = gum.BayesNet("DomainEMBN")
    
    # Add binary variables with string states "0", "1"
    for i in range(n_nodes):
        bn.add(gum.LabelizedVariable(str(i), str(i), ["0", "1"]))
    
    # Add arcs from adjacency matrix
    edges_added = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                bn.addArc(str(i), str(j))
                edges_added += 1    
    # Generate initial random CPDs
    for node_id in bn.nodes():
        bn.generateCPT(node_id)
    
    logger.debug(f"Created pyAgrum BN: {bn.size()} nodes, {edges_added} arcs")
    return bn


def learn_with_pyagrum_em(bn: gum.BayesNet, training_data: pd.DataFrame,
                         max_iter: int = 100, epsilon: float = 1e-3) -> Tuple[gum.BayesNet, int]:
    """
    Learn BN parameters using pyAgrum's EM algorithm.
    
    Args:
        bn: BayesNet with structure and initial parameters
        training_data: DataFrame with missing values marked as "?"
        max_iter: Maximum EM iterations
        epsilon: Convergence threshold
        
    Returns:
        Tuple of (learned BayesNet, number of iterations)
    """
    logger.debug(f"Learning with pyAgrum EM: max_iter={max_iter}, epsilon={epsilon}")
    
    # Create BN learner with missing data support
    learner = gum.BNLearner(training_data, bn, ["?"])  # "?" is missing value marker
    learner.useSmoothingPrior(1.0)  # Laplace smoothing for robust learning
    learner.useEM(epsilon)  # Enable EM with convergence threshold
    learner.setMaxIter(max_iter)  # Set maximum iterations
    
    logger.debug(f"EM configuration: enabled={learner.isUsingEM()}, "
                f"max_iter={learner.EMMaxIter()}, epsilon={learner.EMEpsilon()}")
    
    # Learn parameters - this is where the actual EM happens
    logger.debug("Running pyAgrum EM algorithm...")
    learned_bn = learner.learnParameters(bn)
    
    # Get final statistics
    iterations = learner.EMnbrIterations()
    em_state = learner.EMStateMessage()
    
    logger.info(f"EM completed in {iterations} iterations")
    logger.debug(f"Final EM state: {em_state}")
    
    return learned_bn, iterations


class DomainEMModel(BaseImputationModel):
    """
    Domain-specific EM model for Bayesian Network parameter learning.
    
    Uses pyAgrum's EM algorithm with multiple random restarts to learn BN parameters
    from incomplete data. Provides both KL divergence and log-loss evaluation.
    """
    
    def __init__(self, max_iter: int = 100, epsilon: float = 1e-3, n_restarts: int = 5):
        """
        Initialize EM model with configuration parameters.
        
        Args:
            max_iter: Maximum EM iterations per restart
            epsilon: Convergence threshold for EM
            n_restarts: Number of random restarts for robust learning
        """
        super().__init__("Domain_EM")
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_restarts = n_restarts
        self.learned_bn: Optional[gum.BayesNet] = None
        
        logger.debug(f"Initialized {self.name}: max_iter={max_iter}, epsilon={epsilon}, restarts={n_restarts}")
        
    def train(self, training_data: List[SampleTuple], bn: gum.BayesNet, 
              adj_matrix: np.ndarray, n_nodes: int, **kwargs) -> None:
        """
        Train the EM model with multiple random restarts.
        
        Args:
            training_data: List of training samples with missing data
            bn: BayesNet object (for structure reference, not modified)
            adj_matrix: Adjacency matrix defining BN structure
            n_nodes: Number of nodes in the network
            **kwargs: Additional training parameters (unused)
            
        Raises:
            RuntimeError: If all EM restarts fail
            Exception: Any EM learning failures will bubble up
        """
        logger.info(f"Training {self.name} on {len(training_data)} samples with {self.n_restarts} restarts")
        
        # Convert training data to pyAgrum format
        pyagrum_data = convert_training_data_for_pyagrum(training_data, n_nodes)
        
        # Try multiple random restarts with fallback to iteration-based selection
        best_bn = None
        best_likelihood = float('-inf')  # Use log-likelihood for proper model selection
        fallback_candidates = []  # Store (bn, iterations) for fallback selection
        used_fallback = False  # Track whether we used iteration-based fallback
        
        for restart in range(self.n_restarts):
            logger.debug(f"EM restart {restart + 1}/{self.n_restarts}")
            
            # Create fresh BN structure for this restart (don't modify the original)
            restart_bn = create_pyagrum_bn_from_adjacency(adj_matrix)
            
            # Randomize initial CPDs with different seed
            restart_seed = 42 + restart * 1000
            np.random.seed(restart_seed)
            for node_id in restart_bn.nodes():
                restart_bn.generateCPT(node_id)  # New random initialization
            
            try:
                # EM learning with exception handling
                logger.debug(f"Learning EM parameters with seed {restart_seed}")
                candidate_bn, iterations = learn_with_pyagrum_em(
                    restart_bn, pyagrum_data, self.max_iter, self.epsilon
                )
                
                logger.debug(f"Restart {restart + 1} completed in {iterations} iterations")
                
                if candidate_bn is not None:
                    # Store for potential fallback
                    fallback_candidates.append((candidate_bn, iterations))
                    
                    # Try likelihood-based selection
                    try:
                        # Create learner to compute log-likelihood on training data
                        likelihood_learner = gum.BNLearner(pyagrum_data, candidate_bn, ["?"])
                        
                        # Debug: Check learner properties
                        logger.debug(f"Restart {restart + 1} - Learner has {likelihood_learner.nbRows()} rows, {likelihood_learner.nbCols()} cols")
                        logger.debug(f"Restart {restart + 1} - Variable names: {likelihood_learner.names()}")
                        
                        # Get all variable names for complete likelihood computation
                        all_variable_names = [str(i) for i in range(n_nodes)]
                        log_likelihood = likelihood_learner.logLikelihood(all_variable_names)
                        logger.debug(f"Restart {restart + 1} log-likelihood: {log_likelihood:.6f}")
                        
                        # Check for suspicious zero likelihood
                        if abs(log_likelihood) < 1e-10:
                            logger.warning(f"Restart {restart + 1}: Invalid zero log-likelihood {log_likelihood}")
                            logger.warning(f"Will be available for iteration-based fallback")
                            continue
                            
                        # Normal likelihood-based selection
                        if log_likelihood > best_likelihood:
                            best_bn = candidate_bn
                            best_likelihood = log_likelihood
                            logger.debug(f"New best model from restart {restart + 1} (LL={log_likelihood:.6f})")
                            
                    except Exception as e:
                        logger.info(f"Restart {restart + 1} - logLikelihood computation failed: {e}")
                        logger.info(f"Restart {restart + 1} will be available for iteration-based fallback")
                        # Continue to next restart, but keep this candidate for fallback
                        continue
                        
            except Exception as e:
                logger.info(f"Restart {restart + 1} - EM learning failed: {e}")
                # Continue to next restart
                continue
        
        # If no valid likelihood-based selection, fall back to iteration-based selection
        if best_bn is None and fallback_candidates:
            logger.warning(f"No valid likelihood-based selection possible")
            logger.warning(f"Falling back to iteration-based selection from {len(fallback_candidates)} candidates")
            
            used_fallback = True
            # Select the model with most iterations (assuming more iterations = better convergence)
            best_iterations = max(iterations for _, iterations in fallback_candidates)
            for candidate_bn, iterations in fallback_candidates:
                if iterations == best_iterations:
                    best_bn = candidate_bn
                    best_likelihood = best_iterations  # Store iterations for fallback logging
                    logger.warning(f"Selected model with {iterations} iterations (iteration-based fallback)")
                    break
        
        if best_bn is None:
            raise RuntimeError(f"All {self.n_restarts} EM restarts failed to produce a valid model")
            
        self.learned_bn = best_bn
        self.is_trained = True
        
        # Report final results with appropriate context
        if used_fallback:
            logger.info(f"{self.name} training completed: selected model with {best_likelihood:.0f} iterations (fallback)")
        else:
            logger.info(f"{self.name} training completed: best model with log-likelihood {best_likelihood:.6f}")
        
    def evaluate(self, test_data: List[SampleTuple], bn: gum.BayesNet, 
                n_nodes: int, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the learned EM model using KL divergence.
        
        Args:
            test_data: List of test samples with missing values
            bn: Ground truth BayesNet object for computing true posteriors
            n_nodes: Number of nodes
            **kwargs: Additional evaluation parameters (unused)
            
        Returns:
            dict: Evaluation results with KL divergence metrics
            
        Raises:
            ValueError: If model is not trained
            Exception: Any evaluation failures will bubble up
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        logger.debug(f"Evaluating {self.name} on {len(test_data)} test samples")
        
        # Delegate to the core evaluation function with ground truth BN
        results = evaluate_domain_specific_model(self.learned_bn, test_data, n_nodes, 2, ground_truth_bn=bn)
        
        mean_kl = results.get('mean_kl', float('inf'))
        logger.debug(f"{self.name} evaluation: Mean KL = {mean_kl:.4f}")
        
        return results
    
    def evaluate_log_loss(self, test_data: List[SampleTuple], bn: gum.BayesNet, 
                         n_nodes: int, **kwargs) -> Dict[str, Any]:
        """
        Evaluate log-loss: -log P(true_unobserved | observed, learned_model).
        
        Args:
            test_data: List of test samples with missing values
            bn: BayesNet object (unused, kept for interface compatibility)
            n_nodes: Number of nodes
            **kwargs: Additional evaluation parameters (unused)
            
        Returns:
            dict: Log-loss evaluation results
            
        Raises:
            ValueError: If model is not trained
            Exception: Any inference failures will bubble up
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before log-loss evaluation")
            
        logger.debug(f"Evaluating {self.name} log-loss on {len(test_data)} test samples")
        
        # Use the learned BN for log-loss computation
        return self._compute_log_loss_with_bn(self.learned_bn, test_data, n_nodes)
    
    def _compute_log_loss_with_bn(self, bayesnet: gum.BayesNet, test_data: List[SampleTuple], 
                                 n_nodes: int) -> Dict[str, Any]:
        """
        Compute log-loss using exact Bayesian inference with given BayesNet.
        
        Args:
            bayesnet: PyAgrum BayesNet to use for inference
            test_data: List of test samples
            n_nodes: Number of nodes
            
        Returns:
            dict: Log-loss results with statistics
            
        Raises:
            ImportError: If pyAgrum is not available (graceful degradation)
            Exception: Any inference failures will bubble up for debugging
        """
        # Handle pyAgrum import gracefully (this try-except is legitimate)
        try:
            import pyagrum as gum
        except ImportError:
            logger.error("PyAgrum required for log-loss calculation but not available")
            return {
                'mean_log_loss': float('inf'),
                'std_log_loss': 0.0,
                'failed_rate': 1.0,
                'log_loss_values': []
            }
        
        logger.debug(f"Computing log-loss with exact inference for {len(test_data)} samples")
        
        infer = gum.LazyPropagation(bayesnet)
        log_losses = []
        
        for sample_idx, (inputs, embeddings, dimensions, mask, targets, true_states) in enumerate(test_data):
            logger.debug(f"Processing sample {sample_idx} for log-loss")
            
            # Create evidence from observed nodes
            evidence = {}
            unobserved_nodes = []
            
            for node in range(n_nodes):
                if mask[node] == 0:  # Observed node
                    state = torch.argmax(inputs[node, 1:]).item()
                    evidence[str(node)] = str(state)
                else:  # Unobserved node - we need to predict these
                    unobserved_nodes.append(node)
            
            if not unobserved_nodes:
                logger.debug(f"  Sample {sample_idx}: No unobserved nodes, skipping")
                continue
            
            logger.debug(f"  Sample {sample_idx}: Evidence={evidence}, Unobserved={unobserved_nodes}")
            
            # NO TRY-EXCEPT HERE - Let inference failures bubble up for debugging
            
            # Set evidence and run inference
            if evidence:
                infer.setEvidence(evidence)
                infer.makeInference()
            
            # Compute log P(true_unobserved | observed)
            example_log_loss = 0.0
            
            for node in unobserved_nodes:
                node_str = str(node)
                
                if evidence:
                    # Use posterior given evidence
                    posterior = infer.posterior(node_str)
                    true_state = true_states[node].item()
                    prob_true_state = posterior[{node_str: str(true_state)}]
                else:
                    # No evidence - use marginal probability
                    marginal = bayesnet.cpt(node_str)
                    true_state = true_states[node].item()
                    prob_true_state = marginal[{node_str: str(true_state)}]
                
                # Add to log-loss: -log(prob)
                if prob_true_state > 1e-10:
                    node_log_loss = -np.log(prob_true_state)
                else:
                    node_log_loss = 10  # Large penalty for very low probability
                
                example_log_loss += node_log_loss
            
            log_losses.append(example_log_loss)
            infer.eraseAllEvidence()
                    
        # Compile results
        if not log_losses:
            logger.warning("No samples produced log-loss values!")
            return {
                'mean_log_loss': float('inf'),
                'std_log_loss': 0.0,
                'failed_rate': 1.0,
                'log_loss_values': []
            }
        
        results = {
            'mean_log_loss': np.mean(log_losses),
            'std_log_loss': np.std(log_losses),
            'failed_rate': 0.0,  # All samples succeeded (no try-except masking)
            'log_loss_values': log_losses
        }
        
        logger.info(f"{self.name} log-loss: Mean={results['mean_log_loss']:.4f} ± {results['std_log_loss']:.4f}")
        
        return results
    
    def reset(self) -> None:
        """Reset model parameters for retraining."""
        super().reset()
        self.learned_bn = None
        logger.debug(f"Reset {self.name}: cleared learned BN")


def evaluate_domain_specific_model(learned_bn: gum.BayesNet, test_data: List[SampleTuple],
                                  n_nodes: int, n_states: int = 2, ground_truth_bn: gum.BayesNet = None) -> Dict[str, Any]:
    """
    Evaluate learned BN using KL divergence on test data.
    
    Args:
        learned_bn: Learned pyAgrum BayesNet
        test_data: List of test samples with missing values
        n_nodes: Number of nodes
        n_states: Number of states per node (must be 2)
        
    Returns:
        dict: Evaluation results with KL divergence metrics
        
    Raises:
        Exception: Any inference failures will bubble up
    """
    logger.debug(f"Evaluating domain model on {len(test_data)} samples")
    
    infer = gum.LazyPropagation(learned_bn)
    kl_divergences = []
    
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
            
        # NO TRY-EXCEPT HERE - Let inference failures bubble up
        
        # Set evidence and perform inference
        if evidence:
            infer.setEvidence(evidence)
            infer.makeInference()
        
        # Compute KL divergence for unobserved nodes - per-node aggregation for consistency with neural path
        for node in unobserved_nodes:
            node_str = str(node)
            
            # Get predicted distribution
            if evidence:
                predicted = infer.posterior(node_str)
            else:
                predicted = learned_bn.cpt(node_str)
            
            # Get TRUE Bayesian posterior using ground truth BN (not one-hot sample!)
            if ground_truth_bn is not None:
                # Use ground truth BN to compute true posterior
                if evidence:
                    true_infer = gum.LazyPropagation(ground_truth_bn)
                    true_infer.setEvidence(evidence)
                    true_infer.makeInference()
                    true_posterior = true_infer.posterior(node_str)
                    true_probs = np.array([
                        true_posterior[{node_str: str(state)}]
                        for state in range(n_states)
                    ])
                    true_infer.eraseAllEvidence()
                else:
                    # No evidence - use marginal from ground truth BN
                    true_marginal = ground_truth_bn.cpt(node_str)
                    true_probs = np.array([
                        true_marginal[{node_str: str(state)}]
                        for state in range(n_states)
                    ])
            else:
                # Fallback to one-hot targets (old behavior)
                logger.warning(f"No ground truth BN provided - using one-hot targets for evaluation")
                true_probs = targets[node].numpy()
            
            # Compute KL divergence: sum_i p_true(i) * log(p_true(i) / p_pred(i))
            node_kl = 0.0
            for state in range(n_states):
                p_true = true_probs[state]
                p_pred = predicted[{node_str: str(state)}]
                
                if p_true > 1e-10:  # Avoid log(0)
                    if p_pred > 1e-10:
                        node_kl += p_true * np.log(p_true / p_pred)
                    else:
                        node_kl += 10  # Large penalty for zero predicted probability
            
            # Handle floating point precision issues in KL computation
            if np.isnan(node_kl) or np.isinf(node_kl):
                logger.warning(f"Sample {sample_idx}, Node {node}: Invalid KL={node_kl} (NaN/Inf), skipping")
                continue
            
            # Clamp small negative values to 0 (due to numerical precision when pred ≈ true)
            if node_kl < -1e-6:  # Only warn for significantly negative values
                logger.warning(f"Sample {sample_idx}, Node {node}: Significantly negative KL={node_kl}, skipping")
                continue
                
            node_kl = max(node_kl, 0.0)  # Ensure no negative due to numerical precision
            kl_divergences.append(node_kl)  # Append per-node KL for consistency
        
        infer.eraseAllEvidence()
    
    if not kl_divergences:
        logger.warning("No KL divergences computed!")
        return {
            'mean_kl': float('inf'),
            'std_kl': 0.0,
            'failed_rate': 1.0,
            'kl_values': []
        }
    
    results = {
        'mean_kl': np.mean(kl_divergences),
        'std_kl': np.std(kl_divergences),
        'failed_rate': 0.0,
        'kl_values': kl_divergences
    }
    
    logger.debug(f"Domain evaluation: Mean KL = {results['mean_kl']:.4f} ± {results['std_kl']:.4f}")
    
    return results