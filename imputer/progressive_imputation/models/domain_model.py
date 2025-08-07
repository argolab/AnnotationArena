"""
Domain-specific EM model wrapper for progressive imputation.

Adapts the domain-specific model from the main codebase.
"""

import logging
import numpy as np

from .domain_specific_model import (
    learn_domain_specific_model,
    evaluate_domain_specific_model,
    convert_training_data_for_pyagrum
)

from .base_model import BaseImputationModel

logger = logging.getLogger(__name__)

class DomainEMModel(BaseImputationModel):
    """
    Domain-specific EM model for graph imputation.
    """
    
    def __init__(self, max_iter=100, epsilon=1e-3, n_restarts=2):
        super().__init__("Domain_EM")
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_restarts = 2  # Fixed at 2 for stability
        self.learned_bn = None
        
    def train(self, training_data, bn, adj_matrix, n_nodes, **kwargs):
        """
        Train the domain EM model on training data with random restarts.
        
        Args:
            training_data: List of training samples
            bn: BayesNet object (for structure)
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
        """
        logger.debug(f"Training domain EM model on {len(training_data)} samples with {self.n_restarts} restarts")
        
        # Convert training data to pyAgrum format
        pyagrum_data = convert_training_data_for_pyagrum(training_data, n_nodes)
        
        # Run exactly 2 EM restarts and keep the better one
        first_bn, first_ll = None, -np.inf
        second_bn, second_ll = None, -np.inf
        
        # First restart
        try:
            logger.debug("EM restart 1/2")
            first_bn, first_ll = learn_domain_specific_model(
                adj_matrix, pyagrum_data, n_states=2,
                max_iter=self.max_iter, epsilon=self.epsilon,
                restart_seed=42
            )
            logger.debug(f"Restart 1 log-likelihood: {first_ll:.4f}")
        except Exception as e:
            logger.warning(f"EM restart 1 failed: {e}")
        
        # Second restart
        try:
            logger.debug("EM restart 2/2")
            second_bn, second_ll = learn_domain_specific_model(
                adj_matrix, pyagrum_data, n_states=2,
                max_iter=self.max_iter, epsilon=self.epsilon,
                restart_seed=1042
            )
            logger.debug(f"Restart 2 log-likelihood: {second_ll:.4f}")
        except Exception as e:
            logger.warning(f"EM restart 2 failed: {e}")
        
        # Select the better model
        if first_bn is None and second_bn is None:
            raise RuntimeError("Both EM restarts failed")
        elif first_bn is None:
            self.learned_bn = second_bn
            best_log_likelihood = second_ll
            logger.debug("Using restart 2 (restart 1 failed)")
        elif second_bn is None:
            self.learned_bn = first_bn
            best_log_likelihood = first_ll
            logger.debug("Using restart 1 (restart 2 failed)")
        elif first_ll >= second_ll:
            self.learned_bn = first_bn
            best_log_likelihood = first_ll
            logger.debug("Using restart 1 (better log-likelihood)")
        else:
            self.learned_bn = second_bn
            best_log_likelihood = second_ll
            logger.debug("Using restart 2 (better log-likelihood)")
            
        self.is_trained = True
        logger.info(f"Domain EM training completed with 2 restarts, best log-likelihood: {best_log_likelihood:.4f}")
        
    def evaluate(self, test_data, bn, n_nodes, **kwargs):
        """
        Evaluate the domain EM model on test data.
        
        Args:
            test_data: List of test samples with missing values
            bn: BayesNet object (unused, kept for interface compatibility)
            n_nodes: Number of nodes
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        logger.debug(f"Evaluating domain EM model on {len(test_data)} test samples")
        
        # Evaluate using existing function
        results = evaluate_domain_specific_model(self.learned_bn, test_data, n_nodes, 2)
        
        logger.debug(f"Domain EM evaluation: Mean KL = {results.get('mean_kl', float('inf')):.4f}")
        
        return results
    
    def reset(self):
        """Reset model parameters for retraining."""
        super().reset()
        self.learned_bn = None