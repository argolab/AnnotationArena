"""
Domain-specific EM model wrapper for progressive imputation.

Adapts the domain-specific model from the main codebase.
"""

import logging

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
    
    def __init__(self, max_iter=100, epsilon=1e-3):
        super().__init__("Domain_EM")
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.learned_bn = None
        
    def train(self, training_data, bn, adj_matrix, n_nodes, **kwargs):
        """
        Train the domain EM model on training data.
        
        Args:
            training_data: List of training samples
            bn: BayesNet object (for structure)
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
        """
        logger.debug(f"Training domain EM model on {len(training_data)} samples")
        
        # Convert training data to pyAgrum format
        pyagrum_data = convert_training_data_for_pyagrum(training_data, n_nodes)
        
        # Learn domain-specific model using EM
        self.learned_bn = learn_domain_specific_model(
            adj_matrix, pyagrum_data, n_states=2, 
            max_iter=self.max_iter, epsilon=self.epsilon
        )
        
        self.is_trained = True
        logger.info(f"Domain EM training completed")
        
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