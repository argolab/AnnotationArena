"""
Base model interface for progressive imputation.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseImputationModel(ABC):
    """
    Abstract base class for all imputation models.
    """
    
    def __init__(self, name):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def train(self, training_data, bn, adj_matrix, **kwargs):
        """
        Train the model on the given training data.
        
        Args:
            training_data: List of training samples
            bn: BayesNet object (for structure info)
            adj_matrix: Adjacency matrix
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data, bn, **kwargs):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of test samples with missing values
            bn: BayesNet object
            **kwargs: Additional evaluation parameters
            
        Returns:
            dict: Evaluation results containing 'mean_kl' and other metrics
        """
        pass
    
    def reset(self):
        """Reset model parameters (for retraining from scratch)."""
        self.is_trained = False
        logger.debug(f"Reset {self.name} model")
        
    def __str__(self):
        return f"{self.name}({'trained' if self.is_trained else 'untrained'})"