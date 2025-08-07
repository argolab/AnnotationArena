"""
Two-stream transformer imputer for progressive imputation.

Uses proper two-stream architecture:
- Embedding stream: Node embeddings + structure + observed evidence
- Parameter stream: True CPTs for observed nodes, zeros for unobserved nodes

The model learns to impute missing CPT values through transformer layers
while never seeing true CPT values for unobserved nodes in the input.
"""

import logging
import torch
from torch.utils.data import DataLoader

from .imputer import (
    create_model,
    train_model,
    ImputationDataset,
    collate_batch,
    evaluate_model,
    compute_max_cpt_size
)

from .base_model import BaseImputationModel

logger = logging.getLogger(__name__)

class NeuralParameterEmbeddingImputer(BaseImputationModel):
    """
    Two-stream transformer imputer using proper CPT masking.
    
    Architecture:
    - Embedding stream: Node embeddings + adjacency + observed states
    - Parameter stream: True CPTs for observed, zeros for unobserved
    """
    
    def __init__(self, epochs=100, lr=1e-4, patience=30, model_size="Large"):
        super().__init__(f"Imputer_{model_size}")
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.model_size = model_size
        self.model = None
        
    def train(self, training_data, bn, adj_matrix, n_nodes, use_afa=False, **kwargs):
        """
        Train the two-stream imputer on training data.
        
        Args:
            training_data: List of training samples
            bn: BayesNet object
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
            use_afa: Whether to use self-supervised training for AFA experiments
        """
        logger.debug(f"Training two-stream imputer on {len(training_data)} samples")
        
        # Create training data with proper structure info (adjacency matrix per sample)
        train_data_prepared = []
        for inputs, _, dimensions, mask, targets in training_data:
            # Use adjacency matrix as structure info for each sample
            structure_info = torch.FloatTensor(adj_matrix)
            train_data_prepared.append((inputs, structure_info, dimensions, mask, targets))
        
        # Create dataset and data loader with num_workers=0 to avoid multiprocessing
        train_dataset = ImputationDataset(train_data_prepared, bn)
        batch_size = min(32, len(training_data))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=collate_batch, num_workers=0, persistent_workers=False)
        
        # Create test loader (use training data for validation during training)
        test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=collate_batch, num_workers=0, persistent_workers=False)
        
        # Get dimensions
        input_dim = training_data[0][0].shape[1]
        structure_dim = adj_matrix.shape[1]  # Adjacency matrix width
        cpt_dim = compute_max_cpt_size(bn)
        
        # Create and train model
        self.model = create_model(n_nodes, input_dim, structure_dim, cpt_dim, model_size=self.model_size)
        logger.debug(f"Created {self.model_size} model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Use standard training for testing
        self.model = train_model(
            self.model, train_loader, test_loader, 
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            use_self_supervised=False  # Always use standard training
        )
        
        # Explicit cleanup of DataLoaders to prevent resource leaks
        del train_loader, test_loader, train_dataset
        
        self.is_trained = True
        logger.info(f"Two-stream imputer training completed (standard)")
        
    def evaluate(self, test_data, bn, adj_matrix, n_nodes, **kwargs):
        """
        Evaluate the two-stream imputer on test data.
        
        Args:
            test_data: List of test samples with missing values
            bn: BayesNet object
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        logger.debug(f"Evaluating two-stream imputer on {len(test_data)} test samples")
        
        # Create test data with proper structure info
        test_data_prepared = []
        for inputs, _, dimensions, mask, targets in test_data:
            structure_info = torch.FloatTensor(adj_matrix)
            test_data_prepared.append((inputs, structure_info, dimensions, mask, targets))
        
        # Evaluate using new imputer
        results = evaluate_model(self.model, test_data_prepared, bn, n_nodes, 2)
        
        logger.debug(f"Two-stream evaluation: Mean KL = {results.get('mean_kl', float('inf')):.4f}")
        
        return results
    
    def reset(self):
        """Reset model parameters for retraining."""
        super().reset()
        self.model = None