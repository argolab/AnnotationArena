"""
AnnotationArena: Core class for managing annotation variables, predictions, and training.

This class provides the main interface for the Active Learner framework, managing:
- Variable registration and observation
- Model predictions with proper example_idx handling
- Training integration with the new queue-based system
- Dataset management and reference passing

Author: Prabhav Singh / Haojun Shi
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class AnnotationArena:
    """
    Core class for Annotation Arena framework.
    Manages variables, observations, predictions, and training coordination.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize Annotation Arena.
        
        Args:
            model: Imputer model to use for predictions
            device: Device to use for computations
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Variable management
        self.variables = {}  # variable_id -> metadata
        self.observations = {}  # variable_id -> observed value
        
        # History tracking
        self.prediction_history = []
        self.observation_history = []
        self.training_losses = []
        
        # Dataset reference
        self.dataset = None
        
        # Dynamic masking parameters (optional)
        self.num_patterns_per_example = 5
        self.visible_ratio = 0.5
        
        logger.info(f"AnnotationArena initialized with model: {type(model).__name__}")
    
    def set_dataset(self, dataset):
        """
        Set the dataset to use for predictions and provide reference to model.
        
        Args:
            dataset: Dataset to use
            
        Returns:
            bool: Success status
        """
        self.dataset = dataset
        # CRITICAL: Provide dataset reference to the model
        self.model.set_dataset(dataset)
        logger.info(f"Dataset set with {len(dataset)} examples")
        return True
    
    def set_dynamic_masking_params(self, num_patterns_per_example=5, visible_ratio=0.5):
        """
        Set parameters for dynamic masking training.
        
        Args:
            num_patterns_per_example: Number of masking patterns per example
            visible_ratio: Ratio of observed positions to keep visible
        """
        self.num_patterns_per_example = num_patterns_per_example
        self.visible_ratio = visible_ratio
        logger.debug(f"Dynamic masking params set: patterns={num_patterns_per_example}, visible_ratio={visible_ratio}")
    
    def add(self, variable_id, loss_function="cross_entropy", distribution_family="categorical", cost=1.0):
        """
        Register a new variable to be tracked/predicted.
        
        Args:
            variable_id: Unique identifier for the variable (format: "example_X_position_Y")
            loss_function: Loss function to use for this variable
            distribution_family: Distribution family for this variable
            cost: Cost of observing this variable
            
        Returns:
            bool: Success status
        """
        if variable_id in self.variables:
            logger.debug(f"Variable {variable_id} already exists")
            return False
            
        self.variables[variable_id] = {
            "loss_function": loss_function,
            "distribution_family": distribution_family,
            "timestamp": len(self.variables),
            "cost": cost
        }
        
        logger.debug(f"Added variable: {variable_id}")
        return True
    
    def observe(self, variable_id, value):
        """
        Record an observation for a variable.
        
        Args:
            variable_id: Identifier for the variable
            value: Observed value
            
        Returns:
            bool: Success status
        """
        if variable_id not in self.variables:
            logger.warning(f"Cannot observe unknown variable: {variable_id}")
            return False
            
        observation_time = len(self.observations)
        self.observations[variable_id] = {
            "value": value,
            "timestamp": observation_time
        }
        
        self.observation_history.append({
            "variable_id": variable_id,
            "value": value,
            "timestamp": observation_time,
            "cost": self.variables[variable_id]["cost"]
        })
        
        # Update training supervision in the model
        updated_count = self.model.update_training_supervision(
            observed_values=[value], 
            variable_ids=[variable_id]
        )
        
        logger.debug(f"Observed {variable_id} = {value}, updated {updated_count} training entries")
        return True
    
    def predict(self, variable_id, conditions=None, train=True, weight=1.0):
        """
        Predict distribution of a variable.
        
        Args:
            variable_id: Identifier for the variable
            conditions: Optional conditions for the prediction
            train: Whether to track this prediction for training
            weight: Weight of this example for training
            
        Returns:
            Predicted distribution or None if variable not found
        """
        if variable_id not in self.variables:
            logger.warning(f"Cannot predict unknown variable: {variable_id}")
            return None
        
        # Parse variable ID to get example and position indices
        example_idx, position_idx = self._parse_variable_id(variable_id)
        
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        # Get example data
        known_questions, inputs, answers, annotators, questions, embeddings = self._get_example_data(example_idx)
        
        # Prepare tensors
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        
        # CRITICAL: Pass example_idx to the model's predict method
        predictions = self.model.predict(
            inputs.unsqueeze(0).to(self.device),
            annotators.unsqueeze(0).to(self.device),
            questions.unsqueeze(0).to(self.device),
            embeddings,
            positions=[position_idx],
            train=train,
            weight=weight,
            example_idx=example_idx  # FIXED: Added missing parameter
        )
        
        # Track prediction in history
        if train:
            prediction_entry = {
                "variable_id": variable_id,
                "example_idx": example_idx,
                "position_idx": position_idx,
                "timestamp": len(self.prediction_history),
                "weight": weight,
                "conditions": conditions,
                "needs_revisit": False,
                "loss": None
            }
            self.prediction_history.append(prediction_entry)
        
        logger.debug(f"Predicted {variable_id} for example {example_idx}, position {position_idx}")
        return predictions[0, 0] if predictions is not None else None
    
    def decode(self, variable_id):
        """
        Return minimum-Bayes-risk value for a variable and its expected loss.
        
        Args:
            variable_id: Identifier for the variable
            
        Returns:
            tuple: (decoded_value, expected_loss)
        """
        prediction = self.predict(variable_id, train=False)
        if prediction is None:
            return None, 1.0

        if isinstance(prediction, torch.Tensor):

            probs = torch.softmax(prediction, dim=-1)
            decoded_value = torch.argmax(prediction).item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            expected_loss = entropy
            
            return decoded_value, expected_loss
        else:
            return prediction, 1.0  # Fallback
    
    def register_example(self, example_idx, add_all_positions=True, costs=None):
        """
        Register variables for positions in an example.
        
        Args:
            example_idx: Index of the example
            add_all_positions: Whether to add all positions or just masked ones
            costs: Optional dictionary mapping positions to costs
            
        Returns:
            list: Added variable IDs
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        # Get positions to register
        if add_all_positions:
            # Register all positions in the example
            data_entry = self.dataset.get_data_entry(example_idx)
            positions = list(range(len(data_entry['input'])))
        else:
            # Register only masked positions
            positions = self.dataset.get_masked_positions(example_idx)
        
        # Add variables for each position
        variable_ids = []
        for position in positions:
            variable_id = f"example_{example_idx}_position_{position}"
            
            # Get cost for this position
            cost = 1.0  # Default cost
            if costs and position in costs:
                cost = costs[position]
            
            if self.add(variable_id, cost=cost):
                variable_ids.append(variable_id)
        
        logger.debug(f"Registered {len(variable_ids)} variables for example {example_idx}")
        return variable_ids
    
    def observe_position(self, example_idx, position):
        """
        Observe a position in an example.
        
        Args:
            example_idx: Index of the example
            position: Position to observe
            
        Returns:
            bool: Success status
        """
        # Create variable ID
        variable_id = f"example_{example_idx}_position_{position}"
        
        # Add variable if it doesn't exist
        if variable_id not in self.variables:
            self.add(variable_id)
        
        # Get true value from dataset
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        data_entry = self.dataset.get_data_entry(example_idx)
        true_value = data_entry['answers'][position]
        
        # Observe the variable
        success = self.observe(variable_id, true_value)
        
        # Update dataset observation state
        if success:
            self.dataset.observe_position(example_idx, position)
        
        logger.debug(f"Observed position {position} in example {example_idx}: {true_value}")
        return success
    
    def train(self, training_type='basic', epochs=1, batch_size=8, lr=1e-4, revisit_examples=False):
        """
        Train the model using the current prediction history.
        
        Args:
            training_type: Type of training ('basic', 'random_masking', 'dynamic_masking')
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            revisit_examples: Whether to revisit examples (currently unused)
            
        Returns:
            dict: Training metrics
        """
        if not hasattr(self.model, 'training_queue') or len(self.model.training_queue) == 0:
            logger.warning("No training examples available")
            return {"losses": [], "avg_loss": 0.0, "examples_trained": 0}
        
        logger.info(f"Training model with {training_type} on {len(self.model.training_queue)} examples")
        
        # Get all training examples indices
        examples_to_train = list(range(len(self.model.training_queue)))
        
        # Train based on type
        if training_type == 'basic':
            epoch_losses = self.model.train_on_examples_basic(
                examples_indices=examples_to_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr
            )
        elif training_type == 'dynamic_masking':
            epoch_losses = self.model.train_on_examples_dynamic_masking(
                examples_indices=examples_to_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                num_patterns_per_example=self.num_patterns_per_example,
                visible_ratio=self.visible_ratio
            )
        elif training_type == 'random_masking':
            epoch_losses = self.model.train_on_examples_random_masking(
                examples_indices=examples_to_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr
            )
        else:
            raise ValueError(f"Unknown training type: {training_type}")
        
        # Update training history
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.training_losses.append(avg_loss)
        
        training_metrics = {
            "losses": epoch_losses,
            "avg_loss": avg_loss,
            "examples_trained": len(examples_to_train),
            "training_type": training_type
        }
        
        logger.info(f"Training completed - Avg loss: {avg_loss:.4f}, Examples: {len(examples_to_train)}")
        return training_metrics
    
    def evaluate(self, target_examples, target_questions=None, metrics=None):
        """
        Evaluate predictions on target examples and questions.
        
        Args:
            target_examples: List of example indices to evaluate
            target_questions: List of question indices to evaluate
            metrics: List of metrics to compute
            
        Returns:
            dict: Evaluation metrics
        """
        if metrics is None:
            metrics = ["rmse", "pearson", "spearman", "kendall"]
            
        if target_questions is None:
            target_questions = [0, 1, 2, 3, 4, 5, 6]  # Default to Q0
            
        all_preds = []
        all_true = []
        all_losses = []
        
        for example_idx in target_examples:
            for q_idx in target_questions:
                # Find position with this question
                positions = []
                entry = self.dataset.get_data_entry(example_idx)
                for i, question in enumerate(entry['questions']):
                    if question == q_idx and entry['annotators'][i] >= 0:  # Human annotation
                        positions.append(i)
                        
                for position in positions:
                    # Get variable ID
                    variable_id = f"example_{example_idx}_position_{position}"
                    
                    # Predict value and get expected loss
                    if variable_id in self.variables:
                        pred, expected_loss = self.decode(variable_id)
                    else:
                        # If not registered, register and predict
                        self.add(variable_id)
                        pred, expected_loss = self.decode(variable_id)
                    
                    # Get true value - USE TRUE_ANSWERS for evaluation if available
                    if 'true_answers' in entry:
                        true_label = torch.argmax(torch.tensor(entry['true_answers'][position])).item()
                    else:
                        # Fallback to answers if true_answers not available (backward compatibility)
                        true_label = torch.argmax(torch.tensor(entry['answers'][position])).item()
                    
                    # Convert to scores (1-5 for HANNA, 1-4 for LLM_RUBRIC)
                    pred_score = pred + 1 if pred is not None else 1
                    true_score = true_label + 1
                    
                    all_preds.append(pred_score)
                    all_true.append(true_score)
                    all_losses.append(expected_loss)
        
        # Compute metrics
        from utils import compute_metrics
        results = compute_metrics(np.array(all_preds), np.array(all_true))
        results["avg_expected_loss"] = np.mean(all_losses) if all_losses else 0.0
        
        return results
        
    def _parse_variable_id(self, variable_id):
        """
        Parse variable ID to get example index and position index.
        
        Args:
            variable_id: Variable identifier (format: "example_{example_idx}_position_{position_idx}")
            
        Returns:
            tuple: (example_idx, position_idx)
        """
        if isinstance(variable_id, str) and "example_" in variable_id and "_position_" in variable_id:
            parts = variable_id.split('_')
            if len(parts) >= 4:
                try:
                    example_idx = int(parts[1])
                    position_idx = int(parts[3])
                    return example_idx, position_idx
                except ValueError:
                    pass
        
        # Fallback for unexpected formats
        logger.warning(f"Could not parse variable_id: {variable_id}")
        return 0, 0
    
    def _get_example_data(self, example_idx):
        """
        Get data for a specific example.
        
        Args:
            example_idx: Index of the example
            
        Returns:
            tuple: (known_questions, inputs, answers, annotators, questions, embeddings)
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        return self.dataset[example_idx]
    
    def get_variable_count(self):
        """Get the total number of registered variables."""
        return len(self.variables)
    
    def get_observation_count(self):
        """Get the total number of observations made."""
        return len(self.observations)
    
    def get_training_history(self):
        """Get the training loss history."""
        return self.training_losses.copy()
    
    def is_variable_observed(self, variable_id):
        """Check if a variable has been observed."""
        return variable_id in self.observations
    
    def get_observed_variables(self):
        """Get list of all observed variable IDs."""
        return list(self.observations.keys())
    
    def get_unobserved_variables(self):
        """Get list of all unobserved variable IDs."""
        return [vid for vid in self.variables.keys() if vid not in self.observations]