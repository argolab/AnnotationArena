"""
Code for AnnotationArena Class
"""

import os
import argparse
import torch
import json
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy

from utils_prabhav import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from imputer import Imputer
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy
)

class AnnotationArena:
    """
    Core class for Annotation Arena framework.
    Manages variables, observations, predictions, and suggestions.
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
        self.variables = {}
        self.observations = {}
        self.prediction_history = []
        self.observation_history = []
        self.training_losses = []
        
    def add(self, variable_id, loss_function="cross_entropy", distribution_family="categorical", cost=1.0):
        """
        Register a new variable to be tracked/predicted.
        
        Args:
            variable_id: Unique identifier for the variable
            loss_function: Loss function to use for this variable ("cross_entropy", "l2", etc.)
            distribution_family: Distribution family for this variable ("categorical", etc.)
            cost: Cost of observing this variable (default: 1.0)
            
        Returns:
            bool: Success status
        """
        if variable_id in self.variables:
            return False
            
        self.variables[variable_id] = {
            "loss_function": loss_function,
            "distribution_family": distribution_family,
            "timestamp": len(self.variables),
            "cost": cost
        }
        
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
        affected_examples = None #[ex for ex in self.prediction_history if variable_id in ex["variables"]]
        self.model.update_training_supervision(
            [value], [variable_id], affected_examples
        )
        
        '''for ex in affected_examples:
            ex["needs_revisit"] = True'''
        
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
            Predicted distribution
        """
        if variable_id not in self.variables:
            return None
            
        example_idx, position_idx = self._parse_variable_id(variable_id)
        
        known_questions, inputs, answers, annotators, questions, embeddings = self._get_example_data(example_idx)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(self.device)
        predictions = self.model.predict(
            inputs.unsqueeze(0).to(self.device),
            annotators.unsqueeze(0).to(self.device),
            questions.unsqueeze(0).to(self.device),
            embeddings,
            positions=[position_idx],
            train=train,
            weight=weight
        )
        
        if train:
            prediction_example = {
                "variables": [variable_id],
                "timestamp": len(self.prediction_history),
                "weight": weight,
                "conditions": conditions,
                "needs_revisit": False,  # Will be set to True when new observations arrive
                "loss": None,  # Will be filled during training
                "example_idx": example_idx,
                "position_idx": position_idx
            }
            self.prediction_history.append(prediction_example)
        
        return predictions[0, 0]
        
    def decode(self, variable_id):
        """
        Return minimum-Bayes-risk value for a variable.
        
        Args:
            variable_id: Identifier for the variable
            
        Returns:
            Decoded value and estimated loss
        """
        if variable_id not in self.variables:
            return None, float('inf')
        
        distribution = self.predict(variable_id, train=False)
        loss_function = self.variables[variable_id]["loss_function"]
        
        if loss_function == "cross_entropy":
            probs = torch.softmax(distribution, dim=0)
            mbr_decision = torch.argmax(probs).item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            expected_loss = entropy
            
        elif loss_function == "l2":
            probs = torch.softmax(distribution, dim=0)
            values = torch.arange(1, 6, device=self.device).float()
            mbr_decision = torch.sum(probs * values).item()
            mean = mbr_decision
            variance = torch.sum(probs * (values - mean) ** 2).item()
            expected_loss = variance
        else:
            probs = torch.softmax(distribution, dim=0)
            mbr_decision = torch.argmax(probs).item()
            expected_loss = 1.0 - probs[mbr_decision].item() 
        
        return mbr_decision, expected_loss
        
    def suggest(self, candidate_variables=None, target_variables=None, strategy="voi", loss_type="cross_entropy", **kwargs):
        """
        Recommend variables to observe next, considering benefit/cost ratio.
        
        Args:
            candidate_variables: List of candidate variables to consider
            target_variables: List of target variables to optimize for
            strategy: Selection strategy to use ("voi", "fast_voi", "gradient", etc.)
            loss_type: Type of loss to use ("cross_entropy", "l2")
            **kwargs: Additional arguments for the selection strategy
            
        Returns:
            list: Ranked list of suggested variables to observe with their benefit/cost scores
        """
        if candidate_variables is None:
            candidate_variables = [v for v in self.variables if v not in self.observations]
        
        if not candidate_variables:
            return []
            
        if target_variables is None:
            target_variables = list(self.variables.keys())
            
        candidate_mapping = {}
        for var_id in candidate_variables:
            example_idx, position_idx = self._parse_variable_id(var_id)
            if example_idx not in candidate_mapping:
                candidate_mapping[example_idx] = []
            candidate_mapping[example_idx].append((position_idx, var_id))
        
        target_mapping = {}
        for var_id in target_variables:
            example_idx, position_idx = self._parse_variable_id(var_id)
            if example_idx not in target_mapping:
                target_mapping[example_idx] = []
            target_mapping[example_idx].append((position_idx, var_id))
        
        if strategy in ["voi", "fast_voi", "entropy", "voi_argmax"]:
            feature_strategy = SelectionFactory.create_feature_strategy(strategy, self.model, self.device)
            if hasattr(feature_strategy, 'loss_type'):
                feature_strategy.loss_type = loss_type
            elif hasattr(feature_strategy, 'voi_calculator') and hasattr(feature_strategy.voi_calculator, 'loss_type'):
                feature_strategy.voi_calculator.loss_type = loss_type
                
            suggestions = []
            
            for example_idx, positions in candidate_mapping.items():
                if example_idx in target_mapping:
                    target_positions = [p for p, _ in target_mapping[example_idx]]
                else:
                    # If no targets for this example, use all positions
                    target_positions = None
                    
                selections = feature_strategy.select_features(
                    example_idx, self.dataset, num_to_select=len(positions),
                    target_questions=kwargs.get('target_questions', [0]),
                    loss_type=loss_type
                )
                
                position_to_var = {p: v for p, v in positions}
                for selection in selections:
                    if isinstance(selection, tuple):
                        pos = selection[0]
                        benefit = selection[1]
                        
                        if pos in position_to_var:
                            var_id = position_to_var[pos]
                            cost = self.variables[var_id]["cost"]
                            
                            ratio = benefit / max(cost, 1e-6)
                            if len(selection) > 4:
                                extra_data = selection[4:]
                                suggestions.append((var_id, benefit, cost, ratio) + extra_data)
                            else:
                                suggestions.append((var_id, benefit, cost, ratio))
                    else:
                        if selection in position_to_var:
                            var_id = position_to_var[selection]
                            cost = self.variables[var_id]["cost"]
                            suggestions.append((var_id, 0.0, cost, 0.0))
            
            suggestions.sort(key=lambda x: x[3], reverse=True)
                                
        elif strategy == "gradient":
            example_strategy = SelectionFactory.create_example_strategy(strategy, self.model, self.device)
            example_indices = list(candidate_mapping.keys())
            
            # Select examples
            selections, scores = example_strategy.select_examples(
                self.dataset, num_to_select=len(example_indices), 
                val_dataset=kwargs.get('val_dataset')
            )
            
            # Map back to variable IDs with benefit/cost analysis
            suggestions = []
            for i, example_idx in enumerate(selections):
                if example_idx in candidate_mapping:
                    positions = candidate_mapping[example_idx]
                    if positions:
                        var_id = positions[0][1]
                        benefit = scores[i] if scores else 0.0
                        cost = self.variables[var_id]["cost"]
                        ratio = benefit / max(cost, 1e-6)
                        suggestions.append((var_id, benefit, cost, ratio))
                        
        elif strategy == "random":
            random.shuffle(candidate_variables)
            suggestions = []
            for var_id in candidate_variables:
                cost = self.variables[var_id]["cost"]
                suggestions.append((var_id, cost, cost, 1.0)) 
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
            
        return suggestions
        
    def train(self, epochs=1, batch_size=8, lr=1e-4, revisit_examples=True, training_type='basic'):
        """
        Train imputer model on observed variables with example revisiting.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            revisit_examples: Whether to revisit examples that were affected by new observations
            
        Returns:
            dict: Training metrics including losses and example counts
        """

        examples_to_train = list(range(len(self.prediction_history)))
        
        # Train the model
        if training_type='basic':
            epoch_losses = self.model.train_on_examples_basic(
                examples_indices=examples_to_train,
                epochs=epochs, 
                batch_size=batch_size, 
                lr=lr
            )
        else:
            epoch_losses = self.model.train_on_examples_random_masking(
            examples_indices=examples_to_train,
            epochs=epochs, 
            batch_size=batch_size, 
            lr=lr
        )
        
        # Update needs_revisit flag and record losses
        for idx in examples_to_train:
            self.prediction_history[idx]["needs_revisit"] = False
            if idx < len(epoch_losses):
                self.prediction_history[idx]["loss"] = epoch_losses[idx]
        
        # Track training metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.training_losses.append(avg_loss)
        
        training_metrics = {
            "losses": epoch_losses,
            "avg_loss": avg_loss,
            "examples_trained": len(examples_to_train),
            "examples_revisited": 0
            # "examples_revisited": len(examples_to_revisit) if revisit_examples else 0
        }

        print(f'Training Metrics - {epoch_losses}, {avg_loss}, {len(examples_to_train)}')
        
        return training_metrics
    
    def _parse_variable_id(self, variable_id):
        """
        Parse variable ID to get example index and position index.
        
        Args:
            variable_id: Variable identifier (format: "example_{example_idx}_position_{position_idx}")
            
        Returns:
            tuple: (example_idx, position_idx)
        """
        # Parse from string format
        if isinstance(variable_id, str):
            parts = variable_id.split('_')
            example_idx = int(parts[1])
            position_idx = int(parts[3])
            return example_idx, position_idx
        
        # Direct tuple format
        if isinstance(variable_id, tuple) and len(variable_id) == 2:
            return variable_id
            
        # Default to identity mapping
        return variable_id, variable_id
    
    def _get_example_data(self, example_idx):
        """
        Get data for a specific example.
        
        Args:
            example_idx: Index of the example
            
        Returns:
            tuple: (known_questions, inputs, answers, annotators, questions)
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        return self.dataset[example_idx]
    
    def set_dataset(self, dataset):
        """
        Set the dataset to use for predictions.
        
        Args:
            dataset: Dataset to use
            
        Returns:
            bool: Success status
        """
        self.dataset = dataset
        return True
    
    def register_example(self, example_idx, add_all_positions=True, costs=None):
        """
        Register variables for all positions in an example.
        
        Args:
            example_idx: Index of the example
            add_all_positions: Whether to add all positions or just masked ones
            costs: Optional dictionary mapping positions to costs
            
        Returns:
            list: Added variable IDs
        """
        # Get positions
        if add_all_positions:
            positions = list(range(len(self.dataset.get_data_entry(example_idx)['input'])))
        else:
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
        # Get variable ID
        variable_id = f"example_{example_idx}_position_{position}"
        
        # Check if variable exists
        if variable_id not in self.variables:
            self.add(variable_id)
            
        # Get true value
        entry = self.dataset.get_data_entry(example_idx)
        true_value = entry['answers'][position]
        # true_value = entry['true_answers'][position] #TODO: is "true_answers" a specific key for noisy experiments?
        
        # Observe the variable
        success = self.observe(variable_id, true_value)
        
        # Update dataset
        if success:
            self.dataset.observe_position(example_idx, position)
        
        return success

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
            target_questions = [0]  # Default to Q0
            
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
                    pred_score = pred + 1
                    true_score = true_label + 1
                    
                    all_preds.append(pred_score)
                    all_true.append(true_score)
                    all_losses.append(expected_loss)
        
        # Compute metrics
        results = compute_metrics(np.array(all_preds), np.array(all_true))
        results["avg_expected_loss"] = np.mean(all_losses) if all_losses else 0.0
        
        return results

    def get_metrics_history(self):
        """Get the history of training metrics for plotting."""
        return {
            "training_losses": self.training_losses,
            "observation_history": self.observation_history,
            "prediction_history": [
                {"timestamp": ex["timestamp"], "loss": ex["loss"]} 
                for ex in self.prediction_history if ex["loss"] is not None
            ]
        }
