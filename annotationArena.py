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

from utils import AnnotationDataset, DataManager, compute_metrics, plot_metrics_over_time
from imputer import Imputer
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy
)

def resample_validation_dataset(dataset_train, dataset_val, active_pool, annotated_examples, 
                               strategy="balanced", update_percentage=25):
    """
    Resample the validation dataset using annotated examples and active pool.
    
    Args:
        dataset_train: Training dataset (source of data)
        dataset_val: Current validation dataset
        active_pool: List of indices available in the active pool
        annotated_examples: List of indices of examples that have been annotated
        strategy: Strategy for resampling ("balanced", "add_only", "replace_all")
        update_percentage: Percentage of validation set to update (default: 25%)
        
    Returns:
        tuple: (new_validation_dataset, updated_active_pool)
    """
    current_val_size = len(dataset_val)
    
    if strategy == "balanced":
        num_to_update = max(1, int(current_val_size * update_percentage / 100))
        new_val_indices = []
        
        if annotated_examples:
            num_from_annotated = min(len(annotated_examples), num_to_update // 2)
            if num_from_annotated > 0:
                annotated_sample = random.sample(annotated_examples, num_from_annotated)
                new_val_indices.extend(annotated_sample)
        
        remaining_needed = num_to_update - len(new_val_indices)
        if remaining_needed > 0 and active_pool:
            remaining_active = [idx for idx in active_pool if idx not in annotated_examples]
            num_from_pool = min(len(remaining_active), remaining_needed)
            if num_from_pool > 0:
                pool_sample = random.sample(remaining_active, num_from_pool)
                new_val_indices.extend(pool_sample)
        
        if new_val_indices:
            keep_size = current_val_size - len(new_val_indices)
            
            # Create new validation dataset
            new_val_data = []
            if keep_size > 0:
                for i in range(min(keep_size, current_val_size)):
                    new_val_data.append(dataset_val.get_data_entry(i))
            
            # Add new examples
            for idx in new_val_indices:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            # Create new dataset
            new_dataset_val = AnnotationDataset(new_val_data)
            
            # Update active pool
            updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
            
            print(f"Resampled validation set: {len(new_dataset_val)} examples ({len(new_val_indices)} new)")
            return new_dataset_val, updated_active_pool
    
    elif strategy == "add_only":

        max_to_add = max(1, int(current_val_size * update_percentage / 100))
        num_to_add = min(len(annotated_examples), max_to_add)
        
        if num_to_add > 0:
            # Get examples to add
            examples_to_add = random.sample(annotated_examples, num_to_add)
            
            # Create new validation dataset
            new_val_data = []
            
            # Keep all existing validation examples
            for i in range(current_val_size):
                new_val_data.append(dataset_val.get_data_entry(i))
            
            # Add new examples
            for idx in examples_to_add:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            # Create new dataset
            new_dataset_val = AnnotationDataset(new_val_data)
            
            # No need to update active pool as we're only adding already annotated examples
            print(f"Added {num_to_add} annotated examples to validation set (now {len(new_dataset_val)} examples)")
            return new_dataset_val, active_pool
    
    elif strategy == "replace_all":

        val_size = min(current_val_size, len(active_pool))
        if val_size > 0:
            # Get examples to use
            new_val_indices = random.sample(active_pool, val_size)
            
            # Create new validation dataset
            new_val_data = []
            for idx in new_val_indices:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            # Create new dataset
            new_dataset_val = AnnotationDataset(new_val_data)
            
            # Update active pool
            updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
            
            print(f"Completely replaced validation set with {len(new_dataset_val)} new examples")
            return new_dataset_val, updated_active_pool
    
    # If we get here, no resampling was done
    return dataset_val, active_pool

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
        
        affected_examples = [ex for ex in self.prediction_history if variable_id in ex["variables"]]
        self.model.update_training_supervision(
            [value], [variable_id], affected_examples
        )
        
        for ex in affected_examples:
            ex["needs_revisit"] = True
        
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
        
        known_questions, inputs, answers, annotators, questions = self._get_example_data(example_idx)
        
        predictions = self.model.predict(
            inputs.unsqueeze(0).to(self.device),
            annotators.unsqueeze(0).to(self.device),
            questions.unsqueeze(0).to(self.device),
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
        
        if strategy in ["voi", "fast_voi"]:
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
                        benefit = selection[1]  # VOI
                        
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
        
    def train(self, epochs=1, batch_size=8, lr=1e-4, revisit_examples=True):
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
        if revisit_examples:
            examples_to_revisit = [i for i, ex in enumerate(self.prediction_history) if ex["needs_revisit"]]
            examples_to_train = examples_to_revisit
            if len(examples_to_train) < batch_size * 10:
                other_examples = [i for i, ex in enumerate(self.prediction_history) if not ex["needs_revisit"]]
                if other_examples:
                    examples_to_train.extend(random.sample(other_examples, 
                                                         min(len(other_examples), batch_size * 10 - len(examples_to_train))))
        else:
            examples_to_train = list(range(len(self.prediction_history)))
        
        # Train the model
        epoch_losses = self.model.train_on_examples(
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
            "examples_revisited": len(examples_to_revisit) if revisit_examples else 0
        }
        
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
                    
                    # Get true value
                    true_label = torch.argmax(torch.tensor(entry['answers'][position])).item()
                    
                    # Convert to scores (1-5)
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


def run_experiment(dataset_train, dataset_val, dataset_test, 
                  example_strategy, feature_strategy, model,
                  cycles=5, examples_per_cycle=10, features_per_example=5,
                  epochs_per_cycle=3, batch_size=8, lr=1e-4,
                  device=None, resample_validation=False, loss_type="cross_entropy"):
    """
    Run an active learning experiment with the given strategy.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        dataset_test: Test dataset
        example_strategy: Strategy for selecting examples ("random", "gradient", etc.)
        feature_strategy: Strategy for selecting features ("random", "voi", "fast_voi", etc.)
        model: Model to use for predictions
        cycles: Number of active learning cycles
        examples_per_cycle: Number of examples to select per cycle
        features_per_example: Number of features to select per example
        epochs_per_cycle: Number of training epochs per cycle
        batch_size: Batch size for training
        lr: Learning rate for training
        device: Device to use for computations
        resample_validation: Whether to resample validation set on each cycle
        loss_type: Type of loss to use for VOI calculations
        
    Returns:
        dict: Experiment results
    """
    # Create Annotation Arena
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    # Track metrics
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'benefit_cost_ratios': [],
        'observation_costs': []
    }
    
    # Register examples in the active pool
    active_pool = list(range(len(dataset_train)))
    # Track annotated examples
    annotated_examples = []
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    print(f"Initial validation metrics: {val_metrics}")
    
    for cycle in range(cycles):
        print(f"=== Cycle {cycle+1}/{cycles} ===")
        arena.set_dataset(dataset_train)
        
        # Resample validation set if requested
        if resample_validation and cycle > 0:
            dataset_val, active_pool = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="replace_all", update_percentage=20
            )
        
        if example_strategy == "random":
            selected_examples = random.sample(active_pool, min(examples_per_cycle, len(active_pool)))
        elif example_strategy == "gradient":
            gradient_strategy = GradientSelectionStrategy(model, device)
            selected_indices, _ = gradient_strategy.select_examples(
                dataset_train, num_to_select=examples_per_cycle,
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            selected_examples = [idx for idx in selected_indices if idx in active_pool]
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
            
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        annotated_examples.extend(selected_examples)
            
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            
            candidate_variables = [
                f"example_{example_idx}_position_{pos}" 
                for pos in dataset_train.get_masked_positions(example_idx)
            ]
            
            if not candidate_variables:
                continue
                
            # Select features
            features_to_annotate = min(features_per_example, len(candidate_variables))
            
            if feature_strategy == "random":
                selected_variables = random.sample(candidate_variables, features_to_annotate)
                feature_benefit_costs = [(var, 1.0, 1.0, 1.0) for var in selected_variables]
            elif feature_strategy == "voi":
                feature_suggestions = arena.suggest(
                    candidate_variables=candidate_variables,
                    strategy="voi",
                    loss_type=loss_type
                )
                feature_benefit_costs = feature_suggestions[:features_to_annotate]
                selected_variables = [var for var, _, _, _ in feature_benefit_costs]
            elif feature_strategy == "fast_voi":
                feature_suggestions = arena.suggest(
                    candidate_variables=candidate_variables,
                    strategy="fast_voi",
                    loss_type=loss_type,
                    num_samples=3
                )
                feature_benefit_costs = feature_suggestions[:features_to_annotate]
                selected_variables = [var for var, *_ in feature_benefit_costs]
            elif feature_strategy == "sequential":
                selected_variables = candidate_variables[:features_to_annotate]
                feature_benefit_costs = [(var, 1.0, 1.0, 1.0) for var in selected_variables]
            else:
                raise ValueError(f"Unknown feature strategy: {feature_strategy}")
            
            for i, variable_id in enumerate(selected_variables):
                example_idx, position = arena._parse_variable_id(variable_id)
                arena.observe_position(example_idx, position)
                total_features_annotated += 1
                
                # Record benefit/cost if available
                if i < len(feature_benefit_costs):
                    var_id, benefit, cost, ratio, *_ = feature_benefit_costs[i]
                    cycle_benefit_cost_ratios.append(ratio)
                    cycle_observation_costs.append(cost)
                
                # Make a prediction on this position for training
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        if total_features_annotated > 0:
            training_metrics = arena.train(
                epochs=epochs_per_cycle, 
                batch_size=batch_size, 
                lr=lr,
                revisit_examples=True
            )
            
            metrics['training_losses'].append(training_metrics["avg_loss"])
            print(f"Training metrics: {training_metrics}")
        else:
            metrics['training_losses'].append(0.0)
        
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        print(f"Cycle {cycle+1} metrics: {val_metrics}")
        
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_metrics'] = test_metrics
    
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    print(f"Final test metrics: {test_metrics}")
    
    return metrics

def run_gradient_all_observe_experiment(dataset_train, dataset_val, dataset_test, 
                                      model, cycles=5, examples_per_cycle=10,
                                      epochs_per_cycle=3, batch_size=8, lr=1e-4,
                                      device=None, resample_validation=False):
    # Create Annotation Arena
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'benefit_cost_ratios': [],
        'observation_costs': []
    }
    
    active_pool = list(range(len(dataset_train)))
    # Track annotated examples
    annotated_examples = []
    
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    print(f"Initial validation metrics: {val_metrics}")
    
    # Active learning loop
    for cycle in range(cycles):
        print(f"=== Cycle {cycle+1}/{cycles} ===")
        arena.set_dataset(dataset_train)
        
        # Resample validation set if requested
        if resample_validation and cycle > 0:
            dataset_val, active_pool = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="replace_all", update_percentage=20
            )
        
        gradient_strategy = GradientSelectionStrategy(model, device)
        selected_indices, alignment_scores = gradient_strategy.select_examples(
            dataset_train, num_to_select=examples_per_cycle,
            val_dataset=dataset_val, num_samples=3, batch_size=batch_size
        )
        selected_examples = [idx for idx in selected_indices if idx in active_pool]
            
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        annotated_examples.extend(selected_examples)
            
        # 2. Observe all positions for each example
        total_features_annotated = 0
        cycle_observation_costs = []
        
        for example_idx in selected_examples:
            # Register variables for this example
            arena.register_example(example_idx, add_all_positions=False)
            
            # Get masked positions for this example
            masked_positions = dataset_train.get_masked_positions(example_idx)
            
            # Observe all positions
            for position in masked_positions:
                arena.observe_position(example_idx, position)
                total_features_annotated += 1
                cycle_observation_costs.append(arena.variables[f"example_{example_idx}_position_{position}"]["cost"])

                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        # 3. Train model
        if total_features_annotated > 0:
            training_metrics = arena.train(
                epochs=epochs_per_cycle, 
                batch_size=batch_size, 
                lr=lr,
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
            print(f"Training metrics: {training_metrics}")
        else:
            metrics['training_losses'].append(0.0)
        
        # 4. Evaluate on validation set
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        # Update metrics
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(1.0)  # For all-observe, just set to 1.0
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        print(f"Cycle {cycle+1} metrics: {val_metrics}")
        
    # Final evaluation on test set
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_metrics'] = test_metrics
    
    # Get internal metrics history
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    print(f"Final test metrics: {test_metrics}")
    
    return metrics

def run_all_observe_experiment(dataset_train, dataset_val, dataset_test, 
                              example_strategy, model,
                              cycles=5, examples_per_cycle=10,
                              epochs_per_cycle=3, batch_size=8, lr=1e-4,
                              device=None, resample_validation=False):
    # Create Annotation Arena
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    # Track metrics
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'benefit_cost_ratios': [],
        'observation_costs': []
    }
    
    # Register examples in the active pool
    active_pool = list(range(len(dataset_train)))
    # Track annotated examples
    annotated_examples = []
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    print(f"Initial validation metrics: {val_metrics}")
    
    # Active learning loop
    for cycle in range(cycles):
        print(f"=== Cycle {cycle+1}/{cycles} ===")
        arena.set_dataset(dataset_train)
        
        # Resample validation set if requested
        if resample_validation and cycle > 0:
            dataset_val, active_pool = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="replace_all", update_percentage=20
            )
        
        # 1. Select examples
        if example_strategy == "random":
            selected_examples = random.sample(active_pool, min(examples_per_cycle, len(active_pool)))
        elif example_strategy == "gradient":
            # Use gradient alignment for example selection
            gradient_strategy = GradientSelectionStrategy(model, device)
            selected_indices, _ = gradient_strategy.select_examples(
                dataset_train, num_to_select=examples_per_cycle,
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            selected_examples = [idx for idx in selected_indices if idx in active_pool]
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
            
        # Remove selected examples from active pool
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        # Add to annotated examples list
        annotated_examples.extend(selected_examples)
            
        # 2. Observe all positions for each example
        total_features_annotated = 0
        cycle_observation_costs = []
        
        for example_idx in selected_examples:
            # Register variables for this example
            arena.register_example(example_idx, add_all_positions=False)
            
            # Get masked positions for this example
            masked_positions = dataset_train.get_masked_positions(example_idx)
            
            # Observe all positions
            for position in masked_positions:
                arena.observe_position(example_idx, position)
                total_features_annotated += 1
                cycle_observation_costs.append(arena.variables[f"example_{example_idx}_position_{position}"]["cost"])

                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        # 3. Train model
        if total_features_annotated > 0:
            training_metrics = arena.train(
                epochs=epochs_per_cycle, 
                batch_size=batch_size, 
                lr=lr,
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
            print(f"Training metrics: {training_metrics}")
        else:
            metrics['training_losses'].append(0.0)
        
        # 4. Evaluate on validation set
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        # Update metrics
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(1.0)  # For all-observe, just set to 1.0
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        print(f"Cycle {cycle+1} metrics: {val_metrics}")
        
    # Final evaluation on test set
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_metrics'] = test_metrics
    
    # Get internal metrics history
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    print(f"Final test metrics: {test_metrics}")
    
    return metrics


def plot_loss_reduction(results_dict, save_path):
    """
    Plot loss reduction over active learning cycles for multiple strategies.
    
    Args:
        results_dict: Dictionary with strategy names as keys and experiment results as values
        save_path: Path to save the plot
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1, 1]})
    
    # Define colors and markers
    colors = ['blue', 'red', 'green', 'olive', 'black', 'pink']
    markers = ['o', 's', '^', 'D', 'X', '1']
    
    # Prepare data for plotting
    loss_data = {}
    for strategy, metrics in results_dict.items():
        # Extract validation losses
        val_losses = metrics['val_losses']
        loss_data[strategy] = val_losses
    
    max_iter = max([len(losses) for losses in loss_data.values()])
    x = list(range(max_iter))
    
    # First subplot: Absolute loss reduction
    for i, (strategy, losses) in enumerate(loss_data.items()):
        initial_loss = losses[0]
        reductions = [initial_loss - loss for loss in losses]
        
        axes[0].plot(x[:len(reductions)], reductions, 
                 linestyle='--', 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 label=strategy,
                 linewidth=1.5,
                 markersize=8)
    
    # Add labels and grid for first subplot
    axes[0].set_title('Absolute Loss Reduction', fontsize=14)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Reduction from Initial Value', fontsize=12)
    axes[0].grid(True, linestyle='-', alpha=0.7)
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].set_ylim(bottom=0)
    
    # Second subplot: Percentage improvement
    for i, (strategy, losses) in enumerate(loss_data.items()):
        initial_loss = losses[0]
        percent_reductions = [100 * (initial_loss - loss) / initial_loss if initial_loss != 0 else 0 for loss in losses]
        
        axes[1].plot(x[:len(percent_reductions)], percent_reductions, 
                 linestyle='--', 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 label=strategy,
                 linewidth=1.5,
                 markersize=8)
    
    # Add labels and grid for second subplot
    axes[1].set_title('Percentage Loss Reduction', fontsize=14)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Reduction (% of Initial Value)', fontsize=12)
    axes[1].grid(True, linestyle='-', alpha=0.7)
    axes[1].legend(loc='upper left', frameon=True)
    axes[1].set_ylim(bottom=0)
    
    # Third subplot: Training losses
    for i, (strategy, metrics) in enumerate(results_dict.items()):
        training_losses = metrics['training_losses']
        
        axes[2].plot(x[:len(training_losses)], training_losses, 
                 linestyle='-', 
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], 
                 label=strategy,
                 linewidth=1.5,
                 markersize=8)
    
    # Add labels and grid for third subplot
    axes[2].set_title('Training Losses', fontsize=14)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel('Training Loss', fontsize=12)
    axes[2].grid(True, linestyle='-', alpha=0.7)
    axes[2].legend(loc='upper right', frameon=True)
    axes[2].set_ylim(bottom=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Annotation Arena experiments")
    parser.add_argument("--cycles", type=int, default=5, help="Number of active learning cycles")
    parser.add_argument("--examples_per_cycle", type=int, default=20, help="Number of examples to select per cycle")
    parser.add_argument("--features_per_example", type=int, default=5, help="Number of features to select per example")
    parser.add_argument("--epochs_per_cycle", type=int, default=3, help="Number of training epochs per cycle")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--experiment", type=str, default="all", 
                       help="Experiment to run ('all', 'random_all', 'random_5', 'gradient_all', 'gradient_sequential', 'gradient_voi', 'gradient_fast_voi')")
    parser.add_argument("--resample_validation", action="store_true", help="Resample validation set on each cycle")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", help="Type of loss to use (cross_entropy, l2)")
    args = parser.parse_args()
    
    
    # Set up paths
    base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
    data_path = os.path.join(base_path, "data")
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up data manager and load datasets
    data_manager = DataManager()
    data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
    
    train_dataset = AnnotationDataset(data_manager.paths['train'])
    val_dataset = AnnotationDataset(data_manager.paths['validation'])
    test_dataset = AnnotationDataset(data_manager.paths['test'])
    active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
    
    print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")
    
    model = Imputer(
        question_num=7, 
        max_choices=5, 
        encoder_layers_num=6,
        attention_heads=4, 
        hidden_dim=64, 
        num_annotator=18, 
        annotator_embedding_dim=19, 
        dropout=0.1
    ).to(device)
    
    experiment_results = {}
    
    if args.experiment == "all" or args.experiment == "random_all":
        print("\n=== Running Random-All Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_all_observe_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="random", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["random_all"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "random_all.pth"))
        with open(os.path.join(results_path, "random_all.json"), "w") as f:
            json.dump(results, f, indent=4)

    if args.experiment == "all" or args.experiment == "gradient_fast_voi":
        print("\n=== Running Gradient-FastVOI Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="fast_voi", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["gradient_fast_voi"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_fast_voi.pth"))
        with open(os.path.join(results_path, "gradient_fast_voi.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "gradient_all":
        print("\n=== Running Gradient-All Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_gradient_all_observe_experiment(
            active_pool_dataset, val_dataset, test_dataset,
            model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["gradient_all"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_all.pth"))
        with open(os.path.join(results_path, "gradient_all.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "random_5":
        print("\n=== Running Random-5 Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="random", feature_strategy="random", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["random_5"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "random_5.pth"))
        with open(os.path.join(results_path, "random_5.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "gradient_sequential":
        print("\n=== Running Gradient-Sequential Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="sequential", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["gradient_sequential"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_sequential.pth"))
        with open(os.path.join(results_path, "gradient_sequential.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "gradient_voi":
        print("\n=== Running Gradient-VOI Experiment ===")
        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="voi", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation
        )
        experiment_results["gradient_voi"] = results
        
        # Save model and results
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_voi.pth"))
        with open(os.path.join(results_path, "gradient_voi.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if experiment_results:
        plot_loss_reduction(experiment_results, os.path.join(results_path, "loss_reduction.png"))
        with open(os.path.join(results_path, "combined_results.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"Results saved to {results_path}")
        
if __name__ == "__main__":
    main()
