"""
Code to run active learning experiments.
"""

# Base Imports
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

# Library Imports
from annotationArena import *
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from visualizations import *
from imputer import Imputer
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy
) 

# Base Experiment Runner
def run_experiment(dataset_train, dataset_val, dataset_test, 
                  example_strategy, feature_strategy, model,
                  cycles=5, examples_per_cycle=10, features_per_example=5,
                  epochs_per_cycle=3, batch_size=8, lr=1e-4,
                  device=None, resample_validation=False, loss_type="cross_entropy",
                  run_until_exhausted=False):
    """
    Run active learning experiment with the given strategy.
    """

    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    
    # Initial test set evaluation
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
    metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
    
    max_cycles = float('inf') if run_until_exhausted else cycles
    
    while cycle_count < max_cycles:

        if not active_pool:
            print(f"Active pool exhausted after {cycle_count} cycles")
            break
            
        print(f"=== Cycle {cycle_count+1}/{cycles if not run_until_exhausted else 'until exhausted'} ===")
        print(f"Active pool size: {len(active_pool)}")
        arena.set_dataset(dataset_train)
        
        # Filter active pool to remove examples with no masked positions
        valid_active_pool = []
        for idx in active_pool:
            if dataset_train.get_masked_positions(idx):
                valid_active_pool.append(idx)
            else:
                print(f"Removing example {idx} from active pool (no masked positions)")
        
        if len(valid_active_pool) < len(active_pool):
            print(f"Filtered active pool from {len(active_pool)} to {len(valid_active_pool)} examples with masked positions")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions left in active pool")
            break
        
        # Select examples
        if example_strategy == "random":
            selected_examples = random.sample(active_pool, min(examples_per_cycle, len(active_pool)))

        elif example_strategy == "gradient":
            gradient_strategy = GradientSelectionStrategy(model, device)

            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_pool]
            active_pool_subset = AnnotationDataset(active_pool_examples)

            selected_indices, _ = gradient_strategy.select_examples(
                active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )

            selected_examples = [active_pool[idx] for idx in selected_indices]

        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
            
        print(f"Selected {len(selected_examples)} examples")
        
        # Remove selected examples from active pool
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        annotated_examples.extend(selected_examples)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        # Add selected examples to validation set if requested
        if resample_validation:
            dataset_val, _ = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_selected", selected_examples=selected_examples
            )
            
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        
        # Annotate selected examples
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            
            candidate_variables = [
                f"example_{example_idx}_position_{pos}" 
                for pos in dataset_train.get_masked_positions(example_idx)
            ]
            
            if not candidate_variables:
                print(f"No masked positions found for example {example_idx}")
                continue
                
            print(f"Example {example_idx}: {len(candidate_variables)} candidate positions")
            
            # Select features
            features_to_annotate = min(features_per_example, len(candidate_variables))
            
            if feature_strategy == "random":
                selected_variables = random.sample(candidate_variables, features_to_annotate)
                feature_benefit_costs = [(var, 1.0, 1.0, 1.0) for var in selected_variables]
            elif feature_strategy == "voi":
                feature_suggestions = arena.suggest(
                    candidate_variables=candidate_variables,
                    strategy="voi", loss_type=loss_type
                )
                if not feature_suggestions:
                    print(f"VOI strategy returned no suggestions for example {example_idx}")
                    continue
                feature_benefit_costs = feature_suggestions[:features_to_annotate]
                selected_variables = [var for var, _, _, _ in feature_benefit_costs]
            elif feature_strategy == "fast_voi":
                feature_suggestions = arena.suggest(
                    candidate_variables=candidate_variables,
                    strategy="fast_voi", loss_type=loss_type, num_samples=3
                )
                if not feature_suggestions:
                    print(f"Fast VOI strategy returned no suggestions for example {example_idx}")
                    continue
                feature_benefit_costs = feature_suggestions[:features_to_annotate]
                selected_variables = [var for var, *_ in feature_benefit_costs]
            elif feature_strategy == "sequential":
                selected_variables = candidate_variables[:features_to_annotate]
                feature_benefit_costs = [(var, 1.0, 1.0, 1.0) for var in selected_variables]
            else:
                raise ValueError(f"Unknown feature strategy: {feature_strategy}")
            
            print(f"Selected {len(selected_variables)} features to annotate")
            
            # Check for overlaps with test set
            test_example_idx = example_idx % len(dataset_test)
            if test_example_idx not in test_overlap_annotations:
                test_overlap_annotations[test_example_idx] = []
                
            for i, variable_id in enumerate(selected_variables):
                example_idx, position = arena._parse_variable_id(variable_id)
                annotation_success = arena.observe_position(example_idx, position)
                
                if annotation_success:
                    total_features_annotated += 1
                else:
                    print(f"Failed to observe position {position} for example {example_idx}")
                
                if i < len(feature_benefit_costs):
                    var_id, benefit, cost, ratio, *_ = feature_benefit_costs[i]
                    cycle_benefit_cost_ratios.append(ratio)
                    cycle_observation_costs.append(cost)
                
                # Check for position overlap with test set
                test_positions = dataset_test.get_masked_positions(test_example_idx)
                if position in test_positions:
                    test_overlap_annotations[test_example_idx].append(position)
                
                # Make a prediction for training
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        print(f"Total features annotated in cycle {cycle_count+1}: {total_features_annotated}")
        
        # Training step
        if total_features_annotated > 0:
            print(f"Training on {total_features_annotated} annotated features in cycle {cycle_count+1}")
            training_metrics = arena.train(
                epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, 
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
        else:
            print(f"No features annotated in cycle {cycle_count+1}, skipping training")
            metrics['training_losses'].append(0.0)
        
        # Validation evaluation
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        # Test set evaluation - expected loss without annotations
        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
        
        # Create a clone of the test set with overlap annotations applied
        annotated_test_dataset = copy.deepcopy(dataset_test)
        annotations_applied = 0
        
        # Temporarily set up an arena for the annotated test set
        test_arena = AnnotationArena(model, device)
        test_arena.set_dataset(annotated_test_dataset)
        
        # Apply annotations to overlapping examples
        for test_idx, positions in test_overlap_annotations.items():
            for pos in positions:
                if test_arena.observe_position(test_idx, pos):
                    annotations_applied += 1
        
        # Evaluate with overlap annotations
        if annotations_applied > 0:
            test_arena.set_dataset(annotated_test_dataset)
            annotated_test_metrics = test_arena.evaluate(list(range(len(annotated_test_dataset))))
            metrics['test_annotated_losses'].append(annotated_test_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
        
        # Update metrics
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        cycle_count += 1
        
    metrics['test_metrics'] = test_metrics

    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics

# Gradient All Experiment Runner
def run_gradient_all_observe_experiment(dataset_train, dataset_val, dataset_test, 
                                      model, cycles=5, examples_per_cycle=10,
                                      epochs_per_cycle=3, batch_size=8, lr=1e-4,
                                      device=None, resample_validation=False,
                                      run_until_exhausted=False):
    """
    Run gradient selection with all positions observed.
    """

    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    
    # Initial test set evaluation
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
    metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
    
    max_cycles = float('inf') if run_until_exhausted else cycles
    
    while cycle_count < max_cycles:
        if not active_pool:
            print(f"Active pool exhausted after {cycle_count} cycles")
            break
            
        print(f"=== Cycle {cycle_count+1}/{cycles if not run_until_exhausted else 'until exhausted'} ===")
        print(f"Active pool size: {len(active_pool)}")
        arena.set_dataset(dataset_train)
        
        # Filter active pool to remove examples with no masked positions
        valid_active_pool = []
        for idx in active_pool:
            if dataset_train.get_masked_positions(idx):
                valid_active_pool.append(idx)
            else:
                print(f"Removing example {idx} from active pool (no masked positions)")
        
        if len(valid_active_pool) < len(active_pool):
            print(f"Filtered active pool from {len(active_pool)} to {len(valid_active_pool)} examples with masked positions")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions left in active pool")
            break
            
        if resample_validation and cycle_count > 0:
            dataset_val, active_pool = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_only", update_percentage=20
            )

        active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_pool]
        active_pool_subset = AnnotationDataset(active_pool_examples)
        
        # Select examples using gradient alignment
        gradient_strategy = GradientSelectionStrategy(model, device)
        selected_indices, alignment_scores = gradient_strategy.select_examples(
            active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
            val_dataset=dataset_val, num_samples=3, batch_size=batch_size
        )
        selected_examples = [active_pool[idx] for idx in selected_indices]
        
        print(f"Selected {len(selected_examples)} examples")
        
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        annotated_examples.extend(selected_examples)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        # Add selected examples to validation set if requested
        if resample_validation:
            dataset_val, _ = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_selected", selected_examples=selected_examples
            )
        
        total_features_annotated = 0
        cycle_observation_costs = []
        
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            masked_positions = dataset_train.get_masked_positions(example_idx)
            
            if not masked_positions:
                print(f"No masked positions found for example {example_idx}")
                continue
                
            print(f"Example {example_idx}: {len(masked_positions)} masked positions")
            
            # Check for overlaps with test set
            test_example_idx = example_idx % len(dataset_test)
            if test_example_idx not in test_overlap_annotations:
                test_overlap_annotations[test_example_idx] = []
            
            for position in masked_positions:
                annotation_success = arena.observe_position(example_idx, position)
                
                if annotation_success:
                    total_features_annotated += 1
                else:
                    print(f"Failed to observe position {position} for example {example_idx}")
                    
                cycle_observation_costs.append(1.0)

                # Check for position overlap with test set
                test_positions = dataset_test.get_masked_positions(test_example_idx)
                if position in test_positions:
                    test_overlap_annotations[test_example_idx].append(position)
                
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        print(f"Total features annotated in cycle {cycle_count+1}: {total_features_annotated}")
        
        # Training step
        if total_features_annotated > 0:
            print(f"Training on {total_features_annotated} annotated features in cycle {cycle_count+1}")
            training_metrics = arena.train(
                epochs=epochs_per_cycle, batch_size=batch_size, lr=lr,
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
        else:
            print(f"No features annotated in cycle {cycle_count+1}, skipping training")
            metrics['training_losses'].append(0.0)
        
        # Validation evaluation
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        # Test set evaluation - expected loss without annotations
        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
        
        # Create a clone of the test set with overlap annotations applied
        annotated_test_dataset = copy.deepcopy(dataset_test)
        annotations_applied = 0
        
        # Temporarily set up an arena for the annotated test set
        test_arena = AnnotationArena(model, device)
        test_arena.set_dataset(annotated_test_dataset)
        
        # Apply annotations to overlapping examples
        for test_idx, positions in test_overlap_annotations.items():
            for pos in positions:
                if test_arena.observe_position(test_idx, pos):
                    annotations_applied += 1
        
        # Evaluate with overlap annotations
        if annotations_applied > 0:
            test_arena.set_dataset(annotated_test_dataset)
            annotated_test_metrics = test_arena.evaluate(list(range(len(annotated_test_dataset))))
            metrics['test_annotated_losses'].append(annotated_test_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
        
        # Update metrics
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(1.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs))
        
        cycle_count += 1
        
    # Final test evaluation
    metrics['test_metrics'] = test_metrics
    
    # Get arena metrics history
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics

# All Observe Random Experiments
def run_all_observe_experiment(dataset_train, dataset_val, dataset_test, 
                              example_strategy, model,
                              cycles=5, examples_per_cycle=10,
                              epochs_per_cycle=3, batch_size=8, lr=1e-4,
                              device=None, resample_validation=False,
                              run_until_exhausted=False):
    """Run experiment with all positions observed."""
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    
    # Initial test set evaluation
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
    metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
    
    max_cycles = float('inf') if run_until_exhausted else cycles
    
    while cycle_count < max_cycles:
        if not active_pool:
            print(f"Active pool exhausted after {cycle_count} cycles")
            break
            
        print(f"=== Cycle {cycle_count+1}/{cycles if not run_until_exhausted else 'until exhausted'} ===")
        print(f"Active pool size: {len(active_pool)}")
        arena.set_dataset(dataset_train)
        
        # Filter active pool to remove examples with no masked positions
        valid_active_pool = []
        for idx in active_pool:
            if dataset_train.get_masked_positions(idx):
                valid_active_pool.append(idx)
            else:
                print(f"Removing example {idx} from active pool (no masked positions)")
        
        if len(valid_active_pool) < len(active_pool):
            print(f"Filtered active pool from {len(active_pool)} to {len(valid_active_pool)} examples with masked positions")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions left in active pool")
            break
        
        if resample_validation and cycle_count > 0:
            dataset_val, active_pool = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_only", update_percentage=20
            )
        
        # Select examples based on strategy
        if example_strategy == "random":
            selected_examples = random.sample(active_pool, min(examples_per_cycle, len(active_pool)))
        elif example_strategy == "gradient":
            gradient_strategy = GradientSelectionStrategy(model, device)

            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_pool]
            active_pool_subset = AnnotationDataset(active_pool_examples)
    
            selected_indices, _ = gradient_strategy.select_examples(
                active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            selected_examples = [active_pool[idx] for idx in selected_indices]
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        print(f"Selected {len(selected_examples)} examples")
        
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        annotated_examples.extend(selected_examples)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        # Add selected examples to validation set if requested
        if resample_validation:
            dataset_val, _ = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_selected", selected_examples=selected_examples
            )
        
        total_features_annotated = 0
        cycle_observation_costs = []
        
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            masked_positions = dataset_train.get_masked_positions(example_idx)
            
            if not masked_positions:
                print(f"No masked positions found for example {example_idx}")
                continue
                
            print(f"Example {example_idx}: {len(masked_positions)} masked positions")
            
            # Check for overlaps with test set
            test_example_idx = example_idx % len(dataset_test)
            if test_example_idx not in test_overlap_annotations:
                test_overlap_annotations[test_example_idx] = []
            
            for position in masked_positions:
                annotation_success = arena.observe_position(example_idx, position)
                
                if annotation_success:
                    total_features_annotated += 1
                else:
                    print(f"Failed to observe position {position} for example {example_idx}")
                
                variable_id = f"example_{example_idx}_position_{position}"
                cost = arena.variables.get(variable_id, {}).get("cost", 1.0)
                cycle_observation_costs.append(cost)

                # Check for position overlap with test set
                test_positions = dataset_test.get_masked_positions(test_example_idx)
                if position in test_positions:
                    test_overlap_annotations[test_example_idx].append(position)
                
                arena.predict(variable_id, train=True)
        
        print(f"Total features annotated in cycle {cycle_count+1}: {total_features_annotated}")
        
        # Training step
        if total_features_annotated > 0:
            print(f"Training on {total_features_annotated} annotated features in cycle {cycle_count+1}")
            training_metrics = arena.train(
                epochs=epochs_per_cycle, batch_size=batch_size, lr=lr,
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
        else:
            print(f"No features annotated in cycle {cycle_count+1}, skipping training")
            metrics['training_losses'].append(0.0)
        
        # Validation evaluation
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        # Test set evaluation - expected loss without annotations
        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
        
        # Create a clone of the test set with overlap annotations applied
        annotated_test_dataset = copy.deepcopy(dataset_test)
        annotations_applied = 0
        
        # Temporarily set up an arena for the annotated test set
        test_arena = AnnotationArena(model, device)
        test_arena.set_dataset(annotated_test_dataset)
        
        # Apply annotations to overlapping examples
        for test_idx, positions in test_overlap_annotations.items():
            for pos in positions:
                if test_arena.observe_position(test_idx, pos):
                    annotations_applied += 1
        
        # Evaluate with overlap annotations
        if annotations_applied > 0:
            test_arena.set_dataset(annotated_test_dataset)
            annotated_test_metrics = test_arena.evaluate(list(range(len(annotated_test_dataset))))
            metrics['test_annotated_losses'].append(annotated_test_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
        
        # Update metrics
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(1.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs))
        
        cycle_count += 1
        
    # Final test evaluation
    metrics['test_metrics'] = test_metrics
    
    # Get arena metrics history
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics

def main():
    
    parser = argparse.ArgumentParser(description="Run Active Learning Experiments with AnnotationArena.")
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
    parser.add_argument("--run_until_exhausted", action="store_true", help="Run until annotation pool is exhausted")
    args = parser.parse_args()
    
    # Set up paths and device
    base_path = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
    data_path = os.path.join(base_path, "data")
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = Imputer(
        question_num=7, max_choices=5, encoder_layers_num=6,
        attention_heads=4, hidden_dim=64, num_annotator=18, 
        annotator_embedding_dim=19, dropout=0.1
    ).to(device)
    
    experiment_results = {}

    if args.experiment == "all" or args.experiment == "gradient_fast_voi":
        print("\n=== Running Gradient-FastVOI Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="fast_voi", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["gradient_fast_voi"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_fast_voi.pth"))
        with open(os.path.join(results_path, "gradient_fast_voi.json"), "w") as f:
            json.dump(results, f, indent=4)

    if args.experiment == "all" or args.experiment == "gradient_all":
        print("\n=== Running Gradient-All Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_gradient_all_observe_experiment(
            active_pool_dataset, val_dataset, test_dataset,
            model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["gradient_all"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_all.pth"))
        with open(os.path.join(results_path, "gradient_all.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "random_all":
        print("\n=== Running Random-All Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_all_observe_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="random", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["random_all"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "random_all.pth"))
        with open(os.path.join(results_path, "random_all.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "random_5":
        print("\n=== Running Random-5 Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="random", feature_strategy="random", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["random_5"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "random_5.pth"))
        with open(os.path.join(results_path, "random_5.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "gradient_sequential":
        print("\n=== Running Gradient-Sequential Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="sequential", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["gradient_sequential"] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_sequential.pth"))
        with open(os.path.join(results_path, "gradient_sequential.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if args.experiment == "all" or args.experiment == "gradient_voi":
        print("\n=== Running Gradient-VOI Experiment ===")

        # Load datasets
        data_manager = DataManager()
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0)
        
        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        model_copy = copy.deepcopy(model)
        results = run_experiment(
            active_pool_dataset, val_dataset, test_dataset, 
            example_strategy="gradient", feature_strategy="voi", model=model_copy,
            cycles=args.cycles, examples_per_cycle=args.examples_per_cycle, 
            features_per_example=args.features_per_example,
            epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
            device=device, resample_validation=args.resample_validation,
            run_until_exhausted=args.run_until_exhausted
        )
        experiment_results["gradient_voi"] = results
        
        # Save model and results
        torch.save(model_copy.state_dict(), os.path.join(models_path, "gradient_voi.pth"))
        with open(os.path.join(results_path, "gradient_voi.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if experiment_results:
        
        with open(os.path.join(results_path, "combined_results.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"Results saved to {results_path}")

        create_plots()

        print(f"Plots Generated!")
        
if __name__ == "__main__":
    main()
