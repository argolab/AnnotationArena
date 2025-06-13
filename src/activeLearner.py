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
from feature_recorder import FeatureRecorder

# Library Imports
from annotationArena import *
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from visualizations import *
from imputer import Imputer
from imputer_embedding import ImputerEmbedding
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy,
    EntropyExampleSelectionStrategy,
    EntropyFeatureSelectionStrategy,
    BADGESelectionStrategy,
    ArgmaxVOISelectionStrategy
) 

def run_experiment(
    dataset_train, dataset_val, dataset_test, 
    example_strategy, model,
    feature_strategy=None,
    cycles=5, 
    examples_per_cycle=10, 
    features_per_example=None,
    observe_all_features=False,
    epochs_per_cycle=3, 
    batch_size=8, 
    lr=1e-4,
    device=None, 
    resample_validation=False, 
    loss_type="cross_entropy",
    run_until_exhausted=False,
    gradient_top_only=False,
    cold_start=False,
    use_feature=False
):
    """
    Unified experiment runner for all active learning strategies.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        dataset_test: Test dataset
        example_strategy: Strategy for selecting examples ("random", "gradient", "entropy", "badge", etc.)
        model: Model to use for predictions
        feature_strategy: Strategy for selecting features (None for observe_all_features=True)
        cycles: Number of active learning cycles
        examples_per_cycle: Number of examples to select per cycle
        features_per_example: Number of features to select per example (ignored if observe_all_features=True)
        observe_all_features: If True, observe all features of selected examples
        epochs_per_cycle: Number of training epochs per cycle
        batch_size: Batch size for training
        lr: Learning rate for training
        device: Device to use for computations
        resample_validation: Whether to resample validation set during training
        loss_type: Type of loss to use for VOI calculation
        run_until_exhausted: Whether to run until the active pool is exhausted
        gradient_top_only: Whether to use only the top gradient for example selection
        cold_start: If True, keep annotated examples in the active pool for potential reselection
        
    Returns:
        dict: Metrics and results from the experiment
    """

    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    if use_feature:
        feature_recorder = FeatureRecorder(model, device)
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

    active_pool_list = [dataset_train.get_data_entry(idx) for idx in range(len(dataset_train))]
    val_list = [dataset_val.get_data_entry(idx) for idx in range(len(dataset_val))]
    dataset_train = AnnotationDataset(active_pool_list + val_list)
    
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
            dataset_val, active_pool, val_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, list(set(annotated_examples)), 
                strategy="add_selected", update_percentage=20
            ) 
        active_set_size = 100
        print(f"Applying dynamic K-centers to select {min(active_set_size, len(active_pool))} from {len(active_pool)} examples...")
        if len(active_pool) > active_set_size:
            # Extract model embeddings for current pool
            model_embeddings = extract_model_embeddings(dataset_train, active_pool, model, device)
            selected_subset_indices = greedy_k_centers(model_embeddings, active_set_size, random_seed=42 + cycle_count)
            active_subset = [active_pool[i] for i in selected_subset_indices]
            print(f"K-centers selected {len(active_subset)} diverse examples from pool")
        else:
            active_subset = active_pool.copy()
            print(f"Pool smaller than target, using all {len(active_subset)} examples")
        
        
        # Debug logging
        print(f"DEBUG: Active subset size: {len(active_subset)}")
        
        # Select examples based on strategy
        if example_strategy == "random":
            selected_examples = random.sample(active_subset, min(examples_per_cycle, len(active_subset)))

        elif example_strategy in ["gradient", "entropy", "badge"]:
            strategy_class = SelectionFactory.create_example_strategy(example_strategy, model, device, gradient_top_only=gradient_top_only)
            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_subset]
            active_pool_subset = AnnotationDataset(active_pool_examples)
            
            selected_indices, selected_scores = strategy_class.select_examples(
                active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            
            selected_examples = [active_subset[idx] for idx in selected_indices]

            if example_strategy == "gradient":
                validation_gradient_cache = strategy_class.validation_grad_samples

        elif example_strategy == "combine":
            print("Debug: running combined strategy")
            strategy_class = SelectionFactory.create_example_strategy(example_strategy, model, device, gradient_top_only=gradient_top_only)
            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_subset]
            active_pool_subset = AnnotationDataset(active_pool_examples)

            num_to_select = examples_per_cycle * features_per_example
            
            selected_indices, _ = strategy_class.select_examples(
                active_pool_subset, num_to_select=num_to_select, #TODO: should change this to using argument later on
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )

            print(selected_indices)
            
            selected_examples = list(set([active_subset[idx[0]] for idx in selected_indices]))

        elif example_strategy == "gradient_combine":
            print("Debug: running combined strategy")
            strategy_class = SelectionFactory.create_example_strategy("gradient", model, device, gradient_top_only=gradient_top_only)
            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_subset]
            active_pool_subset = AnnotationDataset(active_pool_examples)
            num_to_select = examples_per_cycle * features_per_example
            selected_indices, selected_scores = strategy_class.select_examples(
                active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )

            selected_examples = [active_subset[idx] for idx in selected_indices]
            selected_list = [dataset_train.get_data_entry(idx) for idx in selected_examples]
            selected_set = AnnotationDataset(selected_list)

            strategy_class = SelectionFactory.create_example_strategy("combine", model, device)
            
            selected_indices, _ = strategy_class.select_examples(
                selected_set, num_to_select=num_to_select,
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )

            print(selected_indices)
            
            selected_examples = list(set([active_subset[idx[0]] for idx in selected_indices]))
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        print(f"Selected {len(selected_examples)} examples")

        if use_feature:
            print("Recording features")
            selected_with_scores = list(zip(selected_examples, selected_scores))
            feature_recorder.record_cycle_features(
                cycle_num=cycle_count,
                dataset=dataset_train,
                active_pool=active_pool,
                annotated_examples=annotated_examples,
                val_dataset=dataset_val,
                selected_examples_with_scores=selected_with_scores
            )

            print("Finished")
        
        #active_pool = [idx for idx in active_pool if idx not in selected_examples]
        
        # Add selected examples to annotated_examples list
        for example in selected_examples:
            if example not in annotated_examples:
                annotated_examples.append(example)
        
        
        # Add selected examples to validation set if requested
        if resample_validation:
            dataset_val, active_pool_after_resample, val_examples = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="add_selected_partial", selected_examples=selected_examples, validation_set_size=len(dataset_val),
            )
            active_pool = active_pool_after_resample
        metrics['remaining_pool_size'].append(len(active_pool))
        
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        selected_variables_for_recorder = []
        # Annotate selected examples
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
        print(selected_examples)
        if not example_strategy in ["combine", "gradient_combine"]:
            for example_idx in tqdm(selected_examples, desc="Selecting features"): 
                masked_positions = dataset_train.get_masked_positions(example_idx)
                if not masked_positions:
                    print(f"No masked positions found for example {example_idx}")
                    continue
                    
                if observe_all_features:

                    # Observe all features of the selected examples
                    positions_to_annotate = masked_positions
                    position_benefit_costs = [(pos, 1.0, 1.0, 1.0) for pos in positions_to_annotate]

                else:

                    # Select features using the specified feature strategy
                    if features_per_example is None:
                        features_per_example = 5  # Default value
                    
                    features_to_annotate = min(features_per_example, len(masked_positions))
                    
                    candidate_variables = [
                        f"example_{example_idx}_position_{pos}" 
                        for pos in masked_positions
                    ]
                    
                    if feature_strategy == "random":
                        selected_variables = random.sample(candidate_variables, features_to_annotate)
                        position_benefit_costs = []
                        for var in selected_variables:
                            _, pos = arena._parse_variable_id(var)
                            position_benefit_costs.append((pos, 1.0, 1.0, 1.0))

                    elif feature_strategy == "sequential":
                        positions_to_annotate = masked_positions[:features_to_annotate]
                        position_benefit_costs = [(pos, 1.0, 1.0, 1.0) for pos in positions_to_annotate]

                    elif feature_strategy in ["voi", "fast_voi", "entropy", "voi_argmax"]:
                        feature_suggestions = arena.suggest(
                            candidate_variables=candidate_variables,
                            strategy=feature_strategy, loss_type=loss_type
                        )
                        if not feature_suggestions:
                            print(f"{feature_strategy} strategy returned no suggestions for example {example_idx}")
                            continue
                        
                        # Extract positions and benefit/cost metrics
                        position_benefit_costs = []
                        for var, benefit, cost, ratio, *extra in feature_suggestions[:features_to_annotate]:
                            _, pos = arena._parse_variable_id(var)
                            position_benefit_costs.append((pos, benefit, cost, ratio))
                    
                    elif feature_strategy == "gradient":
                        feature_selector = SelectionFactory.create_feature_strategy(feature_strategy, model)
                        feature_selector.set_validation_gradients(validation_gradient_cache)
                        selections = feature_selector.select_features(example_idx, dataset_train, num_to_select=features_per_example, val_dataset=dataset_val, batch_size=batch_size)
                        position_benefit_costs = selections
                    
                    else:
                        raise ValueError(f"Unknown feature strategy: {feature_strategy}")
            
                # Check for overlaps with test set
                test_example_idx = example_idx % len(dataset_test)
                if test_example_idx not in test_overlap_annotations:
                    test_overlap_annotations[test_example_idx] = []
                
                # Observe the selected positions
                for pos_data in position_benefit_costs:
                    position = pos_data[0]
                    
                    annotation_success = arena.observe_position(example_idx, position)
                    
                    if annotation_success:
                        total_features_annotated += 1
                        selected_variables_for_recorder.append((example_idx, position))
                    else:
                        print(f"Failed to observe position {position} for example {example_idx}")
                    
                    # Record benefit/cost metrics if available
                    if len(pos_data) >= 4:
                        _, benefit, cost, ratio = pos_data[:4]
                        cycle_benefit_cost_ratios.append(ratio)
                        cycle_observation_costs.append(cost)
                    else:
                        # Default values if not provided
                        cycle_benefit_cost_ratios.append(1.0)
                        cycle_observation_costs.append(1.0)
                    
                    # Check for position overlap with test set
                    test_positions = dataset_test.get_masked_positions(test_example_idx)
                    if position in test_positions:
                        test_overlap_annotations[test_example_idx].append(position)
                    
                    # Make a prediction for training
                    variable_id = f"example_{example_idx}_position_{position}"
                    arena.predict(variable_id, train=True)
        else:
            print("Debug: should print this for combined strategy")
            for example_idx, pos in selected_indices:
                actual_idx = active_pool[example_idx]
                annotation_success = arena.observe_position(actual_idx, pos)

                test_example_idx = example_idx % len(dataset_test)
                if test_example_idx not in test_overlap_annotations:
                    test_overlap_annotations[test_example_idx] = []

                test_positions = dataset_test.get_masked_positions(test_example_idx)
                if pos in test_positions:
                    test_overlap_annotations[test_example_idx].append(pos)
                
                if annotation_success:
                    total_features_annotated += 1
                    selected_variables_for_recorder.append((actual_idx, pos))
                else:
                    print(f"Failed to observe position {position} for example {example_idx}")
                cycle_benefit_cost_ratios.append(1.0)
                cycle_observation_costs.append(1.0) #TODO: might change this later on
                variable_id = f"example_{example_idx}_position_{pos}"
                arena.predict(variable_id, train=True)

        print(f"Total features annotated in cycle {cycle_count+1}: {total_features_annotated}")



        if use_feature:
            feature_recorder.update_from_selections(selected_examples, selected_variables_for_recorder)
        
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
        if use_feature:
            feature_recorder.save_features("features.pik")
        cycle_count += 1
        
    metrics['test_metrics'] = test_metrics

    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]

    
    return metrics

def extract_model_embeddings(dataset, example_indices, model, device):
    """Extract embeddings using the current imputer model state."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for idx in example_indices:
            known_questions, inputs, answers, annotators, questions, model_embeddings = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            annotators = annotators.unsqueeze(0).to(device)
            questions = questions.unsqueeze(0).to(device)
            if model_embeddings is not None:
                model_embeddings = model_embeddings.unsqueeze(0).to(device)
            
            # Get model's internal representation
            if hasattr(model, 'encoder'):
                if model_embeddings is not None:
                    feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions, model_embeddings)
                else:
                    feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions)
                
                # Use the feature representation as embedding
                embedding = feature_x[0].mean(dim=0).cpu().numpy()  # Average across positions
            else:
                # Fallback to statistical features
                entry = dataset.get_data_entry(idx)
                inputs_array = np.array(entry['input'])
                answer_dists = inputs_array[:, 1:]
                embedding = np.concatenate([
                    np.mean(answer_dists, axis=0),
                    np.std(answer_dists, axis=0),
                    [np.mean(answer_dists), np.std(answer_dists)]
                ])
            
            embeddings.append(embedding)
    
    return np.array(embeddings)

def greedy_k_centers(features, k, random_seed=42):
    """Greedy K-centers algorithm for diverse subset selection."""
    np.random.seed(random_seed)
    n = len(features)
    
    if k >= n:
        return list(range(n))
    
    selected = [np.random.randint(n)]
    
    for _ in range(k - 1):
        distances = []
        for i in range(n):
            if i in selected:
                distances.append(0)
            else:
                min_dist = min(np.linalg.norm(features[i] - features[j]) for j in selected)
                distances.append(min_dist)
                
        next_center = np.argmax(distances)
        selected.append(next_center)
    
    return selected

def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Experiments with AnnotationArena.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of active learning cycles")
    parser.add_argument("--examples_per_cycle", type=int, default=20, help="Number of examples to select per cycle")
    parser.add_argument("--features_per_example", type=int, default=5, help="Number of features to select per example")
    parser.add_argument("--epochs_per_cycle", type=int, default=3, help="Number of training epochs per cycle")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--experiment", type=str, default="all", 
                      help="Experiment to run (all, random_all, random_5, gradient_all, gradient_sequential, gradient_voi, "
                           "gradient_fast_voi, entropy_all, entropy_5, badge_all, badge_cold_start, gradient_voi_argmax)")
    parser.add_argument("--resample_validation", action="store_true", help="Resample validation set on each cycle")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", help="Type of loss to use (cross_entropy, l2)")
    parser.add_argument("--run_until_exhausted", action="store_true", help="Run until annotation pool is exhausted")
    parser.add_argument("--dataset", type=str, default="hanna", help="Dataset to run the experiment")
    parser.add_argument("--runner", type=str, default="prabhav", help="Pass name to change directory paths! (prabhav/haojun)")
    parser.add_argument("--cold_start", type=bool, default=False, help="Start with no annotation (true) or partial annotation (false)")
    parser.add_argument("--use_embedding", type=bool, default=False, help="Use embeddings for texts")
    args = parser.parse_args()
    
    # Set up paths and device
    if args.runner == 'prabhav':
        base_path = '/export/fs06/psingh54/ActiveRubric-Internal/outputs'
    else:
        base_path = "outputs"

    dataset = args.dataset
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.use_embedding:
        ModelClass = ImputerEmbedding
    else:
        ModelClass = Imputer
    
    # Initialize model
    if dataset == "hanna":
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)

    elif dataset == "llm_rubric":
        model = ModelClass(
            question_num=9, max_choices=4, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=24, 
            annotator_embedding_dim=24, dropout=0.1
        ).to(device)

    else:
        model = ModelClass(
            question_num=5, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=1, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)

    
    experiment_results = {}
    
    # Configure experiments to run
    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = [
            "random_all", "badge_all", "entropy_all", "gradient_all", "gradient_all_top_only"
        ]
        # experiments_to_run = [
        # "gradient_voi_argmax", "gradient_voi", "gradient_fast_voi", "gradient_sequential"
        # ]
        # experiments_to_run = [
        #     "gradient_random_cold_start", "gradient_voi_cold_start", "gradient_fast_voi_cold_start", "gradient_sequential_cold_start", 
        #     "gradient_all_top_only", "gradient_sequential_top_only", "gradient_voi_top_only", "gradient_fast_voi_top_only", "gradient_random_top_only", 
        #     "entropy_all", "entropy_5", "random_all", "random_5", "gradient_all", 
        #     "gradient_sequential", "gradient_voi", "gradient_fast_voi", "gradient_random", 
        #     "badge_all", "badge_cold_start", "gradient_voi_argmax"
        # ]
    else:
        experiments_to_run = [args.experiment]
    
    # Run each selected experiment
    for experiment in experiments_to_run:
        
        print(f"\n=== Creating Data for {experiment} Experiment ===")
        
        if args.runner == "prabhav":
            data_manager = DataManager(base_path + '/data/')
        else:
            data_manager = DataManager(base_path + f'/data_{dataset}/')

        if dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=dataset, cold_start=args.cold_start, use_embedding=args.use_embedding)
        elif dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=225, initial_train_ratio=0.0, dataset=dataset, cold_start=args.cold_start, use_embedding=args.use_embedding)
        elif dataset == "gaussian":
            data_manager.prepare_data(dataset=dataset)
        else:
            raise ValueError("Unsupported dataset")
        
        model_copy = copy.deepcopy(model)

        print(f"\n=== Created Data for {experiment} Experiment ===")

        if experiment == "random_all":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "badge_all":

            print(f"\n=== Running {experiment} Experiment ===")

            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="badge", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "badge_cold_start":
            # BADGE with cold_start=True for comparison with no initial annotations
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="badge", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                cold_start=True
            )

        elif experiment == "entropy_all":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_all":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")
        
            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_all_top_only":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")
        
            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "random_5":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
                f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_sequential":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="sequential", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_sequential_top_only":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="sequential", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "gradient_sequential_cold_start":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="sequential", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                cold_start=True
            )

        elif experiment == "gradient_random":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "gradient_random_cold_start":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=False,
                cold_start=True
            )

        elif experiment == "gradient_random_top_only":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "gradient_voi":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_voi_argmax":
            # Gradient alignment with ArgmaxVOI feature selection
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi_argmax", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_voi_cold_start":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=True
            )

        elif experiment == "gradient_voi_top_only":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "gradient_fast_voi":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="fast_voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )

        elif experiment == "gradient_fast_voi_cold_start":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="fast_voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=True
            )

        elif experiment == "gradient_fast_voi_top_only":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="fast_voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "both_level_gradient":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "gradient_combine":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
            
            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient_combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        elif experiment == "entropy_5":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])

            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", feature_strategy="entropy", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start
            )
        
        elif experiment == "combine":
            train_dataset = AnnotationDataset(data_manager.paths['train'])
            val_dataset = AnnotationDataset(data_manager.paths['validation'])
            test_dataset = AnnotationDataset(data_manager.paths['test'])
            active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])


            results = run_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                gradient_top_only=True, cold_start=args.cold_start
            )

        else:
            print(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results
        
        # Save model and results
        torch.save(model_copy.state_dict(), os.path.join(models_path, f"{experiment}.pth"))
        file_name = experiment
        if not "cold_start" in experiment and args.cold_start:
            file_name += "_cold_start"
        if args.use_embedding:
            file_name += "_with_embedding"
        with open(os.path.join(results_path, f"{file_name}.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    if experiment_results:
        with open(os.path.join(results_path, "combined_results.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"Results saved to {results_path}")

        # create_plots()
        # print(f"Plots Generated!")
        
if __name__ == "__main__":
    main()
    #create_plots()