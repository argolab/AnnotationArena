"""
Enhanced active learning experiments combining best features from both standard and noisy versions.
Supports dynamic K-centers, enhanced validation resampling, and comparison experiments.
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
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances

from annotationArena import *
from utils_prabhav import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from visualizations import *
from imputer import Imputer
from imputer_embedding import ImputerEmbedding
from selection_fixed import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy,
    EntropyExampleSelectionStrategy,
    EntropyFeatureSelectionStrategy,
    BADGESelectionStrategy,
    ArgmaxVOISelectionStrategy,
    VariableGradientSelectionStrategy
)
# from feature_recorder import FeatureRecorder

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_embeddings_features(dataset_entries, model_name='all-MiniLM-L6-v2'):
    """Extract sentence transformer embeddings for K-centers algorithm."""
    embedding_model = SentenceTransformer(model_name)
    features = []
    
    for entry in dataset_entries:
        if 'text_embedding' in entry and entry['text_embedding']:
            embedding = np.array(entry['text_embedding'][0])
        else:
            inputs = np.array(entry['input'])
            answer_dists = inputs[:, 1:] 
            mean_dist = np.mean(answer_dists, axis=0)
            std_dist = np.std(answer_dists, axis=0)
            entropy_per_pos = []
            
            for dist in answer_dists:
                if np.sum(dist) > 0:
                    normalized = dist / np.sum(dist)
                    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                    entropy_per_pos.append(entropy)
                else:
                    entropy_per_pos.append(0.0)

            embedding = np.concatenate([
                mean_dist, 
                std_dist, 
                [np.mean(entropy_per_pos), np.std(entropy_per_pos)]
            ])
        
        features.append(embedding)
    
    return np.array(features)

def extract_model_embeddings(dataset, example_indices, model, device):

    """Extract embeddings using the current imputer model state."""
    embeddings = []
    
    for idx in example_indices:
        entry = dataset.get_data_entry(idx)
        
        inputs = torch.tensor(entry['input'], dtype=torch.float32).unsqueeze(0).to(device)
        annotators = torch.tensor(entry['annotators'], dtype=torch.long).unsqueeze(0).to(device)
        questions = torch.tensor(entry['questions'], dtype=torch.long).unsqueeze(0).to(device)
        # text_embeddings = torch.tensor(entry['text_embedding'], dtype=torch.float32).to(device)
        text_embeddings = torch.tensor(entry['text_embedding'], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions, text_embeddings)
            
            # Create mask from input  
            mask = inputs[:, :, 0]  # Add this line
            
            for layer in model.encoder.layers:
                feature_x, param_x = layer(feature_x, param_x, questions, mask)  # Add questions, mask
                
            # Use mean of feature representations as embedding
            embedding = feature_x.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
    
    return np.array(embeddings)

def greedy_k_centers(embeddings, k, random_seed=42):
    """Greedy K-centers algorithm for diverse subset selection."""
    np.random.seed(random_seed)
    n = len(embeddings)
    if k >= n:
        return list(range(n))
    
    distances = pairwise_distances(embeddings, metric='euclidean')
    
    centers = [np.random.randint(0, n)]
    
    for _ in range(k - 1):
        min_distances = np.inf * np.ones(n)
        
        for center in centers:
            min_distances = np.minimum(min_distances, distances[center])
        
        next_center = np.argmax(min_distances)
        centers.append(next_center)
    
    return centers

def run_enhanced_experiment(
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
    active_set_size=100,
    validation_set_size=50,
    target_questions=None,
    initial_train_dataset=None
):
    """Enhanced experiment runner with dynamic K-centers and improved validation resampling."""
    
    if initial_train_dataset is not None and len(initial_train_dataset) > 0:
        arena = AnnotationArena(model, device)
        print(f"Initial training on {len(initial_train_dataset)} clean examples...")
        arena.set_dataset(initial_train_dataset)
        
        for idx in range(len(initial_train_dataset)):
            arena.register_example(idx, add_all_positions=False)
            known_positions = initial_train_dataset.get_known_positions(idx)
            for pos in known_positions:
                arena.observe_position(idx, pos)
                variable_id = f"example_{idx}_position_{pos}"
                arena.predict(variable_id, train=True)
        
        arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr)
        print("Initial training completed!")
    else:
        arena = AnnotationArena(model, device)
        arena.set_dataset(dataset_train)

    # feature_recorder = FeatureRecorder(model, device)
    
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
        'remaining_pool_size': [],
        'active_subset_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0

    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
    metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
    
    metrics['examples_annotated'].append(0)
    metrics['features_annotated'].append(0)
    metrics['benefit_cost_ratios'].append(0.0)
    metrics['observation_costs'].append(0.0)
    metrics['remaining_pool_size'].append(len(active_pool))
    metrics['active_subset_size'].append(min(active_set_size, len(active_pool)))

    max_cycles = float('inf') if run_until_exhausted else cycles

    while cycle_count < max_cycles:
        
        if not active_pool:
            print(f"Active pool exhausted after {cycle_count} cycles")
            break
    
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle_count + 1}/{cycles}")
        print(f"{'='*60}")
        print(f"Active pool size: {len(active_pool)}")
        
        valid_active_pool = []
        for idx in active_pool:
            masked_positions = dataset_train.get_masked_positions(idx)
            if masked_positions:
                valid_active_pool.append(idx)
        
        if len(valid_active_pool) != len(active_pool):
            print(f"Filtered active pool: {len(valid_active_pool)} (removed examples with no masked positions)")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions remaining")
            break
        
        if resample_validation and cycle_count > 0:
            current_val_indices = list(range(len(dataset_val)))
            active_pool.extend(current_val_indices)
            
            dataset_val, active_pool, validation_example_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="balanced_fixed_size", 
                selected_examples=annotated_examples[-examples_per_cycle:] if annotated_examples else [],
                validation_set_size=validation_set_size
            )
        
        print(f"Applying dynamic K-centers to select {min(active_set_size, len(active_pool))} from {len(active_pool)} examples...")
        if len(active_pool) > active_set_size:
            model_embeddings = extract_model_embeddings(dataset_train, active_pool, model, device)
            selected_subset_indices = greedy_k_centers(model_embeddings, active_set_size, random_seed=42 + cycle_count)
            active_subset = [active_pool[i] for i in selected_subset_indices]
            print(f"K-centers selected {len(active_subset)} diverse examples from pool")
        else:
            active_subset = active_pool.copy()
            print(f"Pool smaller than target, using all {len(active_subset)} examples")
        
        metrics['active_subset_size'].append(len(active_subset))
        
        print(f"DEBUG: Active subset size: {len(active_subset)}")
        if len(active_subset) > 0:
            sample_idx = active_subset[0]
            sample_entry = dataset_train.get_data_entry(sample_idx)
            sample_features = len(sample_entry['input'])
        
        if example_strategy == "random":
            selected_examples = random.sample(active_subset, min(examples_per_cycle, len(active_subset)))
        elif example_strategy in ["gradient", "entropy", "badge"]:
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])
            
            example_selector = SelectionFactory.create_example_strategy(
                example_strategy, model, device, gradient_top_only=gradient_top_only
            )
            
            selected_indices, scores = example_selector.select_examples(
                active_subset_dataset, 
                num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val,
                num_samples=5,
                batch_size=batch_size
            )
            
            selected_examples = [active_subset[idx] for idx in selected_indices]

        elif example_strategy == "combine":
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])

            variable_selector = VariableGradientSelectionStrategy(model, device)

            # Calculate total features needed
            total_features_needed = examples_per_cycle * features_per_example

            # Request more variables than needed to ensure we have enough options
            # Some examples might not have enough maskable positions
            num_variables_to_request = min(total_features_needed * 3, len(active_subset) * 10)

            selected_variables, scores = variable_selector.select_examples(
                active_subset_dataset,
                num_to_select=num_variables_to_request,
                val_dataset=dataset_val,
                num_samples=5,
                batch_size=batch_size
            )

            print(f"Variable selector returned {len(selected_variables)} candidate variables")

            # Improved selection logic: Guarantee exactly total_features_needed features
            selected_examples_dict_fixed = {}
            total_features_selected = 0

            # First pass: Collect exactly the number of features we need
            for (local_idx, pos), score in zip(selected_variables, scores):
                if total_features_selected >= total_features_needed:
                    break
                    
                global_idx = active_subset[local_idx]
                
                # Initialize example if not seen
                if global_idx not in selected_examples_dict_fixed:
                    selected_examples_dict_fixed[global_idx] = []
                
                # Add feature if we haven't exceeded per-example limit
                if len(selected_examples_dict_fixed[global_idx]) < features_per_example:
                    selected_examples_dict_fixed[global_idx].append(pos)
                    total_features_selected += 1

            # Second pass: If we still need more features, relax per-example limit
            if total_features_selected < total_features_needed:
                print(f"First pass collected {total_features_selected}/{total_features_needed} features")
                print("Second pass: relaxing per-example limits to fill remaining slots")
                
                for (local_idx, pos), score in zip(selected_variables, scores):
                    if total_features_selected >= total_features_needed:
                        break
                        
                    global_idx = active_subset[local_idx]
                    
                    # Skip if already added this position
                    if global_idx in selected_examples_dict_fixed and pos in selected_examples_dict_fixed[global_idx]:
                        continue
                        
                    # Initialize example if not seen
                    if global_idx not in selected_examples_dict_fixed:
                        selected_examples_dict_fixed[global_idx] = []
                    
                    # Add feature (no per-example limit in second pass)
                    selected_examples_dict_fixed[global_idx].append(pos)
                    total_features_selected += 1

            # Third pass: If still not enough, add any remaining maskable positions
            if total_features_selected < total_features_needed:
                print(f"Second pass collected {total_features_selected}/{total_features_needed} features")
                print("Third pass: adding any remaining maskable positions")
                
                for example_idx in active_subset:
                    if total_features_selected >= total_features_needed:
                        break
                        
                    # Get all maskable positions for this example
                    masked_positions = dataset_train.get_masked_positions(example_idx)
                    
                    # Initialize if not seen
                    if example_idx not in selected_examples_dict_fixed:
                        selected_examples_dict_fixed[example_idx] = []
                    
                    # Add any positions we haven't selected yet
                    for pos in masked_positions:
                        if total_features_selected >= total_features_needed:
                            break
                        if pos not in selected_examples_dict_fixed[example_idx]:
                            selected_examples_dict_fixed[example_idx].append(pos)
                            total_features_selected += 1

            # Final selection: All examples that have at least one feature
            selected_examples = list(selected_examples_dict_fixed.keys())
            selected_examples = selected_examples[:examples_per_cycle] if len(selected_examples) > examples_per_cycle else selected_examples

            # Final count verification
            final_feature_count = sum(len(positions) for example, positions in selected_examples_dict_fixed.items() 
                                        if example in selected_examples)

            print(f"Variable gradient selection summary:")
            print(f"  Target features: {total_features_needed}")
            print(f"  Selected features: {final_feature_count}")
            print(f"  Selected examples: {len(selected_examples)}")
            print(f"  Avg features per example: {final_feature_count / len(selected_examples) if selected_examples else 0:.1f}")

            # Ensure selected_examples_dict contains only selected examples
            selected_examples_dict = {ex: selected_examples_dict_fixed[ex] for ex in selected_examples}
            
            # selected_examples_dict = {}
            # for (local_idx, pos), score in zip(selected_variables, scores):
            #     global_idx = active_subset[local_idx]
            #     if global_idx not in selected_examples_dict:
            #         selected_examples_dict[global_idx] = []
            #     selected_examples_dict[global_idx].append(pos)
            
            # selected_examples = list(selected_examples_dict.keys())[:examples_per_cycle]

        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        print(f"Selected {len(selected_examples)} examples for annotation")
        
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        selected_variables_info = []
        
        arena.set_dataset(dataset_train)
        
        for example_idx in tqdm(selected_examples, desc="Annotating selected examples"):
            arena.register_example(example_idx, add_all_positions=False)
            
            if observe_all_features:
                masked_positions = dataset_train.get_masked_positions(example_idx)
                for pos in masked_positions:
                    if arena.observe_position(example_idx, pos):
                        total_features_annotated += 1
                        selected_variables_info.append((example_idx, pos))
                        
                        test_entry = dataset_train.get_data_entry(example_idx)
                        test_question = test_entry['questions'][pos]
                        test_annotator = test_entry['annotators'][pos]
                        
                        if example_idx < len(dataset_test):
                            if example_idx not in test_overlap_annotations:
                                test_overlap_annotations[example_idx] = []
                            test_overlap_annotations[example_idx].append(pos)
                        
                        variable_id = f"example_{example_idx}_position_{pos}"
                        arena.predict(variable_id, train=True)
            
            elif feature_strategy and features_per_example:
                feature_selector = SelectionFactory.create_feature_strategy(feature_strategy, model, device)
                
                if example_strategy == "combine" and example_idx in selected_examples_dict:
                    selected_positions = selected_examples_dict[example_idx][:features_per_example]
                    selected_features = [(pos, 1.0, 1.0, 1.0) for pos in selected_positions]
                else:
                    feature_kwargs = {}
                    if target_questions is not None:
                        feature_kwargs['target_questions'] = target_questions
                    
                    selected_features = feature_selector.select_features(
                        example_idx, dataset_train, 
                        num_to_select=features_per_example,
                        loss_type=loss_type,
                        **feature_kwargs
                    )
                
                for feature_info in selected_features:
                    pos = feature_info[0]
                    benefit = feature_info[1] if len(feature_info) > 1 else 1.0
                    cost = feature_info[2] if len(feature_info) > 2 else 1.0
                    bc_ratio = feature_info[3] if len(feature_info) > 3 else 1.0
                    
                    if arena.observe_position(example_idx, pos):
                        total_features_annotated += 1
                        cycle_benefit_cost_ratios.append(bc_ratio)
                        cycle_observation_costs.append(cost)
                        selected_variables_info.append((example_idx, pos))
                        
                        test_entry = dataset_train.get_data_entry(example_idx)
                        test_question = test_entry['questions'][pos]
                        test_annotator = test_entry['annotators'][pos]
                        
                        if example_idx < len(dataset_test):
                            if example_idx not in test_overlap_annotations:
                                test_overlap_annotations[example_idx] = []
                            test_overlap_annotations[example_idx].append(pos)
                        
                        variable_id = f"example_{example_idx}_position_{pos}"
                        arena.predict(variable_id, train=True)
            
            else:
                masked_positions = dataset_train.get_masked_positions(example_idx)
                if masked_positions:
                    pos = random.choice(masked_positions)
                    if arena.observe_position(example_idx, pos):
                        total_features_annotated += 1
                        selected_variables_info.append((example_idx, pos))
                        
                        if example_idx < len(dataset_test):
                            if example_idx not in test_overlap_annotations:
                                test_overlap_annotations[example_idx] = []
                            test_overlap_annotations[example_idx].append(pos)
                        
                        variable_id = f"example_{example_idx}_position_{pos}"
                        arena.predict(variable_id, train=True)
        
        print(f"Total features annotated this cycle: {total_features_annotated}")
        
        # feature_recorder.record_cycle_features(
        #     cycle_count, dataset_train, active_subset, annotated_examples, 
        #     dataset_val, {idx: 1.0 for idx in selected_examples}
        # )
        
        # feature_recorder.update_from_selections(selected_examples, selected_variables_info)
        
        annotated_examples.extend(selected_examples)
        
        if not cold_start:
            for example_idx in selected_examples:
                if example_idx in active_pool:
                    active_pool.remove(example_idx)
        
        print(f"Training model for {epochs_per_cycle} epochs...")
        arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr)
        
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])

        # Print validation metrics for this cycle
        print(f"Validation - RMSE: {val_metrics['rmse']:.4f}, "
        f"Pearson: {val_metrics['pearson']:.4f}, "
        f"Expected Loss: {val_metrics['avg_expected_loss']:.4f}")
        
        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
        
        annotated_test_dataset = copy.deepcopy(dataset_test)
        annotations_applied = 0
        
        test_arena = AnnotationArena(model, device)
        test_arena.set_dataset(annotated_test_dataset)
        
        for test_idx, positions in test_overlap_annotations.items():
            for pos in positions:
                if test_arena.observe_position(test_idx, pos):
                    annotations_applied += 1

        if annotations_applied > 0:
            test_arena.set_dataset(annotated_test_dataset)
            annotated_test_metrics = test_arena.evaluate(list(range(len(annotated_test_dataset))))
            metrics['test_annotated_losses'].append(annotated_test_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
        
        print(f"Test loss: {test_metrics['avg_expected_loss']:.4f}")
        
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        cycle_count += 1
        
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    metrics['test_metrics'] = test_metrics
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    # feature_recorder.save_features("features.pik")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced Active Learning Experiments with AnnotationArena.")
    parser.add_argument("--experiment", type=str, default="gradient_voi", 
                       help="Experiment to run")
    parser.add_argument("--cycles", type=int, default=5, 
                       help="Number of active learning cycles")
    parser.add_argument("--examples_per_cycle", type=int, default=20, 
                       help="Number of examples to select per cycle")
    parser.add_argument("--features_per_example", type=int, default=5, 
                       help="Number of features to select per example")
    parser.add_argument("--epochs_per_cycle", type=int, default=3, 
                       help="Number of training epochs per cycle")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", 
                       help="Loss type for VOI calculation")
    parser.add_argument("--resample_validation", action="store_true", 
                       help="Whether to resample validation set during training")
    parser.add_argument("--run_until_exhausted", action="store_true", 
                       help="Whether to run until the active pool is exhausted")
    parser.add_argument("--dataset", type=str, default="hanna", 
                       help="Dataset to use")
    parser.add_argument("--runner", type=str, default="local", 
                       help="Runner identifier")
    parser.add_argument("--cold_start", type=bool, default=False, 
                       help="Use cold start approach")
    parser.add_argument("--use_embedding", type=bool, default=False, 
                       help="Use embeddings for texts")
    parser.add_argument("--active_set_size", type=int, default=100, 
                       help="Size of active subset selected by K-centers each cycle")
    parser.add_argument("--validation_set_size", type=int, default=50, 
                       help="Fixed size for validation set")
    
    args = parser.parse_args()
    
    if args.runner == 'prabhav':
        base_path = '/export/fs06/psingh54/ActiveRubric-Internal/outputs'
    else:
        base_path = "outputs"

    dataset = args.dataset
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, f"results_enhanced_{dataset}")
    os.makedirs(results_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Enhanced Active Learning Setup:")
    print(f"   Dataset: {dataset}")
    print(f"   K-centers active subset: {args.active_set_size}")
    print(f"   Validation set size: {args.validation_set_size}")
    print(f"   Use embeddings: {args.use_embedding}")

    if args.use_embedding:
        ModelClass = ImputerEmbedding
    else:
        ModelClass = Imputer
    
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
    
    experiment_results = {}
    
    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = [
            "random_all", "gradient_all", "entropy_all", "random_5", 
            "gradient_voi", "gradient_entropy", "entropy_voi", "gradient_sequential",
            "gradient_voi_q0_human", "gradient_voi_q0_both", "gradient_voi_all_questions",
            "variable_gradient_comparison"
        ]
    elif args.experiment == "comparison":
        experiments_to_run = [
            "gradient_voi_q0_human"
        ]
    else:
        experiments_to_run = [args.experiment]
    
    for experiment in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"RUNNING ENHANCED {experiment.upper()} EXPERIMENT")
        print(f"{'='*80}")
        
        model_copy = copy.deepcopy(model)
        
        if args.runner == "prabhav":
            data_manager = DataManager(base_path + '/data/')
        else:
            data_manager = DataManager(base_path + f'/data_{dataset}/')

        if dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding)
        elif dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=1000, initial_train_ratio=0.0, dataset=dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding)

        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
        initial_train_dataset = None
        if len(train_dataset) > 0:
            initial_train_dataset = train_dataset

        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
              f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        if experiment == "random_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "entropy_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", model=model_copy,
                observe_all_features=True,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "random_5":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_voi":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=args.cold_start, target_questions=[0],
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_voi_q0_human":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=args.cold_start, target_questions=[0],
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_voi_q0_both":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=args.cold_start, target_questions=[0],
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_voi_all_questions":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=args.cold_start, target_questions=[0, 1, 2, 3, 4, 5, 6],
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "variable_gradient_comparison":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, 
                cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_entropy":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="entropy", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "entropy_voi":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                cold_start=args.cold_start, target_questions=[0],
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        elif experiment == "gradient_sequential":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted, cold_start=args.cold_start,
                active_set_size=args.active_set_size, validation_set_size=args.validation_set_size,
                initial_train_dataset=initial_train_dataset
            )

        else:
            print(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, f"enhanced_{experiment}.pth"))
        
        file_name = f"enhanced_{experiment}"
        if args.use_embedding:
            file_name += "_with_embedding"
        
        with open(os.path.join(results_path, f"{file_name}.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment.upper()} SUMMARY")
        print(f"{'='*60}")
        print(f"Final validation loss: {results['val_losses'][-1]:.4f}")
        print(f"Final test loss: {results['test_expected_losses'][-1]:.4f}")
        print(f"Total examples annotated: {sum(results['examples_annotated'])}")
        print(f"Total features annotated: {sum(results['features_annotated'])}")
    
    if experiment_results:
        combined_file_name = "combined_enhanced_results"
        if args.use_embedding:
            combined_file_name += "_with_embedding"
        
        with open(os.path.join(results_path, f"{combined_file_name}.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"\nEnhanced experiment results saved to {results_path}")


if __name__ == "__main__":
    main()