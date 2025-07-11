"""
Enhanced active learning experiments with configuration management and logging.

Author: Prabhav Singh / Haojun Shi
"""

import os
import argparse
import torch
import json
import random
import numpy as np
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances

os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})

from config import Config, ModelConfig, DefaultHyperparams
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset, get_experiment_config
from annotationArena import AnnotationArena
# from imputer import ImputerEmbedding
from imputerExpanded import ImputerEmbedding
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy,
    EntropyExampleSelectionStrategy,
    EntropyFeatureSelectionStrategy,
    BADGESelectionStrategy,
    ArgmaxVOISelectionStrategy,
    VariableGradientSelectionStrategy,
    NewVariableGradientSelectionStrategy
)
from eval import ModelEvaluator, TrainingMetricsTracker, evaluate_training_progress

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

random.seed(90)
torch.manual_seed(90)
np.random.seed(90)

os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})

# Change Based on Usage.
model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer('C:\\Users\\stone\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\\snapshots\\c9745ed1d9f207416be6d2e6f8de32d1f16199bf')

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
        text_embeddings = torch.tensor(entry['text_embedding'], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions, text_embeddings)
            
            mask = inputs[:, :, 0]
            
            for layer in model.encoder.layers:
                feature_x, param_x = layer(feature_x, param_x, questions, mask)
                
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
    dataset_calibration=None,
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
    initial_train_dataset=None,
    training_type='basic',
    num_patterns_per_example=5,
    visible_ratio=0.5,
    config=None,
    use_wandb=False,
    experiment_config=None
):
    """Enhanced experiment runner with dynamic K-centers and improved validation resampling."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    
    # Track calibration metrics across cycles
    cycle_calibration_metrics = {
        'smECE_overall': [],
        'smECE_class_0': [],
        'smECE_class_1': [],
        'smECE_class_2': [],
        'smECE_class_3': [],
        'smECE_class_4': []
    }
    
    if initial_train_dataset is not None and len(initial_train_dataset) > 0:
        arena = AnnotationArena(model, device)
        logger.info(f"Initial training on {len(initial_train_dataset)} clean examples...")
        arena.set_dataset(initial_train_dataset)

        if training_type == 'dynamic_masking':
            arena.set_dynamic_masking_params(num_patterns_per_example, visible_ratio)
        
        for idx in range(len(initial_train_dataset)):
            arena.register_example(idx, add_all_positions=False)
            known_positions = initial_train_dataset.get_known_positions(idx)
            for pos in known_positions:
                arena.observe_position(idx, pos)
                variable_id = f"example_{idx}_position_{pos}"
                arena.predict(variable_id, train=True)
        
        arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, training_type=training_type)
        logger.info("Initial training completed!")
    else:
        arena = AnnotationArena(model, device)
        arena.set_dataset(dataset_train)

    if training_type == 'dynamic_masking':
        arena.set_dynamic_masking_params(num_patterns_per_example, visible_ratio)
    
    metrics = {
        'val_losses': [],
        'val_metrics': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    validation_example_indices = list(range(len(dataset_val)))
    if dataset_calibration:
        calibration_pool = list(range(len(dataset_calibration)))
    test_overlap_annotations = {}
    cycle_count = 0

    logger.info(f"Starting experiment: {example_strategy} strategy, {cycles} cycles, {examples_per_cycle} examples/cycle")

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

    max_cycles = float('inf') if run_until_exhausted else cycles

    while cycle_count < max_cycles:
        
        if not active_pool:
            logger.info(f"Active pool exhausted after {cycle_count} cycles")
            break
    
        logger.info(f"\n\n\n============================ Cycle {cycle_count + 1}/{cycles} ==============================")
        logger.info(f"Active pool size: {len(active_pool)}")

        arena.model.set_current_cycle(cycle_count)
        
        valid_active_pool = []
        for idx in active_pool:
            masked_positions = dataset_train.get_masked_positions(idx)
            if masked_positions:
                valid_active_pool.append(idx)
        
        if len(valid_active_pool) != len(active_pool):
            logger.info(f"Filtered active pool: {len(valid_active_pool)} (removed examples with no masked positions)")
            active_pool = valid_active_pool
        
        if not active_pool:
            logger.info("No examples with masked positions remaining")
            break
        
        if resample_validation and cycle_count > 0:
            logger.info("Resampling validation set...")
            current_val_indices = list(range(len(dataset_val)))
            active_pool.extend(current_val_indices)
            
            dataset_val, active_pool, validation_example_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="balanced_fixed_size", 
                selected_examples=annotated_examples[-examples_per_cycle:] if annotated_examples else [],
                validation_set_size=validation_set_size
            )
        
        logger.info(f"Applying dynamic K-centers to select {min(active_set_size, len(active_pool))} from {len(active_pool)} examples...")
        if len(active_pool) > active_set_size:
            model_embeddings = extract_model_embeddings(dataset_train, active_pool, model, device)
            selected_subset_indices = greedy_k_centers(model_embeddings, active_set_size, random_seed=42 + cycle_count)
            active_subset = [active_pool[i] for i in selected_subset_indices]
            logger.info(f"K-centers selected {len(active_subset)} diverse examples from pool")
        else:
            active_subset = active_pool.copy()
            logger.info(f"Pool smaller than target, using all {len(active_subset)} examples")
        
        # Count available positions in active subset for frequency calculation
        available_positions = {f"Pos-{i}": 0 for i in range(14)}
        for example_idx in active_subset:
            masked_positions = dataset_train.get_masked_positions(example_idx)
            for pos in masked_positions:
                available_positions[f"Pos-{pos}"] += 1
        
        if example_strategy == "random":
            selected_examples = random.sample(active_subset, min(examples_per_cycle, len(active_subset)))
            if dataset_calibration:
                selected_calibration_examples = random.sample(active_subset, min(examples_per_cycle, len(calibration_pool)))
            
        elif example_strategy == "gradient":
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])
            example_selector = SelectionFactory.create_example_strategy(
                example_strategy, model, device, gradient_top_only=gradient_top_only
            )
            
            selected_indices, scores = example_selector.select_examples(
                active_subset_dataset, 
                num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val,
                num_samples=3,
                batch_size=batch_size
            )
            
            selected_examples = [active_subset[idx] for idx in selected_indices]

            calibration_subset_dataset = AnnotationDataset([dataset_calibration.get_data_entry(idx) for idx in calibration_pool])
            selected_calibration_indices, scores = example_selector.select_examples(
                calibration_subset_dataset, 
                num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val,
                num_samples=3,
                batch_size=batch_size
            )
            
            selected_calibration_examples = [calibration_pool[idx] for idx in selected_calibration_indices]

        elif example_strategy == "entropy":
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])
            example_selector = SelectionFactory.create_example_strategy(
                example_strategy, model, device
            )
            
            selected_indices, scores = example_selector.select_examples(
                active_subset_dataset, 
                num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val,
                num_samples=3,
                batch_size=batch_size
            )
            
            selected_examples = [active_subset[idx] for idx in selected_indices]

        elif example_strategy == "combine":
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])
            variable_selector = NewVariableGradientSelectionStrategy(model, device)

            total_features_needed = examples_per_cycle * features_per_example
            num_variables_to_request = min(total_features_needed * 3, len(active_subset) * 10)

            selected_variables, scores = variable_selector.select_examples(
                active_subset_dataset,
                num_to_select=num_variables_to_request,
                val_dataset=dataset_val,
                num_samples=3,
                batch_size=batch_size
            )

            logger.info(f"Variable selector returned {len(selected_variables)} candidate variables")

            selected_examples_dict_fixed = {}
            total_features_selected = 0

            for (local_idx, pos), score in zip(selected_variables, scores):
                if total_features_selected >= total_features_needed:
                    break
                    
                global_idx = active_subset[local_idx]
                
                if global_idx not in selected_examples_dict_fixed:
                    selected_examples_dict_fixed[global_idx] = []
                
                selected_examples_dict_fixed[global_idx].append(pos)
                total_features_selected += 1

            selected_examples = list(selected_examples_dict_fixed.keys())

            final_feature_count = sum(len(positions) for example, positions in selected_examples_dict_fixed.items() 
                                        if example in selected_examples)

            logger.info(f"Variable gradient selection summary:")
            logger.info(f"  Target features: {total_features_needed}")
            logger.info(f"  Selected features: {final_feature_count}")
            logger.info(f"  Selected examples: {len(selected_examples)}")
            logger.info(f"  Avg features per example: {final_feature_count / len(selected_examples) if selected_examples else 0:.1f}")

            selected_examples_dict = {ex: selected_examples_dict_fixed[ex] for ex in selected_examples}

        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        logger.info(f"Selected {len(selected_examples)} examples for annotation")
        
        # Track question selections for this cycle
        question_counts = {f"Pos-{i}": 0 for i in range(14)}
        question_frequencies = {f"Pos-{i}": 0.0 for i in range(14)}
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
                        question_counts[f"Pos-{pos}"] += 1
                        
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

                logger.debug(f"SELECTED FEATURE POSITIONS ARE {selected_features}")
                
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
                        question_counts[f"Pos-{pos}"] += 1
                        
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
                        
                        test_entry = dataset_train.get_data_entry(example_idx)
                        test_question = test_entry['questions'][pos]
                        question_counts[f"Pos-{pos}"] += 1
                        
                        if example_idx < len(dataset_test):
                            if example_idx not in test_overlap_annotations:
                                test_overlap_annotations[example_idx] = []
                            test_overlap_annotations[example_idx].append(pos)
                        
                        variable_id = f"example_{example_idx}_position_{pos}"
                        arena.predict(variable_id, train=True)

        for example_idx in selected_calibration_examples:
            masked_positions = dataset_calibration.get_masked_positions(example_idx)
            for pos in masked_positions:
                arena.observe_position(example_idx, pos)
            calibration_pool.remove(example_idx)
        
        # Calculate frequencies (proportion of available positions selected)
        for pos_key in question_frequencies.keys():
            if available_positions[pos_key] > 0:
                question_frequencies[pos_key] = question_counts[pos_key] / available_positions[pos_key]
            else:
                question_frequencies[pos_key] = 0.0
        
        logger.info(f"Total features annotated this cycle: {total_features_annotated}")

        # NEW: Collect historical patterns for professor's approach
        for example_idx in selected_examples:

            current_data = dataset_train[example_idx]
            current_state = (current_data[1][:, 0] == 0).float() 
            
            # Get query pattern for this cycle (which positions were annotated this cycle)
            query_pattern = torch.zeros(14)
            for (ex_idx, pos) in selected_variables_info:
                if ex_idx == example_idx:
                    query_pattern[pos] = 1.0
            
            # Collect the historical pattern
            arena.model.collect_historical_pattern(current_state, query_pattern, cycle_count)
        
        annotated_examples.extend(selected_examples)
        
        if not cold_start:
            for example_idx in selected_examples:
                if example_idx in active_pool:
                    active_pool.remove(example_idx)

        if training_type == 'dynamic_masking':
            arena.set_dynamic_masking_params(num_patterns_per_example, visible_ratio)
        
        logger.info(f"Training model for {epochs_per_cycle} epochs...")
        arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, training_type=training_type)

        # NEW: Export pattern logs every 5 cycles
        if cycle_count % 3 == 0 and config:
            try:
                experiment_paths = config.get_experiment_paths("current_experiment")
                arena.model.export_pattern_logs(experiment_paths['results_dir'])
            except:
                logger.warning("Could not export pattern logs - continuing without export")
        
        # Evaluation using eval.py
        if config and use_wandb:
            evaluator = ModelEvaluator(config, use_wandb)
            datasets = {'train': dataset_train, 'validation': dataset_val, 'test': dataset_test}
            cycle_eval = evaluator.evaluate_active_learning_cycle(model, datasets, cycle_count, experiment_config=experiment_config)
            
            val_metrics = cycle_eval['evaluations']['validation']['overall']
            test_metrics = cycle_eval['evaluations']['test']['overall']
            
            # Extract calibration metrics from step 7 of test evaluation
            if 'test_trend' in cycle_eval:
                test_trend = cycle_eval['test_trend']
                metrics_trends = test_trend.get('metrics_trends', {})
                
                # Use step 7 (index 7) for cycle calibration tracking
                step_index = min(7, len(metrics_trends.get('smECE_overall', [])) - 1)
                if step_index >= 0:
                    for metric_name in cycle_calibration_metrics.keys():
                        if metric_name in metrics_trends and len(metrics_trends[metric_name]) > step_index:
                            cycle_calibration_metrics[metric_name].append(metrics_trends[metric_name][step_index])
                        else:
                            cycle_calibration_metrics[metric_name].append(0.0)
                else:
                    # If no step 7, use zeros
                    for metric_name in cycle_calibration_metrics.keys():
                        cycle_calibration_metrics[metric_name].append(0.0)
            else:
                # No test trend available, use zeros
                for metric_name in cycle_calibration_metrics.keys():
                    cycle_calibration_metrics[metric_name].append(0.0)
                    
        else:
            arena.set_dataset(dataset_val)
            val_metrics = arena.evaluate(list(range(len(dataset_val))))
            arena.set_dataset(dataset_test)
            test_metrics = arena.evaluate(list(range(len(dataset_test))))
            
            # No calibration tracking for non-wandb runs
            for metric_name in cycle_calibration_metrics.keys():
                cycle_calibration_metrics[metric_name].append(0.0)

        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        logger.info(f"Validation - RMSE: {val_metrics['rmse']:.4f}, "
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
        
        logger.info(f"Test loss: {test_metrics['avg_expected_loss']:.4f}")
        
        # Log question selection counts and frequencies to WandB
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb_data = {
                'cycle': cycle_count,
                'total_features_selected': total_features_annotated,
                'examples_selected': len(selected_examples),
                'pool_size_remaining': len(active_pool)
            }
            
            # Log both counts and frequencies using organized metrics
            for question, count in question_counts.items():
                pos_num = question.split('-')[1]  # Extract position number
                wandb_data[f"position_selection/{question}_count"] = count
                wandb_data[f"position_selection/{question}_frequency"] = question_frequencies[question]
                wandb_data[f"position_selection/{question}_available"] = available_positions[question]
                
                # Also log selection proportion (same as frequency but clearer naming)
                wandb_data[f"position_selection/Pos-{pos_num}_proportion"] = question_frequencies[question]
            
            wandb.log(wandb_data)
        
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        cycle_count += 1
        
    logger.info(f"Experiment complete - {cycle_count} cycles")
    
    # Create explicit plots and log to WandB
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None and cycle_count > 0:
        import matplotlib.pyplot as plt
        
        # Create calibration trend plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Overall smECE trend
        cycles_x = list(range(cycle_count))
        if len(cycle_calibration_metrics['smECE_overall']) >= cycle_count:
            ax1.plot(cycles_x, cycle_calibration_metrics['smECE_overall'][:cycle_count], 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Cycle')
            ax1.set_ylabel('smECE Overall')
            ax1.set_title('Model Calibration (smECE) Across Active Learning Cycles')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)
        
        # Plot 2: Per-class smECE trends
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, color in enumerate(colors):
            metric_name = f'smECE_class_{i}'
            if len(cycle_calibration_metrics[metric_name]) >= cycle_count:
                ax2.plot(cycles_x, cycle_calibration_metrics[metric_name][:cycle_count], 
                        color=color, marker='o', linewidth=2, markersize=4, label=f'Class {i}')
        
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('smECE by Class')
        ax2.set_title('Per-Class Calibration Trends')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save calibration plot to WandB
        wandb.log({"Model_Calibration_Trends_Across_Cycles": wandb.Image(fig)})
        plt.close(fig)

        
        
        # Log calibration trends across cycles using WandB's metric system
        for cycle_idx in range(cycle_count):
            calibration_data = {"cycle": cycle_idx}
            
            # Log overall and per-class calibration metrics for this cycle
            for metric_name, values in cycle_calibration_metrics.items():
                if len(values) > cycle_idx:
                    calibration_data[f"calibration_trends/{metric_name}"] = values[cycle_idx]
            
            wandb.log(calibration_data)
        
        # Log final calibration summary statistics
        final_calibration_data = {'experiment_complete': True}
        for metric_name, values in cycle_calibration_metrics.items():
            if len(values) >= cycle_count and cycle_count > 0:
                final_calibration_data[f"calibration_summary/final_{metric_name}"] = values[cycle_count-1]
                final_calibration_data[f"calibration_summary/mean_{metric_name}"] = np.mean(values[:cycle_count])
                final_calibration_data[f"calibration_summary/std_{metric_name}"] = np.std(values[:cycle_count])
        
        wandb.log(final_calibration_data)
        
        logger.info("Final calibration trends:")
        for metric_name, values in cycle_calibration_metrics.items():
            if len(values) >= cycle_count and cycle_count > 0:
                logger.info(f"  {metric_name}: Final={values[cycle_count-1]:.4f}, Mean={np.mean(values[:cycle_count]):.4f}")
    
    metrics['test_metrics'] = test_metrics
    metrics['calibration_trends'] = cycle_calibration_metrics
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced Active Learning Experiments with AnnotationArena.")
    
    parser.add_argument("--experiment", type=str, default="gradient_voi", 
                       help="Experiment to run")
    parser.add_argument("--cycles", type=int, default=DefaultHyperparams.CYCLES, 
                       help="Number of active learning cycles")
    parser.add_argument("--examples_per_cycle", type=int, default=DefaultHyperparams.EXAMPLES_PER_CYCLE, 
                       help="Number of examples to select per cycle")
    parser.add_argument("--features_per_example", type=int, default=DefaultHyperparams.FEATURES_PER_EXAMPLE, 
                       help="Number of features to select per example")
    parser.add_argument("--epochs_per_cycle", type=int, default=DefaultHyperparams.EPOCHS_PER_CYCLE, 
                       help="Number of training epochs per cycle")
    parser.add_argument("--batch_size", type=int, default=DefaultHyperparams.BATCH_SIZE, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=DefaultHyperparams.LR, 
                       help="Learning rate")
    parser.add_argument("--loss_type", type=str, default=DefaultHyperparams.LOSS_TYPE, 
                       help="Loss type for VOI calculation")
    parser.add_argument("--resample_validation", action="store_true", default=DefaultHyperparams.RESAMPLE_VALIDATION,
                       help="Whether to resample validation set during training")
    parser.add_argument("--run_until_exhausted", action="store_true", default=DefaultHyperparams.RUN_UNTIL_EXHAUSTED,
                       help="Whether to run until the active pool is exhausted")
    parser.add_argument("--dataset", type=str, default="hanna", 
                       help="Dataset to use")
    parser.add_argument("--runner", type=str, default="local", 
                       help="Runner identifier")
    parser.add_argument("--cold_start", type=bool, default=DefaultHyperparams.COLD_START, 
                       help="Use cold start approach")
    parser.add_argument("--use_embedding", type=bool, default=DefaultHyperparams.USE_EMBEDDING, 
                       help="Use embeddings for texts")
    parser.add_argument("--active_set_size", type=int, default=DefaultHyperparams.ACTIVE_SET_SIZE, 
                       help="Size of active subset selected by K-centers each cycle")
    parser.add_argument("--validation_set_size", type=int, default=DefaultHyperparams.VALIDATION_SET_SIZE, 
                       help="Fixed size for validation set")
    parser.add_argument("--train_option", choices=['basic', 'random_masking', 'dynamic_masking'], 
                       default=DefaultHyperparams.TRAIN_OPTION,
                       help="Type of Training to Use - basic / random_masking / dynamic masking")
    parser.add_argument("--gradient_top_only", type=bool, default=DefaultHyperparams.GRADIENT_TOP_ONLY, 
                       help="Faster Approximation with Top Only")
    parser.add_argument('--num_patterns_per_example', type=int, default=DefaultHyperparams.NUM_PATTERNS_PER_EXAMPLE, 
                   help='Number of masking patterns per example for dynamic masking')
    parser.add_argument('--visible_ratio', type=float, default=DefaultHyperparams.VISIBLE_RATIO,
                   help='Ratio of observed positions to keep visible in dynamic masking')
    parser.add_argument('--output_path', type=str,
                   help='Folder to Save In')

    parser.add_argument("--calibration_holdout_ratio", type=float, default=0.1,
                      help="Ratio of data to hold out for conformal calibration (default: 0.1)")
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='active-learner',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str,
                       help='Wandb entity name')
    parser.add_argument('--experiment_name', type=str,
                       help='Experiment name for logging and file naming')
    parser.add_argument('--training_buffer_size', type=int, default=0,
                       help='Buffer size for maximum number of examples seen in the training')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config(args.runner)
    config.ensure_directories()
    
    # Set experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"{args.experiment}_{args.dataset}"
    
    # Setup logging
    exp_paths = config.get_experiment_paths(args.experiment_name)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_paths['log_file']),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model using ModelConfig
    model_config = ModelConfig.get_config(args.dataset, args.training_buffer_size)
    if args.use_embedding:
        ModelClass = ImputerEmbedding
    else:
        logger.error(f"Imputer Without Embeddings is not supported in new codebase. Use v1 for that.")
    
    model = ModelClass(**model_config).to(device)
    logger.info(f"Model initialized: {ModelClass.__name__} with config {model_config}")
    
    experiment_results = {}
    
    # Define experiments to run
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
            "variable_gradient_comparison"
        ]
    else:
        experiments_to_run = [args.experiment]

    for experiment in experiments_to_run:

        logger.info(f"============ Running experiment: {experiment} ============")
        
        # Get experiment-specific configuration
        experiment_config = get_experiment_config(experiment)
        
        # Create experiment-specific paths
        experiment_exp_name = f"{args.experiment_name}_{experiment}"
        experiment_paths = config.get_experiment_paths(experiment_exp_name)
        
        # Initialize Wandb for this specific experiment
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_config = vars(args).copy()
            wandb_config.update({
                'config_timestamp': config.timestamp,
                'base_path': config.BASE_PATH,
                'experiment_type': experiment,
                'feature_selection_strategy': experiment_config['feature_selection_strategy'],
                'target_questions': experiment_config['target_questions']
            })
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{experiment_exp_name}_{config.timestamp}",
                config=wandb_config
            )
            logger.info(f"Wandb initialized for experiment: {experiment}")
        elif args.use_wandb:
            logger.warning("Wandb requested but not available")
        
        model_copy = copy.deepcopy(model)
        
        # Initialize data manager with config
        data_manager = DataManager(config)

        if args.dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=args.dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding, calibration_holdout_ratio=args.calibration_holdout_ratio)
        elif args.dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=1000, initial_train_ratio=0.0, dataset=args.dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding, calibration_holdout_ratio=args.calibration_holdout_ratio)

        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        calibration_dataset = AnnotationDataset(data_manager.paths["calibration"])
        
        initial_train_dataset = None
        if len(train_dataset) > 0:
            initial_train_dataset = train_dataset

        logger.info(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
              f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        # Run experiments based on type
        common_kwargs = {
            'cycles': args.cycles,
            'examples_per_cycle': args.examples_per_cycle,
            'epochs_per_cycle': args.epochs_per_cycle,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': device,
            'resample_validation': args.resample_validation,
            'run_until_exhausted': args.run_until_exhausted,
            'cold_start': args.cold_start,
            'active_set_size': args.active_set_size,
            'validation_set_size': args.validation_set_size,
            'initial_train_dataset': initial_train_dataset,
            'gradient_top_only': args.gradient_top_only,
            'training_type': args.train_option,
            'num_patterns_per_example': args.num_patterns_per_example,
            'visible_ratio': args.visible_ratio,
            'config': config,
            'use_wandb': args.use_wandb,
            'experiment_config': experiment_config
        }

        if experiment == "random_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset, 
                calibration_dataset=calibration_dataset,
                example_strategy="random", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "gradient_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="gradient", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "entropy_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="entropy", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "random_5":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                **common_kwargs
            )

        elif experiment == "gradient_voi":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                **common_kwargs
            )

        elif experiment == "gradient_voi_q0_human":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type, target_questions=[0],
                **common_kwargs
            )

        elif experiment == "gradient_voi_all_questions":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type, target_questions=[0, 1, 2, 3, 4, 5, 6],
                **common_kwargs
            )

        elif experiment == "variable_gradient_comparison":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                calibration_dataset=calibration_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                **common_kwargs
            )

        else:
            logger.warning(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results

        calibration_path = os.path.join(config.INPUT_DATA_DIR, "calibration_holdout.json")
        if os.path.exists(calibration_path):
            logger.info(f"Calibration holdout saved at: {calibration_path}")
            print(f"Calibration holdout available at: {calibration_path}")
        else:
            logger.warning("No calibration holdout found - may need to regenerate data")
        
        # Save model with experiment-specific path
        torch.save(model_copy.state_dict(), experiment_paths['model_file'])
        
        # Save results with experiment-specific path
        results_file = os.path.join(experiment_paths['results_dir'], f"{experiment}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Experiment {experiment} completed")
        logger.info(f"Final validation loss: {results['val_losses'][-1]:.4f}")
        logger.info(f"Final test loss: {results['test_expected_losses'][-1]:.4f}")
        logger.info(f"Total examples annotated: {sum(results['examples_annotated'])}")
        logger.info(f"Total features annotated: {sum(results['features_annotated'])}")
        
        # Finish WandB run for this experiment
        if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    
    # Save combined results
    if experiment_results:
        combined_file = os.path.join(exp_paths['results_dir'], "combined_results.json")
        with open(combined_file, "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        logger.info(f"All results saved to {exp_paths['results_dir']}")
        
if __name__ == "__main__":
    main()