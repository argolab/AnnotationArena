"""
Enhanced active learning experiments with configuration management and logging.
Ablation study version for testing variable gradient methods with different training approaches.

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
from imputerExpandedAblation import ImputerEmbedding
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

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_embeddings_features(dataset_entries, model_name='all-MiniLM-L6-v2'):
    """Extract sentence transformer embeddings for K-centers algorithm."""
    embeddings = []
    for entry in dataset_entries:
        text = entry['text'] if 'text' in entry else entry['content']
        embedding = model.encode(text)
        embeddings.append(embedding)
    return np.array(embeddings)

def extract_model_embeddings(dataset, indices, model, device):
    """Extract embeddings using model encoder."""
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            known_questions, inputs, answers, annotators, questions, text_embeddings = dataset[idx]
            
            text_embeddings = text_embeddings.unsqueeze(0).to(device)
            inputs = inputs.unsqueeze(0).to(device)
            annotators = annotators.unsqueeze(0).to(device) 
            questions = questions.unsqueeze(0).to(device)
            
            feature_x, param_x = model.encoder(inputs, annotators, questions, text_embeddings)
            
            pooled_features = feature_x.mean(dim=1).squeeze(0)
            embeddings.append(pooled_features.cpu().numpy())
    
    return np.array(embeddings)

def greedy_k_centers(embeddings, k, random_seed=42):
    """Greedy k-centers algorithm for diverse subset selection."""
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

    if training_type == 'dynamic_masking' or training_type == 'dynamic_masking_simple':
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
        
        available_positions = {f"Pos-{i}": 0 for i in range(14)}
        for example_idx in active_subset:
            masked_positions = dataset_train.get_masked_positions(example_idx)
            for pos in masked_positions:
                available_positions[f"Pos-{pos}"] += 1
        
        if example_strategy == "random":
            selected_examples = random.sample(active_subset, min(examples_per_cycle, len(active_subset)))
            
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
            logger.info(f"  - Selected {len(selected_examples)} examples")
            logger.info(f"  - Selected {final_feature_count} features total")
            logger.info(f"  - Average features per example: {final_feature_count / len(selected_examples):.2f}")

        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")

        logger.info(f"Selected {len(selected_examples)} examples from active subset of {len(active_subset)}")

        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)

        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        selected_variables_info = []
        total_features_annotated = 0
        question_counts = {f"Pos-{i}": 0 for i in range(14)}

        for example_idx in selected_examples:
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

            elif feature_strategy is not None and feature_strategy != "combine":
                feature_selector = SelectionFactory.create_feature_strategy(feature_strategy, model, device)
                
                if example_strategy == "combine" and example_idx in selected_examples_dict_fixed:
                    selected_positions = selected_examples_dict_fixed[example_idx][:features_per_example]
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

        logger.info(f"Feature annotation summary:")
        logger.info(f"  Total features annotated this cycle: {total_features_annotated}")
        for pos_name, count in question_counts.items():
            if count > 0:
                logger.info(f"  {pos_name}: {count} annotations")

        if len(selected_examples) > 0:
            logger.info(f"Training on {len(arena.model.training_queue)} examples...")
            training_metrics = arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, training_type=training_type)
            logger.info(f"Training completed - Average loss: {training_metrics['avg_loss']:.4f}")

        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])

        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])

        test_subset_indices = list(test_overlap_annotations.keys()) if test_overlap_annotations else []
        if test_subset_indices:
            arena.set_dataset(dataset_test)
            test_subset_metrics = arena.evaluate(test_subset_indices)
            metrics['test_annotated_losses'].append(test_subset_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])

        arena.set_dataset(dataset_train)

        for example_idx in selected_examples:
            if example_idx not in annotated_examples:
                annotated_examples.append(example_idx)
            active_pool.remove(example_idx)

        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        cycle_count += 1
        
    logger.info(f"Experiment complete - {cycle_count} cycles")
    
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None and cycle_count > 0:
        for cycle_idx in range(cycle_count):
            calibration_data = {"cycle": cycle_idx}
            
            for metric_name, values in cycle_calibration_metrics.items():
                if len(values) > cycle_idx:
                    calibration_data[f"calibration_trends/{metric_name}"] = values[cycle_idx]
            
            wandb.log(calibration_data)
        
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
    
    parser.add_argument("--experiment", type=str, default="random_5_base_training", 
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
    
    parser.add_argument('--historical_weight', type=float, default=0.3,
                       help='Weight for historical loss in dynamic masking generation')
    parser.add_argument('--influence_weight', type=float, default=0.7,
                       help='Weight for influence loss in dynamic masking generation')
    
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='active-learner-ablation',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str,
                       help='Wandb entity name')
    parser.add_argument('--experiment_name', type=str,
                       help='Experiment name for logging and file naming')
    parser.add_argument('--training_buffer_size', type=int, default=0,
                       help='Buffer size for maximum number of examples seen in the training')
    
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    config = Config(args.runner)
    config.ensure_directories()
    
    if not args.experiment_name:
        args.experiment_name = f"{args.experiment}_{args.dataset}"
    
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
    
    model_config = ModelConfig.get_config(args.dataset, args.training_buffer_size)
    model_config['historical_weight'] = args.historical_weight
    model_config['influence_weight'] = args.influence_weight
    
    if args.use_embedding:
        ModelClass = ImputerEmbedding
    else:
        logger.error(f"Imputer Without Embeddings is not supported in new codebase. Use v1 for that.")
    
    model = ModelClass(**model_config).to(device)
    logger.info(f"Model initialized: {ModelClass.__name__} with config {model_config}")
    
    experiment_results = {}
    
    experiments_to_run = []
    if args.experiment == "ablation_all":
        experiments_to_run = [
            "random_5_base_training",
            "var_grad_base_training", 
            "var_grad_random_masking",
            "var_grad_dynamic_masking",
            "var_grad_dynamic_masking_hist_only",
            "var_grad_dynamic_masking_inf_only",
            "var_grad_dynamic_masking_70_30",
            "var_grad_dynamic_masking_30_70"
        ]
    elif args.experiment == "ablation_comparison":
        experiments_to_run = [
            "random_5_base_training",
            "var_grad_base_training",
            "var_grad_dynamic_masking",
            "var_grad_dynamic_masking_hist_only",
            "var_grad_dynamic_masking_inf_only"
        ]
    else:
        experiments_to_run = [args.experiment]

    for experiment in experiments_to_run:

        logger.info(f"============ Running experiment: {experiment} ============")
        
        experiment_config = get_experiment_config(experiment)
        
        experiment_exp_name = f"{args.experiment_name}_{experiment}"
        experiment_paths = config.get_experiment_paths(experiment_exp_name)
        
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
        
        initial_train_dataset = None
        if len(train_dataset) > 0:
            initial_train_dataset = train_dataset

        logger.info(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
              f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

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
            'num_patterns_per_example': args.num_patterns_per_example,
            'visible_ratio': args.visible_ratio,
            'config': config,
            'use_wandb': args.use_wandb,
            'experiment_config': experiment_config
        }

        if experiment == "random_5_base_training":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                training_type='basic_ablation',
                **common_kwargs
            )

        elif experiment == "var_grad_base_training":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='basic_ablation',
                **common_kwargs
            )

        elif experiment == "var_grad_random_masking":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='random_masking_ablation',
                **common_kwargs
            )

        elif experiment == "var_grad_dynamic_masking":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='dynamic_masking_simple',
                **common_kwargs
            )

        elif experiment == "var_grad_dynamic_masking_hist_only":
            model_copy.historical_weight = 1.0
            model_copy.influence_weight = 0.0
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='dynamic_masking',
                **common_kwargs
            )

        elif experiment == "var_grad_dynamic_masking_inf_only":
            model_copy.historical_weight = 0.0
            model_copy.influence_weight = 1.0
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='dynamic_masking',
                **common_kwargs
            )

        elif experiment == "var_grad_dynamic_masking_70_30":
            model_copy.historical_weight = 0.3
            model_copy.influence_weight = 0.7
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='dynamic_masking',
                **common_kwargs
            )

        elif experiment == "var_grad_dynamic_masking_30_70":
            model_copy.historical_weight = 0.7
            model_copy.influence_weight = 0.3
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                training_type='dynamic_masking',
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
        
        torch.save(model_copy.state_dict(), experiment_paths['model_file'])
        
        results_file = os.path.join(experiment_paths['results_dir'], f"{experiment}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Experiment {experiment} completed")
        logger.info(f"Final validation loss: {results['val_losses'][-1]:.4f}")
        logger.info(f"Final test loss: {results['test_expected_losses'][-1]:.4f}")
        logger.info(f"Total examples annotated: {sum(results['examples_annotated'])}")
        logger.info(f"Total features annotated: {sum(results['features_annotated'])}")
        
        if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    
    if experiment_results:
        combined_file = os.path.join(exp_paths['results_dir'], "combined_results.json")
        with open(combined_file, "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        logger.info(f"All results saved to {exp_paths['results_dir']}")
        
if __name__ == "__main__":
    main()