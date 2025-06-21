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

from config import Config, ModelConfig, DefaultHyperparams
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from annotationArena import AnnotationArena
from imputer import ImputerEmbedding
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
    use_wandb=False
):
    """Enhanced experiment runner with dynamic K-centers and improved validation resampling."""
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {example_strategy}, features: {feature_strategy}")
    
    metrics = {
        'val_losses': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': []
    }
    
    active_pool = list(range(len(dataset_train)))
    selected_examples = []
    
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

    cycle_count = 0
    
    while cycle_count < cycles:
        if run_until_exhausted and len(active_pool) == 0:
            logger.info("Active pool exhausted, stopping experiment")
            break
            
        if len(active_pool) < examples_per_cycle:
            examples_per_cycle = len(active_pool)
            if examples_per_cycle == 0:
                break

        logger.info(f"\n=== CYCLE {cycle_count + 1} ===")
        logger.info(f"Active pool size: {len(active_pool)}")
        
        if cold_start and len(dataset_train) > 0:
            embeddings = extract_embeddings_features([dataset_train.get_data_entry(idx) for idx in active_pool])
        else:
            embeddings = extract_model_embeddings(dataset_train, active_pool, model, device)
        
        k_centers_size = min(active_set_size, len(active_pool))
        active_subset_indices = greedy_k_centers(embeddings, k_centers_size)
        active_subset = [active_pool[i] for i in active_subset_indices]
        
        logger.info(f"Selected active subset of size {len(active_subset)} using K-centers")

        if example_strategy == "random":
            selected_indices = random.sample(range(len(active_subset)), min(examples_per_cycle, len(active_subset)))
            selected_examples_current_cycle = [active_subset[idx] for idx in selected_indices]

        elif example_strategy == "gradient":
            active_subset_dataset = AnnotationDataset([dataset_train.get_data_entry(idx) for idx in active_subset])
            example_selector = SelectionFactory.create_example_strategy("gradient", model, device)
            
            selected_indices, scores = example_selector.select_examples(
                active_subset_dataset, 
                num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val,
                num_samples=3,
                batch_size=batch_size
            )
            
            selected_examples_current_cycle = [active_subset[idx] for idx in selected_indices]

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

            selected_examples_current_cycle = list(selected_examples_dict_fixed.keys())[:examples_per_cycle]
            selected_examples_dict = selected_examples_dict_fixed

        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")

        for example_idx in selected_examples_current_cycle:
            selected_examples.append(example_idx)
            active_pool.remove(example_idx)

        arena.set_dataset(dataset_train)
        
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        selected_variables_info = []
        test_overlap_annotations = {}

        for example_idx in selected_examples_current_cycle:
            arena.register_example(example_idx, add_all_positions=False)
            
            if observe_all_features:
                masked_positions = dataset_train.get_masked_positions(example_idx)
                for pos in masked_positions:
                    if arena.observe_position(example_idx, pos):
                        total_features_annotated += 1
                        cycle_benefit_cost_ratios.append(1.0)
                        cycle_observation_costs.append(1.0)
                        selected_variables_info.append((example_idx, pos))
                        
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

        training_metrics = arena.train(epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, training_type=training_type)
        logger.info(f"Training completed - Loss: {training_metrics['avg_loss']:.4f}")

        if resample_validation:
            logger.info("Resampling validation set...")
            dataset_val, active_pool, validation_example_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, selected_examples, validation_set_size
            )

        # Evaluation using eval.py
        if config and use_wandb:
            evaluator = ModelEvaluator(config, use_wandb)
            datasets = {'train': dataset_train, 'validation': dataset_val, 'test': dataset_test}
            cycle_eval = evaluator.evaluate_active_learning_cycle(model, datasets, cycle_count)
            
            val_metrics = cycle_eval['evaluations']['validation']['overall']
            test_metrics = cycle_eval['evaluations']['test']['overall']
        else:
            arena.set_dataset(dataset_val)
            val_metrics = arena.evaluate(list(range(len(dataset_val))))
            arena.set_dataset(dataset_test)
            test_metrics = arena.evaluate(list(range(len(dataset_test))))

        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        logger.info(f"Validation - RMSE: {val_metrics['rmse']:.4f}, "
                   f"Pearson: {val_metrics['pearson']:.4f}, "
                   f"Expected Loss: {val_metrics['avg_expected_loss']:.4f}")
        
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
        
        metrics['examples_annotated'].append(len(selected_examples_current_cycle))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        metrics['remaining_pool_size'].append(len(active_pool))
        
        cycle_count += 1
        
    logger.info(f"Experiment complete - {cycle_count} cycles")
    
    metrics['test_metrics'] = test_metrics
    arena_metrics = {
        'training_losses': arena.get_training_history(),
        'observation_history': arena.observation_history,
        'prediction_history': arena.prediction_history
    }
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
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='active-learner',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str,
                       help='Wandb entity name')
    parser.add_argument('--experiment_name', type=str,
                       help='Experiment name for logging and file naming')
    
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
    
    # Initialize Wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_config = vars(args).copy()
        wandb_config.update({
            'config_timestamp': config.timestamp,
            'base_path': config.BASE_PATH
        })
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.experiment_name}_{config.timestamp}",
            config=wandb_config
        )
        logger.info("Wandb initialized")
    elif args.use_wandb:
        logger.warning("Wandb requested but not available")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model using ModelConfig
    model_config = ModelConfig.get_config(args.dataset)
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
            "gradient_voi_q0_human",
            "variable_gradient_comparison"
        ]
    else:
        experiments_to_run = [args.experiment]
    
    for experiment in experiments_to_run:
        logger.info(f"Running experiment: {experiment}")
        
        model_copy = copy.deepcopy(model)
        
        # Initialize data manager with config
        data_manager = DataManager(config)

        if args.dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=args.dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding)
        elif args.dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=1000, initial_train_ratio=0.0, dataset=args.dataset, 
                        cold_start=args.cold_start, use_embedding=args.use_embedding)

        train_dataset = AnnotationDataset(data_manager.paths['train'])
        val_dataset = AnnotationDataset(data_manager.paths['validation'])
        test_dataset = AnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
        
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
            'use_wandb': args.use_wandb
        }

        if experiment == "random_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "gradient_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "entropy_all":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", model=model_copy,
                observe_all_features=True,
                **common_kwargs
            )

        elif experiment == "random_5":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                **common_kwargs
            )

        elif experiment == "gradient_voi":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                **common_kwargs
            )

        elif experiment == "gradient_voi_q0_human":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type, target_questions=[0],
                **common_kwargs
            )

        elif experiment == "gradient_voi_all_questions":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type, target_questions=[0, 1, 2, 3, 4, 5, 6],
                **common_kwargs
            )

        elif experiment == "variable_gradient_comparison":
            results = run_enhanced_experiment(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="combine", feature_strategy="gradient", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                loss_type=args.loss_type,
                **common_kwargs
            )

        else:
            logger.warning(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results
        
        # Save model
        torch.save(model_copy.state_dict(), exp_paths['model_file'])
        
        # Save results
        results_file = os.path.join(exp_paths['results_dir'], f"{experiment}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Experiment {experiment} completed")
        logger.info(f"Final validation loss: {results['val_losses'][-1]:.4f}")
        logger.info(f"Final test loss: {results['test_expected_losses'][-1]:.4f}")
        logger.info(f"Total examples annotated: {sum(results['examples_annotated'])}")
        logger.info(f"Total features annotated: {sum(results['features_annotated'])}")
    
    # Save combined results
    if experiment_results:
        combined_file = os.path.join(exp_paths['results_dir'], "combined_results.json")
        with open(combined_file, "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        logger.info(f"All results saved to {exp_paths['results_dir']}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()