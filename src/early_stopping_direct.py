#!/usr/bin/env python3
"""
Integrated script for Human Q0 evaluation workflow with prediction confidence interval-based stopping.
For each position selection, dynamically computes calibration data from prediction errors on the same observed positions.
"""

import os
import sys
import argparse
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from scipy import stats
from annotationArena import AnnotationArena
from selection import SelectionFactory
from utils import AnnotationDataset
from eval import ModelEvaluator
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_cache_to_disk(cache_file_path: str):
    """Save the current calibration cache to disk."""
    global _calibration_cache
    
    try:
        # Convert frozenset keys to tuples for JSON serialization compatibility
        serializable_cache = {}
        for key, value in _calibration_cache.items():
            if isinstance(key, tuple) and len(key) == 2:
                observed_positions, target_question = key
                # Convert frozenset to sorted tuple
                if isinstance(observed_positions, frozenset):
                    serializable_key = (tuple(sorted(observed_positions)), target_question)
                else:
                    serializable_key = key
                serializable_cache[serializable_key] = value
            else:
                serializable_cache[key] = value
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(serializable_cache, f)
        
        logger.info(f"Cache saved to {cache_file_path} ({len(_calibration_cache)} entries)")
        
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_file_path}: {e}")

def load_cache_from_disk(cache_file_path: str) -> bool:
    """
    Load calibration cache from disk.
    
    Args:
        cache_file_path: Path to the cache file
        
    Returns:
        True if cache was loaded successfully, False otherwise
    """
    global _calibration_cache
    
    if not os.path.exists(cache_file_path):
        logger.info(f"Cache file {cache_file_path} not found, starting with empty cache")
        return False
    
    try:
        with open(cache_file_path, 'rb') as f:
            loaded_cache = pickle.load(f)
        
        # Convert back to frozenset keys
        _calibration_cache = {}
        for key, value in loaded_cache.items():
            if isinstance(key, tuple) and len(key) == 2:
                observed_positions_tuple, target_question = key
                # Convert tuple back to frozenset
                frozenset_key = (frozenset(observed_positions_tuple), target_question)
                _calibration_cache[frozenset_key] = value
            else:
                _calibration_cache[key] = value
        
        logger.info(f"Cache loaded from {cache_file_path} ({len(_calibration_cache)} entries)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_file_path}: {e}")
        _calibration_cache = {}
        return False

def clear_cache():
    """Clear the calibration cache."""
    global _calibration_cache
    _calibration_cache = {}
    logger.info("Cache cleared")

# Global cache for dynamic calibration computations
_calibration_cache = {}

def compute_prediction_confidence_interval(observed_positions: Set[int], 
                                         model, calibration_dataset, 
                                         target_question: int, device: str,
                                         confidence_level: float = 0.95) -> Tuple[float, float, int]:
    """
    Compute prediction confidence interval using dynamically computed position-specific calibration data.
    
    Args:
        observed_positions: Set of positions already observed
        model: The model
        calibration_dataset: The calibration dataset
        target_question: Target question index
        device: Device for computation
        confidence_level: Confidence level for the interval (default 0.95)
    
    Returns:
        Tuple of (interval_width, mean_error, sample_size)
    """
    global _calibration_cache
    
    # Create cache key (no longer need next_position since we're looking at current prediction)
    cache_key = (frozenset(observed_positions), target_question)
    
    # Check if we already computed calibration data for this configuration
    if cache_key in _calibration_cache:
        prediction_errors = _calibration_cache[cache_key]
    else:
        # Dynamically compute calibration data
        prediction_errors = []
        
        logger.debug(f"Computing calibration for observed={observed_positions}, target_q={target_question}")
        
        # Process each example in calibration dataset
        for cal_example_idx in tqdm(range(len(calibration_dataset)), desc="Computing calibration"):
            # Create fresh copy for this calibration example
            cal_dataset_copy = copy.deepcopy(calibration_dataset)
            cal_arena = AnnotationArena(model, device)
            cal_arena.set_dataset(cal_dataset_copy)
            
            # Step 1: Observe all the positions in observed_positions for this calibration example
            valid_example = True
            for pos in observed_positions:
                # Check if this position exists and is valid for this example
                cal_data_entry = cal_dataset_copy.get_data_entry(cal_example_idx)
                if pos >= len(cal_data_entry['questions']):
                    valid_example = False
                    break
                
                # Observe the position
                success = cal_arena.observe_position(cal_example_idx, pos)
                if not success:
                    valid_example = False
                    break
            
            if not valid_example:
                continue
                
            # Step 2: Get current prediction for target question
            try:
                cal_data_entry = cal_dataset_copy.get_data_entry(cal_example_idx)
                known_questions, inputs, answers, annotators, questions, embeddings = cal_dataset_copy[cal_example_idx]
                
                inputs = inputs.unsqueeze(0).to(device)
                annotators_tensor = annotators.unsqueeze(0).to(device)
                questions_tensor = questions.unsqueeze(0).to(device)
                
                if embeddings is not None:
                    embeddings = embeddings.unsqueeze(0).to(device)
                else:
                    seq_len = inputs.shape[1]
                    embeddings = torch.zeros(1, seq_len, 384).to(device)
                
                with torch.no_grad():
                    outputs = model(inputs, annotators_tensor, questions_tensor, embeddings)
                    
                    # Get prediction for target question
                    pred_probs = F.softmax(outputs[0, target_question], dim=0)
                    pred_class = 1 * pred_probs[0] + 2 * pred_probs[1] + 3 * pred_probs[2] + 4 * pred_probs[3] + 5 * pred_probs[4]
                    pred_score = pred_class.cpu().item()
                    
                    # Get true label for target question
                    if 'true_answers' in cal_data_entry and cal_data_entry['true_answers']:
                        true_class = torch.argmax(torch.tensor(cal_data_entry['true_answers'][target_question])).item()
                    else:
                        true_class = torch.argmax(torch.tensor(cal_data_entry['answers'][target_question])).item()
                    true_score = true_class + 1
                    
                    # Compute prediction error
                    prediction_error = pred_score - true_score
                    prediction_errors.append(prediction_error)
                    
            except Exception as e:
                logger.debug(f"Error processing calibration example {cal_example_idx}: {e}")
                continue
        
        # Cache the result
        _calibration_cache[cache_key] = prediction_errors
        logger.debug(f"Cached {len(prediction_errors)} prediction errors for key {cache_key}")
        save_cache_to_disk("cache1.pik")
    
    if len(prediction_errors) == 0:
        return float('inf'), 0.0, 0
    
    # Compute confidence interval using empirical percentiles
    errors = np.array(prediction_errors)
    mean_error = np.mean(errors)
    
    # Calculate confidence interval using empirical percentiles
    alpha = 1 - confidence_level
    n = len(errors)
    
    if n < 2:
        return float('inf'), mean_error, n
    
    # Compute empirical percentiles
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(errors, lower_percentile)
    upper_bound = np.percentile(errors, upper_percentile)
    
    interval_width = upper_bound - lower_bound
    
    return interval_width, mean_error, n

def evaluate_human_q0_workflow_with_confidence_intervals(model, dataset, calibration_dataset,
                                                       dataset_name: str = "unknown", 
                                                       split_type: str = "test",
                                                       target_question: int = 7,
                                                       max_interval_width: float = 0.5,
                                                       min_selections_before_stop: int = 3,
                                                       min_calibration_samples: int = 5,
                                                       confidence_level: float = 0.95,
                                                       device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate model with confidence interval-based stopping decisions using dynamic calibration.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        calibration_dataset: The calibration dataset for computing confidence intervals
        dataset_name: Name of the dataset
        split_type: Type of split (test, val, etc.)
        target_question: Target question index
        max_interval_width: Maximum allowed confidence interval width for stopping
        min_selections_before_stop: Minimum number of selections before considering stopping
        min_calibration_samples: Minimum calibration samples required for reliable interval
        confidence_level: Confidence level for intervals (default 0.95)
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation results and workflow metrics
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Human Q0 Evaluation with Confidence Interval-Based Stopping")
    logger.info(f"Dataset: {dataset_name} ({split_type})")
    logger.info(f"Calibration dataset size: {len(calibration_dataset)}")
    logger.info(f"Target question: {target_question}")
    logger.info(f"Max interval width: {max_interval_width}")
    logger.info(f"Min selections before stop: {min_selections_before_stop}")
    logger.info(f"Min calibration samples: {min_calibration_samples}")
    logger.info(f"Confidence level: {confidence_level}")
    logger.info(f"{'='*60}")
    
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Clear cache at start
    global _calibration_cache
    _calibration_cache = {}
    
    # Initialize results tracking
    workflow_results = {
        'dataset_name': dataset_name,
        'split_type': split_type,
        'target_question': target_question,
        'max_interval_width': max_interval_width,
        'min_selections_before_stop': min_selections_before_stop,
        'min_calibration_samples': min_calibration_samples,
        'confidence_level': confidence_level,
        'individual_phase': {
            'per_example_results': {},
            'total_features_selected': 0,
            'early_stop_count': 0,
            'interval_width_stops': 0,
            'insufficient_calibration_stops': 0,
            'final_rmse': None
        }
    }
    
    # Get initial RMSE
    initial_rmse, initial_smece = evaluate_model_q0_only(model, dataset, target_question, device)
    logger.info(f"Initial Q{target_question} RMSE: {initial_rmse:.4f}")
    if os.path.exists("cache1.pik"):
        load_cache_from_disk("cache1.pik")
    
    # =================================================================
    # PHASE 1: INDIVIDUAL EXAMPLE STOPPING DECISIONS
    # =================================================================
    logger.info(f"\n=== Phase 1: Individual Example Processing ===")
    
    for example_idx in tqdm(range(len(dataset)), desc="Processing examples"):
        if example_idx == 100:
            break
        # Create fresh copy for this example
        dataset_copy = copy.deepcopy(dataset)
        arena = AnnotationArena(model, device)
        arena.set_dataset(dataset_copy)
        feature_selector = SelectionFactory.create_feature_strategy('voi', model, device)
        
        # Initialize per-example tracking
        example_results = {
            'voi_scores': [],
            'interval_widths': [],
            'mean_errors': [],
            'calibration_sample_sizes': [],
            'selected_positions': [],
            'observed_positions_history': [],
            'features_selected': 0,
            'stopped_early': False,
            'stop_reason': None
        }
        
        # Track observed positions
        observed_positions = set()
        features_selected = 0
        
        # Individual example selection loop
        while features_selected < 14:
            # Get current VOI ranking
            current_voi_ranking = feature_selector.select_features(
                example_idx, dataset_copy,
                num_to_select=14-features_selected,
                loss_type="cross_entropy",
                target_questions=[0]
            )
            
            if not current_voi_ranking:
                example_results['stopped_early'] = True
                example_results['stop_reason'] = "no_features_available"
                break
            
            # Filter out human Q0 and already observed positions
            filtered_ranking = []
            data_entry = dataset_copy.get_data_entry(example_idx)
            
            for feature_info in current_voi_ranking:
                pos = feature_info[0]
                question_idx = data_entry['questions'][pos]
                annotator_idx = data_entry['annotators'][pos]
                
                # Skip human Q0 unless it's the only remaining feature
                if question_idx == target_question and annotator_idx != -1 and features_selected < 13:
                    continue
                
                filtered_ranking.append(feature_info)
            
            if not filtered_ranking:
                example_results['stopped_early'] = True
                example_results['stop_reason'] = "no_valid_features"
                break
            
            # Get the top feature and its VOI score
            top_feature = filtered_ranking[0]
            next_position, voi_score = top_feature[0], top_feature[1]
            
            # Compute confidence interval using dynamic calibration
            interval_width, mean_error, calibration_size = compute_prediction_confidence_interval(
                observed_positions, model, calibration_dataset, target_question, device, confidence_level
            )
            
            # Store results
            example_results['voi_scores'].append(voi_score)
            example_results['interval_widths'].append(interval_width)
            example_results['mean_errors'].append(mean_error)
            example_results['calibration_sample_sizes'].append(calibration_size)
            example_results['observed_positions_history'].append(list(observed_positions))
            
            logger.info(f"Example {example_idx}, Step {features_selected}: "
                        f"VOI={voi_score:.4f}, interval_width={interval_width:.4f}, "
                        f"cal_size={calibration_size}")
            
            # Check stopping conditions after minimum selections
            if features_selected >= min_selections_before_stop:
                # Check if we have enough calibration samples
                if calibration_size < min_calibration_samples:
                    logger.debug(f"  ⚠️  Insufficient calibration samples ({calibration_size} < {min_calibration_samples})")
                    workflow_results['individual_phase']['insufficient_calibration_stops'] += 1
                
                # Check confidence interval width
                elif interval_width <= max_interval_width:
                    # Prediction is sufficiently confident, stop
                    example_results['stopped_early'] = True
                    example_results['stop_reason'] = "confident_prediction"
                    workflow_results['individual_phase']['interval_width_stops'] += 1
                    logger.debug(f"  🛑 Stopping: confident prediction (width={interval_width:.4f} <= {max_interval_width})")
                    break
            
            # Select and observe the feature
            success = arena.observe_position(example_idx, next_position)
            observed_positions.add(next_position)
            features_selected += 1
            example_results['selected_positions'].append(next_position)
        
        # Record results for this example
        example_results['features_selected'] = features_selected
        workflow_results['individual_phase']['per_example_results'][example_idx] = example_results
        
        if example_results['stopped_early']:
            workflow_results['individual_phase']['early_stop_count'] += 1
    
    # Calculate summary statistics
    workflow_results['individual_phase']['total_features_selected'] = sum(
        ex['features_selected'] for ex in workflow_results['individual_phase']['per_example_results'].values()
    )
    
    # Calculate final RMSE after individual phase  
    dataset_copy = copy.deepcopy(dataset)
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_copy)
    
    # Apply all the selections made during the individual phase
    for example_idx, example_results in workflow_results['individual_phase']['per_example_results'].items():
        for pos in example_results['selected_positions']:
            arena.observe_position(example_idx, pos)
    
    final_rmse, final_smece = evaluate_model_q0_only(model, dataset_copy, target_question, device)
    workflow_results['individual_phase']['final_rmse'] = final_rmse
    
    # Log results
    total_examples = len(workflow_results['individual_phase']['per_example_results'])
    avg_features = workflow_results['individual_phase']['total_features_selected'] / total_examples
    
    logger.info(f"\n=== Results Summary ===")
    logger.info(f"Average features per example: {avg_features:.2f}")
    logger.info(f"Early stops: {workflow_results['individual_phase']['early_stop_count']}/{total_examples}")
    logger.info(f"Interval width stops: {workflow_results['individual_phase']['interval_width_stops']}")
    logger.info(f"Insufficient calibration stops: {workflow_results['individual_phase']['insufficient_calibration_stops']}")
    logger.info(f"Final RMSE: {final_rmse:.4f}")
    logger.info(f"Final smECE: {final_smece:.4f}")
    logger.info(f"Cache entries created: {len(_calibration_cache)}")
    plot_features_histogram(workflow_results)

    # Additional evaluation for comparison
    experiment_config = {"feature_selection_strategy": "voi", "target_questions": [0]}
    
    # Extract configuration from experiment_config
    if experiment_config:
        feature_selection_type = experiment_config.get('feature_selection_strategy', 'voi')
    else:
        feature_selection_type = 'voi'
        eval_target_questions = list(range(7))

    eval_target_questions = experiment_config.get("target_questions", list(range(1, 7)))
    
    logger.info(f"\n-- Evaluating model on {dataset_name} {split_type} set ({len(dataset)} examples) with {feature_selection_type} feature selection --")
    
    all_results = []
    model.eval()
    
    # Create deep copy of dataset to avoid state persistence between cycles
    dataset_copy = copy.deepcopy(dataset)
    
    # Initialize arena and feature selector
    arena = AnnotationArena(model, "cuda")
    arena.set_dataset(dataset_copy)
    feature_selector = SelectionFactory.create_feature_strategy(feature_selection_type, model, "cuda")
    
    # Initialize metrics tracking
    metrics_trends = {
        'rmse': [],
        'pearson': [],
        'spearman': [],
        'kendall': [],
        'accuracy': [],
        'mae': [],
        'avg_expected_loss': [],
        'smECE_overall': [],
        'smECE_class_0': [],
        'smECE_class_1': [],
        'smECE_class_2': [],
        'smECE_class_3': [],
        'smECE_class_4': []
    }
    
    # Count total features across all examples
    total_features = 14 * len(dataset_copy)
    
    logger.info(f"Starting evaluation with {total_features} total features to collect")
    
    # Initial evaluation with no features observed (all positions unknown) - with calibration
    initial_eval = evaluate_model_q0_only(model, dataset_copy, target_question, device)
    
    logger.info(f"Initial evaluation (0 features): RMSE={initial_eval[0]:.4f}, "
                f"smECE={initial_eval[1]:.4f}")
    
    # Iteratively select and observe features
    features_collected = 0
    while features_collected < total_features:
        # For each example, select one feature if available
        features_selected_this_round = 0
        
        for example_idx in tqdm(range(len(dataset_copy))):
            # Select features for this example (limit to 1 per round)
            selected_features = feature_selector.select_features(
                example_idx, dataset_copy, 
                num_to_select=2,
                loss_type="cross_entropy",
                target_questions=eval_target_questions
            )
            
            # Observe selected features
            for feature_info in selected_features:
                pos = feature_info[0]  # Position index
                if len(selected_features) > 1:
                    if pos == 0:
                        continue
                success_criteria = arena.observe_position(example_idx, pos)
                features_selected_this_round += 1
                features_collected += 1
                
                logger.debug(f"Observed feature at example {example_idx}, position {pos} (total collected: {features_collected}). Success - {success_criteria}")
                
                # Break after selecting one feature per example per round
                break
        
        # If no features were selected this round, break
        if features_selected_this_round == 0:
            logger.info("No more features available for selection")
            break
        
        # Evaluate model with newly observed features - including calibration
        current_eval = evaluate_model_q0_only(model, dataset_copy, target_question, device)
        all_results.append(current_eval)
        
        logger.info(f"After {features_collected} features: RMSE={current_eval[0]:.4f}, "
                f"smECE={current_eval[1]:.4f}, "
                f"Features selected this round: {features_selected_this_round}")
        
        # Early termination if all features have been collected
        if features_collected >= total_features:
            break
    
    logger.info(f"Evaluation completed: {len(metrics_trends['rmse'])} evaluation steps from 0 to {features_collected} features")
    
    return workflow_results

def plot_features_histogram(workflow_results, save_path=None):
    """Plot histogram of number of features collected per example."""
    
    features_per_example = [
        results['features_selected'] 
        for results in workflow_results['individual_phase']['per_example_results'].values()
    ]
    
    plt.figure(figsize=(12, 8))
    
    # Plot features histogram only
    n, bins, patches = plt.hist(features_per_example, 
                               bins=range(0, 16),
                               alpha=0.7, 
                               color='skyblue', 
                               edgecolor='black',
                               linewidth=0.5)
    
    mean_features = np.mean(features_per_example)
    median_features = np.median(features_per_example)
    
    plt.axvline(mean_features, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_features:.2f}')
    plt.axvline(median_features, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_features:.2f}')
    
    plt.xlabel('Number of Features Selected per Example')
    plt.ylabel('Frequency')
    plt.title('Distribution of Features Selected (Confidence Interval-Based Stopping)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_example_loss(model, dataset, example_idx: int, target_question: int, device) -> float:
    """Compute the loss for a specific example and target question."""
    model.eval()
    
    known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
    
    with torch.no_grad():
        outputs = model(
            inputs.unsqueeze(0).to(device), 
            annotators.unsqueeze(0).to(device), 
            questions.unsqueeze(0).to(device), 
            embeddings.unsqueeze(0).to(device)
        )
        
        prediction = torch.softmax(outputs[:, target_question, :], dim=-1)
        target_answer = answers[target_question].to(device)
        
        loss = torch.nn.functional.cross_entropy(
            prediction, 
            target_answer.unsqueeze(0), 
            reduction='none'
        )
        
        return loss.item()

def evaluate_model_q0_only(model, dataset, target_question: int, device) -> tuple:
    """Evaluate model on target question only and return RMSE and calibration."""
    model.eval()
    
    predictions = []
    true_values = []
    all_pred_probs = []
    all_true_labels = []
    
    with torch.no_grad():
        for example_idx in range(len(dataset)):
            if example_idx == 100:
                break
            try:
                data_entry = dataset.get_data_entry(example_idx)
                known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
                
                inputs = inputs.unsqueeze(0).to(device)
                annotators_tensor = annotators.unsqueeze(0).to(device)
                questions_tensor = questions.unsqueeze(0).to(device)
                
                if embeddings is not None:
                    embeddings = embeddings.unsqueeze(0).to(device)
                else:
                    seq_len = inputs.shape[1]
                    embeddings = torch.zeros(1, seq_len, 384).to(device)

                outputs = model(inputs, annotators_tensor, questions_tensor, embeddings)
                
                for pos in range(len(data_entry['questions'])):
                    if pos != target_question:
                        continue

                    annotator = data_entry["annotators"][pos]
                    
                    if annotator == -1:
                        continue
                    
                    pred_probs = F.softmax(outputs[0, pos], dim=0)
                    pred_class = 1 * pred_probs[0] + 2 * pred_probs[1] + 3 * pred_probs[2] + 4 * pred_probs[3] + 5 * pred_probs[4]
                    pred_score = pred_class.cpu()
                    
                    if 'true_answers' in data_entry and data_entry['true_answers']:
                        true_class = torch.argmax(torch.tensor(data_entry['true_answers'][pos])).item()
                    else:
                        true_class = torch.argmax(torch.tensor(data_entry['answers'][pos])).item()
                    true_score = true_class + 1
                    
                    predictions.append(pred_score)
                    true_values.append(true_score)
                    
                    all_pred_probs.append(pred_probs.cpu().numpy())
                    all_true_labels.append(true_class)
                    
            except Exception as e:
                logger.warning(f"Error processing example {example_idx}: {e}")
                continue
    
    if not predictions:
        logger.error("No valid predictions found")
        return float('inf'), 0.0
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    
    calibration_score = 0.0
    if len(all_pred_probs) > 0:
        calibration_metrics = compute_calibration_metrics(all_pred_probs, all_true_labels)
        calibration_score = calibration_metrics.get('smECE_overall', 0.0)
    
    return rmse, calibration_score

def compute_calibration_metrics(all_pred_probs, all_true_labels):
    """Compute calibration metrics (smECE) for each class and overall."""
    try:
        import relplot as rp
    except ImportError:
        logger.warning("relplot not available, returning zero calibration metrics")
        return {'smECE_overall': 0.0}
    
    calibration_metrics = {}
    all_pred_probs = np.array(all_pred_probs)
    all_true_labels = np.array(all_true_labels)
    
    class_smECE = []
    for class_idx in range(all_pred_probs.shape[1]):
        y_true = (all_true_labels == class_idx).astype(int)
        y_pred = all_pred_probs[:, class_idx]
        
        if len(np.unique(y_true)) > 1:
            try:
                smECE = rp.smECE(y_pred, y_true)
                class_smECE.append(smECE)
                calibration_metrics[f'smECE_class_{class_idx}'] = smECE
            except:
                calibration_metrics[f'smECE_class_{class_idx}'] = 0.0
        else:
            calibration_metrics[f'smECE_class_{class_idx}'] = 0.0
    
    if class_smECE:
        calibration_metrics['smECE_overall'] = np.mean(class_smECE)
    else:
        calibration_metrics['smECE_overall'] = 0.0
    
    return calibration_metrics

def main():
    from imputer import ImputerEmbedding
    
    parser = argparse.ArgumentParser(description='Human Q0 Evaluation with Confidence Interval-Based Stopping')
    parser.add_argument('--max_interval_width', type=float, default=0.5, help='Maximum allowed confidence interval width for stopping')
    parser.add_argument('--min_selections', type=int, default=0, help='Minimum selections before stop check')
    parser.add_argument('--min_calibration_samples', type=int, default=5, help='Minimum calibration samples for reliable interval')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='Confidence level for intervals')
    parser.add_argument('--target_question', type=int, default=7, help='Target question index')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets and model
    calibration_dataset = AnnotationDataset("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\input\\data\\calibration_holdout.json")
    test_dataset = AnnotationDataset("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\input\\data\\test.json")
    
    model = ImputerEmbedding(7, 5, 6, 4, 64, 18, 19, 0.1)
    model.load_state_dict(torch.load("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\output\\models\\HANNA_NEW_DM_variable_gradient_comparison_20250706_090905.pth"))
    model.to(args.device)
    
    # Evaluate with confidence interval-based stopping
    logger.info("Evaluating with confidence interval-based stopping...")
    results = evaluate_human_q0_workflow_with_confidence_intervals(
        model=model,
        dataset=test_dataset,
        calibration_dataset=calibration_dataset,
        dataset_name="test",
        target_question=args.target_question,
        max_interval_width=args.max_interval_width,
        min_selections_before_stop=args.min_selections,
        min_calibration_samples=args.min_calibration_samples,
        confidence_level=args.confidence_level,
        device=args.device
    )
    
    # Generate plots and save results
    plot_path = os.path.join(args.output_dir, 'confidence_interval_workflow_plots.png')
    plot_features_histogram(results, plot_path)
    
    results_path = os.path.join(args.output_dir, 'confidence_interval_workflow_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Workflow completed!")

if __name__ == "__main__":
    main()