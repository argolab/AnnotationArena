#!/usr/bin/env python3
"""
Integrated script for Human Q0 evaluation workflow with position-specific statistical significance testing.
For each position selection, dynamically computes calibration data from the same observed positions.
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
            if isinstance(key, tuple) and len(key) == 3:
                observed_positions, next_position, target_question = key
                # Convert frozenset to sorted tuple
                if isinstance(observed_positions, frozenset):
                    serializable_key = (tuple(sorted(observed_positions)), next_position, target_question)
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
            if isinstance(key, tuple) and len(key) == 3:
                observed_positions_tuple, next_position, target_question = key
                # Convert tuple back to frozenset
                frozenset_key = (frozenset(observed_positions_tuple), next_position, target_question)
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

def compute_voi_p_value(predicted_voi: float, observed_positions: Set[int], 
                       next_position: int, model, calibration_dataset, 
                       target_question: int, device: str) -> Tuple[float, int]:
    """
    Compute p-value for VOI prediction using dynamically computed position-specific calibration data.
    
    Args:
        predicted_voi: The VOI prediction to test
        observed_positions: Set of positions already observed
        next_position: The position being considered
        model: The model
        calibration_dataset: The calibration dataset
        target_question: Target question index
        device: Device for computation
    
    Returns:
        Tuple of (p_value, sample_size)
    """
    global _calibration_cache
    
    # Create cache key
    cache_key = (frozenset(observed_positions), next_position, target_question)
    
    # Check if we already computed calibration data for this configuration
    if cache_key in _calibration_cache:
        calibration_errors = _calibration_cache[cache_key]
    else:
        # Dynamically compute calibration data
        calibration_errors = []
        
        logger.debug(f"Computing calibration for observed={observed_positions}, next={next_position}")
        
        # Process each example in calibration dataset
        for cal_example_idx in tqdm(range(len(calibration_dataset))):
            # Create fresh copy for this calibration example
            cal_dataset_copy = copy.deepcopy(calibration_dataset)
            cal_arena = AnnotationArena(model, device)
            cal_arena.set_dataset(cal_dataset_copy)
            cal_feature_selector = SelectionFactory.create_feature_strategy('voi', model, device)
            
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
                print("here")
                continue
                
            # Step 2: Check if next_position is available and valid
            cal_data_entry = cal_dataset_copy.get_data_entry(cal_example_idx)
            if next_position >= len(cal_data_entry['questions']):
                continue
                
            # Check if next_position would be selected by VOI
            current_voi_ranking = cal_feature_selector.select_features(
                cal_example_idx, cal_dataset_copy,
                num_to_select=14 - len(observed_positions),
                loss_type="cross_entropy",
                target_questions=[0]
            )
            
            if not current_voi_ranking:
                print("VOI error")
                continue
                
            # Find next_position in the ranking and get its VOI
            position_voi = None
            for feature_info in current_voi_ranking:
                if feature_info[0] == next_position:
                    position_voi = feature_info[1]
                    break
            
            if position_voi is None:
                continue  # next_position not in VOI ranking
                
            # Step 3: Compute actual loss reduction for next_position
            # Get loss before observing next_position
            loss_before = compute_example_loss(model, cal_dataset_copy, cal_example_idx, target_question, device)
            
            # Observe next_position
            success = cal_arena.observe_position(cal_example_idx, next_position)
            if not success:
                continue
                
            # Get loss after observing next_position
            loss_after = compute_example_loss(model, cal_dataset_copy, cal_example_idx, target_question, device)
            actual_loss_reduction = loss_before - loss_after
            
            # Step 4: Compute non-conformity score
            non_conformity_score = position_voi - actual_loss_reduction
            calibration_errors.append(non_conformity_score)
        
        # Cache the result
        _calibration_cache[cache_key] = calibration_errors
        logger.debug(f"Cached {len(calibration_errors)} calibration errors for key {cache_key}")

        save_cache_to_disk("cache.pik")
    
    if len(calibration_errors) == 0:
        return 1.0, 0
    
    # Perform statistical test
    # One-sample t-test: H0: predicted_voi = 0 vs H1: predicted_voi != 0
    # We test if the predicted VOI is significantly different from the distribution of errors
    
    errors = np.array(calibration_errors)
    
    # Test if predicted_voi is significantly different from the mean of calibration errors
    # H0: predicted_voi is drawn from the same distribution as calibration errors
    # H1: predicted_voi is significantly different

    if predicted_voi > 0:
        more_extreme = np.sum(errors >= predicted_voi) / 120
        p_value = more_extreme
    else:
        p_value = np.sum(errors <= predicted_voi) / 120
    
    return p_value, len(calibration_errors)

def evaluate_human_q0_workflow_with_statistical_stopping(model, dataset, calibration_dataset,
                                                        dataset_name: str = "unknown", 
                                                        split_type: str = "test",
                                                        target_question: int = 7,
                                                        significance_level: float = 0.05,
                                                        min_selections_before_stop: int = 3,
                                                        min_calibration_samples: int = 5,
                                                        device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate model with statistical significance-based stopping decisions using dynamic calibration.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        calibration_dataset: The calibration dataset for computing p-values
        dataset_name: Name of the dataset
        split_type: Type of split (test, val, etc.)
        target_question: Target question index
        significance_level: P-value threshold for stopping (default 0.05)
        min_selections_before_stop: Minimum number of selections before considering stopping
        min_calibration_samples: Minimum calibration samples required for reliable p-value
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation results and workflow metrics
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Human Q0 Evaluation with Dynamic Statistical Stopping")
    logger.info(f"Dataset: {dataset_name} ({split_type})")
    logger.info(f"Calibration dataset size: {len(calibration_dataset)}")
    logger.info(f"Target question: {target_question}")
    logger.info(f"Significance level: {significance_level}")
    logger.info(f"Min selections before stop: {min_selections_before_stop}")
    logger.info(f"Min calibration samples: {min_calibration_samples}")
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
        'significance_level': significance_level,
        'min_selections_before_stop': min_selections_before_stop,
        'min_calibration_samples': min_calibration_samples,
        'individual_phase': {
            'per_example_results': {},
            'total_features_selected': 0,
            'early_stop_count': 0,
            'statistical_stops': 0,
            'insufficient_calibration_stops': 0,
            'final_rmse': None
        }
    }
    
    # Get initial RMSE
    initial_rmse, initial_smece = evaluate_model_q0_only(model, dataset, target_question, device)
    logger.info(f"Initial Q{target_question} RMSE: {initial_rmse:.4f}")
    if os.path.exists("cache.pik"):
        load_cache_from_disk("cache.pik")
    # =================================================================
    # PHASE 1: INDIVIDUAL EXAMPLE STOPPING DECISIONS
    # =================================================================
    logger.info(f"\n=== Phase 1: Individual Example Processing ===")
    
    for example_idx in tqdm(range(len(dataset)), desc="Processing examples"):
        # Create fresh copy for this example
        dataset_copy = copy.deepcopy(dataset)
        arena = AnnotationArena(model, device)
        arena.set_dataset(dataset_copy)
        feature_selector = SelectionFactory.create_feature_strategy('voi', model, device)
        
        # Initialize per-example tracking
        example_results = {
            'voi_scores': [],
            'p_values': [],
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
            
            # Compute p-value using dynamic calibration
            p_value, calibration_size = compute_voi_p_value(
                voi_score, observed_positions, next_position, 
                model, calibration_dataset, target_question, device
            )
            
            # Store results
            example_results['voi_scores'].append(voi_score)
            example_results['p_values'].append(p_value)
            example_results['calibration_sample_sizes'].append(calibration_size)
            example_results['observed_positions_history'].append(list(observed_positions))
            
            logger.info(f"Example {example_idx}, Step {features_selected}: "
                        f"VOI={voi_score:.4f}, p={p_value:.4f}, "
                        f"cal_size={calibration_size}")
            
            # Check stopping conditions after minimum selections
            if features_selected >= min_selections_before_stop:
                
                # Check statistical significance
                if (p_value > significance_level and voi_score > 0) or (p_value < significance_level and voi_score < 0):
                    # VOI is not statistically significant, stop
                    example_results['stopped_early'] = True
                    example_results['stop_reason'] = "not_statistically_significant"
                    workflow_results['individual_phase']['statistical_stops'] += 1
                    logger.debug(f"  🛑 Stopping: not significant (p={p_value:.4f} > {significance_level})")
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
    logger.info(f"Statistical stops: {workflow_results['individual_phase']['statistical_stops']}")
    logger.info(f"Insufficient calibration stops: {workflow_results['individual_phase']['insufficient_calibration_stops']}")
    logger.info(f"Final RMSE: {final_rmse:.4f}")
    logger.info(f"Final smECE: {final_smece:.4f}")
    logger.info(f"Cache entries created: {len(_calibration_cache)}")
    
    return workflow_results

def plot_features_histogram(workflow_results, save_path=None):
    """Plot histogram of number of features collected per example."""
    
    features_per_example = [
        results['features_selected'] 
        for results in workflow_results['individual_phase']['per_example_results'].values()
    ]
    
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Features histogram
    n, bins, patches = ax1.hist(features_per_example, 
                               bins=range(0, 16),
                               alpha=0.7, 
                               color='skyblue', 
                               edgecolor='black',
                               linewidth=0.5)
    
    mean_features = np.mean(features_per_example)
    median_features = np.median(features_per_example)
    
    ax1.axvline(mean_features, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_features:.2f}')
    ax1.axvline(median_features, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_features:.2f}')
    
    ax1.set_xlabel('Number of Features Selected per Example')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Features Selected (Dynamic Statistical Stopping)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Stop reasons
    stop_reasons = {}
    for results in workflow_results['individual_phase']['per_example_results'].values():
        reason = results.get('stop_reason', 'completed_all_features')
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    
    if stop_reasons:
        reasons, counts = zip(*stop_reasons.items())
        ax2.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Stop Reasons Distribution')
    
    # Plot 3: P-values distribution (if available)
    all_p_values = []
    for results in workflow_results['individual_phase']['per_example_results'].values():
        all_p_values.extend(results.get('p_values', []))
    
    if all_p_values:
        ax3.hist(all_p_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(workflow_results.get('significance_level', 0.05), 
                   color='red', linestyle='--', label='Significance Level')
        ax3.set_xlabel('P-values')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of P-values')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Calibration sample sizes
    all_cal_sizes = []
    for results in workflow_results['individual_phase']['per_example_results'].values():
        all_cal_sizes.extend(results.get('calibration_sample_sizes', []))
    
    if all_cal_sizes:
        ax4.hist(all_cal_sizes, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.axvline(workflow_results.get('min_calibration_samples', 5), 
                   color='red', linestyle='--', label='Min Required')
        ax4.set_xlabel('Calibration Sample Size')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Calibration Sample Sizes')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
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
    
    parser = argparse.ArgumentParser(description='Human Q0 Evaluation with Dynamic Statistical Stopping')
    parser.add_argument('--significance_level', type=float, default=0.5, help='P-value threshold for stopping')
    parser.add_argument('--min_selections', type=int, default=0, help='Minimum selections before stop check')
    parser.add_argument('--min_calibration_samples', type=int, default=5, help='Minimum calibration samples for reliable p-value')
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
    
    # Evaluate with dynamic statistical stopping
    logger.info("Evaluating with dynamic statistical stopping...")
    results = evaluate_human_q0_workflow_with_statistical_stopping(
        model=model,
        dataset=test_dataset,
        calibration_dataset=calibration_dataset,
        dataset_name="test",
        target_question=args.target_question,
        significance_level=args.significance_level,
        min_selections_before_stop=args.min_selections,
        min_calibration_samples=args.min_calibration_samples,
        device=args.device
    )
    
    # Generate plots and save results
    plot_path = os.path.join(args.output_dir, 'dynamic_statistical_workflow_plots.png')
    plot_features_histogram(results, plot_path)
    
    results_path = os.path.join(args.output_dir, 'dynamic_statistical_workflow_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Workflow completed!")

if __name__ == "__main__":
    main()