#!/usr/bin/env python3
"""
Standalone script for Human Q0 evaluation workflow with VOI-based sequential selection.
Each example makes individual stopping decisions based on VOI ranking stability.
"""

import os
import sys
import argparse
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from annotationArena import AnnotationArena
from selection import SelectionFactory
from utils import AnnotationDataset
from eval import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_features_histogram(workflow_results, save_path=None):
    """
    Plot histogram of number of features collected per example in the individual phase.
    
    Args:
        workflow_results: Dictionary containing workflow results
        save_path: Optional path to save the plot
    """
    # Extract number of features selected per example
    features_per_example = [
        results['features_selected'] 
        for results in workflow_results['individual_phase']['per_example_results'].values()
    ]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(features_per_example, 
                               bins=range(0, 16),  # 0 to 15 features (since max is 14)
                               alpha=0.7, 
                               color='skyblue', 
                               edgecolor='black',
                               linewidth=0.5)
    
    # Add statistics text
    mean_features = np.mean(features_per_example)
    median_features = np.median(features_per_example)
    early_stop_count = workflow_results['individual_phase']['early_stop_count']
    total_examples = len(features_per_example)
    ci_stops = workflow_results['individual_phase']['ci_contains_zero_stops']
    
    # Add vertical lines for mean and median
    plt.axvline(mean_features, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_features:.2f}')
    plt.axvline(median_features, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_features:.2f}')
    
    # Labels and title
    plt.xlabel('Number of Features Selected per Example')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Features Selected (Early Stopping)\n'
              f'Dataset: HANNA')
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Add statistics text box
    
    # Set x-axis ticks
    plt.xticks(range(0, 15))
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig("hist.png")

def evaluate_human_q0_workflow_with_ci_stopping(model, dataset, conformal_delta: float,
                                               dataset_name: str = "unknown", 
                                               split_type: str = "test", 
                                               min_selections_before_stop: int = 3,
                                               device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate model focusing on human Q0 prediction with VOI confidence interval-based stopping decisions.
    Uses precomputed conformal prediction delta to create confidence intervals for VOI predictions
    and stops when the interval contains 0.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        conformal_delta: Precomputed confidence interval delta from calibration (±delta)
        dataset_name: Name of the dataset
        split_type: Type of split (test, val, etc.)
        min_selections_before_stop: Minimum number of selections before considering stopping
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation results and workflow metrics
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Human Q0 Evaluation Workflow with VOI Confidence Intervals")
    logger.info(f"Dataset: {dataset_name} ({split_type})")
    logger.info(f"Conformal delta (±): {conformal_delta:.4f}")
    logger.info(f"Minimum selections before stop check: {min_selections_before_stop}")
    logger.info(f"{'='*60}")
    
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create dataset copy and initialize components
    dataset_copy = copy.deepcopy(dataset)
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_copy)
    feature_selector = SelectionFactory.create_feature_strategy('voi', model, device)
    
    target_question = 0  # Human Q0
    
    # Initialize results tracking
    workflow_results = {
        'dataset_name': dataset_name,
        'split_type': split_type,
        'target_question': target_question,
        'conformal_delta': conformal_delta,
        'min_selections_before_stop': min_selections_before_stop,
        'individual_phase': {
            'per_example_results': {},
            'total_features_selected': 0,
            'early_stop_count': 0,
            'ci_contains_zero_stops': 0,
            'final_rmse': None
        },
        'full_evaluation_phase': {
            'rmse_values': [],
            'feature_counts': [],
            'final_rmse': None,
            'total_features_collected': 0
        }
    }
    
    # Get initial RMSE
    initial_rmse, initial_smece = evaluate_model_q0_only(model, dataset_copy, target_question, device)
    logger.info(f"Initial Q0 RMSE: {initial_rmse:.4f}")
    
    # =================================================================
    # PHASE 1: INDIVIDUAL EXAMPLE STOPPING DECISIONS
    # =================================================================
    logger.info(f"\n=== Phase 1: Individual Example Processing ===")
    
    for example_idx in range(len(dataset_copy)):
        #logger.info(f"\n--- Processing Example {example_idx} ---")
        
        # Initialize per-example tracking
        example_results = {
            'voi_rankings': [],
            'confidence_intervals': [],
            'selected_positions': [],
            'voi_scores': [],
            'features_selected': 0,
            'stopped_early': False,
            'stop_reason': None
        }
        
        features_selected = 0
        
        # Individual example selection loop
        while True:
            if features_selected == 14:
                break
                
            # Get current full VOI ranking for this example
            current_voi_ranking = feature_selector.select_features(
                example_idx, dataset_copy,
                num_to_select=14-features_selected,
                loss_type="cross_entropy",
                target_questions=[target_question]
            )
            #print([(pos, voi) for pos, voi, _, _, in current_voi_ranking])
            
            if not current_voi_ranking:
                example_results['stopped_early'] = True
                example_results['stop_reason'] = "no_features_available"
                break
            
            # Filter out human Q0 (unless it's the only option left)
            filtered_ranking = []
            data_entry = dataset_copy.get_data_entry(example_idx)

            known_questions, inputs, answers, annotators, questions, embeddings = dataset_copy[example_idx]
            outputs = model(inputs.unsqueeze(0).to(device), annotators.unsqueeze(0).to(device), questions.unsqueeze(0).to(device), embeddings.unsqueeze(0).to(device))
            prediction = torch.softmax(outputs[:, 7, :], dim=-1)
            y_hat = 1 * prediction[0][0] + 2 * prediction[0][1] + 3 * prediction[0][2] + 4 * prediction[0][3] + 5 * prediction[0][4]
            label = torch.argmax(answers[7]) + 1
            non_conformity_scores = abs(y_hat - label)
            
            for feature_info in current_voi_ranking:
                pos = feature_info[0]
                question_idx = data_entry['questions'][pos]
                annotator_idx = data_entry['annotators'][pos]
                
                # Skip human Q0 unless it's the only remaining feature
                if question_idx == 0 and annotator_idx != -1 and not features_selected == 13:
                    continue
                
                filtered_ranking.append(feature_info)
            
            if not filtered_ranking:
                example_results['stopped_early'] = True
                example_results['stop_reason'] = "no_valid_features"
                break
            
            # Store current ranking
            example_results['voi_rankings'].append(filtered_ranking)
            
            # Get the top feature and its VOI score
            top_feature = current_voi_ranking[0]
            pos, voi_score = top_feature[0], top_feature[1]
            
            # Calculate confidence interval for this VOI prediction
            ci_lower = voi_score - conformal_delta
            ci_upper = voi_score + conformal_delta
            
            confidence_interval = {
                'voi_score': voi_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'contains_zero': ci_lower <= 0 <= ci_upper
            }
            
            example_results['confidence_intervals'].append(confidence_interval)
            example_results['voi_scores'].append(voi_score)
            
            logger.debug(f"  VOI: {voi_score:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}], Contains 0: {confidence_interval['contains_zero']}")
            
            # Check if confidence interval contains 0 after minimum selections
            if features_selected >= min_selections_before_stop and confidence_interval['contains_zero']:
                example_results['stopped_early'] = True
                example_results['stop_reason'] = "confidence_interval_contains_zero"
                workflow_results['individual_phase']['ci_contains_zero_stops'] += 1
                logger.info(f"  🛑 Stopping early (CI contains 0: [{ci_lower:.4f}, {ci_upper:.4f}], features selected: {features_selected})")
                break
            
            # Select and observe the top feature
            success = arena.observe_position(example_idx, pos)
            features_selected += 1
            example_results['selected_positions'].append(pos)
            
            question_idx = data_entry['questions'][pos]
            annotator_idx = data_entry['annotators'][pos]
        
        # Record results for this example
        example_results['features_selected'] = features_selected
        #print(f"Example {example_idx}: {features_selected}")
        workflow_results['individual_phase']['per_example_results'][example_idx] = example_results
        
        if example_results['stopped_early']:
            workflow_results['individual_phase']['early_stop_count'] += 1
    
    # Continue with the rest of your original workflow...
    workflow_results['individual_phase']['total_features_selected'] = sum(
        ex['features_selected'] for ex in workflow_results['individual_phase']['per_example_results'].values()
    )
    
    # Calculate final RMSE after individual phase
    final_rmse, final_smece = evaluate_model_q0_only(model, dataset_copy, target_question, device)
    workflow_results['individual_phase']['final_rmse'] = final_rmse

    print(workflow_results['individual_phase']['total_features_selected'] / 240)
    print(final_rmse)
    print(final_smece)
    
    plot_features_histogram(workflow_results)

    experiment_config={"feature_selection_strategy": "voi",
            "target_questions": [0]}
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

def calculate_ranking_stability(prev_ranking: List, curr_ranking: List) -> float:
    """
    Calculate ranking stability by comparing relative order of common positions.
    
    Args:
        prev_ranking: Previous ranking as [(pos, voi), ...] 
        curr_ranking: Current ranking as [(pos, voi), ...]
    
    Returns:
        Stability score (0 = perfectly stable, 1 = maximum change)
    """
    if not prev_ranking or not curr_ranking or len(prev_ranking) <= 1:
        return 1.0
    
    # Get positions that appear in both rankings
    prev_positions = [pos for pos, _, _, _ in prev_ranking]
    curr_positions = [pos for pos, _, _, _ in curr_ranking]
    common_positions = set(prev_positions) & set(curr_positions)
    
    if len(common_positions) <= 1:
        return 1.0  # Can't measure stability with ≤1 common positions
    
    # Create relative order mappings for common positions only
    common_prev = [pos for pos in prev_positions if pos in common_positions]
    common_curr = [pos for pos in curr_positions if pos in common_positions]
    
    # Calculate rank differences in relative order
    rank_changes = []
    for pos in common_positions:
        prev_relative_rank = common_prev.index(pos)
        curr_relative_rank = common_curr.index(pos)
        
        # Normalized rank difference
        max_rank = len(common_positions) - 1
        if max_rank > 0:
            rank_change = abs(prev_relative_rank - curr_relative_rank) / max_rank
            rank_changes.append(rank_change)
    
    # Return average relative rank change
    return np.mean(rank_changes) if rank_changes else 0.0

def evaluate_model_q0_only(model, dataset, target_question: int, device) -> tuple:
    """Evaluate model on human Q0 only and return RMSE and calibration."""
    model.eval()
    
    predictions = []
    true_values = []
    # Add calibration data collection
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
                
                # Process positions for target question
                for pos in range(len(data_entry['questions'])):
                    if data_entry['questions'][pos] != target_question:
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
                    
                    # Store calibration data
                    all_pred_probs.append(pred_probs.cpu().numpy())
                    all_true_labels.append(true_class)
                    
            except Exception as e:
                logger.warning(f"Error processing example {example_idx}: {e}")
                continue
    
    if not predictions:
        logger.error("No valid predictions found")
        return float('inf'), 0.0
    
    # Calculate RMSE
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    
    # Calculate calibration metrics
    calibration_score = 0.0
    if len(all_pred_probs) > 0:
        calibration_metrics = compute_calibration_metrics(all_pred_probs, all_true_labels)
        calibration_score = calibration_metrics.get('smECE_overall', 0.0)
    
    return rmse, calibration_score

def compute_calibration_metrics(all_pred_probs, all_true_labels):
    """Compute calibration metrics (smECE) for each class and overall."""
    try:
        import relplot as rp  # Assuming this is available
    except ImportError:
        logger.warning("relplot not available, returning zero calibration metrics")
        return {'smECE_overall': 0.0}
    
    calibration_metrics = {}
    
    # Convert to numpy arrays if they aren't already
    all_pred_probs = np.array(all_pred_probs)
    all_true_labels = np.array(all_true_labels)
    
    # Compute smECE for each class
    class_smECE = []
    for class_idx in range(all_pred_probs.shape[1]):  # Number of classes
        y_true = (all_true_labels == class_idx).astype(int)
        y_pred = all_pred_probs[:, class_idx]
        
        if len(np.unique(y_true)) > 1:  # Only compute if both classes are present
            try:
                smECE = rp.smECE(y_pred, y_true)
                class_smECE.append(smECE)
                calibration_metrics[f'smECE_class_{class_idx}'] = smECE
            except:
                calibration_metrics[f'smECE_class_{class_idx}'] = 0.0
        else:
            calibration_metrics[f'smECE_class_{class_idx}'] = 0.0
    
    # Overall smECE (average across classes)
    if class_smECE:
        calibration_metrics['smECE_overall'] = np.mean(class_smECE)
    else:
        calibration_metrics['smECE_overall'] = 0.0
    
    return calibration_metrics

def generate_analysis(workflow_results: Dict[str, Any], initial_rmse: float) -> Dict[str, Any]:
    """Generate comprehensive analysis comparing individual vs full evaluation."""
    
    individual_rmse = workflow_results['individual_phase']['final_rmse']
    full_rmse = workflow_results['full_evaluation_phase']['final_rmse']
    
    individual_features = workflow_results['individual_phase']['total_features_selected']
    total_features = workflow_results['full_evaluation_phase']['total_features_collected']
    
    # Calculate efficiency metrics
    total_improvement = initial_rmse - full_rmse if full_rmse else 0
    individual_improvement = initial_rmse - individual_rmse if individual_rmse else 0
    
    individual_efficiency = individual_improvement / total_improvement if total_improvement > 0 else 0
    feature_efficiency = individual_features / total_features if total_features > 0 else 0
    rmse_gap = individual_rmse - full_rmse if individual_rmse and full_rmse else 0
    
    # Find optimal stopping point
    rmse_values = workflow_results['full_evaluation_phase']['rmse_values']
    feature_counts = workflow_results['full_evaluation_phase']['feature_counts']
    
    if rmse_values:
        optimal_idx = np.argmin(rmse_values)
        optimal_rmse = rmse_values[optimal_idx]
        optimal_features = feature_counts[optimal_idx]
    else:
        optimal_rmse = full_rmse
        optimal_features = total_features
    
    # Per-example statistics
    per_example_results = workflow_results['individual_phase']['per_example_results']
    if per_example_results:
        features_per_example = [r['features_selected'] for r in per_example_results.values()]
        early_stops = sum(1 for r in per_example_results.values() if r['stopped_early'])
        
        per_example_stats = {
            'avg_features': np.mean(features_per_example),
            'std_features': np.std(features_per_example),
            'early_stop_rate': early_stops / len(per_example_results),
            'min_features': min(features_per_example),
            'max_features': max(features_per_example)
        }
    else:
        per_example_stats = {}
    
    return {
        'individual_efficiency': individual_efficiency,
        'feature_efficiency': feature_efficiency,
        'rmse_gap': rmse_gap,
        'optimal_rmse': optimal_rmse,
        'optimal_features': optimal_features,
        'individual_vs_optimal_rmse': individual_rmse - optimal_rmse if individual_rmse else 0,
        'individual_vs_optimal_features': individual_features - optimal_features,
        'per_example_stats': per_example_stats
    }

def plot_results(workflow_results: Dict[str, Any], save_path: str = None):
    """Plot comparison between individual stopping and full evaluation."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: RMSE progression
    full_features = workflow_results['full_evaluation_phase']['feature_counts']
    full_rmse = workflow_results['full_evaluation_phase']['rmse_values']
    
    ax1.plot(full_features, full_rmse, 'b.-', label='Full Evaluation', linewidth=2)
    
    # Mark individual stopping point
    individual_features = workflow_results['individual_phase']['total_features_selected']
    individual_rmse = workflow_results['individual_phase']['final_rmse']
    
    ax1.axvline(x=individual_features, color='red', linestyle='--', alpha=0.7, label='Individual Stop')
    ax1.scatter([individual_features], [individual_rmse], color='red', s=100, zorder=5)
    
    # Mark optimal point
    if full_rmse:
        optimal_idx = np.argmin(full_rmse)
        optimal_features = full_features[optimal_idx]
        ax1.axvline(x=optimal_features, color='green', linestyle='--', alpha=0.7, label='Optimal')
    
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Features: Individual vs Full Evaluation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-example feature distribution
    per_example_results = workflow_results['individual_phase']['per_example_results']
    if per_example_results:
        features_counts = [r['features_selected'] for r in per_example_results.values()]
        
        ax2.hist(features_counts, bins=max(1, len(set(features_counts))), alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(features_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(features_counts):.1f}')
        
        ax2.set_xlabel('Features Selected')
        ax2.set_ylabel('Number of Examples')
        ax2.set_title('Distribution of Features Selected per Example')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def save_results(workflow_results: Dict[str, Any], save_path: str):
    """Save results to JSON file."""
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    json_results = json.loads(json.dumps(workflow_results, default=convert_for_json))
    
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {save_path}")

def main():
    from imputer import ImputerEmbedding
    parser = argparse.ArgumentParser(description='Human Q0 Evaluation Workflow')
    parser.add_argument('--stability_threshold', type=float, default=0.05, help='Ranking stability threshold')
    parser.add_argument('--min_selections', type=int, default=0, help='Minimum selections before stop check')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # TODO: Load your model, dataset, and config here
    # model = load_model(args.model_path, YourModelClass, config, args.device)
    # dataset = load_dataset(args.dataset_path)

    dataset = AnnotationDataset("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\input\\data\\test.json")
    model = ImputerEmbedding(7, 5, 6, 4, 64, 18, 19, 0.1)

    model.load_state_dict(torch.load("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\output\\models\\HANNA_NEW_DM_variable_gradient_comparison_20250706_090905.pth"))
    model.to(args.device)
    
    results = evaluate_human_q0_workflow_with_ci_stopping(
        model=model,
        dataset=dataset,
        conformal_delta=0.00624,
        dataset_name="test",
    )
    
    # results_path = os.path.join(args.output_dir, 'workflow_results.json')
    # save_results(results, results_path)
    
    # plot_path = os.path.join(args.output_dir, 'workflow_plot.png')
    # plot_results(results, plot_path)
    
    logger.info("Workflow completed!")

if __name__ == "__main__":
    main()