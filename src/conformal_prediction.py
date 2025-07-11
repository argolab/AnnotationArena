import torch
import copy
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from annotationArena import AnnotationArena
from selection import SelectionFactory
from utils import AnnotationDataset
from eval import ModelEvaluator
from imputer import ImputerEmbedding
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calibrate_voi_confidence_intervals(model, calibration_dataset, dataset_name: str = "calibration",
                                     target_question: int = 0,
                                     confidence_level: float = 0.95,
                                     device: str = "cuda") -> Dict[str, Any]:
    """
    Calibrate confidence intervals for VOI predictions by simulating the full feature acquisition workflow
    on a calibration set and recording non-conformity scores.
    
    Args:
        model: The model to evaluate
        calibration_dataset: The calibration dataset
        dataset_name: Name of the dataset
        target_question: Target question index (default 0 for human Q0)
        confidence_level: Confidence level for intervals (default 0.95)
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing calibration results and confidence intervals
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VOI Confidence Interval Calibration")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Target question: {target_question}")
    logger.info(f"Confidence level: {confidence_level}")
    logger.info(f"{'='*60}")
    
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create dataset copy and initialize components
    dataset_copy = copy.deepcopy(calibration_dataset)
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_copy)
    feature_selector = SelectionFactory.create_feature_strategy('voi', model, device)
    
    # Initialize results tracking
    calibration_results = {
        'dataset_name': dataset_name,
        'target_question': target_question,
        'confidence_level': confidence_level,
        'non_conformity_scores': [],  # All non-conformity scores across all examples and steps
        'per_example_results': {},
        'confidence_intervals': {},
        'quantiles': {}
    }
    
    
    logger.info(f"\n=== Processing {len(dataset_copy)} calibration examples ===")
    
    # Process each example in the calibration set
    for example_idx in tqdm(range(len(dataset_copy))):
        logger.debug(f"\n--- Processing Example {example_idx} ---")
        
        # Initialize per-example tracking
        example_results = {
            'selected_positions': [],
            'voi_scores': [],
            'actual_loss_reductions': [],
            'non_conformity_scores': [],
            'cumulative_loss_history': []
        }
        
        # Get baseline loss for this example before any feature selection
        baseline_loss = compute_example_loss(model, dataset_copy, example_idx, target_question, device)
        current_loss = baseline_loss
        example_results['cumulative_loss_history'].append(current_loss)
        
        # Continue until all features are selected (no early stopping)
        features_selected = 0
        max_features = 14  # Adjust based on your dataset structure
        
        while features_selected < max_features:
            # Get current VOI ranking for this example
            current_voi_ranking = feature_selector.select_features(
                example_idx, dataset_copy,
                num_to_select=max_features - features_selected,
                loss_type="cross_entropy",
                target_questions=[0]
            )
            
            if not current_voi_ranking:
                logger.debug(f"  No more features available for example {example_idx}")
                break
            
            # Filter out human Q0 (unless it's the only option left)
            filtered_ranking = []
            data_entry = dataset_copy.get_data_entry(example_idx)
            
            for feature_info in current_voi_ranking:
                pos = feature_info[0]
                question_idx = data_entry['questions'][pos]
                annotator_idx = data_entry['annotators'][pos]
                
                # Skip human Q0 unless it's the only remaining feature
                if question_idx == 0 and annotator_idx != -1 and features_selected < max_features - 1:
                    continue
                
                filtered_ranking.append(feature_info)
            
            if not filtered_ranking:
                logger.debug(f"  No valid features remaining for example {example_idx}")
                break
            
            # Select the top feature
            top_feature = filtered_ranking[0]
            pos, predicted_voi = top_feature[0], top_feature[1]
            
            # Record the predicted VOI
            example_results['voi_scores'].append(predicted_voi)
            example_results['selected_positions'].append(pos)
            
            # Observe the feature and compute actual loss reduction
            arena.observe_position(example_idx, pos)
            features_selected += 1
            
            # Compute new loss after observing this feature
            new_loss = compute_example_loss(model, dataset_copy, example_idx, target_question, device)
            actual_loss_reduction = current_loss - new_loss
            example_results['actual_loss_reductions'].append(actual_loss_reduction)
            example_results['cumulative_loss_history'].append(new_loss)
            
            # Compute non-conformity score: |predicted_voi - actual_loss_reduction|
            non_conformity_score = abs(predicted_voi - actual_loss_reduction)
            example_results['non_conformity_scores'].append(non_conformity_score)
            calibration_results['non_conformity_scores'].append(non_conformity_score)
            
            logger.debug(f"  Step {features_selected}: Pos {pos}, Predicted VOI: {predicted_voi:.4f}, "
                        f"Actual reduction: {actual_loss_reduction:.4f}, "
                        f"Non-conformity: {non_conformity_score:.4f}")
            
            # Update current loss for next iteration
            current_loss = new_loss
        
        # Store results for this example
        calibration_results['per_example_results'][example_idx] = example_results
        logger.debug(f"  Example {example_idx} completed: {features_selected} features selected")
    
    # Compute confidence intervals from non-conformity scores
    all_non_conformity = np.array(calibration_results['non_conformity_scores'])
    
    if len(all_non_conformity) == 0:
        logger.warning("No non-conformity scores collected!")
        return calibration_results
    
    # Compute quantiles for confidence intervals
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    # Key quantiles for conformal prediction
    calibration_results['quantiles'] = {
        'alpha': alpha,
        'lower_quantile': lower_quantile,
        'upper_quantile': upper_quantile,
        f'q_{lower_quantile:.3f}': np.quantile(all_non_conformity, lower_quantile),
        f'q_{upper_quantile:.3f}': np.quantile(all_non_conformity, upper_quantile),
        'q_0.50': np.quantile(all_non_conformity, 0.5),  # median
        'q_0.90': np.quantile(all_non_conformity, 0.9),
        'q_0.95': np.quantile(all_non_conformity, 0.95),
        'q_0.99': np.quantile(all_non_conformity, 0.99)
    }
    
    # The key threshold for conformal prediction intervals
    conformal_threshold = np.quantile(all_non_conformity, upper_quantile)
    
    calibration_results['confidence_intervals'] = {
        'conformal_threshold': conformal_threshold,
        'confidence_level': confidence_level,
        'method': 'conformal_prediction',
        'description': f"For future VOI predictions, the actual loss reduction will be within "
                     f"[predicted_voi - {conformal_threshold:.4f}, predicted_voi + {conformal_threshold:.4f}] "
                     f"with {confidence_level*100:.1f}% confidence"
    }
    
    # Summary statistics
    calibration_results['summary_stats'] = {
        'total_non_conformity_scores': len(all_non_conformity),
        'mean_non_conformity': np.mean(all_non_conformity),
        'std_non_conformity': np.std(all_non_conformity),
        'min_non_conformity': np.min(all_non_conformity),
        'max_non_conformity': np.max(all_non_conformity),
        'total_examples_processed': len(calibration_results['per_example_results']),
        'avg_features_per_example': np.mean([len(ex['voi_scores']) 
                                           for ex in calibration_results['per_example_results'].values()])
    }
    
    # Log results
    logger.info(f"\n=== Calibration Results ===")
    logger.info(f"Total non-conformity scores: {len(all_non_conformity)}")
    logger.info(f"Mean non-conformity: {np.mean(all_non_conformity):.4f}")
    logger.info(f"Std non-conformity: {np.std(all_non_conformity):.4f}")
    logger.info(f"Conformal threshold ({confidence_level*100:.1f}%): {conformal_threshold:.4f}")
    logger.info(f"Interpretation: Future VOI predictions will be accurate within ±{conformal_threshold:.4f} "
               f"with {confidence_level*100:.1f}% confidence")
    
    return calibration_results


def compute_example_loss(model, dataset, example_idx: int, target_question: int, device) -> float:
    """
    Compute the loss for a specific example and target question.
    
    Args:
        model: The model
        dataset: The dataset
        example_idx: Index of the example
        target_question: Target question index
        device: Device for computation
        
    Returns:
        Loss value for the example
    """
    model.eval()
    
    # Get data for this example
    known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            inputs.unsqueeze(0).to(device), 
            annotators.unsqueeze(0).to(device), 
            questions.unsqueeze(0).to(device), 
            embeddings.unsqueeze(0).to(device)
        )
        
        # Get prediction for target question
        prediction = torch.softmax(outputs[:, target_question, :], dim=-1)
        target_answer = answers[target_question].to(device)
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            prediction, 
            target_answer.unsqueeze(0), 
            reduction='none'
        )
        
        return loss.item()
    
dataset = AnnotationDataset("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\input\\data\\validation.json")
model = ImputerEmbedding(7, 5, 6, 4, 64, 18, 19, 0.1)

model.load_state_dict(torch.load("C:\\Users\\stone\\Projects\\AnnotationArena\\src\\output\\models\\HANNA_NEW_DM_variable_gradient_comparison_20250706_090905.pth"))
model.to("cuda")
calibrate_voi_confidence_intervals(model, dataset, target_question=7)