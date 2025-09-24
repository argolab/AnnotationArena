"""Evaluation engine for ranking/rating imputation experiments."""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.special import softmax
import copy

from .data import RankingData
from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # Total metrics
    total_loss: float
    rating_loss: float
    ranking_loss: float

    # Rating metrics
    rating_accuracy: Optional[float] = None
    rating_rmse: Optional[float] = None
    num_rating_evaluations: int = 0

    # Ranking metrics
    ranking_accuracy: Optional[float] = None
    num_ranking_evaluations: int = 0

    # Breakdown by masked/observed
    masked_metrics: Optional[Dict[str, Any]] = None
    observed_metrics: Optional[Dict[str, Any]] = None


class EvaluationEngine:
    """Engine for evaluating imputation models with configurable masking."""

    def __init__(self, config=None):
        self.config = config
        # Initialize loss strategy with config weights if available
        if config and hasattr(config, 'masked_loss_weight') and hasattr(config, 'observed_loss_weight'):
            self.loss_strategy = DefaultLossStrategy(
                masked_loss_weight=config.masked_loss_weight,
                observed_loss_weight=config.observed_loss_weight
            )
        else:
            self.loss_strategy = DefaultLossStrategy()

    def evaluate_model(self, model, variables: List[RankingData], masking_rate: float, converter=None, device='cpu') -> EvaluationResults:
        """
        Main evaluation function with M% masking.

        Args:
            model: The imputation model to evaluate
            variables: List of variable dictionaries (rating_variables + ranking_variables)
            data: Dictionary containing rating_data and ranking_data
            masking_rate: Fraction of variables to mask (M%)
            converter: DataConverter instance for processing
            device: Device for computation

        Returns:
            EvaluationResults object with comprehensive metrics
        """
        model.eval()

        ref_variables = copy.deepcopy(variables)

        with torch.no_grad():

            # Create batch for evaluation (all variables, but with masking applied)
            if converter is None:
                raise ValueError("DataConverter required for evaluation")

            evaluation_mask = self.create_evaluation_mask(variables)

            # Create batch with evaluation masking
            ranking_data_list = self._create_evaluation_batch(
                variables, evaluation_mask
            )

            # Ensure model is on correct device
            model = model.to(device)
            model_output = model(ranking_data_list)

            # Compute metrics
            results = self._compute_comprehensive_metrics(
                model_output, ref_variables, evaluation_mask, converter
            )

        model.train()
        return results

    def create_evaluation_mask(self, variables: List[RankingData], masking_rate: float) -> List[bool]:
        """
        Create M% random mask across all variables, respecting pre-existing is_masked flags.

        Args:
            variables: List of all variables (ratings + rankings)
            masking_rate: Fraction to mask (0.0 to 1.0)

        Returns:
            List of booleans indicating which variables are masked
        """
        num_variables = len(variables)
        mask = [False] * num_variables

        missing_idx = [i for i in range(num_variables) if variables[i].is_missing]

        for idx in missing_idx:
            mask[idx] = True
        

        return mask

    def split_variables(self, variables: List[RankingData], mask: List[bool]) -> Tuple[List[RankingData], List[RankingData]]:
        """
        Split variables into Test_M (masked) and Test_O (observed).

        Args:
            variables: List of all variables
            mask: Boolean mask (True = masked)

        Returns:
            Tuple of (masked_variables, observed_variables)
        """
        test_m = []  # Masked variables
        test_o = []  # Observed variables

        for var, is_masked in zip(variables, mask):
            if is_masked:
                test_m.append(var)
            else:
                test_o.append(var)

        return test_m, test_o

    def compute_losses(self, model_output: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor],
                      mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute total, rating, and ranking log losses.

        Args:
            model_output: Dictionary with 'rating' and 'ranking' logits
            targets: Dictionary with target tensors
            mask: Mask indicating which positions to evaluate

        Returns:
            Dictionary with loss values
        """
        # Use existing loss strategy but apply to masked positions only
        # This is a simplified version - the main evaluation uses structured approach
        rating_loss = 0.0
        ranking_loss = 0.0

        if 'rating' in model_output and 'rating_targets' in targets:
            rating_logits = model_output['rating']
            rating_targets = targets['rating_targets']
            rating_mask = targets.get('rating_mask', torch.ones_like(rating_targets[:, :, 0]))

            # Apply evaluation mask
            eval_mask = mask & rating_mask
            if eval_mask.any():
                masked_logits = rating_logits[eval_mask]
                masked_targets = rating_targets[eval_mask]
                rating_loss = torch.nn.functional.cross_entropy(
                    masked_logits.view(-1, masked_logits.size(-1)),
                    torch.argmax(masked_targets, dim=-1).view(-1)
                ).item()

        if 'ranking' in model_output and 'ranking_targets' in targets:
            ranking_logits = model_output['ranking']
            ranking_targets = targets['ranking_targets']
            ranking_mask = targets.get('ranking_mask', torch.ones_like(ranking_targets[:, :, 0]))

            # Apply evaluation mask
            eval_mask = mask & ranking_mask
            if eval_mask.any():
                total_ranking_loss = 0.0
                count = 0

                for i in torch.nonzero(eval_mask).flatten():
                    batch_idx = i // eval_mask.size(1)
                    var_idx = i % eval_mask.size(1)

                    # Get logits and target for this ranking
                    logits = ranking_logits[batch_idx, var_idx]
                    target = ranking_targets[batch_idx, var_idx]

                    # Find valid positions (non-zero targets)
                    valid_positions = target > 0
                    if valid_positions.any():
                        valid_target = target[valid_positions]
                        valid_logits = logits[valid_positions]

                        # Get indices sorted by target (lowest rank = best)
                        # E.g., if target=[2,1], sorted_idx=[1,0] (item at pos 1 ranks first)
                        sorted_idx = torch.argsort(valid_target)

                        # Plackett-Luce loss: probability that the top-ranked item is chosen
                        log_probs = torch.log_softmax(valid_logits, dim=-1)
                        ranking_loss_item = -log_probs[sorted_idx[0]]

                        total_ranking_loss += ranking_loss_item.item()
                        count += 1

                ranking_loss = total_ranking_loss / count if count > 0 else 0.0

        return {
            'total_loss': rating_loss + ranking_loss,
            'rating_loss': rating_loss,
            'ranking_loss': ranking_loss
        }

    def compute_accuracies(self, predictions: List, targets: List,
                          variable_types: List[str]) -> Dict[str, float]:
        """
        Compute rating and ranking accuracies.

        Args:
            predictions: List of predictions
            targets: List of ground truth targets
            variable_types: List indicating 'rating' or 'ranking' for each variable

        Returns:
            Dictionary with accuracy metrics
        """
        rating_correct = 0
        rating_total = 0
        ranking_correct = 0
        ranking_total = 0

        for pred, target, var_type in zip(predictions, targets, variable_types):
            if var_type == 'rating':
                if pred == target:
                    rating_correct += 1
                rating_total += 1
            elif var_type == 'ranking':
                if pred == target:  # Simplified - proper ranking comparison needed
                    ranking_correct += 1
                ranking_total += 1

        rating_accuracy = rating_correct / rating_total if rating_total > 0 else None
        ranking_accuracy = ranking_correct / ranking_total if ranking_total > 0 else None

        return {
            'rating_accuracy': rating_accuracy,
            'ranking_accuracy': ranking_accuracy,
            'num_rating_evaluations': rating_total,
            'num_ranking_evaluations': ranking_total
        }

    def compute_rmse(self, rating_predictions: List[int],
                    rating_targets: List[int]) -> float:
        """
        Compute RMSE for rating predictions.

        Args:
            rating_predictions: List of predicted ratings (0-indexed)
            rating_targets: List of true ratings (0-indexed)

        Returns:
            RMSE value
        """
        if len(rating_predictions) == 0:
            return 0.0

        # Convert from 0-indexed to 1-5 scale for RMSE computation
        pred_ratings = [p + 1 for p in rating_predictions]
        true_ratings = [t + 1 for t in rating_targets]

        mse = np.mean([(p - t)**2 for p, t in zip(pred_ratings, true_ratings)])
        return np.sqrt(mse)

    def evaluate_by_type(self, predictions: Dict, targets: Dict,
                        masks: Dict) -> Dict[str, Dict]:
        """
        Separate metrics for ratings vs rankings with masked/observed breakdown.

        Args:
            predictions: Predictions organized by type
            targets: Targets organized by type
            masks: Masks organized by type

        Returns:
            Nested dictionary with metrics by type and mask status
        """
        results = {
            'rating': {'total': {}, 'masked': {}, 'observed': {}},
            'ranking': {'total': {}, 'masked': {}, 'observed': {}}
        }

        # Process each annotation type
        for ann_type in ['rating', 'ranking']:
            if ann_type in predictions:
                type_preds = predictions[ann_type]
                type_targets = targets[ann_type]
                type_mask = masks.get(ann_type, [])

                # Compute metrics for different subsets
                results[ann_type]['total'] = self._compute_subset_metrics(
                    type_preds, type_targets, None, ann_type
                )

                if type_mask:
                    masked_preds = [p for p, m in zip(type_preds, type_mask) if m]
                    masked_targets = [t for t, m in zip(type_targets, type_mask) if m]
                    results[ann_type]['masked'] = self._compute_subset_metrics(
                        masked_preds, masked_targets, None, ann_type
                    )

                    observed_preds = [p for p, m in zip(type_preds, type_mask) if not m]
                    observed_targets = [t for t, m in zip(type_targets, type_mask) if not m]
                    results[ann_type]['observed'] = self._compute_subset_metrics(
                        observed_preds, observed_targets, None, ann_type
                    )

        return results

    def _create_evaluation_batch(self, variables, evaluation_mask):
        """Create batch with evaluation masking applied."""
        # Similar to converter.create_batch but with evaluation masking
        # This applies the evaluation mask to determine which variables have supervision

        if len(variables) != len(evaluation_mask):
            raise ValueError(f"Variables length ({len(variables)}) must match evaluation mask length ({len(evaluation_mask)})")

        masked_variables = []

        for i, var in enumerate(variables):
            if evaluation_mask[i]:
                # Create missing version (remove supervision)
                masked_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_masked=True,  # Mark as masked
                    is_missing=True,
                    rating_value=var.rating_value,  # Keep original value for reference
                    ranking_order=var.ranking_order  # Keep original order for reference
                )
                masked_variables.append(masked_var)
            else:
                # Keep original (observed) for conditioning
                observed_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_missing=False,
                    is_masked=False,  # Mark as observed
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order
                )
                masked_variables.append(observed_var)

        return masked_variables

    def _compute_comprehensive_metrics(self, model_output, variables,
                                     evaluation_mask, converter):
        """Compute comprehensive evaluation metrics."""
        rating_logits = model_output['rating']
        ranking_logits = model_output['ranking']

        # Convert to structured format for loss computation
        predictions_full = adapt_batched_logits_to_predictions(model_output)

        # Separate data by masked status and annotation type
        total_metrics = {'predictions': [], 'references': [],
                        'rating_preds': [], 'rating_targets': [],
                        'ranking_preds': [], 'ranking_targets': []}
        masked_metrics = {'predictions': [], 'references': [],
                         'rating_preds': [], 'rating_targets': [],
                         'ranking_preds': [], 'ranking_targets': []}
        observed_metrics = {'predictions': [], 'references': [],
                           'rating_preds': [], 'rating_targets': [],
                           'ranking_preds': [], 'ranking_targets': []}

        # Process each variable
        for i, (var, is_masked) in enumerate(zip(variables, evaluation_mask)):
            if not var.is_listwise:
                rating_val = var.rating_value
                pred_rating = torch.argmax(rating_logits[0, i]).item()

                # Create structured prediction/reference
                pred_ref_pair = (predictions_full[i], RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=False,
                    item_ids=var.item_ids,
                    rating_value=rating_val,
                    is_masked=is_masked,
                ))

                # Add to total
                total_metrics['predictions'].append(pred_ref_pair[0])
                total_metrics['references'].append(pred_ref_pair[1])
                total_metrics['rating_preds'].append(pred_rating)
                total_metrics['rating_targets'].append(rating_val)

                # Add to masked/observed
                target_metrics = masked_metrics if is_masked else observed_metrics
                target_metrics['predictions'].append(pred_ref_pair[0])
                target_metrics['references'].append(pred_ref_pair[1])
                target_metrics['rating_preds'].append(pred_rating)
                target_metrics['rating_targets'].append(rating_val)

            else:
                ranking_order = var.ranking_order

                # Simplified ranking prediction
                pred_scores = ranking_logits[0, i].cpu().numpy()
                if len(ranking_order) == 2:  # Pairwise ranking
                    position_probs = softmax(pred_scores[:2])
                    pred_first_wins = position_probs[0] > position_probs[1]
                    true_first_wins = ranking_order[0] < ranking_order[1]
                    pred_ranking = [1, 2] if pred_first_wins else [2, 1]
                else:
                    pred_ranking = ranking_order  # Fallback

                # Create structured prediction/reference
                pred_ref_pair = (predictions_full[i], RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=True,
                    is_missing=False,
                    item_ids=var.item_ids,
                    ranking_order=ranking_order,
                    is_masked=is_masked,
                ))

                # Add to total
                total_metrics['predictions'].append(pred_ref_pair[0])
                total_metrics['references'].append(pred_ref_pair[1])
                total_metrics['ranking_preds'].append(pred_ranking)
                total_metrics['ranking_targets'].append(ranking_order)

                # Add to masked/observed
                target_metrics = masked_metrics if is_masked else observed_metrics
                target_metrics['predictions'].append(pred_ref_pair[0])
                target_metrics['references'].append(pred_ref_pair[1])
                target_metrics['ranking_preds'].append(pred_ranking)
                target_metrics['ranking_targets'].append(ranking_order)

        # Compute losses and accuracies for each subset
        def compute_subset_results(metrics_dict):
            if len(metrics_dict['predictions']) == 0:
                return {
                    'total_loss': 0.0, 'rating_loss': 0.0, 'ranking_loss': 0.0,
                    'rating_accuracy': None, 'ranking_accuracy': None, 'rating_rmse': None,
                    'num_rating_evaluations': 0, 'num_ranking_evaluations': 0
                }

            # Compute losses using structured approach
            losses = self.loss_strategy.compute(
                metrics_dict['predictions'], metrics_dict['references']
            )

            # Compute accuracies
            rating_accuracy = None
            rating_rmse = None
            if len(metrics_dict['rating_preds']) > 0:
                correct = sum(p == t for p, t in zip(metrics_dict['rating_preds'],
                                                   metrics_dict['rating_targets']))
                rating_accuracy = correct / len(metrics_dict['rating_preds'])
                rating_rmse = self.compute_rmse(metrics_dict['rating_preds'],
                                              metrics_dict['rating_targets'])

            ranking_accuracy = None
            if len(metrics_dict['ranking_preds']) > 0:
                # Proper ranking accuracy for pairwise rankings
                correct = 0
                for pred_ranking, true_ranking in zip(metrics_dict['ranking_preds'], metrics_dict['ranking_targets']):
                    if len(pred_ranking) == 2 and len(true_ranking) == 2:
                        # For pairwise rankings: correct if both have same relative order
                        pred_first_wins = pred_ranking[0] < pred_ranking[1]
                        true_first_wins = true_ranking[0] < true_ranking[1]
                        if pred_first_wins == true_first_wins:
                            correct += 1
                    else:
                        # For other ranking sizes, exact match
                        if pred_ranking == true_ranking:
                            correct += 1
                ranking_accuracy = correct / len(metrics_dict['ranking_preds'])

            return {
                'total_loss': losses['total_loss'],
                'rating_loss': losses['rating_loss'],
                'ranking_loss': losses['ranking_loss'],
                'rating_accuracy': rating_accuracy,
                'ranking_accuracy': ranking_accuracy,
                'rating_rmse': rating_rmse,
                'num_rating_evaluations': len(metrics_dict['rating_preds']),
                'num_ranking_evaluations': len(metrics_dict['ranking_preds'])
            }

        total_results = compute_subset_results(total_metrics)
        masked_results = compute_subset_results(masked_metrics)
        observed_results = compute_subset_results(observed_metrics)

        return EvaluationResults(
            total_loss=total_results['total_loss'],
            rating_loss=total_results['rating_loss'],
            ranking_loss=total_results['ranking_loss'],
            rating_accuracy=total_results['rating_accuracy'],
            rating_rmse=total_results['rating_rmse'],
            ranking_accuracy=total_results['ranking_accuracy'],
            num_rating_evaluations=total_results['num_rating_evaluations'],
            num_ranking_evaluations=total_results['num_ranking_evaluations'],
            masked_metrics=masked_results,
            observed_metrics=observed_results
        )

    def _compute_subset_metrics(self, predictions, targets, mask, annotation_type):
        """Compute metrics for a subset of predictions/targets."""
        if len(predictions) == 0:
            return {'accuracy': None, 'rmse': None, 'count': 0}

        if annotation_type == 'rating':
            correct = sum(p == t for p, t in zip(predictions, targets))
            accuracy = correct / len(predictions)
            rmse = self.compute_rmse(predictions, targets)
            return {'accuracy': accuracy, 'rmse': rmse, 'count': len(predictions)}
        elif annotation_type == 'ranking':
            # Proper ranking accuracy for pairwise rankings
            correct = 0
            for pred_ranking, true_ranking in zip(predictions, targets):
                if len(pred_ranking) == 2 and len(true_ranking) == 2:
                    # For pairwise rankings: correct if both have same relative order
                    pred_first_wins = pred_ranking[0] < pred_ranking[1]
                    true_first_wins = true_ranking[0] < true_ranking[1]
                    if pred_first_wins == true_first_wins:
                        correct += 1
                else:
                    # For other ranking sizes, exact match
                    if pred_ranking == true_ranking:
                        correct += 1
            accuracy = correct / len(predictions)
            return {'accuracy': accuracy, 'rmse': None, 'count': len(predictions)}

        return {'accuracy': None, 'rmse': None, 'count': 0}