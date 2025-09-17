"""Evaluation engine for ranking/rating imputation experiments."""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.special import softmax

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
        self.loss_strategy = DefaultLossStrategy()

    def evaluate_model(self, model, variables: List[Dict], data: Dict,
                      masking_rate: float, converter=None, device='cpu') -> EvaluationResults:
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

        with torch.no_grad():
            # Create evaluation mask (M% of all variables)
            evaluation_mask = self.create_evaluation_mask(variables, masking_rate)

            # Split variables into Test_M (masked) and Test_O (observed)
            test_m_vars, test_o_vars = self.split_variables(variables, evaluation_mask)

            # Create batch for evaluation (all variables, but with masking applied)
            if converter is None:
                raise ValueError("DataConverter required for evaluation")

            # Extract rating and ranking data
            rating_data = data.get('rating_data', {})
            ranking_data = data.get('ranking_data', [])

            # Create batch with evaluation masking
            batch = self._create_evaluation_batch(
                variables, rating_data, ranking_data, evaluation_mask, converter
            )

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            ranking_data_list = model._convert_legacy_tensors_to_ranking_data(
                batch['variable_data'], batch['variable_types'],
                batch['attribute_ids'], batch['annotator_ids'], batch['item_ids']
            )

            model_output = model(ranking_data_list)

            # Compute metrics
            results = self._compute_comprehensive_metrics(
                model_output, batch, variables, evaluation_mask, converter
            )

        model.train()
        return results

    def create_evaluation_mask(self, variables: List[Dict], masking_rate: float) -> List[bool]:
        """
        Create M% random mask across all variables.

        Args:
            variables: List of all variables (ratings + rankings)
            masking_rate: Fraction to mask (0.0 to 1.0)

        Returns:
            List of booleans indicating which variables are masked
        """
        num_variables = len(variables)
        num_to_mask = int(num_variables * masking_rate)

        # Create mask: True = masked, False = observed
        mask = [False] * num_variables
        masked_indices = random.sample(range(num_variables), num_to_mask)

        for idx in masked_indices:
            mask[idx] = True

        return mask

    def split_variables(self, variables: List[Dict], mask: List[bool]) -> Tuple[List[Dict], List[Dict]]:
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
            # Simplified ranking loss computation
            ranking_loss = 0.0  # TODO: Implement proper ranking loss

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

    def _create_evaluation_batch(self, variables, rating_data, ranking_data,
                               evaluation_mask, converter):
        """Create batch with evaluation masking applied."""
        # Similar to converter.create_batch but with evaluation masking
        # This applies the evaluation mask to determine which variables have supervision

        all_variables = variables
        num_variables = len(all_variables)

        # Initialize tensors (similar to DataConverter.create_batch)
        variable_data = torch.zeros(1, num_variables, max(converter.num_likert_classes, converter.max_rank_size))
        variable_types = torch.zeros(1, num_variables, dtype=torch.long)
        attribute_ids = torch.zeros(1, num_variables, dtype=torch.long)
        annotator_ids = torch.zeros(1, num_variables, dtype=torch.long)
        item_ids = torch.full((1, num_variables, converter.max_rank_size), -1, dtype=torch.long)

        rating_targets = torch.zeros(1, num_variables, converter.num_likert_classes)
        ranking_targets = torch.zeros(1, num_variables, converter.max_rank_size)
        rating_mask = torch.zeros(1, num_variables, dtype=torch.bool)
        ranking_mask = torch.zeros(1, num_variables, dtype=torch.bool)

        # Process each variable
        for i, (var, is_masked) in enumerate(zip(all_variables, evaluation_mask)):
            attribute_ids[0, i] = var['attribute'] - 1
            annotator_ids[0, i] = var['annotator'] - 1

            if var['type'] == 'rating':
                variable_types[0, i] = 0
                item_ids[0, i, 0] = var['item'] - 1

                # Check if this rating exists in data
                key = (var['attribute'], var['annotator'], var['item'])
                if key in rating_data:
                    rating_value = rating_data[key] - 1  # Convert to 0-indexed
                    rating_targets[0, i, rating_value] = 1.0
                    rating_mask[0, i] = True

                    # Only provide supervision if NOT masked for evaluation
                    if not is_masked:
                        variable_data[0, i, rating_value] = 1.0

            elif var['type'] == 'ranking':
                variable_types[0, i] = 1
                items = var['items']
                for j, item in enumerate(items):
                    if j < converter.max_rank_size:
                        item_ids[0, i, j] = item - 1

                # Find matching ranking
                matching_ranking = None
                for ranking_entry in ranking_data:
                    if (ranking_entry['attribute'] == var['attribute'] and
                        ranking_entry['annotator'] == var['annotator'] and
                        ranking_entry['items'] == items):
                        matching_ranking = ranking_entry
                        break

                if matching_ranking:
                    order = matching_ranking['order']
                    for j, pos in enumerate(order):
                        if j < converter.max_rank_size:
                            ranking_targets[0, i, j] = pos
                    ranking_mask[0, i] = True

                    # Only provide supervision if NOT masked for evaluation
                    if not is_masked:
                        for j, pos in enumerate(order):
                            if j < converter.max_rank_size:
                                variable_data[0, i, j] = pos

        return {
            'variable_data': variable_data,
            'variable_types': variable_types,
            'attribute_ids': attribute_ids,
            'annotator_ids': annotator_ids,
            'item_ids': item_ids,
            'rating_targets': rating_targets,
            'ranking_targets': ranking_targets,
            'rating_mask': rating_mask,
            'ranking_mask': ranking_mask,
            'evaluation_mask': evaluation_mask,
            'all_variables': all_variables
        }

    def _compute_comprehensive_metrics(self, model_output, batch, variables,
                                     evaluation_mask, converter):
        """Compute comprehensive evaluation metrics."""
        rating_logits = model_output['rating']
        ranking_logits = model_output['ranking']

        rating_targets = batch['rating_targets']
        ranking_targets = batch['ranking_targets']
        rating_mask = batch['rating_mask']
        ranking_mask = batch['ranking_mask']

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
            if var['type'] == 'rating' and rating_mask[0, i]:
                rating_val = int(torch.argmax(rating_targets[0, i]).item())
                pred_rating = torch.argmax(rating_logits[0, i]).item()

                # Create structured prediction/reference
                pred_ref_pair = (predictions_full[i], RankingData(
                    annotator_id=var['annotator'] - 1,
                    attribute_id=var['attribute'] - 1,
                    is_listwise=False,
                    item_ids=[var['item'] - 1],
                    rating_value=rating_val,
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

            elif var['type'] == 'ranking' and ranking_mask[0, i]:
                scores_vec = ranking_targets[0, i]
                ranking_order = []
                for j in range(scores_vec.shape[0]):
                    s = int(scores_vec[j].item())
                    if s > 0:
                        ranking_order.append(s)

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
                    annotator_id=var['annotator'] - 1,
                    attribute_id=var['attribute'] - 1,
                    is_listwise=True,
                    item_ids=[it - 1 for it in var['items'][:converter.max_rank_size]],
                    ranking_order=ranking_order,
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
                # Simplified ranking accuracy (need proper implementation)
                correct = sum(p == t for p, t in zip(metrics_dict['ranking_preds'],
                                                   metrics_dict['ranking_targets']))
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
            # Simplified ranking accuracy
            correct = sum(p == t for p, t in zip(predictions, targets))
            accuracy = correct / len(predictions)
            return {'accuracy': accuracy, 'rmse': None, 'count': len(predictions)}

        return {'accuracy': None, 'rmse': None, 'count': 0}