"""Evaluation engine for ranking/rating imputation experiments (no masking during eval)."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
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

    # Breakdown by status
    observed_metrics: Optional[Dict[str, Any]] = None
    missing_metrics: Optional[Dict[str, Any]] = None
    masked_metrics: Optional[Dict[str, Any]] = None


class EvaluationEngine:
    """Engine for evaluating imputation models without masking (use observed + missing)."""

    def __init__(self, config=None):
        self.config = config
        # Loss strategy; weights don't matter in eval since masked subset is empty
        self.loss_strategy = DefaultLossStrategy()

    def evaluate_model(self, model, variables: List[RankingData], converter=None, device='cpu') -> EvaluationResults:
        """Evaluate on provided variables as-is; compute metrics/log-loss per status (observed/missing/masked)."""
        if converter is None:
            raise ValueError("DataConverter required for evaluation")

        model.eval()
        ref_variables = copy.deepcopy(variables)

        with torch.no_grad():
            model = model.to(device)
            model_output = model(ref_variables)

            # Build predictions once
            rating_logits = model_output['rating']
            ranking_logits = model_output['ranking']
            predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
            # Partition indices by status
            observed_idx = [i for i, v in enumerate(ref_variables) if v.is_observed]
            missing_idx = [i for i, v in enumerate(ref_variables) if v.is_missing]
            masked_idx = [i for i, v in enumerate(ref_variables) if v.is_masked]

            # Helper to compute loss/metrics for a subset
            def compute_subset(indices: List[int]) -> Dict[str, Any]:
                if not indices:
                    return {
                        'total_loss': 0.0,
                        'rating_loss': 0.0,
                        'ranking_loss': 0.0,
                        'rating_accuracy': None,
                        'rating_rmse': None,
                        'ranking_accuracy': None,
                        'num_rating_evaluations': 0,
                        'num_ranking_evaluations': 0,
                    }

                # Build per-variable predictions and references for this subset
                # - predictions: derived from model logits (already computed)
                # - references: we use the exact original RankingData instances
                #   (ground-truth lives on them; no need to reconstruct)
                preds = []
                refs: List[RankingData] = []
                rating_preds_local: List[int] = []
                rating_targets_local: List[int] = []
                ranking_preds_local: List[List[int]] = []
                ranking_targets_local: List[List[int]] = []

                for i in indices:
                    var = ref_variables[i]
                    preds.append(predictions_full[i])
                    # Use the original ref as-is
                    refs.append(var)
                    if not var.is_listwise:
                        rating_preds_local.append(torch.argmax(rating_logits[0, i]).item())
                        rating_targets_local.append(var.rating_value)
                    else:
                        # For ranking, we compute a simple pairwise prediction if size==2
                        scores = ranking_logits[0, i].cpu().numpy()
                        if len(var.ranking_order or []) == 2:
                            probs = softmax(scores[:2])
                            pred_first_wins = probs[0] > probs[1]
                            pred_ranking = [1, 2] if pred_first_wins else [2, 1]
                        else:
                            pred_ranking = var.ranking_order
                        ranking_preds_local.append(pred_ranking)
                        ranking_targets_local.append(var.ranking_order)

                # Compute losses on this subset using the loss strategy
                losses = self.loss_strategy.compute(preds, refs)
                total_loss_local = losses['total_loss']
                rating_loss_local = losses['rating_loss']
                ranking_loss_local = losses['ranking_loss']

                rating_accuracy_local = None
                rating_rmse_local = None
                if len(rating_preds_local) > 0:
                    correct = sum(p == t for p, t in zip(rating_preds_local, rating_targets_local))
                    rating_accuracy_local = correct / len(rating_preds_local)
                    pred_ratings = [p + 1 for p in rating_preds_local]
                    true_ratings = [t + 1 for t in rating_targets_local]
                    mse = np.mean([(p - t) ** 2 for p, t in zip(pred_ratings, true_ratings)])
                    rating_rmse_local = float(np.sqrt(mse))

                ranking_accuracy_local = None
                if len(ranking_preds_local) > 0:
                    correct = 0
                    for pred_r, true_r in zip(ranking_preds_local, ranking_targets_local):
                        if len(pred_r) == 2 and len(true_r) == 2:
                            pred_first = pred_r[0] < pred_r[1]
                            true_first = true_r[0] < true_r[1]
                            if pred_first == true_first:
                                correct += 1
                        else:
                            if pred_r == true_r:
                                correct += 1
                    ranking_accuracy_local = correct / len(ranking_preds_local)

                # Return a consistent metrics dict for this subset
                return {
                    'total_loss': total_loss_local,
                    'rating_loss': rating_loss_local,
                    'ranking_loss': ranking_loss_local,
                    'rating_accuracy': rating_accuracy_local,
                    'rating_rmse': rating_rmse_local,
                    'ranking_accuracy': ranking_accuracy_local,
                    'num_rating_evaluations': len(rating_preds_local),
                    'num_ranking_evaluations': len(ranking_preds_local),
                }

            observed_metrics = compute_subset(observed_idx)
            missing_metrics = compute_subset(missing_idx)
            masked_metrics = compute_subset(masked_idx)

            # Combine totals across all variables
            total_loss = observed_metrics['total_loss'] + missing_metrics['total_loss'] + masked_metrics['total_loss']
            rating_loss = observed_metrics['rating_loss'] + missing_metrics['rating_loss'] + masked_metrics['rating_loss']
            ranking_loss = observed_metrics['ranking_loss'] + missing_metrics['ranking_loss'] + masked_metrics['ranking_loss']

            # Aggregate counts for top-level fields
            num_rating = observed_metrics['num_rating_evaluations'] + missing_metrics['num_rating_evaluations'] + masked_metrics['num_rating_evaluations']
            num_ranking = observed_metrics['num_ranking_evaluations'] + missing_metrics['num_ranking_evaluations'] + masked_metrics['num_ranking_evaluations']

            # Do not compute aggregated accuracies; keep them per-status for clarity
            results = EvaluationResults(
                total_loss=total_loss,
                rating_loss=rating_loss,
                ranking_loss=ranking_loss,
                rating_accuracy=None,
                rating_rmse=None,
                ranking_accuracy=None,
                num_rating_evaluations=num_rating,
                num_ranking_evaluations=num_ranking,
                observed_metrics=observed_metrics,
                missing_metrics=missing_metrics,
                masked_metrics=masked_metrics,
            )

        model.train()
        return results

    # Deprecated helpers removed; evaluation is fully handled in evaluate_model