"""Evaluation engine for ranking/rating imputation experiments (no masking during eval)."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from scipy.special import softmax
import copy

from imputer.data import RankingData
from imputer.losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from imputer.metrics import compute_rating_accuracy_rmse, compute_ranking_accuracy


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

    def evaluate_model(self, model, variables: List[RankingData], converter=None, device='cuda') -> EvaluationResults:
        """Evaluate on provided variables as-is; compute metrics/log-loss per status (observed/missing/masked)."""
        if converter is None:
            raise ValueError("DataConverter required for evaluation")

        model.eval()
        # Disable full_dropout for evaluation (if embedding provider supports it)
        embed = getattr(model, "embedding_provider", None)
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(False)
        
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

                for i in indices:
                    var = ref_variables[i]
                    preds.append(predictions_full[i])
                    # Use the original ref as-is
                    refs.append(var)

                # Compute losses on this subset using the loss strategy
                losses = self.loss_strategy.compute(preds, refs)
                total_loss_local = losses['total_loss']
                rating_loss_local = losses['rating_loss']
                ranking_loss_local = losses['ranking_loss']

                # Use shared metric computation functions (filter indices by type)
                rating_indices = [i for i in indices if not ref_variables[i].is_listwise and ref_variables[i].rating_value is not None]
                ranking_indices = [
                    i for i in indices
                    if ref_variables[i].is_listwise
                    and ref_variables[i].ranking_order is not None
                    and len(ref_variables[i].ranking_order) == 2
                ]
                
                rating_accuracy_local, rating_rmse_local = compute_rating_accuracy_rmse(
                    rating_logits[0], rating_indices, ref_variables, rating_logits.device
                )
                ranking_accuracy_local = compute_ranking_accuracy(
                    ranking_logits[0], ranking_indices, ref_variables, ranking_logits.device
                )

                # Return a consistent metrics dict for this subset
                return {
                    'total_loss': total_loss_local,
                    'rating_loss': rating_loss_local,
                    'ranking_loss': ranking_loss_local,
                    'rating_accuracy': rating_accuracy_local,
                    'rating_rmse': rating_rmse_local,
                    'ranking_accuracy': ranking_accuracy_local,
                    'num_rating_evaluations': len(rating_indices),
                    'num_ranking_evaluations': len(ranking_indices),
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

            # Compute aggregated accuracies across all subsets for convenience
            all_idx = observed_idx + missing_idx + masked_idx
            overall_metrics = compute_subset(all_idx)
            results = EvaluationResults(
                total_loss=total_loss,
                rating_loss=rating_loss,
                ranking_loss=ranking_loss,
                rating_accuracy=overall_metrics['rating_accuracy'],
                rating_rmse=overall_metrics['rating_rmse'],
                ranking_accuracy=overall_metrics['ranking_accuracy'],
                num_rating_evaluations=num_rating,
                num_ranking_evaluations=num_ranking,
                observed_metrics=observed_metrics,
                missing_metrics=missing_metrics,
                masked_metrics=masked_metrics,
            )

        model.train()
        return results

    # Deprecated helpers removed; evaluation is fully handled in evaluate_model