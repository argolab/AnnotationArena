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

    def evaluate_model(self, model, variables: List[RankingData], converter=None, device='cuda', max_item=30) -> EvaluationResults:
        """Evaluate on provided variables as-is; compute metrics/log-loss per status (observed/missing/masked).
        
        If the total number of unique items exceeds max_item, evaluation is done in chunks
        to match training behavior and prevent train-test mismatch.
        """
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
            
            # Collect all unique items across all variables
            all_items: Set[int] = set()
            for var in ref_variables:
                all_items.update(var.item_ids)
            
            # Create ordered list and split into chunks
            all_items_list = sorted(all_items)
            num_items = len(all_items_list)
            
            if max_item is not None and num_items > max_item:
                # Split into chunks of max_item size
                item_chunks = []
                for i in range(0, num_items, max_item):
                    chunk = set(all_items_list[i:i + max_item])
                    item_chunks.append(chunk)
            else:
                # All items fit in one batch
                item_chunks = [all_items]
            
            # Collect all logits and refs across chunks (for metric computation)
            all_rating_logits: List[torch.Tensor] = []
            all_ranking_logits: List[torch.Tensor] = []
            all_refs: List[RankingData] = []
            all_predictions: List = []
            
            for chunk_idx, available_items in enumerate(item_chunks):
                # Filter variables for this chunk
                chunk_vars: List[RankingData] = []
                for var in ref_variables:
                    if all(item_id in available_items for item_id in var.item_ids):
                        chunk_vars.append(var)
                
                if not chunk_vars:
                    continue
                
                # Forward pass for this chunk
                model_output = model(chunk_vars)
                
                rating_logits = model_output['rating'][0]  # Remove batch dim
                ranking_logits = model_output['ranking'][0]
                
                # Store for later
                all_rating_logits.append(rating_logits)
                all_ranking_logits.append(ranking_logits)
                all_refs.extend(chunk_vars)
                
                # Build predictions for this chunk
                predictions_chunk = adapt_batched_logits_to_predictions({
                    'rating': model_output['rating'],
                    'ranking': model_output['ranking']
                })
                all_predictions.extend(predictions_chunk)
            
            # Concatenate all logits
            if all_rating_logits:
                combined_rating_logits = torch.cat(all_rating_logits, dim=0)
                combined_ranking_logits = torch.cat(all_ranking_logits, dim=0)
            else:
                # No variables processed
                model.train()
                return EvaluationResults(
                    total_loss=0.0,
                    rating_loss=0.0,
                    ranking_loss=0.0,
                    rating_accuracy=None,
                    rating_rmse=None,
                    ranking_accuracy=None,
                    num_rating_evaluations=0,
                    num_ranking_evaluations=0,
                    observed_metrics={},
                    missing_metrics={},
                    masked_metrics={},
                )
            
            # Partition indices by status
            observed_idx = [i for i, v in enumerate(all_refs) if v.is_observed]
            missing_idx = [i for i, v in enumerate(all_refs) if v.is_missing]
            masked_idx = [i for i, v in enumerate(all_refs) if v.is_masked]

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

                preds = [all_predictions[i] for i in indices]
                refs = [all_refs[i] for i in indices]

                # Compute losses on this subset using the loss strategy
                losses = self.loss_strategy.compute(preds, refs)
                total_loss_local = losses['total_loss']
                rating_loss_local = losses['rating_loss']
                ranking_loss_local = losses['ranking_loss']

                # Use shared metric computation functions (filter indices by type)
                rating_indices = [i for i in indices if not all_refs[i].is_listwise and all_refs[i].rating_value is not None]
                ranking_indices = [
                    i for i in indices
                    if all_refs[i].is_listwise
                    and all_refs[i].ranking_order is not None
                    and len(all_refs[i].ranking_order) == 2
                ]
                
                rating_accuracy_local, rating_rmse_local = compute_rating_accuracy_rmse(
                    combined_rating_logits, rating_indices, all_refs, combined_rating_logits.device
                )
                ranking_accuracy_local = compute_ranking_accuracy(
                    combined_ranking_logits, ranking_indices, all_refs, combined_ranking_logits.device
                )

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