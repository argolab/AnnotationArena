"""
Shared metric computation utilities for training and evaluation.

This module provides unified metric computation logic used by both:
- Training-time metrics (lightweight, frequent)
- Epoch-end evaluation metrics (comprehensive, infrequent)

All metric computation follows the same logic to ensure consistency.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from scipy.special import softmax

from imputer.data import RankingData


def compute_rating_accuracy_rmse(
    rating_logits: torch.Tensor,
    rating_indices: List[int],
    variables: List[RankingData],
    device: torch.device
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute rating accuracy and RMSE for a subset of variables.
    
    Args:
        rating_logits: [N, num_classes] logits tensor
        rating_indices: List of indices into variables that are ratings
        variables: List of RankingData variables
        device: Device for tensor operations
    
    Returns:
        (accuracy, rmse) tuple, or (None, None) if no ratings
    """
    if not rating_indices:
        return None, None
    
    # Compute predictions and targets
    preds = torch.argmax(rating_logits[rating_indices], dim=-1)
    targets = torch.tensor(
        [variables[i].rating_value for i in rating_indices],
        device=device,
        dtype=torch.long
    )
    
    # Accuracy: exact match
    correct = (preds == targets).float()
    accuracy = correct.mean().item()
    
    # RMSE: convert to 1-indexed for comparison (same as eval.py:127-130)
    pred_ratings = preds.float() + 1.0
    true_ratings = targets.float() + 1.0
    rmse = torch.sqrt(((pred_ratings - true_ratings) ** 2).mean()).item()
    
    return accuracy, rmse


def compute_ranking_accuracy(
    ranking_logits: torch.Tensor,
    ranking_indices: List[int],
    variables: List[RankingData],
    device: torch.device
) -> Optional[float]:
    """
    Compute pairwise ranking accuracy for a subset of variables.
    
    Args:
        ranking_logits: [N, max_rank_size] logits tensor
        ranking_indices: List of indices into variables that are rankings
        variables: List of RankingData variables
        device: Device for tensor operations
    
    Returns:
        Accuracy as float, or None if no rankings
    """
    if not ranking_indices:
        return None
    
    # For pairwise rankings (size=2), compute accuracy
    correct = 0
    total = 0
    
    for i in ranking_indices:
        var = variables[i]
        if var.ranking_order is None or len(var.ranking_order) != 2:
            continue
        
        # Prediction: first item wins if logits[0] > logits[1]
        logits = ranking_logits[i, :2]
        pred_first_wins = logits[0] > logits[1]
        
        # Ground truth: first item wins if ranking_order[0] < ranking_order[1]
        true_first_wins = var.ranking_order[0] < var.ranking_order[1]
        
        if pred_first_wins.item() == true_first_wins:
            correct += 1
        total += 1
    
    if total == 0:
        return None
    
    return correct / total


def compute_rating_entropy_metrics(
    rating_logits: torch.Tensor,
    rating_indices: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Compute entropy-based metrics for ratings (training-specific).
    
    Args:
        rating_logits: [N, num_classes] logits tensor
        rating_indices: List of indices to compute metrics for
    
    Returns:
        Dict with 'pred_entropy' and 'pred_max_prob' tensors
    """
    if not rating_indices:
        return {}
    
    logits = rating_logits[rating_indices]
    probs = torch.softmax(logits, dim=-1)
    
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    
    return {
        'pred_entropy': entropy.mean(),
        'pred_max_prob': max_prob.mean()
    }


def partition_variables_by_status(
    variables: List[RankingData],
    is_rating: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Partition variable indices by status (observed, missing, masked).
    
    Args:
        variables: List of RankingData variables
        is_rating: If True, filter for ratings; if False, filter for rankings
    
    Returns:
        (observed_indices, missing_indices, masked_indices)
    """
    observed = []
    missing = []
    masked = []
    
    for i, var in enumerate(variables):
        # Filter by type
        if is_rating and var.is_listwise:
            continue
        if not is_rating and not var.is_listwise:
            continue
        
        # Filter by having ground truth
        if is_rating and var.rating_value is None:
            continue
        if not is_rating and (var.ranking_order is None or len(var.ranking_order) != 2):
            continue
        
        # Partition by status
        if var.is_observed:
            observed.append(i)
        elif var.is_missing:
            missing.append(i)
        elif var.is_masked:
            masked.append(i)
    
    return observed, missing, masked


def compute_metrics_for_subset(
    rating_logits: torch.Tensor,
    ranking_logits: torch.Tensor,
    indices: List[int],
    variables: List[RankingData],
    device: torch.device,
    include_entropy: bool = False
) -> Dict[str, any]:
    """
    Compute all metrics (accuracy, RMSE) for a subset of variables.
    
    This is the core shared logic used by both training and evaluation.
    
    Args:
        rating_logits: [N, num_classes] logits tensor
        ranking_logits: [N, max_rank_size] logits tensor
        indices: List of variable indices to compute metrics for
        variables: List of RankingData variables
        device: Device for tensor operations
        include_entropy: If True, include entropy metrics (training-specific)
    
    Returns:
        Dict with rating_accuracy, rating_rmse, ranking_accuracy, and optionally entropy metrics
    """
    if not indices:
        return {
            'rating_accuracy': None,
            'rating_rmse': None,
            'ranking_accuracy': None,
            'num_rating_evaluations': 0,
            'num_ranking_evaluations': 0,
        }
    
    # Partition indices by type
    rating_indices = [i for i in indices if not variables[i].is_listwise and variables[i].rating_value is not None]
    ranking_indices = [
        i for i in indices
        if variables[i].is_listwise
        and variables[i].ranking_order is not None
        and len(variables[i].ranking_order) == 2
    ]
    
    # Compute metrics
    rating_accuracy, rating_rmse = compute_rating_accuracy_rmse(
        rating_logits, rating_indices, variables, device
    )
    ranking_accuracy = compute_ranking_accuracy(
        ranking_logits, ranking_indices, variables, device
    )
    
    result = {
        'rating_accuracy': rating_accuracy,
        'rating_rmse': rating_rmse,
        'ranking_accuracy': ranking_accuracy,
        'num_rating_evaluations': len(rating_indices),
        'num_ranking_evaluations': len(ranking_indices),
    }
    
    # Add entropy metrics if requested (training-specific)
    if include_entropy and rating_indices:
        entropy_metrics = compute_rating_entropy_metrics(rating_logits, rating_indices)
        result.update(entropy_metrics)
    
    return result


def log_instance_metrics_to_tb(pl_module, result: Dict[str, Any], instance_name: str) -> None:
    """
    Log evaluation metrics to TensorBoard with unified structure {instance}/{type}/{metric}/{status}.
    
    This is called from on_train_epoch_end() after EvaluationCallback.on_epoch_end() returns results.
    """
    def _log_rating(metrics: Optional[Dict], status: str) -> None:
        if not metrics:
            return
        prefix = f"{instance_name}/rating"
        if "rating_loss" in metrics:
            pl_module.log(f"{prefix}/xent_loss/{status}", metrics["rating_loss"], on_epoch=True)
        if metrics.get("rating_accuracy") is not None:
            pl_module.log(f"{prefix}/accuracy/{status}", metrics["rating_accuracy"], on_epoch=True)
        if metrics.get("rating_rmse") is not None:
            pl_module.log(f"{prefix}/rmse/{status}", metrics["rating_rmse"], on_epoch=True)

    def _log_ranking(metrics: Optional[Dict], status: str) -> None:
        if not metrics:
            return
        prefix = f"{instance_name}/ranking"
        if "ranking_loss" in metrics:
            pl_module.log(f"{prefix}/xent_loss/{status}", metrics["ranking_loss"], on_epoch=True)
        if metrics.get("ranking_accuracy") is not None:
            pl_module.log(f"{prefix}/accuracy/{status}", metrics["ranking_accuracy"], on_epoch=True)

    _log_rating(result.get("observed_metrics"), "observed")
    _log_rating(result.get("missing_metrics"), "missing")
    _log_rating(result.get("masked_metrics"), "masked")
    _log_ranking(result.get("observed_metrics"), "observed")
    _log_ranking(result.get("missing_metrics"), "missing")
    _log_ranking(result.get("masked_metrics"), "masked")

    pl_module.log(f"{instance_name}/total/xent_loss/total", result["total_loss"], on_epoch=True, prog_bar=True)
    pl_module.log(f"{instance_name}/rating/xent_loss/total", result["rating_loss"], on_epoch=True, prog_bar=True)
    pl_module.log(f"{instance_name}/ranking/xent_loss/total", result["ranking_loss"], on_epoch=True)
    if result.get("rating_accuracy") is not None:
        pl_module.log(f"{instance_name}/rating/accuracy/total", result["rating_accuracy"], on_epoch=True, prog_bar=True)
    if result.get("ranking_accuracy") is not None:
        pl_module.log(f"{instance_name}/ranking/accuracy/total", result["ranking_accuracy"], on_epoch=True)
    if result.get("rating_rmse") is not None:
        pl_module.log(f"{instance_name}/rating/rmse/total", result["rating_rmse"], on_epoch=True)
