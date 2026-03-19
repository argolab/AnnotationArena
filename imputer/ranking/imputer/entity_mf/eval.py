from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from imputer.data import RankingData

from .data import variable_list_to_entity_graph
from .types import LossBreakdown, EntityType


def _aggregate_loss_from_breakdowns(
    params: torch.Tensor,
    graph: Any,
    types: Dict[str, Any],
    global_param_dim: int,
    device: torch.device,
) -> dict:
    """
    Aggregate per-type LossBreakdown into global observed/masked/missing losses
    and counts, and optionally a per-(status, type_name) breakdown.
    """
    weighted_masked_sum = torch.zeros((), device=device)
    weighted_observed_sum = torch.zeros((), device=device)
    sum_observed = 0.0
    sum_masked = 0.0
    sum_missing = 0.0
    n_observed = 0
    n_masked = 0
    n_missing = 0
    # Per-(status, type_name) metrics: metrics[status][type_name] = {"xent": ..., "n": ...}
    per_type: Dict[str, Dict[str, Dict[str, float]]] = {
        "observed": {},
        "masked": {},
        "missing": {},
    }
    
    for type_name, t in types.items():
        type_mask = torch.tensor(
            [tok.type_name == type_name for tok in graph.tokens],
            device=device,
            dtype=torch.bool,
        ).unsqueeze(0)  # [1, L]
        
        b: LossBreakdown = t.compute_loss_breakdown(
            predicted_params=params,
            tokens=graph.tokens,
            type_mask=type_mask,
            global_param_dim=global_param_dim,
        )
        
        if b.n_masked > 0:
            weighted_masked_sum = weighted_masked_sum + b.trainable_loss * b.n_masked
        if b.n_observed > 0:
            weighted_observed_sum = weighted_observed_sum + b.observed_loss_tensor * b.n_observed
            sum_observed += b.loss_observed * b.n_observed
            n_observed += b.n_observed
            per_type["observed"][type_name] = {
                "xent": float(b.loss_observed),
                "n": float(b.n_observed),
            }
        if b.n_masked > 0:
            sum_masked += b.loss_masked * b.n_masked
            n_masked += b.n_masked
            per_type["masked"][type_name] = {
                "xent": float(b.loss_masked),
                "n": float(b.n_masked),
            }
        if b.n_missing > 0:
            sum_missing += b.loss_missing * b.n_missing
            n_missing += b.n_missing
            per_type["missing"][type_name] = {
                "xent": float(b.loss_missing),
                "n": float(b.n_missing),
            }
    
    loss_observed = sum_observed / n_observed if n_observed else 0.0
    loss_masked = sum_masked / n_masked if n_masked else 0.0
    loss_missing = sum_missing / n_missing if n_missing else 0.0
    
    out = {
        "loss_observed": loss_observed,
        "loss_masked": loss_masked,
        "loss_missing": loss_missing,
        "n_observed": n_observed,
        "n_masked": n_masked,
        "n_missing": n_missing,
        # For training: masked and observed losses as separate tensors (with grad).
        "trainable_masked_loss": weighted_masked_sum / n_masked if n_masked else torch.zeros((), device=device),
        "trainable_observed_loss": weighted_observed_sum / n_observed if n_observed else torch.zeros((), device=device),
    }
    out["per_type"] = per_type
    return out

@dataclass
class EntityEvalResults:
    """
    Minimal evaluation results for Entity Marformer on a single split.
    
    Only carries:
      - `split`: name of the evaluated split (e.g., "train", "test").
      - `metrics`: nested dict with per-(status, type_name) metrics.
    
    metrics[status][type_name] = {
        "xent": <cross-entropy per token for this (status,type)>,
        "n":    <number of tokens in this bucket>,
        # Optional fields like "acc" can be added by evaluation helpers.
    }
    """
    split: str
    metrics: Dict[str, Dict[str, Dict[str, float]]]


# Used in train.py
def compute_trainable_loss(
    params: torch.Tensor,
    graph: Any,
    types: Dict[str, Any],
    global_param_dim: int,
    device: torch.device,
    masked_loss_weight: float = 1.0,
    observed_loss_weight: float = 0.0,
) -> tuple:
    """
    Compute the scalar trainable loss as a weighted combination of masked and observed losses.

    total = masked_loss_weight * masked_loss + observed_loss_weight * observed_loss

    Returns:
        (loss_tensor, raw_masked_ce, raw_observed_ce)
        - loss_tensor: weighted objective (has grad, used for backprop)
        - raw_masked_ce: unweighted mean CE on masked tokens (float, for logging)
        - raw_observed_ce: unweighted mean CE on observed tokens (float, for logging)
    """
    out = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim, device)
    loss = masked_loss_weight * out["trainable_masked_loss"]
    if observed_loss_weight > 0:
        loss = loss + observed_loss_weight * out["trainable_observed_loss"]
    return loss, out["loss_masked"], out["loss_observed"]


def evaluate_entity_marformer_split(
    model: torch.nn.Module,
    split: str,
    variables: List[RankingData],
    types: Dict[str, EntityType],
    global_param_dim: int,
    device: torch.device,
) -> EntityEvalResults:
    """
    Evaluate Entity Marformer on a list of RankingData variables.
    Focus on rating tokens with status=0 (missing) for accuracy and xent.
    """
    if not variables:
        raise ValueError("No variables to evaluate on")

    # Build graph and run model to get parameter stream.
    graph = variable_list_to_entity_graph(variables, types)
    params = model(graph, device=device)  # [1, L, P]

    # Aggregate loss breakdown over observed/masked/missing (with per-type details).
    loss_info = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim, device)

    # Rating-only accuracy and cross-entropy on missing tokens.
    rating_type = types["rating"]
    num_classes = rating_type.num_classes

    ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    logits_list: List[torch.Tensor] = []
    targets_list: List[int] = []

    correct = 0
    total = 0

    # Variable tokens are first in the graph, in the same order as `variables`.
    for idx, var in enumerate(variables):
        tok = graph.tokens[idx]
        if tok.type_name != "rating" or tok.status != 0:
            continue

        raw = tok.raw_data or {}
        rating_value = raw.get("rating_value", None)
        if rating_value is None:
            continue

        # Slice out logits for classes 0..C-1 (skip the first mask bit dimension).
        logits = params[0, idx, 1 : 1 + num_classes]
        logits_list.append(logits)
        targets_list.append(int(rating_value))

        pred_class = int(torch.argmax(logits).item())
        if pred_class == int(rating_value):
            correct += 1
        total += 1

    # If there are no missing rating tokens, we still return the per-type xent
    # counts, but do not add any accuracy fields.
    if total == 0 or not logits_list:
        return EntityEvalResults(
            split=split,
            metrics=loss_info["per_type"],
        )

    logits_tensor = torch.stack(logits_list, dim=0)  # [N, C]
    targets_tensor = torch.tensor(targets_list, device=device, dtype=torch.long)  # [N]
    xent = float(ce_loss(logits_tensor, targets_tensor).item())
    acc = float(correct) / float(total)
    
    # Attach accuracy and CE for rating-on-missing into the per-type metrics tree.
    per_type_metrics = loss_info["per_type"]
    rating_missing = per_type_metrics.setdefault("missing", {}).setdefault("rating", {})
    rating_missing["acc"] = acc
    # xent in per_type_metrics["missing"]["rating"]["xent"] is already the CE
    # from the LossBreakdown; we leave it as-is for consistency.

    return EntityEvalResults(
        split=split,
        metrics=per_type_metrics,
    )

