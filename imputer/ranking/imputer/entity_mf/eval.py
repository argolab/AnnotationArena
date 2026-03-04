from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from imputer.data import RankingData
from stan.pipeline.bundle import GroundTruthBundle

from .data import bundle_to_entity_graph
from .types import LossBreakdown, EntityType


@dataclass
class LossStat:
    """Aggregate loss stat across all variable types.

    trainable_loss is the mean loss over all masked variables (used for backprop).
    The *_loss fields are means per token by status for logging / evaluation.
    """

    trainable_loss: torch.Tensor
    loss_observed: float
    loss_masked: float
    loss_missing: float
    n_observed: int
    n_masked: int
    n_missing: int


def compute_loss_stat(
    params: torch.Tensor,
    graph: Any,
    types: Dict[str, Any],
    global_param_dim: int,
    device: torch.device,
) -> LossStat:
    """
    Compute aggregate loss stat from per-type breakdowns. No forward pass here:
    caller already has `params = model(graph)`.

    trainable_loss = mean over all masked variables (same count-weighted mean as loss_masked);
    observed/missing losses are for metrics only.
    """
    weighted_masked_sum = torch.zeros((), device=device)
    sum_observed = 0.0
    sum_masked = 0.0
    sum_missing = 0.0
    n_observed = 0
    n_masked = 0
    n_missing = 0

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
            sum_observed += b.loss_observed * b.n_observed
            n_observed += b.n_observed
        if b.n_masked > 0:
            sum_masked += b.loss_masked * b.n_masked
            n_masked += b.n_masked
        if b.n_missing > 0:
            sum_missing += b.loss_missing * b.n_missing
            n_missing += b.n_missing

    trainable_loss = weighted_masked_sum / n_masked if n_masked else torch.zeros((), device=device)
    loss_observed = sum_observed / n_observed if n_observed else 0.0
    loss_masked = sum_masked / n_masked if n_masked else 0.0
    loss_missing = sum_missing / n_missing if n_missing else 0.0

    return LossStat(
        trainable_loss=trainable_loss,
        loss_observed=loss_observed,
        loss_masked=loss_masked,
        loss_missing=loss_missing,
        n_observed=n_observed,
        n_masked=n_masked,
        n_missing=n_missing,
    )


@dataclass
class EntityEvalResults:
    """Lightweight evaluation results for Entity Marformer on a single split."""

    missing_accuracy: float
    missing_xent: float
    n_missing_ratings: int


def evaluate_entity_marformer_split(
    model: torch.nn.Module,
    bundle: GroundTruthBundle,
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
        return EntityEvalResults(missing_accuracy=0.0, missing_xent=0.0, n_missing_ratings=0)

    # Build graph and run model to get parameter stream.
    graph = bundle_to_entity_graph(bundle, variables, types)
    params = model(graph, device=device)  # [1, L, P]

    # Compute aggregate loss stat (for potential future use).
    _ = compute_loss_stat(params, graph, types, global_param_dim, device)

    # Rating-only accuracy and cross-entropy on missing tokens.
    rating_type = types.get("rating", None)
    if rating_type is None:
        return EntityEvalResults(missing_accuracy=0.0, missing_xent=0.0, n_missing_ratings=0)

    num_classes = getattr(rating_type, "num_classes", None)
    if num_classes is None:
        # Fallback: infer from params dimension.
        num_classes = global_param_dim - 1

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

    if total == 0 or not logits_list:
        return EntityEvalResults(missing_accuracy=0.0, missing_xent=0.0, n_missing_ratings=0)

    logits_tensor = torch.stack(logits_list, dim=0)  # [N, C]
    targets_tensor = torch.tensor(targets_list, device=device, dtype=torch.long)  # [N]
    xent = float(ce_loss(logits_tensor, targets_tensor).item())
    acc = float(correct) / float(total)

    return EntityEvalResults(missing_accuracy=acc, missing_xent=xent, n_missing_ratings=total)

