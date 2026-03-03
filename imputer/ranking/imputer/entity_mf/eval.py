from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .types import LossBreakdown


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

