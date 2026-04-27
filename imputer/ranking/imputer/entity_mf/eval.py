from __future__ import annotations

from dataclasses import dataclass, field
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
    # Per-(status, type_name) metrics: metrics[status][type_name] = {"nll": ..., "n": ...}
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
                "nll": float(b.loss_observed),
                "n": float(b.n_observed),
            }
        if b.n_masked > 0:
            sum_masked += b.loss_masked * b.n_masked
            n_masked += b.n_masked
            per_type["masked"][type_name] = {
                "nll": float(b.loss_masked),
                "n": float(b.n_masked),
            }
        if b.n_missing > 0:
            sum_missing += b.loss_missing * b.n_missing
            n_missing += b.n_missing
            per_type["missing"][type_name] = {
                "nll": float(b.loss_missing),
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
    Evaluation results for Entity Marformer on a single split.

    metrics[status][type_name] = {
        "nll": <gaussian NLL per token>,
        "n":   <token count>,
    }

    missing_preds / missing_true  : scalar mean predictions and true scores for missing rating tokens.
    observed_preds / observed_true: same for observed rating tokens.
    """
    split: str
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    missing_preds: List[float] = field(default_factory=list)
    missing_true:  List[float] = field(default_factory=list)
    observed_preds: List[float] = field(default_factory=list)
    observed_true:  List[float] = field(default_factory=list)


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
        - raw_masked_ce: unweighted mean NLL on masked tokens (float, for logging)
        - raw_observed_ce: unweighted mean NLL on observed tokens (float, for logging)
    """
    out = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim, device)
    loss = masked_loss_weight * out["trainable_masked_loss"]
    if observed_loss_weight > 0:
        loss = loss + observed_loss_weight * out["trainable_observed_loss"]
    return loss, out["loss_masked"], out["loss_observed"]


def _merge_chunk_results(split: str, chunk_results: List["EntityEvalResults"]) -> "EntityEvalResults":
    """Aggregate per-chunk EntityEvalResults into a single result (weighted by n)."""
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for result in chunk_results:
        for status, type_dict in result.metrics.items():
            if status not in agg:
                agg[status] = {}
            for type_name, vals in type_dict.items():
                if type_name not in agg[status]:
                    agg[status][type_name] = {"n": 0.0}
                n = vals.get("n", 0.0)
                agg[status][type_name]["n"] += n
                for k, v in vals.items():
                    if k == "n":
                        continue
                    numer_key = f"{k}_numer"
                    agg[status][type_name][numer_key] = agg[status][type_name].get(numer_key, 0.0) + v * n
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for status, type_dict in agg.items():
        merged[status] = {}
        for type_name, sums in type_dict.items():
            n = sums["n"]
            entry: Dict[str, float] = {"n": n}
            for k, v in sums.items():
                if k == "n" or not k.endswith("_numer"):
                    continue
                metric_name = k[:-6]
                entry[metric_name] = v / n if n > 0 else 0.0
            merged[status][type_name] = entry
    m_preds: List[float] = []
    m_true:  List[float] = []
    o_preds: List[float] = []
    o_true:  List[float] = []
    for result in chunk_results:
        m_preds.extend(result.missing_preds)
        m_true.extend(result.missing_true)
        o_preds.extend(result.observed_preds)
        o_true.extend(result.observed_true)
    return EntityEvalResults(
        split=split,
        metrics=merged,
        missing_preds=m_preds,
        missing_true=m_true,
        observed_preds=o_preds,
        observed_true=o_true,
    )


def evaluate_entity_marformer_split(
    model: torch.nn.Module,
    split: str,
    variables: List[RankingData],
    types: Dict[str, EntityType],
    global_param_dim: int,
    device: torch.device,
    max_item: int | None = None,
) -> EntityEvalResults:
    """
    Evaluate Entity Marformer on a list of RankingData variables.
    Focus on rating tokens with status=0 (missing) for scalar prediction metrics.

    If max_item is set and the variable list spans more items than max_item,
    evaluation is run in item-chunks (matching the training graph size) and
    results are aggregated with n-weighted averaging.
    """
    if not variables:
        raise ValueError("No variables to evaluate on")

    # Chunked evaluation path: mirrors training chunking so graph sizes match.
    if max_item is not None:
        all_item_ids = sorted({iid for v in variables for iid in v.item_ids})
        if len(all_item_ids) > max_item:
            item_chunks = [
                set(all_item_ids[i : i + max_item])
                for i in range(0, len(all_item_ids), max_item)
            ]
            chunk_results: List[EntityEvalResults] = []
            for item_set in item_chunks:
                chunk_vars = [v for v in variables if all(iid in item_set for iid in v.item_ids)]
                if not chunk_vars:
                    continue
                chunk_results.append(
                    evaluate_entity_marformer_split(
                        model, split, chunk_vars, types, global_param_dim, device, max_item=None
                    )
                )
            if not chunk_results:
                return EntityEvalResults(split=split, metrics={})
            return _merge_chunk_results(split, chunk_results)

    # Build graph and run model to get parameter stream.
    graph = variable_list_to_entity_graph(variables, types)
    params = model(graph, device=device)  # [1, L, P]

    # Aggregate loss breakdown over observed/masked/missing (with per-type details).
    loss_info = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim, device)

    # Rating metrics and scalar predictions for missing + observed.
    rating_type = types["rating"]

    missing_preds: List[float] = []
    missing_true:  List[float] = []
    observed_preds: List[float] = []
    observed_true:  List[float] = []

    # Variable tokens are first in the graph, in the same order as `variables`.
    for idx, var in enumerate(variables):
        tok = graph.tokens[idx]
        if tok.type_name != "rating" or tok.status not in (0, 2):
            continue

        raw = tok.raw_data or {}
        rating_value = raw.get("rating_value", None)
        if rating_value is None:
            continue

        pred_mu = float(params[0, idx, 1].item())
        true_y = float(rating_value)

        if tok.status == 0:  # missing
            missing_preds.append(pred_mu)
            missing_true.append(true_y)
        else:  # status == 2, observed
            observed_preds.append(pred_mu)
            observed_true.append(true_y)

    per_type_metrics = loss_info["per_type"]
    if missing_true:
        miss_pred_t = torch.tensor(missing_preds, device=device, dtype=torch.float32)
        miss_true_t = torch.tensor(missing_true, device=device, dtype=torch.float32)
        per_type_metrics.setdefault("missing", {}).setdefault("rating", {})["mse"] = float(
            torch.mean((miss_pred_t - miss_true_t).pow(2)).item()
        )
        per_type_metrics["missing"]["rating"]["mae"] = float(
            torch.mean(torch.abs(miss_pred_t - miss_true_t)).item()
        )
    if observed_true:
        obs_pred_t = torch.tensor(observed_preds, device=device, dtype=torch.float32)
        obs_true_t = torch.tensor(observed_true, device=device, dtype=torch.float32)
        per_type_metrics.setdefault("observed", {}).setdefault("rating", {})["mse"] = float(
            torch.mean((obs_pred_t - obs_true_t).pow(2)).item()
        )
        per_type_metrics["observed"]["rating"]["mae"] = float(
            torch.mean(torch.abs(obs_pred_t - obs_true_t)).item()
        )

    return EntityEvalResults(
        split=split,
        metrics=per_type_metrics,
        missing_preds=missing_preds,
        missing_true=missing_true,
        observed_preds=observed_preds,
        observed_true=observed_true,
    )

