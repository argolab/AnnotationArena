from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from imputer.data import RankingData

from .data import variable_list_to_entity_graph
from .fastgraph import GraphTensors, TypeSlice, graph_tensors
from .types import LossBreakdown, EntityType


def _rating_breakdown_fast(
    params: torch.Tensor,
    status: torch.Tensor,
    sl: TypeSlice,
    t: Any,
) -> LossBreakdown:
    """
    Vectorised replacement for RatingVariableType.compute_loss_breakdown.

    The reference implementation loops over every token of the type to collect
    targets and statuses, allocating a tensor per token, and then makes three
    separate `.item()` calls. Targets are fixed by the graph, so GraphTensors
    already has them; status comes from one gathered tensor; and the six scalars
    we need are read back in a single device->host transfer.
    """
    device = params.device
    C = t.num_classes
    B = params.shape[0]
    logits = params.index_select(1, sl.positions)[..., 1 : 1 + C]       # [B, M, C]

    if sl.soft_targets is not None:
        losses = -(sl.soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        losses = F.cross_entropy(
            logits.reshape(-1, C), sl.targets.repeat(B), reduction="none"
        ).view(B, -1)                                                   # [B, M]

    status = status.index_select(1, sl.positions)                       # [B, M]
    is_masked   = (status == 1).to(losses.dtype)
    is_observed = (status == 2).to(losses.dtype)
    is_missing  = (status == 0).to(losses.dtype)

    masked_sum, observed_sum = (losses * is_masked).sum(), (losses * is_observed).sum()
    missing_sum = (losses * is_missing).sum()
    n_masked_t, n_observed_t, n_missing_t = is_masked.sum(), is_observed.sum(), is_missing.sum()

    # One sync for everything the caller needs as a Python float.
    n_masked, n_observed, n_missing, missing_f = torch.stack(
        [n_masked_t, n_observed_t, n_missing_t, missing_sum]
    ).tolist()
    n_masked, n_observed, n_missing = int(n_masked), int(n_observed), int(n_missing)

    z = torch.zeros((), device=device)
    trainable_loss = masked_sum / n_masked_t if n_masked else z
    observed_loss  = observed_sum / n_observed_t if n_observed else z
    # These two are only read as floats, and both numerators are already on the
    # device; detach so no graph is kept for the reporting path.
    loss_masked_f, loss_observed_f = torch.stack(
        [trainable_loss.detach(), observed_loss.detach()]
    ).tolist()

    return LossBreakdown(
        trainable_loss=trainable_loss,
        observed_loss_tensor=observed_loss,
        loss_observed=loss_observed_f if n_observed else 0.0,
        loss_masked=loss_masked_f if n_masked else 0.0,
        loss_missing=missing_f / n_missing if n_missing else 0.0,
        n_observed=n_observed,
        n_masked=n_masked,
        n_missing=n_missing,
    )


def _aggregate_loss_from_breakdowns(
    params: torch.Tensor,
    graph: Any,
    types: Dict[str, Any],
    global_param_dim: int,
    device: torch.device,
    status: torch.Tensor | None = None,
) -> dict:
    """
    Aggregate per-type LossBreakdown into global observed/masked/missing losses
    and counts, and optionally a per-(status, type_name) breakdown.

    Entity types contribute nothing but used to cost a full L-length Python list
    comprehension each, to build a type mask that GraphTensors now caches.
    """
    gt = graph_tensors(graph, types, global_param_dim, device)
    # [B, L] statuses. The unbatched path reads them off the graph's tokens;
    # the batched training path passes one row per mask augmentation.
    status = gt.status.unsqueeze(0) if status is None else status
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
        if not getattr(t, "is_variable", False):
            continue                      # entity types return a zero breakdown
        sl = gt.slices[type_name]
        if sl.positions.numel() == 0:
            continue

        if sl.vectorised and getattr(t, "num_classes", None) is not None:
            b: LossBreakdown = _rating_breakdown_fast(params, status, sl, t)
        else:
            if params.shape[0] != 1:
                raise NotImplementedError(
                    f"type {type_name!r} has no vectorised loss, so it cannot run "
                    "in the batched-augmentation path"
                )
            b = t.compute_loss_breakdown(
                predicted_params=params,
                tokens=graph.tokens,
                type_mask=sl.mask.unsqueeze(0),
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
    Evaluation results for Entity Marformer on a single split.

    metrics[status][type_name] = {
        "xent": <cross-entropy per token>,
        "n":    <token count>,
        "acc":  <accuracy, rating missing only>,
    }

    missing_preds / missing_true  : expected-value predictions and true labels
                                    (0-indexed) for missing rating tokens.
    observed_preds / observed_true: same for observed rating tokens.
    """
    split: str
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    missing_preds: List[float] = field(default_factory=list)
    missing_true:  List[int]   = field(default_factory=list)
    observed_preds: List[float] = field(default_factory=list)
    observed_true:  List[int]   = field(default_factory=list)


# Used in train.py
def compute_trainable_loss(
    params: torch.Tensor,
    graph: Any,
    types: Dict[str, Any],
    global_param_dim: int,
    device: torch.device,
    masked_loss_weight: float = 1.0,
    observed_loss_weight: float = 0.0,
    status: torch.Tensor | None = None,
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
    out = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim,
                                          device, status=status)
    loss = masked_loss_weight * out["trainable_masked_loss"]
    if observed_loss_weight > 0:
        loss = loss + observed_loss_weight * out["trainable_observed_loss"]
    return loss, out["loss_masked"], out["loss_observed"], out["n_masked"], out["n_observed"]


def _merge_chunk_results(split: str, chunk_results: List["EntityEvalResults"]) -> "EntityEvalResults":
    """Aggregate per-chunk EntityEvalResults into a single result (weighted by n)."""
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for result in chunk_results:
        for status, type_dict in result.metrics.items():
            if status not in agg:
                agg[status] = {}
            for type_name, vals in type_dict.items():
                if type_name not in agg[status]:
                    agg[status][type_name] = {"xent_numer": 0.0, "n": 0.0}
                n = vals.get("n", 0.0)
                agg[status][type_name]["xent_numer"] += vals.get("xent", 0.0) * n
                agg[status][type_name]["n"] += n
                if "acc" in vals:
                    if "acc_numer" not in agg[status][type_name]:
                        agg[status][type_name]["acc_numer"] = 0.0
                    agg[status][type_name]["acc_numer"] += vals["acc"] * n
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for status, type_dict in agg.items():
        merged[status] = {}
        for type_name, sums in type_dict.items():
            n = sums["n"]
            entry: Dict[str, float] = {
                "xent": sums["xent_numer"] / n if n > 0 else 0.0,
                "n": n,
            }
            if "acc_numer" in sums and n > 0:
                entry["acc"] = sums["acc_numer"] / n
            merged[status][type_name] = entry
    m_preds: List[float] = []
    m_true:  List[int]   = []
    o_preds: List[float] = []
    o_true:  List[int]   = []
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
    cache: Dict[str, Any] | None = None,
) -> EntityEvalResults:
    """
    Evaluate Entity Marformer on a list of RankingData variables.
    Focus on rating tokens with status=0 (missing) for accuracy and xent.

    If max_item is set and the variable list spans more items than max_item,
    evaluation is run in item-chunks (matching the training graph size) and
    results are aggregated with n-weighted averaging.
    """
    if not variables:
        raise ValueError("No variables to evaluate on")

    # Chunked evaluation path: mirrors training chunking so graph sizes match.
    #
    # `cache` (optional, keyed by split) holds the item-chunk partition across
    # calls. The variable list and its statuses are the same every epoch, so
    # without it each evaluation re-derives the chunks, rebuilds every chunk
    # graph, and rebuilds the [L, L, R] edge mask and [L, L, 3] K_aug that are
    # cached *on the graph object* -- and then throws all of it away.
    if max_item is not None:
        entry = cache.get(split) if cache is not None else None
        if entry is None:
            all_item_ids = sorted({iid for v in variables for iid in v.item_ids})
            if len(all_item_ids) > max_item:
                chunks: List[List[RankingData]] = []
                for i in range(0, len(all_item_ids), max_item):
                    item_set = set(all_item_ids[i : i + max_item])
                    chunk_vars = [v for v in variables if all(iid in item_set for iid in v.item_ids)]
                    if chunk_vars:
                        chunks.append(chunk_vars)
                entry = {"chunks": chunks}
            else:
                entry = {"chunks": None}          # single graph, no chunking
            if cache is not None:
                cache[split] = entry

        if entry["chunks"] is not None:
            chunk_results: List[EntityEvalResults] = []
            for ci, chunk_vars in enumerate(entry["chunks"]):
                chunk_results.append(
                    evaluate_entity_marformer_split(
                        model, split, chunk_vars, types, global_param_dim, device,
                        max_item=None,
                        cache=entry.setdefault("sub", {}).setdefault(ci, {}),
                    )
                )
            if not chunk_results:
                return EntityEvalResults(split=split, metrics={})
            return _merge_chunk_results(split, chunk_results)

        # Unchunked: keep this split's graph under its own entry, not under the
        # shared cache dict, which several splits pass in.
        cache = entry

    # Build graph and run model to get parameter stream.
    graph = cache.get("graph") if cache is not None else None
    if graph is None:
        graph = variable_list_to_entity_graph(variables, types)
        if cache is not None:
            cache["graph"] = graph
    params = model(graph, device=device)  # [1, L, P]

    # Aggregate loss breakdown over observed/masked/missing (with per-type details).
    loss_info = _aggregate_loss_from_breakdowns(params, graph, types, global_param_dim, device)

    # Rating accuracy, CE, and expected-value predictions for missing + observed.
    # Vectorised: the reference loop called `.item()` twice per rating token,
    # i.e. ~2 device syncs per token, which was most of the evaluation cost.
    gt = graph_tensors(graph, types, global_param_dim, device)
    rating_type = types["rating"]
    num_classes = rating_type.num_classes
    sl = gt.slices["rating"]

    missing_preds: List[float] = []
    missing_true:  List[int]   = []
    observed_preds: List[float] = []
    observed_true:  List[int]   = []
    total = 0
    acc = 0.0

    if sl.positions.numel() and sl.targets is not None:
        logits = params[0].index_select(0, sl.positions)[:, 1 : 1 + num_classes]  # [M, C]
        probs = torch.softmax(logits, dim=-1)
        arange = torch.arange(num_classes, device=device, dtype=probs.dtype)
        expected = probs @ arange                                                 # [M]
        pred = logits.argmax(dim=-1)                                              # [M]
        targets = sl.targets
        status = gt.status.index_select(0, sl.positions)
        is_missing = status == 0
        is_observed = status == 2

        n_missing_t = is_missing.sum()
        correct_t = ((pred == targets) & is_missing).sum()
        total, correct = (int(v) for v in torch.stack([n_missing_t, correct_t]).tolist())

        missing_preds = expected[is_missing].tolist()
        missing_true = targets[is_missing].tolist()
        observed_preds = expected[is_observed].tolist()
        observed_true = targets[is_observed].tolist()
        if total:
            acc = float(correct) / float(total)

    # If there are no missing rating tokens, return without accuracy fields.
    if total == 0:
        return EntityEvalResults(
            split=split,
            metrics=loss_info["per_type"],
            missing_preds=missing_preds,
            missing_true=missing_true,
            observed_preds=observed_preds,
            observed_true=observed_true,
        )

    per_type_metrics = loss_info["per_type"]
    per_type_metrics.setdefault("missing", {}).setdefault("rating", {})["acc"] = acc

    return EntityEvalResults(
        split=split,
        metrics=per_type_metrics,
        missing_preds=missing_preds,
        missing_true=missing_true,
        observed_preds=observed_preds,
        observed_true=observed_true,
    )
