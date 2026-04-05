#!/usr/bin/env python3
"""
Pointer attention diagnostics for a trained EntityMarformer.

Data splits match imputer/data.py / EntityMarformerLightningModule: only ``train`` and
``test`` exist in the bundle (``instance`` field). There is no separate validation split;
training reports metrics on the ``test`` holdout. Default for this script is ``test``,
same as evaluate_entity_marformer_split(..., split="test") in train.py.

Uses **rating-only** variables (``not is_listwise``); pairwise rankings are excluded.
Optional ``--max-item`` chunking matches training/eval (first chunk by default; set ``--chunk-index`` for others).

Runs the model on an EntityGraph for the chosen partition and produces:
  - scatter_layer{L}.png       : Exp 1  — mass_rel vs mass_ptr per relation (ATTR/ANNOT/ITEM)
  - entity_inv_hist_layer{L}.png : Exp 2 — inverse-edge mass histograms for item/annotator/attribute
  - logit_topk_layer{L}.png    : Exp 3a — centered top-K grouped bar chart for a few queries
  - logit_hist_layer{L}.png    : Exp 3b — key-axis-centered softmax-norm histograms
  - qscale_layer{L}.png        : Exp 3c — Q_rel vs Q_ptr scatter (y=x: equal query-side logits; above → ptr larger)
  - all_layers/*.png           : same five plot kinds, all layers stacked as rows in one figure each
  - attention_debug.npz        : raw arrays for offline analysis

Model architecture and training knobs (``use_graph_mask``, ``use_pointer``, ``max_item``, reg weights, etc.)
are read from ``<run-dir>/train_config.json`` — same source as training. You should not need extra flags for
graph mask vs soft attention.

``data.data_dir`` in that json may point at OUTPUT/...; if missing locally, resolution also tries
``<imputer/ranking>/OLD_DATA/<leaf>/``. Use ``--data-dir`` only when that automatic resolution fails.

For an **interactive** layer/token attention viewer (full ``attn_mean``, fan chart, edge/pointer filters),
run ``scripts/export_attention_explorer_data.py`` and serve the output directory over HTTP; the page’s
“See also” panel points here for **three-logit** PNG/NPZ from this script.

Usage:
    cd imputer/ranking
    PYTHONPATH=. python scripts/analyze_pointer_attention.py \\
        --run-dir RESULTS/MARFORMER/my_run \\
        --partition test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import (
    variable_list_to_entity_graph,
    EDGE_REL_ATTR, EDGE_REL_ATTR_INV,
    EDGE_REL_ANNOT, EDGE_REL_ANNOT_INV,
    EDGE_REL_ITEM, EDGE_REL_ITEM_INV,
)

# imputer/ranking (directory that contains scripts/ and package imputer/)
RANKING_ROOT = Path(__file__).resolve().parents[1]

# train helpers are imported inside load_model() so importing this script does not
# pull pytorch_lightning → torchmetrics → scipy/transformers (slow on NFS).


def normalize_cli_path(p: str | Path) -> Path:
    """
    Resolve a CLI path relative to cwd.

    If the user is already in ``imputer/ranking`` but passes ``imputer/ranking/RESULTS/...``,
    that would wrongly nest; strip the redundant prefix and anchor under RANKING_ROOT.
    """
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    s = path.as_posix().lstrip("./")
    if s.startswith("imputer/ranking/"):
        return (RANKING_ROOT / s[len("imputer/ranking/") :]).resolve()
    return (Path.cwd() / path).resolve()


def resolve_run_dir(run_dir: Path) -> Path:
    """Find a directory that contains ``train_config.json`` (handles common path mistakes)."""
    run_dir = Path(run_dir).expanduser()
    tried: list[Path] = []

    def _check(p: Path) -> Path | None:
        r = p.resolve()
        if r in tried:
            return None
        tried.append(r)
        return r if (r / "train_config.json").is_file() else None

    candidates: list[Path] = []
    if run_dir.is_absolute():
        candidates.append(run_dir)
    else:
        candidates.append(Path.cwd() / run_dir)
        candidates.append(RANKING_ROOT / run_dir)
        s = run_dir.as_posix().lstrip("./")
        if s.startswith("imputer/ranking/"):
            rel = Path(s[len("imputer/ranking/") :])
            candidates.append(Path.cwd() / rel)
            candidates.append(RANKING_ROOT / rel)

    for c in candidates:
        hit = _check(c)
        if hit is not None:
            return hit

    fallback = (Path.cwd() / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    raise FileNotFoundError(
        "Could not find train_config.json for --run-dir. Tried:\n  "
        + "\n  ".join(str(t) for t in tried)
        + "\nHint: from imputer/ranking use --run-dir RESULTS/... (not imputer/ranking/RESULTS/...)."
    )


# ── Relation / channel metadata ────────────────────────────────────────────────
# Graphs are built from rating-only variables; all variable tokens are type "rating".
VARIABLE_TYPES = {"rating"}
ENTITY_TYPES   = {"item", "annotator", "attribute"}

# Exp 1: (subplot_label, mass_rel_idx, mass_ptr_channel)
EXP1_CHANNELS = [
    ("ATTR",  EDGE_REL_ATTR,  0),
    ("ANNOT", EDGE_REL_ANNOT, 1),
    ("ITEM",  EDGE_REL_ITEM,  2),
]

# Exp 2: (subplot_label, token_type, mass_rel_idx)
EXP2_ENTITY_INV = [
    ("item",       "item",       EDGE_REL_ITEM_INV),
    ("annotator",  "annotator",  EDGE_REL_ANNOT_INV),
    ("attribute",  "attribute",  EDGE_REL_ATTR_INV),
]


# ── Model loading ──────────────────────────────────────────────────────────────

def _resolve_data_dir(run_dir: Path, configured_data_dir: str) -> Path:
    """
    Resolve data_dir from train_config robustly across cwd/run_dir layouts.

    We prefer the first candidate that contains data_bundle.json.

    If OUTPUT/... paths from an old machine are absent, we also try
    ``RANKING_ROOT / OLD_DATA / <basename(configured_data_dir)>`` (e.g. OLD_DATA/llm_rubric_dist).
    """
    raw = Path(configured_data_dir)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Most common/desired interpretation: relative to current working dir.
        candidates.append(raw)
        # Also try relative to run_dir and its ancestors.
        candidates.append(run_dir / raw)
        candidates.extend(parent / raw for parent in run_dir.parents)

    leaf = raw.name
    if leaf:
        candidates.append(RANKING_ROOT / "OLD_DATA" / leaf)

    seen = set()
    unique_candidates: List[Path] = []
    for cand in candidates:
        r = cand.resolve()
        if r in seen:
            continue
        seen.add(r)
        unique_candidates.append(r)

    for cand in unique_candidates:
        if (cand / "data_bundle.json").exists():
            return cand

    raise FileNotFoundError(
        "Unable to resolve data_dir from train_config. Checked: "
        + ", ".join(str(c) for c in unique_candidates)
        + ". Pass --data-dir explicitly (directory containing data_bundle.json)."
    )


def load_model(run_dir: Path, device: torch.device, data_dir: Path | None = None):
    from imputer.entity_mf.train import load_bundle_and_converter, build_entity_marformer_from_bundle

    cfg_path = run_dir / "train_config.json"
    with open(cfg_path) as f:
        tc = json.load(f)

    if data_dir is not None:
        resolved = Path(data_dir).resolve()
        if not (resolved / "data_bundle.json").exists():
            raise FileNotFoundError(f"No data_bundle.json under --data-dir: {resolved}")
    else:
        resolved = _resolve_data_dir(run_dir, tc["data"]["data_dir"])
    data_dir = resolved

    print(f"Data dir  : {data_dir}")
    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    mc = tc["model"]
    # Keep in sync with imputer.entity_mf.config.EntityMarformerConfig (train.py main() + saved json).
    config = EntityMarformerConfig(
        embedding_dim=mc["embedding_dim"],
        num_layers=mc["num_layers"],
        attention_heads=mc["attention_heads"],
        dropout=mc.get("dropout", 0.1),
        d_ff=mc.get("d_ff", 128),
        num_ffn_layers=mc.get("num_ffn_layers", 1),
        logit_high=mc.get("logit_high", 20.0),
        temperature=mc.get("temperature", 1.0),
        use_per_head_rel=mc.get("use_per_head_rel", True),
        use_pointer=mc.get("use_pointer", False),
        use_rel_value=mc.get("use_rel_value", False),
        use_addone_attn=mc.get("use_addone_attn", False),
        type_embedding_init=mc.get("type_embedding_init", "normal"),
        use_deviation_norm=mc.get("use_deviation_norm", False),
        scale_shared_rel=mc.get("scale_shared_rel", False),
        use_learned_embedding=mc.get("use_learned_embedding", False),
        use_graph_mask=mc.get("use_graph_mask", False),
    )

    tr = tc.get("training", {})
    model, _ = build_entity_marformer_from_bundle(
        bundle=bundle,
        converter=converter,
        sizes=sizes,
        config=config,
        annotator_reg_weight=tr.get("annotator_reg_weight", 0.0),
        item_reg_weight=tr.get("item_reg_weight", 0.0),
        attribute_reg_weight=tr.get("attribute_reg_weight", 0.0),
        llm_input_dist=tr.get("llm_input_dist", False),
        item_dropout_rate=tr.get("item_dropout_rate", 1.0),
    )

    # Load checkpoint — find latest .ckpt
    ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpts[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    # Strip "model." prefix added by Lightning
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    print(
        "train_config model flags: "
        f"use_graph_mask={config.use_graph_mask} use_pointer={config.use_pointer} "
        f"use_per_head_rel={config.use_per_head_rel}"
    )
    return model, bundle, converter, sizes, tc


def _rating_only(variables: List[Any]) -> List[Any]:
    """Keep ratings; drop listwise / pairwise ranking variables."""
    return [v for v in variables if not v.is_listwise]


def _chunk_variables_by_max_item(variables: List[Any], max_item: int) -> List[List[Any]]:
    """Mirror evaluate_entity_marformer_split chunking: sorted item ids, contiguous chunks of size max_item."""
    if max_item is None or max_item <= 0:
        return [variables]
    all_item_ids = sorted({iid for v in variables for iid in v.item_ids})
    if len(all_item_ids) <= max_item:
        return [variables]
    chunks: List[List[Any]] = []
    for i in range(0, len(all_item_ids), max_item):
        item_set = set(all_item_ids[i : i + max_item])
        chunk_vars = [v for v in variables if all(iid in item_set for iid in v.item_ids)]
        if chunk_vars:
            chunks.append(chunk_vars)
    return chunks


def build_partition_graphs(
    bundle,
    converter,
    sizes,
    tc,
    max_graphs: int,
    partition: str,
    max_item: int | None = None,
    chunk_index: int = 0,
):
    """Build EntityGraph(s) for ``partition`` using **rating-only** variables.

    When ``max_item`` is set, variables are split into item chunks (same logic as
    ``evaluate_entity_marformer_split``); only ``chunk_index`` is built (default 0).
    If multiple chunks exist, prints how many and which index is used.

    Matches DataConverter.create_variables_from_bundle — there is no ``val`` split in the bundle.
    """
    if partition not in ("train", "test"):
        raise ValueError(f"partition must be 'train' or 'test', got {partition!r}")

    from imputer.entity_mf.types import build_default_domain3_types
    tr = tc.get("training", {})
    mc = tc["model"]
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=converter.max_rank_size,
        logit_high=mc.get("logit_high", 20.0),
        annotator_reg_weight=tr.get("annotator_reg_weight", 0.0),
        item_reg_weight=tr.get("item_reg_weight", 0.0),
        attribute_reg_weight=tr.get("attribute_reg_weight", 0.0),
        llm_input_dist=tr.get("llm_input_dist", False),
        item_dropout_rate=tr.get("item_dropout_rate", 1.0),
    )

    obs = converter.create_variables_from_bundle(bundle, partition=partition, status="observed")
    miss = converter.create_variables_from_bundle(bundle, partition=partition, status="missing")
    all_vars = _rating_only(obs + miss)
    n_dropped = len(obs) + len(miss) - len(all_vars)
    if n_dropped:
        print(f"  (Excluded {n_dropped} non-rating / listwise variables.)")

    note = (
        "same holdout as test_eval in train.py."
        if partition == "test"
        else "training fit partition (cf. test_eval in train.py)."
    )
    print(
        f"Partition {partition!r}: {len(all_vars)} rating variables "
        f"(from {len(obs)} observed + {len(miss)} missing raw rows); {note}"
    )

    chunks = _chunk_variables_by_max_item(all_vars, max_item)
    if len(chunks) > 1:
        print(
            f"  max_item={max_item}: {len(chunks)} item chunks; using chunk_index={chunk_index} "
            f"({len(chunks[chunk_index])} variables in this chunk)."
        )
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(f"chunk_index {chunk_index} out of range for {len(chunks)} chunk(s)")

    chunk_vars = chunks[chunk_index]
    if not chunk_vars:
        raise ValueError("No rating variables in selected chunk — try another chunk_index or disable max_item.")

    graph = variable_list_to_entity_graph(chunk_vars, types)
    print(f"Graph: {graph.num_tokens} tokens")
    return [graph]


# ── Token index helpers ────────────────────────────────────────────────────────

def token_indices(graph, type_names: set) -> List[int]:
    return [i for i, t in enumerate(graph.tokens) if t.type_name in type_names]


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_scatter(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path, partition: str):
    """Exp 1: scatter of mass_rel vs mass_ptr for ATTR / ANNOT / ITEM."""
    mass_rel = debug.get("mass_rel")  # [B, L, R] numpy
    mass_ptr = debug.get("mass_ptr")  # [B, L, 3] numpy
    if mass_rel is None or mass_ptr is None or not var_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Layer {layer_idx}: rel-mass vs ptr-mass (rating tokens) [partition={partition}]",
        fontsize=11,
    )
    for ax, (label, rel_idx, ptr_ch) in zip(axes, EXP1_CHANNELS):
        x = mass_rel[0, var_rows, rel_idx]  # [N_var]
        y = mass_ptr[0, var_rows, ptr_ch]   # [N_var]
        ax.scatter(x, y, s=4, alpha=0.4, rasterized=True)
        lim = max(x.max(), y.max()) * 1.05 + 1e-6
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"mass_rel_{label}")
        ax.set_ylabel(f"mass_ptr_{ptr_ch} ({label})")
        ax.set_title(label)
    _save(fig, out_dir / f"scatter_layer{layer_idx}.png")


def plot_entity_inv_hist(layer_idx: int, entity_rows_by_type: dict, debug: dict, out_dir: Path, partition: str):
    """Exp 2: histograms of inverse-edge mass for item / annotator / attribute tokens."""
    mass_rel = debug.get("mass_rel")
    if mass_rel is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Layer {layer_idx}: entity token inverse-edge mass [partition={partition}]",
        fontsize=11,
    )
    for ax, (label, tok_type, rel_idx) in zip(axes, EXP2_ENTITY_INV):
        rows = entity_rows_by_type.get(tok_type, [])
        if not rows:
            ax.set_title(f"{label} (no tokens)")
            continue
        vals = mass_rel[0, rows, rel_idx]  # [N_entity]
        ax.hist(vals, bins=40, range=(0, 1), color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(float(vals.mean()), color="red", lw=1.2, linestyle="--", label=f"mean={vals.mean():.3f}")
        ax.set_xlabel(f"mass_{label.upper()}_INV")
        ax.set_ylabel("count")
        ax.set_title(f"{label} (n={len(rows)})")
        ax.legend(fontsize=8)
    _save(fig, out_dir / f"entity_inv_hist_layer{layer_idx}.png")


def plot_logit_topk(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path,
                    partition: str, topk_queries: int = 3, K: int = 10):
    """Exp 3a: centered top-K grouped bar chart for a few example rating queries."""
    lc = debug.get("logit_content_mean")  # [B, L, L]
    lr = debug.get("logit_rel_mean")      # [B, L, L]
    lp = debug.get("logit_ptr")           # [B, L, L]
    if lc is None or lr is None or not var_rows:
        return

    n_q = min(topk_queries, len(var_rows))
    # Pick queries with the highest max attention mass as interesting examples
    mass_rel = debug.get("mass_rel")
    if mass_rel is not None:
        total_mass = mass_rel[0, var_rows, :].max(axis=-1)  # proxy for "active" queries
        top_q_local = np.argsort(total_mass)[::-1][:n_q]
    else:
        top_q_local = np.arange(n_q)
    query_indices = [var_rows[i] for i in top_q_local]

    fig, axes = plt.subplots(1, n_q, figsize=(5 * n_q, 5), squeeze=False)
    fig.suptitle(
        f"Layer {layer_idx}: top-{K} key logit breakdown (centered) [partition={partition}]",
        fontsize=11,
    )

    for col, qi in enumerate(query_indices):
        ax = axes[0][col]
        # Determine top-K keys by attention mass (use logit sum as proxy for attention)
        total_logit = lc[0, qi, :] + lr[0, qi, :]
        if lp is not None:
            total_logit = total_logit + lp[0, qi, :]
        topk_keys = np.argsort(total_logit)[::-1][:K]

        c_vals = lc[0, qi, topk_keys]  # [K]
        r_vals = lr[0, qi, topk_keys]  # [K]
        p_vals = lp[0, qi, topk_keys] if lp is not None else np.zeros(K)

        # Center each component over the top-K keys (key-axis centering)
        c_vals = c_vals - c_vals.mean()
        r_vals = r_vals - r_vals.mean()
        p_vals = p_vals - p_vals.mean()

        x = np.arange(K)
        w = 0.25
        ax.bar(x - w, c_vals, w, label="content", color="#4c72b0")
        ax.bar(x,     r_vals, w, label="rel",     color="#dd8452")
        ax.bar(x + w, p_vals, w, label="ptr",     color="#55a868")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"k{j}" for j in topk_keys], rotation=45, fontsize=7)
        ax.set_xlabel("key token"); ax.set_ylabel("centered logit")
        ax.set_title(f"query tok {qi}")
        if col == 0:
            ax.legend(fontsize=8)

    _save(fig, out_dir / f"logit_topk_layer{layer_idx}.png")


def plot_logit_hist(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path, partition: str, K: int = 10):
    """Exp 3b: key-axis-centered softmax-normalized histograms over all rating queries."""
    lc = debug.get("logit_content_mean")
    lr = debug.get("logit_rel_mean")
    lp = debug.get("logit_ptr")
    if lc is None or lr is None or not var_rows:
        return

    w_c_all, w_r_all, w_p_all = [], [], []

    for qi in var_rows:
        # Top-K keys for this query
        total = lc[0, qi, :] + lr[0, qi, :]
        if lp is not None:
            total = total + lp[0, qi, :]
        topk = np.argsort(total)[::-1][:K]

        c = lc[0, qi, topk]
        r = lr[0, qi, topk]
        p = lp[0, qi, topk] if lp is not None else np.zeros(K)

        # Key-axis centering: remove per-query mean of each component over top-K keys
        c = c - c.mean()
        r = r - r.mean()
        p = p - p.mean()

        # 3-way softmax over the centered triple for each (i,j)
        stack = np.stack([c, r, p], axis=-1)  # [K, 3]
        stack = stack - stack.max(axis=-1, keepdims=True)  # numerical stability
        exp_s = np.exp(stack)
        w = exp_s / exp_s.sum(axis=-1, keepdims=True)  # [K, 3]

        w_c_all.append(w[:, 0])
        w_r_all.append(w[:, 1])
        w_p_all.append(w[:, 2])

    w_c = np.concatenate(w_c_all)
    w_r = np.concatenate(w_r_all)
    w_p = np.concatenate(w_p_all)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Layer {layer_idx}: key-axis-centered softmax logit shares (top-{K} keys per query) "
        f"[partition={partition}]",
        fontsize=11,
    )
    for ax, vals, label, color in zip(
        axes,
        [w_c, w_r, w_p],
        ["content (w_c)", "relational (w_r)", "pointer (w_p)"],
        ["#4c72b0", "#dd8452", "#55a868"],
    ):
        ax.hist(vals, bins=50, range=(0, 1), color=color, edgecolor="none", alpha=0.8)
        ax.axvline(float(vals.mean()), color="red", lw=1.2, linestyle="--",
                   label=f"mean={vals.mean():.3f}")
        ax.set_xlabel("softmax weight w"); ax.set_ylabel("count")
        ax.set_title(label); ax.legend(fontsize=8)

    _save(fig, out_dir / f"logit_hist_layer{layer_idx}.png")


def _qscale_square_lim(rel_vals: np.ndarray, ptr_vals: np.ndarray) -> tuple[float, float]:
    """Shared square axis limits so y=x is a 45° visual reference (equal aspect)."""
    lo = min(float(np.min(rel_vals)), float(np.min(ptr_vals)))
    hi = max(float(np.max(rel_vals)), float(np.max(ptr_vals)))
    pad = max((hi - lo) * 0.06 + 1e-9, 0.15)
    return lo - pad, hi + pad


def _draw_qscale_scatter_panel(
    ax,
    rel_vals: np.ndarray,
    ptr_vals: np.ndarray,
    title: str,
    *,
    xlabel: str | None = r"$Q_{\mathrm{rel}}$ (shared)",
    ylabel: str | None = r"$Q_{\mathrm{ptr}}$",
) -> None:
    ax.scatter(
        rel_vals, ptr_vals, s=12, alpha=0.35, c="#2c5282", edgecolors="none", rasterized=True,
    )
    lo2, hi2 = _qscale_square_lim(rel_vals, ptr_vals)
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.85, alpha=0.55)
    ax.axhline(0, color="gray", lw=0.45, linestyle=":", alpha=0.65)
    ax.axvline(0, color="gray", lw=0.45, linestyle=":", alpha=0.65)
    if title:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_qscale(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path, partition: str):
    """Exp 3c: Q_rel vs Q_ptr scatter per matched channel (above y=x → larger pointer query logit)."""
    Q_ptr = debug.get("Q_ptr")          # [B, L, 3]
    Q_rel = debug.get("Q_rel_shared")   # [B, L, R]
    if Q_ptr is None or Q_rel is None or not var_rows:
        return

    labels = [("ATTR", 0, 0), ("ANNOT", 1, 2), ("ITEM", 2, 4)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Layer {layer_idx}: $Q_{{\\mathrm{{rel}}}}$ vs $Q_{{\\mathrm{{ptr}}}}$ (above diag → ptr > rel) "
        f"[partition={partition}]",
        fontsize=11,
    )

    for ax, (name, ptr_ch, rel_idx) in zip(axes, labels):
        ptr_vals = Q_ptr[0, var_rows, ptr_ch]
        rel_vals = Q_rel[0, var_rows, rel_idx]
        _draw_qscale_scatter_panel(ax, rel_vals, ptr_vals, name)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out_dir / f"qscale_layer{layer_idx}.png")


# ── All-layers vertical strips (one figure per plot kind) ──────────────────────

def _row_label(ax, layer_idx: int) -> None:
    ax.text(
        -0.12, 0.5, f"L{layer_idx}",
        transform=ax.transAxes, va="center", ha="right", fontsize=10, fontweight="bold",
    )


def plot_scatter_all_layers(
    layer_debugs: List[dict],
    var_rows: List[int],
    out_dir: Path,
    partition: str,
) -> None:
    rows: List[tuple[int, dict]] = []
    for layer_idx, debug in enumerate(layer_debugs):
        mr, mp = debug.get("mass_rel"), debug.get("mass_ptr")
        if mr is None or mp is None or not var_rows:
            continue
        rows.append((layer_idx, debug))
    if not rows:
        return

    R = len(rows)
    fig, axes = plt.subplots(R, 3, figsize=(12, max(3.0, 2.8 * R)), squeeze=False)
    fig.suptitle(f"All layers: rel-mass vs ptr-mass [partition={partition}]", fontsize=12, y=1.002)

    for r, (layer_idx, debug) in enumerate(rows):
        mass_rel, mass_ptr = debug["mass_rel"], debug["mass_ptr"]
        for c, (label, rel_idx, ptr_ch) in enumerate(EXP1_CHANNELS):
            ax = axes[r, c]
            x = mass_rel[0, var_rows, rel_idx]
            y = mass_ptr[0, var_rows, ptr_ch]
            ax.scatter(x, y, s=3, alpha=0.35, rasterized=True)
            lim = max(float(x.max()), float(y.max())) * 1.05 + 1e-6
            ax.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.5)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            if r == 0:
                ax.set_title(label)
            if r == R - 1:
                ax.set_xlabel(f"mass_rel_{label}")
            if c == 0:
                ax.set_ylabel("mass_ptr")
                _row_label(ax, layer_idx)

    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    _save(fig, out_dir / "all_layers" / "scatter_all_layers.png")


def plot_entity_inv_hist_all_layers(
    layer_debugs: List[dict],
    entity_rows_by_type: dict,
    out_dir: Path,
    partition: str,
) -> None:
    rows: List[tuple[int, dict]] = []
    for layer_idx, debug in enumerate(layer_debugs):
        if debug.get("mass_rel") is None:
            continue
        rows.append((layer_idx, debug))
    if not rows:
        return

    R = len(rows)
    fig, axes = plt.subplots(R, 3, figsize=(12, max(3.0, 2.8 * R)), squeeze=False)
    fig.suptitle(f"All layers: entity inverse-edge mass [partition={partition}]", fontsize=12, y=1.002)

    for r, (layer_idx, debug) in enumerate(rows):
        mass_rel = debug["mass_rel"]
        for c, (label, tok_type, rel_idx) in enumerate(EXP2_ENTITY_INV):
            ax = axes[r, c]
            erows = entity_rows_by_type.get(tok_type, [])
            if not erows:
                ax.set_title(f"{label} (∅)" if r == 0 else "")
                ax.text(0.5, 0.5, "no tokens", ha="center", va="center", transform=ax.transAxes)
                if c == 0:
                    _row_label(ax, layer_idx)
                continue
            vals = mass_rel[0, erows, rel_idx]
            ax.hist(vals, bins=40, range=(0, 1), color="steelblue", edgecolor="none", alpha=0.8)
            ax.axvline(float(vals.mean()), color="red", lw=1.0, linestyle="--", label=f"μ={vals.mean():.3f}")
            if r == 0:
                ax.set_title(f"{label} (n={len(erows)})")
            if r == R - 1:
                ax.set_xlabel(f"mass_{label.upper()}_INV")
            ax.set_ylabel("count")
            if c == 0:
                _row_label(ax, layer_idx)
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    _save(fig, out_dir / "all_layers" / "entity_inv_hist_all_layers.png")


def _logit_topk_query_indices(var_rows: List[int], debug: dict, topk_queries: int) -> List[int]:
    """Same query choice as single-layer logit_topk; uses this layer's mass_rel."""
    n_q = min(topk_queries, len(var_rows))
    mass_rel = debug.get("mass_rel")
    if mass_rel is not None:
        total_mass = mass_rel[0, var_rows, :].max(axis=-1)
        top_q_local = np.argsort(total_mass)[::-1][:n_q]
    else:
        top_q_local = np.arange(n_q)
    return [var_rows[i] for i in top_q_local]


def plot_logit_topk_all_layers(
    layer_debugs: List[dict],
    var_rows: List[int],
    out_dir: Path,
    partition: str,
    topk_queries: int,
    K: int,
) -> None:
    ref_idx = next((i for i, d in enumerate(layer_debugs) if d.get("logit_content_mean") is not None), None)
    if ref_idx is None or not var_rows:
        return
    query_indices = _logit_topk_query_indices(var_rows, layer_debugs[ref_idx], topk_queries)
    n_q = len(query_indices)

    layer_rows: List[tuple[int, dict]] = []
    for layer_idx, debug in enumerate(layer_debugs):
        lc, lr = debug.get("logit_content_mean"), debug.get("logit_rel_mean")
        if lc is None or lr is None:
            continue
        layer_rows.append((layer_idx, debug))
    if not layer_rows:
        return

    R = len(layer_rows)
    fig, axes = plt.subplots(R, n_q, figsize=(max(4.0, 3.8 * n_q), max(2.5, 2.4 * R)), squeeze=False)
    fig.suptitle(
        f"All layers: top-{K} logit breakdown (queries from L{ref_idx}) [partition={partition}]",
        fontsize=11,
        y=1.01,
    )

    for r, (layer_idx, debug) in enumerate(layer_rows):
        lc, lr, lp = debug["logit_content_mean"], debug["logit_rel_mean"], debug.get("logit_ptr")
        for col, qi in enumerate(query_indices):
            ax = axes[r, col]
            total_logit = lc[0, qi, :] + lr[0, qi, :]
            if lp is not None:
                total_logit = total_logit + lp[0, qi, :]
            topk_keys = np.argsort(total_logit)[::-1][:K]

            c_vals = lc[0, qi, topk_keys] - lc[0, qi, topk_keys].mean()
            r_vals = lr[0, qi, topk_keys] - lr[0, qi, topk_keys].mean()
            p_raw = lp[0, qi, topk_keys] if lp is not None else np.zeros(K)
            p_vals = p_raw - p_raw.mean()

            x = np.arange(K)
            w = 0.25
            ax.bar(x - w, c_vals, w, label="content", color="#4c72b0")
            ax.bar(x, r_vals, w, label="rel", color="#dd8452")
            ax.bar(x + w, p_vals, w, label="ptr", color="#55a868")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{j}" for j in topk_keys], rotation=45, fontsize=6)
            if r == R - 1:
                ax.set_xlabel("key idx")
            if col == 0:
                ax.set_ylabel("centered logit")
                _row_label(ax, layer_idx)
            if r == 0:
                ax.set_title(f"query tok {qi}")
            if col == 0 and r == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    _save(fig, out_dir / "all_layers" / "logit_topk_all_layers.png")


def plot_logit_hist_all_layers(
    layer_debugs: List[dict],
    var_rows: List[int],
    out_dir: Path,
    partition: str,
    K: int,
) -> None:
    layer_rows: List[tuple[int, dict, np.ndarray, np.ndarray, np.ndarray]] = []
    for layer_idx, debug in enumerate(layer_debugs):
        lc, lr = debug.get("logit_content_mean"), debug.get("logit_rel_mean")
        if lc is None or lr is None or not var_rows:
            continue
        lp = debug.get("logit_ptr")
        w_c_all, w_r_all, w_p_all = [], [], []
        for qi in var_rows:
            total = lc[0, qi, :] + lr[0, qi, :]
            if lp is not None:
                total = total + lp[0, qi, :]
            topk = np.argsort(total)[::-1][:K]
            c = lc[0, qi, topk] - lc[0, qi, topk].mean()
            r = lr[0, qi, topk] - lr[0, qi, topk].mean()
            p = lp[0, qi, topk] if lp is not None else np.zeros(K)
            p = p - p.mean()
            stack = np.stack([c, r, p], axis=-1)
            stack = stack - stack.max(axis=-1, keepdims=True)
            exp_s = np.exp(stack)
            w = exp_s / exp_s.sum(axis=-1, keepdims=True)
            w_c_all.append(w[:, 0])
            w_r_all.append(w[:, 1])
            w_p_all.append(w[:, 2])
        w_c = np.concatenate(w_c_all)
        w_r = np.concatenate(w_r_all)
        w_p = np.concatenate(w_p_all)
        layer_rows.append((layer_idx, debug, w_c, w_r, w_p))

    if not layer_rows:
        return

    R = len(layer_rows)
    fig, axes = plt.subplots(R, 3, figsize=(12, max(3.0, 2.6 * R)), squeeze=False)
    fig.suptitle(
        f"All layers: key-centered softmax logit shares (top-{K} / query) [partition={partition}]",
        fontsize=11,
        y=1.002,
    )

    labels_cols = ["content (w_c)", "relational (w_r)", "pointer (w_p)"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for r, (layer_idx, _d, w_c, w_r, w_p) in enumerate(layer_rows):
        for c, (vals, lab, col) in enumerate(zip([w_c, w_r, w_p], labels_cols, colors)):
            ax = axes[r, c]
            ax.hist(vals, bins=50, range=(0, 1), color=col, edgecolor="none", alpha=0.8)
            ax.axvline(float(vals.mean()), color="red", lw=1.0, linestyle="--", label=f"μ={vals.mean():.3f}")
            if r == 0:
                ax.set_title(lab)
            if r == R - 1:
                ax.set_xlabel("softmax weight w")
            ax.set_ylabel("count")
            if c == 0:
                _row_label(ax, layer_idx)
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    _save(fig, out_dir / "all_layers" / "logit_hist_all_layers.png")


def plot_qscale_all_layers(
    layer_debugs: List[dict],
    var_rows: List[int],
    out_dir: Path,
    partition: str,
) -> None:
    labels = [("ATTR", 0, 0), ("ANNOT", 1, 2), ("ITEM", 2, 4)]
    rows: List[tuple[int, dict]] = []
    for layer_idx, debug in enumerate(layer_debugs):
        if debug.get("Q_ptr") is None or debug.get("Q_rel_shared") is None or not var_rows:
            continue
        rows.append((layer_idx, debug))
    if not rows:
        return

    R = len(rows)
    fig, axes = plt.subplots(R, 3, figsize=(12, max(3.0, 2.8 * R)), squeeze=False)
    fig.suptitle(
        f"All layers: $Q_{{\\mathrm{{rel}}}}$ vs $Q_{{\\mathrm{{ptr}}}}$ (above diag → ptr > rel) "
        f"[partition={partition}]",
        fontsize=12,
        y=1.002,
    )

    for r, (layer_idx, debug) in enumerate(rows):
        Q_ptr, Q_rel = debug["Q_ptr"], debug["Q_rel_shared"]
        for c, (name, ptr_ch, rel_idx) in enumerate(labels):
            ax = axes[r, c]
            ptr_vals = Q_ptr[0, var_rows, ptr_ch]
            rel_vals = Q_rel[0, var_rows, rel_idx]
            xl = r"$Q_{\mathrm{rel}}$" if r == R - 1 else None
            yl = r"$Q_{\mathrm{ptr}}$" if c == 0 else None
            _draw_qscale_scatter_panel(ax, rel_vals, ptr_vals, name if r == 0 else "", xlabel=xl, ylabel=yl)
            if c == 0:
                _row_label(ax, layer_idx)

    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    _save(fig, out_dir / "all_layers" / "qscale_all_layers.png")


def plot_all_layers_strips(
    layer_records: List[dict],
    num_layers: int,
    var_rows: List[int],
    entity_rows: dict,
    out_dir: Path,
    partition: str,
    topk_queries: int,
    topk_keys: int,
) -> None:
    layer_debugs = [layer_records[i].get("g0", {}) for i in range(num_layers)]
    if not any(layer_debugs):
        return
    sub = out_dir / "all_layers"
    sub.mkdir(parents=True, exist_ok=True)
    print(f"\nAll-layers strips → {sub}/")
    plot_scatter_all_layers(layer_debugs, var_rows, out_dir, partition)
    plot_entity_inv_hist_all_layers(layer_debugs, entity_rows, out_dir, partition)
    plot_logit_topk_all_layers(layer_debugs, var_rows, out_dir, partition, topk_queries, topk_keys)
    plot_logit_hist_all_layers(layer_debugs, var_rows, out_dir, partition, topk_keys)
    plot_qscale_all_layers(layer_debugs, var_rows, out_dir, partition)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pointer attention diagnostics for EntityMarformer. "
        "Architecture and training defaults come from <run-dir>/train_config.json (no graph-mask / pointer flags needed).",
    )
    parser.add_argument("--run-dir", required=True, help="Path to training run directory.")
    parser.add_argument("--output-dir", default=None, help="Where to save plots (default: <run-dir>/attn_diagnostics).")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with data_bundle.json (only if automatic resolution from train_config data_dir fails).",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "test"),
        default="test",
        help="Bundle partition: only train/test exist (no val). Default test matches training holdout metrics.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-graphs", type=int, default=1,
                        help="Number of graphs to process (one full graph per partition).")
    parser.add_argument("--topk-queries", type=int, default=3,
                        help="Number of example queries for the 3a grouped-bar chart.")
    parser.add_argument("--topk-keys", type=int, default=10,
                        help="K top keys used in 3a and 3b.")
    parser.add_argument(
        "--max-item",
        type=int,
        default=None,
        help="Override training.max_item from train_config (default: same as training json).",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Which item chunk to use when max_item chunking applies (0-based).",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))
    out_dir = normalize_cli_path(args.output_dir) if args.output_dir else (run_dir / "attn_diagnostics")
    device = torch.device(args.device)

    print(f"Run dir   : {run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Partition : {args.partition} (train/test only in bundle; default=test = holdout in train.py)")
    print(f"Device    : {device}")

    data_override = normalize_cli_path(args.data_dir) if args.data_dir else None
    model, bundle, converter, sizes, tc = load_model(run_dir, device, data_dir=data_override)
    max_item = args.max_item
    if max_item is None:
        max_item = tc.get("training", {}).get("max_item")
        if max_item is not None:
            print(f"Using max_item={max_item} from train_config.json (override with --max-item / --max-item 0 to disable).")
    if max_item is not None and max_item <= 0:
        max_item = None

    graphs = build_partition_graphs(
        bundle,
        converter,
        sizes,
        tc,
        args.max_graphs,
        partition=args.partition,
        max_item=max_item,
        chunk_index=args.chunk_index,
    )

    num_layers = model.config.num_layers

    # Accumulate per-layer records across graphs
    # Each entry is a dict of {key: list_of_numpy_arrays_across_graphs}
    layer_records: List[dict] = [{} for _ in range(num_layers)]

    for g_idx, graph in enumerate(graphs[:args.max_graphs]):
        print(f"\nGraph {g_idx+1}/{min(len(graphs), args.max_graphs)} ({graph.num_tokens} tokens) ...")

        attention_debug: list = []
        with torch.no_grad():
            model(graph, device=device, attention_debug=attention_debug)

        var_rows    = token_indices(graph, VARIABLE_TYPES)
        entity_rows = {t: token_indices(graph, {t}) for t in ENTITY_TYPES}

        if not var_rows:
            print("  No rating tokens found, skipping graph.")
            continue

        for layer_idx, layer_debug in enumerate(attention_debug):
            # Convert tensors to numpy
            rec = {}
            for k, v in layer_debug.items():
                rec[k] = v.numpy() if isinstance(v, torch.Tensor) else v
            layer_records[layer_idx][f"g{g_idx}"] = rec

    # ── Per-layer plots ────────────────────────────────────────────────────────
    # For simplicity (and because we use a single large graph), work with the
    # first (and only) graph's records directly.
    for layer_idx in range(num_layers):
        if not layer_records[layer_idx]:
            continue
        # Use first graph's record for plotting (single-graph case)
        debug = layer_records[layer_idx].get("g0", {})
        if not debug:
            continue

        # We need var_rows and entity_rows for the first graph
        graph = graphs[0]
        var_rows    = token_indices(graph, VARIABLE_TYPES)
        entity_rows = {t: token_indices(graph, {t}) for t in ENTITY_TYPES}

        print(f"\nLayer {layer_idx}: plotting ({len(var_rows)} rating tokens) ...")

        plot_scatter(layer_idx, var_rows, debug, out_dir, args.partition)
        plot_entity_inv_hist(layer_idx, entity_rows, debug, out_dir, args.partition)
        plot_logit_topk(
            layer_idx, var_rows, debug, out_dir, args.partition,
            topk_queries=args.topk_queries, K=args.topk_keys,
        )
        plot_logit_hist(layer_idx, var_rows, debug, out_dir, args.partition, K=args.topk_keys)
        plot_qscale(layer_idx, var_rows, debug, out_dir, args.partition)

    if graphs and any(layer_records[i] for i in range(num_layers)):
        g0 = graphs[0]
        vr = token_indices(g0, VARIABLE_TYPES)
        er = {t: token_indices(g0, {t}) for t in ENTITY_TYPES}
        if vr:
            plot_all_layers_strips(
                layer_records,
                num_layers,
                vr,
                er,
                out_dir,
                args.partition,
                args.topk_queries,
                args.topk_keys,
            )

    # ── Save raw arrays ────────────────────────────────────────────────────────
    npz_path = out_dir / "attention_debug.npz"
    flat = {}
    for layer_idx, gdict in enumerate(layer_records):
        for g_key, rec in gdict.items():
            for k, v in rec.items():
                flat[f"layer{layer_idx}_{g_key}_{k}"] = v
    if flat:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **flat)
        print(f"\nSaved raw arrays to {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
