#!/usr/bin/env python3
"""
Pointer attention diagnostics for a trained EntityMarformer.

Runs the model on the test partition (all test variables) and produces:
  - scatter_layer{L}.png       : Exp 1  — mass_rel vs mass_ptr per relation (ATTR/ANNOT/ITEM)
  - entity_inv_hist_layer{L}.png : Exp 2 — inverse-edge mass histograms for item/annotator/attribute
  - logit_topk_layer{L}.png    : Exp 3a — centered top-K grouped bar chart for a few queries
  - logit_hist_layer{L}.png    : Exp 3b — key-axis-centered softmax-norm histograms
  - qscale_layer{L}.png        : Exp 3c — Q_ptr vs Q_rel_shared scale histograms
  - attention_debug.npz        : raw arrays for offline analysis

Usage:
    cd /home/xwang397/AnnotationArena/imputer/ranking
    PYTHONPATH=. python scripts/analyze_pointer_attention.py \\
        --run-dir OUTPUT/ENTITY_MF/POINTER_SWEEPS/ptrswp_valmcar_ptr1_drop0.7_8L4H_emb80_300ep_dist_run2 \\
        --output-dir OUTPUT/ENTITY_MF/POINTER_SWEEPS/ptrswp_valmcar_ptr1_drop0.7_8L4H_emb80_300ep_dist_run2/attn_diagnostics \\
        --max-graphs 80 \\
        --topk-queries 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

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
from imputer.entity_mf.train import load_bundle_and_converter, build_entity_marformer_from_bundle


# ── Relation / channel metadata ────────────────────────────────────────────────
VARIABLE_TYPES = {"rating", "ranking_pairwise"}
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
    )


def load_model(run_dir: Path, device: torch.device):
    cfg_path = run_dir / "train_config.json"
    with open(cfg_path) as f:
        tc = json.load(f)

    data_dir = _resolve_data_dir(run_dir, tc["data"]["data_dir"])

    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    mc = tc["model"]
    config = EntityMarformerConfig(
        embedding_dim=mc["embedding_dim"],
        num_layers=mc["num_layers"],
        attention_heads=mc["attention_heads"],
        d_ff=mc.get("d_ff", 128),
        num_ffn_layers=mc.get("num_ffn_layers", 1),
        dropout=mc.get("dropout", 0.1),
        use_per_head_rel=mc.get("use_per_head_rel", False),
        use_pointer=mc.get("use_pointer", False),
        use_rel_value=mc.get("use_rel_value", False),
        use_addone_attn=mc.get("use_addone_attn", False),
        type_embedding_init=mc.get("type_embedding_init", "kaiming"),
        use_deviation_norm=mc.get("use_deviation_norm", False),
        scale_shared_rel=mc.get("scale_shared_rel", False),
        use_learned_embedding=mc.get("use_learned_embedding", False),
        logit_high=mc.get("logit_high", 20.0),
        temperature=mc.get("temperature", 1.0),
        global_param_dim=mc.get("global_param_dim", 5),
        pointer_channels=mc.get("pointer_channels", None),
        freeze_ptr=mc.get("freeze_ptr", False),
    )

    tr = tc.get("training", {})
    model, _ = build_entity_marformer_from_bundle(
        bundle=bundle,
        converter=converter,
        sizes=sizes,
        config=config,
        annotator_reg_weight=tr.get("annotator_reg_weight", 0.0),
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
    return model, bundle, converter, sizes, tc


def build_test_graphs(bundle, converter, sizes, tc, max_graphs: int):
    """Build EntityGraph objects from the test partition."""
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
        llm_input_dist=tr.get("llm_input_dist", False),
        item_dropout_rate=tr.get("item_dropout_rate", 1.0),
    )

    test_obs  = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_miss = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    test_all  = test_obs + test_miss
    print(f"Test variables: {len(test_all)} total ({len(test_obs)} observed, {len(test_miss)} missing)")

    # Build one graph per variable (same as the evaluation loop in train.py)
    # For attention diagnostics we want one graph that contains ALL test variables
    # so we can see the full attention pattern — build a single large graph.
    # Cap by using only the first max_graphs*10 variables, then build 1 graph.
    # Actually: the model processes one graph at a time; the "graph" contains all
    # the variables + entity tokens needed. There is only one graph for a dataset.
    graph = variable_list_to_entity_graph(test_all, types)
    print(f"Graph: {graph.num_tokens} tokens")
    return [graph]  # list for consistent loop below


# ── Token index helpers ────────────────────────────────────────────────────────

def token_indices(graph, type_names: set) -> List[int]:
    return [i for i, t in enumerate(graph.tokens) if t.type_name in type_names]


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_scatter(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path):
    """Exp 1: scatter of mass_rel vs mass_ptr for ATTR / ANNOT / ITEM."""
    mass_rel = debug.get("mass_rel")  # [B, L, R] numpy
    mass_ptr = debug.get("mass_ptr")  # [B, L, 3] numpy
    if mass_rel is None or mass_ptr is None or not var_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Layer {layer_idx}: rel-mass vs ptr-mass (variable tokens)", fontsize=11)
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


def plot_entity_inv_hist(layer_idx: int, entity_rows_by_type: dict, debug: dict, out_dir: Path):
    """Exp 2: histograms of inverse-edge mass for item / annotator / attribute tokens."""
    mass_rel = debug.get("mass_rel")
    if mass_rel is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Layer {layer_idx}: entity token inverse-edge mass", fontsize=11)
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
                    topk_queries: int = 3, K: int = 10):
    """Exp 3a: centered top-K grouped bar chart for a few example variable queries."""
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
    fig.suptitle(f"Layer {layer_idx}: top-{K} key logit breakdown (centered)", fontsize=11)

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


def plot_logit_hist(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path, K: int = 10):
    """Exp 3b: key-axis-centered softmax-normalized histograms over all variable queries."""
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
        f"Layer {layer_idx}: key-axis-centered softmax logit shares (top-{K} keys per query)",
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


def plot_qscale(layer_idx: int, var_rows: List[int], debug: dict, out_dir: Path):
    """Exp 3c: Q_ptr vs Q_rel_shared amplitude histograms (one per matched channel/relation)."""
    Q_ptr = debug.get("Q_ptr")          # [B, L, 3]
    Q_rel = debug.get("Q_rel_shared")   # [B, L, R]
    if Q_ptr is None or Q_rel is None or not var_rows:
        return

    labels = [("ATTR", 0, 0), ("ANNOT", 1, 2), ("ITEM", 2, 4)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Layer {layer_idx}: Q_ptr vs Q_rel amplitude (variable tokens)", fontsize=11)

    for ax, (name, ptr_ch, rel_idx) in zip(axes, labels):
        ptr_vals = Q_ptr[0, var_rows, ptr_ch]   # [N_var]
        rel_vals = Q_rel[0, var_rows, rel_idx]  # [N_var]

        all_vals = np.concatenate([ptr_vals, rel_vals])
        lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo - 0.1, hi + 0.1, 60)

        ax.hist(ptr_vals, bins=bins, alpha=0.6, label=f"Q_ptr[{ptr_ch}]", color="#55a868", edgecolor="none")
        ax.hist(rel_vals, bins=bins, alpha=0.6, label=f"Q_rel[{rel_idx}={name}]", color="#dd8452", edgecolor="none")
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("logit value"); ax.set_ylabel("count")
        ax.set_title(name); ax.legend(fontsize=8)

    _save(fig, out_dir / f"qscale_layer{layer_idx}.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pointer attention diagnostics for EntityMarformer.")
    parser.add_argument("--run-dir", required=True, help="Path to training run directory.")
    parser.add_argument("--output-dir", default=None, help="Where to save plots (default: <run-dir>/attn_diagnostics).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-graphs", type=int, default=1,
                        help="Number of graphs to process. Use 1 for the full test graph.")
    parser.add_argument("--topk-queries", type=int, default=3,
                        help="Number of example queries for the 3a grouped-bar chart.")
    parser.add_argument("--topk-keys", type=int, default=10,
                        help="K top keys used in 3a and 3b.")
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else run_dir / "attn_diagnostics"
    device   = torch.device(args.device)

    print(f"Run dir   : {run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Device    : {device}")

    model, bundle, converter, sizes, tc = load_model(run_dir, device)
    graphs = build_test_graphs(bundle, converter, sizes, tc, args.max_graphs)

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
            print("  No variable tokens found, skipping graph.")
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

        print(f"\nLayer {layer_idx}: plotting ({len(var_rows)} var tokens) ...")

        plot_scatter(layer_idx, var_rows, debug, out_dir)
        plot_entity_inv_hist(layer_idx, entity_rows, debug, out_dir)
        plot_logit_topk(layer_idx, var_rows, debug, out_dir,
                        topk_queries=args.topk_queries, K=args.topk_keys)
        plot_logit_hist(layer_idx, var_rows, debug, out_dir, K=args.topk_keys)
        plot_qscale(layer_idx, var_rows, debug, out_dir)

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
