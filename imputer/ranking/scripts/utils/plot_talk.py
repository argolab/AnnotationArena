#!/usr/bin/env python3
"""
Generate talk-quality plots for LLM Rubric results.

Outputs to PLOTS/TALK/:
  1. llm_rubric_train_val_curves.png  — training masked CE (left) + val missing CE (right),
                                        side-by-side, for Marformer vs Marformer Hard Mask.
                                        Uses the largest split (175) as representative.
  2. llm_rubric_test_log_loss.png     — test log loss vs training size for all methods
                                        + at-random baseline.

Usage (run from imputer/ranking):
    python scripts/utils/plot_talk.py
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── ICML-style rcParams ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    14,
    "axes.titlesize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.75",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "0.88",
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
    "lines.markersize":  7,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

RESULTS_ROOT = Path("RESULTS")
OUT_DIR      = Path("PLOTS/TALK")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_MF   = "#1f6fba"   # Marformer
C_HM   = "#c0392b"   # Marformer Hard Mask
C_SF   = "#27ae60"   # STAN Factor (transductive)
C_SN   = "#e67e22"   # STAN Normal (transductive)
C_RAND = "0.55"      # at-random

SPLIT_RE    = re.compile(r"LLMRubric_225_25_9_(\d+)$")
STAN_T_RE   = re.compile(r"LLMRubric_225_25_9_(\d+)_nt_(Factor|Normal)_eval$")

LLM_RUBRIC_NUM_CLASSES = 4
LLM_RUBRIC_DATA_ROOT   = Path("DATA/LLM_RUBRIC")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_tfevents(run_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Return (train_masked_ce, val_missing_ce) epoch-indexed lists from TFEvents.
    Searches lightning_logs/version_*/events.out.tfevents.* under run_dir.
    """
    log_root = run_dir / "lightning_logs"
    if not log_root.exists():
        return [], []

    # Pick the first version folder (version_0 normally)
    version_dirs = sorted(log_root.glob("version_*"))
    if not version_dirs:
        return [], []

    ea = EventAccumulator(str(version_dirs[0]))
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    train_masked, val_missing = [], []
    if "train/masked_ce_epoch" in tags:
        train_masked = [s.value for s in ea.Scalars("train/masked_ce_epoch")]
    if "val/missing_ce" in tags:
        val_missing = [s.value for s in ea.Scalars("val/missing_ce")]

    return train_masked, val_missing


def _load_test_loss(run_dir: Path, which: str = "best") -> Optional[float]:
    p = run_dir / "TEST_RESULTS" / f"{which}.json"
    if not p.exists():
        return None
    data = json.load(open(p))
    v = data.get("missing", {}).get("log_loss")
    return float(v) if v is not None else None


def _load_stan_t_loss(eval_dir: Path) -> Optional[float]:
    p = eval_dir / "predictive_metrics.json"
    if not p.exists():
        return None
    data = json.load(open(p))
    ll = data.get("rating_missing_log_likelihood")
    return float(-ll) if ll is not None else None


def _mf_runs(model_type: str) -> Dict[int, Path]:
    root = RESULTS_ROOT / model_type / "LLM_RUBRIC"
    runs = {}
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        m = SPLIT_RE.match(d.name)
        if m and d.is_dir():
            runs[int(m.group(1))] = d
    return runs


def _empirical_marginal_baseline(sizes: List[int]) -> Dict[int, float]:
    """
    For each training size, compute the cross-entropy of predicting the training
    observed marginal distribution against the test missing labels.
    H(p_test, q_train) = -sum_c p_test(c) * log(q_train(c))
    """
    C = LLM_RUBRIC_NUM_CLASSES
    result = {}
    for size in sizes:
        p = LLM_RUBRIC_DATA_ROOT / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if not p.exists():
            continue
        b = json.load(open(p))
        from collections import Counter
        train_obs  = [r for r in b.get("observed_ratings", []) if r["instance"] == "train"]
        test_miss  = [r for r in b.get("missing_ratings",  []) if r["instance"] == "test"]
        if not train_obs or not test_miss:
            continue
        train_cnt  = Counter(r["value"] for r in train_obs)
        test_cnt   = Counter(r["value"] for r in test_miss)
        train_n    = len(train_obs)
        test_n     = len(test_miss)
        q = {c: train_cnt.get(c, 0) / train_n for c in range(1, C + 1)}
        p_test = {c: test_cnt.get(c, 0) / test_n for c in range(1, C + 1)}
        ce = -sum(p_test[c] * math.log(q[c] + 1e-12) for c in range(1, C + 1))
        result[size] = ce
    return result


def _stan_t_runs(variant: str) -> Dict[int, Path]:
    root = RESULTS_ROOT / "STAN" / "LLM_RUBRIC_T"
    runs = {}
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        m = STAN_T_RE.match(d.name)
        if m and d.is_dir() and m.group(2) == variant:
            runs[int(m.group(1))] = d
    return runs


# ── Plot 1: training masked CE + val missing CE curves (largest split) ────────

def plot_curves(split_size: int = 175) -> None:
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 4.5))

    for model_type, label, color in [
        ("MARFORMER",           "Marformer",             C_MF),
        ("MARFORMER_HARD_MASK", "Marformer + Hard Mask", C_HM),
    ]:
        run_dir = (
            RESULTS_ROOT / model_type / "LLM_RUBRIC"
            / f"LLMRubric_225_25_9_{split_size}"
        )
        if not run_dir.exists():
            print(f"  [skip] {run_dir.name} not found")
            continue

        train_masked, val_missing = _read_tfevents(run_dir)
        epochs_train = list(range(len(train_masked)))
        epochs_val   = list(range(len(val_missing)))

        if train_masked:
            ax_train.plot(epochs_train, train_masked, color=color, label=label)
        if val_missing:
            ax_val.plot(epochs_val, val_missing, color=color, label=label)

    for ax, title, ylabel in [
        (ax_train, "Training Masked CE", "Cross-Entropy"),
        (ax_val,   "Validation Missing CE", "Cross-Entropy"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")

    fig.suptitle(f"LLM Rubric — Training Size {split_size}", y=1.02)
    out = OUT_DIR / "llm_rubric_train_val_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Plot 2: test log loss vs training size ─────────────────────────────────────

def plot_test_vs_size() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # ── Marformer series ──────────────────────────────────────────────────────
    for model_type, label, color, marker in [
        ("MARFORMER",           "Marformer",             C_MF, "o"),
        ("MARFORMER_HARD_MASK", "Marformer + Hard Mask", C_HM, "s"),
    ]:
        runs = _mf_runs(model_type)
        pairs = sorted(
            [(sz, _load_test_loss(d)) for sz, d in runs.items()],
            key=lambda x: x[0]
        )
        pairs = [(sz, ll) for sz, ll in pairs if ll is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color=color, marker=marker, label=label, linestyle="-")

    # ── STAN Transductive ─────────────────────────────────────────────────────
    for variant, label, color, marker in [
        ("Factor", "STAN Factor (Transductive)", C_SF, "^"),
        ("Normal", "STAN Normal (Transductive)", C_SN, "D"),
    ]:
        runs = _stan_t_runs(variant)
        pairs = sorted(
            [(sz, _load_stan_t_loss(d)) for sz, d in runs.items()],
            key=lambda x: x[0]
        )
        pairs = [(sz, ll) for sz, ll in pairs if ll is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color=color, marker=marker, label=label, linestyle="--")

    # ── Empirical marginal baseline ───────────────────────────────────────────
    all_sizes = sorted(set(
        list(_mf_runs("MARFORMER").keys()) +
        list(_mf_runs("MARFORMER_HARD_MASK").keys())
    ))
    marginal = _empirical_marginal_baseline(all_sizes)
    if marginal:
        mxs, mys = zip(*sorted(marginal.items()))
        ax.plot(mxs, mys, color=C_RAND, linestyle=":", linewidth=1.8,
                marker="x", label="Empirical Marginal Baseline")

    ax.set_xlabel("Training Items")
    ax.set_ylabel("Test Log Loss")
    ax.set_title("LLM Rubric — Test Log Loss vs Training Size")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    out = OUT_DIR / "llm_rubric_test_log_loss.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== LLM Rubric — Talk Plots ===\n")
    print("-- Plot 1: train masked CE + val missing CE curves --")
    plot_curves(split_size=175)

    print("\n-- Plot 2: test log loss vs training size --")
    plot_test_vs_size()

    print(f"\nDone. Output → {OUT_DIR}/")


if __name__ == "__main__":
    main()
