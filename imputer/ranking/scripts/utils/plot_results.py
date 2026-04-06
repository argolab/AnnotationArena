#!/usr/bin/env python3
"""
Plot training and test results for Entity Marformer.

Generates three plot types for each MARFORMER / MARFORMER_HARD_MASK × dataset combo:
  1. Val log loss curve per split folder   → {run_dir}/val_log_loss_curve.png
  2. Best val log loss vs training size    → {model_type}/{dataset}/best_val_log_loss_vs_size.png
  3. Test log loss vs training size
       – best checkpoint                  → {model_type}/{dataset}/test_log_loss_best_vs_size.png
       – last checkpoint                  → {model_type}/{dataset}/test_log_loss_last_vs_size.png

Usage (run from imputer/ranking):
    python scripts/utils/plot_results.py llm_rubric
    python scripts/utils/plot_results.py summeval
    python scripts/utils/plot_results.py stan        # not yet implemented
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── ICML-style rcParams ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "0.88",
    "grid.linewidth":     0.6,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
    "figure.dpi":         150,   # screen preview; savefig uses 300
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

RESULTS_ROOT = Path("RESULTS")
MODEL_TYPES  = ["MARFORMER", "MARFORMER_HARD_MASK"]
MODEL_LABELS = {"MARFORMER": "Marformer", "MARFORMER_HARD_MASK": "Marformer + Hard Mask"}
MODEL_COLOR  = {"MARFORMER": "#1f6fba", "MARFORMER_HARD_MASK": "#c0392b"}
MODEL_MARKER = {"MARFORMER": "o",       "MARFORMER_HARD_MASK": "s"}

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "llm_rubric": {
        "subdir":     "LLM_RUBRIC",
        "split_re":   re.compile(r"LLMRubric_225_25_9_(\d+)$"),
        "size_label": "Training Items",
    },
    "summeval": {
        "subdir":     "SUMMEVAL",
        "split_re":   re.compile(r"SummEval_1600_8_4_(\d+)$"),
        "size_label": "Training Documents",
    },
}


# ── Data extraction ───────────────────────────────────────────────────────────

def load_val_curve(run_dir: Path) -> Tuple[List[int], List[float]]:
    """Read (epochs, val_log_loss) from training_history.json."""
    path = run_dir / "training_history.json"
    if not path.exists():
        return [], []
    with open(path) as f:
        history = json.load(f)
    epochs, losses = [], []
    for entry in history:
        epoch = entry.get("epoch")
        xent = (
            entry.get("val_eval", {})
                 .get("metrics", {})
                 .get("missing", {})
                 .get("rating", {})
                 .get("xent", None)
        )
        if epoch is not None and xent is not None:
            epochs.append(epoch)
            losses.append(float(xent))
    return epochs, losses


def load_test_loss(run_dir: Path, which: str) -> Optional[float]:
    """Read missing.log_loss from TEST_RESULTS/{which}.json."""
    path = run_dir / "TEST_RESULTS" / f"{which}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    val = data.get("missing", {}).get("log_loss", None)
    return float(val) if val is not None else None


def find_runs(model_type: str, dataset_cfg: dict) -> Dict[int, Path]:
    """Return {training_size: run_dir} for an existing model_type/dataset combo."""
    root = RESULTS_ROOT / model_type / dataset_cfg["subdir"]
    runs: Dict[int, Path] = {}
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = dataset_cfg["split_re"].match(d.name)
        if m:
            runs[int(m.group(1))] = d
    return runs


# ── Plot 1: Val log loss curve per run dir ────────────────────────────────────

def plot_val_curve(run_dir: Path, training_size: int, size_label: str) -> None:
    epochs, losses = load_val_curve(run_dir)
    if not epochs:
        print(f"    [skip plot1] no val curve data in {run_dir.name}")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, losses, color="#1f6fba")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Log Loss")
    ax.set_title(f"{size_label} = {training_size}")
    out_path = run_dir / "val_log_loss_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    saved → {out_path.relative_to(RESULTS_ROOT)}")


# ── Plot 2: Best val log loss vs training size ────────────────────────────────

def plot_best_val_vs_size(
    out_dir: Path,
    runs_by_model: Dict[str, Dict[int, Path]],
    size_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    any_data = False

    for mt in MODEL_TYPES:
        runs = runs_by_model.get(mt, {})
        sizes, bests = [], []
        for size in sorted(runs):
            _, losses = load_val_curve(runs[size])
            if losses:
                sizes.append(size)
                bests.append(min(losses))
        if sizes:
            ax.plot(
                sizes, bests,
                marker=MODEL_MARKER[mt],
                color=MODEL_COLOR[mt],
                label=MODEL_LABELS[mt],
            )
            any_data = True

    if not any_data:
        plt.close(fig)
        print(f"    [skip plot2] no val data for {out_dir}")
        return

    ax.set_xlabel(size_label)
    ax.set_ylabel("Best Val Log Loss")
    ax.set_title("Best Validation Log Loss vs Training Size")
    ax.legend()
    out_path = out_dir / "best_val_log_loss_vs_size.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(RESULTS_ROOT)}")


# ── Plot 3: Test log loss vs training size ────────────────────────────────────

def plot_test_vs_size(
    out_dir: Path,
    runs_by_model: Dict[str, Dict[int, Path]],
    which: str,
    size_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    any_data = False

    for mt in MODEL_TYPES:
        runs = runs_by_model.get(mt, {})
        sizes, losses = [], []
        for size in sorted(runs):
            ll = load_test_loss(runs[size], which)
            if ll is not None:
                sizes.append(size)
                losses.append(ll)
        if sizes:
            ax.plot(
                sizes, losses,
                marker=MODEL_MARKER[mt],
                color=MODEL_COLOR[mt],
                label=MODEL_LABELS[mt],
            )
            any_data = True

    if not any_data:
        plt.close(fig)
        print(f"    [skip plot3/{which}] no test data for {out_dir}")
        return

    ckpt_label = "Best Checkpoint" if which == "best" else "Last Checkpoint"
    ax.set_xlabel(size_label)
    ax.set_ylabel("Test Log Loss")
    ax.set_title(f"Test Log Loss vs Training Size ({ckpt_label})")
    ax.legend()
    out_path = out_dir / f"test_log_loss_{which}_vs_size.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(RESULTS_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("llm_rubric", "summeval", "stan"):
        print("Usage: python scripts/utils/plot_results.py <llm_rubric|summeval|stan>")
        sys.exit(1)

    dataset_key = sys.argv[1]

    if dataset_key == "stan":
        print("Stan plotting not yet implemented.")
        sys.exit(0)

    cfg = DATASET_CONFIGS[dataset_key]
    size_label = cfg["size_label"]

    # Collect all runs per model type
    runs_by_model: Dict[str, Dict[int, Path]] = {}
    for mt in MODEL_TYPES:
        runs_by_model[mt] = find_runs(mt, cfg)

    total_runs = sum(len(v) for v in runs_by_model.values())
    if total_runs == 0:
        print(f"No runs found under RESULTS/*/({cfg['subdir']})")
        sys.exit(1)

    print(f"\n=== {dataset_key.upper()} — {total_runs} run(s) found ===\n")

    # ── Plot 1: per-run val curve ─────────────────────────────────────────────
    print("-- Plot 1: val log loss curves --")
    for mt, runs in runs_by_model.items():
        for size, run_dir in sorted(runs.items()):
            print(f"  [{mt}] size={size}")
            plot_val_curve(run_dir, size, size_label)

    # ── Plots 2 & 3: vs training size — one set of PNGs per dataset root ─────
    # Save to the dataset-level parent: RESULTS/{MODEL_TYPE}/{DATASET}/
    # Plots 2 & 3 compare both model types on the same axes.
    # Use the first model type's dataset dir as the output location,
    # OR write to RESULTS/{DATASET}/ — but user said root of dataset results.
    # We write one combined set under each model_type dir that has data,
    # plus a combined set directly under RESULTS/{DATASET}/ for side-by-side.

    # Combined output dir at RESULTS/{subdir}/
    combined_out = RESULTS_ROOT / cfg["subdir"]
    combined_out.mkdir(parents=True, exist_ok=True)

    print("\n-- Plot 2: best val log loss vs training size --")
    plot_best_val_vs_size(combined_out, runs_by_model, size_label)

    print("\n-- Plot 3: test log loss vs training size --")
    for which in ("best", "last"):
        plot_test_vs_size(combined_out, runs_by_model, which, size_label)

    print(f"\nDone. Combined plots → RESULTS/{cfg['subdir']}/")


if __name__ == "__main__":
    main()
