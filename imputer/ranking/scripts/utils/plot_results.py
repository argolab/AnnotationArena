#!/usr/bin/env python3
"""
Plot training and test results for Entity Marformer + STAN baselines.

Generates:
  1. Val log loss curve per Marformer split folder
       → {run_dir}/val_log_loss_curve.png
  2. Best val log loss vs training size  (Marformer + STAN)
       → RESULTS/MARFORMER/{DATASET}/best_val_log_loss_vs_size.png
  3. Test log loss vs training size — best checkpoint  (Marformer + STAN)
       → RESULTS/MARFORMER/{DATASET}/test_log_loss_best_vs_size.png
  4. Test log loss vs training size — last checkpoint  (Marformer + STAN)
       → RESULTS/MARFORMER/{DATASET}/test_log_loss_last_vs_size.png

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

# ── ICML-style rcParams ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     13,
    "axes.titlesize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.75",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "0.88",
    "grid.linewidth":     0.6,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

RESULTS_ROOT  = Path("RESULTS")
STAN_ROOT     = RESULTS_ROOT / "STAN"

# ── Series styling ────────────────────────────────────────────────────────────
SERIES = {
    "MARFORMER":          {"label": "Marformer",              "color": "#1f6fba", "marker": "o", "ls": "-"},
    "MARFORMER_HARD_MASK":{"label": "Marformer + Hard Mask",  "color": "#c0392b", "marker": "s", "ls": "-"},
    "STAN_Factor":        {"label": "STAN Factor",            "color": "#27ae60", "marker": "^", "ls": "--"},
    "STAN_Normal":        {"label": "STAN Normal",            "color": "#e67e22", "marker": "D", "ls": "--"},
}

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "llm_rubric": {
        "subdir":          "LLM_RUBRIC",
        "mf_split_re":     re.compile(r"LLMRubric_225_25_9_(\d+)$"),
        "stan_split_re":   re.compile(r"LLMRubric_225_25_9_(\d+)_(Factor|Normal)_eval$"),
        "size_label":      "Training Items",
    },
    "summeval": {
        "subdir":          "SUMMEVAL",
        "mf_split_re":     re.compile(r"SummEval_1600_8_4_(\d+)$"),
        "stan_split_re":   re.compile(r"SummEval_1600_8_4_(\d+)_(Factor|Normal)_eval$"),
        "size_label":      "Training Items",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_val_curve(run_dir: Path) -> Tuple[List[int], List[float]]:
    """(epochs, val_log_loss) from training_history.json."""
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


def load_mf_test_loss(run_dir: Path, which: str) -> Optional[float]:
    """missing.log_loss from TEST_RESULTS/{which}.json."""
    path = run_dir / "TEST_RESULTS" / f"{which}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    val = data.get("missing", {}).get("log_loss", None)
    return float(val) if val is not None else None


def load_stan_loss(stan_eval_dir: Path) -> Optional[float]:
    """Negate rating_missing_log_likelihood from predictive_metrics.json."""
    path = stan_eval_dir / "predictive_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    ll = data.get("rating_missing_log_likelihood", None)
    return float(-ll) if ll is not None else None


def find_mf_runs(model_type: str, cfg: dict) -> Dict[int, Path]:
    """Return {size: run_dir} for a Marformer model_type × dataset."""
    root = RESULTS_ROOT / model_type / cfg["subdir"]
    runs: Dict[int, Path] = {}
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        m = cfg["mf_split_re"].match(d.name)
        if m and d.is_dir():
            runs[int(m.group(1))] = d
    return runs


def find_stan_runs(variant: str, cfg: dict) -> Dict[int, Path]:
    """Return {size: eval_dir} for a STAN variant × dataset."""
    root = STAN_ROOT / cfg["subdir"]
    runs: Dict[int, Path] = {}
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        m = cfg["stan_split_re"].match(d.name)
        if m and d.is_dir() and m.group(2) == variant:
            runs[int(m.group(1))] = d
    return runs


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _add_series(ax, sizes_losses: Dict[int, Optional[float]], key: str) -> bool:
    """Plot one series onto ax. Returns True if anything was drawn."""
    pairs = [(s, l) for s, l in sorted(sizes_losses.items()) if l is not None]
    if not pairs:
        return False
    xs, ys = zip(*pairs)
    st = SERIES[key]
    ax.plot(xs, ys, marker=st["marker"], color=st["color"],
            linestyle=st["ls"], label=st["label"])
    return True


# ── Plot 1: Val log loss curve per Marformer run dir ──────────────────────────

def plot_val_curve(run_dir: Path, training_size: int, size_label: str) -> None:
    epochs, losses = load_val_curve(run_dir)
    if not epochs:
        print(f"    [skip] no val data in {run_dir.name}")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, losses, color=SERIES["MARFORMER"]["color"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Log Loss")
    ax.set_title(f"{size_label} = {training_size}")
    out = run_dir / "val_log_loss_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    saved → {out.relative_to(RESULTS_ROOT)}")


# ── Plots 2 / 3: vs training size ─────────────────────────────────────────────

def plot_vs_size(
    out_path: Path,
    data_by_series: Dict[str, Dict[int, Optional[float]]],
    title: str,
    size_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    any_data = False
    for key in ("MARFORMER", "MARFORMER_HARD_MASK", "STAN_Factor", "STAN_Normal"):
        if key in data_by_series:
            any_data |= _add_series(ax, data_by_series[key], key)

    if not any_data:
        plt.close(fig)
        print(f"    [skip] no data for {out_path.name}")
        return

    ax.set_xlabel(size_label)
    ax.set_ylabel("Log Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
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
        print("Stan-only plotting not yet implemented.")
        sys.exit(0)

    cfg        = DATASET_CONFIGS[dataset_key]
    size_label = cfg["size_label"]
    out_dir    = RESULTS_ROOT / "MARFORMER" / cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all runs ──────────────────────────────────────────────────────
    mf_runs   = {mt: find_mf_runs(mt, cfg) for mt in ("MARFORMER", "MARFORMER_HARD_MASK")}
    stan_runs = {v: find_stan_runs(v, cfg) for v in ("Factor", "Normal")}

    total = sum(len(v) for v in mf_runs.values()) + sum(len(v) for v in stan_runs.values())
    print(f"\n=== {dataset_key.upper()} — {total} run(s) found ===\n")

    # ── Plot 1: val curves (Marformer only) ───────────────────────────────────
    print("-- Plot 1: val log loss curves (Marformer) --")
    for mt, runs in mf_runs.items():
        for size, run_dir in sorted(runs.items()):
            print(f"  [{mt}] size={size}")
            plot_val_curve(run_dir, size, size_label)

    # ── Build per-series size→loss dicts ──────────────────────────────────────
    # Val (best across epochs)
    val_data: Dict[str, Dict[int, Optional[float]]] = {}
    for mt, runs in mf_runs.items():
        d = {}
        for size, run_dir in runs.items():
            _, losses = load_val_curve(run_dir)
            d[size] = min(losses) if losses else None
        val_data[mt] = d
    # STAN test loss also shown on val plot for reference
    for variant, runs in stan_runs.items():
        val_data[f"STAN_{variant}"] = {s: load_stan_loss(d) for s, d in runs.items()}

    # Test — best checkpoint
    test_best_data: Dict[str, Dict[int, Optional[float]]] = {}
    for mt, runs in mf_runs.items():
        test_best_data[mt] = {s: load_mf_test_loss(d, "best") for s, d in runs.items()}
    for variant, runs in stan_runs.items():
        test_best_data[f"STAN_{variant}"] = {s: load_stan_loss(d) for s, d in runs.items()}

    # Test — last checkpoint
    test_last_data: Dict[str, Dict[int, Optional[float]]] = {}
    for mt, runs in mf_runs.items():
        test_last_data[mt] = {s: load_mf_test_loss(d, "last") for s, d in runs.items()}
    for variant, runs in stan_runs.items():
        test_last_data[f"STAN_{variant}"] = {s: load_stan_loss(d) for s, d in runs.items()}

    # ── Plot 2: best val / STAN test log loss vs size ─────────────────────────
    print("\n-- Plot 2: best val log loss vs training size --")
    plot_vs_size(
        out_dir / "best_val_log_loss_vs_size.png",
        val_data,
        "Best Val Log Loss vs Training Size",
        size_label,
    )

    # ── Plot 3a: test log loss (best ckpt) vs size ────────────────────────────
    print("\n-- Plot 3: test log loss vs training size --")
    plot_vs_size(
        out_dir / "test_log_loss_best_vs_size.png",
        test_best_data,
        "Test Log Loss vs Training Size (Best Checkpoint)",
        size_label,
    )

    # ── Plot 3b: test log loss (last ckpt) vs size ────────────────────────────
    plot_vs_size(
        out_dir / "test_log_loss_last_vs_size.png",
        test_last_data,
        "Test Log Loss vs Training Size (Last Checkpoint)",
        size_label,
    )

    print(f"\nDone. Combined plots → RESULTS/MARFORMER/{cfg['subdir']}/")


if __name__ == "__main__":
    main()
