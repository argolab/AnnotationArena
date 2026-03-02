#!/usr/bin/env python3
"""
Plot Train and Test Log Loss (cross-entropy on held-out missing ratings)
for all 6 Marformer runs from scripts/all_datagen_marformer.sh.

Produces two figures:
  1. Train Log Loss (missing_metrics.rating_loss from train_instance)
  2. Test  Log Loss (missing_metrics.rating_loss from test_instance)

Usage:
    PYTHONPATH=. python scripts/plot_all_6_marformer.py
    PYTHONPATH=. python scripts/plot_all_6_marformer.py --output-dir plots/
    PYTHONPATH=. python scripts/plot_all_6_marformer.py --metric rating_loss
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Run definitions ────────────────────────────────────────────────────────────
RUNS = [
    # (run_name,               display_label,          color,    linestyle)
    ("spherical_gauss_marformer_e72", "Spherical Gauss", "#1f77b4", "-"),
    ("spherical_exp_marformer_e72",   "Spherical Exp",   "#1f77b4", "--"),
    ("mmatrix_gauss_marformer_e72",   "M-matrix Gauss",  "#ff7f0e", "-"),
    ("mmatrix_exp_marformer_e72",     "M-matrix Exp",    "#ff7f0e", "--"),
    ("tcp_exp_marformer_e72",         "TCP Exp",         "#2ca02c", "--"),
]

# Color groups data-gen method; solid=Gaussian, dashed=Exp


def load_history(run_dir: Path, filename: str):
    path = run_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_curve(history, metric_path: str):
    """
    Extract (epochs, values) from a history list.
    metric_path supports nested keys with '/', e.g. 'missing_metrics/rating_loss'.
    """
    keys = metric_path.split("/")
    epochs, values = [], []
    for entry in history:
        v = entry
        try:
            for k in keys:
                v = v[k]
        except (KeyError, TypeError):
            continue
        if v is None:
            continue
        epochs.append(entry["epoch"])
        values.append(float(v))
    return epochs, values


def plot_curves(ax, runs_data, title, ylabel):
    for label, color, ls, epochs, values in runs_data:
        if not epochs:
            ax.plot([], [], color=color, linestyle=ls, label=f"{label} (no data)")
            continue
        ax.plot(epochs, values, color=color, linestyle=ls, linewidth=1.8,
                label=label, alpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))


def main():
    parser = argparse.ArgumentParser(description="Plot train/test log loss for 6 Marformer runs")
    parser.add_argument(
        "--imputer-dir",
        type=str,
        default="OUTPUT/IMPUTER",
        help="Base directory containing all Marformer run sub-directories (default: OUTPUT/IMPUTER)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots as PNG (default: show interactively)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="missing_metrics/rating_loss",
        help=(
            "Metric to plot (nested keys separated by '/'). "
            "Default: 'missing_metrics/rating_loss' — cross-entropy on truly missing ratings. "
            "Other options: 'rating_loss' (total), 'observed_metrics/rating_loss'."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful for headless environments)",
    )
    args = parser.parse_args()

    base_dir = Path(args.imputer_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load histories ─────────────────────────────────────────────────────────
    train_runs_data = []
    test_runs_data  = []
    missing_runs = []

    for run_name, label, color, ls in RUNS:
        run_dir = base_dir / run_name

        if not run_dir.exists():
            print(f"  [WARN] Not found: {run_dir}")
            missing_runs.append(run_name)
            train_runs_data.append((label, color, ls, [], []))
            test_runs_data.append((label, color, ls, [], []))
            continue

        train_hist = load_history(run_dir, "train_instance_training_history.json")
        if train_hist is None:
            print(f"  [WARN] No train history in {run_dir}")
            train_runs_data.append((label, color, ls, [], []))
        else:
            epochs, values = extract_curve(train_hist, args.metric)
            train_runs_data.append((label, color, ls, epochs, values))
            print(f"  [OK] {label:25s} train: {len(epochs)} epochs, last={values[-1]:.4f}" if values else f"  [OK] {label}: no values")

        test_hist = load_history(run_dir, "test_instance_training_history.json")
        if test_hist is None:
            print(f"  [WARN] No test history in {run_dir}")
            test_runs_data.append((label, color, ls, [], []))
        else:
            epochs, values = extract_curve(test_hist, args.metric)
            test_runs_data.append((label, color, ls, epochs, values))
            print(f"  [OK] {label:25s} test:  {len(epochs)} epochs, last={values[-1]:.4f}" if values else f"  [OK] {label}: no values")

    if missing_runs:
        print(f"\n[INFO] {len(missing_runs)} run(s) not found (runs may not be complete yet):")
        for r in missing_runs:
            print(f"       {r}")

    metric_label_map = {
        "missing_metrics/rating_loss":  "Log Loss on Missing Ratings (Cross-Entropy)",
        "rating_loss":                  "Total Rating Log Loss (Cross-Entropy)",
        "observed_metrics/rating_loss": "Log Loss on Observed Ratings (Cross-Entropy)",
    }
    ylabel = metric_label_map.get(args.metric, f"Log Loss [{args.metric}]")

    # ── Figure 1: Train ────────────────────────────────────────────────────────
    fig_train, ax_train = plt.subplots(figsize=(9, 5))
    plot_curves(ax_train, train_runs_data,
                title="Train Log Loss — 6 Data Generation Methods", ylabel=ylabel)
    fig_train.tight_layout()
    if output_dir is not None:
        p = output_dir / "train_log_loss_all6.png"
        fig_train.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {p}")

    # ── Figure 2: Test ─────────────────────────────────────────────────────────
    fig_test, ax_test = plt.subplots(figsize=(9, 5))
    plot_curves(ax_test, test_runs_data,
                title="Test Log Loss — 6 Data Generation Methods", ylabel=ylabel)
    fig_test.tight_layout()
    if output_dir is not None:
        p = output_dir / "test_log_loss_all6.png"
        fig_test.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")

    print("\nLegend: color=method (blue=Spherical, orange=M-matrix, green=TCP) | style=dist (solid=Gauss, dashed=Exp)")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
