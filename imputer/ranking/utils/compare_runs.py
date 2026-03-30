#!/usr/bin/env python3
"""
Comparison plots: Marformer Pretrained vs Marformer Transductive vs Stan.

Produces (test missing only):
  test_missing_log_loss.png
  test_missing_accuracy.png

Usage:
    python utils/compare_runs.py \\
        --marformer-pretrained   OUTPUT/IMPUTER/run_a \\
        --marformer-transductive OUTPUT/IMPUTER/run_b \\
        --stan-eval              OUTPUT/domain_model/eval/.../predictive_metrics.json \\
        --output-dir             plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "pretrained":   "blue",
    "transductive": "red",
    "stan":         "green",
}
LABELS = {
    "pretrained":   "Marformer (pretrained only)",
    "transductive": "Marformer (transductive)",
    "stan":         "Stan (transductive)",
}


def load_history(run_dir: Path) -> list:
    path = run_dir / "test_instance_training_history.json"
    with open(path) as f:
        return json.load(f)


def load_stan(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marformer-pretrained",   default=None)
    parser.add_argument("--marformer-transductive", default=None)
    parser.add_argument("--stan-eval",              default=None)
    parser.add_argument("--output-dir",             default="plots")
    args = parser.parse_args()

    if not any([args.marformer_pretrained, args.marformer_transductive, args.stan_eval]):
        parser.error("Provide at least one data source.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    series = {}
    for key, path_arg in [("pretrained",   args.marformer_pretrained),
                           ("transductive", args.marformer_transductive)]:
        if path_arg is None:
            continue
        h = load_history(Path(path_arg))
        series[key] = {
            "epochs": [e["epoch"] for e in h],
            "loss":   [e["missing_metrics"]["rating_loss"]     for e in h],
            "acc":    [e["missing_metrics"]["rating_accuracy"] for e in h],
        }
        print(f"Loaded {key}: {len(h)} epochs from {path_arg}")

    stan = None
    if args.stan_eval:
        stan = load_stan(Path(args.stan_eval))
        print(f"Loaded Stan: acc={stan['rating_missing_accuracy']:.3f}  "
              f"NLL={stan['rating_missing_log_likelihood']:.3f}")

    # ── Log loss ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, s in series.items():
        ax.plot(s["epochs"], s["loss"], label=LABELS[key], linewidth=2, color=COLORS[key])
    if stan:
        val = -stan["rating_missing_log_likelihood"]
        ax.axhline(y=val, color=COLORS["stan"], linestyle=":", linewidth=2,
                   label=f"{LABELS['stan']}: {val:.3f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss (Missing)", fontsize=12)
    ax.set_title("Test Set: Rating Loss over Epochs (Missing Only)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "test_missing_log_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out / 'test_missing_log_loss.png'}")

    # ── Accuracy ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, s in series.items():
        ax.plot(s["epochs"], s["acc"], label=LABELS[key], linewidth=2, color=COLORS[key])
    if stan:
        val = stan["rating_missing_accuracy"]
        ax.axhline(y=val, color=COLORS["stan"], linestyle=":", linewidth=2,
                   label=f"{LABELS['stan']}: {val:.3f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Rating Accuracy (Missing)", fontsize=12)
    ax.set_title("Test Set: Rating Accuracy over Epochs (Missing Only)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(out / "test_missing_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out / 'test_missing_accuracy.png'}")


if __name__ == "__main__":
    main()
