#!/usr/bin/env python3
"""
Training and validation missing loss curves for a single Marformer run.

Reads training_history.json and plots:
  - Combined missing CE  (train+val combined, epoch-end eval)
  - Val missing CE       (val only, epoch-end eval)

Usage (run from imputer/ranking):
    python scripts/utils/plot_sparse_tensor_test_loss.py \
        --run-dir RESULTS/MARFORMER/STAN/SPARSE/Tensor_100_25_9_ItemTest_10_CLUSTER_NOITEMDEV_TRANS

Output: PLOTS/TALK/<run_name>_loss_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    "lines.markersize":  4,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "training_history.json"
    history = json.load(open(history_path))

    epochs, combined_miss, val_miss = [], [], []
    for entry in history:
        e = entry["epoch"]
        cm = entry.get("combined_eval", {}).get("metrics", {}).get("missing", {}).get("rating", {}).get("xent")
        vm = entry.get("val_eval", {}).get("metrics", {}).get("missing", {}).get("rating", {}).get("xent")
        if cm is not None and vm is not None:
            epochs.append(e)
            combined_miss.append(cm)
            val_miss.append(vm)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, combined_miss, color="#1f6fba", label="Combined Missing CE (train+val)")
    ax.plot(epochs, val_miss,     color="#e67e22", label="Val Missing CE")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Missing Cross-Entropy")
    ax.set_title(run_dir.name)
    ax.legend()

    out = Path("PLOTS/TALK") / f"{run_dir.name}_loss_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
