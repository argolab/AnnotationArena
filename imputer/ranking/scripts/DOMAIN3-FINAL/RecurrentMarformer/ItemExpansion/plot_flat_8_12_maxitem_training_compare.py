#!/usr/bin/env python3
"""Ad hoc training-curve comparison for flat 8/12-layer runs at different max_item."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]

RUNS = [
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
]

OUT_DIR = ROOT / "PLOTS/TALK/RECURRENT_MARFORMER/ADHOC-FLAT-8-12-MAXITEM-COMPARE"
OUT_PATH = OUT_DIR / "combined_missing_log_loss_8_12_maxitem_compare.png"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_training_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    history = json.loads((run_dir / "training_history.json").read_text())
    epochs: list[int] = []
    values: list[float] = []
    for entry in history:
        if "epoch" not in entry:
            continue
        combined_eval = entry.get("combined_eval") or {}
        missing = (combined_eval.get("metrics") or {}).get("missing") or {}
        rating = missing.get("rating") or {}
        xent = rating.get("xent")
        if xent is None:
            continue
        epochs.append(int(entry["epoch"]))
        values.append(float(xent))
    return epochs, values


def label_for(run_dir: Path) -> str:
    cfg = json.loads((run_dir / "train_config.json").read_text())
    model = cfg["model"]
    training = cfg["training"]
    layers = int(model["effective_depth"])
    max_item = training.get("max_item")
    epochs = training.get("epochs")
    return f"{layers} layers, train max_item={max_item} ({epochs} epochs)"


def plot(runs: Iterable[Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    found = 0
    for run_dir in runs:
        if not (run_dir / "training_history.json").exists():
            print(f"skip missing history: {run_dir}")
            continue
        epochs, values = load_training_curve(run_dir)
        if not epochs:
            print(f"skip empty curve: {run_dir}")
            continue
        ax.plot(epochs, values, label=label_for(run_dir), alpha=0.9)
        found += 1

    if found == 0:
        raise SystemExit("No curves found")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined missing log loss (xent, nats)")
    ax.set_title("Flat 8/12-layer training curves by training max_item")
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    plot(RUNS)
