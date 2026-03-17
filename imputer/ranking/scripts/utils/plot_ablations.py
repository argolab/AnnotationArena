#!/usr/bin/env python3
"""
Plot ablation comparison for Entity Marformer runs.

Scans OUTPUT/ENTITY_MF (or any given root) for training_history.json files,
then produces two plots:
  1. Test CE Loss  (missing ratings)
  2. Test Accuracy (missing ratings)

Usage:
    python scripts/utils/plot_ablations.py OUTPUT/ENTITY_MF
    python scripts/utils/plot_ablations.py OUTPUT/ENTITY_MF --out plots/ablations.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Known ablation labels ──────────────────────────────────────────────────────
# Maps the short key extracted from the folder name to a human-readable legend label.
# Folder name pattern: ablation_<key>_<rest>
LABEL_MAP = {
    "base":       "Base",
    "noperhead":  "No Per-Head Rel",
    "noptr":      "No Pointer",
    "norelv":     "No Rel Value",
    "llmhard":    "LLM Hard",
    "human02":    "Human 0.2",
}

# Base always drawn first and with a distinct style
BASE_KEY = "base"

# Color cycle — visually distinct for up to 6 runs
COLORS = [
    "#1f77b4",  # blue     — Base
    "#d62728",  # red      — No Per-Head Rel
    "#ff7f0e",  # orange   — No Pointer
    "#2ca02c",  # green    — No Rel Value
    "#9467bd",  # purple   — LLM Hard
    "#8c564b",  # brown    — Human 0.2
]


def _extract_label(folder_name: str) -> str:
    """
    Extract a human-readable legend label from a run folder name.

    Tries LABEL_MAP first; falls back to the raw ablation key; then the folder name.
    """
    # Pattern: ablation_<key>_...
    if folder_name.startswith("ablation_"):
        parts = folder_name.split("_")
        if len(parts) >= 2:
            key = parts[1]
            if key in LABEL_MAP:
                return LABEL_MAP[key]
            return key  # unknown ablation — use raw key
    return folder_name  # not an ablation run — use full name


def _sort_key(label: str) -> int:
    """Sort order: Base first, then alphabetical."""
    order = list(LABEL_MAP.values())
    try:
        return order.index(label)
    except ValueError:
        return len(order)


def load_runs(root: Path) -> list[dict]:
    """
    Recursively find all training_history.json files under root.
    Returns a list of dicts: {label, epochs, xent, acc}
    """
    runs = []
    for history_path in sorted(root.rglob("training_history.json")):
        folder_name = history_path.parent.name
        label = _extract_label(folder_name)

        with open(history_path, "r") as f:
            history = json.load(f)

        epochs, xents, accs = [], [], []
        for entry in history:
            ep = entry.get("epoch")
            test_eval = entry.get("test_eval", {})
            missing = test_eval.get("metrics", {}).get("missing", {}).get("rating", {})
            xent = missing.get("xent", None)
            acc  = missing.get("acc",  None)
            if ep is not None and xent is not None and acc is not None:
                epochs.append(ep)
                xents.append(xent)
                accs.append(acc)

        if not epochs:
            print(f"  Warning: no test missing data found in {history_path}, skipping.")
            continue

        runs.append({"label": label, "folder": folder_name, "epochs": epochs, "xent": xents, "acc": accs})

    # Sort: Base first, then by LABEL_MAP order
    runs.sort(key=lambda r: _sort_key(r["label"]))
    return runs


def plot(runs: list[dict], out_path: Path) -> None:
    if not runs:
        print("No runs found — nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Entity Marformer — Ablation Comparison (Test, Missing Ratings)", fontsize=13)

    ax_loss, ax_acc = axes

    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        lw = 2.2 if run["label"] == LABEL_MAP.get(BASE_KEY, "Base") else 1.6
        alpha = 1.0 if run["label"] == LABEL_MAP.get(BASE_KEY, "Base") else 0.85

        ax_loss.plot(run["epochs"], run["xent"], label=run["label"],
                     color=color, linewidth=lw, alpha=alpha)
        ax_acc.plot(run["epochs"],  run["acc"],  label=run["label"],
                    color=color, linewidth=lw, alpha=alpha)

    # ── Loss plot ──────────────────────────────────────────────────────────────
    ax_loss.set_title("Test CE Loss (Missing)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy plot ──────────────────────────────────────────────────────────
    ax_acc.set_title("Test Accuracy (Missing)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(bottom=0.0)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Entity Marformer ablation comparisons.")
    parser.add_argument(
        "output_root",
        type=Path,
        help="Root directory containing Entity Marformer run folders (e.g. OUTPUT/ENTITY_MF).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the plot PNG (default: <output_root>/ablation_comparison.png).",
    )
    args = parser.parse_args()

    root = args.output_root
    if not root.exists():
        raise FileNotFoundError(f"Output root not found: {root}")

    out_path = args.out if args.out is not None else root / "ablation_comparison.png"

    print(f"Scanning {root} for training_history.json files...")
    runs = load_runs(root)

    if runs:
        print(f"Found {len(runs)} run(s):")
        for r in runs:
            print(f"  [{r['label']}]  {len(r['epochs'])} epochs  —  {r['folder']}")
    else:
        print("No valid runs found.")
        return

    plot(runs, out_path)


if __name__ == "__main__":
    main()
