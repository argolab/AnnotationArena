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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Known ablation labels ──────────────────────────────────────────────────────
# Maps the short key extracted from the folder name to a human-readable legend label.
# Folder name pattern: ablation_<key>_<rest>
LABEL_MAP = {
    "base":         "Base",
    "noperhead":    "No Per-Head Rel",
    "noptr":        "No Pointer",
    "norelv":       "No Rel Value",
    "llmhard":      "LLM Hard",
    "human02":      "Human 0.2",
    "drop09":       "Item Drop 0.9",
    "normalinit":   "Normal Init",
    "scalednormal": "Scaled Normal Init",
    "itemreg":      "Item L2 Reg",
    "attrreg":      "Attr L2 Reg",
    "bothreg":      "Item + Attr L2 Reg",
    "devnorm":          "Deviation Norm",
    "cosine":           "Cosine LR (2e-3→1e-5)",
    # ablations_reg_2
    "allreg":           "Item + Attr + Annot L2 Reg",
    "bothregdevnorm":   "Item + Attr L2 + Dev Norm",
    "cosinebothreg":    "Cosine LR + Item + Attr L2",
}

# Base always drawn first and with a distinct style
BASE_KEY = "base"

# Color cycle — visually distinct for up to 9 runs
COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#17becf",  # cyan
    "#bcbd22",  # yellow-green
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
]


# ── Sweep label extraction ─────────────────────────────────────────────────────
# Base values for the No Per-Head Rel sweep — runs matching all defaults are
# labelled "Sweep Base". Folder name pattern: sweep_noperhead_<NL>L<NH>H_..._ffn<F>_..._itemdrop<D>_..._lr<LR>_...
_SWEEP_BASE = {"layers": 6, "ffn": 1, "itemdrop": 0.7, "lr": "2e-4"}


def _extract_sweep_label(folder_name: str) -> str:
    """Parse a sweep_noperhead_... folder name and return the swept parameter as a label."""
    layers_m = re.search(r"_(\d+)L\d+H_",      folder_name)
    ffn_m    = re.search(r"_ffn(\d+)_",         folder_name)
    idrop_m  = re.search(r"_itemdrop([\d.]+)_", folder_name)
    lr_m     = re.search(r"_lr([^_]+)_",        folder_name)

    layers   = int(layers_m.group(1))   if layers_m else _SWEEP_BASE["layers"]
    ffn      = int(ffn_m.group(1))      if ffn_m    else _SWEEP_BASE["ffn"]
    itemdrop = float(idrop_m.group(1))  if idrop_m  else _SWEEP_BASE["itemdrop"]
    lr       = lr_m.group(1)            if lr_m     else _SWEEP_BASE["lr"]

    parts = []
    if layers != _SWEEP_BASE["layers"]:
        parts.append(f"{layers} Layers")
    if ffn != _SWEEP_BASE["ffn"]:
        parts.append(f"{ffn} FFN Layers")
    if abs(itemdrop - _SWEEP_BASE["itemdrop"]) > 1e-6:
        parts.append(f"Item Drop {itemdrop}")
    if lr != _SWEEP_BASE["lr"]:
        parts.append(f"LR {lr}")

    return ", ".join(parts) if parts else "Sweep Base"


def _extract_label(folder_name: str) -> str:
    """
    Extract a human-readable legend label from a run folder name.

    Handles three patterns:
      ablation_<key>_... → LABEL_MAP lookup
      sweep_noperhead_...  → parse swept parameter from name
      anything else        → full folder name (fallback)
    """
    parts = folder_name.split("_")
    if parts[0].startswith("ablation") and len(parts) >= 2:
        key = parts[1]
        if key in LABEL_MAP:
            return LABEL_MAP[key]
        return key
    if folder_name.startswith("sweep_"):
        return _extract_sweep_label(folder_name)
    return folder_name


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
    fig.suptitle("Entity Marformer — Run Comparison (Test, Missing Ratings)", fontsize=13)

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

    out_path = args.out if args.out is not None else root / "comparison.png"

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
