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
import colorsys
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


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
    "bothregscale":     "Item + Attr L2 + Shared Rel Scale",
    # best run replications
    "bestrun":          "Best Run",
    # section 1: feature-only layernorm experiments
    "lnbase":           "LN Base",
    "lnpointer":        "LN Pointer",
    "lnrelvalue":       "LN Rel Value",
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

    Handles four patterns:
      ablation_<key>_...   → LABEL_MAP lookup
      sweep_noperhead_...  → parse swept parameter from name
      multirun_<key>_runN_... → "Base Run N" or "Split-Norm Run N"
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
    if folder_name.startswith("final_") and len(parts) >= 2:
        _FINAL_LABELS = {
            "base":       "Base",
            "pointer":    "Pointer",
            "relvalue":   "Rel Value",
            "perheadrel": "Per-Head Rel",
            "addone":     "Add-One Attn",
            "combo124":   "Combo (Ptr+RelV+AddOne)",
            "itemreg":    "Item Reg",
            "attrreg":    "Attr Reg",
            "bothreg":    "Both Reg",
            "devnorm":    "Dev Norm",
            "all":        "All Together",
        }
        exp = parts[1]
        run_num = parts[-1][3:] if parts[-1].startswith("run") and parts[-1][3:].isdigit() else ""
        label = _FINAL_LABELS.get(exp, exp)
        return f"{label} Run {run_num}" if run_num else label
    if folder_name.startswith("lnexp_") and len(parts) >= 2:
        _LN_LABELS = {
            "base":     "LN Base",
            "pointer":  "LN Pointer",
            "relvalue": "LN Rel Value",
        }
        exp = parts[1]
        run_num = parts[-1][3:] if parts[-1].startswith("run") and parts[-1][3:].isdigit() else ""
        label = _LN_LABELS.get(exp, exp)
        return f"{label} Run {run_num}" if run_num else label
    if folder_name.startswith("best_run") and len(parts) >= 2:
        run_num = parts[1][3:] if parts[1].startswith("run") else ""
        return f"Best Run {run_num}" if run_num else "Best Run"
    if folder_name.startswith("multirun_") and len(parts) >= 3:
        variant = parts[1]   # "base" or "splitnorm"
        # Extract run number from "runN"
        run_num = ""
        for p in parts[2:]:
            if p.startswith("run") and p[3:].isdigit():
                run_num = p[3:]
                break
        suffix = f" Run {run_num}" if run_num else ""
        if variant == "base":
            return f"Base{suffix}"
        return f"{variant}{suffix}"
    return folder_name


# ── Experiment color helpers ───────────────────────────────────────────────────

def _get_exp_name(label: str) -> str:
    """Strip trailing ' Run N' to get experiment family name."""
    return re.sub(r'\s+Run\s+\d+$', '', label)


def _get_run_num(label: str) -> int:
    """Extract run number from label, default 1 if not found."""
    m = re.search(r'\s+Run\s+(\d+)$', label)
    return int(m.group(1)) if m else 1


def _make_shade(hex_color: str, run_num: int) -> str:
    """Lighter/darker shade of hex_color: run 1=darkest, run 2=base, run 3=lightest."""
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    factors = {1: 0.70, 2: 1.0, 3: 1.38}
    new_l = min(0.88, max(0.15, l * factors.get(run_num, 1.0)))
    r2, g2, b2 = colorsys.hls_to_rgb(h, new_l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


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

    # ── Assign one base color per experiment family ────────────────────────────
    exp_names: list[str] = []
    for run in runs:
        name = _get_exp_name(run["label"])
        if name not in exp_names:
            exp_names.append(name)
    exp_color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(exp_names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Entity Marformer — Run Comparison (Test, Missing Ratings)", fontsize=13)

    ax_loss, ax_acc = axes

    for run in runs:
        exp_name = _get_exp_name(run["label"])
        run_num  = _get_run_num(run["label"])
        color    = _make_shade(exp_color[exp_name], run_num)
        is_base  = exp_name == LABEL_MAP.get(BASE_KEY, "Base")
        lw    = 2.2 if is_base else 1.6
        alpha = 1.0 if is_base else 0.85

        ax_loss.plot(run["epochs"], run["xent"], color=color, linewidth=lw, alpha=alpha)
        ax_acc.plot(run["epochs"],  run["acc"],  color=color, linewidth=lw, alpha=alpha)

    # ── Loss plot ──────────────────────────────────────────────────────────────
    ax_loss.set_title("Test CE Loss (Missing)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy plot ──────────────────────────────────────────────────────────
    ax_acc.set_title("Test Accuracy (Missing)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(bottom=0.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Single figure-level legend outside the plots (right side) ─────────────
    legend_handles = [
        Line2D([0], [0], color=exp_color[name], linewidth=2.0, label=name)
        for name in exp_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()
    fig.subplots_adjust(right=0.86)
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
