#!/usr/bin/env python3
"""Compare flat Recurrent Marformer runs trained with different max_item values."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

_RANKING_ROOT = Path(__file__).resolve().parents[2]
_RUN_GLOB = "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
_TAG_RE = re.compile(r"RECURRENT_MF_(.+)$")
_EPOCH_RE = re.compile(r"periodic-epoch=(\d+)")

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


def _short_label(run_dir: Path) -> str:
    m = _TAG_RE.search(run_dir.name)
    return m.group(1) if m else run_dir.name


def _load_max_item(run_dir: Path) -> int | None:
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("training", {}).get("max_item")


def _load_training(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "training_history.json"
    if not path.exists():
        return [], []
    history = json.loads(path.read_text())
    epochs: list[int] = []
    values: list[float] = []
    for entry in history:
        ce = entry.get("combined_eval") or {}
        miss = (ce.get("metrics") or {}).get("missing") or {}
        rating = miss.get("rating") or {}
        xent = rating.get("xent")
        if xent is None or "epoch" not in entry:
            continue
        epochs.append(int(entry["epoch"]))
        values.append(float(xent))
    return epochs, values


def _load_test(run_dir: Path, max_item: int) -> tuple[list[int], list[float], list[float]]:
    test_dir = run_dir / f"TEST_RESULTS_MAXITEM{max_item}"
    if not test_dir.is_dir():
        return [], [], []
    epochs: list[int] = []
    log_losses: list[float] = []
    rmses: list[float] = []
    for path in sorted(test_dir.glob("periodic-epoch=*.json")):
        m = _EPOCH_RE.search(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        miss = data.get("missing") or {}
        ll = miss.get("log_loss")
        rmse = miss.get("rmse")
        if ll is None:
            continue
        epochs.append(int(m.group(1)))
        log_losses.append(float(ll))
        rmses.append(float(rmse) if rmse is not None else float("nan"))
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    return (
        [epochs[i] for i in order],
        [log_losses[i] for i in order],
        [rmses[i] for i in order],
    )


def _iter_runs(roots: Iterable[Path]) -> list[tuple[Path, int | None]]:
    out: list[tuple[Path, int | None]] = []
    for root in roots:
        for run_dir in sorted(root.glob(_RUN_GLOB)):
            if not run_dir.is_dir():
                continue
            out.append((run_dir, _load_max_item(run_dir)))
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_training(runs: list[tuple[Path, int | None]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    missing: list[str] = []
    for run_dir, max_item in runs:
        epochs, values = _load_training(run_dir)
        label = f"{_short_label(run_dir)} m{max_item}"
        if not epochs:
            missing.append(label)
            continue
        ax.plot(epochs, values, label=label, alpha=0.9)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print("No training histories found.")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined missing log loss (xent, nats)")
    ax.set_title("Flat Recurrent Marformer: max_item comparison")
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    if missing:
        ax.text(
            0.99,
            0.02,
            "missing history: " + ", ".join(missing),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.9),
        )
        print("missing training_history.json:", ", ".join(missing))
    fig.tight_layout()
    _save(fig, out_dir / "combined_missing_log_loss_flat_maxitem_comparison.png")


def plot_test(runs: list[tuple[Path, int | None]], out_dir: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for run_dir, max_item in runs:
        if max_item is None:
            continue
        epochs, lls, rmses = _load_test(run_dir, int(max_item))
        if not epochs:
            continue
        ys = lls if metric == "log_loss" else rmses
        ax.plot(epochs, ys, label=f"{_short_label(run_dir)} m{max_item}", alpha=0.9)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print(f"skip test {metric}: no TEST_RESULTS_MAXITEM* curves found")
        return
    ax.set_xlabel("Epoch (periodic checkpoint)")
    ax.set_ylabel("Test missing log loss (nats)" if metric == "log_loss" else "Test missing RMSE")
    ax.set_title(f"Flat Recurrent Marformer: test {metric} by max_item")
    if metric == "log_loss":
        ax.set_ylim(0.3, 0.8)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / f"global_test_{metric}_flat_maxitem_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300",
            "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500",
        ],
    )
    parser.add_argument(
        "--output-dir",
        default="PLOTS/TALK/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM-COMPARISON",
    )
    args = parser.parse_args()

    roots = [Path(p) if Path(p).is_absolute() else _RANKING_ROOT / p for p in args.roots]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _RANKING_ROOT / out_dir

    runs = _iter_runs(roots)
    print(f"found {len(runs)} runs")
    plot_training(runs, out_dir)
    plot_test(runs, out_dir, "log_loss")
    plot_test(runs, out_dir, "rmse")


if __name__ == "__main__":
    main()
