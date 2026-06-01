#!/usr/bin/env python3
"""Ad hoc comparison for flat 8/12/16-layer runs at different max_item."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]

# (run_dir, test_results_subdir)
TEST_RUNS = [
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
        "TEST_RESULTS",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
        "TEST_RESULTS_MAXITEM300",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
        "TEST_RESULTS",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
        "TEST_RESULTS_MAXITEM300",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
        "TEST_RESULTS_MAXITEM500",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
        "TEST_RESULTS_MAXITEM500",
    ),
    (
        ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
        "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c16r1c0",
        "TEST_RESULTS_MAXITEM500",
    ),
]

# Training histories to overlay. Some of these may be configs only if a run failed
# before writing training_history.json; those are skipped with a printed message.
TRAIN_RUNS = [
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0",
    ROOT / "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM500/"
    "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c16r1c0",
]

OUT_DIR = ROOT / "PLOTS/TALK/RECURRENT_MARFORMER/ADHOC-FLAT-8-12-MAXITEM-COMPARE"
OUT_TRAIN_PATH = OUT_DIR / "combined_missing_log_loss_8_12_16_maxitem_compare.png"
OUT_LOG_LOSS_PATH = OUT_DIR / "test_missing_log_loss_8_12_16_maxitem_compare.png"
OUT_RMSE_PATH = OUT_DIR / "test_missing_rmse_8_12_16_maxitem_compare.png"

_EPOCH_RE = re.compile(r"periodic-epoch=(\d+)")

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 8,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _epoch_from_name(path: Path) -> int | None:
    m = _EPOCH_RE.search(path.name)
    return int(m.group(1)) if m else None


def load_test_curve(
    run_dir: Path,
    subdir: str,
    *,
    metric: str,
) -> tuple[list[int], list[float], int | None]:
    test_dir = run_dir / subdir
    by_epoch: dict[int, Path] = {}
    for path in sorted(test_dir.glob("periodic-epoch=*.json")):
        epoch = _epoch_from_name(path)
        if epoch is not None:
            by_epoch[epoch] = path

    epochs: list[int] = []
    values: list[float] = []
    eval_max_item: int | None = None
    for epoch, path in sorted(by_epoch.items()):
        data = json.loads(path.read_text())
        if eval_max_item is None:
            eval_max_item = data.get("eval_max_item", data.get("max_item"))
        missing = data.get("missing") or {}
        value = missing.get(metric)
        if value is None:
            continue
        epochs.append(epoch)
        values.append(float(value))
    return epochs, values, eval_max_item


def load_training_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "training_history.json"
    if not path.exists():
        return [], []

    history = json.loads(path.read_text())
    epochs: list[int] = []
    values: list[float] = []
    for entry in history:
        if "epoch" not in entry:
            continue
        ce = entry.get("combined_eval") or {}
        miss = (ce.get("metrics") or {}).get("missing") or {}
        rating = miss.get("rating") or {}
        xent = rating.get("xent")
        if xent is None:
            continue
        epochs.append(int(entry["epoch"]))
        values.append(float(xent))
    return epochs, values


def label_for(run_dir: Path, eval_max_item: int | None = None) -> str:
    cfg = json.loads((run_dir / "train_config.json").read_text())
    model = cfg["model"]
    training = cfg["training"]
    layers = int(model["effective_depth"])
    train_max_item = training.get("max_item")
    if eval_max_item is None:
        return f"{layers} layers, train max_item={train_max_item}"
    return f"{layers} layers, train max_item={train_max_item}, eval max_item={eval_max_item}"


def plot_training(runs: Iterable[Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    found = 0
    for run_dir in runs:
        if not run_dir.is_dir():
            print(f"skip missing run dir: {run_dir}")
            continue
        epochs, values = load_training_curve(run_dir)
        if not epochs:
            print(f"skip missing/empty training curve: {run_dir / 'training_history.json'}")
            continue
        ax.plot(epochs, values, label=label_for(run_dir), alpha=0.9)
        found += 1

    if found == 0:
        print("skip training plot: no curves found")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined missing log loss (xent, nats)")
    ax.set_title("Flat 8/12/16-layer training trace by max_item")
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TRAIN_PATH)
    plt.close(fig)
    print(f"wrote {OUT_TRAIN_PATH}")


def plot_metric(
    runs: Iterable[tuple[Path, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    found = 0
    for run_dir, subdir in runs:
        if not (run_dir / subdir).is_dir():
            print(f"skip missing test dir: {run_dir / subdir}")
            continue
        epochs, values, eval_max_item = load_test_curve(run_dir, subdir, metric=metric)
        if not epochs:
            print(f"skip empty test curve: {run_dir / subdir}")
            continue
        ax.plot(epochs, values, "o-", label=label_for(run_dir, eval_max_item), alpha=0.9)
        found += 1

    if found == 0:
        raise SystemExit("No curves found")

    ax.set_xlabel("Epoch (periodic checkpoint)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric == "log_loss":
        ax.set_ylim(0.3, 0.8)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot(runs: Iterable[tuple[Path, str]]) -> None:
    run_list = list(runs)
    plot_metric(
        run_list,
        metric="log_loss",
        ylabel="Test missing log loss (nats)",
        title="Flat 8/12/16-layer test log loss by training/eval max_item",
        out_path=OUT_LOG_LOSS_PATH,
    )
    plot_metric(
        run_list,
        metric="rmse",
        ylabel="Test missing RMSE",
        title="Flat 8/12/16-layer test RMSE by training/eval max_item",
        out_path=OUT_RMSE_PATH,
    )


if __name__ == "__main__":
    plot_training(TRAIN_RUNS)
    plot(TEST_RUNS)
