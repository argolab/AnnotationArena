#!/usr/bin/env python3
"""
Visualize a Recurrent Marformer DOMAIN3 sweep (training + test + recurrence scaling).

Mirrors analysis under RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD:
  - combined missing log loss (transductive combined_eval xent) per run + overlay
  - global test log loss / RMSE from periodic TEST_RESULTS checkpoints
  - aggregate recurrence-at-eval curves (log loss + RMSE)

Run from imputer/ranking:

  python scripts/utils/plot_recurrent_marformer_domain3_sweep.py \\
      --results-root RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12

  python scripts/utils/plot_recurrent_marformer_domain3_sweep.py --all-sweeps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

_RANKING_ROOT = Path(__file__).resolve().parents[2]
_RUN_GLOB = "DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
_TAG_RE = re.compile(r"RECURRENT_MF_(.+)$")

_DEFAULT_SWEEPS = (
    "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12",
    "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-P0C1RX",
    "RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE8-DEEP",
)

_TEST_LOG_LOSS_YLIM = (0.3, 0.8)
_COMBINED_MISSING_LOG_LOSS_YLIM = (0.4, 0.8)

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
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _short_label(run_dir: Path) -> str:
    m = _TAG_RE.search(run_dir.name)
    return m.group(1) if m else run_dir.name


def discover_runs(results_root: Path) -> List[Path]:
    runs = sorted(
        d for d in results_root.glob(_RUN_GLOB)
        if d.is_dir() and (d / "training_history.json").exists()
    )
    return runs


def load_training_missing_xent(run_dir: Path) -> Tuple[List[int], List[float]]:
    history = json.loads((run_dir / "training_history.json").read_text())
    epochs: List[int] = []
    values: List[float] = []
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


def _parse_periodic_epoch(name: str) -> Optional[int]:
    m = re.search(r"periodic-epoch=(\d+)", name)
    return int(m.group(1)) if m else None


def load_test_periodic_metrics(
    run_dir: Path,
) -> Tuple[List[int], List[float], List[float]]:
    test_dir = run_dir / "TEST_RESULTS"
    if not test_dir.is_dir():
        return [], [], []

    epochs: List[int] = []
    log_losses: List[float] = []
    rmses: List[float] = []

    for path in sorted(test_dir.glob("periodic-epoch=*.json")):
        epoch = _parse_periodic_epoch(path.name)
        if epoch is None:
            continue
        data = json.loads(path.read_text())
        miss = data.get("missing") or {}
        ll = miss.get("log_loss")
        rmse = miss.get("rmse")
        if ll is None:
            continue
        epochs.append(epoch)
        log_losses.append(float(ll))
        rmses.append(float(rmse) if rmse is not None else float("nan"))

    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in order]
    log_losses = [log_losses[i] for i in order]
    rmses = [rmses[i] for i in order]
    return epochs, log_losses, rmses


def load_recurrence_scaling(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "RECURRENCE_SCALING" / "recurrence_scaling.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_combined_missing_log_loss(
    runs: Sequence[Path],
    out_path: Path,
    *,
    title: str,
    also_per_run: bool,
) -> None:
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False
    for run_dir in runs:
        epochs, values = load_training_missing_xent(run_dir)
        if not epochs:
            continue
        label = _short_label(run_dir)
        ax.plot(epochs, values, label=label, alpha=0.9)
        any_curve = True
        if also_per_run:
            fig_one, ax_one = plt.subplots(figsize=(7, 4.5))
            ax_one.plot(epochs, values, color="#1f77b4", alpha=0.9)
            ax_one.set_xlabel("Epoch")
            ax_one.set_ylabel("Combined missing log loss (xent, nats)")
            ax_one.set_title(f"{label} — training combined_eval missing rating xent")
            ax_one.set_ylim(*_COMBINED_MISSING_LOG_LOSS_YLIM)
            per_path = run_dir / "combined_missing_log_loss.png"
            _save(fig_one, per_path)

    if not any_curve:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined missing log loss (xent, nats)")
    ax.set_title(title)
    ax.set_ylim(*_COMBINED_MISSING_LOG_LOSS_YLIM)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_global_test_metric(
    runs: Sequence[Path],
    out_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False
    for run_dir in runs:
        epochs, log_losses, rmses = load_test_periodic_metrics(run_dir)
        if not epochs:
            continue
        values = log_losses if metric == "log_loss" else rmses
        if all(v != v for v in values):  # all NaN
            continue
        ax.plot(epochs, values, label=_short_label(run_dir), alpha=0.9)
        any_curve = True

    if not any_curve:
        plt.close(fig)
        print(f"  (skip {out_path.name}: no periodic TEST_RESULTS)")
        return

    ax.set_xlabel("Epoch (periodic checkpoint)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric == "log_loss":
        ax.set_ylim(*_TEST_LOG_LOSS_YLIM)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_recurrence_scaling_aggregate(
    runs: Sequence[Path],
    out_dir: Path,
    *,
    sweep_name: str,
) -> None:
    series: List[Tuple[str, List[int], List[float], List[float], Optional[int]]] = []
    for run_dir in runs:
        summary = load_recurrence_scaling(run_dir)
        if not summary:
            continue
        rows = summary.get("results") or []
        if not rows:
            continue
        rs = [int(r["num_recurrence"]) for r in rows]
        lls = [float(r["missing"]["log_loss"]) for r in rows]
        rmses = [float(r["missing"]["rmse"]) for r in rows]
        trained = int(rows[0].get("trained_num_recurrence", rs[-1]))
        series.append((_short_label(run_dir), rs, lls, rmses, trained))

    if not series:
        print("  (skip recurrence scaling: no RECURRENCE_SCALING/*.json)")
        return

    for metric, fname, ylabel in (
        ("log_loss", "recurrence_scaling_log_loss.png", "Test missing log loss (nats)"),
        ("rmse", "recurrence_scaling_rmse.png", "Test missing RMSE"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rs, lls, rmses, trained in series:
            ys = lls if metric == "log_loss" else rmses
            ax.plot(rs, ys, "o-", label=label, alpha=0.85)
            if trained in rs:
                i = rs.index(trained)
                ax.scatter([trained], [ys[i]], s=80, zorder=5)

        ax.set_xlabel("num_recurrence at eval")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sweep_name} — recurrence scaling ({metric})")
        if metric == "log_loss":
            ax.set_ylim(*_TEST_LOG_LOSS_YLIM)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        _save(fig, out_dir / fname)

    # Per-run recurrence plots with log loss + RMSE (upgrade from eval-only log loss)
    for run_dir in runs:
        summary = load_recurrence_scaling(run_dir)
        if not summary:
            continue
        rows = summary.get("results") or []
        if not rows:
            continue
        rs = [int(r["num_recurrence"]) for r in rows]
        lls = [float(r["missing"]["log_loss"]) for r in rows]
        rmses = [float(r["missing"]["rmse"]) for r in rows]
        trained = int(rows[0].get("trained_num_recurrence", rs[-1]))
        cfg = summary.get("trained_config", _short_label(run_dir))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, ys, ylab, is_log_loss in zip(
            axes, (lls, rmses), ("log loss (nats)", "RMSE"), (True, False)
        ):
            ax.plot(rs, ys, "o-", color="#1f77b4", linewidth=2, markersize=7, alpha=0.85)
            if trained in rs:
                i = rs.index(trained)
                ax.scatter([trained], [ys[i]], color="#d62728", s=100, zorder=5, label=f"trained r={trained}")
            ax.set_xlabel("num_recurrence at eval")
            ax.set_ylabel(f"Test missing {ylab}")
            if is_log_loss:
                ax.set_ylim(*_TEST_LOG_LOSS_YLIM)
            ax.grid(True, alpha=0.3)
            if trained in rs:
                ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"Recurrence scaling — {cfg} ({summary.get('checkpoint', '')} weights)")
        fig.tight_layout()
        out = run_dir / "RECURRENCE_SCALING" / "recurrence_scaling.png"
        _save(fig, out)


def plot_sweep(results_root: Path, output_dir: Optional[Path], *, per_run: bool) -> None:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        print(f"Missing results root: {results_root}")
        return

    runs = discover_runs(results_root)
    if not runs:
        print(f"No runs with training_history.json under {results_root}")
        return

    sweep_name = results_root.name
    out_dir = (output_dir or (_RANKING_ROOT / "PLOTS/TALK/RECURRENT_MARFORMER" / sweep_name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sweep_name} ({len(runs)} runs) -> {out_dir} ===")

    plot_combined_missing_log_loss(
        runs,
        out_dir / "combined_missing_log_loss.png",
        title=f"{sweep_name} — combined missing log loss (training)",
        also_per_run=per_run,
    )

    plot_global_test_metric(
        runs,
        out_dir / "global_test_log_loss.png",
        metric="log_loss",
        title=f"{sweep_name} — test missing log loss (periodic eval)",
        ylabel="Test missing log loss (nats)",
    )
    plot_global_test_metric(
        runs,
        out_dir / "global_test_rmse.png",
        metric="rmse",
        title=f"{sweep_name} — test missing RMSE (periodic eval)",
        ylabel="Test missing RMSE",
    )

    plot_recurrence_scaling_aggregate(runs, out_dir, sweep_name=sweep_name)

    # Mirror sweep-level training curve into RESULTS root (DOMAIN3-OLD convention)
    plot_combined_missing_log_loss(
        runs,
        results_root / "combined_missing_log_loss.png",
        title=f"{sweep_name} — combined missing log loss (training)",
        also_per_run=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Sweep directory, e.g. RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: PLOTS/TALK/RECURRENT_MARFORMER/<sweep_name>",
    )
    parser.add_argument(
        "--all-sweeps",
        action="store_true",
        help=f"Plot all default sweeps: {', '.join(_DEFAULT_SWEEPS)}",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also write <run-dir>/combined_missing_log_loss.png",
    )
    args = parser.parse_args(argv)

    if args.all_sweeps:
        roots = [_RANKING_ROOT / p for p in _DEFAULT_SWEEPS]
    elif args.results_root:
        roots = [Path(args.results_root)]
        if not roots[0].is_absolute():
            roots = [_RANKING_ROOT / roots[0]]
    else:
        parser.error("Specify --results-root or --all-sweeps")

    out_override = Path(args.output_dir) if args.output_dir else None
    for root in roots:
        plot_sweep(root, out_override, per_run=args.per_run)


if __name__ == "__main__":
    main()
