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
_COMBINED_MISSING_LOG_LOSS_YLIM = (0.3, 0.8)

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
    *,
    test_results_subdir: str = "TEST_RESULTS",
) -> Tuple[List[int], List[float], List[float], Optional[int]]:
    test_dir = run_dir / test_results_subdir
    if not test_dir.is_dir():
        return [], [], [], None

    epochs: List[int] = []
    log_losses: List[float] = []
    rmses: List[float] = []
    eval_max_item: Optional[int] = None

    for path in sorted(test_dir.glob("periodic-epoch=*.json")):
        epoch = _parse_periodic_epoch(path.name)
        if epoch is None:
            continue
        data = json.loads(path.read_text())
        if eval_max_item is None:
            mi = data.get("eval_max_item", data.get("max_item"))
            if mi is not None or "eval_max_item" in data or "max_item" in data:
                eval_max_item = mi
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
    return epochs, log_losses, rmses, eval_max_item


def load_recurrence_scaling(
    run_dir: Path,
    *,
    scaling_subdir: str = "RECURRENCE_SCALING",
) -> Optional[Dict[str, Any]]:
    path = run_dir / scaling_subdir / "recurrence_scaling.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _scaling_max_item_tag(max_item: Optional[int]) -> str:
    if max_item is None:
        return "fullgraph"
    return f"maxitem{max_item}"


def _infer_recurrence_scaling_max_item(
    runs: Sequence[Path],
    *,
    scaling_subdir: str = "RECURRENCE_SCALING",
) -> tuple[Optional[int], str]:
    """Return (representative max_item, filename/title tag) from scaling JSONs."""
    seen: List[Optional[int]] = []
    for run_dir in runs:
        summary = load_recurrence_scaling(run_dir, scaling_subdir=scaling_subdir)
        if not summary:
            continue
        seen.append(summary.get("max_item"))
    if not seen:
        return None, "fullgraph"
    unique = {v for v in seen}
    if len(unique) == 1:
        mi = seen[0]
        return mi, _scaling_max_item_tag(mi)
    parts = sorted(_scaling_max_item_tag(v) for v in unique)
    return None, "mixed_" + "_".join(parts)


def _scaling_eval_label(max_item: Optional[int], tag: str) -> str:
    if max_item is None and tag == "fullgraph":
        return "full graph (max_item=None)"
    if max_item is None:
        return tag
    return f"max_item={max_item}"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def _annotate_test_max_item(ax: plt.Axes, eval_max_item: Optional[int]) -> None:
    label = _scaling_eval_label(eval_max_item, _scaling_max_item_tag(eval_max_item))
    ax.text(
        0.99,
        0.99,
        f"test eval: {label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
        ),
    )


def _apply_resume_phase_marker(
    ax: plt.Axes,
    *,
    resume_epoch: Optional[int],
    phase_note: Optional[str],
    x_max: Optional[float] = None,
) -> None:
    """Shade epochs after a resume boundary and annotate the new training phase."""
    if resume_epoch is None:
        return
    if x_max is None:
        xlim = ax.get_xlim()
        x_max = float(xlim[1])
    ax.axvspan(
        resume_epoch,
        x_max,
        color="#ffcc80",
        alpha=0.28,
        zorder=0,
        label="_nolegend_",
    )
    ax.axvline(
        resume_epoch,
        color="#e65100",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        zorder=1,
    )
    if phase_note:
        ax.text(
            0.99,
            0.99,
            phase_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.92,
            ),
        )


def plot_combined_missing_log_loss(
    runs: Sequence[Path],
    out_path: Path,
    *,
    title: str,
    also_per_run: bool,
    resume_epoch: Optional[int] = None,
    phase_note: Optional[str] = None,
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
            if resume_epoch is not None:
                _apply_resume_phase_marker(
                    ax_one,
                    resume_epoch=resume_epoch,
                    phase_note=phase_note,
                    x_max=max(epochs) if epochs else None,
                )
            per_path = run_dir / "combined_missing_log_loss.png"
            _save(fig_one, per_path)

    if not any_curve:
        plt.close(fig)
        return

    global_x_max = max(
        (max(epochs) for run_dir in runs for epochs, _ in [load_training_missing_xent(run_dir)] if epochs),
        default=None,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined missing log loss (xent, nats)")
    ax.set_title(title)
    ax.set_ylim(*_COMBINED_MISSING_LOG_LOSS_YLIM)
    _apply_resume_phase_marker(
        ax,
        resume_epoch=resume_epoch,
        phase_note=phase_note,
        x_max=global_x_max,
    )
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
    resume_epoch: Optional[int] = None,
    phase_note: Optional[str] = None,
    test_results_subdir: str = "TEST_RESULTS",
    eval_max_item: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False
    seen_max_item: List[Optional[int]] = []
    for run_dir in runs:
        epochs, log_losses, rmses, run_mi = load_test_periodic_metrics(
            run_dir, test_results_subdir=test_results_subdir
        )
        seen_max_item.append(run_mi)
        if not epochs:
            continue
        values = log_losses if metric == "log_loss" else rmses
        if all(v != v for v in values):  # all NaN
            continue
        ax.plot(epochs, values, label=_short_label(run_dir), alpha=0.9)
        any_curve = True

    if not any_curve:
        plt.close(fig)
        print(f"  (skip {out_path.name}: no periodic {test_results_subdir})")
        return

    mi_for_plot = eval_max_item
    if mi_for_plot is None and seen_max_item:
        unique_vals = set(seen_max_item)
        if len(unique_vals) == 1:
            mi_for_plot = next(iter(unique_vals))

    eval_label = _scaling_eval_label(mi_for_plot, _scaling_max_item_tag(mi_for_plot))
    ax.set_xlabel("Epoch (periodic checkpoint)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({eval_label})")
    _annotate_test_max_item(ax, mi_for_plot)
    if metric == "log_loss":
        ax.set_ylim(*_TEST_LOG_LOSS_YLIM)
    else:
        ax.set_ylim(bottom=0)
    test_x_max = max(
        (
            max(epochs)
            for run_dir in runs
            for epochs, _, _, _ in [load_test_periodic_metrics(run_dir, test_results_subdir=test_results_subdir)]
            if epochs
        ),
        default=None,
    )
    _apply_resume_phase_marker(
        ax,
        resume_epoch=resume_epoch,
        phase_note=phase_note,
        x_max=test_x_max,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_run_test_metrics(
    runs: Sequence[Path],
    *,
    test_results_subdir: str = "TEST_RESULTS",
    eval_max_item: Optional[int] = None,
    resume_epoch: Optional[int] = None,
    phase_note: Optional[str] = None,
) -> None:
    """Write per-run periodic test curves next to each run directory."""
    for run_dir in runs:
        epochs, log_losses, rmses, run_mi = load_test_periodic_metrics(
            run_dir, test_results_subdir=test_results_subdir
        )
        if not epochs:
            continue

        mi_for_plot = eval_max_item if eval_max_item is not None else run_mi
        mi_tag = _scaling_max_item_tag(mi_for_plot)
        eval_label = _scaling_eval_label(mi_for_plot, mi_tag)
        run_label = _short_label(run_dir)

        # Per-run test missing log loss
        fig_ll, ax_ll = plt.subplots(figsize=(7, 4.5))
        ax_ll.plot(epochs, log_losses, "o-", color="#1f77b4", alpha=0.9)
        ax_ll.set_xlabel("Epoch (periodic checkpoint)")
        ax_ll.set_ylabel("Test missing log loss (nats)")
        ax_ll.set_title(f"{run_label} — test missing log loss ({eval_label})")
        ax_ll.set_ylim(*_TEST_LOG_LOSS_YLIM)
        _annotate_test_max_item(ax_ll, mi_for_plot)
        _apply_resume_phase_marker(
            ax_ll,
            resume_epoch=resume_epoch,
            phase_note=phase_note,
            x_max=max(epochs) if epochs else None,
        )
        _save(
            fig_ll,
            run_dir / test_results_subdir / f"test_missing_log_loss_{mi_tag}.png",
        )

        # Per-run test missing RMSE
        if not all(v != v for v in rmses):  # not all NaN
            fig_rmse, ax_rmse = plt.subplots(figsize=(7, 4.5))
            ax_rmse.plot(epochs, rmses, "o-", color="#ff7f0e", alpha=0.9)
            ax_rmse.set_xlabel("Epoch (periodic checkpoint)")
            ax_rmse.set_ylabel("Test missing RMSE")
            ax_rmse.set_title(f"{run_label} — test missing RMSE ({eval_label})")
            ax_rmse.set_ylim(bottom=0)
            _annotate_test_max_item(ax_rmse, mi_for_plot)
            _apply_resume_phase_marker(
                ax_rmse,
                resume_epoch=resume_epoch,
                phase_note=phase_note,
                x_max=max(epochs) if epochs else None,
            )
            _save(
                fig_rmse,
                run_dir / test_results_subdir / f"test_missing_rmse_{mi_tag}.png",
            )


def plot_recurrence_scaling_aggregate(
    runs: Sequence[Path],
    out_dir: Path,
    *,
    sweep_name: str,
    scaling_subdir: str = "RECURRENCE_SCALING",
) -> None:
    series: List[Tuple[str, List[int], List[float], List[float], Optional[int]]] = []
    for run_dir in runs:
        summary = load_recurrence_scaling(run_dir, scaling_subdir=scaling_subdir)
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

    max_item, max_item_tag = _infer_recurrence_scaling_max_item(
        runs, scaling_subdir=scaling_subdir
    )
    eval_label = _scaling_eval_label(max_item, max_item_tag)

    for metric, metric_suffix, ylabel in (
        ("log_loss", "log_loss", "Test missing log loss (nats)"),
        ("rmse", "rmse", "Test missing RMSE"),
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
        ax.set_title(
            f"{sweep_name} — recurrence scaling ({metric}, {eval_label})"
        )
        if metric == "log_loss":
            ax.set_ylim(*_TEST_LOG_LOSS_YLIM)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        tagged_path = out_dir / f"recurrence_scaling_{metric_suffix}_{max_item_tag}.png"
        _save(fig, tagged_path)

    # Per-run recurrence plots with log loss + RMSE (upgrade from eval-only log loss)
    for run_dir in runs:
        summary = load_recurrence_scaling(run_dir, scaling_subdir=scaling_subdir)
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
        run_max_item = summary.get("eval_max_item", summary.get("max_item"))
        mi_tag = _scaling_max_item_tag(run_max_item)
        eval_label = _scaling_eval_label(run_max_item, mi_tag)

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
        fig.suptitle(
            f"Recurrence scaling — {cfg} ({summary.get('checkpoint', '')} weights, {eval_label})"
        )
        fig.tight_layout()
        out = run_dir / scaling_subdir / f"recurrence_scaling_{mi_tag}.png"
        _save(fig, out)


def plot_sweep(
    results_root: Path,
    output_dir: Optional[Path],
    *,
    per_run: bool,
    resume_epoch: Optional[int] = None,
    phase_note: Optional[str] = None,
    test_results_subdir: str = "TEST_RESULTS",
    scaling_subdir: str = "RECURRENCE_SCALING",
    eval_max_item: Optional[int] = None,
) -> None:
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
        resume_epoch=resume_epoch,
        phase_note=phase_note,
    )

    mi_tag = _scaling_max_item_tag(eval_max_item)
    plot_global_test_metric(
        runs,
        out_dir / f"global_test_log_loss_{mi_tag}.png",
        metric="log_loss",
        title=f"{sweep_name} — test missing log loss (periodic eval)",
        ylabel="Test missing log loss (nats)",
        resume_epoch=resume_epoch,
        phase_note=phase_note,
        test_results_subdir=test_results_subdir,
        eval_max_item=eval_max_item,
    )
    plot_global_test_metric(
        runs,
        out_dir / f"global_test_rmse_{mi_tag}.png",
        metric="rmse",
        title=f"{sweep_name} — test missing RMSE (periodic eval)",
        ylabel="Test missing RMSE",
        resume_epoch=resume_epoch,
        phase_note=phase_note,
        test_results_subdir=test_results_subdir,
        eval_max_item=eval_max_item,
    )
    if per_run:
        plot_per_run_test_metrics(
            runs,
            test_results_subdir=test_results_subdir,
            eval_max_item=eval_max_item,
            resume_epoch=resume_epoch,
            phase_note=phase_note,
        )

    plot_recurrence_scaling_aggregate(
        runs, out_dir, sweep_name=sweep_name, scaling_subdir=scaling_subdir
    )

    # Mirror sweep-level training curve into RESULTS root (DOMAIN3-OLD convention)
    plot_combined_missing_log_loss(
        runs,
        results_root / "combined_missing_log_loss.png",
        title=f"{sweep_name} — combined missing log loss (training)",
        also_per_run=False,
        resume_epoch=resume_epoch,
        phase_note=phase_note,
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
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Shade epochs after this boundary (e.g. 600 when resuming from a copied checkpoint).",
    )
    parser.add_argument(
        "--phase-note",
        type=str,
        default=None,
        help="Annotation in the top-right corner (e.g. 'max_item=150 after epoch 600').",
    )
    parser.add_argument(
        "--test-results-subdir",
        type=str,
        default="TEST_RESULTS",
        help="Per-run test JSON directory (e.g. TEST_RESULTS_MAXITEM300).",
    )
    parser.add_argument(
        "--scaling-subdir",
        type=str,
        default="RECURRENCE_SCALING",
        help="Per-run recurrence scaling directory (e.g. RECURRENCE_SCALING_MAXITEM300).",
    )
    parser.add_argument(
        "--eval-max-item",
        type=int,
        default=None,
        help="Label test/scaling plots with this eval max_item (default: read from JSON).",
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
        plot_sweep(
            root,
            out_override,
            per_run=args.per_run,
            resume_epoch=args.resume_epoch,
            phase_note=args.phase_note,
            test_results_subdir=args.test_results_subdir,
            scaling_subdir=args.scaling_subdir,
            eval_max_item=args.eval_max_item,
        )


if __name__ == "__main__":
    main()
