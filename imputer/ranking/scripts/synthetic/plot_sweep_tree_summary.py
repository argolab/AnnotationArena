#!/usr/bin/env python3
"""
Aggregate synthetic tree sweep results into sweep-specific plots.

Reads each run's `summary.json` and uses:
  - min.train_mse / min.test_mse (best over epoch)
  - min.best_test_epoch (epoch index achieving min test MSE)

Produces improved per-sweep plots under `<base-out>/plots`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


_MSE_FLOOR = 1e-12


@dataclass(frozen=True)
class RunMetrics:
    train_mse: float
    test_mse: float
    best_test_epoch: Optional[int]


@dataclass(frozen=True)
class RunFinalMetrics:
    train_mse: float
    test_mse: float


def load_min_metrics(run_dir: Path) -> Optional[RunMetrics]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None

    m = summary.get("min", {})
    train_mse = m.get("train_mse", None)
    test_mse = m.get("test_mse", None)
    best_test_epoch = m.get("best_test_epoch", None)
    if train_mse is None or test_mse is None:
        return None
    try:
        return RunMetrics(
            train_mse=float(train_mse),
            test_mse=float(test_mse),
            best_test_epoch=None if best_test_epoch is None else int(best_test_epoch),
        )
    except (TypeError, ValueError):
        return None


def load_training_curves(run_dir: Path) -> Optional[Tuple[List[float], List[float]]]:
    curves_path = run_dir / "training_curves.json"
    if not curves_path.exists():
        return None
    try:
        data = json.loads(curves_path.read_text())
    except json.JSONDecodeError:
        return None

    train = data.get("train_loss", [])
    test = data.get("test_loss", [])
    if not isinstance(train, list) or not isinstance(test, list):
        return None
    if len(train) == 0 and len(test) == 0:
        return None
    try:
        train_out = [float(x) for x in train]
        test_out = [float(x) for x in test]
    except (TypeError, ValueError):
        return None
    return train_out, test_out


def load_final_metrics(run_dir: Path) -> Optional[RunFinalMetrics]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None

    m = summary.get("final", {})
    train_mse = m.get("train_mse", None)
    test_mse = m.get("test_mse", None)
    if train_mse is None or test_mse is None:
        return None
    try:
        return RunFinalMetrics(train_mse=float(train_mse), test_mse=float(test_mse))
    except (TypeError, ValueError):
        return None


def safe_log10(x: float) -> float:
    return math.log10(max(float(x), _MSE_FLOOR))


def format_mse(x: float) -> str:
    # Keep annotations readable across orders of magnitude.
    if x >= 1e-2:
        return f"{x:.3g}"
    if x >= 1e-4:
        return f"{x:.2e}"
    return f"{x:.1e}"


def savefig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_train_test_heatmaps(
    *,
    train_grid: np.ndarray,
    test_grid: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    out_path: Path,
    log_color: bool,
    cmap_train: str = "viridis",
    cmap_test: str = "magma",
) -> None:
    assert train_grid.shape == test_grid.shape

    fig, axes = plt.subplots(1, 2, figsize=(1.35 * len(x_labels) * 1.9, 6.0))
    (ax_tr, ax_te) = axes

    # Shared tick locations for both plots.
    xticks = np.arange(len(x_labels))
    yticks = np.arange(len(y_labels))

    def render(ax: plt.Axes, grid: np.ndarray, cmap: str, subtitle: str) -> None:
        data = grid.astype(float)
        mask = np.isnan(data)

        if log_color:
            # Convert to log10 for the color mapping.
            data_plot = np.array(data, dtype=float)
            for i in range(data_plot.shape[0]):
                for j in range(data_plot.shape[1]):
                    if mask[i, j]:
                        data_plot[i, j] = np.nan
                    else:
                        data_plot[i, j] = safe_log10(data_plot[i, j])
            data_masked = np.ma.masked_invalid(data_plot)
            vmin = float(np.nanmin(data_masked)) if not np.all(mask) else None
            vmax = float(np.nanmax(data_masked)) if not np.all(mask) else None
            im = ax.imshow(
                data_masked,
                origin="upper",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            cbar_label = "log10(best MSE)"
        else:
            data_masked = np.ma.masked_invalid(data)
            im = ax.imshow(data_masked, origin="upper", aspect="auto", cmap=cmap)
            cbar_label = "best MSE"

        ax.set_xticks(xticks)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_yticks(yticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(f"{subtitle}")

        # Annotate with linear MSE values (not log-transformed).
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    continue
                ax.text(
                    j,
                    i,
                    format_mse(data[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if log_color else "black",
                    alpha=0.95,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel(cbar_label, rotation=90)

    fig.suptitle(title, y=0.98, fontsize=12)
    render(ax_tr, train_grid, cmap_train, subtitle="Train (best over epochs)")
    render(ax_te, test_grid, cmap_test, subtitle="Test (best over epochs)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, out_path)


def plot_train_test_curves(
    *,
    xs: List[Any],
    train_vals: List[float],
    test_vals: List[float],
    x_label: str,
    title: str,
    out_path: Path,
    log_y: bool,
    y_value_desc: str = "best MSE (min over epochs)",
    highlight_test_best: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    (ax_tr, ax_te) = axes

    for ax, vals, lab in ((ax_tr, train_vals, "Train"), (ax_te, test_vals, "Test")):
        ax.plot(range(len(xs)), vals, marker="o", linewidth=2.0)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_value_desc)
        ax.set_title(f"{lab} MSE")
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        # Mark the best test point if we're plotting test, otherwise just annotate minima.
        if lab == "Test" and highlight_test_best:
            best_i = int(np.nanargmin([v for v in vals]))
            ax.axvline(best_i, color="red", alpha=0.35, linewidth=2.0)
            ax.text(
                best_i,
                vals[best_i],
                " best",
                color="red",
                fontsize=9,
                ha="left",
                va="bottom",
            )

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_individual_train_test_learning_curves_log(
    *,
    curve_runs: List[Tuple[str, List[float], List[float]]],
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Plot individual train/test learning curves (log y-scale).

    curve_runs: list of (label, train_loss_over_epochs, test_loss_over_epochs)
    """
    if not curve_runs:
        return

    n = len(curve_runs)
    cmap = plt.cm.get_cmap("viridis", n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    (ax_tr, ax_te) = axes

    handles = []
    labels = []

    for idx, (lab, train_loss, test_loss) in enumerate(curve_runs):
        color = cmap(idx)
        if train_loss:
            x_tr = np.arange(1, len(train_loss) + 1)
            y_tr = np.maximum(np.asarray(train_loss, dtype=float), _MSE_FLOOR)
            h = ax_tr.plot(x_tr, y_tr, color=color, linewidth=2.0, label=lab)
            handles.append(h[0])
            labels.append(lab)
        if test_loss:
            x_te = np.arange(1, len(test_loss) + 1)
            y_te = np.maximum(np.asarray(test_loss, dtype=float), _MSE_FLOOR)
            ax_te.plot(x_te, y_te, color=color, linewidth=2.0)

    for ax, lab in ((ax_tr, "Train"), (ax_te, "Test")):
        ax.set_title(f"{lab} (log MSE)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("MSE (log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_yscale("log")

    # Legend can get large for sweep 1/3; keep it small and outside.
    ax_tr.legend(
        handles,
        labels,
        fontsize=6,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.95,
    )
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    savefig(fig, out_path)


def plot_bars(
    *,
    labels: List[str],
    train_vals: List[float],
    test_vals: List[float],
    title: str,
    out_path: Path,
    log_y: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    (ax_tr, ax_te) = axes

    for ax, vals, lab in ((ax_tr, train_vals, "Train"), (ax_te, test_vals, "Test")):
        x = np.arange(len(labels))
        ax.bar(x, vals, color="#4c72b0" if lab == "Train" else "#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(f"{lab} best MSE")
        ax.set_ylabel("best MSE (min over epochs)")
        ax.grid(True, axis="y", alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        # annotate values
        for i, v in enumerate(vals):
            ax.text(i, v, format_mse(v), ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path)


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    import csv

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate synthetic tree sweep summaries into plots.")
    parser.add_argument(
        "--base-out",
        type=str,
        default="OUTPUT/SYNTHETIC/tree_new",
        help="Base output directory containing sweep subfolders.",
    )
    parser.add_argument(
        "--plot-root",
        type=str,
        default=None,
        help="Output directory for plots (default: <base-out>/plots).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scaling for y-axis / heatmap colors.",
    )
    parser.add_argument(
        "--annotate-heatmap",
        action="store_true",
        help="Annotate heatmap cells with numeric MSE values.",
    )
    args = parser.parse_args()

    base_out = Path(args.base_out)
    plot_root = Path(args.plot_root) if args.plot_root else base_out / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    log_color = bool(args.log_y)
    annotate_heatmap = bool(args.annotate_heatmap)

    # -------------------- Sweep 1: Depth vs Layers --------------------
    sweep1_root = base_out / "depth_vs_layers"
    depths: List[int] = []
    layers: List[int] = []
    s1_records: List[Dict[str, Any]] = []
    if sweep1_root.exists():
        pat = re.compile(r"^d(?P<depth>\d+)_L(?P<layers>\d+)$")
        for child in sweep1_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            depth = int(m.group("depth"))
            num_layers = int(m.group("layers"))
            metrics = load_min_metrics(child)
            if metrics is None:
                continue
            depths.append(depth)
            layers.append(num_layers)
            s1_records.append(
                {
                    "depth": depth,
                    "layers": num_layers,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": metrics.best_test_epoch,
                }
            )

    depths = sorted(set(depths))
    layers = sorted(set(layers))
    if depths and layers:
        train_grid = np.full((len(depths), len(layers)), np.nan, dtype=float)
        test_grid = np.full((len(depths), len(layers)), np.nan, dtype=float)
        best_epoch_grid: np.ndarray = np.full((len(depths), len(layers)), np.nan, dtype=float)
        for r in s1_records:
            i = depths.index(int(r["depth"]))
            j = layers.index(int(r["layers"]))
            train_grid[i, j] = float(r["train_mse"])
            test_grid[i, j] = float(r["test_mse"])
            best_epoch_grid[i, j] = float(r["best_test_epoch"] if r["best_test_epoch"] is not None else np.nan)

        # Optional: allow disabling annotation for readability.
        if not annotate_heatmap:
            # Temporarily plot without numeric overlay by zeroing text rendering:
            # simplest: pass log_color False? No. We'll just set annotations off by overriding colors and
            # not drawing them. We'll implement by calling a thin wrapper:
            orig_plot_fn = plot_train_test_heatmaps

            def _plot_no_annot(**kwargs: Any) -> None:
                # copy/paste with annotation disabled by setting log_color and skipping text draw
                fig, axes = plt.subplots(1, 2, figsize=(1.35 * len(layers) * 1.9, 6.0))
                (ax_tr, ax_te) = axes
                xticks = np.arange(len(kwargs["x_labels"]))
                yticks = np.arange(len(kwargs["y_labels"]))
                fig.suptitle(kwargs["title"], y=0.98, fontsize=12)

                def render_no(ax: plt.Axes, grid: np.ndarray, cmap: str, subtitle: str) -> None:
                    data = grid.astype(float)
                    mask = np.isnan(data)
                    if kwargs["log_color"]:
                        data_plot = np.array(data, dtype=float)
                        for i in range(data_plot.shape[0]):
                            for j in range(data_plot.shape[1]):
                                if mask[i, j]:
                                    data_plot[i, j] = np.nan
                                else:
                                    data_plot[i, j] = safe_log10(data_plot[i, j])
                        data_masked = np.ma.masked_invalid(data_plot)
                        vmin = float(np.nanmin(data_masked)) if not np.all(mask) else None
                        vmax = float(np.nanmax(data_masked)) if not np.all(mask) else None
                        im = ax.imshow(
                            data_masked,
                            origin="upper",
                            aspect="auto",
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        cbar_label = "log10(best MSE)"
                    else:
                        data_masked = np.ma.masked_invalid(data)
                        im = ax.imshow(data_masked, origin="upper", aspect="auto", cmap=cmap)
                        cbar_label = "best MSE"

                    ax.set_xticks(xticks)
                    ax.set_xticklabels(kwargs["x_labels"])
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(kwargs["y_labels"])
                    ax.set_xlabel(kwargs["x_axis_label"])
                    ax.set_ylabel(kwargs["y_axis_label"])
                    ax.set_title(subtitle)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cbar.ax.set_ylabel(cbar_label, rotation=90)

                render_no(ax_tr, kwargs["train_grid"], "viridis", "Train (best over epochs)")
                render_no(ax_te, kwargs["test_grid"], "magma", "Test (best over epochs)")
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                savefig(fig, kwargs["out_path"])

            _plot_no_annot(
                train_grid=train_grid,
                test_grid=test_grid,
                x_labels=[str(v) for v in layers],
                y_labels=[str(v) for v in depths],
                    x_axis_label="num_layers",
                    y_axis_label="tree_depth",
                title="Sweep 1: depth vs layers (best over epochs)",
                out_path=plot_root / "sweep1_depth_vs_layers_best_min_log_train_test_heatmap.png",
                log_color=log_color,
            )
        else:
            plot_train_test_heatmaps(
                train_grid=train_grid,
                test_grid=test_grid,
                x_labels=[str(v) for v in layers],
                y_labels=[str(v) for v in depths],
                    x_axis_label="num_layers",
                    y_axis_label="tree_depth",
                title="Sweep 1: depth vs layers (best over epochs)",
                out_path=plot_root / "sweep1_depth_vs_layers_best_min_log_train_test_heatmap.png",
                log_color=log_color,
            )

        # Save sweep1 CSV.
        write_csv(s1_records, plot_root / "sweep1_depth_vs_layers_best_min.csv")

        # Individual learning curves (log y) for each (tree_depth, num_layers) setup.
        s1_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for d in depths:
            for L in layers:
                run_dir = sweep1_root / f"d{d}_L{L}"
                loaded = load_training_curves(run_dir)
                if loaded is None:
                    continue
                tr, te = loaded
                s1_curve_runs.append((f"depth={d}, L={L}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s1_curve_runs,
            x_label="epoch",
            title="Sweep 1: depth vs layers (individual curves, log y)",
            out_path=plot_root / "sweep1_depth_vs_layers_individual_curves_log.png",
        )

    # -------------------- Sweep 2: Width scaling --------------------
    sweep2_root = base_out / "width"
    width_vals: List[int] = []
    s2_train: List[float] = []
    s2_test: List[float] = []
    s2_records: List[Dict[str, Any]] = []
    if sweep2_root.exists():
        pat = re.compile(r"^w(?P<w>\d+)$")
        for child in sweep2_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            w = int(m.group("w"))
            metrics = load_final_metrics(child)
            if metrics is None:
                continue
            width_vals.append(w)
            s2_train.append(metrics.train_mse)
            s2_test.append(metrics.test_mse)
            s2_records.append(
                {
                    "width": w,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": None,
                }
            )

    if width_vals:
        # Sort consistently.
        order = np.argsort(width_vals)
        width_sorted = [width_vals[i] for i in order]
        train_sorted = [s2_train[i] for i in order]
        test_sorted = [s2_test[i] for i in order]
        plot_train_test_curves(
            xs=width_sorted,
            train_vals=train_sorted,
            test_vals=test_sorted,
            x_label="width (tree_width)",
            title="Sweep 2: width scaling (final epoch)",
            out_path=plot_root / "sweep2_width_best_min_train_test_log.png",
            log_y=log_color,
            y_value_desc="final MSE (epoch end)",
            highlight_test_best=False,
        )
        write_csv(s2_records, plot_root / "sweep2_width_best_min.csv")

        # Individual learning curves (log y) for each width setup.
        s2_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for w in width_sorted:
            run_dir = sweep2_root / f"w{w}"
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            s2_curve_runs.append((f"w={w}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s2_curve_runs,
            x_label="epoch",
            title="Sweep 2: width scaling (individual curves, log y)",
            out_path=plot_root / "sweep2_width_individual_curves_log.png",
        )

    # -------------------- Sweep 3: Edge direction ablation --------------------
    sweep3_root = base_out / "edge_dir"
    dir_cats: List[str] = []
    depths3: List[int] = []
    s3_records: List[Dict[str, Any]] = []
    if sweep3_root.exists():
        pat = re.compile(r"^(?P<dir>both|c2p|p2c)_d(?P<depth>\d+)$")
        for child in sweep3_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            d = m.group("dir")
            depth = int(m.group("depth"))
            metrics = load_min_metrics(child)
            if metrics is None:
                continue
            dir_cats.append(d)
            depths3.append(depth)
            s3_records.append(
                {
                    "edge_direction": d,
                    "depth": depth,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": metrics.best_test_epoch,
                }
            )

    dir_cats = sorted(set(dir_cats), key=lambda s: {"both": 0, "c2p": 1, "p2c": 2}.get(s, 99))
    depths3 = sorted(set(depths3))
    if dir_cats and depths3:
        train_grid = np.full((len(dir_cats), len(depths3)), np.nan, dtype=float)
        test_grid = np.full((len(dir_cats), len(depths3)), np.nan, dtype=float)
        for r in s3_records:
            i = dir_cats.index(str(r["edge_direction"]))
            j = depths3.index(int(r["depth"]))
            train_grid[i, j] = float(r["train_mse"])
            test_grid[i, j] = float(r["test_mse"])

        if annotate_heatmap:
            plot_train_test_heatmaps(
                train_grid=train_grid,
                test_grid=test_grid,
                x_labels=[str(v) for v in depths3],
                y_labels=[str(v) for v in dir_cats],
                x_axis_label="tree_depth",
                y_axis_label="edge direction",
                title="Sweep 3: edge direction vs depth (best over epochs)",
                out_path=plot_root / "sweep3_edge_dir_best_min_log_train_test_heatmap.png",
                log_color=log_color,
            )
        else:
            # Use the same plot function (annotations are optional).
            plot_train_test_heatmaps(
                train_grid=train_grid,
                test_grid=test_grid,
                x_labels=[str(v) for v in depths3],
                y_labels=[str(v) for v in dir_cats],
                x_axis_label="tree_depth",
                y_axis_label="edge direction",
                title="Sweep 3: edge direction vs depth (best over epochs)",
                out_path=plot_root / "sweep3_edge_dir_best_min_log_train_test_heatmap.png",
                log_color=log_color,
            )

        write_csv(s3_records, plot_root / "sweep3_edge_dir_best_min.csv")

        # Individual learning curves (log y) for each edge-direction/depth setup.
        s3_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for dcat in dir_cats:
            for d in depths3:
                run_dir = sweep3_root / f"{dcat}_d{d}"
                loaded = load_training_curves(run_dir)
                if loaded is None:
                    continue
                tr, te = loaded
                s3_curve_runs.append((f"{dcat}, d={d}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s3_curve_runs,
            x_label="epoch",
            title="Sweep 3: edge direction vs depth (individual curves, log y)",
            out_path=plot_root / "sweep3_edge_dir_individual_curves_log.png",
        )

    # -------------------- Sweep 4: Counting variants --------------------
    sweep4_root = base_out / "variant"
    variant_order = ["empty-count", "count", "sum"]
    s4_records: List[Dict[str, Any]] = []
    v_train: List[float] = []
    v_test: List[float] = []
    v_labels: List[str] = []
    for v in variant_order:
        run_dir = sweep4_root / v
        if not run_dir.exists():
            continue
        metrics = load_min_metrics(run_dir)
        if metrics is None:
            continue
        s4_records.append(
            {
                "variant": v,
                "train_mse": metrics.train_mse,
                "test_mse": metrics.test_mse,
                "best_test_epoch": metrics.best_test_epoch,
            }
        )
        v_labels.append(v)
        v_train.append(metrics.train_mse)
        v_test.append(metrics.test_mse)
    if v_labels:
        plot_bars(
            labels=v_labels,
            train_vals=v_train,
            test_vals=v_test,
            title="Sweep 4: counting variants (best over epochs)",
            out_path=plot_root / "sweep4_variant_best_min_train_test_log.png",
            log_y=log_color,
        )
        write_csv(s4_records, plot_root / "sweep4_variant_best_min.csv")

        # Individual learning curves (log y) for each variant.
        s4_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for v in v_labels:
            run_dir = sweep4_root / v
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            s4_curve_runs.append((v, tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s4_curve_runs,
            x_label="epoch",
            title="Sweep 4: counting variants (individual curves, log y)",
            out_path=plot_root / "sweep4_variant_individual_curves_log.png",
        )

    # -------------------- Sweep 5: Leaf-only vs all-node --------------------
    sweep5_root = base_out / "leaf"
    leaf_order = ["all", "leaf_only"]
    s5_records: List[Dict[str, Any]] = []
    s5_train: List[float] = []
    s5_test: List[float] = []
    s5_labels: List[str] = []
    for v in leaf_order:
        run_dir = sweep5_root / v
        if not run_dir.exists():
            continue
        metrics = load_min_metrics(run_dir)
        if metrics is None:
            continue
        s5_records.append(
            {
                "mode": v,
                "train_mse": metrics.train_mse,
                "test_mse": metrics.test_mse,
                "best_test_epoch": metrics.best_test_epoch,
            }
        )
        s5_labels.append("all" if v == "all" else "leaf_only")
        s5_train.append(metrics.train_mse)
        s5_test.append(metrics.test_mse)
    if s5_labels:
        plot_bars(
            labels=s5_labels,
            train_vals=s5_train,
            test_vals=s5_test,
            title="Sweep 5: leaf-only vs all-node (best over epochs)",
            out_path=plot_root / "sweep5_leaf_best_min_train_test_log.png",
            log_y=log_color,
        )
        write_csv(s5_records, plot_root / "sweep5_leaf_best_min.csv")

        # Individual learning curves (log y) for leaf mode.
        s5_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for v in leaf_order:
            run_dir = sweep5_root / v
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            label = "all" if v == "all" else "leaf_only"
            s5_curve_runs.append((label, tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s5_curve_runs,
            x_label="epoch",
            title="Sweep 5: leaf-only vs all-node (individual curves, log y)",
            out_path=plot_root / "sweep5_leaf_individual_curves_log.png",
        )

    # -------------------- Sweep 6: Forest disconnected components --------------------
    sweep6_root = base_out / "forest"
    s6_records: List[Dict[str, Any]] = []
    Ts: List[int] = []
    s6_train: List[float] = []
    s6_test: List[float] = []
    if sweep6_root.exists():
        pat = re.compile(r"^T(?P<T>\d+)$")
        for child in sweep6_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            T = int(m.group("T"))
            metrics = load_min_metrics(child)
            if metrics is None:
                continue
            Ts.append(T)
            s6_train.append(metrics.train_mse)
            s6_test.append(metrics.test_mse)
            s6_records.append(
                {
                    "T": T,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": metrics.best_test_epoch,
                }
            )
    if Ts:
        order = np.argsort(Ts)
        Ts_sorted = [Ts[i] for i in order]
        train_sorted = [s6_train[i] for i in order]
        test_sorted = [s6_test[i] for i in order]
        plot_train_test_curves(
            xs=Ts_sorted,
            train_vals=train_sorted,
            test_vals=test_sorted,
            x_label="num_trees (T)",
            title="Sweep 6: forest components (best over epochs)",
            out_path=plot_root / "sweep6_forest_best_min_train_test_log.png",
            log_y=log_color,
        )
        write_csv(s6_records, plot_root / "sweep6_forest_best_min.csv")

        # Individual learning curves (log y) for each T.
        s6_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for T in Ts_sorted:
            run_dir = sweep6_root / f"T{T}"
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            s6_curve_runs.append((f"T={T}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s6_curve_runs,
            x_label="epoch",
            title="Sweep 6: forest components (individual curves, log y)",
            out_path=plot_root / "sweep6_forest_individual_curves_log.png",
        )

    # -------------------- Sweep 7: Param-dim scaling (vector sum) --------------------
    sweep7_root = base_out / "param_dim"
    Ds: List[int] = []
    s7_train: List[float] = []
    s7_test: List[float] = []
    s7_records: List[Dict[str, Any]] = []
    if sweep7_root.exists():
        pat = re.compile(r"^D(?P<D>\d+)$")
        for child in sweep7_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            D = int(m.group("D"))
            metrics = load_min_metrics(child)
            if metrics is None:
                continue
            Ds.append(D)
            s7_train.append(metrics.train_mse)
            s7_test.append(metrics.test_mse)
            s7_records.append(
                {
                    "param_dim": D,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": metrics.best_test_epoch,
                }
            )
    if Ds:
        order = np.argsort(Ds)
        Ds_sorted = [Ds[i] for i in order]
        train_sorted = [s7_train[i] for i in order]
        test_sorted = [s7_test[i] for i in order]
        plot_train_test_curves(
            xs=Ds_sorted,
            train_vals=train_sorted,
            test_vals=test_sorted,
            x_label="param_dim (D)",
            title="Sweep 7: param-dim scaling (best over epochs)",
            out_path=plot_root / "sweep7_param_dim_best_min_train_test_log.png",
            log_y=log_color,
        )
        write_csv(s7_records, plot_root / "sweep7_param_dim_best_min.csv")

        # Individual learning curves (log y) for each param_dim D.
        s7_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for D in Ds_sorted:
            run_dir = sweep7_root / f"D{D}"
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            s7_curve_runs.append((f"D={D}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s7_curve_runs,
            x_label="epoch",
            title="Sweep 7: param-dim scaling (individual curves, log y)",
            out_path=plot_root / "sweep7_param_dim_individual_curves_log.png",
        )

    # -------------------- Sweep 8: Bold width scaling (fixed depth/layers) --------------------
    # This sweep changes the task/topology more aggressively than sweep 2, so we summarize using
    # final epoch MSE rather than "best over epochs" (same rationale as sweep 2).
    sweep8_root = base_out / "width8" / "d3_L3"
    width8_vals: List[int] = []
    s8_train: List[float] = []
    s8_test: List[float] = []
    s8_records: List[Dict[str, Any]] = []

    if sweep8_root.exists():
        pat = re.compile(r"^w(?P<w>\d+)$")
        for child in sweep8_root.iterdir():
            m = pat.match(child.name)
            if not m or not child.is_dir():
                continue
            w = int(m.group("w"))
            metrics = load_final_metrics(child)
            if metrics is None:
                continue
            width8_vals.append(w)
            s8_train.append(metrics.train_mse)
            s8_test.append(metrics.test_mse)
            s8_records.append(
                {
                    "width": w,
                    "train_mse": metrics.train_mse,
                    "test_mse": metrics.test_mse,
                    "best_test_epoch": None,
                }
            )

    if width8_vals:
        order = np.argsort(width8_vals)
        width8_sorted = [width8_vals[i] for i in order]
        train_sorted = [s8_train[i] for i in order]
        test_sorted = [s8_test[i] for i in order]

        plot_train_test_curves(
            xs=width8_sorted,
            train_vals=train_sorted,
            test_vals=test_sorted,
            x_label="width (tree_width)",
            title="Sweep 8: bold width scaling (depth=3, layers=3, final epoch)",
            out_path=plot_root / "sweep8_width_d3_L3_train_test_log.png",
            log_y=log_color,
            y_value_desc="final MSE (epoch end)",
            highlight_test_best=False,
        )
        write_csv(s8_records, plot_root / "sweep8_width_d3_L3_final.csv")

        s8_curve_runs: List[Tuple[str, List[float], List[float]]] = []
        for w in width8_sorted:
            run_dir = sweep8_root / f"w{w}"
            loaded = load_training_curves(run_dir)
            if loaded is None:
                continue
            tr, te = loaded
            s8_curve_runs.append((f"w={w}", tr, te))

        plot_individual_train_test_learning_curves_log(
            curve_runs=s8_curve_runs,
            x_label="epoch",
            title="Sweep 8: bold width scaling (individual curves, log y)",
            out_path=plot_root / "sweep8_width_d3_L3_individual_curves_log.png",
        )

    print(f"Wrote sweep summary plots under: {plot_root}")


if __name__ == "__main__":
    main()

