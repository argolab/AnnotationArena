#!/usr/bin/env python3
"""
Diagnostics-only DOMAIN3-OLD-SMALL item plots:
  - Marformer from existing item summary/cache
  - Stan Oracle
  - Stan Oracle + draw selection on a 10% dev split from test-missing

Outputs:
  - PLOTS/DOMAIN3-OLD-SMALL/Diagnostics/item_generalization_log_loss.png
  - PLOTS/DOMAIN3-OLD-SMALL/Diagnostics/item_generalization_mbr_l2.png
  - PLOTS/DOMAIN3-OLD-SMALL/Diagnostics/calibration/*
  - PLOTS/DOMAIN3-OLD-SMALL/Diagnostics/item_diagnostics_summary.json
  - PLOTS/DOMAIN3-OLD-SMALL/Diagnostics/item_diagnostics_cache.npz
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import plot_domain3_generalization as shared


ITEM_DISPLAY_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ITEM_X_LEFT = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ITEM_X_RIGHT = []
ITEM_STAN_CUTOFF = 500
PROB_PREFIX = "prob_cat_"

MARFORMER_SUMMARY = ROOT / "PLOTS/DOMAIN3-OLD-SMALL/ITEM_GENERALIZATION/item_generalization_summary.json"
MARFORMER_CACHE = ROOT / "PLOTS/DOMAIN3-OLD-SMALL/ITEM_GENERALIZATION/item_generalization_cache.npz"
DATA_ROOT = ROOT / "DATA/STAN/DOMAIN3-OLD-SMALL/ItemSplits/Transductive"
STAN_ORACLE_ROOT = ROOT / "RESULTS/STAN/DOMAIN3-OLD-SMALL/ORACLE/ITEM"
OUT_DIR = ROOT / "PLOTS/DOMAIN3-OLD-SMALL/Diagnostics"
SUMMARY_PATH = OUT_DIR / "item_diagnostics_summary.json"
CACHE_PATH = OUT_DIR / "item_diagnostics_cache.npz"

SERIES_ORDER = [
    "Marformer",
    "Stan Oracle",
    "Stan Oracle Draw-Selected",
]

COLORS = {
    "Marformer": shared.COLORS["Marformer"],
    "Stan Oracle": shared.COLORS["Stan Oracle"],
    "Stan Oracle Draw-Selected": "#b24a6e",
}

MARKERS = {
    "Marformer": shared.MARKERS["Marformer"],
    "Stan Oracle": shared.MARKERS["Stan Oracle"],
    "Stan Oracle Draw-Selected": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOMAIN3-OLD-SMALL diagnostics with Marformer cache and Stan Oracle variants.")
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Reuse saved diagnostics summary/cache instead of recomputing Stan diagnostics.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _bundle_path(size: int) -> Path:
    return DATA_ROOT / f"DOMAIN3-OLD-SMALL_Item_T_{size}" / "data_bundle.json"


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in idxs], dtype=np.int64)
    return idxs, labels


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return -np.log(clipped)


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    return (expected - truth) ** 2


def _metric_row(model: str, display_size: int, source_size: int, metric: str, values: np.ndarray, mean_override: float | None = None) -> dict:
    mean = float(np.mean(values)) if mean_override is None else float(mean_override)
    std = float(np.std(values, ddof=0))
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "dataset": "DOMAIN3-OLD-SMALL-DIAGNOSTICS",
        "model": model,
        "metric": metric,
        "size": int(display_size),
        "source_size": int(source_size),
        "mean": mean,
        "std": std,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": int(values.size),
    }


def _runtime_row(model: str, display_size: int, source_size: int, minutes: float) -> dict:
    return {
        "dataset": "DOMAIN3-OLD-SMALL-DIAGNOSTICS",
        "model": model,
        "metric": "runtime_minutes",
        "size": int(display_size),
        "source_size": int(source_size),
        "mean": float(minutes),
    }


def _summary_lookup(rows: list[dict], metric: str, model: str, display_size: int) -> dict | None:
    for row in rows:
        if row.get("metric") == metric and row.get("model") == model and int(row.get("size", -1)) == display_size:
            return row
    return None


def _save_cache(
    cache_path: Path,
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    oracle_payloads: dict[int, tuple[np.ndarray, np.ndarray]],
    draw_payloads: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for size, (checkpoint_name, probs, labels) in marformer_payloads.items():
        arrays[f"ckpt__marformer__{size}"] = np.asarray([checkpoint_name])
        arrays[f"probs__marformer__{size}"] = probs
        arrays[f"labels__marformer__{size}"] = labels
    for model_key, payloads in {
        "stan_oracle": oracle_payloads,
        "stan_oracle_draw_selected": draw_payloads,
    }.items():
        for size, (probs, labels) in payloads.items():
            arrays[f"probs__{model_key}__{size}"] = probs
            arrays[f"labels__{model_key}__{size}"] = labels
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)


def _load_cache(cache_path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] = {}
    label_map = {
        "marformer": "Marformer",
        "stan_oracle": "Stan Oracle",
        "stan_oracle_draw_selected": "Stan Oracle Draw-Selected",
    }
    for key in data.files:
        if not key.startswith("probs__"):
            continue
        _, model_key, size_s = key.split("__")
        size = int(size_s)
        label = label_map.get(model_key)
        if label is None:
            continue
        labels_key = f"labels__{model_key}__{size}"
        if labels_key not in data:
            continue
        ckpt_name = None
        ckpt_key = f"ckpt__{model_key}__{size}"
        if ckpt_key in data:
            ckpt_name = str(data[ckpt_key][0])
        out[(label, size)] = (data[key], data[labels_key], ckpt_name)
    return out


def _load_marformer_from_existing_cache() -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    if not MARFORMER_SUMMARY.exists():
        raise FileNotFoundError(f"Missing Marformer summary: {MARFORMER_SUMMARY}")
    if not MARFORMER_CACHE.exists():
        raise FileNotFoundError(f"Missing Marformer cache: {MARFORMER_CACHE}")

    rows = _read_json(MARFORMER_SUMMARY).get("rows", [])
    cache = _load_cache(MARFORMER_CACHE)
    if cache is None:
        raise RuntimeError(f"Failed to load Marformer cache: {MARFORMER_CACHE}")

    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    logloss_override: dict[int, float] = {}

    for size in ITEM_DISPLAY_SIZES:
        cached = cache.get(("Marformer", size))
        if cached is not None:
            probs, labels, checkpoint_name = cached
            payloads[size] = (checkpoint_name or "unknown.ckpt", probs, labels)
        runtime_row = _summary_lookup(rows, "runtime_minutes", "Marformer", size)
        if runtime_row is not None:
            runtimes[size] = float(runtime_row["mean"])
        logloss_row = _summary_lookup(rows, "log_loss", "Marformer", size)
        if logloss_row is not None:
            logloss_override[size] = float(logloss_row["mean"])
    print(f"Used cached Marformer summary/cache -> {MARFORMER_SUMMARY}")
    return payloads, runtimes, logloss_override


def _runtime_from_launcher(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    value = payload.get(key)
    return None if value is None else float(value)


def _load_oracle_draw_table(size: int) -> tuple[pd.DataFrame, list[int], np.ndarray]:
    bundle = _read_json(_bundle_path(size))
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    eval_dir = STAN_ORACLE_ROOT / f"DOMAIN3-OLD-SMALL_Item_T_{size}_TENSOR_eval"
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        raise FileNotFoundError(f"Missing Stan rating probabilities: {probs_path}")
    df = pd.read_csv(probs_path)
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
    df = df[df["missing_rating_idx"].isin(test_idxs)].copy()
    df = df[["sample", "missing_rating_idx", *prob_cols]]
    return df, test_idxs, labels


def _mean_probs_for_prefix(df: pd.DataFrame, test_idxs: list[int], sample_limit: int) -> np.ndarray:
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
    kept_samples = sorted(df["sample"].unique())[:sample_limit]
    grouped = (
        df[df["sample"].isin(kept_samples)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        raise RuntimeError("Stan grouped probabilities contain missing rows after reindex.")
    return grouped.to_numpy(dtype=np.float64)


def _collect_oracle_payloads() -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, float],
    dict[int, dict[str, float | int]],
]:
    oracle_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    draw_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    runtime_minutes: dict[int, float] = {}
    selection_meta: dict[int, dict[str, float | int]] = {}

    for size in [100, 200, 300, 400, 500]:
        print(f"[Stan Oracle] size={size}")
        df, test_idxs, labels = _load_oracle_draw_table(size)
        sample_count = int(df["sample"].nunique())
        oracle_probs = _mean_probs_for_prefix(df, test_idxs, sample_count)
        oracle_payloads[size] = (oracle_probs, labels)
        print(
            f"  oracle n={labels.shape[0]} log_loss={float(np.mean(_per_example_nll(oracle_probs, labels))):.6f} "
            f"mbr_l2={float(np.mean(_per_example_mse(oracle_probs, labels))):.6f}"
        )

        dev_count = max(1, int(round(0.10 * labels.shape[0])))
        rng = np.random.default_rng(42 + size)
        perm = rng.permutation(labels.shape[0])
        dev_pos = np.sort(perm[:dev_count])
        test_pos = np.sort(perm[dev_count:])
        if test_pos.size == 0:
            raise RuntimeError(f"Draw-selection split produced empty test subset for size {size}")

        prefix_candidates = list(range(50, sample_count + 1, 50))
        if prefix_candidates[-1] != sample_count:
            prefix_candidates.append(sample_count)

        best_prefix = prefix_candidates[0]
        best_score = math.inf
        best_dev_mean = math.inf
        best_dev_std = math.inf
        for prefix in prefix_candidates:
            prefix_probs = _mean_probs_for_prefix(df, test_idxs, prefix)
            dev_nll = _per_example_nll(prefix_probs[dev_pos], labels[dev_pos])
            score = float(np.mean(dev_nll) + np.std(dev_nll, ddof=0))
            if score < best_score:
                best_score = score
                best_prefix = prefix
                best_dev_mean = float(np.mean(dev_nll))
                best_dev_std = float(np.std(dev_nll, ddof=0))

        selected_probs = _mean_probs_for_prefix(df, test_idxs, best_prefix)[test_pos]
        selected_labels = labels[test_pos]
        draw_payloads[size] = (selected_probs, selected_labels)
        selection_meta[size] = {
            "sample_count": sample_count,
            "dev_count": int(dev_pos.size),
            "test_count": int(test_pos.size),
            "selected_prefix": int(best_prefix),
            "selection_objective": best_score,
            "dev_log_loss_mean": best_dev_mean,
            "dev_log_loss_std": best_dev_std,
        }
        print(
            f"  draw-selected prefix={best_prefix}/{sample_count} dev={dev_pos.size} test={test_pos.size} "
            f"log_loss={float(np.mean(_per_example_nll(selected_probs, selected_labels))):.6f} "
            f"mbr_l2={float(np.mean(_per_example_mse(selected_probs, selected_labels))):.6f}"
        )

        runtime = _runtime_from_launcher(
            STAN_ORACLE_ROOT / f"DOMAIN3-OLD-SMALL_Item_T_{size}_TENSOR" / "launcher_metadata.json",
            "fit_elapsed_seconds",
        )
        if runtime is not None:
            runtime_minutes[size] = runtime / 60.0

    return oracle_payloads, draw_payloads, runtime_minutes, selection_meta


def _plot_x_jump_marker(ax: plt.Axes, x: float, y: float = 0.0) -> None:
    trans = ax.get_xaxis_transform()
    dx = 0.12
    dy = 0.02
    kwargs = dict(transform=trans, color="0.25", linewidth=1.0, clip_on=False)
    ax.plot([x - dx, x - dx / 4], [y - dy, y + dy], **kwargs)
    ax.plot([x + dx / 4, x + dx], [y - dy, y + dy], **kwargs)


def _series_style(label: str) -> tuple[str, str, str | None]:
    if label == "Stan Oracle Draw-Selected":
        return "-", COLORS[label], MARKERS[label]
    linestyle, color, marker = shared._series_style(label)
    return linestyle, color, marker


def _plot_series_with_stan_extension(
    ax: plt.Axes,
    xs: np.ndarray,
    display_sizes: np.ndarray,
    ys: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    label: str,
) -> None:
    valid = ~np.isnan(ys)
    if not np.any(valid):
        return
    xs_v = xs[valid]
    ys_v = ys[valid]
    lo_v = lo[valid]
    hi_v = hi[valid]
    linestyle, color, marker = _series_style(label)
    line_kwargs = dict(
        color=color,
        marker=marker,
        linewidth=shared._series_linewidth("Stan Oracle" if label.startswith("Stan Oracle") else label),
        alpha=shared._series_alpha("Stan Oracle" if label.startswith("Stan Oracle") else label),
        label=label,
    )
    is_stan_family = label in {"Stan Oracle", "Stan Oracle Draw-Selected"}
    if not is_stan_family or np.all(display_sizes[valid] <= ITEM_STAN_CUTOFF):
        ax.plot(xs_v, ys_v, linestyle=linestyle, **line_kwargs)
        ax.fill_between(xs_v, lo_v, hi_v, color=color, alpha=shared._series_fill_alpha("Stan Oracle" if label.startswith("Stan Oracle") else label), linewidth=0)
        return

    sizes_v = display_sizes[valid]
    pre_mask = sizes_v <= ITEM_STAN_CUTOFF
    post_mask = sizes_v >= ITEM_STAN_CUTOFF
    if np.any(pre_mask):
        ax.plot(xs_v[pre_mask], ys_v[pre_mask], linestyle=linestyle, **line_kwargs)
        ax.fill_between(xs_v[pre_mask], lo_v[pre_mask], hi_v[pre_mask], color=color, alpha=shared._series_fill_alpha("Stan Oracle"), linewidth=0)
    if np.any(post_mask):
        ax.plot(
            xs_v[post_mask],
            ys_v[post_mask],
            linestyle=":",
            color=color,
            linewidth=shared._series_linewidth("Stan Oracle"),
            alpha=shared._series_alpha("Stan Oracle"),
            label="_nolegend_",
        )
        ax.fill_between(xs_v[post_mask], lo_v[post_mask], hi_v[post_mask], color=color, alpha=shared._series_fill_alpha("Stan Oracle"), linewidth=0)


def _series_from_rows(rows: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    stats = {
        stat_name: {label: [np.nan] * len(ITEM_DISPLAY_SIZES) for label in SERIES_ORDER}
        for stat_name in ("mean", "std")
    }
    index_by_size = {size: idx for idx, size in enumerate(ITEM_DISPLAY_SIZES)}
    for row in rows:
        if row.get("metric") != metric:
            continue
        label = row.get("model")
        size = int(row.get("size"))
        if label not in stats["mean"] or size not in index_by_size:
            continue
        idx = index_by_size[size]
        stats["mean"][label][idx] = float(row["mean"])
        stats["std"][label][idx] = float(row["std"])
    return stats


def _plot_metric(rows: list[dict], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    stats = _series_from_rows(rows, metric)
    fig = plt.figure(figsize=(17.6, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[6.3, 1.5], height_ratios=[1.0, 4.8], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_leg = fig.add_subplot(gs[:, 1])
    display_sizes = np.asarray(ITEM_DISPLAY_SIZES, dtype=int)
    xs_all = np.arange(len(display_sizes), dtype=float)

    for label in SERIES_ORDER:
        ys = np.asarray(stats["mean"][label], dtype=float)
        stds = np.asarray(stats["std"][label], dtype=float)
        lo = ys - 0.05 * stds
        hi = ys + 0.05 * stds
        _plot_series_with_stan_extension(ax_top, xs_all, display_sizes, ys, lo, hi, label)
        _plot_series_with_stan_extension(ax_bottom, xs_all, display_sizes, ys, lo, hi, label)

    if metric == "log_loss":
        ax_bottom.set_ylim(0.2, 0.82)
        ax_bottom.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax_top.set_ylim(0.95, 3.05)
        ax_top.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    else:
        ax_bottom.set_ylim(0.0, 0.52)
        ax_bottom.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax_top.set_ylim(0.85, 2.55)
        ax_top.set_yticks([1.0, 1.5, 2.0, 2.5])

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
    ax_bottom.tick_params(top=False)

    ax_bottom.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax_bottom.set_xticks(xs_all)
    ax_bottom.set_xticklabels([str(size) for size in display_sizes])
    ax_bottom.set_xlabel("Training Items")
    ax_bottom.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=16.0, y=0.985)

    ax_top.tick_params(axis="y", labelsize=10.5)
    ax_bottom.tick_params(axis="both", labelsize=10.5)
    ax_bottom.xaxis.label.set_size(12.5)
    ax_bottom.yaxis.label.set_size(12.5)

    xs_break = np.linspace(0.0, 1.0, 61)
    top_break = np.where(np.arange(xs_break.size) % 2 == 0, -0.005, 0.005)
    bottom_break = np.where(np.arange(xs_break.size) % 2 == 0, 1.005, 0.995)
    ax_top.plot(xs_break, top_break, transform=ax_top.transAxes, color="0.25", linewidth=1.0, clip_on=False)
    ax_bottom.plot(xs_break, bottom_break, transform=ax_bottom.transAxes, color="0.25", linewidth=1.0, clip_on=False)
    _plot_x_jump_marker(ax_bottom, x=float(ITEM_DISPLAY_SIZES.index(1000) + 0.5), y=0.0)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    for label in SERIES_ORDER:
        ax_bottom.plot(
            [],
            [],
            linestyle=_series_style(label)[0],
            color=COLORS[label],
            marker=_series_style(label)[2],
            linewidth=shared._series_linewidth("Stan Oracle" if label.startswith("Stan Oracle") else label),
            alpha=shared._series_alpha("Stan Oracle" if label.startswith("Stan Oracle") else label),
            label=label,
        )
    shared._style_legend_right(ax_leg, ax_bottom, fontsize=11.0, labelspacing=0.65, handlelength=2.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.11, hspace=0.05, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _payload_from_cache(cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]], model: str, source_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    payload = cache.get((model, source_size))
    if payload is None:
        return None
    return payload[0], payload[1]


def _plot_calibration(rows: list[dict], cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]]) -> None:
    cal_dir = OUT_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    exact_sizes = [100, 200, 300, 400, 500]
    for size in exact_sizes:
        payloads: list[tuple[str, tuple[np.ndarray, np.ndarray], str]] = []
        for label in SERIES_ORDER:
            row = _summary_lookup(rows, "log_loss", label, size)
            if row is None:
                continue
            source_size = int(row["source_size"])
            payload = _payload_from_cache(cache, label, source_size)
            if payload is None:
                continue
            payloads.append((label, payload, label))
        if not payloads:
            continue

        fig, axes = plt.subplots(1, len(payloads), figsize=(5.0 * len(payloads), 4.6))
        axes = np.atleast_1d(axes).flatten()
        for ax, (title, payload, color_key) in zip(axes, payloads):
            probs, labels = payload
            shared._plot_ece(ax, probs, labels, title, COLORS[color_key])
        fig.suptitle(f"Calibration Plots at {size} Training Items", fontsize=17, y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=1.0, w_pad=1.8)
        output_path = cal_dir / f"item_diagnostics_calibration_size{size}.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved -> {output_path}")


def _build_rows_and_cache() -> tuple[list[dict], dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]]]:
    rows: list[dict] = []
    marformer_payloads, marformer_runtimes, marformer_logloss_override = _load_marformer_from_existing_cache()
    oracle_payloads, draw_payloads, oracle_runtimes, selection_meta = _collect_oracle_payloads()

    for size in ITEM_DISPLAY_SIZES:
        if size in marformer_payloads:
            checkpoint_name, probs, labels = marformer_payloads[size]
            rows.append(_metric_row("Marformer", size, size, "log_loss", _per_example_nll(probs, labels), mean_override=marformer_logloss_override.get(size)))
            rows.append(_metric_row("Marformer", size, size, "mbr_l2", _per_example_mse(probs, labels)))
            runtime = marformer_runtimes.get(size)
            if runtime is not None:
                rows.append(_runtime_row("Marformer", size, size, runtime))

        source_size = size if size <= ITEM_STAN_CUTOFF else ITEM_STAN_CUTOFF
        if source_size in oracle_payloads:
            probs, labels = oracle_payloads[source_size]
            rows.append(_metric_row("Stan Oracle", size, source_size, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row("Stan Oracle", size, source_size, "mbr_l2", _per_example_mse(probs, labels)))
            if size == source_size and size in oracle_runtimes:
                rows.append(_runtime_row("Stan Oracle", size, size, oracle_runtimes[size]))

        if source_size in draw_payloads:
            probs, labels = draw_payloads[source_size]
            rows.append(_metric_row("Stan Oracle Draw-Selected", size, source_size, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row("Stan Oracle Draw-Selected", size, source_size, "mbr_l2", _per_example_mse(probs, labels)))

    _write_json(SUMMARY_PATH, {
        "dataset": "item",
        "display_sizes": ITEM_DISPLAY_SIZES,
        "stan_cutoff": ITEM_STAN_CUTOFF,
        "rows": rows,
        "draw_selection": {str(size): meta for size, meta in selection_meta.items()},
        "marformer_source_summary": str(MARFORMER_SUMMARY),
        "marformer_source_cache": str(MARFORMER_CACHE),
    })
    _save_cache(CACHE_PATH, marformer_payloads, oracle_payloads, draw_payloads)
    cache = _load_cache(CACHE_PATH)
    return rows, {} if cache is None else cache


def main() -> None:
    args = parse_args()
    if args.use_summary:
        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(f"Missing diagnostics summary: {SUMMARY_PATH}")
        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Missing diagnostics cache: {CACHE_PATH}")
        rows = _read_json(SUMMARY_PATH).get("rows", [])
        cache = _load_cache(CACHE_PATH)
        if cache is None:
            raise RuntimeError(f"Failed to load diagnostics cache: {CACHE_PATH}")
        print(f"Used cached diagnostics summary -> {SUMMARY_PATH}")
    else:
        rows, cache = _build_rows_and_cache()
        print(f"Saved diagnostics summary -> {SUMMARY_PATH}")

    _plot_metric(
        rows,
        metric="log_loss",
        ylabel="Test Log Loss",
        title="Item Generalization Diagnostics: Test Log Loss (mean ± 0.05 SD)",
        output_path=OUT_DIR / "item_generalization_log_loss.png",
    )
    _plot_metric(
        rows,
        metric="mbr_l2",
        ylabel="MBR L2 (MSE)",
        title="Item Generalization Diagnostics: MBR L2 (mean ± 0.05 SD)",
        output_path=OUT_DIR / "item_generalization_mbr_l2.png",
    )
    _plot_calibration(rows, cache)


if __name__ == "__main__":
    main()
