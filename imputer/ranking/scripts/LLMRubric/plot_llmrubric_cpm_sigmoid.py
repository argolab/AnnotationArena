#!/usr/bin/env python3
"""
Plot LLMRubric CPM-Sigmoid generalization figures in the DOMAIN3-SIGMOID style.

Outputs:
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_log_loss.png
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_mbr_l2.png
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_runtime.png
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/calibration/*
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_summary.json
  - PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_payload_cache.npz

Use --cache to reuse cached probabilities. Without --cache, probabilities are
rebuilt and the cache is overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from BASELINES.structured_baselines import calibration_probs_labels, load_and_fit
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _load_checkpoint, _reconstruct
from scripts.utils import plot_domain3_generalization as shared


from scripts.utils.font_scale import fs, pfs, ts, scale_rcparams

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.4,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# Grow every font relative to the plot: these figures are reproduced at ~0.48
# of the text width, so the shipped 11pt ticks land near 3pt on the page.
scale_rcparams(mpl.rcParams)


PROB_PREFIX = "prob_cat_"
SERIES_ORDER = ["Marformer", "Stan CPM", "MIWAE", "REMASKER", "Unigram Pool ij"]
CALIBRATION_SIZES = [10, 50, 150, 175]
CALIBRATION_SERIES = ["Marformer", "Stan CPM", "MIWAE", "REMASKER"]

COLORS = {
    "Marformer": "#7a1f3d",
    "Stan CPM": "#0d47a1",
    "MIWAE": "#8b5a2b",
    "REMASKER": "#b07a45",
    "Unigram Pool ij": "#111111",
}

MARKERS = {
    "Marformer": "o",
    "Stan CPM": "^",
    "MIWAE": "s",
    "REMASKER": "P",
    "Unigram Pool ij": "h",
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title_prefix: str
    sizes: list[int]
    x_label: str
    num_classes: int
    data_root: Path
    marformer_root: Path
    stan_root: Path
    baseline_roots: dict[str, Path]
    out_dir: Path
    cache_path: Path
    summary_path: Path
    marformer_total_runtime_seconds: dict[int, float]
    stan_runtime_seconds: dict[int, float]


SPEC = DatasetSpec(
    slug="llmrubric_cpm_sigmoid",
    title_prefix="LLMRubric Item Generalization",
    sizes=[10, 20, 30, 40, 50, 75, 100, 125, 150, 175],
    x_label="Training Items",
    num_classes=4,
    data_root=ROOT / "DATA/LLM_RUBRIC",
    marformer_root=ROOT / "RESULTS/MARFORMER/LLM_RUBRIC",
    stan_root=ROOT / "RESULTS/STAN/CPMSigmoid",
    baseline_roots={
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/LLMRUBRIC",
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/LLMRUBRIC",
    },
    out_dir=ROOT / "PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION",
    cache_path=ROOT / "PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_payload_cache.npz",
    summary_path=ROOT / "PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_summary.json",
    marformer_total_runtime_seconds={
        10: (4 * 60 + 47) * 70 / 300.0,
        20: (7 * 60 + 25) * 82 / 300.0,
        30: (10 * 60 + 27) * 114 / 300.0,
        40: (13 * 60 + 29) * 120 / 300.0,
        50: (18 * 60 + 4) * 124 / 300.0,
        75: (24 * 60 + 23) * 98 / 300.0,
        100: (33 * 60 + 58) * 119 / 300.0,
        125: (39 * 60 + 44) * 146 / 300.0,
        150: (50 * 60 + 24) * 142 / 300.0,
        175: (54 * 60 + 55) * 170 / 300.0,
    },
    stan_runtime_seconds={
        10: 20 * 60,
        20: 23 * 60,
        75: 43 * 60,
        100: 74 * 60,
        125: 91 * 60,
        150: 153 * 60,
        175: 162 * 60,
    },
)


Payload = tuple[np.ndarray, np.ndarray, str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLMRubric CPM-Sigmoid figures.")
    parser.add_argument("--cache", action="store_true", help="Load cached probability payloads if available.")
    parser.add_argument("--best", action="store_true",
                        help="Force Marformer to use TEST_RESULTS/best.json instead of choosing best vs last.")
    parser.add_argument("--device", default=None, help="Torch device for Marformer eval; default: cuda > mps > cpu.")
    return parser.parse_args()


def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_key(label: str, size: int) -> str:
    label_key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{label_key}__{size}"


def _bundle_path(size: int) -> Path:
    return SPEC.data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    if "missing_ratings_indexes_in_test_instance" in bundle:
        idxs = list(bundle["missing_ratings_indexes_in_test_instance"])
    else:
        idxs = [
            i
            for i, row in enumerate(bundle.get("missing_ratings", []))
            if str(row.get("instance")) == "test"
        ]
    labels = np.asarray([bundle["missing_ratings"][i]["value"] - 1 for i in idxs], dtype=np.int64)
    return idxs, labels


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0))


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    return (expected - (labels.astype(np.float64) + 1.0)) ** 2


def _parse_best_epoch(checkpoint_name: str | None, total_epochs: int = 300) -> int:
    if not checkpoint_name or checkpoint_name == "last.ckpt":
        return total_epochs
    match = re.search(r"epoch=(\d+)", checkpoint_name)
    return int(match.group(1)) + 1 if match else total_epochs


def _marformer_run_dir(size: int) -> Path | None:
    run_dir = SPEC.marformer_root / f"LLMRubric_225_25_9_{size}"
    return run_dir if run_dir.exists() else None


def _select_marformer_result_json(size: int, force_best: bool) -> Path | None:
    run_dir = _marformer_run_dir(size)
    if run_dir is None:
        return None
    if force_best:
        path = run_dir / "TEST_RESULTS" / "best.json"
        return path if path.exists() else None
    candidates: list[tuple[float, str, Path]] = []
    for name in ("best.json", "last.json"):
        path = run_dir / "TEST_RESULTS" / name
        if not path.exists():
            continue
        payload = _read_json(path)
        log_loss = payload.get("missing", {}).get("log_loss")
        if log_loss is None:
            continue
        candidates.append((float(log_loss), name, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _marformer_best_json_log_loss(size: int) -> float | None:
    run_dir = _marformer_run_dir(size)
    if run_dir is None:
        return None
    path = run_dir / "TEST_RESULTS" / "best.json"
    if not path.exists():
        return None
    value = _read_json(path).get("missing", {}).get("log_loss")
    return None if value is None else float(value)


def _chunk_variables_by_item_limit(variables: list, max_item: int) -> list[list]:
    item_ids = sorted({iid for var in variables for iid in var.item_ids})
    if len(item_ids) <= max_item:
        return [variables]
    chunks: list[list] = []
    for start in range(0, len(item_ids), max_item):
        item_set = set(item_ids[start: start + max_item])
        chunk = [var for var in variables if all(iid in item_set for iid in var.item_ids)]
        if chunk:
            chunks.append(chunk)
    return chunks


@torch.no_grad()
def _marformer_probs_and_labels(
    size: int,
    device_name: str,
    force_best: bool,
    max_item: int = 10,
) -> Payload | None:
    run_dir = _marformer_run_dir(size)
    result_json = _select_marformer_result_json(size, force_best=force_best)
    if run_dir is None or result_json is None:
        return None
    result_payload = _read_json(result_json)
    checkpoint_name = result_payload.get("checkpoint")
    if checkpoint_name is None:
        return None
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        return None

    model, eval_vars, _train_cfg = _reconstruct(run_dir)
    device = torch.device(device_name)
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    num_classes = model.types["rating"].num_classes
    prob_rows: list[np.ndarray] = []
    labels: list[int] = []
    for chunk_vars in _chunk_variables_by_item_limit(eval_vars, max_item=max_item):
        graph = variable_list_to_entity_graph(chunk_vars, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk_vars):
            if not getattr(var, "is_missing", False):
                continue
            logits = params[0, idx, 1: 1 + num_classes]
            prob_rows.append(torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64))
            labels.append(int(var.rating_value))
    if not prob_rows:
        return None
    return np.vstack(prob_rows), np.asarray(labels, dtype=np.int64), checkpoint_name


def _stan_probs_and_labels(size: int) -> Payload | None:
    eval_dir = SPEC.stan_root / f"LLMRubric_225_25_9_{size}_CPMSigmoid_eval"
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        return None

    bundle = _read_json(_bundle_path(size))
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float64)
    if probs.shape[0] != labels.shape[0]:
        return None
    return probs, labels, str(probs_path)


def _external_baseline_probs_and_labels(method: str, size: int) -> Payload | None:
    pred_path = SPEC.baseline_roots[method] / f"LLMRubric_225_25_9_{size}" / "test_predictions.json"
    if not pred_path.exists():
        return None
    rows = _read_json(pred_path)
    if not rows:
        return None
    probs = np.asarray([row["probs"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    if probs.shape[0] != labels.shape[0]:
        return None
    return probs, labels, str(pred_path)


def _unigram_probs_and_labels(size: int) -> Payload | None:
    bundle, fitted = load_and_fit(_bundle_path(size))
    payloads = calibration_probs_labels(fitted, bundle, split="test")
    payload = payloads.get("unigram_ij")
    if payload is None:
        return None
    probs, labels = payload
    if probs.shape[0] != labels.shape[0]:
        return None
    return probs.astype(np.float64), labels.astype(np.int64), "structured_baselines.unigram_ij"


def _build_payloads(device_name: str, force_best: bool) -> dict[tuple[str, int], Payload]:
    payloads: dict[tuple[str, int], Payload] = {}
    for size in SPEC.sizes:
        loaders = [
            ("Marformer", lambda s=size: _marformer_probs_and_labels(
                s, device_name=device_name, force_best=force_best, max_item=10
            )),
            ("Stan CPM", lambda s=size: _stan_probs_and_labels(s)),
            ("MIWAE", lambda s=size: _external_baseline_probs_and_labels("MIWAE", s)),
            ("REMASKER", lambda s=size: _external_baseline_probs_and_labels("REMASKER", s)),
            ("Unigram Pool ij", lambda s=size: _unigram_probs_and_labels(s)),
        ]
        for label, loader in loaders:
            payload = loader()
            if payload is not None:
                payloads[(label, size)] = payload
    return payloads


def _save_payload_cache(path: Path, payloads: dict[tuple[str, int], Payload]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []
    for (label, size), (probs, labels, source) in sorted(payloads.items(), key=lambda item: (item[0][1], item[0][0])):
        key = _safe_key(label, size)
        arrays[f"probs__{key}"] = probs
        arrays[f"labels__{key}"] = labels
        entries.append({"label": label, "size": size, "source": source, "key": key})
    arrays["meta_json"] = np.asarray(json.dumps({"entries": entries}), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    print(f"Saved cache -> {path}")


def _load_payload_cache(path: Path) -> dict[tuple[str, int], Payload]:
    payloads: dict[tuple[str, int], Payload] = {}
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        for entry in meta.get("entries", []):
            key = entry["key"]
            label = str(entry["label"])
            size = int(entry["size"])
            probs = np.asarray(data[f"probs__{key}"], dtype=np.float64)
            labels = np.asarray(data[f"labels__{key}"], dtype=np.int64)
            payloads[(label, size)] = (probs, labels, entry.get("source"))
    return payloads


def _cache_path(force_best: bool) -> Path:
    if not force_best:
        return SPEC.cache_path
    return SPEC.cache_path.with_name(f"{SPEC.cache_path.stem}_best{SPEC.cache_path.suffix}")


def _payloads_from_args(args: argparse.Namespace, device_name: str) -> dict[tuple[str, int], Payload]:
    cache_path = _cache_path(args.best)
    if args.cache and cache_path.exists():
        print(f"Loading cache -> {cache_path}")
        return _load_payload_cache(cache_path)
    payloads = _build_payloads(device_name, force_best=args.best)
    _save_payload_cache(cache_path, payloads)
    return payloads


def _metric_rows(payloads: dict[tuple[str, int], Payload], *, use_best_json_log_loss: bool = False) -> list[dict]:
    rows: list[dict] = []
    for size in SPEC.sizes:
        for label in SERIES_ORDER:
            payload = payloads.get((label, size))
            if payload is None:
                continue
            probs, labels, source = payload
            metric_values = {
                "log_loss": _per_example_nll(probs, labels),
                "mbr_l2": _per_example_mse(probs, labels),
            }
            for metric, values in metric_values.items():
                mean = float(np.mean(values))
                row_source = source
                if use_best_json_log_loss and label == "Marformer" and metric == "log_loss":
                    direct_log_loss = _marformer_best_json_log_loss(size)
                    if direct_log_loss is not None:
                        mean = direct_log_loss
                        row_source = f"{source}; log_loss_mean=TEST_RESULTS/best.json"
                rows.append({
                    "dataset": SPEC.slug,
                    "size": size,
                    "model": label,
                    "metric": metric,
                    "mean": mean,
                    "std": float(np.std(values, ddof=0)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.90)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": int(values.size),
                    "source": row_source,
                })
    return rows


def _summary_lookup(rows: list[dict], metric: str, model: str, size: int) -> dict | None:
    for row in rows:
        if row["metric"] == metric and row["model"] == model and int(row["size"]) == size:
            return row
    return None


def _series_stats(rows: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    stats = {
        stat: {label: [np.nan] * len(SPEC.sizes) for label in SERIES_ORDER}
        for stat in ("mean", "std")
    }
    index_by_size = {size: idx for idx, size in enumerate(SPEC.sizes)}
    for row in rows:
        if row.get("metric") != metric:
            continue
        label = row.get("model")
        size = int(row["size"])
        if label not in stats["mean"] or size not in index_by_size:
            continue
        idx = index_by_size[size]
        stats["mean"][label][idx] = float(row["mean"])
        stats["std"][label][idx] = float(row["std"])
    return stats


def _series_style(label: str) -> tuple[str, str, str]:
    linestyle = ":" if label == "Unigram Pool ij" else "-"
    return linestyle, COLORS[label], MARKERS[label]


def _series_linewidth(label: str) -> float:
    if label == "Marformer":
        return 2.9
    if label == "Stan CPM":
        return 2.7
    if label == "Unigram Pool ij":
        return 2.1
    return 2.3


def _series_alpha(label: str) -> float:
    if label == "Unigram Pool ij":
        return 0.95
    return 1.0


def _series_fill_alpha(label: str) -> float:
    if label == "Stan CPM":
        return 0.12
    if label in {"MIWAE", "REMASKER"}:
        return 0.08
    return 0.14


def _style_legend_right(
    ax_leg: plt.Axes,
    source_ax: plt.Axes,
    *,
    fontsize: float = fs(10.5),
    labelspacing: float = 0.65,
    handlelength: float = 2.3,
) -> None:
    handles, labels = source_ax.get_legend_handles_labels()
    ax_leg.axis("off")
    if not handles:
        return
    leg = ax_leg.legend(
        handles,
        labels,
        loc="center left",
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.75",
        borderpad=0.8,
        labelspacing=labelspacing,
        handlelength=handlelength,
        fontsize=fontsize,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(1.0)


def _plot_one_axis_series(
    ax: plt.Axes,
    stats: dict[str, dict[str, list[float]]],
    transform=lambda values: np.asarray(values, dtype=float),
) -> tuple[list[float], list[float]]:
    xs = np.arange(len(SPEC.sizes), dtype=float)
    ymin_values: list[float] = []
    ymax_values: list[float] = []
    for label in SERIES_ORDER:
        ys = np.asarray(stats["mean"][label], dtype=float)
        stds = np.asarray(stats["std"][label], dtype=float)
        valid = ~np.isnan(ys)
        if not np.any(valid):
            continue
        lo = ys - 0.05 * stds
        hi = ys + 0.05 * stds
        linestyle, color, marker = _series_style(label)
        ax.plot(
            xs[valid],
            transform(ys)[valid],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=_series_linewidth(label),
            alpha=_series_alpha(label),
            label=label,
        )
        if label != "Unigram Pool ij":
            ax.fill_between(
                xs[valid],
                transform(lo)[valid],
                transform(hi)[valid],
                color=color,
                alpha=_series_fill_alpha(label),
                linewidth=0,
            )
        ymin_values.extend(lo[valid].tolist())
        ymax_values.extend(hi[valid].tolist())
    return ymin_values, ymax_values


def _finish_metric_axis(ax: plt.Axes, ylabel: str) -> None:
    xs = np.arange(len(SPEC.sizes), dtype=float)
    ax.set_xlim(-0.4, len(SPEC.sizes) - 0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(size) for size in SPEC.sizes])
    ax.set_xlabel(SPEC.x_label)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=fs(10.5))
    ax.xaxis.label.set_size(fs(12.5))
    ax.yaxis.label.set_size(fs(12.5))
    ax.grid(True, alpha=0.35)


def _plot_mbr_l2(rows: list[dict]) -> None:
    stats = _series_stats(rows, "mbr_l2")
    fig = plt.figure(figsize=(17.0, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[6.5, 1.4], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    ymins, ymaxs = _plot_one_axis_series(ax, stats)
    if ymins and ymaxs:
        ymin = max(0.0, min(ymins) - 0.05)
        ymax = max(ymaxs) + 0.08
        step = 0.25 if ymax > 2.0 else 0.10
        upper = math.ceil(ymax / step) * step
        lower = math.floor(ymin / step) * step
        ax.set_ylim(lower, upper)
        ticks = np.arange(lower, upper + step * 0.5, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])

    _finish_metric_axis(ax, "MBR L2")
    fig.suptitle(f"{SPEC.title_prefix}: MBR L2 (mean +/- 0.05 SD)", fontsize=ts(16.0), y=0.985)
    _style_legend_right(ax_leg, ax)

    output_path = SPEC.out_dir / "llmrubric_cpm_mbr_l2.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.11, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _draw_horizontal_zigzag(ax: plt.Axes, y: float, *, at_top: bool) -> None:
    xs = np.linspace(0.0, 1.0, 81)
    base = 1.0 if at_top else 0.0
    amp = 0.006
    ys = base + np.where(np.arange(xs.size) % 2 == 0, -amp, amp)
    ax.plot(xs, ys, transform=ax.transAxes, color="0.25", linewidth=1.0, clip_on=False)


def _plot_log_loss(rows: list[dict]) -> None:
    stats = _series_stats(rows, "log_loss")
    fig = plt.figure(figsize=(17.0, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[6.5, 1.4], height_ratios=[1.0, 4.0], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_leg = fig.add_subplot(gs[:, 1])

    for ax in (ax_top, ax_bottom):
        _plot_one_axis_series(ax, stats)

    all_means = [
        float(v)
        for label in SERIES_ORDER
        for v in stats["mean"][label]
        if not math.isnan(float(v))
    ]
    ax_bottom.set_ylim(0.78, 1.5)
    top_upper = max([v for v in all_means if v >= 4.0], default=4.5)
    top_upper = max(5.5, math.ceil((top_upper + 0.20) * 2.0) / 2.0)
    ax_top.set_ylim(3.5, top_upper)

    # Stop at 1.4, not 1.5: the 1.5 label sits directly against the top panel's
    # 3.5 across the axis break and the two collide once the type is 3x. 1.5 is
    # the panel's upper bound, so no data point is left unlabelled.
    bottom_ticks = np.arange(0.8, 1.45, 0.1)
    top_ticks = np.asarray([3.5, 4.5, 5.5], dtype=float)
    ax_bottom.yaxis.set_major_locator(FixedLocator(bottom_ticks))
    ax_bottom.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in bottom_ticks]))
    ax_top.yaxis.set_major_locator(FixedLocator(top_ticks))
    ax_top.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in top_ticks]))

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
    ax_bottom.tick_params(top=False)
    ax_top.minorticks_off()
    ax_bottom.minorticks_off()
    _draw_horizontal_zigzag(ax_top, 0.0, at_top=False)
    _draw_horizontal_zigzag(ax_bottom, 1.0, at_top=True)

    _finish_metric_axis(ax_bottom, "Test Log Loss")
    ax_top.tick_params(axis="y", labelsize=fs(10.5))
    ax_top.grid(True, alpha=0.35)
    fig.suptitle(f"{SPEC.title_prefix}: Test Log Loss (mean +/- 0.05 SD)", fontsize=ts(16.0), y=0.985)
    _style_legend_right(ax_leg, ax_bottom)

    output_path = SPEC.out_dir / "llmrubric_cpm_log_loss.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.11, wspace=0.04, hspace=0.05)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _runtime_minutes_for_marformer(size: int, payloads: dict[tuple[str, int], Payload]) -> float | None:
    total = SPEC.marformer_total_runtime_seconds.get(size)
    payload = payloads.get(("Marformer", size))
    if total is None or payload is None:
        return None
    checkpoint_name = payload[2]
    best_epoch_count = _parse_best_epoch(checkpoint_name, total_epochs=300)
    return total * (best_epoch_count / 300.0) / 60.0


def _runtime_minutes_for_stan(size: int) -> float | None:
    value = SPEC.stan_runtime_seconds.get(size)
    return None if value is None else value / 60.0


def _plot_runtime(payloads: dict[tuple[str, int], Payload]) -> list[dict]:
    rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(13.0, 5.9))
    for label, runtime_fn in (
        ("Marformer", lambda size: _runtime_minutes_for_marformer(size, payloads)),
        ("Stan CPM", _runtime_minutes_for_stan),
    ):
        xs: list[float] = []
        ys: list[float] = []
        for idx, size in enumerate(SPEC.sizes):
            value = runtime_fn(size)
            if value is None:
                continue
            xs.append(float(idx))
            ys.append(float(value))
            rows.append({
                "dataset": SPEC.slug,
                "size": size,
                "model": label,
                "metric": "runtime_minutes",
                "mean": float(value),
            })
        if not xs:
            continue
        linestyle, color, marker = _series_style(label)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
                linewidth=_series_linewidth(label), alpha=_series_alpha(label), label=label)

    ax.set_xlim(-0.4, len(SPEC.sizes) - 0.6)
    ax.set_xticks(np.arange(len(SPEC.sizes), dtype=float))
    ax.set_xticklabels([str(size) for size in SPEC.sizes])
    ax.set_xlabel(SPEC.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.tick_params(axis="both", labelsize=fs(10.5))
    ax.xaxis.label.set_size(fs(12.5))
    ax.yaxis.label.set_size(fs(12.5))
    fig.suptitle(f"{SPEC.title_prefix}: Runtime by {SPEC.x_label}", fontsize=ts(16.0), y=0.98)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=fs(10.5), frameon=True, framealpha=0.9)

    output_path = SPEC.out_dir / "llmrubric_cpm_runtime.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return rows


def _draw_empty_calibration(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=pfs(12), fontweight="normal", pad=8)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", color="0.6", transform=ax.transAxes)


def _plot_calibration(payloads: dict[tuple[str, int], Payload]) -> list[dict]:
    rows: list[dict] = []
    cal_dir = SPEC.out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    for size in CALIBRATION_SIZES:
        # Grown from (7.3, 6.7): the panels need physical room for the larger
        # type, and the grid is still reproduced at one column width.
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 12.4))
        flat_axes = axes.flatten()
        for ax, label in zip(flat_axes, CALIBRATION_SERIES):
            payload = payloads.get((label, size))
            if payload is None:
                _draw_empty_calibration(ax, label)
                continue
            probs, labels, _source = payload
            shared._plot_ece(ax, probs, labels, label, COLORS[label])
            ax.set_title(ax.get_title(), fontsize=ts(12), fontweight="normal", pad=8)
            rows.append({
                "dataset": SPEC.slug,
                "size": size,
                "model": label,
                "metric": "calibration_panel",
            })
        shared.share_panel_axis_labels(flat_axes, 2, 2)
        # Padding widened with the type: at 3x the left panel's '1.0' tick and the
        # right panel's '0.0' met at the panel boundary.
        fig.tight_layout(pad=1.2, w_pad=2.6, h_pad=2.0)
        output_path = cal_dir / f"llmrubric_cpm_calibration_size{size}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved -> {output_path}")
    return rows


def _save_summary(rows: list[dict], payloads: dict[tuple[str, int], Payload], cache_path: Path) -> None:
    selected = {}
    for size in SPEC.sizes:
        payload = payloads.get(("Marformer", size))
        if payload is not None:
            selected[str(size)] = payload[2]
    out = {
        "dataset": SPEC.slug,
        "title_prefix": SPEC.title_prefix,
        "sizes": SPEC.sizes,
        "series_order": SERIES_ORDER,
        "calibration_sizes": CALIBRATION_SIZES,
        "marformer_selected_checkpoints": selected,
        "cache_path": str(cache_path),
        "rows": rows,
    }
    _write_json(SPEC.summary_path, out)
    print(f"Saved summary -> {SPEC.summary_path}")


def main() -> None:
    args = parse_args()
    device_name = _choose_device(args.device)
    payloads = _payloads_from_args(args, device_name)
    rows = _metric_rows(payloads, use_best_json_log_loss=args.best)
    _plot_log_loss(rows)
    _plot_mbr_l2(rows)
    rows.extend(_plot_runtime(payloads))
    rows.extend(_plot_calibration(payloads))
    _save_summary(rows, payloads, _cache_path(args.best))


if __name__ == "__main__":
    main()
