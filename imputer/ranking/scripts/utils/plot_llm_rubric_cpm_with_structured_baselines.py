#!/usr/bin/env python3
"""
LLM Rubric: test-missing log loss, RMSE, and calibration vs train size.

Curves: CPM SharedThreshold STAN, unigram P(y|i,j), NB IJK, structured NB (−CHANGEK); optional log-linear (``--log-linear``).

Run from imputer/ranking (all three outputs by default):

  python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py

Calibration only at one train size:

  python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py --calibration-only

Single-bundle calibration (any dataset):

  python scripts/utils/plot_structured_baselines_calibration.py \\
      --bundle DATA/.../data_bundle.json --output PLOTS/cal.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RANKING_ROOT = Path(__file__).resolve().parents[2]
_UTILS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))
sys.path.insert(0, str(_UTILS_DIR))

from structured_baselines.cli_defaults import (
    DEFAULT_LOG_LINEAR_BATCH,
    DEFAULT_LOG_LINEAR_EPOCHS,
    DEFAULT_LOG_LINEAR_LR,
    DEFAULT_LOG_LINEAR_PATIENCE,
    DEFAULT_SNB_ALPHA,
    DEFAULT_UNIGRAM_ALPHA,
)
from structured_baselines.dataset_adapter import build_test_examples, load_bundle_dict
from structured_baselines.plate_graph_factorized import StructuredFactorMask
from structured_baselines.runner import calibration_probs_labels, fit_baselines

# Structured NB: attr-pair + CHANGEJ only (no cross-item CHANGEK)
SNB_FACTOR_MASK = StructuredFactorMask(attr_pair=True, change_j=True, change_k=False)

SIZE_RE = re.compile(r"LLMRubric_225_25_9_(\d+)_eval$")

PANEL_COLORS = {
    "cpm": "#1b9e77",
    "unigram_ij": "#0b7285",
    "ijk": "#111111",
    "snb": "#e7298a",
    "log_linear": "#7570b3",
}
PANEL_TITLES = {
    "cpm": "CPM SharedThreshold STAN",
    "unigram_ij": "Unigram (pool ij)",
    "ijk": "Naive Bayes (i,j,k)",
    "snb": "Structured NB (−CHANGEK)",
    "log_linear": "Structured log-linear",
}


def _log_linear_fit_kw(args: argparse.Namespace) -> dict:
    if not args.log_linear:
        return {}
    return {
        "fit_log_linear": True,
        "log_linear_epochs": args.log_linear_epochs,
        "log_linear_lr": args.log_linear_lr,
        "log_linear_batch_size": args.log_linear_batch,
        "log_linear_early_stopping_patience": (
            None if args.log_linear_patience == 0 else args.log_linear_patience
        ),
        "log_linear_show_progress": args.log_linear_progress,
    }


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_size(name: str) -> int | None:
    m = SIZE_RE.match(name)
    return int(m.group(1)) if m else None


def _sort_pts(pts: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return sorted(pts, key=lambda x: x[0])


def _rmse_from_proba(examples, probs: np.ndarray) -> float | None:
    if len(examples) == 0:
        return None
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    pred = probs @ classes
    truth = np.array([ex.y + 1 for ex in examples], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _cpm_test_probs_and_labels(
    data_root: Path, size: int, eval_dir: Path, split: str = "test"
) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = eval_dir / "rating_probabilities.csv"
    bundle_path = data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
    if not probs_path.exists() or not bundle_path.exists():
        return None
    bundle = _read_json(bundle_path)
    missing = bundle.get("missing_ratings", [])
    idxs = [i for i, row in enumerate(missing) if row.get("instance") == split]
    if not idxs:
        return None
    labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
    df = pd.read_csv(probs_path)
    n_c = labels.max() + 1 if len(labels) else 4
    prob_cols = [f"prob_cat_{k}" for k in range(1, n_c + 1)]
    if not all(c in df.columns for c in prob_cols):
        return None
    grouped = (
        df[df["missing_rating_idx"].isin(idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(idxs)
    )
    if grouped.isnull().any().any():
        return None
    return grouped.to_numpy(dtype=np.float64), labels


def _cpm_rmse(data_root: Path, size: int, eval_dir: Path) -> float | None:
    out = _cpm_test_probs_and_labels(data_root, size, eval_dir, "test")
    if out is None:
        return None
    probs, labels = out
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    return float(np.sqrt(np.mean((probs @ classes - (labels.astype(np.float64) + 1.0)) ** 2)))


def _plot_curves(
    series: dict[str, list[tuple[int, float]]],
    *,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(9.0, 5.4))
    styles = {
        "cpm": ("#1b9e77", "o", "-"),
        "unigram_ij": ("#0b7285", ">", ":"),
        "ijk": ("#111111", "*", "-."),
        "snb": ("#e7298a", "s", "-"),
        "log_linear": ("#7570b3", "d", "--"),
    }
    labels_map = dict(PANEL_TITLES)
    for key, pts in series.items():
        if not pts:
            continue
        pts = _sort_pts(pts)
        color, marker, ls = styles.get(key, ("#333", "o", "-"))
        plt.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=marker,
            linestyle=ls,
            color=color,
            linewidth=2.0,
            label=labels_map.get(key, key),
        )
    plt.xlabel("Training items (+25 test items with observed LLM ratings)")
    plt.ylabel(ylabel)
    plt.title(title)
    xs = sorted({p[0] for pts in series.values() for p in pts})
    if xs:
        plt.xticks(xs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def _plot_calibration(
    args: argparse.Namespace,
    sizes: list[int],
) -> None:
    from reliability_diagram import plot_reliability_panels

    split = args.calibration_split
    cal_size = int(args.calibration_size) if args.calibration_size > 0 else max(sizes)
    if cal_size not in sizes:
        print(f"[calibration] size {cal_size} not in {sizes}; skip")
        return
    eval_dir = None
    for mp in args.results_root.glob("LLMRubric_225_25_9_*_eval/predictive_metrics.json"):
        sz = _extract_size(mp.parent.name)
        if sz == cal_size:
            eval_dir = mp.parent
            break
    if eval_dir is None:
        print("[calibration] no eval dir; skip")
        return
    bundle_path = args.data_root / f"LLMRubric_225_25_9_{cal_size}" / "data_bundle.json"
    if not bundle_path.exists():
        print(f"[calibration] no bundle {bundle_path}; skip")
        return

    bundle = load_bundle_dict(bundle_path)
    fitted = fit_baselines(
        bundle,
        bundle_path,
        snb_alpha=args.snb_alpha,
        unigram_alpha=args.unigram_alpha,
        snb_factor_mask=SNB_FACTOR_MASK,
        **_log_linear_fit_kw(args),
    )
    arrays = calibration_probs_labels(fitted, bundle, split)

    panels: list[tuple[str, np.ndarray | None, np.ndarray | None, str]] = []
    cpm = _cpm_test_probs_and_labels(args.data_root, cal_size, eval_dir, split)
    if cpm is not None:
        panels.append((PANEL_TITLES["cpm"], cpm[0], cpm[1], PANEL_COLORS["cpm"]))
    for key in ("unigram_ij", "ijk", "snb", "log_linear"):
        if key in arrays:
            probs, labels = arrays[key]
            panels.append((PANEL_TITLES[key], probs, labels, PANEL_COLORS[key]))
    if not panels:
        print("[calibration] no panels; skip")
        return
    plot_reliability_panels(
        panels,
        suptitle=f"LLM Rubric reliability at train size {cal_size} ({split} missing; SNB −CHANGEK)",
        output_path=args.output_calibration,
    )
    print(f"Saved: {args.output_calibration}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD"),
    )
    ap.add_argument("--data-root", type=Path, default=Path("DATA/STAN/LLM_RUBRIC"))
    ap.add_argument("--snb-alpha", type=float, default=DEFAULT_SNB_ALPHA)
    ap.add_argument("--unigram-alpha", type=float, default=DEFAULT_UNIGRAM_ALPHA)
    ap.add_argument(
        "--output-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_log_loss.png"),
    )
    ap.add_argument(
        "--output-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_rmse.png"),
    )
    ap.add_argument(
        "--output-calibration",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_calibration.png"),
    )
    ap.add_argument("--plot-calibration", action="store_true", default=True)
    ap.add_argument("--no-plot-calibration", action="store_false", dest="plot_calibration")
    ap.add_argument("--calibration-only", action="store_true", help="Skip log-loss / RMSE curves")
    ap.add_argument("--calibration-size", type=int, default=0, help="0 = largest train size")
    ap.add_argument(
        "--calibration-split",
        choices=("test", "val"),
        default="test",
        help="Missing split for reliability panels (default: test)",
    )
    ap.add_argument("--log-linear", action="store_true")
    ap.add_argument("--log-linear-epochs", type=int, default=DEFAULT_LOG_LINEAR_EPOCHS)
    ap.add_argument("--log-linear-lr", type=float, default=DEFAULT_LOG_LINEAR_LR)
    ap.add_argument("--log-linear-batch", type=int, default=DEFAULT_LOG_LINEAR_BATCH)
    ap.add_argument(
        "--log-linear-patience",
        type=int,
        default=DEFAULT_LOG_LINEAR_PATIENCE,
        help="0 = no val early stopping on log-linear",
    )
    ap.add_argument("--log-linear-progress", action="store_true")
    args = ap.parse_args()

    if args.calibration_only:
        sizes: list[int] = []
        for mp in args.results_root.glob("LLMRubric_225_25_9_*_eval/predictive_metrics.json"):
            sz = _extract_size(mp.parent.name)
            if sz is not None:
                sizes.append(sz)
        if not sizes:
            raise SystemExit(f"No CPM eval dirs under {args.results_root}")
        _plot_calibration(args, sizes)
        return

    ll: dict[str, list[tuple[int, float]]] = {k: [] for k in ("cpm", "unigram_ij", "ijk", "snb")}
    rmse: dict[str, list[tuple[int, float]]] = {k: [] for k in ("cpm", "unigram_ij", "ijk", "snb")}
    if args.log_linear:
        ll["log_linear"] = []
        rmse["log_linear"] = []
    sizes = []

    for metrics_path in sorted(args.results_root.glob("LLMRubric_225_25_9_*_eval/predictive_metrics.json")):
        size = _extract_size(metrics_path.parent.name)
        if size is None:
            continue
        sizes.append(size)
        metrics = _read_json(metrics_path)
        rll = metrics.get("rating_missing_log_likelihood")
        if rll is not None:
            ll["cpm"].append((size, float(-rll)))
        r = _cpm_rmse(args.data_root, size, metrics_path.parent)
        if r is not None:
            rmse["cpm"].append((size, r))

        bundle_path = args.data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if not bundle_path.exists():
            print(f"[skip] no bundle for size {size}")
            continue
        print(f"size={size}  fitting baselines…")
        bundle = load_bundle_dict(bundle_path)
        fitted = fit_baselines(
            bundle,
            bundle_path,
            snb_alpha=args.snb_alpha,
            unigram_alpha=args.unigram_alpha,
            snb_factor_mask=SNB_FACTOR_MASK,
            **_log_linear_fit_kw(args),
        )
        test_ex = build_test_examples(bundle)
        ev_u = fitted.unigram_ij.evaluate_split(bundle, "test")
        ev_i = fitted.nb_ijk.evaluate(test_ex)
        ev_s = fitted.snb.evaluate(test_ex)
        if ev_u["n"] > 0:
            ll["unigram_ij"].append((size, float(ev_u["mean_nll"])))
            rmse["unigram_ij"].append((size, float(ev_u["rmse"])))
        ll["ijk"].append((size, float(ev_i["mean_nll"])))
        ll["snb"].append((size, float(ev_s["mean_nll"])))
        r_i = _rmse_from_proba(test_ex, fitted.nb_ijk.predict_proba(test_ex))
        r_s = _rmse_from_proba(test_ex, fitted.snb.predict_proba(test_ex))
        if r_i is not None:
            rmse["ijk"].append((size, r_i))
        if r_s is not None:
            rmse["snb"].append((size, r_s))
        if args.log_linear and fitted.log_linear is not None:
            ev_ll = fitted.log_linear.evaluate(test_ex)
            if ev_ll["n"] > 0:
                ll["log_linear"].append((size, float(ev_ll["mean_nll"])))
                r_l = _rmse_from_proba(test_ex, fitted.log_linear.predict_proba(test_ex))
                if r_l is not None:
                    rmse["log_linear"].append((size, r_l))

    if not ll["cpm"]:
        raise SystemExit(f"No CPM metrics under {args.results_root}")

    _plot_curves(
        ll,
        ylabel="Test missing mean NLL",
        title="LLM Rubric: CPM vs structured baselines (log loss; SNB −CHANGEK)",
        output=args.output_logloss,
    )
    if any(rmse[k] for k in rmse):
        _plot_curves(
            rmse,
            ylabel="Test missing RMSE",
            title="LLM Rubric: CPM vs structured baselines (RMSE; SNB −CHANGEK)",
            output=args.output_rmse,
        )
    if args.plot_calibration and sizes:
        _plot_calibration(args, sizes)


if __name__ == "__main__":
    main()
