#!/usr/bin/env python3
"""
Learning curves (log loss, RMSE) for structured baselines over a folder of bundles.

Each immediate subdirectory of --data-root must contain data_bundle.json.
Training size is parsed from the directory name via --size-regex (default: last _<digits>).

Optional STAN overlay: --stan-results-root with eval dirs named by --stan-eval-regex.

Run from imputer/ranking:

  # DOMAIN3-FINAL item expansion (transductive)
  python scripts/utils/plot_structured_baselines_learning_curve.py \\
      --data-root DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive \\
      --size-regex 'DOMAIN3-FINAL_Item_T_(\\d+)$' \\
      --xlabel 'Training items' \\
      --title 'DOMAIN3-FINAL: structured baselines (item, transductive)' \\
      --output-logloss PLOTS/TALK/DOMAIN3-FINAL/item_T_structured_log_loss.png \\
      --output-rmse PLOTS/TALK/DOMAIN3-FINAL/item_T_structured_rmse.png

  # DOMAIN3-FINAL annotator expansion
  python scripts/utils/plot_structured_baselines_learning_curve.py \\
      --data-root DATA/STAN/DOMAIN3-FINAL/AnnotSplits/Transductive \\
      --size-regex 'DOMAIN3-FINAL_Annot_T_(\\d+)$' \\
      --xlabel 'Training annotators' \\
      --title 'DOMAIN3-FINAL: structured baselines (annot, transductive)' \\
      --output-logloss PLOTS/TALK/DOMAIN3-FINAL/annot_T_structured_log_loss.png
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

from structured_baselines.cli_defaults import DEFAULT_SNB_ALPHA, DEFAULT_UNIGRAM_ALPHA
from structured_baselines.dataset_adapter import (
    build_eval_examples,
    build_test_examples,
    load_bundle_dict,
)
from structured_baselines.runner import calibration_probs_labels, fit_baselines

PANEL_TITLES = {
    "stan": "STAN",
    "unigram_ij": "Unigram (pool ij)",
    "ijk": "Naive Bayes (i,j,k)",
    "snb": "Structured NB",
}
CURVE_STYLES = {
    "stan": ("#1b9e77", "o", "-"),
    "unigram_ij": ("#0b7285", ">", ":"),
    "ijk": ("#111111", "*", "-."),
    "snb": ("#e7298a", "s", "-"),
}


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _sort_pts(pts: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return sorted(pts, key=lambda x: x[0])


def _rmse_from_proba(examples, probs: np.ndarray) -> float | None:
    if len(examples) == 0:
        return None
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    pred = probs @ classes
    truth = np.array([ex.y + 1 for ex in examples], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _discover_bundles(data_root: Path, size_re: re.Pattern[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for bundle_path in sorted(data_root.glob("*/data_bundle.json")):
        m = size_re.match(bundle_path.parent.name)
        if m is None:
            continue
        out.append((int(m.group(1)), bundle_path))
    return out


def _stan_curves(
    stan_root: Path | None,
    stan_eval_re: re.Pattern[str] | None,
    bundle_by_size: dict[int, Path],
    split: str,
) -> tuple[dict[str, list[tuple[int, float]]], dict[int, Path]]:
    ll: dict[str, list[tuple[int, float]]] = {"stan": []}
    rmse: dict[str, list[tuple[int, float]]] = {"stan": []}
    eval_by_size: dict[int, Path] = {}
    if stan_root is None or stan_eval_re is None or not stan_root.is_dir():
        return ll, rmse, eval_by_size

    for metrics_path in sorted(stan_root.glob("*/predictive_metrics.json")):
        m = stan_eval_re.match(metrics_path.parent.name)
        if m is None:
            continue
        size = int(m.group(1))
        eval_by_size[size] = metrics_path.parent
        metrics = _read_json(metrics_path)
        rll = metrics.get("rating_missing_log_likelihood")
        if rll is not None:
            ll["stan"].append((size, float(-rll)))

        bundle_path = bundle_by_size.get(size)
        probs_path = metrics_path.parent / "rating_probabilities.csv"
        if bundle_path is None or not probs_path.exists():
            continue
        bundle = _read_json(bundle_path)
        missing = bundle.get("missing_ratings", [])
        idxs = [i for i, row in enumerate(missing) if str(row.get("instance")) == split]
        if not idxs:
            continue
        labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
        df = pd.read_csv(probs_path)
        n_c = int(labels.max()) + 1 if len(labels) else 4
        prob_cols = [f"prob_cat_{k}" for k in range(1, n_c + 1)]
        if not all(c in df.columns for c in prob_cols):
            continue
        grouped = (
            df[df["missing_rating_idx"].isin(idxs)]
            .groupby("missing_rating_idx")[prob_cols]
            .mean()
            .reindex(idxs)
        )
        if grouped.isnull().any().any():
            continue
        probs = grouped.to_numpy(dtype=np.float64)
        classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
        r = float(np.sqrt(np.mean((probs @ classes - (labels.astype(np.float64) + 1.0)) ** 2)))
        rmse["stan"].append((size, r))
    return ll, rmse, eval_by_size


def _plot_curves(
    series: dict[str, list[tuple[int, float]]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(9.0, 5.4))
    for key, pts in series.items():
        if not pts:
            continue
        pts = _sort_pts(pts)
        color, marker, ls = CURVE_STYLES.get(key, ("#333", "o", "-"))
        plt.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=marker,
            linestyle=ls,
            color=color,
            linewidth=2.0,
            label=PANEL_TITLES.get(key, key),
        )
    plt.xlabel(xlabel)
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
    bundle_path: Path,
    eval_dir: Path | None,
    *,
    size: int,
    split: str,
    snb_alpha: float,
    unigram_alpha: float,
    output: Path,
    stan_label: str,
) -> None:
    from reliability_diagram import plot_reliability_panels

    bundle = load_bundle_dict(bundle_path)
    fitted = fit_baselines(bundle, bundle_path, snb_alpha=snb_alpha, unigram_alpha=unigram_alpha)
    arrays = calibration_probs_labels(fitted, bundle, split)

    panels: list[tuple[str, np.ndarray | None, np.ndarray | None, str]] = []
    if eval_dir is not None:
        probs_path = eval_dir / "rating_probabilities.csv"
        if probs_path.exists():
            missing = bundle.get("missing_ratings", [])
            idxs = [i for i, row in enumerate(missing) if str(row.get("instance")) == split]
            if idxs:
                labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
                df = pd.read_csv(probs_path)
                n_c = int(labels.max()) + 1 if len(labels) else 4
                prob_cols = [f"prob_cat_{k}" for k in range(1, n_c + 1)]
                if all(c in df.columns for c in prob_cols):
                    grouped = (
                        df[df["missing_rating_idx"].isin(idxs)]
                        .groupby("missing_rating_idx")[prob_cols]
                        .mean()
                        .reindex(idxs)
                    )
                    if not grouped.isnull().any().any():
                        panels.append(
                            (stan_label, grouped.to_numpy(dtype=np.float64), labels, CURVE_STYLES["stan"][0])
                        )

    colors = {"unigram_ij": "#0b7285", "ijk": "#111111", "snb": "#e7298a"}
    for key in ("unigram_ij", "ijk", "snb"):
        if key in arrays:
            probs, labels = arrays[key]
            panels.append((PANEL_TITLES[key], probs, labels, colors[key]))
    if not panels:
        print(f"[calibration] no panels for size={size}; skip")
        return
    plot_reliability_panels(
        panels,
        suptitle=f"Reliability at train size {size} ({split} missing) — {bundle_path.parent.name}",
        output_path=output,
    )
    print(f"Saved: {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Structured-baseline learning curves over bundle folders")
    ap.add_argument("--data-root", type=Path, required=True, help="Parent of per-size bundle directories")
    ap.add_argument(
        "--size-regex",
        type=str,
        default=r"_(\d+)$",
        help="Regex on bundle dir name; first capture group = train size",
    )
    ap.add_argument("--stan-results-root", type=Path, default=None, help="Optional STAN eval parent directory")
    ap.add_argument(
        "--stan-eval-regex",
        type=str,
        default="",
        help="Regex on STAN eval dir name; first capture = size (required if --stan-results-root set)",
    )
    ap.add_argument("--stan-label", type=str, default="STAN")
    ap.add_argument("--xlabel", type=str, default="Training size")
    ap.add_argument("--title", type=str, default="Structured baselines")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--snb-alpha", type=float, default=DEFAULT_SNB_ALPHA)
    ap.add_argument("--unigram-alpha", type=float, default=DEFAULT_UNIGRAM_ALPHA)
    ap.add_argument("--output-logloss", type=Path, required=True)
    ap.add_argument("--output-rmse", type=Path, default=None)
    ap.add_argument("--output-calibration", type=Path, default=None)
    ap.add_argument("--calibration-size", type=int, default=0, help="0 = largest size")
    ap.add_argument("--no-calibration", action="store_true")
    args = ap.parse_args()

    size_re = re.compile(args.size_regex)
    bundles = _discover_bundles(args.data_root, size_re)
    if not bundles:
        raise SystemExit(f"No bundles under {args.data_root} matching {args.size_regex!r}")

    bundle_by_size = {size: path for size, path in bundles}
    stan_eval_re = re.compile(args.stan_eval_regex) if args.stan_eval_regex else None
    stan_ll, stan_rmse, eval_by_size = _stan_curves(
        args.stan_results_root, stan_eval_re, bundle_by_size, args.split
    )

    ll: dict[str, list[tuple[int, float]]] = {
        "unigram_ij": [],
        "ijk": [],
        "snb": [],
        **stan_ll,
    }
    rmse: dict[str, list[tuple[int, float]]] = {
        "unigram_ij": [],
        "ijk": [],
        "snb": [],
        **stan_rmse,
    }

    for size, bundle_path in bundles:
        print(f"size={size}  {bundle_path.parent.name} …")
        bundle = load_bundle_dict(bundle_path)
        fitted = fit_baselines(
            bundle, bundle_path, snb_alpha=args.snb_alpha, unigram_alpha=args.unigram_alpha
        )
        if args.split == "test":
            ex = build_test_examples(bundle)
        else:
            ex = build_eval_examples(bundle, args.split)
        ev_u = fitted.unigram_ij.evaluate_split(bundle, args.split)
        ev_i = fitted.nb_ijk.evaluate(ex)
        ev_s = fitted.snb.evaluate(ex)
        if ev_u["n"] > 0:
            ll["unigram_ij"].append((size, float(ev_u["mean_nll"])))
            rmse["unigram_ij"].append((size, float(ev_u["rmse"])))
        if ev_i["n"] > 0:
            ll["ijk"].append((size, float(ev_i["mean_nll"])))
            r_i = _rmse_from_proba(ex, fitted.nb_ijk.predict_proba(ex))
            if r_i is not None:
                rmse["ijk"].append((size, r_i))
        if ev_s["n"] > 0:
            ll["snb"].append((size, float(ev_s["mean_nll"])))
            r_s = _rmse_from_proba(ex, fitted.snb.predict_proba(ex))
            if r_s is not None:
                rmse["snb"].append((size, r_s))

    if not any(ll[k] for k in ("unigram_ij", "ijk", "snb")):
        raise SystemExit("No structured-baseline metrics computed")

    _plot_curves(
        ll,
        xlabel=args.xlabel,
        ylabel=f"{args.split.title()} missing mean NLL",
        title=f"{args.title} (log loss)",
        output=args.output_logloss,
    )
    if args.output_rmse is not None and any(rmse[k] for k in rmse):
        _plot_curves(
            rmse,
            xlabel=args.xlabel,
            ylabel=f"{args.split.title()} missing RMSE",
            title=f"{args.title} (RMSE)",
            output=args.output_rmse,
        )

    if not args.no_calibration and args.output_calibration is not None:
        sizes = sorted(bundle_by_size)
        cal_size = args.calibration_size if args.calibration_size > 0 else sizes[-1]
        if cal_size in bundle_by_size:
            _plot_calibration(
                bundle_by_size[cal_size],
                eval_by_size.get(cal_size),
                size=cal_size,
                split=args.split,
                snb_alpha=args.snb_alpha,
                unigram_alpha=args.unigram_alpha,
                output=args.output_calibration,
                stan_label=args.stan_label,
            )


if __name__ == "__main__":
    main()
