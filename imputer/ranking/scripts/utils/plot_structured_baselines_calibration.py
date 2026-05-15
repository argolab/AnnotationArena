#!/usr/bin/env python3
"""
Reliability diagram (calibration) for structured baselines on one data_bundle.json.

Panels: unigram (ij), NB IJK, structured NB — all on the same missing split.
Optionally add CPM STAN if you pass --cpm-eval-dir with rating_probabilities.csv.

Run from imputer/ranking:

  python scripts/utils/plot_structured_baselines_calibration.py \\
      --bundle DATA/LLMRubric_225_25_8_175/data_bundle.json \\
      --output PLOTS/calibration.png

  python scripts/utils/plot_structured_baselines_calibration.py \\
      --bundle DATA/LLMRubric_225_25_9_175/data_bundle.json \\
      --cpm-eval-dir RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD/LLMRubric_225_25_9_175_eval \\
      --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_UTILS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "BASELINES"))
sys.path.insert(0, str(_UTILS))

from structured_baselines.cli_defaults import DEFAULT_SNB_ALPHA, DEFAULT_UNIGRAM_ALPHA
from structured_baselines.runner import calibration_probs_labels, load_and_fit
from reliability_diagram import plot_reliability_panels

PANEL_COLORS = {
    "cpm": "#1b9e77",
    "unigram_ij": "#0b7285",
    "ijk": "#111111",
    "snb": "#e7298a",
}
PANEL_TITLES = {
    "cpm": "CPM SharedThreshold STAN",
    "unigram_ij": "Unigram (pool ij)",
    "ijk": "Naive Bayes (i,j,k)",
    "snb": "Structured NB",
}


def _cpm_probs_labels(bundle: dict, eval_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        return None
    missing = bundle.get("missing_ratings", [])
    idxs = [i for i, row in enumerate(missing) if str(row.get("instance")) == split]
    if not idxs:
        return None
    labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
    df = pd.read_csv(probs_path)
    prob_cols = [f"prob_cat_{k}" for k in range(1, int(labels.max()) + 2)]
    if not all(c in df.columns for c in prob_cols):
        prob_cols = [c for c in df.columns if c.startswith("prob_cat_")]
    grouped = (
        df[df["missing_rating_idx"].isin(idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(idxs)
    )
    if grouped.isnull().any().any():
        return None
    return grouped.to_numpy(dtype=np.float64), labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration / reliability diagram for structured baselines")
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument(
        "--cpm-eval-dir",
        type=Path,
        default=None,
        help="Optional STAN eval dir containing rating_probabilities.csv",
    )
    ap.add_argument("--snb-alpha", type=float, default=DEFAULT_SNB_ALPHA)
    ap.add_argument("--unigram-alpha", type=float, default=DEFAULT_UNIGRAM_ALPHA)
    args = ap.parse_args()

    bundle, fitted = load_and_fit(
        args.bundle, snb_alpha=args.snb_alpha, unigram_alpha=args.unigram_alpha
    )
    arrays = calibration_probs_labels(fitted, bundle, args.split)

    panels: list[tuple[str, np.ndarray | None, np.ndarray | None, str]] = []
    if args.cpm_eval_dir is not None:
        cpm = _cpm_probs_labels(bundle, args.cpm_eval_dir, args.split)
        if cpm is not None:
            panels.append((PANEL_TITLES["cpm"], cpm[0], cpm[1], PANEL_COLORS["cpm"]))

    for key in ("unigram_ij", "ijk", "snb"):
        if key in arrays:
            probs, labels = arrays[key]
            panels.append((PANEL_TITLES[key], probs, labels, PANEL_COLORS[key]))

    if not panels:
        raise SystemExit(f"No missing rows for split={args.split!r} in {args.bundle}")

    plot_reliability_panels(
        panels,
        suptitle=f"Reliability ({args.split} missing) — {args.bundle.parent.name}",
        output_path=args.output,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
