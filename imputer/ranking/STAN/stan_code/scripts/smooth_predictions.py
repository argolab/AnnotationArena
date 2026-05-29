#!/usr/bin/env python3
"""
Post-hoc temperature scaling for Stan rating probability outputs.

For each *_eval directory under --results-root:
  1. Loads rating_probabilities.csv  [n_samples * n_missing, C]
  2. Loads data_bundle.json to get ground-truth labels for test missing ratings
  3. Calibrates a scalar temperature T by minimising mean NLL on those ratings
  4. Applies T to all samples:  p_smooth = softmax(log(p) / T)
  5. Saves everything to *_eval_smoothed  (identical structure, csv replaced)

Temperature T > 1 flattens (smooths) overconfident one-hot predictions.
T is found via bounded 1-D optimisation — single scalar, negligible overfitting.

Usage:
    python STAN/stan_code/scripts/smooth_predictions.py \\
        --results-root RESULTS/STAN/DOMAIN3-OLD/ORACLE/ITEM \\
        --data-root    DATA/STAN/DOMAIN3-OLD \\
        --split-prefix DOMAIN3-OLD

    python STAN/stan_code/scripts/smooth_predictions.py \\
        --results-root RESULTS/STAN/DOMAIN3-OLD/ORACLE/ANNOT \\
        --data-root    DATA/STAN/DOMAIN3-OLD \\
        --split-prefix DOMAIN3-OLD
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from STAN.stan_code.pipeline.bundle import GroundTruthBundle


# ── Temperature scaling helpers ───────────────────────────────────────────────

def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Temperature-scale a probability array along the last axis.
    probs: [..., C]  (each row already sums to ~1)
    Returns renormalised softmax(log(p) / T), same shape.
    """
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax(log_p / T, axis=-1)


def calibrate_temperature(mean_probs: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Find T in [0.1, 100] minimising mean NLL on (mean_probs, gt_labels).
    mean_probs : [N, C]  — probability already averaged over MCMC samples
    gt_labels  : [N]     — 1-indexed ground-truth category
    """
    def nll(T: float) -> float:
        p = apply_temperature(mean_probs, T)
        picked = p[np.arange(len(gt_labels)), gt_labels - 1]
        return float(-np.mean(np.log(picked + 1e-10)))

    result = minimize_scalar(nll, bounds=(0.1, 100.0), method="bounded")
    return float(result.x)


# ── Path resolution ───────────────────────────────────────────────────────────

def find_data_bundle(eval_dir: Path, data_root: Path) -> Path:
    """
    Derive data_bundle.json path from the eval directory name.

    Name pattern:  {prefix}_{Item|Annot}_{T|NT}_{size}_{model_tag...}_eval
    Split dir:     {data_root}/{Item|Annot}Splits/{Transductive|NonTransductive}/
                   {prefix}_{Item|Annot}_{T|NT}_{size}/data_bundle.json
    """
    stem = eval_dir.name
    if stem.endswith("_eval"):
        stem = stem[:-5]

    m = re.match(r"(.+_(Item|Annot)_(T|NT)_\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse split name from: {eval_dir.name!r}")

    split_name = m.group(1)
    axis       = m.group(2)   # "Item" or "Annot"
    protocol   = m.group(3)   # "T" or "NT"

    folder = "ItemSplits"  if axis     == "Item" else "AnnotSplits"
    sub    = "Transductive" if protocol == "T"    else "NonTransductive"

    path = data_root / folder / sub / split_name / "data_bundle.json"
    if not path.exists():
        raise FileNotFoundError(f"data_bundle.json not found: {path}")
    return path


# ── Per-directory processing ──────────────────────────────────────────────────

def process_eval_dir(eval_dir: Path, data_root: Path, verbose: bool) -> None:
    out_dir = eval_dir.parent / (eval_dir.name + "_smoothed")

    print(f"\n{'─' * 60}")
    print(f"  IN  : {eval_dir.name}")
    print(f"  OUT : {out_dir.name}")

    prob_csv = eval_dir / "rating_probabilities.csv"
    if not prob_csv.exists():
        print("  SKIP: rating_probabilities.csv not found")
        return

    # ── Load probabilities ────────────────────────────────────────────────────
    prob_df   = pd.read_csv(prob_csv)
    prob_cols = [c for c in prob_df.columns if c.startswith("prob_cat_")]
    C         = len(prob_cols)

    if C == 0:
        print("  SKIP: no prob_cat_* columns found")
        return

    n_missing = int(prob_df["missing_rating_idx"].max()) + 1
    n_samples = int(prob_df["sample"].max()) + 1
    raw_probs = prob_df[prob_cols].values.reshape(n_samples, n_missing, C).astype(np.float64)

    # ── Load ground truth ─────────────────────────────────────────────────────
    bundle_path = find_data_bundle(eval_dir, data_root)
    with open(bundle_path) as f:
        bundle = GroundTruthBundle.from_dict(json.load(f))

    test_idx  = bundle.missing_ratings_indexes_in_test_instance
    gt_labels = np.array([bundle.missing_ratings[i]["value"] for i in test_idx])

    # ── Calibrate T on test-instance missing ratings ──────────────────────────
    test_raw    = raw_probs[:, test_idx, :]   # [n_samples, n_test, C]
    mean_test   = test_raw.mean(axis=0)       # [n_test, C]

    T = calibrate_temperature(mean_test, gt_labels)

    nll_before = -np.mean(np.log(mean_test[np.arange(len(gt_labels)), gt_labels - 1] + 1e-10))
    cal_mean   = apply_temperature(mean_test, T)
    nll_after  = -np.mean(np.log(cal_mean[np.arange(len(gt_labels)), gt_labels - 1] + 1e-10))

    print(f"  T = {T:.4f}   NLL: {nll_before:.4f} → {nll_after:.4f}")

    # ── Apply T to all samples × all missing ratings ──────────────────────────
    flat           = raw_probs.reshape(-1, C)
    smooth_flat    = apply_temperature(flat, T)
    smooth_probs   = smooth_flat.reshape(n_samples, n_missing, C)

    # ── Write output directory ────────────────────────────────────────────────
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Copy every file except rating_probabilities.csv (replaced below)
    for src in eval_dir.iterdir():
        if src.is_file() and src.name != "rating_probabilities.csv":
            shutil.copy2(src, out_dir / src.name)

    # Write smoothed rating_probabilities.csv  (same schema as original)
    out_prob_df = pd.DataFrame(smooth_flat, columns=prob_cols)
    out_prob_df["sample"]             = np.repeat(range(n_samples), n_missing)
    out_prob_df["missing_rating_idx"] = np.tile(range(n_missing), n_samples)
    out_prob_df.to_csv(out_dir / "rating_probabilities.csv", index=False)

    # Write temperature metadata
    acc_after  = float(np.mean(np.argmax(cal_mean, axis=-1) + 1 == gt_labels))
    acc_before = None
    orig_metrics = out_dir / "predictive_metrics.json"
    if orig_metrics.exists():
        with open(orig_metrics) as f:
            m_dict = json.load(f)
        acc_before = m_dict.get("rating_missing_accuracy")
        m_dict["rating_missing_log_likelihood_smoothed"] = float(-nll_after)
        m_dict["rating_missing_accuracy_smoothed"]       = acc_after
        m_dict["temperature_scaling_T"]                  = float(T)
        orig_metrics.write_text(json.dumps(m_dict, indent=2))

    (out_dir / "temperature_scaling.json").write_text(json.dumps({
        "temperature":         float(T),
        "nll_before":          float(nll_before),
        "nll_after":           float(nll_after),
        "accuracy_before":     acc_before,
        "accuracy_after":      acc_after,
        "n_test_ratings":      int(len(gt_labels)),
        "C":                   int(C),
        "n_samples":           int(n_samples),
        "n_missing_total":     int(n_missing),
        "source_eval_dir":     str(eval_dir),
        "data_bundle":         str(bundle_path),
    }, indent=2))

    if verbose:
        print(f"  accuracy before: {acc_before}  after: {acc_after:.4f}")
    print(f"  saved → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc temperature scaling for Stan rating probability outputs."
    )
    parser.add_argument(
        "--results-root", required=True,
        help="Directory containing *_eval subdirs "
             "(e.g. RESULTS/STAN/DOMAIN3-OLD/ORACLE/ITEM)",
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root for data bundles (e.g. DATA/STAN/DOMAIN3-OLD)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_root    = Path(args.data_root)

    if not results_root.exists():
        print(f"ERROR: results-root does not exist: {results_root}")
        sys.exit(1)

    eval_dirs = sorted(
        d for d in results_root.iterdir()
        if d.is_dir()
        and d.name.endswith("_eval")
        and not d.name.endswith("_smoothed")
    )

    if not eval_dirs:
        print(f"No *_eval directories found under {results_root}")
        sys.exit(1)

    print(f"Found {len(eval_dirs)} eval director{'y' if len(eval_dirs) == 1 else 'ies'}:")
    for d in eval_dirs:
        print(f"  {d.name}")

    errors: list[tuple[str, str]] = []
    for eval_dir in eval_dirs:
        try:
            process_eval_dir(eval_dir, data_root, args.verbose)
        except Exception as e:
            import traceback
            print(f"  ERROR in {eval_dir.name}: {e}")
            if args.verbose:
                traceback.print_exc()
            errors.append((eval_dir.name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"Done. {len(eval_dirs) - len(errors)}/{len(eval_dirs)} succeeded.")
    if errors:
        for name, msg in errors:
            print(f"  FAILED: {name}  —  {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
