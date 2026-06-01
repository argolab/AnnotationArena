#!/usr/bin/env python3
"""
Evaluate structured baselines on data_bundle.json missing-cell prediction.

Models (always transductive: train+val+test observed):
  - Pooled unigram P(y | i, j)
  - Naive Bayes IJK
  - Structured NB (attr-pair + CHANGEJ factors)
  - Optional: structured log-linear (softmax over same features; PyTorch; val early stopping when val missing exists; train-missing supervision or train-observed fallback for LLM Rubric–style splits)

Run from imputer/ranking:

  python BASELINES/run_structured_baselines.py \\
      --bundle DATA/LLMRubric_225_25_8_175/data_bundle.json

  python BASELINES/run_structured_baselines.py \\
      --bundle path/to/data_bundle.json --log-linear --log-linear-progress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from structured_baselines.cli_defaults import (
    DEFAULT_IJK_ALPHA,
    DEFAULT_LOG_LINEAR_BATCH,
    DEFAULT_LOG_LINEAR_EPOCHS,
    DEFAULT_LOG_LINEAR_LR,
    DEFAULT_LOG_LINEAR_PATIENCE,
    DEFAULT_SNB_ALPHA,
    DEFAULT_UNIGRAM_ALPHA,
)
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from structured_baselines.plate_graph_factorized import accumulate_transductive_counts
from structured_baselines.runner import evaluate_split, load_and_fit
from structured_baselines.dataset_adapter import transductive_observed_cells


def _parse_alpha_sweep(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        a = float(p)
        if a <= 0.0:
            raise ValueError(f"SNB α must be positive, got {a!r}")
        out.append(a)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Structured baselines on data_bundle.json")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--eval-val", action="store_true")
    p.add_argument("--unigram-alpha", type=float, default=DEFAULT_UNIGRAM_ALPHA)
    p.add_argument("--ijk-alpha", type=float, default=DEFAULT_IJK_ALPHA)
    p.add_argument("--snb-alpha", type=float, default=DEFAULT_SNB_ALPHA)
    p.add_argument("--snb-alpha-sweep", type=str, default="")
    p.add_argument("--log-linear", action="store_true", help="Fit structured log-linear (train-missing CE; val early stopping if val missing exists)")
    p.add_argument("--log-linear-epochs", type=int, default=DEFAULT_LOG_LINEAR_EPOCHS)
    p.add_argument("--log-linear-lr", type=float, default=DEFAULT_LOG_LINEAR_LR)
    p.add_argument("--log-linear-batch", type=int, default=DEFAULT_LOG_LINEAR_BATCH)
    p.add_argument(
        "--log-linear-patience",
        type=int,
        default=DEFAULT_LOG_LINEAR_PATIENCE,
        help="Val NLL early-stopping patience; 0 = run all epochs",
    )
    p.add_argument("--log-linear-progress", action="store_true", help="tqdm training progress for log-linear")
    p.add_argument("--out", type=Path, default=None, help="Write metrics JSON")
    args = p.parse_args()

    bundle, fitted = load_and_fit(
        args.bundle,
        unigram_alpha=args.unigram_alpha,
        ijk_alpha=args.ijk_alpha,
        snb_alpha=args.snb_alpha,
        fit_log_linear=args.log_linear,
        log_linear_epochs=args.log_linear_epochs,
        log_linear_lr=args.log_linear_lr,
        log_linear_batch_size=args.log_linear_batch,
        log_linear_early_stopping_patience=(
            None if args.log_linear_patience == 0 else args.log_linear_patience
        ),
        log_linear_show_progress=args.log_linear_progress,
    )
    n_cells = len(transductive_observed_cells(bundle))
    print(f"bundle: {args.bundle}  transductive cells={n_cells}")

    results: dict = {"bundle": str(args.bundle), "n_transductive_cells": n_cells}

    if args.snb_alpha_sweep.strip():
        from structured_baselines.dataset_adapter import bundle_dims, build_test_examples, build_eval_examples

        I, J, C = bundle_dims(bundle, args.bundle)
        K = max(int(r["item"]) for r in bundle["observed_ratings"] + bundle["missing_ratings"])
        counts = accumulate_transductive_counts(
            transductive_observed_cells(bundle),
            num_attrs=I,
            num_classes=C,
            num_anns=J,
            num_items=K,
        )
        alphas = _parse_alpha_sweep(args.snb_alpha_sweep)
        test_ex = build_test_examples(bundle)
        val_ex = build_eval_examples(bundle, "val")
        sweep: dict = {}
        for a in alphas:
            snb = StructuredNaiveBayes(counts=counts, alpha=a)
            sweep[str(a)] = {
                "test": snb.evaluate(test_ex),
                "val": snb.evaluate(val_ex) if val_ex else {},
            }
        results["snb_alpha_sweep"] = sweep
        print(json.dumps(sweep, indent=2))
    else:
        test_m = evaluate_split(fitted, bundle, "test")
        results["test"] = test_m
        for name, m in test_m.items():
            print(f"--- {name} (test missing) ---")
            print(json.dumps(m, indent=2))
        if args.eval_val:
            val_m = evaluate_split(fitted, bundle, "val")
            results["val"] = val_m
            print("--- (val missing) ---")
            print(json.dumps(val_m, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
