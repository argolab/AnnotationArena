#!/usr/bin/env python3
"""
Evaluate three simple baselines on domain-3 missing-rating prediction (same task as Marformer).

  1) NaiveBayesIJK — P(y|i,j,k) ∝ P(y)P(i|y)P(j|y)P(k|y), fit on observed pool (transductive opt).
  2) StructuredNaiveBayes — relation-aware conditional NB from leave-one-out train plates.
  3) StructuredLogLinear — same unigram + bigram features, softmax / cross-entropy.

Run from imputer/ranking (recommended):

  python BASELINES/run_structured_baselines.py \\
      --bundle DATA/STAN/DOMAIN3-ITEM/Tensor_.../data_bundle.json

Or from BASELINES/ with PYTHONPATH including imputer/ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Allow `python BASELINES/run_structured_baselines.py` from ranking cwd
sys.path.insert(0, str(BASE))

from structured_baselines.dataset_adapter import (
    build_eval_examples,
    build_test_examples,
    build_training_examples,
    bundle_dims,
    load_bundle_dict,
)
from structured_baselines.log_linear_structured import StructuredLogLinear
from structured_baselines.naive_bayes_ijk import NaiveBayesIJK
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes


def main() -> None:
    p = argparse.ArgumentParser(description="Structured + IJK baselines on data_bundle.json")
    p.add_argument("--bundle", type=Path, required=True, help="Path to data_bundle.json")
    p.add_argument(
        "--train-instances",
        default="train",
        help="Comma-separated instances for LOO training plates (default: train). "
        "Example: train,val",
    )
    p.add_argument(
        "--no-ijk-transductive",
        action="store_true",
        help="If set, IJK NB fits on train+val observed only (excludes test-observed pool).",
    )
    p.add_argument("--ll-epochs", type=int, default=40)
    p.add_argument("--ll-lr", type=float, default=0.05)
    p.add_argument("--ll-batch", type=int, default=256)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 for log-linear")
    p.add_argument(
        "--no-ll-tqdm",
        action="store_true",
        help="Disable tqdm during log-linear training (use with --ll-verbose for text progress)",
    )
    p.add_argument(
        "--ll-tqdm-batches",
        action="store_true",
        help="Per-epoch batch-level tqdm (finer progress, more console noise)",
    )
    p.add_argument(
        "--ll-verbose",
        action="store_true",
        help="Print per-epoch mean NLL in addition to tqdm",
    )
    p.add_argument("--eval-val", action="store_true", help="Also print val missing metrics.")
    args = p.parse_args()

    bundle = load_bundle_dict(args.bundle)
    I, _J, C = bundle_dims(bundle, args.bundle)
    K = max(
        int(r["item"]) for r in (bundle.get("observed_ratings", []) + bundle.get("missing_ratings", []))
    )
    train_inst = {s.strip() for s in args.train_instances.split(",") if s.strip()}

    train_ex = build_training_examples(bundle, instances=train_inst)
    test_ex = build_test_examples(bundle)

    print(f"bundle: {args.bundle}")
    print(f"dims I={I}, C={C}  |  train LOO examples={len(train_ex)}  test missing={len(test_ex)}")

    nb_ijk = NaiveBayesIJK.fit_from_bundle(bundle, transductive=not args.no_ijk_transductive)
    print("--- Naive Bayes IJK (test missing) ---")
    print(json.dumps(nb_ijk.evaluate(test_ex), indent=2))

    snb = StructuredNaiveBayes.fit(
        train_ex,
        num_attrs=I,
        num_classes=C,
        num_anns=_J,
        num_items=K,
    )
    print("--- Structured Naive Bayes (test missing) ---")
    print(json.dumps(snb.evaluate(test_ex), indent=2))

    ll = StructuredLogLinear.fit(
        train_ex,
        num_attrs=I,
        num_classes=C,
        epochs=args.ll_epochs,
        lr=args.ll_lr,
        batch_size=args.ll_batch,
        device=args.device,
        verbose=args.ll_verbose,
        show_progress=not args.no_ll_tqdm,
        tqdm_batches=args.ll_tqdm_batches,
        tqdm_desc=f"Log-linear | {args.bundle.parent.name}",
    )
    print("--- Structured log-linear (test missing) ---")
    print(json.dumps(ll.evaluate(test_ex), indent=2))

    if args.eval_val:
        val_ex = build_eval_examples(bundle, "val")
        print("--- (val missing) ---")
        print("IJK:", json.dumps(nb_ijk.evaluate(val_ex)))
        print("SNB:", json.dumps(snb.evaluate(val_ex)))
        print("LL: ", json.dumps(ll.evaluate(val_ex)))


if __name__ == "__main__":
    main()
