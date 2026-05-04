#!/usr/bin/env python3
"""
Numerical sanity check for "small $\\hat P(i'\\mid y,\\mathrm{rel})$ but not fatal" stories.

Uses the same ``emit[y, rel, i, v]`` counts as Orange (LOO training examples on
``train,val`` by default).

For a fixed plate class ``y`` (0-based; bundle **1-based** class 4 → ``--y 3``) and
source rubric slice ``i'`` (0-based; bundle label 9 → ``--i-prime 8``):

1. **Per-relation** ``rel`` (default ``1`` = ``SAME_ITEM_SAME_ATTR_DIFF_ANNOT``):
   - ``\\hat\\pi = \\hat P(i'\\mid y,\\mathrm{rel})`` (Laplace on the **rel** slice)
   - ``E^*[\\log P(v\\mid i',y,\\mathrm{rel})]`` weighted by empirical ``v`` counts in
     that (y, rel, i') bucket (same structure as ``plot_llm_rubric_orange_rel_bucket_term_decomposition``)
   - **Base (emission-only)** score uses only the expectation above
   - **Full (conditional + correction)** adds ``\\log\\hat\\pi``

2. Optional **pooled** block: sum count tensors over several relations (e.g. same-plate
   item family ``0,1,3``) and repeat the same π / emission / full split on the **pooled**
   slice (diagnostic only—Orange scores **per** ``rel`` at runtime).

Run from ``imputer/ranking``::

  python scripts/utils/check_llm_rubric_orange_pi_correction_capture.py \\
      --y 3 --i-prime 8 --rel 1

  # optional same-item-k trio for a pooled summary
  python scripts/utils/check_llm_rubric_orange_pi_correction_capture.py \\
      --y 3 --i-prime 8 --pooled-relations 0,1,3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))

from structured_baselines.dataset_adapter import (
    build_training_examples,
    bundle_dims,
    load_bundle_dict,
)
from structured_baselines.feature_utils import RelationKind, relation_label


def _emit_tensor(train_examples, I: int, Cv: int) -> np.ndarray:
    Rd = 7
    emit = np.zeros((Cv, Rd, I, Cv), dtype=np.float64)
    for ex in train_examples:
        y = int(ex.y)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y, rel, i_s, v_s] += 1.0
    return emit


def _log_emit(rel_mat: np.ndarray, i_prime: int, v: int, alpha: float, Cv: int) -> float:
    n_row = float(rel_mat[i_prime, :].sum())
    return math.log(
        (float(rel_mat[i_prime, v]) + alpha) / (n_row + alpha * float(Cv)),
    )


def _log_pi(rel_mat: np.ndarray, i_prime: int, alpha: float, Idims: int) -> float:
    rel_total = float(rel_mat.sum())
    n_i = float(rel_mat[i_prime, :].sum())
    return math.log((n_i + alpha) / (rel_total + alpha * float(Idims)))


def _pi_hat(rel_mat: np.ndarray, i_prime: int, alpha: float, Idims: int) -> float:
    rel_total = float(rel_mat.sum())
    n_i = float(rel_mat[i_prime, :].sum())
    return (n_i + alpha) / (rel_total + alpha * float(Idims))


def _empirical_expect_log_emit(rel_mat: np.ndarray, *, i_prime: int, alpha: float, Cv: int) -> float:
    row = rel_mat[i_prime, :].astype(np.float64, copy=False)
    z = float(row.sum())
    if z <= 0.0:
        return float("nan")
    acc = 0.0
    for v in range(Cv):
        nv = row[v]
        if nv <= 0.0:
            continue
        acc += (nv / z) * _log_emit(rel_mat, i_prime, int(v), alpha, Cv)
    return float(acc)


def _report_block(
    title: str,
    rel_mat: np.ndarray,
    *,
    y: int,
    i_prime: int,
    Idims: int,
    Cv: int,
    alpha: float,
    pi_small_thresh: float,
) -> None:
    print(f"\n=== {title} ===")
    zz = float(rel_mat.sum())
    print(f"y={y}, i_prime={i_prime}, slice mass Σ counts = {zz:.1f}")

    nv_i = float(rel_mat[i_prime, :].sum())
    print(f"  count at (y,*,i',{i_prime},*) summed over v: {nv_i:.1f}")

    pi = _pi_hat(rel_mat, i_prime, alpha, Idims)
    lgpi = _log_pi(rel_mat, i_prime, alpha, Idims)
    ee = _empirical_expect_log_emit(rel_mat, i_prime=i_prime, alpha=alpha, Cv=Cv)

    msg_pi = ""
    if pi < pi_small_thresh:
        msg_pi = f"  [flag] smoothed Pi_hat(i'|y,rel) = {pi:.4e} < thresh {pi_small_thresh:.1e}"
    else:
        msg_pi = f"  Pi_hat(i'|y,rel) = {pi:.6f} (smoothed, α={alpha})"
    print(msg_pi)
    print(f"  log Pi_hat = {lgpi:+.6f} nats")

    if math.isfinite(ee):
        print(f"  E*[log P(v|i',y,rel)] = {ee:+.6f} nats (empirical v weights)")
        print(f"  base (emission-only)     = {ee:+.6f}")
        print(f"  full (+ log Pi)        = {ee + lgpi:+.6f}")
        print(f"  correction share of |full| = {abs(lgpi) / max(abs(ee + lgpi), 1e-12):.3f}")
    else:
        print("  E*[log P(v|...)] = NaN (no v support at this (y, rel, i'))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--bundle",
        type=Path,
        default=_RANKING_ROOT / "DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json",
    )
    ap.add_argument("--train-instances", type=str, default="train,val")
    ap.add_argument(
        "--y",
        type=int,
        default=3,
        help="Target plate class **0-based** (bundle class 4 → 3).",
    )
    ap.add_argument(
        "--i-prime",
        type=int,
        default=8,
        help="Source rubric slice **0-based** (bundle label 9 → 8).",
    )
    ap.add_argument(
        "--rel",
        type=int,
        default=1,
        choices=list(range(7)),
        help="Relation bucket for the primary block (default 1 = SAME_ITEM_SAME_ATTR_DIFF_ANNOT).",
    )
    ap.add_argument(
        "--pooled-relations",
        type=str,
        default="",
        help="Optional comma list (e.g. 0,1,3) for a second pooled summary block.",
    )
    ap.add_argument("--alpha-emit", type=float, default=1.0)
    ap.add_argument(
        "--pi-small-thresh",
        type=float,
        default=1e-3,
        help="Print a flag when Laplace-smoothed Pi_hat falls below this.",
    )
    args = ap.parse_args()

    inst = {s.strip() for s in args.train_instances.split(",") if s.strip()}
    bundle = load_bundle_dict(args.bundle)
    Idims, _J, Cv = bundle_dims(bundle, args.bundle)
    emit = _emit_tensor(build_training_examples(bundle, instances=inst), Idims, Cv)

    if not (0 <= args.y < Cv):
        raise SystemExit(f"--y must be in [0,{Cv})")
    if not (0 <= args.i_prime < Idims):
        raise SystemExit(f"--i-prime must be in [0,{Idims})")

    rel_mat_1 = emit[args.y, args.rel, :, :].astype(np.float64, copy=False)
    _report_block(
        f"Per-rel {args.rel} ({RelationKind(args.rel).name})",
        rel_mat_1,
        y=args.y,
        i_prime=args.i_prime,
        Idims=Idims,
        Cv=Cv,
        alpha=args.alpha_emit,
        pi_small_thresh=args.pi_small_thresh,
    )

    if args.pooled_relations.strip():
        rels = [int(x.strip()) for x in args.pooled_relations.split(",") if x.strip()]
        for r in rels:
            if r < 0 or r > 6:
                raise SystemExit(f"bad rel {r}")
        pooled = np.zeros((Idims, Cv), dtype=np.float64)
        for r in rels:
            pooled += emit[args.y, r, :, :]
        names = "+".join(RelationKind(r).name for r in rels)
        _report_block(
            f"Pooled rels {rels} ({names})",
            pooled,
            y=args.y,
            i_prime=args.i_prime,
            Idims=Idims,
            Cv=Cv,
            alpha=args.alpha_emit,
            pi_small_thresh=args.pi_small_thresh,
        )


if __name__ == "__main__":
    main()
