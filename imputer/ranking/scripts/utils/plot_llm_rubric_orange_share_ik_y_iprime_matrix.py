#!/usr/bin/env python3
"""
Pooled Orange counts over a chosen set of relation buckets.

Bundle cells are ``X_{i,j,k}`` with **attribute** ``i``, **annotator** ``j``, **item**
``k`` ((see ``feature_utils.relation_label``)).

**Interpretation “share attribute *i* and annotator *j*”.** Sources that match the plate
attribute *and* the plate annotator but sit on another item have
``i_src = i_tgt`` and ``j_src = j_tgt`` and ``k_src ≠ k_tgt``. That Orange bucket is
``RelationKind.SAME_ANNOT_SAME_ATTR_DIFF_ITEM`` (code ``2``). On standard LOO
``LocalExample`` plates every source shares the plate item coordinate with the target,
``k_src = k_tgt``, so **relation 2 never fires** — the pooled matrix is often **all zeros**
(not a bug).

If you intended “same rubric slice *i* and same plate item *k*, different annotators”
(typo naming the third coordinate), that intra-plate bucket is ``1``::

    SAME_ITEM_SAME_ATTR_DIFF_ANNOT (same ``i``, same item ``k``, ``j`` differs).

To pool “same-plate-item” neighbourhoods generally, use ``--pool-relations 0,1,3``.

Plate label ``y`` (Orange ``emit`` axis 0) × source slice ``i′``::

    N[y, i′] = Σ_{rel ∈ POOL} Σ_v emit[y, rel, i′, v]

Plots: log₁p(N) heatmap and row‑Dirichlet smoothed \\hat P(i′\\mid y) (row Laplace).

Run::

  python scripts/utils/plot_llm_rubric_orange_share_ik_y_iprime_matrix.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))

from structured_baselines.dataset_adapter import (
    build_training_examples,
    bundle_dims,
    load_bundle_dict,
)
from structured_baselines.feature_utils import RelationKind, relation_label

def _orange_emit(train_examples, num_attrs: int, num_classes: int):
    Rd = 7
    I, Cv = num_attrs, num_classes
    emit = np.zeros((Cv, Rd, I, Cv), dtype=np.float64)
    for ex in train_examples:
        y_tgt = int(ex.y)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y_tgt, rel, i_s, v_s] += 1.0
    return emit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--bundle",
        type=Path,
        default=_RANKING_ROOT / "DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_RANKING_ROOT
        / "PLOTS/TALK/LLM_RUBRIC/"
        "llm_rubric_orange_same_attr_same_annot_bucket_y_iprime_matrix.png",
    )
    ap.add_argument(
        "--train-instances",
        type=str,
        default="train,val",
    )
    ap.add_argument(
        "--pool-relations",
        type=str,
        default="2",
        help=(
            "Comma-separated RelationKind codes to pool. Default ``2`` = same attribute "
            "**i** AND same annotator **j**, different item (**SAME_ANNOT_SAME_ATTR_DIFF_ITEM**)."
            " On LOO intra-item plates rel 2 seldom appears (see docstring—likely all zeros)."
            " For same **i**, same plate item **k**, different **j**, use ``1``."
            " Same plate item neighbourhoods broadly: ``0,1,3``."
        ),
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet smoothing per row when showing P_hat(i′|y).",
    )
    args = ap.parse_args()

    rel_set = frozenset(
        int(x.strip()) for x in args.pool_relations.split(",") if x.strip()
    )

    inst = {s.strip() for s in args.train_instances.split(",") if s.strip()}
    bundle = load_bundle_dict(args.bundle)
    Idims, _Jdims, Cv = bundle_dims(bundle, args.bundle)

    emit = _orange_emit(build_training_examples(bundle, instances=inst), Idims, Cv)

    N_yip = np.zeros((Cv, Idims), dtype=np.float64)
    for rel in sorted(rel_set):
        if rel < 0 or rel >= 7:
            raise SystemExit(f"Invalid rel index {rel}")
        N_yip += emit[:, rel, :, :].sum(axis=-1)

    row_sums = N_yip.sum(axis=1, keepdims=True)
    denom = row_sums + float(args.alpha) * Idims
    with np.errstate(invalid="ignore", divide="ignore"):
        Phat = np.divide(N_yip + float(args.alpha), denom)
    Phat = np.nan_to_num(Phat, nan=1.0 / Idims)

    log1p_counts = np.log1p(N_yip)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.35))
    for ax, Z, cmap, ylab in zip(
        axes,
        [log1p_counts, Phat],
        ["viridis", "magma"],
        [
            r"log₁p N[y, i′ | pooled rel set]",
            r"$\hat P(i^{\prime}\!\mid y)$ (Dirichlet smoothing over i′)",
        ],
    ):
        im = ax.imshow(Z, aspect="equal", origin="upper", cmap=cmap)
        ax.set_xticks(np.arange(Idims))
        ax.set_xticklabels([str(i + 1) for i in range(Idims)], fontsize=8)
        ax.set_yticks(np.arange(Cv))
        ax.set_yticklabels([str(y + 1) for y in range(Cv)], fontsize=8)
        ax.set_xlabel("source slice i′ (1-based)")
        ax.set_ylabel("plate label y")
        ax.set_title(ylab.replace(r"^{\prime}", "′"))
        plt.colorbar(im, ax=ax, shrink=0.82, fraction=0.046)

    pool_names = [RelationKind(r).name for r in sorted(rel_set)]
    fig.suptitle(
        f"{args.bundle.parent.name} · pool rel {sorted(rel_set)} ({', '.join(pool_names)})\n"
        f"splits={args.train_instances} · row Laplace α={args.alpha}",
        fontsize=11,
        y=1.06,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")
    ts = float(N_yip.sum())
    print(f"Σ N[y,i′]={ts:.0f}; row sums (per y): {N_yip.sum(axis=1)}")
    if ts <= 0.0:
        print(
            "[note] pooled mass is zero — see script docstring "
            "(e.g. pool rel 2 is cross-item-only; intra-plate data uses pool 1 or 0,1,3).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
