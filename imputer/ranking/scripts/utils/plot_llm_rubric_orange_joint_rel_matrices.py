#!/usr/bin/env python3
"""
Compare three |V| × |I| matrices built from Orange training counts emit[y_b, rel, i, v].

Same count construction as
``plot_llm_rubric_cpm_with_structured_baselines._orange_factorized_preprocess``:
rows indexed by emitted rating v′, columns by rubric dimension i′, conditioning plate class y_b.

1. **Generative**: per rel slice, smoothed joint over (i′, v′), then arithmetic mean across rel.

2. **Factored**: per rel,
   $\\hat{p}(v^{\\prime}\\!\\mid\\! i^{\\prime},y_b,r)\\,\\hat{p}(i^{\\prime}\\!\\mid\\! y_b,r)$
   then mean across rel.

3. **Factored, ablated “share item k”**: same as (2) averaging only relation buckets whose source
    does **not** share plate item coordinate k with tgt (drops rel buckets 0, 1, 3).

Optional ``--average-over-y`` adds a second row with each matrix averaged uniformly over plate
labels y_b ∈ {0,…,C−1} (average of per-y matrices after per-(y,r) normalization inside (1)-(3)).

Usage (from ``imputer/ranking``)::

    python scripts/utils/plot_llm_rubric_orange_joint_rel_matrices.py \\
        --bundle DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json
"""

from __future__ import annotations

import argparse
import math
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
from structured_baselines.feature_utils import relation_label


# Relation buckets where k_src == k_tgt on the Orange plate topology.
SHARED_ITEM_K_RELS = frozenset({0, 1, 3})


def _orange_emit_preprocess(train_examples, num_attrs: int, num_classes: int):
    """emit[y_plate, rel, i_src, v_src]; y_plate and v_src live on same C-way label set."""
    I, C_rating = num_attrs, num_classes
    Rdim = 7
    emit = np.zeros((C_rating, Rdim, I, C_rating), dtype=np.float64)
    for ex in train_examples:
        y_plate = int(ex.y)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y_plate, rel, i_s, v_s] += 1.0
    return emit, I, C_rating, Rdim


def _slice_joint_generative(rel_mat: np.ndarray, *, alpha: float, I: int, C_v: int) -> np.ndarray:
    """Laplace on I×V slice → categorical; return shape (V, I) rows v′."""
    K = rel_mat.astype(np.float64, copy=False) + float(alpha)
    s = float(K.sum())
    if s <= 0.0:
        return np.full((C_v, I), 1.0 / float(C_v * I), dtype=np.float64)
    return (K.T / s).astype(np.float64, copy=False)


def _slice_joint_factored(rel_mat: np.ndarray, *, alpha: float, I: int, C_v: int) -> np.ndarray:
    """Orange chain-rule joint on slice; return (V, I)."""
    rel_mat = rel_mat.astype(np.float64, copy=False)
    n_i = rel_mat.sum(axis=1)
    rel_total = float(n_i.sum())
    denom_ip = rel_total + float(alpha) * I
    if denom_ip <= 0.0:
        pi = np.full(I, 1.0 / I)
    else:
        pi = (n_i + float(alpha)) / denom_ip

    denom_v = n_i.reshape(I, 1) + float(alpha) * C_v
    with np.errstate(invalid="ignore", divide="ignore"):
        emit_c = np.divide(rel_mat + alpha, denom_v)
    emit_c = np.nan_to_num(emit_c, nan=0.0, posinf=0.0, neginf=0.0)

    joint_i_v = emit_c * pi.reshape(I, 1)
    return joint_i_v.T.astype(np.float64, copy=False)


def _average_over_rel_slices(
    emit: np.ndarray,
    y_known: int,
    *,
    alpha: float,
    I: int,
    C_v: int,
    rel_filter: frozenset[int] | None,
    mode: str,
) -> tuple[np.ndarray, int]:
    acc = np.zeros((C_v, I), dtype=np.float64)
    cnt = 0
    Rdim = emit.shape[1]
    for r in range(Rdim):
        if rel_filter is not None and r not in rel_filter:
            continue
        sl = emit[int(y_known), int(r), :, :]
        if mode == "joint":
            Jr = _slice_joint_generative(sl, alpha=alpha, I=I, C_v=C_v)
        elif mode == "factored":
            Jr = _slice_joint_factored(sl, alpha=alpha, I=I, C_v=C_v)
        else:
            raise ValueError(mode)
        acc += Jr
        cnt += 1
    if cnt == 0:
        return np.full((C_v, I), float("nan")), 0
    return acc / float(cnt), cnt


def _matrices_for_y(
    emit: np.ndarray,
    y_known: int,
    *,
    alpha: float,
    I: int,
    C_v: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mj, _ = _average_over_rel_slices(
        emit, y_known, alpha=alpha, I=I, C_v=C_v, rel_filter=None, mode="joint"
    )
    mf, _ = _average_over_rel_slices(
        emit, y_known, alpha=alpha, I=I, C_v=C_v, rel_filter=None, mode="factored"
    )
    no_share = frozenset(set(range(7)) - SHARED_ITEM_K_RELS)
    ma, _ = _average_over_rel_slices(
        emit,
        y_known,
        alpha=alpha,
        I=I,
        C_v=C_v,
        rel_filter=no_share,
        mode="factored",
    )
    return mj, mf, ma


def _draw_row(
    axs: tuple,
    mj: np.ndarray,
    mf: np.ndarray,
    ma: np.ndarray,
    *,
    I: int,
    C_v: int,
    vmax: float,
    row_label: str,
    titles: tuple[str, str, str],
):
    mats = [mj, mf, ma]
    for ax, Mat, ttl in zip(axs, mats, titles):
        loc_vmax = vmax if math.isfinite(vmax) and vmax > 0 else float(np.nanmax(Mat))
        if not math.isfinite(loc_vmax) or loc_vmax <= 0:
            loc_vmax = 1.0
        im = ax.imshow(Mat, aspect="equal", vmin=0.0, vmax=loc_vmax, origin="upper", cmap="inferno")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks(np.arange(I))
        ax.set_xticklabels([str(i + 1) for i in range(I)], fontsize=8)
        ax.set_yticks(np.arange(C_v))
        ax.set_yticklabels([str(v + 1) for v in range(C_v)], fontsize=8)
        ax.set_xlabel("rubric slice i′ (bundle 1-index labels)")
        ax.set_ylabel("emitted rating v′")
        fig = ax.figure
        fig.colorbar(im, ax=ax, shrink=0.78, fraction=0.046)

    axs[0].text(
        -0.18,
        1.02,
        row_label,
        transform=axs[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )


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
        / "PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_joint_rel_triptych.png",
    )
    ap.add_argument(
        "--train-instances",
        type=str,
        default="train,val",
        help="Comma-separated plate instances contributing counts.",
    )
    ap.add_argument("--y-fixed", type=int, default=0)
    ap.add_argument("--alpha-emit", type=float, default=1.0)
    ap.add_argument(
        "--average-over-y",
        action="store_true",
        help="Append a second row averaging the three matrices uniformly over plate y.",
    )
    args = ap.parse_args()

    inst = {s.strip() for s in args.train_instances.split(",") if s.strip()}
    bundle = load_bundle_dict(args.bundle)
    Idims, _Jdims, Cv = bundle_dims(bundle, args.bundle)
    train_ex = build_training_examples(bundle, instances=inst)
    emit_full, Ip, Cv_emit, _R_unused = _orange_emit_preprocess(train_ex, Idims, Cv)
    assert Ip == Idims == emit_full.shape[2]
    assert Cv == Cv_emit == emit_full.shape[0] == emit_full.shape[3]

    if not (0 <= args.y_fixed < Cv):
        raise SystemExit(f"--y-fixed must be in [0,{Cv}), got {args.y_fixed}")

    mj_fix, mf_fix, ma_fix = _matrices_for_y(
        emit_full,
        args.y_fixed,
        alpha=args.alpha_emit,
        I=Idims,
        C_v=Cv,
    )

    rows_mats_fix = [mj_fix, mf_fix, ma_fix]

    avg_row_mats = None
    if args.average_over_y:
        mj_ys = []
        mf_ys = []
        ma_ys = []
        for y in range(Cv):
            a, b, c = _matrices_for_y(emit_full, y, alpha=args.alpha_emit, I=Idims, C_v=Cv)
            mj_ys.append(a)
            mf_ys.append(b)
            ma_ys.append(c)
        avg_row_mats = (
            np.mean(np.stack(mj_ys, axis=0), axis=0),
            np.mean(np.stack(mf_ys, axis=0), axis=0),
            np.mean(np.stack(ma_ys, axis=0), axis=0),
        )

    all_stack = rows_mats_fix[:]
    if avg_row_mats is not None:
        all_stack.extend(list(avg_row_mats))

    vmax = float(np.max([np.nanmax(np.asarray(X)) for X in all_stack if np.any(np.isfinite(X))]))

    nrow = 2 if args.average_over_y else 1
    fig, axarr = plt.subplots(nrow, 3, figsize=(13.2, 4.55 * nrow), squeeze=False)

    titles_common = (
        "(1) Generative joint,\nmean over rel",
        "(2) Factored p(v′|i′)·p(i′),\nmean over rel",
        "(3) Factored omit same-item-{k},\nmean over remaining rel",
    )

    _draw_row(
        tuple(axarr[0]),
        mj_fix,
        mf_fix,
        ma_fix,
        I=Idims,
        C_v=Cv,
        vmax=vmax,
        row_label=f"Plate y_b = {args.y_fixed}",
        titles=titles_common,
    )

    if avg_row_mats is not None:
        _draw_row(
            tuple(axarr[1]),
            avg_row_mats[0],
            avg_row_mats[1],
            avg_row_mats[2],
            I=Idims,
            C_v=Cv,
            vmax=vmax,
            row_label="Mean over plate y_b",
            titles=titles_common,
        )

    fig.suptitle(
        f"{args.bundle.parent.name} ({args.train_instances}) α_emit={args.alpha_emit}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close()

    fv = (_frobenius(mj_fix, mf_fix), _frobenius(mj_fix, ma_fix), _frobenius(mf_fix, ma_fix))
    print(f"Saved: {args.output}")
    print(
        "Frob. ‖·‖_F diffs vs (2) [(1)-(2),(1)-(3),(2)-(3)]: "
        + ", ".join(f"{x:.4e}" for x in fv),
    )


def _frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel(), ord=2))


if __name__ == "__main__":
    main()
