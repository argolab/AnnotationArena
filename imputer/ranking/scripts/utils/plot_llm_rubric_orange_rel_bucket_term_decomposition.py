#!/usr/bin/env python3
"""
Orange relation-bucket decomposition (matches ``_orange_factorized_eval`` log terms).

For a **fixed** relation bucket ``rel0`` (default: SAME_ITEM_SAME_ATTR_DIFF_ANNOT),

  full(i′, v, y; rel0)   = log P(v | i′, y, rel0) + log P(i′ | y, rel0)
  ablated(i′, v, y; rel0) = log P(v | i′, y, rel0)
  Δ(i′, y; rel0)       = log P(i′ | y, rel0)

Counts come from ``emit[y, rel, i, v]`` — same preprocessing as the main plotting script::

  structured_baselines.relation_label + LOO-style training LOCALExample targets.

Heatmaps (Panel A):
  Rows    = candidate target class y_C (indexed like Orange ``emit`` axis 0)
  Columns = source rubric slice i′
  Cell(y, i′) averages over emitted source values ``v`` **observed at** (y, rel0, i′)
      in ``emit``: weight ``n(y,rel0,i′,v)/Z_{y,i′}``.
  If Z_{y,i′}=0 the cell stays NaN (empty).

Secondary curves (Panel B, fixed slice i_focus):
  vs y ∈ {0,…,C−1}: full-cell, emission-only cell, and dropped log Pi.

Run from ``imputer/ranking``::

  python scripts/utils/plot_llm_rubric_orange_rel_bucket_term_decomposition.py
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
from structured_baselines.feature_utils import RelationKind, relation_label


def _orange_preprocess(train_examples, num_attrs: int, num_classes: int):
    I, Cv = num_attrs, num_classes
    Rd = 7
    emit = np.zeros((Cv, Rd, I, Cv), dtype=np.float64)
    for ex in train_examples:
        y_tgt = int(ex.y)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y_tgt, rel, i_s, v_s] += 1.0
    return emit, I, Cv, Rd


def _log_emit_v(
    rel_mat: np.ndarray,
    *,
    i_prime: int,
    v_obs: int,
    alpha_emit: float,
    C_rating: int,
) -> float:
    n_row = float(rel_mat[i_prime, :].sum())
    denom = n_row + alpha_emit * C_rating
    cnt = float(rel_mat[i_prime, v_obs])
    return math.log((cnt + alpha_emit) / denom)


def _log_pi_i(
    rel_mat: np.ndarray,
    *,
    i_prime: int,
    alpha_emit: float,
    Idims: int,
) -> float:
    rel_total = float(rel_mat.sum())
    n_i = float(rel_mat[i_prime, :].sum())
    denom_ip = rel_total + alpha_emit * float(Idims)
    return math.log((n_i + alpha_emit) / denom_ip)


def _fill_heat_triplet(
    emit: np.ndarray,
    *,
    rel0: int,
    alpha_emit: float,
    Idims: int,
    Cv: int,
    v_average: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    v_average ∈ {"empirical", "uniform"} — how to average emission log terms over supportive v.
    """
    hf = np.full((Cv, Idims), np.nan, dtype=np.float64)
    he = np.full((Cv, Idims), np.nan, dtype=np.float64)
    hd = np.full((Cv, Idims), np.nan, dtype=np.float64)

    for y_row in range(Cv):
        rel_mat = emit[y_row, rel0, :, :].astype(np.float64)
        for ip in range(Idims):
            row_counts = rel_mat[ip, :]
            zs = float(row_counts.sum())
            log_pi_term = _log_pi_i(rel_mat, i_prime=ip, alpha_emit=alpha_emit, Idims=Idims)

            if zs <= 0.0:
                continue

            if v_average == "empirical":
                acc_e = 0.0
                for v_obs in range(Cv):
                    nv = row_counts[v_obs]
                    if nv <= 0.0:
                        continue
                    w = nv / zs
                    le = _log_emit_v(
                        rel_mat,
                        i_prime=ip,
                        v_obs=v_obs,
                        alpha_emit=alpha_emit,
                        C_rating=Cv,
                    )
                    acc_e += w * le
                he[y_row, ip] = acc_e
                hf[y_row, ip] = acc_e + log_pi_term
                hd[y_row, ip] = log_pi_term

            elif v_average == "uniform":
                nz = []
                for v_obs in range(Cv):
                    if row_counts[v_obs] <= 0.0:
                        continue
                    nz.append(v_obs)
                if not nz:
                    continue
                ww = 1.0 / float(len(nz))
                acc_e = 0.0
                for v_obs in nz:
                    acc_e += ww * _log_emit_v(
                        rel_mat,
                        i_prime=ip,
                        v_obs=v_obs,
                        alpha_emit=alpha_emit,
                        C_rating=Cv,
                    )
                he[y_row, ip] = acc_e
                hf[y_row, ip] = acc_e + log_pi_term
                hd[y_row, ip] = log_pi_term
            else:
                raise ValueError(v_average)

    return hf, he, hd


def _masked_imshow(ax, Mat: np.ndarray, *, title: str, Cv: int, Idims: int, cmap: str = "magma"):
    finite = Mat[np.isfinite(Mat)].ravel()
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            vmin -= 1e-12
            vmax += 1e-12
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(Mat, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(Idims))
    ax.set_xticklabels([str(i + 1) for i in range(Idims)], fontsize=7)
    ax.set_yticks(np.arange(Cv))
    ax.set_yticklabels([str(y + 1) for y in range(Cv)], fontsize=7)
    ax.set_xlabel("source slice i′ (bundle 1-based labels)")
    ax.set_ylabel("candidate plate class y")
    plt.colorbar(im, ax=ax, shrink=0.72, fraction=0.046)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        type=Path,
        default=_RANKING_ROOT / "DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_RANKING_ROOT
        / "PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_rel0_term_decomposition.png",
    )
    ap.add_argument(
        "--train-instances",
        type=str,
        default="train,val",
        help="Comma-separated LOCALExample splits feeding emit.",
    )
    ap.add_argument(
        "--rel0",
        type=int,
        choices=list(range(7)),
        default=int(RelationKind.SAME_ITEM_SAME_ATTR_DIFF_ANNOT),
        help="Fixed Orange relation bucket.",
    )
    ap.add_argument(
        "--alpha-emit",
        type=float,
        default=1.0,
        help="Orange Laplace (same denominator pattern as scorer).",
    )
    ap.add_argument(
        "--v-average",
        choices=("empirical", "uniform"),
        default="empirical",
        help="Empirical-frequency vs uniform averaging over supportive v counts.",
    )
    ap.add_argument(
        "--focus-i-prime-index",
        type=int,
        default=8,
        help="Zero-based bundle attribute index i′ tabulated in curves (default 8 = ninth criterion).",
    )
    args = ap.parse_args()

    inst_keep = {s.strip() for s in args.train_instances.split(",") if s.strip()}
    bundle = load_bundle_dict(args.bundle)
    Idims, _Jdims, Cv = bundle_dims(bundle, args.bundle)
    train_ex = build_training_examples(bundle, instances=inst_keep)

    emit, Ip, Cv_emit, _Rd = _orange_preprocess(train_ex, Idims, Cv)
    assert Ip == Idims and Cv_emit == Cv

    hf, he, hd = _fill_heat_triplet(
        emit,
        rel0=args.rel0,
        alpha_emit=args.alpha_emit,
        Idims=Idims,
        Cv=Cv,
        v_average=args.v_average,
    )

    if not (0 <= args.focus_i_prime_index < Idims):
        raise SystemExit(
            f"--focus-i-prime-index must satisfy 0≤i<{Idims}, got {args.focus_i_prime_index}"
        )

    ip = args.focus_i_prime_index
    y_axis = np.arange(Cv)

    fig = plt.figure(figsize=(14.2, 6.4))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[2.05, 1.0],
        hspace=0.35,
        wspace=0.28,
        left=0.06,
        right=0.99,
        top=0.91,
        bottom=0.10,
    )
    axes_h = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    for ax, Mat, ttl in zip(
        axes_h,
        [hf, he, hd],
        [
            "Full: E*[log P(v|i′,y,r0)] + log P(i′|y,r0)",
            "Emission: E*[log P(v|i′,y,r0)]",
            "Dropped term: log P(i′|y,r0)",
        ],
    ):
        _masked_imshow(ax, Mat, title=ttl.strip(), Cv=Cv, Idims=Idims)

    resid = hf - he - hd
    resid_mask = np.isfinite(resid)
    if resid_mask.any():
        mr = float(np.nanmax(np.abs(resid[resid_mask])))
        if mr > 1e-9:
            print(f"[warn] max|full - emission - pi|={mr}")

    rel_name = RelationKind(args.rel0).name
    fig.suptitle(
        f"{args.bundle.parent.name} | rel₀={rel_name}\nα_emit={args.alpha_emit}; "
        f"v-average={args.v_average}; splits={args.train_instances}",
        fontsize=10,
    )

    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    ax_curve.plot(y_axis + 1, hf[:, ip], marker="o", label="full", linewidth=1.6)
    ax_curve.plot(y_axis + 1, he[:, ip], marker="s", label="emission-only", linewidth=1.6)
    ax_curve.plot(y_axis + 1, hd[:, ip], marker="^", label="log P(i′|y,r₀)", linewidth=1.6)
    ax_curve.set_xticks(np.arange(1, Cv + 1))
    ax_curve.set_xlabel("candidate plate class y (1-based)")
    ax_curve.set_ylabel("average log-contribution")
    ttl_ip = ip + 1
    ax_curve.set_title(f"Slices for bundle i′ = {ttl_ip} [0-index {ip}], same r₀ bucket")
    ax_curve.grid(alpha=0.33)
    ax_curve.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close()

    nz_cells = int(np.sum(np.isfinite(hf)))
    print(f"Saved: {args.output}")
    print(
        f"finite heatmap triple-cells ({rel_name}): {nz_cells}/{Cv * Idims} "
        f"(NaN ⇒ no arcs with that (y, i′, r₀) support)",
    )


if __name__ == "__main__":
    main()
