#!/usr/bin/env python3
"""
Empirical marginal distributions from a tensor ``data_bundle.json`` (``all_ratings`` rows).

Plots:
  - P(rating | attribute=i) pooling all annotators and items (heatmap I × C), plus
    the same conditional for selected annotators j.
  - P(rating | annotator=j) pooling attributes and items (heatmap J × C), plus
    slices for selected attributes i.
  - For selected items k, the marginal over the 4 Likert bins pooling all (i, j).

Run from ``imputer/ranking``::

  python scripts/utils/plot_tensor_data_bundle_ic_jc_marginals.py \\
      --bundle DATA/STAN/DOMAIN3/ItemSplits/Transductive/Tensor_400_25_9_DOMAIN3_Item_T_400/data_bundle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _filter_rows(
    rows: list[dict],
    inst_keep: set[str] | None,
) -> list[dict]:
    out = []
    for r in rows:
        if inst_keep is not None and str(r["instance"]) not in inst_keep:
            continue
        out.append(r)
    return out


def _ic_table(rows: list[dict], j_filter: int | None) -> tuple[np.ndarray, int, int]:
    """Counts [I,C] (0-based rows/cols), 1-based indices in data."""
    filtered = [r for r in rows if j_filter is None or int(r["annotator"]) == j_filter]
    if not filtered:
        return np.zeros((0, 0), dtype=np.float64), 0, 0
    I = max(int(r["attribute"]) for r in filtered)
    C = max(int(r["value"]) for r in filtered)
    mtx = np.zeros((I, C), dtype=np.float64)
    for r in filtered:
        mtx[int(r["attribute"]) - 1, int(r["value"]) - 1] += 1.0
    return mtx, I, C


def _jc_table(rows: list[dict], i_filter: int | None) -> tuple[np.ndarray, int, int]:
    filtered = [r for r in rows if i_filter is None or int(r["attribute"]) == i_filter]
    if not filtered:
        return np.zeros((0, 0), dtype=np.float64), 0, 0
    J = max(int(r["annotator"]) for r in filtered)
    C = max(int(r["value"]) for r in filtered)
    mtx = np.zeros((J, C), dtype=np.float64)
    for r in filtered:
        mtx[int(r["annotator"]) - 1, int(r["value"]) - 1] += 1.0
    return mtx, J, C


def _row_normalize(m: np.ndarray) -> np.ndarray:
    s = m.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.divide(m, np.maximum(s, 1e-12))
    return np.nan_to_num(p, nan=0.0)


def _plot_heatmap(
    ax: plt.Axes,
    p: np.ndarray,
    ylabel: str,
    xlabels: list[str],
    yticklabels: list[str],
    title: str,
) -> None:
    im = ax.imshow(p, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(p.shape[1]))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(p.shape[0]))
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Likert bin (value)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _item_marginal(rows: list[dict], k: int, n_bins: int) -> np.ndarray:
    sub = [r for r in rows if int(r["item"]) == k]
    if not sub:
        return np.zeros(n_bins, dtype=np.float64)
    cts = np.zeros(n_bins, dtype=np.float64)
    for r in sub:
        cts[int(r["value"]) - 1] += 1.0
    return cts / max(cts.sum(), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to data_bundle.json (must contain all_ratings).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: PLOTS/DOMAIN3/tensor_bundle_marginals/<bundle_dir_name>).",
    )
    ap.add_argument(
        "--instances",
        type=str,
        default="all",
        help="Comma-separated: train,val,test or 'all'.",
    )
    ap.add_argument(
        "--example-js",
        type=str,
        default="1,2,3",
        help="Annotator ids (1-based) for I×C slices.",
    )
    ap.add_argument(
        "--example-is",
        type=str,
        default="1,5,9",
        help="Attribute ids (1-based) for J×C slices.",
    )
    ap.add_argument(
        "--example-items",
        type=str,
        default="1,25,100,200",
        help="Item ids (1-based) for Likert histograms.",
    )
    args = ap.parse_args()

    inst_keep = (
        None
        if args.instances.strip().lower() == "all"
        else {s.strip() for s in args.instances.split(",") if s.strip()}
    )

    with args.bundle.open() as f:
        bundle = json.load(f)
    rows = _filter_rows(bundle["all_ratings"], inst_keep)
    if not rows:
        raise SystemExit("No ratings after instance filter.")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _RANKING_ROOT / "PLOTS/DOMAIN3/tensor_bundle_marginals" / args.bundle.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_tag = args.instances.replace(",", "-")
    base = f"{args.bundle.parent.name}_inst-{inst_tag}"

    example_js = _parse_int_list(args.example_js)
    example_is = _parse_int_list(args.example_is)
    example_items = _parse_int_list(args.example_items)

    # --- I × C ---
    m_ic, I, C = _ic_table(rows, j_filter=None)
    p_ic = _row_normalize(m_ic)
    c_labels = [str(c + 1) for c in range(C)]
    i_labels = [str(i + 1) for i in range(I)]

    n_ic = len(example_js) + 1
    fig_ic, axes_ic = plt.subplots(1, n_ic, figsize=(4.0 * n_ic + 1, 4.2))
    if n_ic == 1:
        axes_ic = np.array([axes_ic])
    _plot_heatmap(
        axes_ic[0],
        p_ic,
        ylabel="Attribute i",
        xlabels=c_labels,
        yticklabels=i_labels,
        title=f"I×C | P(c|i), all j,k\n({base})",
    )
    for idx, j in enumerate(example_js, start=1):
        m_ij, _, Cj = _ic_table(rows, j_filter=j)
        if Cj != C:
            raise RuntimeError("Inconsistent C across slices")
        p_ij = _row_normalize(m_ij)
        _plot_heatmap(
            axes_ic[idx],
            p_ij,
            ylabel="Attribute i",
            xlabels=c_labels,
            yticklabels=i_labels,
            title=f"I×C | P(c|i, j={j})",
        )
    fig_ic.tight_layout()
    fig_ic.savefig(out_dir / f"{base}_heatmap_IxC.png", dpi=160)
    plt.close(fig_ic)

    # --- J × C ---
    m_jc, J, Cc = _jc_table(rows, i_filter=None)
    if Cc != C:
        raise RuntimeError("Inconsistent C")
    p_jc = _row_normalize(m_jc)
    j_labels = [str(j + 1) for j in range(J)]

    n_jc = len(example_is) + 1
    fig_jc, axes_jc = plt.subplots(1, n_jc, figsize=(4.0 * n_jc + 1, 4.2))
    if n_jc == 1:
        axes_jc = np.array([axes_jc])
    _plot_heatmap(
        axes_jc[0],
        p_jc,
        ylabel="Annotator j",
        xlabels=c_labels,
        yticklabels=j_labels,
        title=f"J×C | P(c|j), all i,k\n({base})",
    )
    for idx, i_attr in enumerate(example_is, start=1):
        m_ji, J2, _ = _jc_table(rows, i_filter=i_attr)
        if J2 != J:
            raise RuntimeError("Inconsistent J across slices")
        p_ji = _row_normalize(m_ji)
        _plot_heatmap(
            axes_jc[idx],
            p_ji,
            ylabel="Annotator j",
            xlabels=c_labels,
            yticklabels=j_labels,
            title=f"J×C | P(c|j, i={i_attr})",
        )
    fig_jc.tight_layout()
    fig_jc.savefig(out_dir / f"{base}_heatmap_JxC.png", dpi=160)
    plt.close(fig_jc)

    # --- Items: marginal Likert distribution ---
    n_items_plot = len(example_items)
    ncols = min(4, max(1, n_items_plot))
    nrows = int(np.ceil(n_items_plot / ncols))
    fig_b, axes_b = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 1.0, 2.8 * nrows + 1.2))
    if n_items_plot == 1:
        axes_flat = np.array([axes_b])
    else:
        axes_flat = np.atleast_1d(axes_b).ravel()
    for idx, ax in enumerate(axes_flat):
        if idx >= n_items_plot:
            ax.set_visible(False)
            continue
        k = example_items[idx]
        probs = _item_marginal(rows, k, C)
        n_k = float(probs.sum())
        if n_k == 0.0:
            print(
                f"WARNING: item k={k} has no rows in all_ratings (after instance filter). "
                "The bundle only stores observed tuples; many item IDs are never rated.",
                file=sys.stderr,
            )
        xs = np.arange(1, C + 1)
        ax.bar(xs, probs, color="#457b9d", edgecolor="white")
        ax.set_xticks(xs)
        ax.set_xlim(0.5, C + 0.5)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Likert bin")
        ax.set_ylabel("Frequency")
        note = (
            "(no observations)" if n_k == 0.0 else f"mean={float(np.dot(probs, xs)):.2f}"
        )
        ax.set_title(f"Item k={k} (all i, j)\n{note}")
        ax.grid(axis="y", alpha=0.3)
    fig_b.suptitle(f"Per-item pooled Likert marginal\n({base})", fontsize=11)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / f"{base}_items_likert_marginals.png", dpi=160)
    plt.close(fig_b)

    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
