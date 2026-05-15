#!/usr/bin/env python3
"""
For a fixed attribute ``i``, build 4×4 joint count heatmaps for selected annotator pairs
(two Likert scores on the **same** items when both rated that item).

The tensor observation protocol usually samples a small set of annotators per item, so any
pair (j1, j2) only co-occurs on a modest number of items --- the figure title reports ``n``.

Run from ``imputer/ranking``::

  python scripts/utils/plot_tensor_bundle_annot_pair_joint_heatmap.py \\
      --bundle DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold_C4/
              Tensor_400_25_9_ItemTest_SharedThreshold_300_C4/data_bundle.json \\
      --attribute 1 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]


def _filter_rows(rows: list[dict], inst_keep: set[str] | None) -> list[dict]:
    out = []
    for r in rows:
        if inst_keep is not None and str(r["instance"]) not in inst_keep:
            continue
        out.append(r)
    return out


def _item_ann_values(
    rows: list[dict], attr_1indexed: int
) -> dict[int, dict[int, int]]:
    """item -> { annotator -> value }; last write wins if duplicate triples."""
    m: dict[int, dict[int, int]] = defaultdict(dict)
    for r in rows:
        if int(r["attribute"]) != attr_1indexed:
            continue
        it = int(r["item"])
        j = int(r["annotator"])
        m[it][j] = int(r["value"])
    return dict(m)


def _joint_matrix(
    item_ann: dict[int, dict[int, int]],
    ja: int,
    jb: int,
) -> tuple[np.ndarray, int, list[int]]:
    M = np.zeros((4, 4), dtype=np.float64)
    overlap_items: list[int] = []
    for it, d in item_ann.items():
        if ja not in d or jb not in d:
            continue
        va, vb = d[ja], d[jb]
        if not (1 <= va <= 4 and 1 <= vb <= 4):
            continue
        M[va - 1, vb - 1] += 1.0
        overlap_items.append(it)
    return M, int(M.sum()), overlap_items


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = rx.std(ddof=0)
    sy = ry.std(ddof=0)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--attribute", type=int, default=1)
    ap.add_argument(
        "--instances",
        type=str,
        default="all",
        help="Comma-separated instances or all.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=10,
        help="Keep annotator pairs with at least this many jointly rated items.",
    )
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=6,
        help="Random pairs to plot (among those meeting min-overlap).",
    )
    ap.add_argument(
        "--sample-weighted-by-overlap",
        action="store_true",
        help="Bias random pair choice toward larger joint n (default: uniform among eligible pairs).",
    )
    ap.add_argument(
        "--explicit-pairs",
        type=str,
        default="",
        help="Comma-separated annotator pairs 'j1-j2:j1-j2'; if set overrides random picks.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path default under PLOTS/DOMAIN3/tensor_bundle_marginals/...",
    )
    args = ap.parse_args()

    inst_keep = (
        None
        if args.instances.strip().lower() == "all"
        else {s.strip() for s in args.instances.split(",") if s.strip()}
    )
    with args.bundle.open() as f:
        rows = _filter_rows(json.load(f)["all_ratings"], inst_keep)

    item_ann = _item_ann_values(rows, args.attribute)
    J_seen = sorted({int(r["annotator"]) for r in rows})

    cand: list[tuple[int, int, int]] = []
    for ja, jb in combinations(J_seen, 2):
        _, n_ij, _ = _joint_matrix(item_ann, ja, jb)
        if n_ij >= args.min_overlap:
            cand.append((n_ij, ja, jb))

    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    if not cand:
        raise SystemExit(
            f"No annotator pairs with overlap >= {args.min_overlap}. "
            f"Relax --min-overlap (max observed is often modest with num_annotate_annotator=4)."
        )

    rng = np.random.default_rng(args.seed)

    pairs: list[tuple[int, int]] = []
    if args.explicit_pairs.strip():
        for chunk in args.explicit_pairs.split(":"):
            chunk = chunk.strip()
            if not chunk:
                continue
            a, b = chunk.split("-")
            pairs.append((int(a), int(b)))
    else:
        npick = min(args.n_pairs, len(cand))
        if args.sample_weighted_by_overlap:
            weight = np.array([c[0] for c in cand], dtype=np.float64)
            weight = weight / weight.sum()
            idxs = rng.choice(len(cand), size=npick, replace=False, p=weight)
        else:
            idxs = rng.choice(len(cand), size=npick, replace=False)
        for ix in idxs:
            pairs.append((cand[ix][1], cand[ix][2]))

    nplot = len(pairs)
    ncols = min(3, nplot)
    nrows = int(np.ceil(nplot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for axidx, ax in enumerate(axes_flat):
        if axidx >= nplot:
            ax.set_visible(False)
            continue
        ja, jb = pairs[axidx]
        M, n_tot, overlap_items = _joint_matrix(item_ann, ja, jb)
        joint = M / max(n_tot, 1)

        ims = ax.imshow(joint, aspect="equal", vmin=0.0, vmax=max(0.12, joint.max()), cmap="BuPu")
        for r in range(4):
            for c in range(4):
                cn = int(M[r, c])
                ax.text(c, r, str(cn) if cn else "", ha="center", va="center", fontsize=10, color="white" if joint[r,c] > 0.06 else "#333333")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([str(ii + 1) for ii in range(4)])
        ax.set_yticklabels([str(ii + 1) for ii in range(4)])
        ax.set_xlabel(f"Annotator j={jb} Likert bin")
        ax.set_ylabel(f"Annotator j={ja} Likert bin")

        xs: list[float] = []
        ys: list[float] = []
        for it in overlap_items:
            d = item_ann[it]
            xs.append(float(d[ja]))
            ys.append(float(d[jb]))
        x_arr = np.array(xs)
        y_arr = np.array(ys)
        r_p = float(np.corrcoef(x_arr, y_arr)[0, 1]) if n_tot >= 2 else float("nan")
        r_sp = _spearman_corr(x_arr, y_arr)

        ax.set_title(
            f"pair (ja,jb)=({ja},{jb})  n={n_tot}\n"
            f"Pearson r={r_p:.2f}  Spearman rho={r_sp:.2f}"
        )
        plt.colorbar(ims, ax=ax, fraction=0.046, pad=0.04)

    attr = args.attribute
    fig.suptitle(
        f"Joint rating counts (row=ja, col=jb) | attribute i={attr}\n"
        f"{args.bundle.parent.name}  (seed={args.seed}, min_overlap={args.min_overlap})",
        fontsize=11,
    )
    fig.tight_layout()

    out = args.output
    if out is None:
        sub = _RANKING_ROOT / "PLOTS/DOMAIN3/tensor_bundle_marginals" / args.bundle.parent.name
        sub.mkdir(parents=True, exist_ok=True)
        out = sub / f"joint_4x4_i{attr}_seed{args.seed}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    print("top overlap pairs (n, ja, jb):", cand[:12])


if __name__ == "__main__":
    main()
