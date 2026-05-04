#!/usr/bin/env python3
"""
Histogram-style summaries of ratings in an LLM Rubric ``data_bundle.json`` by rubric
dimension (attribute index i, 1-based in JSON → 1..I).

Plots:
  - observed count per dimension (and optional missing-slot count for the same dims);
  - heatmap of empirical P(value | dimension) from observed ratings only.

Run from ``imputer/ranking``:

  python scripts/utils/plot_llm_rubric_bundle_dim_rating_counts.py \\
      --bundle DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    _RANKING_ROOT / "PLOTS/TALK/LLM_RUBRIC/llm_rubric_bundle_rating_counts_by_dimension.png"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--bundle",
        type=Path,
        default=_RANKING_ROOT / "DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json",
    )
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--instances",
        type=str,
        default="all",
        help="Comma-separated instances to include on observed ratings (train,val,test); "
        "'all' means no filter.",
    )
    ap.add_argument(
        "--also-missing-bars",
        action="store_true",
        help="Second bar series for missing-slot rows (counts by attribute).",
    )
    args = ap.parse_args()

    with args.bundle.open() as f:
        bundle = json.load(f)

    inst_keep = (
        None
        if args.instances.strip().lower() == "all"
        else {s.strip() for s in args.instances.split(",") if s.strip()}
    )

    attr_obs = Counter()
    attr_mis = Counter()
    mtx = Counter()  # (attr 1-index, value 1-index) -> count

    for r in bundle["observed_ratings"]:
        if inst_keep is not None and str(r["instance"]) not in inst_keep:
            continue
        ai = int(r["attribute"])
        vi = int(r["value"])
        attr_obs[ai] += 1
        mtx[(ai, vi)] += 1

    if args.also_missing_bars:
        for r in bundle["missing_ratings"]:
            if inst_keep is not None and str(r["instance"]) not in inst_keep:
                continue
            attr_mis[int(r["attribute"])] += 1

    all_ar = bundle["observed_ratings"] + bundle["missing_ratings"]
    I = max(int(r["attribute"]) for r in all_ar)
    C = max(int(r["value"]) for r in all_ar)

    xs = np.arange(1, I + 1)
    y_obs = np.array([attr_obs.get(int(i), 0) for i in xs], dtype=float)
    y_mis = np.array([attr_mis.get(int(i), 0) for i in xs], dtype=float)

    count_mat = np.zeros((I, C), dtype=np.float64)
    for ai in range(1, I + 1):
        for vi in range(1, C + 1):
            count_mat[ai - 1, vi - 1] = float(mtx.get((ai, vi), 0))
    row_sums = count_mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_v_given_i = np.divide(count_mat, np.maximum(row_sums, 1e-12))
    p_v_given_i = np.nan_to_num(p_v_given_i, nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    w = 0.36
    x0 = xs - w / 2
    axes[0].bar(x0, y_obs, width=w, label="observed_ratings", color="#457b9d")
    if args.also_missing_bars:
        axes[0].bar(x0 + w, y_mis, width=w, label="missing_ratings slots", color="#f4a261")
    axes[0].set_xticks(xs)
    axes[0].set_xlabel("Rubric dimension (attribute index)")
    axes[0].set_ylabel("Number of ratings (rows)")
    axes[0].set_title(f"Ratings count by dimension ({args.bundle.parent.name})")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if not args.also_missing_bars:
        vmin = float(y_obs.min()) if len(y_obs) else 0.0
        vmax = float(y_obs.max()) if len(y_obs) else 1.0
        if vmin == vmax:
            axes[0].annotate(
                "All dimensions have the same observed count.",
                xy=(0.5, 0.96),
                xycoords="axes fraction",
                ha="center",
                fontsize=9,
                color="#555555",
            )

    im = axes[1].imshow(
        p_v_given_i,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xticks(np.arange(C))
    axes[1].set_xticklabels([str(v + 1) for v in range(C)])
    axes[1].set_yticks(np.arange(I))
    axes[1].set_yticklabels([str(i + 1) for i in range(I)])
    axes[1].set_xlabel("Rating value class (v, 1-based in bundle)")
    axes[1].set_ylabel("Rubric dimension (i)")
    axes[1].set_title(r"Empirical $\hat P(v\,|\, i)$ over observed filtered ratings")
    fig.colorbar(im, ax=axes[1], shrink=0.82, fraction=0.046, label="fraction of dim i rows")

    if inst_keep is not None:
        _ins = ",".join(sorted(inst_keep))
        fig.suptitle(f"instances ⊆ {{{_ins}}}", fontsize=9, y=1.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
