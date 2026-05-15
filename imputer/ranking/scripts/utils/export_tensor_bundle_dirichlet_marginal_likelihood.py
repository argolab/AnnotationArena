#!/usr/bin/env python3
"""
Per-annotator Dirichlet-multinomial (Polya) log marginal likelihood from ``all_ratings``.

If for annotator ``j`` the category counts ``x^(j)`` were generated as

    p^(j) ~ Dirichlet(alpha, ..., alpha)   with alpha = kappa/C
    x^(j)  ~ Multinomial(n_j; p^(j))

then the latent ``p^(j)`` can be integrated in closed form, giving::

    log P(x^(j)) = logΓ(n_j+1) - sum_c logΓ(x_c^(j)+1)
                 + logΓ(C*alpha) - logΓ(n_j + C*alpha)
                 + sum_c ( logΓ(x_c^(j) + alpha) - logΓ(alpha) )

where ``C * alpha == kappa`` under ``alpha := kappa/C`` for each symmetric component.

Interpretation caveat
---------------------
The simulator does **not** draw ``x^(j)`` exactly from Multi(n; p^(j)): thresholds are calibrated
against a surrogate mixture CDF, and scores come from dot products correlated across ``i`` sharing
each item. Treat this number as "**how typical are these multinomial margins if latent category
rates were IID Dirichlet-drawn and ratings were IID Multinomial**", not as the exact causal
density of Stan.

Run from ``imputer/ranking``::

  python scripts/utils/export_tensor_bundle_dirichlet_marginal_likelihood.py \\
      --bundle DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold_C4/
              Tensor_400_25_9_ItemTest_SharedThreshold_300_C4/data_bundle.json
"""

from __future__ import annotations

import argparse
import csv
import json
from math import lgamma
from pathlib import Path

import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]


def _kappa_from_metadir(bundle_path: Path) -> float | None:
    d = bundle_path.parent
    for name in ("stan_data.json", "configs.json"):
        p = d / name
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        if name == "stan_data.json":
            if "kappa" in obj:
                return float(obj["kappa"])
        else:
            dg = obj.get("datagen") or {}
            if "kappa" in dg:
                return float(dg["kappa"])
    return None


def log_dirichlet_multinomial_joint(counts_row: np.ndarray, alpha_scalar: float) -> float:
    """Symmetric Dirichlet: each category uses the same concentration ``alpha_scalar``."""
    x = counts_row.astype(np.float64).ravel()
    n = float(x.sum())
    C_local = len(x)
    a0 = C_local * alpha_scalar
    if n == 0.0:
        return 0.0
    ell = lgamma(n + 1.0) - np.sum([lgamma(float(xt) + 1.0) for xt in x])
    ell += lgamma(a0) - lgamma(n + a0)
    ell += sum(lgamma(float(xt) + alpha_scalar) - lgamma(alpha_scalar) for xt in x)
    return float(ell)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Total Dirichlet concentration (sum of symmetric alphas). Default: stan_data/datagen.",
    )
    ap.add_argument(
        "--instances",
        type=str,
        default="all",
        help="Comma-separated train,val,test or all.",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Default: beside bundle named dirichlet_dm_marginals.csv",
    )
    args = ap.parse_args()

    inst_keep = (
        None
        if args.instances.strip().lower() == "all"
        else {s.strip() for s in args.instances.split(",") if s.strip()}
    )

    bundle_path = args.bundle
    with bundle_path.open() as f:
        rows = json.load(f)["all_ratings"]
    if inst_keep is not None:
        rows = [r for r in rows if str(r["instance"]) in inst_keep]

    if not rows:
        raise SystemExit("No ratings after filtering.")

    J_count = max(int(r["annotator"]) for r in rows)
    C_cat = max(int(r["value"]) for r in rows)
    counts = np.zeros((J_count, C_cat), dtype=np.float64)
    for r in rows:
        j = int(r["annotator"]) - 1
        c = int(r["value"]) - 1
        counts[j, c] += 1

    kappa = args.kappa
    if kappa is None:
        kappa = _kappa_from_metadir(bundle_path)
    if kappa is None:
        raise SystemExit("Could not read kappa; pass --kappa explicitly.")
    alpha = kappa / C_cat

    out_csv = args.output_csv or (bundle_path.parent / "dirichlet_dm_marginals.csv")

    count_heads = [f"count_c{cc}" for cc in range(1, C_cat + 1)]
    phat_heads = [f"p_hat_c{cc}" for cc in range(1, C_cat + 1)]
    heads = ["annotator_j", "n_j", *count_heads, *phat_heads, "kappa_total", "alpha_symmetric", "log_DM_marginal"]

    total_log_joint = 0.0
    per_logs: list[tuple[int, float]] = []
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(heads)
        for j in range(J_count):
            x = counts[j]
            n_j = float(x.sum())
            phat = (x / n_j).tolist() if n_j > 0 else [0.0] * C_cat
            log_dm = log_dirichlet_multinomial_joint(x, alpha)
            total_log_joint += log_dm
            per_logs.append((j + 1, log_dm))
            w.writerow(
                [j + 1, int(n_j)]
                + [int(cv) for cv in x]
                + [f"{pv:.12g}" for pv in phat]
                + [f"{kappa:g}", f"{alpha:.12g}", f"{log_dm:.16g}"]
            )

    summary_path = out_csv.with_name(out_csv.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "instances_filter": args.instances,
                "kappa_total": float(kappa),
                "alpha_symmetric": float(alpha),
                "C_categories": int(C_cat),
                "annotators_J": int(J_count),
                "log_joint_DM_marginal_products_independent_anns": total_log_joint,
                "formula": (
                    "per row: conjugate multinomial marginal under p~DirSymmetric(kappa/C); independent across j"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_csv}")
    print(f"wrote {summary_path}")
    print(f"kappa={kappa:g} symmetric alpha=kappa/C={alpha:.12g}")
    print(f"log prod_j P_DM(x^(j)|kappa,C) assuming independent rows: {total_log_joint:.6f}")
    worst = sorted(per_logs, key=lambda t: t[1])[:5]
    best = sorted(per_logs, key=lambda t: -t[1])[:5]
    print("five smallest log_DM marginal annotators:", worst)
    print("five largest log_DM marginal annotators:", best)


if __name__ == "__main__":
    main()
