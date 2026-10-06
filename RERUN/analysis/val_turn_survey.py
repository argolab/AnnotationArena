#!/usr/bin/env python
"""How early does the validation block turn, across every run we have?

At item/500 the val curve bottoms at epoch 105 and is 25% worse by 400, while the
test corner is 20% BETTER at 400. That is one cell and one seed. This surveys the
validation side of that question over every run with a val block -- 42 final-dual
cells (1 seed, 13 sizes x 2 arms x 2 axes) and 84 shaw-d2 cells (3 seeds, 4
architecture arms) -- to see whether an early turn is the rule.

Validation only, so it is free and needs no GPU. The other half -- whether the
corner keeps improving past the turn -- needs the corner scored at those epochs
and comes from fixed_budget_vs_f13.py (final-dual) and s7_curves.py (shaw-d2).
"""
from __future__ import annotations
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "RERUN")
from side.s7_curves import val_curve

RUNS = Path("RERUN/runs")


def smooth(v, w=5):
    return [(v[i][0], st.fmean(x[1] for x in v[max(0, i - 2):i + 3]))
            for i in range(len(v))]


def survey(label: str, roots: list[Path]):
    rows = []
    for root in roots:
        for run in sorted(root.rglob("*_MARFORMER")):
            if not (run / "training_history.json").is_file():
                continue
            vc = val_curve(run)
            if len(vc) < 150:
                continue
            sm = smooth(vc)
            d = dict(sm)
            n = len(sm)
            ep, best = min(sm[25:], key=lambda t: t[1])
            end = sm[-1][1]
            at400 = d.get(400)
            rows.append(dict(run=run.name, n=n, argmin=ep, best=best,
                             frac=ep / n, at400=at400, end=end))
    if not rows:
        print(f"\n{label}: no runs with a validation curve"); return rows
    print(f"\n=== {label}: {len(rows)} runs with a validation block ===")
    fr = [r["frac"] for r in rows]
    am = [r["argmin"] for r in rows]
    print(f"  val argmin epoch: median {int(st.median(am))}, range {min(am)}-{max(am)}")
    print(f"  as a fraction of the budget: median {st.median(fr):.2f}, "
          f"{sum(1 for f in fr if f < 0.5)}/{len(fr)} in the first half")
    have400 = [r for r in rows if r["at400"] is not None and r["argmin"] < 400]
    if have400:
        deg = [r["at400"] - r["best"] for r in have400]
        print(f"  of the {len(have400)} runs whose val turns BEFORE 400, val at 400 is")
        print(f"    worse by median {st.median(deg):+.4f} nats "
              f"({sum(1 for x in deg if x > 0)}/{len(deg)} worse, "
              f"{sum(1 for x in deg if x > 0.05)}/{len(deg)} worse by >0.05)")
    rise = [r["end"] - r["best"] for r in rows]
    print(f"  val at end of budget vs its own minimum: median {st.median(rise):+.4f} "
          f"({sum(1 for x in rise if x > 0)}/{len(rise)} rose)")
    return rows


fd = survey("final-dual (1 seed, 42 cells)",
            [RUNS / "final-dual/RESULTS/MARFORMER/DOMAIN3-SIGMOID",
             RUNS / "final-dual/RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID"])
sh = survey("shaw-d2-s101 (3 seeds, 4 arms, 84 cells)",
            [RUNS / "shaw-d2-s101/RESULTS"])
