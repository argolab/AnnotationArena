#!/usr/bin/env python
"""Is K=500 special, or does argmin-on-test flatter calibration at every size?

The published (pre-rerun) numbers selected the reported checkpoint by argmin on the
TEST corner -- bug D14, which F13 exists to remove. At item/500 that is worth a lot:
smECE 0.030 at epoch ~400 against 0.077 at the epoch F13 picks. The question is
whether that cell is representative or extreme.

Isolation: SAME models, same data fraction, same architecture. Only the checkpoint
selection rule varies, read off the 85% search run at {F13-selected, 200, 400, 800}.

Two readings of "the old number", because D14 has two forms:
  d14_ll    pick the budget with the best corner LOG-LOSS, report ITS smECE.
            This is the actual published policy.
  d14_ce    pick the budget with the best corner smECE directly. Not what was done,
            but it bounds how flattering the old policy could ever be.

CAVEAT: this is a 4-point grid, so it UNDERSTATES the true argmin-on-test -- the
real one searched every checkpoint. Treat these gaps as lower bounds.

    python d14_smece_gap.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
LABELS = ["F13-selected", "fixed-200", "fixed-400", "fixed-800"]


def main() -> int:
    rows = json.loads((RUNS / "reports/budget_table.json").read_text())
    idx = {(r["family"], r["axis"], r["K"], r["label"]): r for r in rows}
    cells = sorted({(r["family"], r["axis"], r["K"]) for r in rows})

    recs = []
    for c in cells:
        rs = [idx[c + (l,)] for l in LABELS if c + (l,) in idx]
        rs = [r for r in rs if r.get("smece") is not None]
        f13 = idx.get(c + ("F13-selected",))
        if f13 is None or f13.get("smece") is None or len(rs) < 2:
            continue
        d14_ll = min(rs, key=lambda r: r["log_loss"])
        d14_ce = min(rs, key=lambda r: r["smece"])
        recs.append(dict(cell=c,
                         f13=f13["smece"],
                         d14_ll=d14_ll["smece"], d14_ll_budget=d14_ll["label"],
                         d14_ce=d14_ce["smece"],
                         gap_ll=f13["smece"] - d14_ll["smece"],
                         gap_ce=f13["smece"] - d14_ce["smece"]))

    for key, nm in (("gap_ll", "D14 by log-loss (the published policy)"),
                    ("gap_ce", "D14 by smECE directly (upper bound)")):
        v = [r[key] for r in recs]
        print(f"\n=== smECE gap, F13 minus {nm} ===")
        print(f"  n={len(v)}  mean {st.mean(v):+.4f} +/-{st.stdev(v)/len(v)**0.5:.4f}"
              f"   median {st.median(v):+.4f}   max {max(v):+.4f}")
        over = sum(1 for x in v if x > 0.015)
        print(f"  cells where F13 is worse by more than 0.015: {over}/{len(v)}")
        print(f"  cells where F13 is worse at all:              "
              f"{sum(1 for x in v if x > 0)}/{len(v)}")

    print("\n=== every cell, sorted by the published-policy gap ===")
    print(f"{'cell':<26}{'F13':>9}{'D14(ll)':>10}{'gap':>9}{'budget D14 picked':>20}")
    for r in sorted(recs, key=lambda r: r["gap_ll"], reverse=True):
        c = r["cell"]
        star = "   <-- item/500" if c == ("Marformer", "item", 500) else ""
        print(f"{c[0]+'/'+c[1]+'/'+str(c[2]):<26}{r['f13']:>9.4f}{r['d14_ll']:>10.4f}"
              f"{r['gap_ll']:>+9.4f}{r['d14_ll_budget']:>20}{star}")

    tgt = [r for r in recs if r["cell"] == ("Marformer", "item", 500)]
    if tgt:
        g = tgt[0]["gap_ll"]
        rank = sum(1 for r in recs if r["gap_ll"] >= g)
        v = [r["gap_ll"] for r in recs]
        print(f"\nitem/500 gap {g:+.4f} ranks {rank}/{len(recs)} "
              f"(median {st.median(v):+.4f}); it is "
              f"{g / st.median(v):.1f}x the median cell" if st.median(v) > 0 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
