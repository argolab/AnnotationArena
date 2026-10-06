#!/usr/bin/env python
"""Marformer vs the Stan baselines, under BOTH training protocols, on both metrics.

The protocol question (F13 dev selection vs a fixed 200-epoch budget) is only worth
having if it changes where Marformer sits relative to Stan. This puts the two
protocols and the four Stan variants in one frame, per size, on corner log-loss and
corner smECE.

Stan Oracle is the data-generating model -- it is an upper bound, not a competitor.
Stan Proj is the near-oracle misspecification; Stan Bin and Stan Both are the
genuinely misspecified ones. Which of these the paper's "better or on par" claim is
against matters, so they are reported separately rather than pooled.

    python vs_stan.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
STAN = ["Stan Oracle", "Stan Proj", "Stan Bin", "Stan Both"]


def main() -> int:
    fx = json.loads((RUNS / "reports/tables_fixed200.json").read_text())["synthetic"]
    f13 = json.loads((RUNS / "reports/tables_f13.json").read_text())

    for axis, col in (("item", "K"), ("annot", "J")):
        stan = {s: {int(k): v for k, v in fx["axes"][axis].get(s, {}).items()}
                for s in STAN}
        mar = {
            "Marformer F13": {int(k): v for k, v in
                              f13["axes"][axis].get("Marformer (T)", {}).items()},
            "Marformer @200": {int(k): v for k, v in
                               fx["axes"][axis].get("Marformer (T)", {}).items()},
        }
        sizes = sorted(set(mar["Marformer @200"]) & set(mar["Marformer F13"]))

        for field, nm in (("ll", "corner log-loss"), ("smece", "corner smECE")):
            print(f"\n{'='*86}\n{axis.upper()}  --  {nm}: Marformer MINUS Stan "
                  f"(negative = Marformer BETTER)\n{'='*86}")
            hdr = f"{col:>6}" + "".join(f"{p:>16}" for p in mar)
            print(f"{'':<14}{hdr}")
            for sv in STAN:
                print(f"{sv}")
                for s in sizes:
                    if s not in stan[sv]:
                        continue
                    base = stan[sv][s].get(field)
                    if base is None:
                        continue
                    line = f"{'':<14}{s:>6}"
                    for p, d in mar.items():
                        v = d.get(s, {}).get(field)
                        line += "            --  " if v is None else f"{v - base:>+16.4f}"
                    print(line)
                # summary
                for p, d in mar.items():
                    diffs = [d[s][field] - stan[sv][s][field] for s in sizes
                             if s in stan[sv] and d.get(s, {}).get(field) is not None
                             and stan[sv][s].get(field) is not None]
                    if diffs:
                        better = sum(1 for x in diffs if x < 0)
                        print(f"{'':<14}  {p:<16} mean {st.mean(diffs):+.4f}   "
                              f"Marformer better in {better}/{len(diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
