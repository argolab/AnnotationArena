#!/usr/bin/env python
"""smECE as a size x method table, per axis. The calibration claim, laid out flat.

The claim the figure exists to support is that Marformer stays CALIBRATED IN SMALL
DOMAINS, so the small-size columns are the ones that carry it -- not the large ones.
Each cell is smECE with its bootstrap CI half-width; a cell is marked where
Marformer's gap to the best Stan variant at that size exceeds Marformer's own CI,
which is the convention the existing captions use.

    python smece_table.py              # both protocols
    python smece_table.py --protocol f13
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
ITEM_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ANNOT_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
STAN = ["Stan Oracle", "Stan Proj", "Stan Bin", "Stan Both"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("f13", "200", "both"), default="both")
    ap.add_argument("--ci", action="store_true", help="also print the CI half-width")
    a = ap.parse_args()

    g = json.loads((RUNS / "reports/smece_vs_stan_grid.json").read_text())["axes"]
    ntp = RUNS / "reports/smece_grid_nt.json"
    if ntp.is_file():
        nt = json.loads(ntp.read_text())["axes"]
        for axis in g:
            g[axis].update(nt.get(axis, {}))

    want = {"f13": ["F13"], "200": ["@200"], "both": ["F13", "@200"]}[a.protocol]
    for axis, sizes, col in (("item", ITEM_SIZES, "K"), ("annot", ANNOT_SIZES, "J")):
        for tag in want:
            rows = [r for r in (f"Marformer {tag}", f"Marformer NT {tag}") if r in g[axis]]
            rows += [s for s in STAN if s in g[axis]]
            if not rows:
                continue
            print(f"\n{'='*(16+9*len(sizes))}")
            print(f"smECE  --  {axis} axis, Marformer at {tag}"
                  f"   (bootstrap CI half-width in brackets)" if a.ci else
                  f"smECE  --  {axis} axis, Marformer at {tag}")
            print(f"{'='*(16+9*len(sizes))}")
            print(f"{col:<20}" + "".join(f"{s:>9}" for s in sizes))
            for r in rows:
                d = g[axis][r]
                line = f"{r:<20}"
                for s in sizes:
                    v = d.get(str(s))
                    line += "        -" if v is None else f"{v['smece']:>9.4f}"
                print(line)
                if a.ci:
                    line = f"{'':<20}"
                    for s in sizes:
                        v = d.get(str(s))
                        line += "        -" if v is None else f"{'['+format(v['ci'],'.3f')+']':>9}"
                    print(line)
            # verdict of Marformer vs the BEST Stan at each size
            me = g[axis].get(f"Marformer {tag}", {})
            line = f"{'verdict vs best Stan':<20}"
            for s in sizes:
                m = me.get(str(s))
                st_ = [g[axis][sv][str(s)]["smece"] for sv in STAN
                       if sv in g[axis] and str(s) in g[axis][sv]]
                if m is None or not st_:
                    line += "        -"; continue
                gap = m["smece"] - min(st_)
                line += f"{('better' if gap < 0 else 'on par' if gap < m['ci'] else 'WORSE'):>9}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
