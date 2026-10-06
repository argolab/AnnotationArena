#!/usr/bin/env python
"""Where is Marformer's smECE better than or on par with Stan -- and which size may
the paper legitimately report?

Reads smece_vs_stan_grid.json. Two separate things, deliberately not conflated:

  1. THE PRE-COMMITTED CHOICE. Largest size on each axis at which every baseline was
     fitted: K=4000, J=40 (Stan Oracle has no annot fit past 40). Fixed by a rule
     stated before looking at any number, so whatever it says is reportable.

  2. THE FULL GRID, for understanding only. Reading the best cell off this and
     reporting it is selection on test; it is printed so the shape of the
     size-dependence is visible, not so a winner can be chosen.

"on par" = gap smaller than the Marformer estimate's bootstrap CI half-width,
the convention the existing captions use. "better" = negative gap.

    python pick_calibration_size.py
"""
from __future__ import annotations
import json
from pathlib import Path

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
STAN = ["Stan Oracle", "Stan Proj", "Stan Bin", "Stan Both"]
PROTOCOLS = ["Marformer F13", "Marformer @200"]
PRECOMMIT = {"item": 4000, "annot": 40}


def verdict(gap: float, ci: float) -> str:
    if gap < 0:
        return "BETTER"
    return "on par" if gap < ci else "WORSE"


def main() -> int:
    g = json.loads((RUNS / "reports/smece_vs_stan_grid.json").read_text())["axes"]

    print("=" * 78)
    print("1. THE PRE-COMMITTED CHOICE  (largest size every baseline was fitted at)")
    print("=" * 78)
    for axis, size in PRECOMMIT.items():
        print(f"\n{axis} @ {size}:")
        for p in PROTOCOLS:
            me = g[axis].get(p, {}).get(str(size))
            if me is None:
                print(f"  {p:<16} no value"); continue
            print(f"  {p:<16} smECE {me['smece']:.4f} +/- {me['ci']:.4f}")
            for sv in STAN:
                s = g[axis].get(sv, {}).get(str(size))
                if s is None:
                    continue
                gap = me["smece"] - s["smece"]
                print(f"      vs {sv:<12} {s['smece']:.4f}   gap {gap:+.4f}   "
                      f"-> {verdict(gap, me['ci'])}")

    print("\n" + "=" * 78)
    print("2. FULL GRID -- for understanding only; choosing from it is selection on test")
    print("=" * 78)
    for axis in ("item", "annot"):
        for p in PROTOCOLS:
            print(f"\n{axis} / {p}:  verdict vs each Stan variant")
            sizes = sorted(int(k) for k in g[axis].get(p, {}))
            hdr = f"{'size':>6}{'smECE':>9}{'CI':>8}" + "".join(f"{s.split()[1]:>12}" for s in STAN)
            print(hdr)
            for s in sizes:
                me = g[axis][p][str(s)]
                line = f"{s:>6}{me['smece']:>9.4f}{me['ci']:>8.4f}"
                for sv in STAN:
                    st_ = g[axis].get(sv, {}).get(str(s))
                    line += f"{'--':>12}" if st_ is None else \
                        f"{verdict(me['smece'] - st_['smece'], me['ci']):>12}"
                print(line)
            # the smallest size at which it is better-or-on-par against ALL FOUR
            ok = []
            for s in sizes:
                me = g[axis][p][str(s)]
                vs = [g[axis].get(sv, {}).get(str(s)) for sv in STAN]
                vs = [v for v in vs if v is not None]
                if len(vs) == 4 and all(
                        verdict(me["smece"] - v["smece"], me["ci"]) != "WORSE" for v in vs):
                    ok.append(s)
            print(f"  better-or-on-par against ALL FOUR at: "
                  f"{ok if ok else 'no size'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
