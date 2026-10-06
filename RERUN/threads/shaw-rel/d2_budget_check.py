#!/usr/bin/env python
"""Is the fixed epoch budget binding for any arm?

The comparison reads each run at the epoch its own validation block selects, so
the budget only has to be long enough to contain that epoch for EVERY arm. If
one arm's argmin sits at the end of the budget, that arm is being read before
its optimum and "worse" cannot be told apart from "slower" -- which is exactly
the learning-dynamics claim under test, so it would hollow out the experiment.

Prints the smoothed validation argmin against the budget, and how much of the
improvement happens after epoch 400, so a late argmin on a flat noisy tail is
not mistaken for a curve still descending.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from side.s7_curves import val_curve

ROOT = Path(__file__).resolve().parents[2] / "runs/shaw-d2-s101/RESULTS"


def smooth(v, w=5):
    out = []
    for i in range(len(v)):
        lo, hi = max(0, i - w // 2), min(len(v), i + w // 2 + 1)
        out.append((v[i][0], sum(x[1] for x in v[lo:hi]) / (hi - lo)))
    return out


def main() -> None:
    print("smoothed val (5), 25-epoch floor as in F13\n")
    hdr = (f"{'cell':<32}{'budget':>7}{'argmin':>8}{'val*':>9}"
           f"{'val@400':>9}{'val@end':>9}{'400->min':>10}{'min->end':>10}")
    print(hdr); print("-" * len(hdr))
    late = []
    for r in sorted(ROOT.rglob("*_MARFORMER")):
        if not (r / "training_history.json").is_file():
            continue
        vc = val_curve(r)
        if len(vc) < 100:
            continue
        sv = smooth(vc)
        d = dict(sv)
        ep, best = min(sv[25:], key=lambda t: t[1])
        tag, seed = r.parts[-4], r.parts[-2]
        cell = r.name.replace("DOMAIN3-SIGMOID_", "").replace("_MARFORMER", "")
        n = len(sv)
        v400 = d.get(400, float("nan"))
        vend = sv[-1][1]
        frac = ep / n
        name = f"{tag} {seed} {cell}"
        flag = "  <== argmin in the last 15% of the budget" if frac > 0.85 else ""
        if flag:
            late.append(name)
        print(f"{name:<32}{n:>7}{ep:>8}{best:>9.4f}{v400:>9.4f}{vend:>9.4f}"
              f"{v400 - best:>+10.4f}{vend - best:>+10.4f}{flag}")
    print()
    if late:
        print(f"{len(late)} cell(s) may be truncated by the budget:")
        for x in late:
            print(f"  {x}")
        print("\nRead the '400->min' column: if it is near zero the tail is flat and a late\n"
              "argmin is noise, not a descending curve. If it is clearly negative the\n"
              "budget is binding and that arm needs a longer search.")
    else:
        print("No cell has its argmin in the last 15% of its budget.")


if __name__ == "__main__":
    main()
