#!/usr/bin/env python
"""Does the validation block pick the epoch the test corner wants?

Two independent sources:
  * runs/final-dual        -- 42 cells, 1 seed, the shipped architecture
                              (reports/budget_table.json, 4 budgets per cell)
  * runs/shaw-d2-s101      -- 84 runs, 3 seeds, 4 architecture arms
                              (threads/shaw-rel/out_d2.json, ~15 checkpoints per run)

The Shaw sweep is the cleaner read: it records val_argmin_epoch and
corner_argmin_epoch per run over a dense checkpoint grid, so the two optima are
compared directly rather than through a handful of fixed budgets.

`corner_at_val_argmin - corner_at_corner_argmin` is the selector's regret. It is
non-negative by construction (the corner argmin is an oracle over the same grid),
so its SIZE is the finding, not its sign. What carries information about direction
is the epoch gap: if the carve stopped too EARLY, corner_argmin > val_argmin.

    python val_corner_divergence.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RERUN = HERE.parent


def main() -> int:
    rows = json.loads((RERUN / "threads/shaw-rel/out_d2.json").read_text())
    ok = [r for r in rows
          if r.get("val_argmin_epoch") is not None
          and r.get("corner_argmin_epoch") is not None
          and r.get("corner_at_val_argmin") is not None]
    skipped = len(rows) - len(ok)
    print(f"shaw-d2: {len(ok)}/{len(rows)} runs usable"
          f"{f' ({skipped} have no validation block to carve)' if skipped else ''}")

    gap = [r["corner_argmin_epoch"] - r["val_argmin_epoch"] for r in ok]
    reg = [r["corner_at_val_argmin"] - r["corner_at_corner_argmin"] for r in ok]

    later = sum(1 for g in gap if g > 0)
    earlier = sum(1 for g in gap if g < 0)
    same = sum(1 for g in gap if g == 0)
    print(f"\n=== epoch gap: corner_argmin - val_argmin ===")
    print(f"  corner wants a LATER epoch : {later}/{len(gap)}")
    print(f"  corner wants an EARLIER one: {earlier}/{len(gap)}")
    print(f"  identical                  : {same}/{len(gap)}")
    print(f"  mean {st.mean(gap):+.1f} +/-{st.stdev(gap)/len(gap)**0.5:.1f} epochs"
          f"   median {st.median(gap):+.0f}   range [{min(gap)}, {max(gap)}]")
    # Sign test on the gap, ignoring ties: is "later" more common than chance?
    n = later + earlier
    if n:
        print(f"  sign test (ties dropped): {later}/{n} later = {100*later/n:.0f}%")

    print(f"\n=== selector regret on the corner (nats) ===")
    print(f"  mean {st.mean(reg):+.4f} +/-{st.stdev(reg)/len(reg)**0.5:.4f}"
          f"   median {st.median(reg):+.4f}   max {max(reg):+.4f}")
    big = sum(1 for v in reg if v > 0.015)
    print(f"  runs where regret exceeds the 0.015-nat threshold: {big}/{len(reg)}")

    print(f"\n=== by architecture arm (regret, nats) ===")
    for arm in sorted({r.get("rel_arm", r["run"].split("/DOMAIN3-SIGMOID/")[1].split("/")[1])
                       for r in ok}):
        sub = [r for r in ok
               if r["run"].split("/DOMAIN3-SIGMOID/")[1].split("/")[1] == arm]
        if not sub:
            continue
        v = [r["corner_at_val_argmin"] - r["corner_at_corner_argmin"] for r in sub]
        g = [r["corner_argmin_epoch"] - r["val_argmin_epoch"] for r in sub]
        print(f"  {arm:<10} regret {st.mean(v):+.4f}  gap {st.mean(g):+6.1f} ep  n={len(v)}")

    print(f"\n=== by axis ===")
    for ax in ("item", "annot"):
        sub = [r for r in ok if r["axis"] == ax]
        if not sub:
            continue
        v = [r["corner_at_val_argmin"] - r["corner_at_corner_argmin"] for r in sub]
        g = [r["corner_argmin_epoch"] - r["val_argmin_epoch"] for r in sub]
        print(f"  {ax:<6} regret {st.mean(v):+.4f} +/-{st.stdev(v)/len(v)**0.5:.4f}"
              f"  gap {st.mean(g):+6.1f} ep  later {sum(1 for x in g if x>0)}/{len(g)}  n={len(v)}")

    print(f"\n=== by seed (is this one unlucky draw?) ===")
    for sd in sorted({r["seed"] for r in ok}):
        sub = [r for r in ok if r["seed"] == sd]
        v = [r["corner_at_val_argmin"] - r["corner_at_corner_argmin"] for r in sub]
        g = [r["corner_argmin_epoch"] - r["val_argmin_epoch"] for r in sub]
        print(f"  seed {sd}  regret {st.mean(v):+.4f}  gap {st.mean(g):+6.1f} ep"
              f"  later {sum(1 for x in g if x>0)}/{len(g)}  n={len(v)}")

    print(f"\n=== worst 8 runs by regret ===")
    for r in sorted(ok, key=lambda r: r["corner_at_val_argmin"]
                    - r["corner_at_corner_argmin"], reverse=True)[:8]:
        arm = r["run"].split("/DOMAIN3-SIGMOID/")[1].split("/")[1]
        d = r["corner_at_val_argmin"] - r["corner_at_corner_argmin"]
        print(f"  {arm:<9}{r['axis']}/{r['size']:<5} s{r['seed']}  "
              f"val@{r['val_argmin_epoch']:<4} -> {r['corner_at_val_argmin']:.4f}   "
              f"corner@{r['corner_argmin_epoch']:<4} -> "
              f"{r['corner_at_corner_argmin']:.4f}   regret {d:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
