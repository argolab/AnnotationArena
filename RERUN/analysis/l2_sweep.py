#!/usr/bin/env python
"""Does more L2 change the overfitting, or only delay it?

Jason asked (2026-10-06) whether more weight decay would help, after the
overconfidence result. The sweep is one knob at five settings, paired across
K=300 and K=1000, everything else at the final-run config.

Two questions, not one:

  does the FLOOR drop?   min of the 5-smoothed dev curve. A better floor is a
                         better model under correct early stopping -- the only
                         thing that reaches the paper.
  does the RISE flatten? dev end minus dev floor. L2 that merely postpones the
                         turn-up is worth much less, because F13 stops at the
                         floor anyway.

The decision rule stated in advance: ship a larger weight decay only if it
improves the dev floor at K=300 by more than the model-seed spread AND does not
regress K=1000. Anything else and 0.01 stays.

    python l2_sweep.py
"""
from __future__ import annotations
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/l2-s101")
K_RE = re.compile(r"_T_(\d+)_MARFORMER$")


def curve(run: Path):
    # A run still in flight has train_config.json but not yet a history file.
    f = run / "training_history.json"
    if not f.is_file():
        return None
    h = json.loads(f.read_text())
    ce = [(int(e.get("epoch", i)),
           ((e.get("val_eval") or {}).get("metrics") or {}).get(
               "missing", {}).get("rating", {}).get("xent"))
          for i, e in enumerate(h)]
    ce = [(a, b) for a, b in ce if b is not None]
    if len(ce) < 50:
        return None
    sm = [(ce[i][0], st.mean(v for _, v in ce[max(0, i - 2): i + 3]))
          for i in range(len(ce))]
    ep, v = min(sm, key=lambda t: t[1])
    return {"floor": v, "argmin": ep, "end": sm[-1][1], "n": len(ce)}


def main() -> int:
    rows = []
    for f in sorted(ROOT.rglob("DOMAIN3-SIGMOID_Item_T_*_MARFORMER/train_config.json")):
        run = f.parent
        c = json.loads(f.read_text())
        cu = curve(run)
        if cu is None:
            print(f"  [incomplete] wd={c['training']['weight_decay']} "
                  f"K={K_RE.search(run.name).group(1)} seed={c['run']['seed']}")
            continue
        rows.append({"wd": c["training"]["weight_decay"],
                     "K": int(K_RE.search(run.name).group(1)),
                     "seed": c["run"]["seed"], **cu})
    if not rows:
        return 1

    by = defaultdict(list)
    for r in rows:
        by[(r["K"], r["wd"])].append(r)

    for K in sorted({r["K"] for r in rows}):
        print(f"\n=== K={K} ===")
        print(f"{'wd':>7}{'n':>4}{'dev floor':>12}{'argmin':>8}{'dev end':>10}{'rise':>9}")
        print("-" * 50)
        base = None
        for wd in sorted({r["wd"] for r in rows if r["K"] == K}):
            g = by[(K, wd)]
            if not g:
                continue
            fl = st.mean(r["floor"] for r in g)
            sd = st.stdev([r["floor"] for r in g]) if len(g) > 1 else 0.0
            en = st.mean(r["end"] for r in g)
            am = st.mean(r["argmin"] for r in g)
            if wd == 0.01:
                base = fl
            mark = "" if base is None else f"  ({fl - base:+.4f} vs 0.01)"
            print(f"{wd:>7}{len(g):>4}{fl:>12.4f}{am:>8.0f}{en:>10.4f}"
                  f"{en - fl:>9.4f}{mark}"
                  + (f"   sd {sd:.4f}" if len(g) > 1 else ""))

    seeds = [g for g in by.values() if len(g) > 1]
    if seeds:
        spread = st.mean(max(r["floor"] for r in g) - min(r["floor"] for r in g)
                         for g in seeds)
        print(f"\nmodel-seed spread in the dev floor: {spread:.4f} "
              f"(mean range over {len(seeds)} cells with >1 seed)")
        print("A weight-decay effect smaller than this is not an effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
