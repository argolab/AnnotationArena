#!/usr/bin/env python
"""Facts the writing agent needs, read off runs that already exist. No new experiments.

Covers coding tasks C2 (runtime), C3 (Stan sampler config), C5 (judge-axis status),
C7 (smallest-size fallback epochs). Everything is read from run artifacts; nothing is fitted.
"""
from __future__ import annotations
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs")
RUN = "final-v2"
CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- C2 runtime
def c2():
    print("=" * 78)
    print("C2  RUNTIME, final-v2, minutes.  search = the 2000-epoch validation search")
    print("    refit = retrain on 100% for the selected epochs.  total = what it costs")
    print("=" * 78)
    out = defaultdict(dict)
    for fam_dir, fam in (("MARFORMER", "T"), ("MARFORMER-NT", "NT")):
        for ax in ("ITEM", "ANNOT"):
            for stage, sub in (("search", "valblock-search/seed42"), ("refit", "valblock/seed42")):
                base = R / RUN / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID" / ax / sub
                if not base.is_dir():
                    continue
                for run in sorted(base.glob("*_MARFORMER")):
                    m = CELL.search(run.name)
                    if not m:
                        continue
                    d = jload(run / "launcher_metadata.json")
                    if d and d.get("elapsed_seconds") is not None:
                        out[(fam, ax.lower(), int(m.group(3)))][stage] = d["elapsed_seconds"] / 60.0
    for (fam, ax) in (("T", "item"), ("NT", "item"), ("T", "annot"), ("NT", "annot")):
        ks = sorted(k for k in out if k[0] == fam and k[1] == ax)
        if not ks:
            continue
        print(f"\n  Marformer {fam}, {ax} axis")
        print(f"    {'size':>6}{'search':>9}{'refit':>9}{'total':>9}")
        tot_s = tot_r = 0.0
        for k in ks:
            s = out[k].get("search"); r = out[k].get("refit")
            tot_s += s or 0; tot_r += r or 0
            f = lambda v: f"{v:9.1f}" if v is not None else f"{'--':>9}"
            t = (s or 0) + (r or 0)
            print(f"    {k[2]:>6}{f(s)}{f(r)}{t:9.1f}")
        print(f"    {'SUM':>6}{tot_s:9.1f}{tot_r:9.1f}{tot_s+tot_r:9.1f}")


# ------------------------------------------------------------ C3 Stan config
def c3():
    print("\n" + "=" * 78)
    print("C3  STAN SAMPLER CONFIGURATION -- is it uniform across every fit?")
    print("=" * 78)
    seen = defaultdict(list)
    base = R / RUN / "RESULTS" / "STAN"
    n = 0
    for cfg in base.rglob("configs.json"):
        d = jload(cfg)
        inf = (d or {}).get("inference")
        if not inf:
            continue
        n += 1
        key = (inf.get("chains"), inf.get("iter_warmup"), inf.get("iter_sampling"),
               inf.get("adapt_delta"), inf.get("max_treedepth"), inf.get("init_strategy"))
        seen[key].append(str(cfg.parent.name))
    print(f"  {n} Stan fits inspected under {RUN}")
    for key, cells in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  chains={key[0]} warmup={key[1]} sampling={key[2]} "
              f"adapt_delta={key[3]} max_treedepth={key[4]} init={key[5]}")
        print(f"    {len(cells)} fits, e.g. {', '.join(sorted(cells)[:3])}")
    print(f"\n  UNIFORM: {'YES' if len(seen) == 1 else 'NO -- ' + str(len(seen)) + ' distinct configurations'}")


# -------------------------------------------------------- C5 judge-axis status
def c5():
    print("\n" + "=" * 78)
    print("C5  JUDGE-AXIS STATUS -- which sizes actually exist, per series")
    print("=" * 78)
    for fam_dir, fam in (("MARFORMER", "Marformer T"), ("MARFORMER-NT", "Marformer NT")):
        base = R / RUN / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID" / "ANNOT" / "valblock/seed42"
        sizes = sorted(int(CELL.search(p.name).group(3)) for p in base.glob("*_MARFORMER")
                       if CELL.search(p.name))
        print(f"  {fam:<14} {sizes}")
    so = R / RUN / "RESULTS" / "STAN" / "DOMAIN3-SIGMOID" / "ORACLE" / "ANNOT"
    sizes = sorted({int(m.group(1)) for p in so.glob("*") if (m := re.search(r"_T_(\d+)_", p.name))})
    print(f"  {'Stan Oracle':<14} {sizes}")
    cal = R / RUN / "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/calibration_clean"
    have = sorted(int(m.group(1)) for p in cal.glob("*.png")
                  if (m := re.search(r"size(\d+)\.png", p.name)))
    print(f"  calibration panels rendered: {have}")


# ------------------------------------------------------ C7 fallback epochs
def c7():
    print("\n" + "=" * 78)
    print("C7  SELECTED EPOCHS, and the smallest-size fallback")
    print("=" * 78)
    d = jload(R / RUN / "valblock_epochs.json")
    if not d:
        print("  valblock_epochs.json not found"); return
    fb = d.get("fallback_by_axis", {})
    print(f"  selection rule : {d.get('select')}, smoothing window {d.get('smooth')}")
    print(f"  fallback epochs: {fb}")
    eps = d.get("epochs", {})
    used = [k for k, v in eps.items() if any(v == f for f in fb.values())]
    print(f"\n  cells with no validation block to carve (these take the axis median):")
    for axis, cells in (("item", ["Marformer|item|100|42", "MarformerNT|item|100|42"]),
                        ("annot", ["Marformer|annot|10|42", "MarformerNT|annot|10|42"])):
        for c in cells:
            if c in eps:
                flag = "  <- fallback" if eps[c] == fb.get(axis) else ""
                print(f"    {c:<30} epoch {eps[c]}{flag}")
    print(f"\n  cells whose epoch equals an axis fallback value: {len(used)}")


for f in (c2, c3, c5, c7):
    f()
