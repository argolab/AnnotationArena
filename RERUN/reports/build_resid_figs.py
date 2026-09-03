#!/usr/bin/env python
"""
Residual ablation, rebuilt on the test corner.

The report's original numbers are the pooled metric -- every missing token in
the graph, most of which sits on training-partition items (D22). This rebuilds
both the learning curves and the final table on the 414-cell corner the paper
says it reports.

    python build_resid_figs.py --in reports/build -o reports/build
"""
from __future__ import annotations
import argparse, json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ordered worst-to-best is deliberate: the legend then reads as the conclusion.
VARIANTS = [
    ("all",             "all — shipped (double stream)",      "#1b3a6b", "-",  2.6),
    ("none_id",         "none_id — I+M folded in",            "#2E7357", "-",  2.2),
    ("none",            "none — outer add deleted",           "#C2871A", "--", 2.0),
    ("all+inner_none",  "all + inner=none — inner add deleted", "#A33B31", ":", 2.0),
]
VAR_RE = re.compile(r"/ITEM/([^/]+)/seed(\d+)/")

def load(indir: Path, size: int):
    p = indir / f"resid_corner_curves_K{size}.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    out = defaultdict(dict)                      # variant -> seed -> [(ep, ll)]
    for path, series in raw.items():
        m = VAR_RE.search(path + "/")
        if not m:
            continue
        out[m.group(1)][int(m.group(2))] = [(s["epoch"], s["corner_log_loss"]) for s in series]
    return out

def mean_curve(per_seed):
    """Mean over seeds at each epoch present in all of them."""
    if not per_seed:
        return []
    common = set.intersection(*[{e for e, _ in c} for c in per_seed.values()])
    return [(e, st.mean(dict(c)[e] for c in per_seed.values())) for e in sorted(common)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="reports/build")
    ap.add_argument("--sizes", default="100,300,500")
    ap.add_argument("-o", "--outdir", default="reports/build")
    a = ap.parse_args()
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in a.sizes.split(",")]
    data = {s: load(indir, s) for s in sizes}
    have = [s for s in sizes if data[s]]
    if not have:
        raise SystemExit("no corner-curve payloads found")

    # ---- learning curves -------------------------------------------------
    fig, axes = plt.subplots(1, len(have), figsize=(5.0 * len(have), 4.4), squeeze=False)
    for ax, size in zip(axes[0], have):
        for key, label, colour, ls, lw in VARIANTS:
            cur = mean_curve(data[size].get(key, {}))
            if not cur:
                continue
            ax.plot([e for e, _ in cur], [v for _, v in cur], ls, color=colour, lw=lw,
                    label=label if size == have[0] else None)
        ax.axhline(1.3863, color="#787F92", lw=.8, ls=(0, (1, 3)))
        ax.text(ax.get_xlim()[1], 1.3863, " chance", va="center", fontsize=7,
                color="#787F92")
        ax.set_title(f"K = {size} training items", fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=.22, lw=.5)
    axes[0][0].set_ylabel("log-loss on the 414-cell test corner")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Residual ablation — corner log-loss through training, mean of 3 seeds",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "resid_curves_corner.png", dpi=170)

    # ---- final table -----------------------------------------------------
    table = []
    for size in have:
        base = {s: c[-1][1] for s, c in data[size].get("all", {}).items()}
        for key, label, *_ in VARIANTS:
            per = data[size].get(key, {})
            if not per:
                continue
            finals = {s: c[-1][1] for s, c in per.items()}
            mins = {s: min(v for _, v in c) for s, c in per.items()}
            paired = [finals[s] - base[s] for s in finals if s in base]
            row = {"size": size, "variant": key, "label": label,
                   "n_seeds": len(finals),
                   "final_mean": st.mean(finals.values()),
                   "final_sd": st.stdev(finals.values()) if len(finals) > 1 else 0.0,
                   "min_mean": st.mean(mins.values()),
                   "inflation": st.mean(finals.values()) - st.mean(mins.values()),
                   "paired_vs_all": st.mean(paired) if paired else None,
                   "paired_se": (st.stdev(paired) / len(paired) ** .5)
                                if len(paired) > 1 else None}
            table.append(row)
    (outdir / "resid_corner_table.json").write_text(json.dumps(table, indent=2))

    print(f"{'K':>5}{'variant':>22}{'final':>9}{'sd':>8}{'best':>9}{'inflation':>11}"
          f"{'vs all':>10}{'se':>8}")
    for r in table:
        pv = f"{r['paired_vs_all']:+.4f}" if r["paired_vs_all"] is not None else "—"
        se = f"{r['paired_se']:.4f}" if r["paired_se"] is not None else "—"
        print(f"{r['size']:>5}{r['variant']:>22}{r['final_mean']:>9.4f}{r['final_sd']:>8.4f}"
              f"{r['min_mean']:>9.4f}{r['inflation']:>+11.4f}{pv:>10}{se:>8}")
    print(f"\n-> {outdir/'resid_curves_corner.png'}")

if __name__ == "__main__":
    main()
