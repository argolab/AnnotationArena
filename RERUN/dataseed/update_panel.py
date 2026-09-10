#!/usr/bin/env python
"""
The September panel: what changed since the corrected Figure 6 was drawn.

Three things, all of which need the same data and none of which were visible on
the earlier panels:

1. **Stan now has points past K=500.** The paper stops the Stan curve at 500 for
   HMC cost and draws it flat beyond, and the earlier panels reproduced that
   dotted flat line as a reference rather than a measurement. Stan Proj and Stan
   Both have since been fit out to K=4000 on both draws. Stan does NOT flatten:
   Proj goes on improving all the way out, so the flat extension understated it.

2. **The gap is the stable quantity, not the level.** The absolute corner
   log-loss moves ~0.08 nats between generative draws -- the whole measured
   draw-to-draw sd -- but the Marformer-minus-Stan difference moves 0.001-0.02 at
   K>=1000, because a hard draw is hard for both models. That is what decides how
   many draws the final figure needs, and it is invisible on a single-draw panel.

3. **The epoch fix (F13) lands where Stan is not drawn.** Its gain is 0.002 at
   K=1000 and 0.023 at K=4000. Plotted against the extended Stan curve it is
   clear that it does not close the gap; it is worth having, and it is not the
   answer to the gap.

Job-only (see the "never compute on the login node" note): matplotlib rendering
and JSON aggregation both belong in a job, however small they look.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("dataseed/out")

# F13 (raised cap + per-run carve selection + refit) measured against the flat
# 400-epoch budget, mean of model seeds 42 and 43. FUTURE_EXPERIMENTS, "How F13
# is implemented". These are DELTAS from a different training run on the same
# draw and seed, applied here to the flat-400 curve; they are an estimate of
# where the shipped protocol would put the curve, not an independently scored
# series, and the panel says so.
F13_DELTA = {1000: -0.0022, 2000: -0.0116, 4000: -0.0228}

STAN_STYLE = {
    "Stan Oracle": dict(color="#0d47a1", marker="^", ls="-"),
    "Stan Proj":   dict(color="#42a5f5", marker="X", ls="-"),
    "Stan Both":   dict(color="#1565c0", marker="s", ls="-"),
}
MARF_STYLE = {
    "Marformer":    dict(color="#B02A37", marker="o", ls="-",
                         label="Marformer (T), 400 epochs"),
    "MarformerNT":  dict(color="#E08379", marker="s", ls="-",
                         label="Marformer NT, 400 epochs"),
}


def load():
    stan = {}
    for r in json.loads((OUT / "stan_corner.json").read_text())["rows"]:
        stan.setdefault(r["data_seed"], {}).setdefault(r["model"], {})[r["size"]] = r["corner"]
    marf = {}
    for ds in (101, 105):
        t = json.loads((OUT / f"seed{ds}_table.json").read_text())
        for fam in MARF_STYLE:
            marf.setdefault(ds, {})[fam] = {
                r["size"]: r["corner"] for r in t["rows"] if r["family"] == fam}
    return marf, stan


def best_stan(stan_ds, k):
    vals = [m[k] for m in stan_ds.values() if k in m]
    return min(vals) if vals else None


def main() -> None:
    marf, stan = load()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, ds in zip(axes, (101, 105)):
        for fam, style in MARF_STYLE.items():
            m = marf[ds][fam]
            ks = sorted(m)
            ax.plot(ks, [m[k] for k in ks], lw=2, ms=5, **style)
        # F13 is a transductive measurement; it is not carried over to NT.
        base = marf[ds]["Marformer"]
        fk = sorted(F13_DELTA)
        ax.plot(fk, [base[k] + F13_DELTA[k] for k in fk], color="#B02A37",
                marker="o", mfc="white", ls="--", lw=1.6,
                label="Marformer (T) + F13 epoch budget (est.)")
        for name, style in STAN_STYLE.items():
            m = stan[ds].get(name, {})
            if not m:
                continue
            sk = sorted(m)
            ax.plot(sk, [m[k] for k in sk], lw=1.8, ms=5, label=name, **style)
        # Where the published figure stops measuring Stan and starts drawing it flat.
        ax.axvline(500, color="#787F92", lw=1, ls=":")
        # Not "where Stan stops" -- the fits exist to K=4000 and always did.
        # This is only where the PAPER's curve stops.
        ax.text(560, 0.98, "paper's Stan curve stops here", rotation=90, fontsize=8,
                color="#787F92", va="top", ha="left",
                transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlabel(r"$K_\mathrm{train}$ (items available for training)")
        ax.set_title(f"Data draw {ds}")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("test-corner log-loss (nats)")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=.95)
    fig.suptitle("Item generalization, corner-scored: Marformer against Stan refit on the same draw",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "update_curves.png", dpi=150)
    print("wrote", OUT / "update_curves.png")

    # The gap table: absolute level swings with the draw, the difference does not.
    rows = []
    for k in sorted(set(marf[101]["Marformer"]) & set(marf[105]["Marformer"])):
        b1, b5 = best_stan(stan[101], k), best_stan(stan[105], k)
        rows.append({
            "size": k,
            "marf_101": marf[101]["Marformer"][k], "marf_105": marf[105]["Marformer"][k],
            "marf_spread": abs(marf[101]["Marformer"][k] - marf[105]["Marformer"][k]),
            "stan_101": b1, "stan_105": b5,
            "gap_101": (marf[101]["Marformer"][k] - b1) if b1 else None,
            "gap_105": (marf[105]["Marformer"][k] - b5) if b5 else None,
            "gap_spread": (abs((marf[101]["Marformer"][k] - b1)
                               - (marf[105]["Marformer"][k] - b5))
                           if b1 and b5 else None),
        })
    (OUT / "update_table.json").write_text(json.dumps({"rows": rows,
                                                       "f13_delta": F13_DELTA}, indent=1))
    print(f"{'K':>5} {'|dMarf|':>9} {'|dGap|':>9}")
    for r in rows:
        gs = f"{r['gap_spread']:9.4f}" if r["gap_spread"] is not None else "        -"
        print(f"{r['size']:>5} {r['marf_spread']:9.4f} {gs}")


if __name__ == "__main__":
    main()
