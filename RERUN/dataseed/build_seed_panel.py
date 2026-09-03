#!/usr/bin/env python
"""
One panel per generative draw: that draw's Marformer curves against the Stan
curves the paper reports.

Stan is REFIT on each draw, so both series come from the same data and are
scored on the same corner cells. That removes the cross-draw uncertainty band
earlier versions of this panel had to carry: the digitized published curves are
seed-42 measurements, and D28 measured up to 0.09 nats of movement between
draws -- the same order as the gap the figure is read as showing.

The digitized published curve is still drawn, faintly, as a reference for how
far the draw moves Stan itself.

    python build_seed_panel.py --seed 101 -o dataseed/out
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_STYLE = {
    "Marformer":   dict(color="#B02A37", marker="o", label="Marformer (T)"),
    "MarformerNT": dict(color="#E08379", marker="s", label="Marformer NT"),
}
STAN_STYLE = {
    "Stan Oracle": dict(color="#0d47a1", marker="^"),
    "Stan Both":   dict(color="#1565c0", marker="s"),
    "Stan Bin":    dict(color="#1976d2", marker="P"),
    "Stan Proj":   dict(color="#42a5f5", marker="X"),
}

def draw_spread(all_rows, kmax):
    """Across-draw sd of corner log-loss, pooled over arms and sizes <= kmax."""
    per = {}
    for r in all_rows:
        per.setdefault((r["family"], r["size"]), {})[r["data_seed"]] = r["corner"]["log_loss"]
    sds = [st.stdev(v.values()) for (fam, K), v in per.items()
           if K <= kmax and len(v) > 2]
    return st.mean(sds) if sds else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--in", dest="inp", default="dataseed/out")
    ap.add_argument("--stan", default="fig6/stan_digitized.json")
    ap.add_argument("--stan-measured", default="dataseed/out/stan_corner.json")
    ap.add_argument("--kmax", type=int, default=1000)
    ap.add_argument("-o", "--outdir", default="dataseed/out")
    a = ap.parse_args()

    rows = []
    for p in sorted(Path(a.inp).glob("eval_s*.json")):
        s = int(p.stem.split("_s")[1])
        for r in json.loads(p.read_text())["rows"]:
            if r["eval_max_item"] == 300 and r["corner_whole"]:
                r["data_seed"] = s; rows.append(r)
    mine = [r for r in rows if r["data_seed"] == a.seed and r["size"] <= a.kmax]
    if not mine:
        raise SystemExit(f"no rows for seed {a.seed}")
    stan = json.loads(Path(a.stan).read_text())["item"]
    band = draw_spread(rows, a.kmax)
    measured = {}
    mp = Path(a.stan_measured)
    if mp.is_file():
        for r in json.loads(mp.read_text())["rows"]:
            if r["data_seed"] == a.seed and r["size"] <= a.kmax:
                measured.setdefault(r["model"], {})[r["size"]] = r["corner"]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    # Stan first, so Marformer sits on top.
    for name, style in STAN_STYLE.items():
        if measured.get(name):
            pts = sorted(measured[name].items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.8, ms=5,
                    alpha=.95, label=f"{name} — refit on seed {a.seed}", zorder=3,
                    **style)
            # The published curve stays visible, faint, so the size of the
            # draw effect on Stan itself is legible rather than asserted.
            pub = sorted((int(k), v) for k, v in stan[name].items() if int(k) <= a.kmax)
            ax.plot([p[0] for p in pub], [p[1] for p in pub], ":", lw=1.0,
                    color=style["color"], alpha=.42, zorder=1,
                    label="published (seed 42)" if name == "Stan Oracle" else None)
        else:
            pts = sorted((int(k), v) for k, v in stan[name].items() if int(k) <= a.kmax)
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, "--", lw=1.5, ms=5, alpha=.9,
                    label=f"{name} (published, seed 42)", **style)
            if name == "Stan Oracle" and band:
                ax.fill_between(xs, [y - band for y in ys], [y + band for y in ys],
                                color=style["color"], alpha=.12, lw=0, zorder=0)
    for fam, style in FAMILY_STYLE.items():
        pts = sorted((r["size"], r["corner"]["log_loss"]) for r in mine if r["family"] == fam)
        if not pts: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=2.6, ms=6.5,
                color=style["color"], marker=style["marker"],
                label=f"{style['label']} — data seed {a.seed}", zorder=4)

    ax.set_xscale("log")
    ticks = sorted({r["size"] for r in mine})
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel(r"$K_\mathrm{train}$ (items available for training)")
    ax.set_ylabel("log-loss on the 414-cell test corner")
    same = "Stan refit on the same draw" if measured else "the published Stan curves"
    ax.set_title(f"Item generalization — generative draw {a.seed}: Marformer vs {same}",
                 fontsize=11)
    ax.grid(alpha=.25, lw=.5)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    if measured:
        ax.text(.012, .03, "solid = refit on this draw   ·   dotted = as published (seed 42)",
                transform=ax.transAxes, fontsize=7.6, color="#0d47a1", alpha=.85)
    elif band:
        ax.text(.012, .03, f"shaded band = ±{band:.3f}, the measured across-draw spread "
                           f"of corner log-loss (D28)",
                transform=ax.transAxes, fontsize=7.6, color="#0d47a1", alpha=.85)
    fig.tight_layout()
    out = Path(a.outdir) / f"seed{a.seed}_vs_stan.png"
    fig.savefig(out, dpi=170)
    print(f"band=±{band:.4f} -> {out}")

    tbl = []
    for fam in FAMILY_STYLE:
        for r in sorted((r for r in mine if r["family"] == fam), key=lambda r: r["size"]):
            tbl.append({"family": fam, "size": r["size"],
                        "corner": r["corner"]["log_loss"], "n": r["corner"]["n"],
                        # train is None where the split leaves the training partition
                        # with no missing cells to score.
                        "train": (r["train"] or {}).get("log_loss"),
                        "stan_oracle": (measured.get("Stan Oracle", {}).get(r["size"])
                                        or stan["Stan Oracle"].get(str(r["size"]))),
                        "stan_measured": bool(measured)})
    Path(a.outdir, f"seed{a.seed}_table.json").write_text(
        json.dumps({"seed": a.seed, "band": band, "stan_measured": bool(measured),
                    "stan": {k: v for k, v in measured.items()}, "rows": tbl}, indent=2))

if __name__ == "__main__":
    main()
