#!/usr/bin/env python
"""
Build a corrected Figure 6(a) and 6(b).

The published panels plot Marformer against Stan on the same axes, but the two
series are scored on different populations:

  Stan       -> missing_ratings_indexes_in_test_instance   (the 414-cell corner)
  Marformer  -> every is_missing token in
                train_obs + test_obs + test_miss + train_miss   (D22/D23)

The paper says the corner is the target -- "All reported log-losses are
evaluated on missing entries within this corner" (main.tex:1269, and the Fig. 6
caption). This script plots the corner metric for both Marformer arms, which is
what the paper describes, and overlays the published pooled metric so the size
of the correction is visible.

Stan is absent: no Domain 3 sigmoid Stan results survived the migration to
skipjack. The series is left out rather than approximated.

    python build_fig6.py --corner fig6/out/corner_seeds-01_last.json \
                         --corner fig6/out/corner_fig6-01_last.json \
                         --outdir fig6/out
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Marformer crimson / NT a muted rose, matching the existing Domain 3 palette.
STYLE = {
    ("Marformer", "corner"):   dict(color="#B02A37", marker="o", ls="-",  label="Marformer (T)"),
    ("MarformerNT", "corner"): dict(color="#E08379", marker="s", ls="-",  label="Marformer NT"),
    ("Marformer", "pooled"):   dict(color="#B02A37", marker="", ls=":",   label="Marformer (T), as published"),
    ("MarformerNT", "pooled"): dict(color="#E08379", marker="", ls=":",   label="Marformer NT, as published"),
}
AXIS_TITLE = {"item": "Item generalization", "annot": "Annotator generalization"}
AXIS_XLABEL = {"item": r"$K_\mathrm{train}$ (items available for training)",
               "annot": r"$J_\mathrm{train}$ (annotators available for training)"}


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def load(paths):
    """cells[(family, axis, size)][metric] = {seed: log_loss}"""
    cells = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for r in json.load(open(p))["rows"]:
            key = (r["family"], r["axis"], r["size"])
            for metric, blob in (("corner", r.get("test")), ("pooled", r.get("pooled"))):
                if blob and blob.get("n"):
                    cells[key][metric][r["seed"]] = blob["log_loss"]
    return cells


def series(cells, family, axis, metric):
    pts = []
    for (fam, ax, size), m in cells.items():
        if fam != family or ax != axis or metric not in m or not m[metric]:
            continue
        mu, sd = mean_sd(list(m[metric].values()))
        pts.append((size, mu, sd, len(m[metric])))
    return sorted(pts)


def panel(ax, cells, axis, *, show_published):
    for family in ("Marformer", "MarformerNT"):
        if show_published:
            pub = series(cells, family, axis, "pooled")
            if pub:
                st = STYLE[(family, "pooled")]
                ax.plot([p[0] for p in pub], [p[1] for p in pub],
                        color=st["color"], ls=st["ls"], lw=1.2, alpha=0.65,
                        label=st["label"], zorder=2)
        cor = series(cells, family, axis, "corner")
        if not cor:
            continue
        st = STYLE[(family, "corner")]
        xs = [p[0] for p in cor]
        ys = [p[1] for p in cor]
        sds = [p[2] for p in cor]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"],
                lw=1.9, ms=4.5, label=st["label"], zorder=4)
        ax.fill_between(xs, [y - s for y, s in zip(ys, sds)],
                        [y + s for y, s in zip(ys, sds)],
                        color=st["color"], alpha=0.15, lw=0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(AXIS_XLABEL[axis])
    ax.set_ylabel("Log-loss on the test corner (nats)")
    ax.set_title(AXIS_TITLE[axis], fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", action="append", required=True,
                    help="corner metric JSON; repeatable")
    ap.add_argument("--outdir", default="fig6/out")
    ap.add_argument("--no-published", action="store_true",
                    help="omit the dotted as-published overlay")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cells = load(args.corner)

    for axis, tag in (("item", "a"), ("annot", "b")):
        fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=200)
        panel(ax, cells, axis, show_published=not args.no_published)
        fig.tight_layout()
        out = outdir / f"fig6{tag}_{axis}_log_loss_corrected.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    # Machine-readable payload + a table, so the numbers are quotable without
    # reading pixels off the figure.
    payload = {}
    lines = []
    for axis in ("item", "annot"):
        for family in ("Marformer", "MarformerNT"):
            cor = series(cells, family, axis, "corner")
            pub = {p[0]: p[1] for p in series(cells, family, axis, "pooled")}
            if not cor:
                continue
            lines.append(f"\n{AXIS_TITLE[axis]} -- {family}")
            lines.append(f"{'size':>6} {'corner':>9} {'sd':>7} {'seeds':>6} "
                         f"{'published':>10} {'delta':>8}")
            for size, mu, sd, n in cor:
                p = pub.get(size)
                payload[f"{axis}|{family}|{size}"] = {
                    "corner_log_loss": mu, "sd": sd, "seeds": n,
                    "published_pooled_log_loss": p,
                    "delta": (mu - p) if p is not None else None}
                lines.append(f"{size:>6} {mu:>9.4f} {sd:>7.4f} {n:>6} "
                             f"{(f'{p:.4f}' if p is not None else '-'):>10} "
                             f"{(f'{mu-p:+.4f}' if p is not None else '-'):>8}")
    (outdir / "fig6_payload.json").write_text(json.dumps(payload, indent=2))
    table = "\n".join(lines)
    (outdir / "fig6_table.txt").write_text(table + "\n")
    print(table)
    print(f"\nwrote {outdir/'fig6_payload.json'} and {outdir/'fig6_table.txt'}")


if __name__ == "__main__":
    main()
