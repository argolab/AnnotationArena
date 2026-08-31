#!/usr/bin/env python
"""
Build a corrected Figure 6(a) and 6(b).

Two things are corrected relative to the published panels.

1. **The evaluation set.** The published Marformer points are scored on
   `train_obs + test_obs + test_miss + train_miss` -- every missing token in the
   graph -- while Stan is scored on the 414-cell test corner alone (D22/D23).
   The paper says the corner is the target: "All reported log-losses are
   evaluated on missing entries within this corner" (main.tex:1269, and the
   Fig. 6 caption). Every series here is the corner metric.

2. **Checkpoint selection.** The published points are an argmin over ~18
   checkpoints scored on test (D14). The primary series here is instead the
   block-validation protocol: carve part of the training partition off as
   validation, select the stopping epoch on it, then refit on the whole
   training partition for that many epochs and report the last checkpoint. Test
   is never consulted, and the refit's data budget equals Stan's.

Stan is absent: no Domain 3 sigmoid Stan results survived the migration to
skipjack, and they are not being regenerated. The series is omitted rather
than approximated.

    python build_fig6.py \
        --corner valblock=fig6/out/corner_val-block-01_last.json \
        --corner last=fig6/out/corner_seeds-01_last.json \
        --corner last=fig6/out/corner_fig6-01_last.json \
        --primary valblock --reference last --outdir fig6/out
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

FAMILY_STYLE = {
    "Marformer":   dict(color="#B02A37", marker="o", label="Marformer (T)"),
    "MarformerNT": dict(color="#E08379", marker="s", label="Marformer NT"),
}
SEL_LABEL = {"valblock": "block-val + refit", "last": "last checkpoint",
             "published": "as published (pooled, test-selected)"}
AXIS_TITLE = {"item": "Item generalization", "annot": "Annotator generalization"}
AXIS_XLABEL = {"item": r"$K_\mathrm{train}$ (items available for training)",
               "annot": r"$J_\mathrm{train}$ (annotators available for training)"}


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    return (m, 0.0) if n == 1 else (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)))


def load(specs):
    """cells[(family, axis, size)][selection] = {seed: corner_log_loss}

    `published` is carried alongside: it is the pooled metric from the same
    forward pass, i.e. the number the current figure plots.
    """
    cells = defaultdict(lambda: defaultdict(dict))
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--corner expects label=path, got {spec!r}")
        for r in json.load(open(path))["rows"]:
            key = (r["family"], r["axis"], r["size"])
            if r.get("test") and r["test"].get("n"):
                cells[key][label][r["seed"]] = r["test"]["log_loss"]
    return cells


def series(cells, family, axis, selection):
    pts = []
    for (fam, ax, size), sels in cells.items():
        if fam != family or ax != axis:
            continue
        vals = sels.get(selection)
        if not vals:
            continue
        mu, sd = mean_sd(list(vals.values()))
        pts.append((size, mu, sd, len(vals)))
    return sorted(pts)


def panel(ax, cells, axis, primary, reference):
    for family, st in FAMILY_STYLE.items():
        if reference:
            ref = series(cells, family, axis, reference)
            if ref:
                ax.plot([p[0] for p in ref], [p[1] for p in ref], color=st["color"],
                        ls=":", lw=1.2, alpha=0.7, zorder=2,
                        label=f"{st['label']} — {SEL_LABEL.get(reference, reference)}")
        pri = series(cells, family, axis, primary)
        if not pri:
            continue
        xs, ys, sds = [p[0] for p in pri], [p[1] for p in pri], [p[2] for p in pri]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls="-", lw=1.9, ms=4.5,
                zorder=4, label=f"{st['label']} — {SEL_LABEL.get(primary, primary)}")
        ax.fill_between(xs, [y - s for y, s in zip(ys, sds)],
                        [y + s for y, s in zip(ys, sds)],
                        color=st["color"], alpha=0.15, lw=0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(AXIS_XLABEL[axis])
    ax.set_ylabel("Log-loss on the test corner (nats)")
    ax.set_title(AXIS_TITLE[axis], fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable; several files may share a label")
    ap.add_argument("--primary", default="valblock")
    ap.add_argument("--reference", default="last",
                    help="second series drawn dotted; empty string to omit")
    ap.add_argument("--published", default="",
                    help="published.json from extract_published.py; adds an "
                         "'as published' column (seed 42 only) to the table")
    ap.add_argument("--outdir", default="fig6/out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cells = load(args.corner)

    for axis, tag in (("item", "a"), ("annot", "b")):
        fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=200)
        panel(ax, cells, axis, args.primary, args.reference or None)
        fig.tight_layout()
        out = outdir / f"fig6{tag}_{axis}_log_loss_corrected.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    published = json.loads(Path(args.published).read_text()) if args.published else {}
    payload, lines = {}, []
    sels = [s for s in (args.primary, args.reference) if s]
    for axis in ("item", "annot"):
        for family in FAMILY_STYLE:
            base = {s: {p[0]: (p[1], p[2], p[3]) for p in series(cells, family, axis, s)}
                    for s in sels}
            sizes = sorted(set().union(*(d.keys() for d in base.values())) if base else [])
            if not sizes:
                continue
            lines.append(f"\n{AXIS_TITLE[axis]} -- {family}")
            hdr = f"{'size':>6}" + "".join(f"{SEL_LABEL.get(s,s)[:16]:>18}" for s in sels)
            if published:
                hdr += f"{'published(s42)':>16}{'delta':>9}"
            lines.append(hdr)
            for size in sizes:
                row = f"{size:>6}"
                rec = {}
                for s in sels:
                    v = base[s].get(size)
                    row += f"{(f'{v[0]:.4f} ({v[2]})' if v else '-'):>18}"
                    if v:
                        rec[s] = {"log_loss": v[0], "sd": v[1], "seeds": v[2]}
                pub = published.get(f"{axis}|{family}|{size}")
                if published:
                    pv = pub["log_loss"] if pub else None
                    pri = base.get(args.primary, {}).get(size)
                    row += f"{(f'{pv:.4f}' if pv is not None else '-'):>16}"
                    row += f"{(f'{pri[0]-pv:+.4f}' if (pv is not None and pri) else '-'):>9}"
                    if pub:
                        rec["published_seed42"] = pub
                payload[f"{axis}|{family}|{size}"] = rec
                lines.append(row)
    (outdir / "fig6_payload.json").write_text(json.dumps(payload, indent=2))
    table = "\n".join(lines)
    (outdir / "fig6_table.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
