#!/usr/bin/env python
"""The two appendix figures, redrawn from logit_lens.py's JSON.

Panel layout follows the shipped figures exactly so the appendix text and
captions still describe what is on the page:

  layerwise_metrics.png     log-loss and MBR-L2 risk vs depth, three lenses
                            plus the final model as a dashed floor
  selected_calibration.png  3x3 reliability grid, one row per lens, at layers
                            0 / mid / L

Reliability panels are drawn by rp.plot_rel_diagram on the stored probabilities,
which is the same call plot_domain3_generalization._plot_ece makes, so these are
the paper's own diagrams and not a lookalike.

Type is scaled through utils.font_scale like every other Domain 3 figure -- the
appendix prints these at 0.85\\linewidth rather than 0.48, so SCALE is divided
back down to match the width they are actually reproduced at.

    python plot_logit_lens.py --json LENS.json --outdir DIR
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RANK / "scripts"))
sys.path.insert(0, str(RANK))

from utils.font_scale import scale_rcparams, fs, ts, pfs   # noqa: E402
import relplot as rp                                        # noqa: E402

LENSES = [("param_lens", "Param Lens", "#7a1f3d", "o"),
          ("tuned_param_lens", "Tuned Param Lens", "#b8566e", "s"),
          ("tuned_80d_lens", "Tuned 80d-Lens", "#dd93a4", "^")]
FINAL_C = "#2f3a4a"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 15,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.framealpha": 0.92, "legend.edgecolor": "0.75",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "lines.linewidth": 2.4, "lines.markersize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
scale_rcparams(mpl.rcParams)
rp.config.use_tex_fonts = False


def metrics_figure(d: dict, out: Path):
    layers = [r["layer"] for r in d["layers"]]
    # Title carries the full name; the y label is kept SHORT on purpose. At
    # DOMAIN3_FONT_SCALE=3 a 21-character rotated label is ~420pt of type against
    # a ~360pt axes, so it runs off the canvas and the saved bbox clips it -- the
    # same way a long x label overruns its panel. Short label, full title.
    panels = [("log_loss", "Test Missing Log Loss", "Log loss"),
              ("mbr_l2", "Test Missing MBR L2", "MBR L2 risk")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), layout="constrained")
    for ax, (key, title, ylab) in zip(axes, panels):
        for lk, label, color, marker in LENSES:
            ax.plot(layers, [r[lk][key] for r in d["layers"]],
                    color=color, marker=marker, label=label)
        ax.axhline(d["final_model"][key], color=FINAL_C, ls="--", lw=2.2,
                   label="Final Model")
        ax.set_title(title, fontsize=ts(15))
        ax.set_xlabel("Layer", fontsize=fs(13))
        ax.set_ylabel(ylab, fontsize=fs(13))
        ax.set_xticks(layers)
    # One legend for both panels. constrained_layout reserves room for a figure
    # legend it is told about, so the legend is laid out rather than overlaid and
    # savefig has nothing left to crop.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=fs(11))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def calibration_figure(d: dict, out: Path, cols=None):
    L = d["num_layers"]
    cols = cols or [0, L // 2, L]
    labels = np.asarray(d["labels"])
    fig, axes = plt.subplots(len(LENSES), len(cols),
                             figsize=(6.8 * len(cols), 5.8 * len(LENSES)),
                             layout="constrained")
    for r, (lk, lname, color, _) in enumerate(LENSES):
        for c, li in enumerate(cols):
            ax = axes[r][c]
            rec = d["layers"][li][lk]
            probs = np.asarray(rec["probs"], dtype=float)
            conf, acc = np.asarray(rec["_conf"]), np.asarray(rec["_acc"])
            diag = rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                          report_CE=True, report_CE_std=True)
            rp.plot_rel_diagram(diag, fig=fig, ax=ax, color=color,
                                use_default_style=True, plot_density_ticks=True,
                                plot_labels=True, legend=False)
            for t in ax.texts:
                t.remove()
            # ts(11), not ts(14): the title is two ~21-character lines centred on
            # the AXES, so at ts(14) the right-hand column's smECE interval runs
            # past the figure edge and is clipped. Measured on this grid.
            ax.set_title(f"{lname} L{li}\nsmECE = {diag['ce']:.3f} "
                         f"± {diag.get('ce_ci_width', 0.0):.3f}",
                         fontsize=ts(11), fontweight="bold", pad=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.tick_params(labelsize=pfs(11))
            ax.xaxis.label.set_size(pfs(12)); ax.yaxis.label.set_size(pfs(12))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cols", default=None, help="comma-separated layers for the grid")
    a = ap.parse_args()

    d = json.loads(Path(a.json).read_text())
    if "labels" not in d:
        raise SystemExit("this JSON predates --dump-probs; rerun logit_lens.py")

    # the calibration panels need the all-class flattening the paper uses
    sys.path.insert(0, str(RANK / "scripts/utils"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec); sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    labels = np.asarray(d["labels"])
    for row in d["layers"]:
        for lk, *_ in LENSES:
            probs = np.asarray(row[lk]["probs"], dtype=float)
            conf, acc = m.shared._all_class_calibration(probs, labels)
            row[lk]["_conf"], row[lk]["_acc"] = conf, acc

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    metrics_figure(d, outdir / "layerwise_metrics.png")
    cols = [int(x) for x in a.cols.split(",")] if a.cols else None
    calibration_figure(d, outdir / "selected_calibration.png", cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
