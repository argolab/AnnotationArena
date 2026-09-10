#!/usr/bin/env python
"""
Rough wall-clock comparison: what each method cost to fit, by training size.

TEMPORARY, and the panel says so. These are `elapsed_seconds` /
`fit_elapsed_seconds` scraped from the launcher metadata of runs that were
submitted for other reasons, on a shared cluster, with no repetition and no
control over node contention. They are the right order of magnitude and the
wrong number to quote to three digits. The paper's runtime figure should come
from a controlled pass, not from this.

Two things make the comparison less apples-to-apples than a single axis suggests,
and both are stated on the panel rather than hidden:

  * different hardware. Marformer gets a GPU, Stan does not, because Stan's HMC
    is CPU-only. That IS the comparison being made -- "what does it cost to get
    this curve" -- but it is not a like-for-like per-core measurement.
  * different work. Every Stan fit here is ONE chain. The final run moves to 4
    chains and 2000 sampling iterations (frozen config row 14), which is 4x the
    chains and 4x the post-warmup draws, so these Stan numbers are a LOWER bound
    on what the shipped configuration will cost.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RERUN = Path(__file__).resolve().parents[1]
OUT = RERUN / "dataseed/out"

# Verified from sacct ReqTRES on the jobs that produced these trees.
HW_GPU = "1x H100 + 8 CPU cores"
HW_CPU = "8 CPU cores, no GPU"

SERIES = {
    "Marformer (T)":   dict(color="#B02A37", marker="o", hw=HW_GPU),
    "Marformer NT":    dict(color="#E08379", marker="s", hw=HW_GPU),
    "Stan Oracle":     dict(color="#0d47a1", marker="^", hw=HW_CPU),
    "Stan Proj":       dict(color="#42a5f5", marker="X", hw=HW_CPU),
    "Stan Both":       dict(color="#1565c0", marker="s", hw=HW_CPU),
    "Stan Bin":        dict(color="#90caf9", marker="P", hw=HW_CPU),
}
STAN_DIR_TAG = {"ORACLE": "Stan Oracle", "MISSPEC": None}
TAG_LABEL = {"PROJ": "Stan Proj", "BOTH": "Stan Both", "BIN": "Stan Bin",
             "TENSOR-SIG": "Stan Oracle"}


def collect(tree: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for meta in tree.glob("RESULTS/MARFORMER*/DOMAIN3-SIGMOID/ITEM/**/launcher_metadata.json"):
        d = json.loads(meta.read_text())
        if d.get("launcher_type") != "marformer_train":
            continue
        label = "Marformer NT" if d.get("protocol") == "nontransductive" else "Marformer (T)"
        sec = d.get("elapsed_seconds")
        if sec:
            out.setdefault(label, {})[int(d["size"])] = float(sec)
    for meta in tree.glob("RESULTS/STAN/**/launcher_metadata.json"):
        d = json.loads(meta.read_text())
        if d.get("launcher_type") != "stan_fit":
            continue
        label = TAG_LABEL.get(d.get("model_tag", ""))
        sec = d.get("fit_elapsed_seconds")
        if label and sec:
            out.setdefault(label, {})[int(d["size"])] = float(sec)
    return out


def main() -> None:
    trees = {"101": RERUN / "runs/legacy/dataseed-s101",
             "105": RERUN / "runs/legacy/dataseed-s105"}
    data = {k: collect(v) for k, v in trees.items()}
    pilot = collect(RERUN / "runs/legacy/pilot-10k")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, ds in zip(axes, ("101", "105")):
        for label, style in SERIES.items():
            m = data[ds].get(label, {})
            if not m:
                continue
            ks = sorted(m)
            ax.plot(ks, [m[k] / 60.0 for k in ks], lw=1.9, ms=5,
                    color=style["color"], marker=style["marker"],
                    label=f"{label} — {style['hw']}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$K_\mathrm{train}$ (items available for training)")
        ax.set_title(f"Data draw {ds}")
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel("wall clock to fit (minutes, log scale)")
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=.95)
    fig.suptitle("Rough fit cost by training size — scraped from run logs, NOT a controlled benchmark",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "runtime_curves.png", dpi=150)
    print("wrote", OUT / "runtime_curves.png")

    rows = []
    for ds in ("101", "105"):
        for label in SERIES:
            for k, sec in sorted(data[ds].get(label, {}).items()):
                rows.append({"draw": ds, "method": label, "size": k,
                             "seconds": sec, "minutes": sec / 60.0,
                             "hardware": SERIES[label]["hw"]})
    payload = {"rows": rows,
               "pilot10k": {m: {str(k): v for k, v in sorted(s.items())}
                            for m, s in pilot.items()},
               "hardware": {"marformer": HW_GPU, "stan": HW_CPU},
               "caveat": "scraped from launcher metadata; shared cluster, n=1, no contention control"}
    (OUT / "runtime_table.json").write_text(json.dumps(payload, indent=1))

    print(f"\n{'method':<16}{'K':>7}{'s101 min':>10}{'s105 min':>10}")
    for label in SERIES:
        for k in sorted(set(data['101'].get(label, {})) | set(data['105'].get(label, {}))):
            a = data['101'].get(label, {}).get(k)
            b = data['105'].get(label, {}).get(k)
            f = lambda v: f"{v/60:10.1f}" if v else "         -"
            print(f"{label:<16}{k:>7}{f(a)}{f(b)}")


if __name__ == "__main__":
    main()
