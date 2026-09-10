#!/usr/bin/env python
"""
What happens past K=4000: the 10k pilot, and what it says about extrapolating.

The pilot is a SEPARATE generative draw with its own corner, so its absolute
level cannot be read against seed 42's or against s101/s105 -- D28 measured 0.085
nats of draw-to-draw movement, which is most of the effect anyone would be
looking for. Two panels, therefore:

  (a) the pilot on its own axis, K=1000..10000, absolute. Honest about what it is.
  (b) every draw shifted to its own K=1000 value, so only the SHAPE is compared.
      A draw offset is a constant and drops out of a difference from that draw's
      own anchor, which makes "does the curve flatten" answerable across draws
      even though "which curve is lower" is not.

The reason (b) is the panel that matters: the question is whether returns die
after 4000, and that is a statement about slope, not level.
"""
from __future__ import annotations
import json, re, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RERUN = Path(__file__).resolve().parents[1]
OUT = RERUN / "dataseed/out"
TAIL = 5


def pilot_curves():
    """corner log-loss per size, from the mi300 arm (the only one that reaches 10k)."""
    src = RERUN / "side/out/corner_pilot10k.json"
    rows = json.loads(src.read_text())
    out = {}
    for r in rows:
        if "/mi300/" not in r["run"]:
            continue
        fam = "Marformer NT" if "MARFORMER-NT" in r["run"] else "Marformer (T)"
        pts = [p["corner"] for p in r["series"] if p["epoch"] >= 0]
        if pts:
            out.setdefault(fam, {})[r["size"]] = statistics.fmean(pts[-TAIL:])
    return out


def dataseed_curves():
    out = {}
    for ds in (101, 105):
        t = json.loads((OUT / f"seed{ds}_table.json").read_text())
        out[f"draw {ds}"] = {r["size"]: r["corner"] for r in t["rows"]
                             if r["family"] == "Marformer"}
    return out


def main() -> None:
    pilot = pilot_curves()
    seeds = dataseed_curves()
    if not pilot:
        raise SystemExit("no pilot rows scored yet")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    for fam, style in (("Marformer (T)", dict(color="#B02A37", marker="o")),
                       ("Marformer NT", dict(color="#E08379", marker="s"))):
        m = pilot.get(fam, {})
        if not m:
            continue
        ks = sorted(m)
        axA.plot(ks, [m[k] for k in ks], lw=2.2, ms=6, label=fam, **style)
    axA.axvspan(4000, 10000, color="#787F92", alpha=.09, lw=0)
    axA.text(4300, axA.get_ylim()[1], " new ground", fontsize=8, color="#787F92",
             va="top")
    axA.set_xscale("log")
    axA.set_xticks([1000, 2000, 4000, 6000, 8000, 10000])
    axA.set_xticklabels(["1000", "2000", "4000", "6000", "8000", "10000"], fontsize=8.5)
    axA.minorticks_off()
    axA.set_xlabel(r"$K_\mathrm{train}$")
    axA.set_ylabel("corner log-loss (nats)")
    axA.set_title("(a) The 10k pilot, on its own draw", fontsize=11)
    axA.grid(alpha=.25); axA.legend(fontsize=8.5, frameon=False)

    # (b) shape only: each curve minus its own K=1000 value.
    anchor = 1000
    for name, m in list(seeds.items()) + [("10k pilot", pilot.get("Marformer (T)", {}))]:
        m = {k: v for k, v in m.items() if k >= anchor}
        if anchor not in m:
            continue
        ks = sorted(m)
        style = dict(color="#B02A37", marker="o", lw=2.4, ms=6) if "pilot" in name \
            else dict(color="#787F92", marker="o", lw=1.5, ms=4, alpha=.75)
        axB.plot(ks, [m[k] - m[anchor] for k in ks], label=name, **style)
    axB.axhline(0, color="#B6BBCA", lw=1)
    axB.set_xscale("log")
    axB.set_xticks([1000, 2000, 4000, 6000, 8000, 10000])
    axB.set_xticklabels(["1000", "2000", "4000", "6000", "8000", "10000"], fontsize=8.5)
    axB.minorticks_off()
    axB.set_xlabel(r"$K_\mathrm{train}$")
    axB.set_ylabel("corner log-loss relative to that curve's own K=1000")
    axB.set_title("(b) Shape only — each draw shifted to its own K=1000", fontsize=11)
    axB.grid(alpha=.25); axB.legend(fontsize=8.5, frameon=False)

    fig.suptitle("Does the item-generalization curve flatten after K=4000?", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "extrapolation.png", dpi=150)
    print("wrote", OUT / "extrapolation.png")

    t = pilot.get("Marformer (T)", {})
    rows = []
    ks = sorted(t)
    for i, k in enumerate(ks):
        prev = ks[i - 1] if i else None
        rows.append({"size": k, "corner": t[k],
                     "delta_prev": (t[k] - t[prev]) if prev else None,
                     "from_1000": (t[k] - t[1000]) if 1000 in t else None})
    (OUT / "extrapolation_table.json").write_text(json.dumps(
        {"pilot": {f: {str(k): v for k, v in sorted(m.items())} for f, m in pilot.items()},
         "rows": rows}, indent=1))
    print(f"{'K':>7} {'corner':>9} {'vs prev':>9} {'vs 1000':>9}")
    for r in rows:
        dp = f"{r['delta_prev']:+9.4f}" if r["delta_prev"] is not None else "        -"
        f1 = f"{r['from_1000']:+9.4f}" if r["from_1000"] is not None else "        -"
        print(f"{r['size']:>7} {r['corner']:9.4f} {dp} {f1}")


if __name__ == "__main__":
    main()
