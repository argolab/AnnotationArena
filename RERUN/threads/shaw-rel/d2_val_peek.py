#!/usr/bin/env python
"""Early read of the Shaw arms from the validation curves alone.

No GPU and no test data: just each run's own validation block, smoothed 5 with
F13's 25-epoch floor. This is NOT the result -- the thread reports the test
corner -- but validation is what the protocol selects on, so if an arm were
clearly better or worse it would show here first.
"""
from __future__ import annotations
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from side.s7_curves import val_curve

ROOT = Path(__file__).resolve().parents[2] / "runs/shaw-d2-s101/RESULTS"
ARMS = ["bits", "shaw_k", "shaw_kv", "bits_v"]
REF = "bits"


def main() -> None:
    best: dict = {}
    ep: dict = {}
    for r in sorted(ROOT.rglob("*_MARFORMER")):
        if not (r / "training_history.json").is_file():
            continue
        vc = val_curve(r)
        if len(vc) < 100:
            continue
        sm = [(vc[i][0], st.fmean(x[1] for x in vc[max(0, i - 2):i + 3]))
              for i in range(len(vc))]
        e, v = min(sm[25:], key=lambda t: t[1])
        tag, seed = r.parts[-4], int(r.parts[-2][4:])
        cell = r.name.replace("DOMAIN3-SIGMOID_", "").replace("_MARFORMER", "")
        axis, K = ("annot", cell.split("_")[-1]) if "Annot" in cell else ("item", cell.split("_")[-1])
        best[(tag, axis, int(K), seed)] = v
        ep[(tag, axis, int(K), seed)] = e

    cells = sorted({k[1:] for k in best})
    print(f"cells with a validation curve: {len(cells)}   (annot J=10 carves no block)\n")
    print("paired validation loss vs 'bits'  (negative = BETTER than shipped)")
    hdr = f"{'arm':<9}{'item':>22}{'annot':>22}{'ALL':>22}"
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        line = f"{arm:<9}"
        allds = []
        for grp in ("item", "annot", None):
            ds = [best[(arm,) + c] - best[(REF,) + c] for c in cells
                  if (grp is None or c[0] == grp)
                  and (arm,) + c in best and (REF,) + c in best]
            if grp is None:
                ds = allds
            else:
                allds += ds
            if not ds:
                line += f"{'-':>22}"; continue
            m = st.fmean(ds)
            se = st.stdev(ds) / len(ds) ** 0.5 if len(ds) > 1 else 0.0
            line += f"{m:>+9.4f} +/-{se:>6.4f} n={len(ds):<2}".rjust(22)
        print(line)

    print("\nepochs to the validation optimum (median over cells with all four arms)")
    full = [c for c in cells if all((a,) + c in ep for a in ARMS)]
    for arm in ARMS:
        vals = [ep[(arm,) + c] for c in full]
        print(f"  {arm:<9} median {int(st.median(vals)):>4}   range {min(vals)}-{max(vals)}   n={len(vals)}")
    print(f"\n(cells complete in all four arms: {len(full)} of {len(cells)})")


if __name__ == "__main__":
    main()
