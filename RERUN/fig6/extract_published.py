#!/usr/bin/env python
"""
Recover the numbers the current figure actually plots.

A published Marformer point is TEST_RESULTS/best.json: the argmin over ~18
checkpoints of the POOLED missing log-loss (D14 for the selection, D22 for the
pooling). Both defects are in that one number, so it is the right baseline to
correct against.

Only replicate-01 carries best.json, and only for seed 42 -- the plot script
writes it, and the multi-seed sweep never ran the plot script. So this is a
single-seed reference and is labelled as such.

    python extract_published.py --run-dir runs/replicate-01 -o fig6/out/published.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = {}
    root = Path(a.run_dir) / "RESULTS"
    for fam_dir, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        for bj in base.rglob("TEST_RESULTS/best.json"):
            m = RUN_RE.search(bj.parent.parent.name)
            if not m:
                continue
            d = json.loads(bj.read_text())
            out[f"{m.group(1).lower()}|{family}|{int(m.group(3))}"] = {
                "log_loss": d["missing"]["log_loss"],
                "n": d["missing"]["n"],
                "selected_from": d.get("selected_from"),
                "selection_metric": d.get("selection_metric"),
            }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"{len(out)} published points -> {a.out}")

if __name__ == "__main__":
    main()
