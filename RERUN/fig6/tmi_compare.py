#!/usr/bin/env python
"""
Train-cell vs corner log-loss for explicit run directories.

The question is whether the transductive arm's ~0.09 gap between
training-partition missing cells and the 414-cell corner is genuine
generalization or an artifact of test items lacking the status-0 nodes that
training items carry. So the quantity of interest is the *gap*, not either
number alone -- corner_curve.py reports only the corner.

    python tmi_compare.py --run-dir A --run-dir B -o out.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate, summarise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = {}
    print(f"{'run':<52} {'train':>9} {'corner':>9} {'gap':>9}")
    for rd in a.run_dir:
        rd = Path(rd)
        mi = json.loads((rd / "train_config.json").read_text())["training"]["max_item"]
        rows, _ = evaluate(rd, dev, a.checkpoint, mi)
        tr, te = summarise(rows, "train"), summarise(rows, "test")
        gap = te["log_loss"] - tr["log_loss"] if (tr and te) else None
        # Runs differ by variant directory, not by run name -- keying on
        # rd.name alone silently collapses the off/on pair into one entry.
        label = "/".join(rd.parts[-3:])
        out[label] = {"train": tr, "corner": te, "gap": gap, "max_item": mi}
        print(f"{label:<52} {tr['log_loss']:>9.4f} {te['log_loss']:>9.4f} {gap:>+9.4f}")
    Path(a.out).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
