#!/usr/bin/env python
"""
Score every periodic checkpoint of a run on the 414-cell test corner.

Answers "is this overfitting?" directly: if the corner log-loss bottoms out
mid-training and rises, the last checkpoint is past the optimum. The training
history cannot answer it, because for the non-transductive arm the per-epoch
`combined_missing` curve covers training-partition cells, not the corner.

    python corner_curve.py --run-dir <run> [--run-dir <run> ...] -o out.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate, summarise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = {}
    for rd in a.run_dir:
        rd = Path(rd)
        mi = json.loads((rd / "train_config.json").read_text())["training"]["max_item"]
        ckpts = sorted(rd.glob("checkpoints/periodic-epoch=*.ckpt"),
                       key=lambda p: int(re.search(r"(\d+)\.ckpt$", p.name).group(1)))
        series = []
        for c in ckpts:
            ep = int(re.search(r"(\d+)\.ckpt$", c.name).group(1))
            rows, _ = evaluate(rd, dev, c.name, mi)
            t = summarise(rows, "test")
            series.append({"epoch": ep, "corner_log_loss": t["log_loss"], "n": t["n"]})
            print(f"{rd.name:<44} ep {ep:>4}  corner {t['log_loss']:.4f}", flush=True)
        out[str(rd)] = series
        best = min(series, key=lambda s: s["corner_log_loss"])
        print(f"  -> min {best['corner_log_loss']:.4f} @ epoch {best['epoch']}; "
              f"final {series[-1]['corner_log_loss']:.4f}  "
              f"(inflation {series[-1]['corner_log_loss']-best['corner_log_loss']:+.4f})")
    Path(a.out).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
