#!/usr/bin/env python
"""
Is the corner's log-loss distinguishable from a random 414-cell draw of training
cells, or is the "gap" within sampling noise of a small fixed set?

The corner is a *fixed* 414 cells. Seed-to-seed variation cannot answer this,
because every seed scores the same 414. The right reference is the distribution
of means over random equal-sized subsets of the training cells.

    python corner_bootstrap.py --run-dir <run> --max-item 1200
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-item", type=int, default=1200)
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    rows, _ = evaluate(Path(a.run_dir), dev, a.checkpoint, a.max_item)
    tr = [r[1] for r in rows if r[0] == "train"]
    te = [r[1] for r in rows if r[0] == "test"]
    n = len(te)
    mt, mtr = sum(te)/n, sum(tr)/len(tr)
    rng = random.Random(0)
    means = []
    for _ in range(a.draws):
        s = rng.sample(tr, n)
        means.append(sum(s)/n)
    means.sort()
    lo, hi = means[int(0.025*len(means))], means[int(0.975*len(means))]
    above = sum(1 for m in means if m >= mt) / len(means)
    sd = math.sqrt(sum((m-mtr)**2 for m in means)/(len(means)-1))
    print(f"train cells      n={len(tr):<6} mean {mtr:.4f}")
    print(f"corner cells     n={n:<6} mean {mt:.4f}   gap {mt-mtr:+.4f}")
    print(f"random {n}-cell draws from train: 95% CI [{lo:.4f}, {hi:.4f}]  sd {sd:.4f}")
    print(f"fraction of draws >= corner mean: {above:.4f}   (z = {(mt-mtr)/sd:+.2f})")
    Path(a.out).write_text(json.dumps({"train_mean": mtr, "corner_mean": mt,
        "gap": mt-mtr, "ci95": [lo, hi], "sd": sd, "p_draw_ge_corner": above,
        "z": (mt-mtr)/sd, "n_corner": n, "n_train": len(tr)}, indent=2))

if __name__ == "__main__":
    main()
