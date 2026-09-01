#!/usr/bin/env python
"""
Re-score one trained checkpoint at several evaluation chunk sizes.

The test corner is always the highest item ids, so contiguous-id chunking always
puts it in the final chunk. Raising max_item until the whole graph is one chunk
removes that confound entirely: if the train-vs-corner gap survives a single
chunk, position and chunking are not the cause.

    python chunk_sweep_eval.py --run-dir <run> --max-items 300,600,1200,2400
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate, summarise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-items", default="300,600,1200,2400")
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rd = Path(a.run_dir)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = {}
    print(f"{'max_item':>9} {'chunks':>7} {'train':>9} {'corner':>9} {'gap':>9}")
    for mi in [int(x) for x in a.max_items.split(",")]:
        try:
            rows, _ = evaluate(rd, dev, a.checkpoint, mi)
        except torch.cuda.OutOfMemoryError:
            print(f"{mi:>9} {'OOM':>7}")
            torch.cuda.empty_cache(); continue
        tr, te = summarise(rows, "train"), summarise(rows, "test")
        gap = te["log_loss"] - tr["log_loss"]
        out[mi] = {"train": tr, "corner": te, "gap": gap}
        print(f"{mi:>9} {'-':>7} {tr['log_loss']:>9.4f} {te['log_loss']:>9.4f} {gap:>+9.4f}")
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
