#!/usr/bin/env python
"""
Is the mi600 advantage more training context, or just a better-matched eval?

The 10k pilot ran two arms that differ ONLY in training chunk (max_item 600 vs
300), same seed, same data. Corner-scored at eval chunk 500, the 600 arm is
~0.027 better on the transductive curve at every size. Two readings:

  A. more context at training time genuinely helps, and the 300 ceiling costs us;
  B. it is an evaluation artifact -- mi600 is scored at 500, which is closer to
     its own training chunk than 500 is to mi300's, and S8 established that a
     model does better when the evaluation chunk matches training.

They are separated by scoring each arm at SEVERAL evaluation chunks and comparing
each arm at its own best. If A, mi600 wins at every eval chunk. If B, each arm
peaks near its own training chunk and the advantage collapses.

Last checkpoint only, 4 sizes, 3 eval chunks, 2 arms = 24 evaluations.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

RERUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RERUN))
from analysis.split_missing_metrics import evaluate, summarise

BASE = RERUN / "runs/legacy/pilot-10k/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
SIZES = [1000, 2000, 3000, 4000]
CHUNKS = [300, 500, 600]


def main() -> None:
    dev = torch.device("cpu")
    rows = []
    for arm, sub in (("mi600", ""), ("mi300", "mi300")):
        for k in SIZES:
            run = (BASE / sub / f"DOMAIN3-SIGMOID_Item_T_{k}_MARFORMER") if sub \
                else (BASE / f"DOMAIN3-SIGMOID_Item_T_{k}_MARFORMER")
            if not (run / "training_history.json").is_file():
                continue
            for c in CHUNKS:
                r, _ = evaluate(run, dev, "last.ckpt", c, corner_whole=True)
                t = summarise(r, "test")
                rows.append({"arm": arm, "size": k, "eval_chunk": c,
                             "corner": t["log_loss"], "n": t["n"]})
                print(f"{arm:>6} K={k:<6} eval@{c:<4} corner {t['log_loss']:.4f}", flush=True)
    (RERUN / "side/out/chunk_disentangle.json").write_text(json.dumps(rows, indent=1))

    print(f"\n{'K':>6} " + "".join(f"{f'300@{c}':>11}" for c in CHUNKS)
          + "".join(f"{f'600@{c}':>11}" for c in CHUNKS) + f"{'best600-best300':>17}")
    for k in SIZES:
        g = {(r["arm"], r["eval_chunk"]): r["corner"] for r in rows if r["size"] == k}
        if not g:
            continue
        b3 = min(g[("mi300", c)] for c in CHUNKS if ("mi300", c) in g)
        b6 = min(g[("mi600", c)] for c in CHUNKS if ("mi600", c) in g)
        line = f"{k:>6} "
        for a in ("mi300", "mi600"):
            for c in CHUNKS:
                v = g.get((a, c))
                line += f"{v:11.4f}" if v else f"{'-':>11}"
        print(line + f"{b6 - b3:+17.4f}")


if __name__ == "__main__":
    main()
