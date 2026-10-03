#!/usr/bin/env python
"""A1 part 2: re-score the two historical training-chunk arms on a common footing.

The 2026-09-04 reading averaged over all ten architectures of the capacity
screen. Nine of those are architectures we do not ship. This scores the SHIPPED
architecture (`base-search`, 80 wide / 8 deep) only, at several evaluation
chunks, so the training-chunk effect is read on the model the paper describes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

RERUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RERUN))
from analysis.split_missing_metrics import evaluate, summarise

ARMS = {600: "legacy/hp-01", 300: "legacy/_hp-mi300-collided"}
CHUNKS = [300, 500, 600]

def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for chunk, tree in ARMS.items():
        base = RERUN / "runs" / tree / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
        for fam, proto in (("Marformer", "T"), ("MarformerNT", "NT")):
            run = base / f"{fam}_base-search" / f"DOMAIN3-SIGMOID_Item_{proto}_4000_MARFORMER"
            if not (run / "training_history.json").is_file():
                print(f"skip {run}", flush=True); continue
            for c in CHUNKS:
                r, _ = evaluate(run, dev, "last.ckpt", c, corner_whole=True)
                t = summarise(r, "test")
                rows.append({"train_chunk": chunk, "protocol": proto,
                             "eval_chunk": c, "corner": t["log_loss"], "n": t["n"]})
                print(f"train@{chunk} {proto:>2} eval@{c:<4} corner {t['log_loss']:.4f}", flush=True)
    out = RERUN / "threads/context-window/out/a1_rescore.json"
    out.write_text(json.dumps(rows, indent=1))

    print(f"\n{'proto':>6} " + "".join(f"{f'300@{c}':>10}" for c in CHUNKS)
          + "".join(f"{f'600@{c}':>10}" for c in CHUNKS) + f"{'best600-best300':>18}")
    for proto in ("T", "NT"):
        g = {(r["train_chunk"], r["eval_chunk"]): r["corner"] for r in rows if r["protocol"] == proto}
        if not g: continue
        line = f"{proto:>6} "
        for a in (300, 600):
            line += "".join(f"{g.get((a,c), float('nan')):>10.4f}" for c in CHUNKS)
        b3 = min(g[(300, c)] for c in CHUNKS if (300, c) in g)
        b6 = min(g[(600, c)] for c in CHUNKS if (600, c) in g)
        print(line + f"{b6-b3:>+18.4f}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
