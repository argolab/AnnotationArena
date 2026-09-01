#!/usr/bin/env python
"""
Why is the transductive corner log-loss ~0.09 above the training-partition cells?

Ruled out already: it is not graph presence (--test-missing-in-graph makes test
items structurally identical and moves the gap by 0.001-0.003), it is not
item-specific parameters (item_dropout_rate=1.0 keeps every item deviation at
exactly 0), and it is not label shift (the marginal difference is worth 0.0069
nats).

What is left is context: test items carry 5.45 observed cells each against the
training block's 5.92, and they sit at the end of the item ordering so they land
in the last chunk. This scores every missing cell tagged with its partition, its
item, and how many observed cells that item has, so train and test can be
compared at *matched* context density.

    python gap_anatomy.py --run-dir <run> -o out.json
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _find_checkpoint, _load_checkpoint, _reconstruct

def mean(v): return sum(v)/len(v) if v else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rd = Path(a.run_dir)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    mi = json.loads((rd/"train_config.json").read_text())["training"]["max_item"]
    model, eval_vars, _ = _reconstruct(rd)
    _load_checkpoint(model, _find_checkpoint(rd/"checkpoints", a.checkpoint), dev)
    model.to(dev).eval()
    C = model.types["rating"].num_classes

    # observed cells per item, over the whole evaluation graph
    obs_per_item = defaultdict(int)
    for v in eval_vars:
        if getattr(v, "is_observed", False):
            for iid in v.item_ids:
                obs_per_item[iid] += 1

    cells = []
    with torch.no_grad():
        for ci, chunk in enumerate(chunk_by_item(eval_vars, mi)):
            graph = variable_list_to_entity_graph(chunk, model.types)
            params = model(graph, device=dev)
            for idx, var in enumerate(chunk):
                if not getattr(var, "is_missing", False) or var.rating_value is None:
                    continue
                logp = torch.log_softmax(params[0, idx, 1:1+C], dim=-1)
                iid = var.item_ids[0]
                cells.append({"instance": getattr(var, "instance", "?"),
                              "loss": -float(logp[int(var.rating_value)]),
                              "item": iid, "obs": obs_per_item.get(iid, 0), "chunk": ci})

    tr = [c for c in cells if c["instance"] == "train"]
    te = [c for c in cells if c["instance"] == "test"]
    print(f"train n={len(tr)} loss={mean([c['loss'] for c in tr]):.4f}   "
          f"test n={len(te)} loss={mean([c['loss'] for c in te]):.4f}   "
          f"gap {mean([c['loss'] for c in te])-mean([c['loss'] for c in tr]):+.4f}")

    print(f"\n{'obs/item':>9} {'train n':>8} {'train LL':>9} {'test n':>7} {'test LL':>9} {'gap':>8}")
    buckets = {}
    for lo, hi in ((0,3),(4,5),(6,7),(8,9),(10,99)):
        t = [c["loss"] for c in tr if lo <= c["obs"] <= hi]
        s = [c["loss"] for c in te if lo <= c["obs"] <= hi]
        if not t and not s: continue
        g = (mean(s)-mean(t)) if (t and s) else float("nan")
        buckets[f"{lo}-{hi}"] = {"train_n": len(t), "train_ll": mean(t),
                                 "test_n": len(s), "test_ll": mean(s), "gap": g}
        print(f"{lo}-{hi:>7} {len(t):>8} {mean(t):>9.4f} {len(s):>7} {mean(s):>9.4f} {g:>+8.4f}")

    # chunk placement
    print(f"\n{'chunk':>6} {'train n':>8} {'train LL':>9} {'test n':>7} {'test LL':>9}")
    chunks = {}
    for ci in sorted({c["chunk"] for c in cells}):
        t = [c["loss"] for c in tr if c["chunk"] == ci]
        s = [c["loss"] for c in te if c["chunk"] == ci]
        chunks[ci] = {"train_n": len(t), "train_ll": mean(t), "test_n": len(s), "test_ll": mean(s)}
        print(f"{ci:>6} {len(t):>8} {mean(t):>9.4f} {len(s):>7} "
              f"{(f'{mean(s):.4f}' if s else '-'):>9}")

    Path(a.out).write_text(json.dumps(
        {"overall": {"train_ll": mean([c['loss'] for c in tr]),
                     "test_ll": mean([c['loss'] for c in te])},
         "by_obs_per_item": buckets, "by_chunk": chunks}, indent=2))

if __name__ == "__main__":
    main()
