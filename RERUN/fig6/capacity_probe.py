#!/usr/bin/env python
"""
How large a graph fits, and is chunking actually necessary?

Attention is quadratic in tokens, and tokens ~ items x ratings-per-item (~12
here), so an unchunked K=4000 graph is ~29k tokens. This measures, for
forward-only evaluation, the peak memory and wall time at increasing max_item
until it fails -- which says whether the corner could simply be evaluated whole.

    python capacity_probe.py --run-dir <run> --max-items 300,600,1200,2400,5000
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _find_checkpoint, _load_checkpoint, _reconstruct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-items", default="300,600,1200,2400,5000")
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rd = Path(a.run_dir)
    model, eval_vars, _ = _reconstruct(rd)
    _load_checkpoint(model, _find_checkpoint(rd/"checkpoints", a.checkpoint), dev)
    model.to(dev).eval()
    total_items = len({i for v in eval_vars for i in v.item_ids})
    print(f"graph: {len(eval_vars)} variables over {total_items} items")
    print(f"gpu: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
    print(f"\n{'max_item':>9} {'chunks':>7} {'max tok':>9} {'peak GiB':>9} {'sec':>7}  status")
    out={}
    for mi in [int(x) for x in a.max_items.split(",")]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            chunks = chunk_by_item(eval_vars, mi)
            t0=time.time(); ntok=0
            with torch.no_grad():
                for ch in chunks:
                    g = variable_list_to_entity_graph(ch, model.types)
                    ntok = max(ntok, len(g.tokens))
                    model(g, device=dev)
            dt=time.time()-t0
            pk=torch.cuda.max_memory_allocated()/2**30
            out[mi]={"chunks":len(chunks),"max_tokens":ntok,"peak_gib":pk,"sec":dt,"ok":True}
            print(f"{mi:>9} {len(chunks):>7} {ntok:>9} {pk:>9.2f} {dt:>7.1f}  ok")
        except torch.cuda.OutOfMemoryError:
            out[mi]={"ok":False,"error":"OOM"}
            print(f"{mi:>9} {'-':>7} {'-':>9} {'-':>9} {'-':>7}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            out[mi]={"ok":False,"error":str(e)[:120]}
            print(f"{mi:>9} {'-':>7} {'-':>9} {'-':>9} {'-':>7}  {type(e).__name__}: {str(e)[:60]}")
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
