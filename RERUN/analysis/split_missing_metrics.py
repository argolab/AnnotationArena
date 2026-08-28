#!/usr/bin/env python
"""
Recompute Domain 3 metrics separately for test-partition and training-partition
missing cells.

`test.py` builds one evaluation graph out of
`train_obs + test_obs + test_miss + train_miss` and then reports metrics over
every status-0 token in it. That pools the held-out test cells with the
training partition's own missing cells, and the pooling is not incidental:

    Item_T_500    414 test  +  1317 train  ->  test is 24% of the number
    Item_T_4000   414 test  + 13836 train  ->  test is  2.9%
    Annot_T_30    414 test  +   701 train  ->  test is 37%

So the published item-generalization curve is, at its right-hand end, almost
entirely in-domain matrix completion on training items rather than
generalization to unseen ones. This script runs the same forward pass and the
same loss, and splits the result by partition, so the two can be looked at
separately.

    python split_missing_metrics.py --run-dir RERUN/runs/seeds-01 -o split.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _find_checkpoint, _load_checkpoint, _reconstruct

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")


def chunk_by_item(variables, max_item):
    if max_item is None:
        return [variables]
    ids = sorted({i for v in variables for i in v.item_ids})
    out = []
    for i in range(0, len(ids), max_item):
        keep = set(ids[i:i + max_item])
        sel = [v for v in variables if all(j in keep for j in v.item_ids)]
        if sel:
            out.append(sel)
    return out or [variables]


@torch.no_grad()
def evaluate(run_dir: Path, device: torch.device, which: str, max_item):
    """Per-cell loss over every missing rating, tagged with its partition."""
    model, eval_vars, cfg = _reconstruct(run_dir)
    _load_checkpoint(model, _find_checkpoint(run_dir / "checkpoints", which), device)
    model.to(device).eval()
    C = model.types["rating"].num_classes
    rows = []
    for chunk in chunk_by_item(eval_vars, max_item):
        graph = variable_list_to_entity_graph(chunk, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk):
            if not getattr(var, "is_missing", False) or var.rating_value is None:
                continue
            logits = params[0, idx, 1:1 + C]
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            y = int(var.rating_value)
            ev = float((p * torch.arange(C, device=p.device, dtype=p.dtype)).sum())
            rows.append((getattr(var, "instance", "?"), -float(logp[y]),
                         int(torch.argmax(logits)) == y, (ev - y) ** 2))
    return rows, cfg


def summarise(rows, want):
    sel = [r for r in rows if r[0] == want]
    if not sel:
        return None
    return {"n": len(sel),
            "log_loss": sum(r[1] for r in sel) / len(sel),
            "accuracy": sum(r[2] for r in sel) / len(sel),
            "rmse": math.sqrt(sum(r[3] for r in sel) / len(sel))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("--exclude", default="-search")
    ap.add_argument("--include", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.run_dir) / "RESULTS"
    out = []
    for fam_dir, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run_dir in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run_dir.name)
            if not m:
                continue
            rel = str(run_dir.relative_to(base))
            if (args.exclude and args.exclude in rel) or (args.include and args.include not in rel):
                continue
            if not (run_dir / "checkpoints" / "last.ckpt").is_file():
                continue
            sm = next((SEED_RE.match(p.name) for p in run_dir.parents if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            with open(run_dir / "train_config.json") as fh:
                trained_max_item = json.load(fh)["training"].get("max_item")
            # test.py hardcodes max_item=500; use the trained value (D20).
            rows, _ = evaluate(run_dir, device, args.checkpoint, trained_max_item)
            entry = {"family": family, "axis": m.group(1).lower(), "size": int(m.group(3)),
                     "seed": seed, "max_item": trained_max_item,
                     "test": summarise(rows, "test"), "train": summarise(rows, "train"),
                     "pooled": {"n": len(rows),
                                "log_loss": sum(r[1] for r in rows) / len(rows),
                                "accuracy": sum(r[2] for r in rows) / len(rows)}}
            out.append(entry)
            t, tr = entry["test"], entry["train"]
            print(f"{family:<12}{entry['axis']:<6}{entry['size']:>5} s{seed}  "
                  f"pooled {entry['pooled']['log_loss']:.4f} (n={entry['pooled']['n']:>5})   "
                  f"TEST {t['log_loss']:.4f} (n={t['n']})   "
                  f"train {tr['log_loss']:.4f} (n={tr['n']})" if t and tr else f"{family} {entry['size']} incomplete")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"checkpoint": args.checkpoint, "rows": out}, fh, indent=1)
    print(f"\n{len(out)} runs -> {args.out}")


if __name__ == "__main__":
    main()
