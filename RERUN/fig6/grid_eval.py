#!/usr/bin/env python
"""
Score every grid checkpoint across the evaluation axes.

Training varies shuffle-before-chunk and the training chunk size. Evaluation
adds two more axes on top of each checkpoint, cheaply (~1.4s per config):
  eval chunk size      300 / 600 / 1200
  corner handling      contiguous (ships today) / whole (corner never split)

Emits one row per (run, eval_max_item, corner_whole) with corner, train and
pooled log-loss, so the grid can be pivoted any way afterwards.

    python grid_eval.py --run-dir RERUN/runs/grid-01 -o out.json
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.split_missing_metrics import evaluate, summarise

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-max-items", default="300,600,1200")
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    root = Path(a.run_dir) / "RESULTS"
    evals = [int(x) for x in a.eval_max_items.split(",")]
    rows = []
    for fam_dir, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run.name)
            if not m or not (run / "checkpoints" / "last.ckpt").is_file():
                continue
            rel = run.relative_to(base)
            variant = str(rel.parent.parent) if rel.parent.parent != Path(".") else ""
            sm = next((SEED_RE.match(p.name) for p in run.parents if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            cfg = json.loads((run / "train_config.json").read_text())["training"]
            train_mi = cfg.get("max_item")
            # variant looks like "<Family>_sh<bool>_mi<int>"
            vm = re.search(r"_sh(true|false)_mi(\d+)", variant)
            shuffle = (vm.group(1) == "true") if vm else None
            for emi in evals:
                for whole in (False, True):
                    t0 = time.time()
                    try:
                        r, _ = evaluate(run, dev, a.checkpoint, emi, corner_whole=whole)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); continue
                    te, tr = summarise(r, "test"), summarise(r, "train")
                    rows.append({"family": family, "axis": m.group(1).lower(),
                                 "size": int(m.group(3)), "seed": seed,
                                 "variant": variant, "shuffle": shuffle,
                                 "train_max_item": train_mi, "eval_max_item": emi,
                                 "corner_whole": whole,
                                 "corner": te, "train": tr,
                                 "pooled_n": len(r), "sec": time.time() - t0})
                    print(f"{variant:<34} K={m.group(3):<5} eval@{emi:<5} whole={int(whole)} "
                          f"corner {te['log_loss']:.4f}", flush=True)
                    torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n{len(rows)} rows -> {a.out}")

if __name__ == "__main__":
    main()
