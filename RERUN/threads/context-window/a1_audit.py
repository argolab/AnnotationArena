#!/usr/bin/env python
"""A1: why do the two historical training-chunk comparisons disagree at K=4000?

Evidence 2 (hp-01 vs hp-mi300) says training chunk 600 is +0.009 WORSE at K=4000.
Evidence 3 (the 10k pilot) says it is -0.045 BETTER. Before spending a GPU on the
question, establish what each tree actually contains: chunk, epochs, seed, draw,
how many runs completed, and whether several seeds shared one directory (the
collision bug that has cost this project results five times).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

RERUN = Path(__file__).resolve().parents[2]
TREES = ["legacy/hp-01", "legacy/hp-mi300", "legacy/_hp-mi300-collided",
         "legacy/pilot-10k", "pilot-10k"]

def main() -> None:
    report = {}
    for tree in TREES:
        root = RERUN / "runs" / tree
        if not root.is_dir():
            report[tree] = {"exists": False}
            continue
        runs = []
        for cfg in sorted(root.rglob("train_config.json")):
            d = cfg.parent
            c = json.loads(cfg.read_text())
            t, m = c.get("training", {}), c.get("model", {})
            hist = d / "training_history.json"
            done = None
            if hist.is_file():
                try:
                    h = json.loads(hist.read_text())
                    done = len(h.get("epochs", h.get("train_loss", [])))
                except Exception:
                    done = "unreadable"
            runs.append({
                "dir": str(d.relative_to(root)),
                "max_item": t.get("max_item"), "epochs": t.get("epochs"),
                "seed": t.get("seed", c.get("seed")),
                "data_dir": (c.get("data") or {}).get("data_dir"),
                "mtime": cfg.stat().st_mtime,
                "completed_epochs": done,
                "has_last_ckpt": (d / "last.ckpt").is_file(),
            })
        # collision detection: one directory, several distinct config mtimes is
        # normal (overwrite), but distinct seeds landing in the same dir is not.
        by_dir = defaultdict(set)
        for r in runs:
            by_dir[r["dir"]].add(r["seed"])
        report[tree] = {
            "exists": True, "n_runs": len(runs),
            "chunks": sorted({r["max_item"] for r in runs}),
            "seeds": sorted({str(r["seed"]) for r in runs}),
            "seed_in_path": any("seed" in r["dir"] for r in runs),
            "runs": runs,
        }
    out = RERUN / "threads/context-window/out/a1_audit.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"{'tree':<32} {'runs':>5} {'chunks':<12} {'seeds':<14} seed-in-path")
    for tree, r in report.items():
        if not r["exists"]:
            print(f"{tree:<32} ABSENT"); continue
        print(f"{tree:<32} {r['n_runs']:>5} {str(r['chunks']):<12} "
              f"{','.join(r['seeds']):<14} {r['seed_in_path']}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
