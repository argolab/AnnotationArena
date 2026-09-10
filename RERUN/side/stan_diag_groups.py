#!/usr/bin/env python
"""
Split the Stan diagnostics by parameter GROUP.

R-hat over every sampled parameter is misleading for a factor model: the
per-item / per-annotator latents carry sign and rotation symmetries, so they can
show enormous R-hat while predictions -- which is all we score Stan on -- are
completely unaffected. What matters is whether the *invariant* quantities mixed:
lp__ and the variance hyperparameters. This separates them.

Job-only; parses large CmdStan CSVs. See stan_diag.sbatch.
"""
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_diagnostics import read_cmdstan_csv, split_rhat, ess

def group_of(name: str) -> str:
    if name == "lp__": return "lp__"
    if name.startswith("sigma"): return "hyperparameter"
    if re.match(r"^(u|v|uit|z|theta|beta|alpha)\b|\[", name): return "latent"
    return "other"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    ap.add_argument("--pattern", default="*_4000_*,*_1000_*")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    results = []
    for root in a.roots.split(","):
        for pat in a.pattern.split(","):
            for run_dir in sorted(glob.glob(os.path.join(root, "**", pat), recursive=True)):
                if not os.path.isdir(run_dir) or run_dir.endswith("_eval"):
                    continue          # _eval dirs hold predictions, not draws
                for csv in sorted(Path(run_dir).glob("*.csv")):
                    if csv.name.startswith("posterior_hyperparameters"):
                        continue
                    names, draws, _ = read_cmdstan_csv(csv)
                    if names is None or draws.shape[0] > 5000:
                        continue      # guard: sampler CSVs are 500 draws
                    print(f"{os.path.basename(run_dir)}  ({draws.shape[0]} draws)", flush=True)
                    rec = {"run": os.path.basename(run_dir), "groups": {}}
                    buckets: dict[str, list] = {}
                    for i, nm in enumerate(names):
                        if nm.endswith("__") and nm != "lp__":
                            continue
                        x = draws[:, i]
                        if np.allclose(x, x[0]):
                            continue
                        buckets.setdefault(group_of(nm), []).append((nm, split_rhat(x), ess(x)))
                    for g, items in buckets.items():
                        rh = np.array([r for _, r, _ in items if np.isfinite(r)])
                        es = np.array([e for _, _, e in items if np.isfinite(e)])
                        worst = max(items, key=lambda t: (t[1] if np.isfinite(t[1]) else 0))
                        rec["groups"][g] = {
                            "n": len(items),
                            "rhat_max": float(rh.max()) if rh.size else None,
                            "rhat_median": float(np.median(rh)) if rh.size else None,
                            "rhat_gt_1.01_pct": 100.0*float((rh > 1.01).mean()) if rh.size else None,
                            "ess_min": float(es.min()) if es.size else None,
                            "ess_median": float(np.median(es)) if es.size else None,
                            "worst_param": worst[0],
                        }
                        print(f"    {g:<15} n={len(items):<6} rhat_max={rec['groups'][g]['rhat_max']:.3f}"
                              f"  ess_min={rec['groups'][g]['ess_min']:.0f}  worst={worst[0]}", flush=True)
                    results.append(rec)
    Path(a.out).write_text(json.dumps({"runs": results}, indent=1))
    print(f"\nwrote {len(results)} runs -> {a.out}")

if __name__ == "__main__":
    main()
