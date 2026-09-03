#!/usr/bin/env python
"""
Corner log-loss for every Stan fit on the regenerated data.

Uses the same rule the paper's plot script uses -- restrict to
`missing_ratings_indexes_in_test_instance`, average the posterior draws'
category probabilities per cell, then cross-entropy against the realised label
-- so these numbers are produced the way the published Stan points were, and
land on exactly the cells Marformer is scored on.

    python score_stan.py --seeds 101,105 -o dataseed/out/stan_corner.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

PROB_PREFIX = "prob_cat_"
TAG_LABEL = {"TENSOR-SIG": "Stan Oracle", "BIN": "Stan Bin",
             "PROJ": "Stan Proj", "BOTH": "Stan Both"}
EVAL_RE = re.compile(r"DOMAIN3-SIGMOID_Item_T_(\d+)_([A-Z-]+)_eval$")


def score(bundle_path: Path, eval_dir: Path):
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.is_file():
        return None
    bundle = json.loads(bundle_path.read_text())
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    if not idxs:
        return None
    labels = np.asarray([bundle["missing_ratings"][i]["value"] - 1 for i in idxs],
                        dtype=np.int64)
    df = pd.read_csv(probs_path)
    cols = sorted(c for c in df.columns if c.startswith(PROB_PREFIX))
    g = (df[df["missing_rating_idx"].isin(idxs)]
         .groupby("missing_rating_idx")[cols].mean().reindex(idxs))
    if g.isnull().any().any():
        return None
    p = g.to_numpy(dtype=np.float64)
    nll = -np.log(np.clip(p[np.arange(labels.shape[0]), labels], 1e-12, 1.0))
    return float(nll.mean()), int(labels.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,105")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("-o", "--out", default="dataseed/out/stan_corner.json")
    a = ap.parse_args()
    rows = []
    for s in [x for x in a.seeds.split(",") if x]:
        root = Path(a.runs_root) / f"dataseed-s{s}"
        for eval_dir in sorted(root.glob("RESULTS/STAN/**/ITEM/*_eval")):
            m = EVAL_RE.search(eval_dir.name)
            if not m:
                continue
            size, tag = int(m.group(1)), m.group(2)
            bundle = (root / "DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive"
                      / f"DOMAIN3-SIGMOID_Item_T_{size}" / "data_bundle.json")
            r = score(bundle, eval_dir)
            if r is None:
                print(f"  s{s} {tag:<11} K={size:<5} incomplete"); continue
            ll, n = r
            rows.append({"data_seed": int(s), "model": TAG_LABEL.get(tag, tag),
                         "tag": tag, "size": size, "corner": ll, "n": n})
            print(f"  s{s} {TAG_LABEL.get(tag, tag):<12} K={size:<5} corner {ll:.4f}  n={n}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n{len(rows)} fits -> {a.out}")

if __name__ == "__main__":
    main()
