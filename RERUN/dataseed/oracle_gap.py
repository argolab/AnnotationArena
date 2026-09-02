#!/usr/bin/env python
"""
Is a corner "hard" because the model generalises badly, or because the cells
themselves are intrinsically harder to predict?

The generator writes rating_cumprobs -- the TRUE categorical distribution it
sampled each rating from. Its cross-entropy against the realised label is the
Bayes-optimal log-loss: no estimator, Marformer or Stan, can beat it. Computing
it separately on the corner and on the training cells gives the floor of the
train-vs-corner gap that is pure draw, before any model is involved.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import defaultdict

def oracle(bundle_path: Path):
    b = json.loads(bundle_path.read_text())
    md = b["domain3_metadata"]
    ti = set(md["test_block_item_orig_ids"]); ta = set(md["test_block_annotator_orig_ids"])
    i2o = {k + 1: v for k, v in enumerate(md["selected_item_orig_ids"])}
    a2o = {k + 1: v for k, v in enumerate(md["selected_annotator_orig_ids"])}
    out = defaultdict(lambda: [0.0, 0])
    counts = defaultdict(int)
    for part, key in (("missing", "missing_ratings"), ("observed", "observed_ratings")):
        for r in b.get(key, []):
            p = r.get("rating_dist")
            in_corner = i2o[r["item"]] in ti and a2o[r["annotator"]] in ta
            zone = "corner" if in_corner else "outside"
            counts[(zone, part)] += 1
            if part != "missing" or p is None:
                continue
            q = p[r["value"] - 1]
            out[zone][0] += -math.log(max(q, 1e-12)); out[zone][1] += 1
    return {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in out.items()}, dict(counts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,102,103,104,105")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--split", default="ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_1000")
    a = ap.parse_args()
    print(f"{'seed':>6}{'corner_miss':>13}{'corner_obs':>12}{'miss_rate':>11}"
          f"{'oracle_corner':>15}{'oracle_outside':>16}{'oracle_gap':>12}")
    for s in [x for x in a.seeds.split(",") if x]:
        p = Path(a.runs_root) / f"dataseed-s{s}" / "DATA/STAN/DOMAIN3-SIGMOID" / a.split / "data_bundle.json"
        if not p.is_file():
            print(f"{s:>6}  missing {p}"); continue
        o, c = oracle(p)
        cm, co = c[("corner", "missing")], c[("corner", "observed")]
        oc = o["corner"][0]; oo = o["outside"][0]
        print(f"{s:>6}{cm:>13}{co:>12}{cm/(cm+co):>11.3f}"
              f"{oc:>15.4f}{oo:>16.4f}{oc-oo:>+12.4f}")

if __name__ == "__main__":
    main()
