#!/usr/bin/env python
"""
Model-free test of the train-vs-corner gap.

If the gap is Marformer failing to generalise, it should not appear in a
predictor that does no generalisation at all. If it IS the draw -- if seed 42
simply dealt a harder 414 cells -- then even a marginal-frequency predictor,
fitted on the training partition and scored on both regions, shows the same
sign.

Three predictors, increasing in strength, none of which can transfer anything
item-specific to an unseen item:
  unigram      P(c) over all observed training labels
  by-attr      P(c | attribute)
  by-annot-attr P(c | annotator, attribute)      <- the strongest thing a model
                                                    can know about a corner cell
                                                    without knowing the item
All are fitted on OBSERVED TRAINING cells only and scored on the missing cells
of each region, with add-one smoothing.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

C = 4

def load(p: Path):
    b = json.loads(p.read_text())
    md = b["domain3_metadata"]
    ti = set(md["test_block_item_orig_ids"]); ta = set(md["test_block_annotator_orig_ids"])
    i2o = {k + 1: v for k, v in enumerate(md["selected_item_orig_ids"])}
    a2o = {k + 1: v for k, v in enumerate(md["selected_annotator_orig_ids"])}
    def tag(r):
        return "corner" if (i2o[r["item"]] in ti and a2o[r["annotator"]] in ta) else "train"
    return b, tag

def fit(rows, keyf):
    tab = defaultdict(lambda: [1.0] * C)          # add-one
    for r in rows:
        tab[keyf(r)][r["value"] - 1] += 1.0
    return {k: [x / sum(v) for x in v] for k, v in tab.items()}

def score(tab, rows, keyf):
    tot, n = 0.0, 0
    unif = [1.0 / C] * C
    for r in rows:
        p = tab.get(keyf(r), unif)
        tot += -math.log(max(p[r["value"] - 1], 1e-12)); n += 1
    return (tot / n if n else float("nan")), n

PREDICTORS = {
    "unigram":       lambda r: 0,
    "by-attr":       lambda r: r["attribute"],
    "by-annot-attr": lambda r: (r["annotator"], r["attribute"]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,102,103,104,105")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--split", default="ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_1000")
    ap.add_argument("-o", "--out", default="dataseed/out/corner_difficulty.json")
    a = ap.parse_args()
    out = []
    for name, keyf in PREDICTORS.items():
        print(f"\n=== {name} (fitted on observed training cells) ===")
        print(f"{'seed':>6}{'n_corner':>10}{'n_train':>9}{'corner':>9}{'train':>9}{'gap':>9}")
        for s in [x for x in a.seeds.split(",") if x]:
            p = Path(a.runs_root) / f"dataseed-s{s}" / "DATA/STAN/DOMAIN3-SIGMOID" / a.split / "data_bundle.json"
            if not p.is_file(): continue
            b, tag = load(p)
            train_obs = [r for r in b["observed_ratings"] if tag(r) == "train"]
            tab = fit(train_obs, keyf)
            miss = b["missing_ratings"]
            lc, nc = score(tab, [r for r in miss if tag(r) == "corner"], keyf)
            lt, nt = score(tab, [r for r in miss if tag(r) == "train"], keyf)
            print(f"{s:>6}{nc:>10}{nt:>9}{lc:>9.4f}{lt:>9.4f}{lc-lt:>+9.4f}")
            out.append({"predictor": name, "seed": int(s), "n_corner": nc,
                        "n_train": nt, "corner": lc, "train": lt, "gap": lc - lt})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {a.out}")

if __name__ == "__main__":
    main()
