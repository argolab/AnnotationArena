#!/usr/bin/env python
"""Fixed (criterion, judge) pooling vs honestly dev-selected pooling, on the shipped path.

Tom's prediction (2026-10-06): "I think we will end up observing the same curve
(most of the time)." This tests it on the series the Domain 3 figures actually
draw, so the answer decides whether the baseline needs to change at all.

  fixed    what ships today: pool by (attribute, annotator), no selection.
  dev-sel  carve 15% of the observed pool (seeded), fit all eight variants on the
           rest, pick the lowest dev log-loss, MERGE THE CARVE BACK, refit the
           chosen variant on the full pool, score the corner.

Counting only -- no GPU, no model.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RANK))
spec = importlib.util.spec_from_file_location(
    "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
m = importlib.util.module_from_spec(spec); sys.modules["pd3"] = m
spec.loader.exec_module(m)
sh = m.shared

FIXED = ("attribute", "annotator")      # what PooledUnigramIJ does


def corner_rows(cfg, size):
    b = sh._read_json(m._bundle_path(cfg, size))
    idxs, labels = sh._test_missing_indices_and_labels(b)
    return b, [b["missing_ratings"][i] for i in idxs], labels


def score(observed, fields, rows, labels, C):
    gp = sh._global_probs_from_observed(observed, C)
    tbl = sh._unigram_counts(observed, fields, C)
    probs = sh._unigram_predict(tbl, gp, rows, fields, C)
    return float(np.mean(sh._per_example_nll(probs, labels)))


def main() -> int:
    print(f"{'cell':<13}{'fixed (ij)':>12}{'dev-selected':>14}{'delta':>9}  variant chosen on dev")
    print("-" * 74)
    deltas, same = [], 0
    for cfg, sizes in ((m.ITEM_CFG, (100, 200, 300, 500, 1000, 2000, 4000)),
                       (m.ANNOT_CFG, (10, 20, 30, 40, 50))):
        for size in sizes:
            try:
                b, rows, labels = corner_rows(cfg, size)
            except Exception as e:
                print(f"{cfg.slug}/{size:<8} [skip] {type(e).__name__}"); continue
            observed = b.get("observed_ratings", [])
            if not observed or not rows:
                continue
            C = max(max(r["value"] for r in observed), max(r["value"] for r in rows))

            ll_fixed = score(observed, FIXED, rows, labels, C)

            rng = np.random.default_rng(7)
            order = rng.permutation(len(observed))
            n_dev = int(round(len(observed) * 0.15))
            dev = [observed[i] for i in order[:n_dev]]
            fit = [observed[i] for i in order[n_dev:]]
            dev_lab = np.asarray([int(r["value"]) - 1 for r in dev], dtype=np.int64)
            best, bname = None, None
            for name, fields in sh.UNIGRAM_VARIANTS.items():
                v = score(fit, fields, dev, dev_lab, C)
                if best is None or v < best:
                    best, bname = v, name
            ll_dev = score(observed, sh.UNIGRAM_VARIANTS[bname], rows, labels, C)

            d = ll_dev - ll_fixed
            deltas.append(d)
            if bname == "attribute-annotator":
                same += 1
            print(f"{cfg.slug+'/'+str(size):<13}{ll_fixed:>12.4f}{ll_dev:>14.4f}{d:>+9.4f}  {bname}")

    if deltas:
        print(f"\n{len(deltas)} cells; dev selection lands on (criterion, judge) -- "
              f"i.e. exactly the fixed choice -- in {same}/{len(deltas)}")
        print(f"mean |difference| {np.mean(np.abs(deltas)):.4f} nats, max {np.max(np.abs(deltas)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
