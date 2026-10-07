#!/usr/bin/env python
"""Does choosing the unigram variant honestly change the Best Unigram curve?

Until 2026-10-06 `_best_unigram_result` scored all eight variants against the
TEST labels and took the argmin, which is the same error as D14 -- the baseline
saw the answers. It now selects on a seeded 15% carve of the observed pool, then
merges that carve back and refits the chosen variant on everything.

That can only make the baseline weaker or equal, so the comparison moves in our
disfavour, which is the right direction for a fix to move. This measures by how
much, and whether the chosen variant actually differs.

No GPU: unigram is counting.
"""
from __future__ import annotations
import importlib.util, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RANK))
spec = importlib.util.spec_from_file_location(
    "shared", RANK / "scripts/utils/plot_domain3_generalization.py")
shared = importlib.util.module_from_spec(spec); sys.modules["shared"] = shared
spec.loader.exec_module(shared)


def oracle_pick(slug, size):
    """The OLD behaviour, reconstructed: argmin of test log-loss over variants."""
    bundle = shared._bundle(slug, size)
    observed = bundle.get("observed_ratings", [])
    idxs, labels = shared._test_missing_indices_and_labels(bundle)
    test_missing = [bundle["missing_ratings"][i] for i in idxs]
    if not observed or not test_missing:
        return None
    C = max(max(r["value"] for r in observed), max(r["value"] for r in test_missing))
    gp = shared._global_probs_from_observed(observed, C)
    best = None
    for name, fields in shared.UNIGRAM_VARIANTS.items():
        table = shared._unigram_counts(observed, fields, C)
        probs = shared._unigram_predict(table, gp, test_missing, fields, C)
        ll = float(np.mean(shared._per_example_nll(probs, labels)))
        if best is None or ll < best[1]:
            best = (name, ll)
    return best


def main() -> int:
    print(f"{'axis/size':<14}{'dev-selected':<26}{'test-argmin (old)':<26}{'cost':>9}")
    print("-" * 76)
    rows = []
    for slug, sizes in (("item", [100, 200, 300, 400, 500, 600, 700, 800, 900,
                                 1000, 2000, 3000, 4000]),
                        ("annot", [10, 15, 20, 25, 30, 35, 40, 45, 50])):
        for size in sizes:
            try:
                new = shared._best_unigram_result(slug, size)
                old = oracle_pick(slug, size)
            except Exception as e:                      # bundle absent for this cell
                print(f"{slug}/{size:<9} [skip] {type(e).__name__}")
                continue
            if new is None or old is None:
                continue
            d = new["mean_log_loss"] - old[1]
            same = "same" if new["variant"] == old[0] else "DIFFERENT"
            print(f"{slug+'/'+str(size):<14}"
                  f"{new['variant']+' '+format(new['mean_log_loss'],'.4f'):<26}"
                  f"{old[0]+' '+format(old[1],'.4f'):<26}{d:>+9.4f}  {same}")
            rows.append((slug, size, new["variant"], old[0], d))

    if rows:
        ds = [r[4] for r in rows]
        diff = [r for r in rows if r[2] != r[3]]
        print(f"\n{len(rows)} cells; the honest selection picks a different variant in "
              f"{len(diff)}/{len(rows)}")
        print(f"mean cost to the baseline {np.mean(ds):+.4f} nats, max {max(ds):+.4f}")
        print("A positive cost means the baseline got worse, i.e. it had been "
              "benefiting from seeing test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
