#!/usr/bin/env python
"""How wide should the band be? Naive SE assumes cells are independent. They are not.

The 316 corner cells are not 316 independent observations: they are ~a few dozen
items x a few judges x 9 criteria, and two cells sharing an item share that
item's difficulty. SD/sqrt(n) therefore UNDERSTATES the real sampling error of
the mean. This measures by how much, by bootstrapping over the unit the axis
actually generalises over (items on the item axis, judges on the judge axis)
and comparing against the naive figure and against the band currently drawn.

No GPU and no model: reads the plotted payload cache and the data bundle.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RANK))
spec = importlib.util.spec_from_file_location(
    "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
m = importlib.util.module_from_spec(spec); sys.modules["pd3"] = m
spec.loader.exec_module(m)


def cells_for(cfg, size):
    """(per-cell nll, cluster id) for the plotted Marformer payload."""
    payloads, _ = m._load_marformer_cache(cfg.cache_path)
    if size not in payloads:
        return None, None
    _ckpt, probs, labels = payloads[size]
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))
    bundle = m._read_json(m._bundle_path(cfg, size))
    idxs, _lab = m._test_missing_indices_and_labels(bundle)
    rows = [bundle["missing_ratings"][i] for i in idxs]
    if len(rows) != len(nll):
        return nll, None
    key = "item" if cfg.slug == "item" else "annotator"
    return nll, np.asarray([r[key] for r in rows])


def cluster_boot(nll, cluster, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, c in enumerate(cluster):
        groups.setdefault(c, []).append(i)
    gl = [np.asarray(v) for v in groups.values()]
    G = len(gl)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        means[b] = np.concatenate([gl[i] for i in pick]).size and \
                   nll[np.concatenate([gl[i] for i in pick])].mean()
    return float(means.std(ddof=1)), G


def main() -> int:
    print(f"{'cell':<13}{'n':>5}{'clusters':>9}{'mean':>8}"
          f"{'plotted':>10}{'SD/sqrt(n)':>12}{'clustered':>11}{'ratio':>7}")
    print("-" * 76)
    for cfg, sizes in ((m.ITEM_CFG, (500, 2000, 4000)), (m.ANNOT_CFG, (30, 40))):
        for size in sizes:
            try:
                nll, cl = cells_for(cfg, size)
            except Exception as e:
                print(f"{cfg.slug}/{size:<8} [skip] {type(e).__name__}"); continue
            if nll is None:
                continue
            sd = float(nll.std(ddof=0)); n = len(nll)
            naive = sd / np.sqrt(n)
            plotted = 0.05 * sd
            if cl is None:
                print(f"{cfg.slug+'/'+str(size):<13}{n:>5}{'?':>9}{nll.mean():>8.4f}"
                      f"{plotted:>10.4f}{naive:>12.4f}{'--':>11}{'--':>7}")
                continue
            cse, G = cluster_boot(nll, cl)
            print(f"{cfg.slug+'/'+str(size):<13}{n:>5}{G:>9}{nll.mean():>8.4f}"
                  f"{plotted:>10.4f}{naive:>12.4f}{cse:>11.4f}{cse/naive:>7.2f}x")
    print("\nplotted   = the current band, 0.05 x SD")
    print("clustered = bootstrap over the generalisation unit (items / judges)")
    print("ratio     = how much the naive SE understates the clustered one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
