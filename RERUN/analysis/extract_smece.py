#!/usr/bin/env python
"""Write out the smECE values the calibration figures display.

The paper quotes smECE in prose (0.016 +/- 0.007 and so on) but the figure
pipeline only ever renders those numbers into a plot title -- they are in no
summary JSON, so after a rerun the prose and the figure can drift apart with
nothing to catch it. This reads the SAME payloads the calibration panels are
drawn from and calls the SAME relplot path (`_all_class_calibration` ->
`rp.prepare_rel_diagram(..., report_CE=True, report_CE_std=True)`), so the
numbers are the plotted ones by construction rather than a reimplementation.

No GPU: the Marformer and NT probabilities come from the figure build's own
`*_cache.npz`, and the Stan payloads are read from CSVs the way the plot script
reads them.

    python extract_smece.py --run final-dual --item-size 500 --annot-size 30
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")


def load_plot_module():
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m      # dataclass() resolves cls.__module__ via sys.modules
    spec.loader.exec_module(m)
    return m


def smece(shared, rp, probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    conf, acc = shared._all_class_calibration(probs, labels)
    diag = rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                  report_CE=True, report_CE_std=True)
    return float(diag.get("ce", float("nan"))), float(diag.get("ce_ci_width", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final-dual")
    ap.add_argument("--item-size", type=int, default=500)
    ap.add_argument("--annot-size", type=int, default=30)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    m = load_plot_module()
    shared, rp = m.shared, m.shared.rp
    out: dict = {"run": a.run, "sizes": {"item": a.item_size, "annot": a.annot_size},
                 "note": "same payloads and same relplot call as the calibration panels",
                 "cells": []}

    for cfg, size in ((m.ITEM_CFG, a.item_size), (m.ANNOT_CFG, a.annot_size)):
        payloads, nt_payloads = m._load_marformer_cache(cfg.cache_path)
        oracle_payloads, _ = m._collect_oracle_payloads(cfg)
        series = {
            "Marformer": payloads.get(size),
            "Marformer NT": nt_payloads.get(size),
        }
        for name, pay in series.items():
            if pay is None:
                print(f"  {cfg.slug}/{size:<5} {name:<14} MISSING from the cache")
                continue
            _ckpt, probs, labels = pay
            ce, ci = smece(shared, rp, probs, labels)
            out["cells"].append({"axis": cfg.slug, "size": size, "model": name,
                                 "smece": ce, "ci_width": ci, "n": int(len(labels))})
            print(f"  {cfg.slug}/{size:<5} {name:<14} smECE {ce:.3f} +/- {ci:.3f}  (n={len(labels)})")
        src = m._source_size_for_cfg(cfg, size, set(oracle_payloads))
        if src is None:
            print(f"  {cfg.slug}/{size:<5} {'Stan Oracle':<14} MISSING (no fit at this size)")
        else:
            probs, labels = oracle_payloads[src]
            ce, ci = smece(shared, rp, probs, labels)
            out["cells"].append({"axis": cfg.slug, "size": size, "model": "Stan Oracle",
                                 "source_size": src, "smece": ce, "ci_width": ci,
                                 "n": int(len(labels))})
            print(f"  {cfg.slug}/{size:<5} {'Stan Oracle':<14} smECE {ce:.3f} +/- {ci:.3f}  (n={len(labels)})")

    dest = Path(a.out) if a.out else RERUN / "runs" / a.run / "reports" / "smece.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out['cells'])} cells -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
