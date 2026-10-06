#!/usr/bin/env python
"""The Marformer NT half of smece_vs_stan_grid.py.

The calibration figure carries three series -- Marformer, Marformer NT and Stan
Oracle -- so a size-by-method smECE table that omits NT does not match the figure
it is meant to justify. Separate script rather than a flag so the transductive grid
job already in flight is not disturbed.

    python smece_grid_nt.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"
RUNS = MAR / "RERUN/runs/final-dual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("-o", "--out", default=str(RUNS / "reports/smece_grid_nt.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3 = importlib.util.module_from_spec(spec); sys.modules["pd3"] = pd3
    spec.loader.exec_module(pd3)
    pd3.METRIC_POPULATION = "corner"
    shared = pd3.shared

    def ce_ci(probs, labels):
        conf, acc = shared._all_class_calibration(probs, labels)
        d = shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                          report_CE=True, report_CE_std=True)
        return float(d.get("ce", float("nan"))), float(d.get("ce_ci_width", 0.0))

    corner = json.loads((RUNS / "corner_metrics/corner.json").read_text())
    out: dict = {"axes": {"item": {}, "annot": {}}}

    f13 = {}
    for r in corner["rows"]:
        if r["family"] != "MarformerNT":
            continue
        f13[(r["axis"], int(r["size"]))] = (
            RUNS / "RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID" / r["variant"]
            / f"seed{r['seed']}"
            / f"DOMAIN3-SIGMOID_{r['axis'].capitalize()}_NT_{r['size']}_MARFORMER")
    search = {}
    for axis, ad in (("item", "ITEM"), ("annot", "ANNOT")):
        root = RUNS / "RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID" / ad / "valblock-search"
        if root.is_dir():
            for run in root.rglob("*_MARFORMER"):
                m = re.search(r"_(Item|Annot)_NT_(\d+)_MARFORMER$", run.name)
                if m:
                    search[(axis, int(m.group(2)))] = run

    for label, src, ck in (("Marformer NT F13", f13, "last.ckpt"),
                           ("Marformer NT @200", search, "periodic-epoch=0199.ckpt")):
        for (axis, size), run in sorted(src.items()):
            if not (run / "checkpoints" / ck).exists():
                print(f"  [miss] {label} {axis}/{size}"); continue
            probs, labels = pd3._marformer_probs_all_missing(run, ck, a.max_item, a.device)
            ce, ci = ce_ci(probs, labels)
            out["axes"][axis].setdefault(label, {})[size] = {"smece": ce, "ci": ci}
            print(f"  {axis}/{size:<5} {label:<20} smECE {ce:.4f} +/- {ci:.4f}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
