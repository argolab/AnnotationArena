#!/usr/bin/env python
"""smECE with bootstrap CI for every size, both protocols, all four Stan variants.

PURPOSE AND ITS LIMIT. The paper's calibration figure has to report SOME size, and
the question is which. Choosing the size where Marformer looks best is selection on
test -- the same bug as D14, and the resulting "on par" would not be defensible.
This script therefore reports the WHOLE grid so the choice can be made on an a
priori rule and then simply read off. The defensible rule is the largest size on
each axis that every baseline was fitted at (the asymptotic regime, fixed before
looking): K=4000 on the item axis, J=40 on the annotator axis, since Stan Oracle
has no annot fit past 40.

"On par" here follows the paper's own convention: the gap is smaller than the
bootstrap CI half-width of the Marformer estimate. That is a weak test -- a paired
bootstrap of the DIFFERENCE would be stronger -- but it is the convention the
existing figure captions use, so it is what the claim has to clear.

    python smece_vs_stan_grid.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"
RUNS = MAR / "RERUN/runs/final-dual"
ITEM_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ANNOT_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
STAN_SERIES = [("Stan Oracle", "ORACLE", "TENSOR-SIG"), ("Stan Proj", "MISSPEC", "PROJ"),
               ("Stan Bin", "MISSPEC", "BIN"), ("Stan Both", "MISSPEC", "BOTH")]


def load(name, path):
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def ce_ci(shared, probs, labels):
    conf, acc = shared._all_class_calibration(probs, labels)
    d = shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                      report_CE=True, report_CE_std=True)
    return float(d.get("ce", float("nan"))), float(d.get("ce_ci_width", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("-o", "--out", default=str(RUNS / "reports/smece_vs_stan_grid.json"))
    a = ap.parse_args()
    pd3 = load("pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3.METRIC_POPULATION = "corner"
    shared = pd3.shared
    out: dict = {"axes": {}}

    corner = json.loads((RUNS / "corner_metrics/corner.json").read_text())
    f13_runs = {}
    for r in corner["rows"]:
        if r["family"] != "Marformer":
            continue
        f13_runs[(r["axis"], int(r["size"]))] = (
            RUNS / "RESULTS/MARFORMER/DOMAIN3-SIGMOID" / r["variant"]
            / f"seed{r['seed']}"
            / f"DOMAIN3-SIGMOID_{r['axis'].capitalize()}_T_{r['size']}_MARFORMER")
    search_runs = {}
    for axis, ad in (("item", "ITEM"), ("annot", "ANNOT")):
        root = RUNS / "RESULTS/MARFORMER/DOMAIN3-SIGMOID" / ad / "valblock-search"
        for run in root.rglob("*_MARFORMER"):
            import re
            m = re.search(r"_(Item|Annot)_T_(\d+)_MARFORMER$", run.name)
            if m:
                search_runs[(axis, int(m.group(2)))] = run

    for axis, cfg, sizes in (("item", pd3.ITEM_CFG, ITEM_SIZES),
                             ("annot", pd3.ANNOT_CFG, ANNOT_SIZES)):
        res: dict[str, dict] = {}
        for label, src, ck in (("Marformer F13", f13_runs, "last.ckpt"),
                               ("Marformer @200", search_runs, "periodic-epoch=0199.ckpt")):
            d = {}
            for s in sizes:
                run = src.get((axis, s))
                if run is None or not (run / "checkpoints" / ck).exists():
                    continue
                probs, labels = pd3._marformer_probs_all_missing(run, ck, a.max_item, a.device)
                ce, ci = ce_ci(shared, probs, labels)
                d[s] = {"smece": ce, "ci": ci}
                print(f"  {axis}/{s:<5} {label:<16} smECE {ce:.4f} +/- {ci:.4f}", flush=True)
            res[label] = d
        oracle, _ = pd3._collect_oracle_payloads(cfg)
        for label, variant, tag in STAN_SERIES:
            pool = oracle if variant == "ORACLE" else pd3._collect_misspec_payloads(cfg, tag)[0]
            d = {}
            for s in sizes:
                pay = pool.get(s)
                if pay is None:
                    continue
                ce, ci = ce_ci(shared, np.asarray(pay[0], dtype=np.float64),
                               np.asarray(pay[1], dtype=np.int64))
                d[s] = {"smece": ce, "ci": ci}
                print(f"  {axis}/{s:<5} {label:<16} smECE {ce:.4f} +/- {ci:.4f}", flush=True)
            res[label] = d
        out["axes"][axis] = res
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
