#!/usr/bin/env python
"""Separate the two corrections that moved smECE.

The regenerated calibration panels report smECE far above the published values
and reverse the ordering against Stan. Two corrections landed at once:

  population  pooled -> corner   (D22/D23: score only var.instance == "test")
  checkpoint  test-argmin -> last (D14 at the plot layer: F13 reports the refit's
                                   last epoch, never an argmin on test)

Reporting a cause without separating them would be a guess, so this runs the full
2x2 on the cells the paper quotes and reports smECE and corner/pooled log-loss for
each. The checkpoint axis reuses the OLD selection rule verbatim -- argmin of
TEST_RESULTS/*.json `missing.log_loss`, ties broken by rmse then name -- so the
"test-argmin" column is the checkpoint the published figures actually used.

Stan is on neither axis: it has no checkpoints, and its payloads are already the
test corner. Any change in Stan's smECE is because Stan was refit, not because of
these two corrections.

    python smece_decompose.py            # needs a GPU; ~16 forward passes
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys
from pathlib import Path

import numpy as np

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")  # overridable

CELLS = [
    ("item", 500, "Marformer",
     "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_500_MARFORMER"),
    ("item", 500, "Marformer NT",
     "MARFORMER-NT/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_NT_500_MARFORMER"),
    ("annot", 30, "Marformer",
     "MARFORMER/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_T_30_MARFORMER"),
    ("annot", 30, "Marformer NT",
     "MARFORMER-NT/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_NT_30_MARFORMER"),
]
# What the paper currently prints, for reference only.
PUBLISHED = {("item", 500, "Marformer"): 0.016, ("item", 500, "Marformer NT"): 0.016,
             ("annot", 30, "Marformer"): 0.022, ("annot", 30, "Marformer NT"): 0.014}


def load_module():
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    return m


def test_argmin_ckpt(run_dir: Path) -> str:
    """The old rule, verbatim: argmin of TEST_RESULTS missing.log_loss."""
    cands = []
    for path in sorted((run_dir / "TEST_RESULTS").glob("*.json")):
        if path.name in {"best.json", "launcher_metadata.json"}:
            continue
        d = json.loads(path.read_text())
        mi = d.get("missing", {})
        ll = mi.get("log_loss")
        if ll is None or mi.get("n", 0) == 0:
            continue
        cands.append((float(ll), float(mi.get("rmse") or math.inf), path.name))
    if not cands:
        raise RuntimeError(f"no checkpoint JSONs in {run_dir}")
    cands.sort()
    return cands[0][2].replace(".json", ".ckpt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-max-item", type=int, default=None,
                    help="force the evaluation chunk instead of using each run's "
                         "trained value. replicate-01 trained at 300 and final-dual "
                         "at 500, so comparing them at their own values confounds "
                         "the model change with a change of evaluation context.")
    ap.add_argument("--run", default=None,
                    help="score a different run tree (e.g. replicate-01, the "
                         "pre-rerun models) to test whether retraining, rather "
                         "than either policy, moved the numbers")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    global RUNS
    if a.run:
        RUNS = RUNS.parent / a.run
    m = load_module()
    shared, rp = m.shared, m.shared.rp

    rows = []
    for axis, size, model, rel in CELLS:
        run_dir = RUNS / "RESULTS" / rel
        cfg = m.ITEM_CFG if axis == "item" else m.ANNOT_CFG
        max_item = (a.eval_max_item if a.eval_max_item is not None
                    else m._marformer_eval_max_item(cfg, size))
        ckpts = {"last": "last.ckpt", "test-argmin": test_argmin_ckpt(run_dir)}
        for pop in ("corner", "pooled"):
            m.METRIC_POPULATION = pop
            for cname, ckpt in ckpts.items():
                probs, labels = m._marformer_probs_all_missing(
                    run_dir, ckpt, max_item, a.device)
                conf, acc = shared._all_class_calibration(probs, labels)
                diag = rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                              report_CE=True, report_CE_std=True)
                nll = float(-np.log(np.clip(
                    probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
                rows.append({"axis": axis, "size": size, "model": model,
                             "population": pop, "checkpoint": cname,
                             "ckpt_file": ckpt, "n": int(len(labels)),
                             "smece": float(diag.get("ce", float("nan"))),
                             "ci_width": float(diag.get("ce_ci_width", 0.0)),
                             "log_loss": nll})
                print(f"  {axis}/{size} {model:<13} {pop:<7} {cname:<12} "
                      f"n={len(labels):<6} smECE={rows[-1]['smece']:.4f} "
                      f"ll={nll:.4f}  [{ckpt}]", flush=True)

    out = Path(a.out) if a.out else RUNS / "reports/smece_decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    def get(axis, size, model, pop, ck, field="smece"):
        for r in rows:
            if (r["axis"], r["size"], r["model"], r["population"],
                    r["checkpoint"]) == (axis, size, model, pop, ck):
                return r[field]
        return float("nan")

    for field in ("smece", "log_loss"):
        print(f"\n=== {field}: the 2x2, and each factor's effect ===")
        hdr = (f"{'cell':<26}{'pooled/argmin':>15}{'pooled/last':>13}"
               f"{'corner/argmin':>15}{'corner/last':>13}"
               f"{'pop effect':>12}{'ckpt effect':>13}")
        print(hdr); print("-" * len(hdr))
        for axis, size, model, _ in CELLS:
            pa = get(axis, size, model, "pooled", "test-argmin", field)
            pl = get(axis, size, model, "pooled", "last", field)
            ca = get(axis, size, model, "corner", "test-argmin", field)
            cl = get(axis, size, model, "corner", "last", field)
            # published == pooled/argmin ; current == corner/last
            # population effect, averaged over both checkpoint settings
            pop_eff = ((ca - pa) + (cl - pl)) / 2
            ckpt_eff = ((pl - pa) + (cl - ca)) / 2
            print(f"{axis+'/'+str(size)+' '+model:<26}{pa:>15.4f}{pl:>13.4f}"
                  f"{ca:>15.4f}{cl:>13.4f}{pop_eff:>+12.4f}{ckpt_eff:>+13.4f}")
        if field == "smece":
            print("\n  'published' corresponds to pooled/test-argmin, 'current' to corner/last.")
            print("  Each effect is the main effect: the mean change from flipping that one"
                  "\n  factor, holding the other at both of its levels in turn.")
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
