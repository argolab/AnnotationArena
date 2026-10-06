#!/usr/bin/env python
"""Old block vs standard block: matched epoch, matched 100% data, several sizes.

Generalises arch_at_500.py. OLD = replicate-01 (400 epochs, 100% of the training
partition). NEW = the F13 refit (100% of the training partition). Only the
overlapping periodic checkpoints are scored, so epoch and data fraction are both
held fixed and the architecture is the only difference.

The number that matters is the diff at the LAST overlapping epoch -- the refit stops
at its F13-selected epoch, so that is the checkpoint the paper actually reports.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path
import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"; RERUN = MAR / "RERUN"
OLDR = RERUN / "runs/replicate-01/RESULTS"
NEWR = RERUN / "runs/final-dual/RESULTS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    help="comma list like item/200,item/400,item/1000,annot/30")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-o", "--out", default=str(RERUN / "runs/final-dual/reports/arch_sweep.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3 = importlib.util.module_from_spec(spec); sys.modules["pd3"] = pd3
    spec.loader.exec_module(pd3)
    pd3.METRIC_POPULATION = "corner"; shared = pd3.shared

    def score(run, ck):
        probs, labels = pd3._marformer_probs_all_missing(run, ck, 500, a.device)
        ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
        conf, acc = shared._all_class_calibration(probs, labels)
        ce = float(shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                                 report_CE=True).get("ce", float("nan")))
        return ll, ce

    out = []
    for spec_s in a.cells.split(","):
        axis, size = spec_s.split("/"); size = int(size)
        Ax = axis.capitalize()
        old = OLDR / f"MARFORMER/DOMAIN3-SIGMOID/{axis.upper()}/DOMAIN3-SIGMOID_{Ax}_T_{size}_MARFORMER"
        new = NEWR / f"MARFORMER/DOMAIN3-SIGMOID/{axis.upper()}/valblock/seed42/DOMAIN3-SIGMOID_{Ax}_T_{size}_MARFORMER"
        if not old.is_dir() or not new.is_dir():
            print(f"[miss] {axis}/{size}"); continue
        def eps(p):
            return {int(m.group(1)) for f in (p / "checkpoints").glob("periodic-epoch=*.ckpt")
                    if (m := re.search(r"(\d+)\.ckpt$", f.name))}
        common = sorted(eps(old) & eps(new))
        if not common:
            print(f"[no overlap] {axis}/{size}"); continue
        rows = []
        for ep in common:
            ck = f"periodic-epoch={ep:04d}.ckpt"
            oll, oce = score(old, ck); nll, nce = score(new, ck)
            rows.append({"epoch": ep, "old_smece": oce, "new_smece": nce,
                         "old_ll": oll, "new_ll": nll})
            print(f"  {axis}/{size:<5} ep={ep:<4} old {oce:.4f} / new {nce:.4f}"
                  f"   diff {nce-oce:+.4f}", flush=True)
        d = [r["new_smece"] - r["old_smece"] for r in rows]
        rec = {"cell": f"{axis}/{size}", "rows": rows,
               "mean_diff": float(np.mean(d)),
               "diff_at_last_overlap": d[-1], "last_epoch": common[-1]}
        out.append(rec)
        print(f"  == {axis}/{size}: mean {np.mean(d):+.4f}, "
              f"at reported epoch {common[-1]} {d[-1]:+.4f}\n", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\n=== ARCHITECTURE EFFECT ON smECE (new block minus old block) ===")
    print(f"{'cell':<14}{'mean':>10}{'at reported epoch':>20}")
    for r in out:
        print(f"{r['cell']:<14}{r['mean_diff']:>+10.4f}{r['diff_at_last_overlap']:>+20.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
