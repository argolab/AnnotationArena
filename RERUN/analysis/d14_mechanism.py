#!/usr/bin/env python
"""Does argmin-on-test-LOG-LOSS drag smECE along with it?

The proposed explanation for why the new protocol looks worse calibrated: the old
numbers picked the checkpoint with the best test log-loss (D14 -- confirmed, the old
item/500 best.json is periodic-epoch=0224, the argmin over all 16 checkpoints), and
that selection also lands on a well-calibrated checkpoint. We cannot select that way,
so we lose the bias.

To test the MECHANISM rather than argue it, score every checkpoint of the OLD runs on
both metrics and ask:
  (a) how well does test log-loss rank checkpoints by smECE? (Spearman)
  (b) what smECE does the log-loss argmin get, vs the best available, vs a typical
      checkpoint, vs the epoch F13 would have selected?

If the correlation is strong and the argmin lands near the smECE optimum, the
explanation holds. If it is weak, the old advantage came from somewhere else.

    python d14_mechanism.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"
RERUN = MAR / "RERUN"
OLD = RERUN / "runs/replicate-01/RESULTS"

CELLS = [("Marformer", "item", 500, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_500_MARFORMER"),
         ("Marformer", "annot", 30, "MARFORMER/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_T_30_MARFORMER"),
         ("Marformer", "item", 300, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"),
         ("Marformer", "item", 2000, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_2000_MARFORMER")]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = float(p)
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("-o", "--out",
                    default=str(RERUN / "runs/final-dual/reports/d14_mechanism.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3 = importlib.util.module_from_spec(spec); sys.modules["pd3"] = pd3
    spec.loader.exec_module(pd3)
    pd3.METRIC_POPULATION = "corner"
    shared = pd3.shared

    out = []
    for fam, axis, size, rel in CELLS:
        run = OLD / rel
        if not run.is_dir():
            print(f"[miss] {run}"); continue
        cks = sorted((int(re.search(r"(\d+)\.ckpt$", p.name).group(1)), p.name)
                     for p in (run / "checkpoints").glob("periodic-epoch=*.ckpt"))
        rows = []
        for ep, name in cks:
            probs, labels = pd3._marformer_probs_all_missing(run, name, a.max_item, a.device)
            ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
            conf, acc = shared._all_class_calibration(probs, labels)
            ce = float(shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                                     report_CE=True).get("ce", float("nan")))
            rows.append({"epoch": ep, "ll": ll, "smece": ce})
            print(f"  {axis}/{size} ep={ep:<5} ll={ll:.4f} smECE={ce:.4f}", flush=True)
        if len(rows) < 4:
            continue
        lls = [r["ll"] for r in rows]; ces = [r["smece"] for r in rows]
        amin_ll = min(rows, key=lambda r: r["ll"])
        amin_ce = min(rows, key=lambda r: r["smece"])
        rho = spearman(lls, ces)
        rec = {"cell": f"{axis}/{size}", "rows": rows, "spearman_ll_smece": rho,
               "argmin_ll": amin_ll, "argmin_smece": amin_ce,
               "median_smece": float(np.median(ces))}
        out.append(rec)
        print(f"\n  == {axis}/{size} ==")
        print(f"     log-loss argmin @ep{amin_ll['epoch']}: smECE {amin_ll['smece']:.4f}")
        print(f"     best smECE       @ep{amin_ce['epoch']}: smECE {amin_ce['smece']:.4f}")
        print(f"     median checkpoint smECE            : {np.median(ces):.4f}")
        print(f"     Spearman(test log-loss, test smECE): {rho:+.2f}  (n={len(rows)})")
        print(f"     selection bonus vs a median checkpoint: "
              f"{amin_ll['smece'] - float(np.median(ces)):+.4f}\n", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
