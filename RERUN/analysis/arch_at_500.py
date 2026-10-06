#!/usr/bin/env python
"""Old block vs standard block at item/500, matched epoch AND matched 100% data.

The one cell the architecture factor was never measured at. Old = replicate-01
(400 epochs, 100% of the training partition). New = the F13 refit (100% of the
training partition). Overlapping periodic checkpoints: 24, 49, 74, 99.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"; RERUN = MAR / "RERUN"
OLD = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_500_MARFORMER"
NEW = RERUN / "runs/final-dual/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/valblock/seed42/DOMAIN3-SIGMOID_Item_T_500_MARFORMER"

sys.path.insert(0, str(RANK))
spec = importlib.util.spec_from_file_location("pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
pd3 = importlib.util.module_from_spec(spec); sys.modules["pd3"] = pd3; spec.loader.exec_module(pd3)
pd3.METRIC_POPULATION = "corner"; shared = pd3.shared

def score(run, ck):
    probs, labels = pd3._marformer_probs_all_missing(run, ck, 500, "cuda")
    ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
    conf, acc = shared._all_class_calibration(probs, labels)
    ce = float(shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500, report_CE=True).get("ce", float("nan")))
    return ll, ce

print(f"{'epoch':>7}{'OLD block ll':>14}{'OLD smECE':>11}{'NEW block ll':>14}{'NEW smECE':>11}{'smECE diff':>12}")
out = []
for ep in (24, 49, 74, 99):
    ck = f"periodic-epoch={ep:04d}.ckpt"
    if not (OLD / "checkpoints" / ck).exists() or not (NEW / "checkpoints" / ck).exists():
        print(f"{ep:>7}   missing"); continue
    oll, oce = score(OLD, ck); nll, nce = score(NEW, ck)
    out.append({"epoch": ep, "old_ll": oll, "old_smece": oce, "new_ll": nll, "new_smece": nce})
    print(f"{ep:>7}{oll:>14.4f}{oce:>11.4f}{nll:>14.4f}{nce:>11.4f}{nce-oce:>+12.4f}", flush=True)
if out:
    d = [r["new_smece"] - r["old_smece"] for r in out]
    print(f"\nARCHITECTURE EFFECT at item/500, matched epoch + matched data: "
          f"mean {sum(d)/len(d):+.4f}  (n={len(d)})")
Path(RERUN / "runs/final-dual/reports/arch_at_500.json").write_text(json.dumps(out, indent=2))
