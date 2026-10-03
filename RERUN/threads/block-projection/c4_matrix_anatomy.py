#!/usr/bin/env python
"""C4: what has the block's output projection actually learned?

M is the d x d map applied to the block body before it is added back to the
stream. Written as two halves, its rows split where the stream does: the first
`feature_dim` rows write the feature slice, the last `param_dim` rows write the
value slice that carries the prediction. Its COLUMNS split the same way -- which
part of the stream each output reads from.

That gives four blocks, and their relative sizes say what the projection is for:

    M = [ F<-F   F<-V ]      rows: where the output goes
        [ V<-F   V<-V ]      cols: where the input came from

V<-F is the interesting one: value written from features, i.e. the block turning
what it has inferred about an item and annotator into a prediction. If the
projection is load-bearing, that is where we should see it.

Run on trained checkpoints across several training sizes so nothing is read off
one model.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch

RANKING_ROOT = Path(os.environ["RANKING_ROOT"]).resolve()
RERUN = RANKING_ROOT.parent.parent / "RERUN"
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(RANKING_ROOT))
from imputer.entity_mf.test import _reconstruct, _find_checkpoint, _load_checkpoint

BASE = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
SIZES = [300, 1000, 2000, 4000]
OUT = Path(__file__).resolve().parent / "out/c4_matrix.json"


def blocks_of(M: torch.Tensor, F: int):
    return {"F<-F": M[:F, :F], "F<-V": M[:F, F:], "V<-F": M[F:, :F], "V<-V": M[F:, F:]}


def main() -> None:
    dev = torch.device("cpu")
    report = {}
    for K in SIZES:
        run = BASE / f"DOMAIN3-SIGMOID_Item_T_{K}_MARFORMER"
        if not (run / "training_history.json").is_file():
            continue
        model, _, _ = _reconstruct(run)
        _load_checkpoint(model, _find_checkpoint(run / "checkpoints", "last.ckpt"), dev)
        F, P, D = model.feature_dim, model.param_dim, model.model_dim
        per_layer = []
        for i, b in enumerate(model.blocks):
            M = torch.cat([b["proj_out"].weight.data, b["W_param"].weight.data], 0)
            sv = torch.linalg.svdvals(M)
            # participation ratio: a scale-free "how many directions is it using"
            eff_rank = float((sv.sum() ** 2) / (sv.pow(2).sum()))
            # how much of M is just a scaled identity?
            alpha = float((M * torch.eye(D)).sum() / D)
            resid = float((M - alpha * torch.eye(D)).norm() / M.norm())
            bl = blocks_of(M, F)
            # root-mean-square weight per block, so blocks of different shape compare
            rms = {k: float(v.pow(2).mean().sqrt()) for k, v in bl.items()}
            per_layer.append({
                "layer": i, "frobenius": round(float(M.norm()), 3),
                "top_sv": round(float(sv[0]), 3),
                "effective_rank": round(eff_rank, 1), "of_dim": D,
                "identity_alpha": round(alpha, 4),
                "residual_after_removing_identity": round(resid, 4),
                "rms": {k: round(v, 5) for k, v in rms.items()},
                "rms_ratio_VfromF_over_VfromV": round(rms["V<-F"] / rms["V<-V"], 3),
            })
        report[str(K)] = per_layer
        print(f"\n=== K={K} ===")
        print(f"{'layer':>5}{'|M|':>8}{'top sv':>8}{'eff.rank':>9}{'alpha':>8}"
              f"{'F<-F':>9}{'F<-V':>9}{'V<-F':>9}{'V<-V':>9}")
        for r in per_layer:
            m = r["rms"]
            print(f"{r['layer']:>5}{r['frobenius']:>8.2f}{r['top_sv']:>8.2f}"
                  f"{r['effective_rank']:>9.1f}{r['identity_alpha']:>8.3f}"
                  f"{m['F<-F']:>9.4f}{m['F<-V']:>9.4f}{m['V<-F']:>9.4f}{m['V<-V']:>9.4f}")
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}  (effective rank is out of {80})")


if __name__ == "__main__":
    main()
