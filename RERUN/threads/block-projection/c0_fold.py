#!/usr/bin/env python
"""C0: is the block's trailing d->d projection foldable into the feed-forward?

The claim from the meeting: the feed-forward already ends in a d_ff -> d linear
with no activation after it, so the block's trailing d -> d projection should
compose with it into a single projection and be removable for free.

This script settles it numerically, both ways.

  * With the inner residual OFF, the block's output is the feed-forward output
    alone, the two linears are adjacent, and the fold is exact. We build the
    folded model and assert the forward passes agree.
  * With the inner residual ON -- the shipped architecture -- the projection is
    applied to (identity + attention + feed-forward), not to the feed-forward
    output, so no fold exists. We show the same substitution changes the output.

Nothing is trained here; this is algebra checked against the real forward pass.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import torch
import torch.nn as nn

RERUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RERUN))
sys.path.insert(0, str(RERUN.parent / "imputer/ranking"))
from analysis.split_missing_metrics import chunk_by_item          # noqa: E402
from imputer.entity_mf.test import _reconstruct                   # noqa: E402
from imputer.entity_mf.data import variable_list_to_entity_graph  # noqa: E402

RUN = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"


def fold(model: nn.Module) -> nn.Module:
    """Push each block's trailing projection back into its feed-forward.

    For one block, with M = [proj_out ; W_param] stacked into a single d x d map
    and the feed-forward ending in y = W_out z + b_out:

        M (W_out z + b_out) = (M W_out) z + (M b_out + b_M)

    so the last feed-forward linear absorbs M exactly, and M becomes the
    identity split back into the two slices.
    """
    m = copy.deepcopy(model)
    F, P, D = m.feature_dim, m.param_dim, m.model_dim
    for block in m.blocks:
        W = torch.cat([block["proj_out"].weight.data,
                       block["W_param"].weight.data], dim=0)          # [D, D]
        b = torch.cat([block["proj_out"].bias.data,
                       block["W_param"].bias.data], dim=0)            # [D]
        last = block["ff"].net[-1]
        last.weight.data = W @ last.weight.data
        last.bias.data   = W @ last.bias.data + b
        eye = torch.eye(D, device=W.device, dtype=W.dtype)
        block["proj_out"].weight.data = eye[:F].clone()
        block["proj_out"].bias.data.zero_()
        block["W_param"].weight.data = eye[F:F + P].clone()
        block["W_param"].bias.data.zero_()
    return m


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, eval_vars, _ = _reconstruct(RUN)
    model.to(dev).eval()
    chunk = chunk_by_item(eval_vars, 300)[0]
    graph = variable_list_to_entity_graph(chunk, model.types)

    out = {}
    for inner in ("none", "both"):
        m = copy.deepcopy(model)
        m.inner_residual = inner
        m.to(dev).eval()
        folded = fold(m).to(dev).eval()
        with torch.no_grad():
            a = m(graph, device=dev)
            b = folded(graph, device=dev)
        diff = float((a - b).abs().max())
        scale = float(a.abs().max())
        out[inner] = {"max_abs_diff": diff, "output_scale": scale,
                      "relative": diff / scale}
        print(f"inner_residual={inner:<5}  max|A-B| = {diff:.3e}   "
              f"(output scale {scale:.3f}, relative {diff/scale:.3e})")

    verdict = ("fold is exact with the inner residual off, and is not available "
               "with it on" if out["none"]["relative"] < 1e-5
               < out["both"]["relative"] else "UNEXPECTED -- inspect")
    out["verdict"] = verdict
    print("\nverdict:", verdict)
    p = RERUN / "threads/block-projection/out/c0_fold.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
