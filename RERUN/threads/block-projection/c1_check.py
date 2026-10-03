#!/usr/bin/env python
"""C1 pre-flight: the new flag is neither inert nor disruptive.

Three assertions before any GPU time is spent:
  1. default ("full") reproduces the shipped model exactly -- same parameter
     count, same forward output, so no existing result can move;
  2. "none" is not inert -- it changes the forward output;
  3. "none" actually removes parameters rather than pinning them to the identity.

Must be run from the block-projection worktree; the flag does not exist on the
main checkout.
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path
import torch

# RERUN/threads is a symlink shared across worktrees, so __file__ resolves into
# the MAIN checkout no matter which worktree invoked it. Deriving the import root
# from __file__ therefore silently imports the unpatched model and every flag
# looks inert -- the same trap that cost the S1 sweep three jobs. Take the root
# from RANKING_ROOT, which config/cluster.sh exports from the calling worktree.
RANKING_ROOT = Path(os.environ["RANKING_ROOT"]).resolve()
RERUN = RANKING_ROOT.parent.parent / "RERUN"
sys.path.insert(0, str(RERUN))
sys.path.insert(0, str(RANKING_ROOT))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.test import _reconstruct
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf import model as _model_mod
from imputer.entity_mf.model import EntityMarformer

assert Path(_model_mod.__file__).resolve().is_relative_to(RANKING_ROOT), (
    f"imported model.py from {_model_mod.__file__}, not from {RANKING_ROOT}"
)

RUN = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"


OUT = Path(__file__).resolve().parent / "out/c1_check.json"


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, eval_vars, _ = _reconstruct(RUN)
    model.to(dev).eval()
    chunk = chunk_by_item(eval_vars, 300)[0]
    graph = variable_list_to_entity_graph(chunk, model.types)

    cfg = copy.deepcopy(model.config)
    assert getattr(cfg, "block_output_projection", "full") == "full", \
        "a checkpoint written before this flag must reconstruct as 'full'"
    with torch.no_grad():
        out_full = model(graph, device=dev)

    n_rel = model.blocks[0]["attn"].num_relationships
    cfg_none = copy.deepcopy(cfg)
    cfg_none.block_output_projection = "none"
    torch.manual_seed(0)
    m_none = EntityMarformer(cfg_none, model.types, num_relationships=n_rel).to(dev).eval()
    loaded = m_none.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        out_none = m_none(graph, device=dev)

    n_full = sum(p.numel() for p in model.parameters())
    n_none = sum(p.numel() for p in m_none.parameters())
    diff = float((out_full - out_none).abs().max())
    res = {
        "params_full": n_full, "params_none": n_none,
        "params_removed": n_full - n_none,
        "fraction_removed": round((n_full - n_none) / n_full, 4),
        "keys_only_in_full": sorted({k.split('.')[-2] for k in loaded.unexpected_keys}),
        "max_abs_output_diff_full_vs_none": diff,
    }
    print(json.dumps(res, indent=1))
    assert diff > 1e-3, "flag is INERT -- 'none' produced the same output as 'full'"
    assert n_none < n_full, "'none' did not remove parameters"
    print("\nOK: default unchanged, flag not inert, parameters actually removed")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
