#!/usr/bin/env python
"""Does the standard block compose with the typed codec?

The two were developed on separate branches and never run together. This
asserts the combination the final run uses -- value_codec=typed, codec_init=unit,
block_output_projection=none, dual_outer_residual=none -- constructs, forwards
and backwards, and that the shipped baseline still does too.

Also AST-checks train.py's argparse: the B9 temperature purge left an
add_argument() with no option string and argparse raised at import, which the
equivalence test never caught because it builds models directly.
"""
from __future__ import annotations
import ast, os, sys
from pathlib import Path
import torch

RANKING_ROOT = Path(os.environ["RANKING_ROOT"]).resolve()
RERUN = RANKING_ROOT.parent.parent / "RERUN"
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(RANKING_ROOT))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.test import _reconstruct
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.model import EntityMarformer

RUN = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"

def ast_check(path: Path) -> None:
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            pos = [a for a in node.args if isinstance(a, ast.Constant)
                   and isinstance(a.value, str)]
            if not pos and not any(k.arg == "dest" for k in node.keywords):
                bad.append(node.lineno)
    assert not bad, f"add_argument with no option string at lines {bad}"
    print(f"ok  argparse: every add_argument in {path.name} has an option string")

def run_one(label, cfg_edits, types, n_rel, chunk, expect_proj):
    model, _, _ = _reconstruct(RUN)
    cfg = model.config if hasattr(model, "config") else None
    # Rebuild from a fresh config so the flags take effect at construction.
    import copy
    from imputer.entity_mf.config import EntityMarformerConfig
    c = EntityMarformerConfig()
    base, _, train_cfg = _reconstruct(RUN)
    for k in ("embedding_dim","num_layers","attention_heads","d_ff","num_ffn_layers",
              "dropout","use_per_head_rel","use_pointer","use_rel_value",
              "use_addone_attn","type_embedding_init","use_deviation_norm",
              "scale_shared_rel","use_graph_mask"):
        if hasattr(base, k):
            setattr(c, k, getattr(base, k))
    for k, v in cfg_edits.items():
        setattr(c, k, v)
    m = EntityMarformer(c, types, num_relationships=n_rel)
    has_proj = any("proj_out" in dict(b.named_children()) for b in m.blocks)
    assert has_proj == expect_proj, f"{label}: proj_out present={has_proj}, expected {expect_proj}"
    graph = variable_list_to_entity_graph(chunk, m.types)
    out = m(graph, device=torch.device("cpu"))
    loss = out.float().pow(2).mean()
    loss.backward()
    gnorm = sum((p.grad.detach().float().pow(2).sum() for p in m.parameters()
                 if p.grad is not None)) ** 0.5
    nparam = sum(p.numel() for p in m.parameters())
    assert torch.isfinite(out).all(), f"{label}: non-finite forward"
    assert torch.isfinite(torch.as_tensor(gnorm)), f"{label}: non-finite grad"
    print(f"ok  {label:<34} params {nparam:>9,}  proj_out={has_proj}  "
          f"out {tuple(out.shape)}  grad_norm {float(gnorm):.3e}")
    return nparam

def main():
    ast_check(RANKING_ROOT / "imputer/entity_mf/train.py")
    base, eval_vars, _ = _reconstruct(RUN)
    types = base.types
    # The attention block carries the real count; guessing it silently builds a
    # model whose relational bias indexes out of bounds on the first forward.
    n_rel = base.blocks[0]["attn"].num_relationships
    chunk = chunk_by_item(eval_vars, 60)[0]
    print(f"    graph chunk: {len(chunk)} variables, num_relationships={n_rel}\n")
    shipped = run_one("shipped (dual + full proj)",
                      dict(value_codec="dual", block_output_projection="full",
                           dual_outer_residual="all"),
                      types, n_rel, chunk, expect_proj=True)
    final = run_one("FINAL (typed/unit + std block)",
                    dict(value_codec="typed", codec_init="unit",
                         block_output_projection="none", dual_outer_residual="none"),
                    types, n_rel, chunk, expect_proj=False)
    print(f"\nok  the standard block drops {shipped - final:,} parameters "
          f"({100*(shipped-final)/shipped:.1f}%)")
    # The guard must still refuse the combination that forced the C1 retraction.
    try:
        run_one("broken (none proj + outer all)",
                dict(value_codec="dual", block_output_projection="none",
                     dual_outer_residual="all"), types, n_rel, chunk, expect_proj=False)
    except ValueError as e:
        assert "doubles" in str(e), e
        print("ok  guard still refuses block_output_projection=none with outer_residual=all")
    else:
        raise AssertionError("the C1 guard is GONE -- the broken arm constructed")
    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()
