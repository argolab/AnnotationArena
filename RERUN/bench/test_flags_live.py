"""Check that each model flag actually changes the forward pass.

The configuration matrix in regression.py only proves a flag does not *break*
training. Several flags moved the 15-epoch loss by less than 1e-3, which is
also what an inert flag looks like -- a path that silently does nothing would
pass the matrix while covering nothing. This asserts each flag changes the
output of a forward pass on identical weights and inputs.

`use_rel_value` is initialised to zeros deliberately (a no-op at step 0), so its
parameter is randomised first; the check is that it is wired, not that it is
non-trivial at init.

  python -m RERUN.bench.test_flags_live --data-dir <split>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.train import load_bundle_and_converter
from imputer.entity_mf.types import build_default_domain3_types

FLAGS = [
    ("use_per_head_rel", True),
    ("use_rel_value", True),
    ("use_addone_attn", True),
    ("use_graph_mask", True),
    ("use_deviation_norm", True),
    ("use_param_output_head", True),
    ("use_pointer", False),          # default is on in the runners; turn it off
    ("scale_shared_rel", False),
]


def build(types, graph, device, **over):
    cfg = EntityMarformerConfig()
    cfg.embedding_dim, cfg.num_layers, cfg.attention_heads = 80, 8, 4
    cfg.d_ff, cfg.num_ffn_layers, cfg.dropout = 128, 1, 0.1
    cfg.use_per_head_rel, cfg.scale_shared_rel, cfg.use_pointer = False, True, True
    for k, v in over.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    m = EntityMarformer(cfg, types, graph.num_relationships).to(device).eval()
    with torch.no_grad():
        for t in m.deviation_tables.values():
            t.normal_(0.0, 0.1)
        for blk in m.blocks:
            if "attn" in blk and hasattr(blk["attn"], "value_rel"):
                blk["attn"].value_rel.normal_(0.0, 0.1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-vars", type=int, default=3000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle, converter, sizes = load_bundle_and_converter(Path(args.data_dir))
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"], num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"], num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=20.0, item_dropout_rate=1.0, annotator_dropout_rate=0.0)
    variables = (converter.create_variables_from_bundle(bundle, partition="train", status="observed")
                 + converter.create_variables_from_bundle(bundle, partition="train", status="missing")
                 )[: args.max_vars]
    graph = variable_list_to_entity_graph(variables, types)

    with torch.no_grad():
        base = build(types, graph, device)(graph, device=device)

    print(f"device={device}  L={graph.num_tokens}\n")
    print(f"{'flag':<26}{'max|delta| vs default':>24}  verdict")
    print("-" * 60)
    failures = []
    for name, value in FLAGS:
        with torch.no_grad():
            out = build(types, graph, device, **{name: value})(graph, device=device)
        if out.shape != base.shape:
            print(f"{name:<26}{'shape change':>24}  live")
            continue
        d = (out - base).abs().max().item()
        live = d > 1e-6
        print(f"{name}={value}".ljust(26) + f"{d:>24.3e}  {'live' if live else 'INERT'}")
        if not live:
            failures.append(name)

    print()
    if failures:
        raise SystemExit("flags with no effect on the forward pass: " + ", ".join(failures))
    print("every flag changes the forward pass")


if __name__ == "__main__":
    main()
