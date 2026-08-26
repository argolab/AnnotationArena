"""Kernel-level profile of one training step, via torch.profiler.

Builds the real chunk graph the trainer would use and runs forward+backward on
it a few times, so the ranking below is what the GPU actually spends time on
once the Python marshalling is out of the way.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.eval import compute_trainable_loss
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.train import load_bundle_and_converter
from imputer.entity_mf.types import build_default_domain3_types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-item", type=int, default=100)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda")
    bundle, converter, sizes = load_bundle_and_converter(Path(args.data_dir))
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"], num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"], num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=20.0, item_dropout_rate=1.0, annotator_dropout_rate=0.0)

    variables = []
    for part in ("train", "val", "test"):
        for status in ("observed", "missing"):
            variables += converter.create_variables_from_bundle(bundle, partition=part, status=status)
    item_ids = sorted({i for v in variables for i in v.item_ids})[: args.max_item]
    chunk = [v for v in variables if all(i in set(item_ids) for i in v.item_ids)]
    graph = variable_list_to_entity_graph(chunk, types)

    cfg = EntityMarformerConfig()
    cfg.embedding_dim, cfg.num_layers, cfg.attention_heads = 80, 8, 4
    cfg.d_ff, cfg.num_ffn_layers, cfg.dropout = 128, 1, 0.1
    cfg.use_per_head_rel, cfg.scale_shared_rel, cfg.use_pointer = False, True, True
    model = EntityMarformer(cfg, types, graph.num_relationships).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    def step():
        opt.zero_grad(set_to_none=True)
        params = model(graph, device=device)
        loss, _, _ = compute_trainable_loss(params, graph, model.types,
                                            model.global_param_dim, device,
                                            masked_loss_weight=15.0, observed_loss_weight=1.0)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()

    print(f"L={graph.num_tokens}  chunk_vars={len(chunk)}  iters={args.iters}")
    print(f"peak cuda mem: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.rows))


if __name__ == "__main__":
    main()
