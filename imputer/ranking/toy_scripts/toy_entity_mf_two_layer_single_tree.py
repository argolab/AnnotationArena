from __future__ import annotations

"""
Toy overfitting experiment using the real EntityMarformer on a single depth-1 tree.

Differences from toy_rel_attention_two_layer:
- Uses the actual EntityMarformer implementation (1 head, 2 layers).
- Uses SyntheticRegressionType for a single node type with MSE loss.
- Uses per-entity variation (deviation embeddings) to break node symmetry.

Run from repo root:

  python toy_entity_mf_two_layer_single_tree.py

This should overfit to targets [4,1,1,1] if optimization is healthy.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import EntityGraph, Relationship, Token
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.synthetic.types import RegressionSlices, SyntheticRegressionType
from imputer.entity_mf.types import EntityType, VariationConfig


@dataclass
class SingleTreeGraph:
    graph: EntityGraph
    types: Dict[str, EntityType]


def build_single_tree_graph(device: torch.device) -> SingleTreeGraph:
    """
    Build a tiny depth-1 tree as an EntityGraph:
      - 4 nodes: 0=root, 1..3 children
      - One type 'tree_node' with SyntheticRegressionType (input_dim=1, output_dim=1)
      - Per-node variation enabled via entity_id (breaks symmetry)
    """
    N = 4
    root = 0
    children = [1, 2, 3]

    # Relationships: P2C and C2P for completeness
    relationships = [
        Relationship(name="P2C", source_type="tree_node", target_type="tree_node", inverse="C2P"),
        Relationship(name="C2P", source_type="tree_node", target_type="tree_node", inverse="P2C"),
    ]

    # Types: single node type with param layout [input_dim=1, output_dim=1].
    # Enable variation so each node can have its own fixed embedding, but set
    # reg_weight=0.0 and freeze deviations later (symmetry break at embedding
    # level without learning the deviations).
    slices = RegressionSlices(input_dim=1, output_dim=1)
    var_cfg = VariationConfig(enabled=False, num_entities=N, reg_weight=0.0)
    tree_type = SyntheticRegressionType(name="tree_node", slices=slices, has_target=True, variation=var_cfg)
    types: Dict[str, EntityType] = {"tree_node": tree_type}

    # Tokens: each node has input_value=[1.0], target_value=subtree size
    # subtree sizes for depth-1: [4,1,1,1]
    subtree_sizes = [4.0, 1.0, 1.0, 1.0]
    input_values = [1.0, 1.0, 1.0, 1.0]
    tokens: List[Token] = []
    for i in range(N):
        raw = {
            "input_value": [input_values[i]],
            "target_value": [subtree_sizes[i]],
        }
        tokens.append(Token(type_name="tree_node", entity_id=i, status=2, raw_data=raw))

    # Edges: P2C and C2P
    edges: List[Tuple[int, int, str]] = []
    for c in children:
        edges.append((root, c, "P2C"))
        edges.append((c, root, "C2P"))

    graph = EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)
    return SingleTreeGraph(graph=graph, types=types)


def _compute_graph_mse(model: EntityMarformer, tree: SingleTreeGraph, device: torch.device) -> torch.Tensor:
    """
    Compute MSE between predicted param slice and target_value for 'tree_node' tokens.
    """
    graph = tree.graph
    params = model(graph, device=device)  # [1, L, P]

    # Extract the output slice for SyntheticRegressionType
    node_type = tree.types["tree_node"]
    assert isinstance(node_type, SyntheticRegressionType)
    out_slice = node_type.slices.output_slice()

    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    for idx, tok in enumerate(graph.tokens):
        raw = tok.raw_data or {}
        tgt = raw.get("target_value", None)
        if tgt is None:
            continue
        p = params[0, idx, out_slice]  # [output_dim]
        preds.append(p)
        tgts.append(torch.tensor(tgt, device=device, dtype=p.dtype))

    if not preds:
        return torch.zeros((), device=device)

    pred = torch.stack(preds, dim=0)  # [N,1]
    tgt = torch.stack(tgts, dim=0)    # [N,1]
    return F.mse_loss(pred, tgt, reduction="mean")


def run_entity_mf_single_tree(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tree = build_single_tree_graph(device=device)

    # Config: 2 layers, 1 head, small embedding dimension.
    cfg = EntityMarformerConfig(
        embedding_dim=8,
        num_layers=2,
        attention_heads=1,
        dropout=0.0,
        d_ff=16,
        num_ffn_layers=1,
    )

    model = EntityMarformer(
        config=cfg,
        types=tree.types,
        num_relationships=len(tree.graph.relationships),
    ).to(device)

    # # Symmetry break at embedding level only: random, fixed deviations per node.
    # dev = model.deviation_tables.get("tree_node", None)
    # if dev is not None:
    #     with torch.no_grad():
    #         dev.normal_(mean=0.0, std=0.1)
    #     dev.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    print("Starting EntityMarformer single-tree overfit experiment (fixed deviations)...")
    for step in range(10000):
        opt.zero_grad()
        mse_loss = _compute_graph_mse(model, tree, device=device)
        loss = mse_loss
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            with torch.no_grad():
                graph = tree.graph
                params = model(graph, device=device)
                node_type = tree.types["tree_node"]
                assert isinstance(node_type, SyntheticRegressionType)
                out_slice = node_type.slices.output_slice()

                preds = []
                for idx, tok in enumerate(graph.tokens):
                    p = params[0, idx, out_slice]
                    preds.append(float(p.item()))
                print(f"step {step+1:4d} | mse={mse_loss.item():.6f}")
                print("  preds:", [round(v, 4) for v in preds])

    with torch.no_grad():
        graph = tree.graph
        params = model(graph, device=device)
        node_type = tree.types["tree_node"]
        assert isinstance(node_type, SyntheticRegressionType)
        out_slice = node_type.slices.output_slice()

        preds = [float(params[0, idx, out_slice].item()) for idx in range(len(graph.tokens))]
        print("\nFinal predictions:", [round(v, 4) for v in preds])
        print("Targets:", [tok.raw_data["target_value"][0] for tok in graph.tokens])

        # Inspect learned feature embeddings (type base + deviation) per node.
        print("\nLearned feature embeddings for 'tree_node':")
        base = model.type_embeddings["tree_node"].detach().cpu()[0]  # [D_feat]
        dev_table = model.deviation_tables.get("tree_node", None)
        if dev_table is None:
            print("  No deviation table for 'tree_node' (variation disabled).")
        else:
            dev_cpu = dev_table.detach().cpu()
            for idx, tok in enumerate(graph.tokens):
                ent_id = tok.entity_id
                dev = dev_cpu[ent_id]
                feat = base + dev
                print(f"  node {idx} (entity_id={ent_id}):")
                print(f"    base: {base.tolist()}")
                print(f"    dev : {dev.tolist()}")
                print(f"    feat: {feat.tolist()}")


if __name__ == "__main__":
    run_entity_mf_single_tree()

