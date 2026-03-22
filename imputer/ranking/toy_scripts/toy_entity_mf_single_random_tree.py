from __future__ import annotations

"""
Toy overfitting experiment using EntityMarformer on a single random tree
with variable depth and branching factor (checkpoint-1 style).

Differences from toy_entity_mf_two_layer_single_tree.py:
- The tree structure is randomized:
  * Root at node 0.
  * Each node samples a random number of children up to max_branch.
  * Depth is at most max_depth, but actual subtree depths vary.
- Targets are subtree sizes (inclusive).

We use:
- A single SyntheticRegressionType "tree_node" with (input_dim=1, output_dim=1).
- VariationConfig(enabled=True, reg_weight=10.0) so there is a regularized
  per-entity deviation embedding (symmetry-breaking at embedding level).
- EntityMarformer with 2 layers, 1 head, small embedding dimension.

Run from repo root:

  python toy_entity_mf_single_random_tree.py
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import EntityGraph, Relationship, Token
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.synthetic.types import RegressionSlices, SyntheticRegressionType
from imputer.entity_mf.types import EntityType, VariationConfig


@dataclass
class RandomTreeGraph:
    graph: EntityGraph
    types: Dict[str, EntityType]


def build_random_tree(
    device: torch.device,
    rng: random.Random,
    max_nodes: int = 20,
    max_depth: int = 3,
    max_branch: int = 4,
) -> RandomTreeGraph:
    """
    Build a random rooted tree as an EntityGraph:
      - Node 0 is the root.
      - Each node at depth < max_depth samples a random number of children
        in [0, max_branch], subject to a global node budget <= max_nodes.
      - Tree may be unbalanced; subtree depths vary; branching factors vary.
    Targets are subtree sizes (inclusive) but that is handled in training.
    """
    root = 0
    parents: List[int] = [-1]  # parents[i] = parent index of node i, -1 for root
    depths: List[int] = [0]
    frontier: List[int] = [root]
    next_id = 1

    while frontier and next_id < max_nodes:
        i = frontier.pop(0)
        depth_i = depths[i]
        if depth_i >= max_depth:
            continue
        # Sample how many children, but ensure we don't exceed max_nodes.
        remaining = max_nodes - next_id
        if remaining <= 0:
            break
        # For the root, force at least 1 child so the tree is non-trivial.
        upper = min(max_branch, remaining)
        if i == root and depth_i == 0:
            num_children = rng.randint(1, upper)
        else:
            num_children = rng.randint(0, upper)
        for _ in range(num_children):
            parents.append(i)
            depths.append(depth_i + 1)
            frontier.append(next_id)
            next_id += 1
            if next_id >= max_nodes:
                break

    N = len(parents)

    # Relationships: P2C and C2P for tree_node
    relationships = [
        Relationship(name="P2C", source_type="tree_node", target_type="tree_node", inverse="C2P"),
        Relationship(name="C2P", source_type="tree_node", target_type="tree_node", inverse="P2C"),
    ]

    # Compute children lists
    children: List[List[int]] = [[] for _ in range(N)]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    # Subtree sizes (inclusive) via post-order DFS
    subtree_sizes = [0.0 for _ in range(N)]

    def dfs(u: int) -> float:
        total = 1.0
        for v in children[u]:
            total += dfs(v)
        subtree_sizes[u] = total
        return total

    dfs(root)

    # Types: single node type with param layout [input_dim=1, output_dim=1]
    slices = RegressionSlices(input_dim=1, output_dim=1)
    var_cfg = VariationConfig(enabled=False, num_entities=N, reg_weight=10.0)
    tree_type = SyntheticRegressionType(name="tree_node", slices=slices, has_target=True, variation=var_cfg)
    types: Dict[str, EntityType] = {"tree_node": tree_type}

    # Tokens: each node has input_value=[1.0], target_value=subtree size
    tokens: List[Token] = []
    for i in range(N):
        raw = {
            "input_value": [1.0],
            "target_value": [subtree_sizes[i]],
        }
        tokens.append(Token(type_name="tree_node", entity_id=i, status=2, raw_data=raw))

    # Edges: P2C and C2P
    edges: List[Tuple[int, int, str]] = []
    for j, p in enumerate(parents):
        if p < 0:
            continue
        edges.append((p, j, "P2C"))
        edges.append((j, p, "C2P"))

    graph = EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)
    return RandomTreeGraph(graph=graph, types=types)


def _compute_graph_mse(model: EntityMarformer, tree: RandomTreeGraph, device: torch.device) -> torch.Tensor:
    """
    Compute MSE between predicted param slice and target_value for 'tree_node' tokens.
    """
    graph = tree.graph
    params = model(graph, device=device)  # [1, L, P]

    node_type = tree.types["tree_node"]
    assert isinstance(node_type, SyntheticRegressionType)
    out_slice = node_type.slices.output_slice()

    preds = []
    tgts = []
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


def run_entity_mf_single_random_tree(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(122)
    tree = build_random_tree(device=device, rng=rng)

    # Config: 2 layers, 1 head, small embedding dimension.
    cfg = EntityMarformerConfig(
        embedding_dim=2,
        num_layers=2,
        attention_heads=1,
        dropout=0.0,
        d_ff=32,
        num_ffn_layers=1,
    )

    model = EntityMarformer(
        config=cfg,
        types=tree.types,
        num_relationships=len(tree.graph.relationships),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("Starting EntityMarformer single-random-tree overfit experiment...")
    for step in range(3000):
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
                tgts = []
                for idx, tok in enumerate(graph.tokens):
                    p = params[0, idx, out_slice]
                    preds.append(float(p.item()))
                    tgts.append(float(tok.raw_data["target_value"][0]))
                print(f"step {step+1:4d} | mse={mse_loss.item():.6f}")
                print("  preds:", [round(v, 4) for v in preds])
                print("  tgts :", [round(v, 4) for v in tgts])

    with torch.no_grad():
        graph = tree.graph
        params = model(graph, device=device)
        node_type = tree.types["tree_node"]
        assert isinstance(node_type, SyntheticRegressionType)
        out_slice = node_type.slices.output_slice()

        preds = [float(params[0, idx, out_slice].item()) for idx in range(len(graph.tokens))]
        print("\nFinal predictions:", [round(v, 4) for v in preds])
        print("Targets:", [float(tok.raw_data["target_value"][0]) for tok in graph.tokens])


if __name__ == "__main__":
    run_entity_mf_single_random_tree()

