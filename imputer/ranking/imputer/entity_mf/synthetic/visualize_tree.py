from __future__ import annotations

"""
Minimal visualization for synthetic tree tasks.

This script generates a single tree/forest instance using TreeTaskConfig and
pretty-prints the structure. For each node it shows:
  - its index
  - its input_value
  - its target_value

Usage example:

  python -m imputer.entity_mf.synthetic.visualize_tree \\
    --tree-depth 3 --tree-width 3 --num-trees 1 --seed 0
"""

import argparse
from typing import Dict, List, Tuple

from .datagen_tree import TreeTaskConfig, generate_tree_graph


def _build_parent_children(edges: List[Tuple[int, int, str]]) -> Tuple[List[int], List[List[int]]]:
    """
    Recover parent/children structure from P2C edges.
    """
    if not edges:
        return [], []

    max_idx = 0
    for u, v, _name in edges:
        max_idx = max(max_idx, u, v)
    n = max_idx + 1

    parents = [-1 for _ in range(n)]
    children: List[List[int]] = [[] for _ in range(n)]

    for u, v, name in edges:
        if name == "P2C":
            parents[v] = u
            children[u].append(v)

    return parents, children


def _print_tree(graph, parents: List[int], children: List[List[int]]) -> None:
    """
    Depth-first pretty print of each tree in the forest.
    """
    n = len(graph.tokens)
    roots = [i for i in range(n) if parents[i] < 0]

    def fmt_token(idx: int) -> str:
        tok = graph.tokens[idx]
        raw = tok.raw_data or {}
        inp = raw.get("input_value", [])
        tgt = raw.get("target_value", [])
        return f"{idx}: input={inp}, target={tgt}"

    def dfs(u: int, indent: int) -> None:
        print("  " * indent + fmt_token(u))
        for v in children[u]:
            dfs(v, indent + 1)

    for r in roots:
        print(f"Tree rooted at {r}:")
        dfs(r, indent=0)
        print()


def main() -> None:
    parser = argparse.ArgumentParser("Visualize a synthetic tree/forest instance.")
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--tree-width", type=int, default=3)
    parser.add_argument("--num-trees", type=int, default=1)
    parser.add_argument("--randomize-tree", action="store_true")
    parser.add_argument("--aggregate", type=str, choices=["count", "sum"], default="count")
    parser.add_argument("--leaf-only", action="store_true")
    parser.add_argument("--empty-param", action="store_true")
    parser.add_argument("--param-dim", type=int, default=1)
    parser.add_argument("--edge-direction", type=str, choices=["both", "p2c", "c2p"], default="both")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    cfg = TreeTaskConfig(
        tree_depth=int(args.tree_depth),
        tree_width=int(args.tree_width),
        num_trees=int(args.num_trees),
        randomize_tree=bool(args.randomize_tree),
        aggregate=args.aggregate,  # type: ignore[arg-type]
        leaf_only=bool(args.leaf_only),
        empty_param=bool(args.empty_param),
        param_dim=int(args.param_dim),
        edge_direction=args.edge_direction,  # type: ignore[arg-type]
    )

    graph = generate_tree_graph(cfg, seed=int(args.seed))
    parents, children = _build_parent_children(graph.edges)

    print("Synthetic tree/forest instance")
    print(f"  num_tokens = {len(graph.tokens)}")
    print(f"  num_edges  = {len(graph.edges)}")
    print(f"  aggregate  = {cfg.aggregate}, leaf_only={cfg.leaf_only}, "
          f"empty_param={cfg.empty_param}, param_dim={cfg.param_dim}")
    print()

    _print_tree(graph, parents, children)


if __name__ == "__main__":
    main()

