from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import torch

from imputer.entity_mf.data import EntityGraph, Relationship, Token
from imputer.entity_mf.synthetic.types import RegressionSlices, SyntheticRegressionType
from imputer.entity_mf.types import EntityType


AggregateMode = Literal["count", "sum"]
EdgeDirection = Literal["both", "p2c", "c2p"]


@dataclass(frozen=True)
class TreeTaskConfig:
    tree_depth: int = 3
    # Target branching factor per internal node when randomize_tree is False.
    # When randomize_tree is True this is an upper bound on sampled child count.
    tree_width: int = 3
    num_trees: int = 1
    # If True, use stochastic branching: for each node at level < tree_depth,
    # sample num_children ~ Uniform{0..tree_width}. A safeguard ensures at
    # least one child is created at each level until max depth so the tree
    # does not die out early.
    randomize_tree: bool = False
    # What quantity to aggregate over each subtree:
    #   - "count": scalar count of nodes (or leaves when leaf_only=True).
    #   - "sum":   vector sum of per-node inputs (dim = param_dim).
    aggregate: AggregateMode = "count"
    # If True, only leaf nodes contribute to the target:
    #   - count: target = number of leaves in the subtree.
    #   - sum:   target = sum of leaf input vectors only.
    leaf_only: bool = False
    # If True with aggregate="count", the param stream input is always zero
    # and the model must infer counts from tree structure alone.
    empty_param: bool = False
    # Dimensionality of per-node input / output vectors for aggregate="sum".
    param_dim: int = 1
    # Which directed edges to materialize in the graph:
    #   - "both": parent->child (P2C) and child->parent (C2P).
    #   - "p2c":  only parent->child edges.
    #   - "c2p":  only child->parent edges.
    edge_direction: EdgeDirection = "both"
    # If True, randomly permute node indices after building the canonical tree/forest so that
    # token position is not a fixed function of depth/role. Targets/inputs move with nodes;
    # edges are relabeled. Different seeds -> different permutations (reduces position memoization).
    shuffle_nodes: bool = True


def _build_relationships() -> List[Relationship]:
    # Always declare both directions so num_relationships is stable across ablations.
    return [
        Relationship(name="P2C", source_type="tree_node", target_type="tree_node", inverse="C2P"),
        Relationship(name="C2P", source_type="tree_node", target_type="tree_node", inverse="P2C"),
    ]


def _sample_tree_parents(
    rng: random.Random,
    depth: int,
    width: int,
    randomize: bool,
) -> List[int]:
    """
    Build a rooted tree as a parent array.

    Returns:
      parents: list of length N, with parents[0] = -1 for root.

    The tree is generated level-by-level up to max depth.
    - If randomize=False: full W-ary tree with exact depth (all internal nodes have W children).
    - If randomize=True: each node at level < depth samples num_children ~ Uniform{0..W}.
      We still ensure at least one node exists at each level until depth by forcing
      at least one child somewhere if a whole level would be empty.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if width < 0:
        raise ValueError("width must be >= 0")

    parents: List[int] = [-1]  # root
    current_level: List[int] = [0]
    next_node_id = 1

    for level in range(depth):
        next_level: List[int] = []
        if not current_level:
            break

        if not randomize:
            for u in current_level:
                for _ in range(width):
                    parents.append(u)
                    next_level.append(next_node_id)
                    next_node_id += 1
        else:
            # Random branching
            for u in current_level:
                num_children = rng.randint(0, width) if width > 0 else 0
                for _ in range(num_children):
                    parents.append(u)
                    next_level.append(next_node_id)
                    next_node_id += 1

            # Ensure progress: if we would terminate early but still have remaining depth,
            # force one random parent to have a child.
            if not next_level and width > 0:
                u = rng.choice(current_level)
                parents.append(u)
                next_level.append(next_node_id)
                next_node_id += 1

        current_level = next_level

    return parents


def _parents_to_children(parents: Sequence[int]) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for v, p in enumerate(parents):
        if p >= 0:
            children[p].append(v)
    return children


def _postorder(children: Sequence[Sequence[int]], root: int = 0) -> List[int]:
    order: List[int] = []

    def dfs(u: int) -> None:
        for v in children[u]:
            dfs(v)
        order.append(u)

    dfs(root)
    return order


def _tree_roots(num_nodes: int, parents: Sequence[int]) -> List[int]:
    return [i for i in range(num_nodes) if parents[i] < 0]


def _compute_subtree_targets_count(
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    leaf_only: bool,
) -> List[float]:
    n = len(parents)
    targets = [0.0 for _ in range(n)]
    roots = _tree_roots(n, parents)

    for r in roots:
        order = _postorder(children, root=r)
        for u in order:
            if leaf_only:
                own = 1.0 if len(children[u]) == 0 else 0.0
            else:
                own = 1.0
            s = own
            for v in children[u]:
                s += targets[v]
            targets[u] = s

    return targets


def _compute_subtree_targets_sum(
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    values: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    n = len(parents)
    targets: List[torch.Tensor] = [torch.zeros_like(values[0]) for _ in range(n)]
    roots = _tree_roots(n, parents)

    for r in roots:
        order = _postorder(children, root=r)
        for u in order:
            s = values[u].clone()
            for v in children[u]:
                s = s + targets[v]
            targets[u] = s

    return targets


def _make_values(
    rng: random.Random,
    n: int,
    dim: int,
    leaf_only: bool,
    children: Sequence[Sequence[int]],
) -> List[torch.Tensor]:
    values: List[torch.Tensor] = []
    for u in range(n):
        if leaf_only and len(children[u]) > 0:
            values.append(torch.zeros(dim))
        else:
            # Use Python RNG for reproducibility independent of torch global seed.
            vals = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            values.append(torch.tensor(vals, dtype=torch.float32))
    return values


def _shuffle_tokens_and_edges(
    rng: random.Random,
    n: int,
    global_parents: List[int],
    input_values: List[List[float]],
    target_values: List[List[float]],
    edge_direction: EdgeDirection,
) -> Tuple[List[Token], List[Tuple[int, int, str]]]:
    """
    Apply a random permutation to node indices: perm[new_idx] = old_idx.
    Inputs/targets follow canonical nodes; edges are relabeled consistently.
    """
    perm = list(range(n))
    rng.shuffle(perm)
    inv_perm = [0] * n
    for new_idx in range(n):
        inv_perm[perm[new_idx]] = new_idx

    tokens: List[Token] = []
    for new_idx in range(n):
        old_idx = perm[new_idx]
        raw = {
            "input_value": list(input_values[old_idx]),
            "target_value": list(target_values[old_idx]),
        }
        tokens.append(Token(type_name="tree_node", entity_id=new_idx, status=2, raw_data=raw))

    edges: List[Tuple[int, int, str]] = []
    for v_old, p_old in enumerate(global_parents):
        if p_old < 0:
            continue
        p_new = inv_perm[p_old]
        v_new = inv_perm[v_old]
        if edge_direction in ("both", "p2c"):
            edges.append((p_new, v_new, "P2C"))
        if edge_direction in ("both", "c2p"):
            edges.append((v_new, p_new, "C2P"))

    return tokens, edges


def build_tree_task_types(cfg: TreeTaskConfig) -> Dict[str, EntityType]:
    if cfg.aggregate == "count":
        if cfg.empty_param:
            slices = RegressionSlices(input_dim=0, output_dim=1)
        else:
            slices = RegressionSlices(input_dim=1, output_dim=1)
    else:
        if cfg.param_dim <= 0:
            raise ValueError("param_dim must be > 0 for sum aggregation")
        slices = RegressionSlices(input_dim=cfg.param_dim, output_dim=cfg.param_dim)

    return {"tree_node": SyntheticRegressionType(name="tree_node", slices=slices, has_target=True)}


def generate_tree_graph(cfg: TreeTaskConfig, seed: int) -> EntityGraph:
    """
    Generate one synthetic instance (one graph) for the tree/forest task.

    When ``cfg.shuffle_nodes`` is True (default), node indices are randomly permuted so
    token order does not align with a fixed BFS role; targets travel with nodes.
    """
    rng = random.Random(int(seed))

    # Build a forest by concatenating T trees.
    global_parents: List[int] = []
    offset = 0
    for t in range(cfg.num_trees):
        parents_t = _sample_tree_parents(
            rng=rng,
            depth=cfg.tree_depth,
            width=cfg.tree_width,
            randomize=cfg.randomize_tree,
        )
        # Shift indices and append; root stays -1.
        for p in parents_t:
            global_parents.append(p if p < 0 else p + offset)
        offset += len(parents_t)

    n = len(global_parents)
    children = _parents_to_children(global_parents)

    # Node input values and targets.
    if cfg.aggregate == "count":
        targets_count = _compute_subtree_targets_count(global_parents, children, leaf_only=cfg.leaf_only)
        if cfg.empty_param:
            input_values: List[List[float]] = [[] for _ in range(n)]
        else:
            # Scalar-1 count; leaf-only uses 1 on leaves, 0 otherwise (pass-through nodes).
            input_values = []
            for u in range(n):
                if cfg.leaf_only and len(children[u]) > 0:
                    input_values.append([0.0])
                else:
                    input_values.append([1.0])
        target_values: List[List[float]] = [[float(c)] for c in targets_count]
    else:
        values = _make_values(rng, n=n, dim=cfg.param_dim, leaf_only=cfg.leaf_only, children=children)
        targets = _compute_subtree_targets_sum(global_parents, children, values=values)
        input_values = [[float(x) for x in v.tolist()] for v in values]
        target_values = [[float(x) for x in t.tolist()] for t in targets]

    if cfg.shuffle_nodes:
        tokens, edges = _shuffle_tokens_and_edges(
            rng=rng,
            n=n,
            global_parents=global_parents,
            input_values=input_values,
            target_values=target_values,
            edge_direction=cfg.edge_direction,
        )
    else:
        tokens = []
        for u in range(n):
            raw = {"input_value": input_values[u], "target_value": target_values[u]}
            tokens.append(Token(type_name="tree_node", entity_id=u, status=2, raw_data=raw))

        edges = []
        for v, p in enumerate(global_parents):
            if p < 0:
                continue
            if cfg.edge_direction in ("both", "p2c"):
                edges.append((p, v, "P2C"))
            if cfg.edge_direction in ("both", "c2p"):
                edges.append((v, p, "C2P"))

    types = build_tree_task_types(cfg)
    relationships = _build_relationships()
    return EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)

