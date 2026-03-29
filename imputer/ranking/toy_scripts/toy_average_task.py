from __future__ import annotations

"""
Toy generalization experiment for EntityMarformer on random bounded-depth trees.

**Task (subtree average):**
- Each node i gets a scalar a_i (uniform i.i.d. in [scalar_low, scalar_high] per graph).
- Target at node u is the mean of a_v over all v in the subtree rooted at u
  (including u).

Graph generation (same as toy_final):
- Random rooted trees with depth <= max_depth, out-degree <= max_degree,
  size <= max_nodes; structure varies per sample.
- Optional **shuffle node indices** so roles are not tied to fixed positions.

Note: `guide_attention` uses a sum-leaning V init (from toy_final); for strict
mean aggregation you may get better results with `guide_attention=False`.

We use:
- A single SyntheticRegressionType "tree_node" with (input_dim=1, output_dim=1);
  input is the node's scalar, target is its subtree mean.
- VariationConfig with num_entities=MAX_NUM_ENTITIES (upper bound for all graphs).
- EntityMarformer with a small configuration.
- Architecture knobs on `run_entity_mf_random_trees`.

Run from repo root:

  python toy_average_task.py
"""

import random
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


# Upper bound on the number of entities we ever use in a single graph.
# All graphs will have num_nodes <= MAX_NUM_ENTITIES, but individual trees
# can be much smaller.
MAX_NUM_ENTITIES = 64


@dataclass
class TreeSample:
    graph: EntityGraph
    targets: List[float]  # subtree-mean targets per token index (aligned with tokens)


def build_random_bounded_tree(
    device: torch.device,
    rng: random.Random,
    max_depth: int,
    max_degree: int,
    max_nodes: int,
    shuffle_nodes: bool = True,
    scalar_low: float = -10.0,
    scalar_high: float = 10.0,
) -> TreeSample:
    """
    Build a random rooted tree with constraints:
      - depth <= max_depth (root at depth 0)
      - out-degree at each node <= max_degree
      - total number of nodes <= max_nodes

    Each node gets a scalar input in [scalar_low, scalar_high] (uniform).
    Targets are subtree means of those scalars (per node).
    """
    assert max_depth >= 0
    assert max_degree >= 0
    assert 1 <= max_nodes <= MAX_NUM_ENTITIES

    # Canonical tree construction in "old" index space.
    # Root is index 0; we grow the tree breadth-first.
    parents: List[int] = [-1]  # parent of each node; root has parent -1
    depths: List[int] = [0]
    N = 1

    i = 0
    while i < N:
        if depths[i] >= max_depth:
            i += 1
            continue
        remaining = max_nodes - N
        if remaining <= 0:
            break
        # Sample number of children for node i.
        max_children = min(max_degree, remaining)
        if max_children > 0:
            num_children = rng.randint(0, max_children)
        else:
            num_children = 0
        for _ in range(num_children):
            parents.append(i)
            depths.append(depths[i] + 1)
            N += 1
        i += 1

    # Relationships: P2C and C2P for tree_node.
    relationships = [
        Relationship(name="P2C", source_type="tree_node", target_type="tree_node", inverse="C2P"),
        Relationship(name="C2P", source_type="tree_node", target_type="tree_node", inverse="P2C"),
    ]

    # Per-node scalar (canonical index); used as input and to define subtree means.
    node_scalar: List[float] = [
        rng.uniform(scalar_low, scalar_high) for _ in range(N)
    ]

    # Subtree sum of scalars and subtree node counts -> mean = sum / count.
    children: List[List[int]] = [[] for _ in range(N)]
    for j in range(1, N):
        children[parents[j]].append(j)

    subtree_sum: List[float] = [0.0] * N
    subtree_cnt: List[int] = [0] * N

    def _dfs_subtree(u: int) -> None:
        s = node_scalar[u]
        c = 1
        for v in children[u]:
            _dfs_subtree(v)
            s += subtree_sum[v]
            c += subtree_cnt[v]
        subtree_sum[u] = s
        subtree_cnt[u] = c

    _dfs_subtree(0)

    subtree_mean: List[float] = [
        subtree_sum[i] / float(subtree_cnt[i]) for i in range(N)
    ]

    # Optional node index shuffling.
    #
    # We treat `perm` as mapping *new_idx -> old_idx* (canonical index), and
    # `inv_perm` as the inverse mapping *old_idx -> new_idx*. This way, the
    # tokens, edges, and targets are all expressed in the same "new" index
    # space when shuffling is enabled.
    if shuffle_nodes:
        perm = list(range(N))  # new_idx -> old_idx
        rng.shuffle(perm)
        inv_perm = {old: new for new, old in enumerate(perm)}
    else:
        perm = list(range(N))  # identity
        inv_perm = {i: i for i in range(N)}

    # Tokens: one type "tree_node" with scalar input=a_old and target=subtree mean.
    # We assign both token index and entity_id in the *new* index space, while
    # inputs/targets come from the corresponding canonical node (old_idx).
    tokens: List[Token] = []
    targets_shuffled: List[float] = []
    for new_idx in range(N):
        old_idx = perm[new_idx]
        raw = {
            "input_value": [node_scalar[old_idx]],
            "target_value": [subtree_mean[old_idx]],
        }
        tokens.append(Token(type_name="tree_node", entity_id=new_idx, status=2, raw_data=raw))
        targets_shuffled.append(subtree_mean[old_idx])

    # Edges: P2C and C2P using the *new* indices.
    edges: List[Tuple[int, int, str]] = []
    for node in range(1, N):
        p_old = parents[node]
        c_old = node
        p_new = inv_perm[p_old]
        c_new = inv_perm[c_old]
        edges.append((p_new, c_new, "P2C"))
        edges.append((c_new, p_new, "C2P"))

    # Types registry (shared across all graphs). We fix num_entities to an
    # upper bound so that the model can, in principle, be reused for trees up
    # to MAX_NUM_ENTITIES nodes.
    slices = RegressionSlices(input_dim=1, output_dim=1)
    var_cfg = VariationConfig(enabled=False, num_entities=MAX_NUM_ENTITIES, reg_weight=1.0)
    tree_type = SyntheticRegressionType(name="tree_node", slices=slices, has_target=True, variation=var_cfg)
    types: Dict[str, EntityType] = {"tree_node": tree_type}

    graph = EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)
    return TreeSample(graph=graph, targets=targets_shuffled)


def _compute_mse_on_sample(
    model: EntityMarformer,
    sample: TreeSample,
    device: torch.device,
) -> torch.Tensor:
    graph = sample.graph
    params = model(graph, device=device)  # [1, L, P]

    node_type = graph.types["tree_node"]
    assert isinstance(node_type, SyntheticRegressionType)
    out_slice = node_type.slices.output_slice()

    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    for idx, tok in enumerate(graph.tokens):
        tgt = sample.targets[idx]
        p = params[0, idx, out_slice]  # [1]
        preds.append(p)
        tgts.append(torch.tensor([tgt], device=device, dtype=p.dtype))

    pred = torch.stack(preds, dim=0)  # [N,1]
    tgt = torch.stack(tgts, dim=0)    # [N,1]
    return F.mse_loss(pred, tgt, reduction="mean")


def _compute_deviation_reg_loss(model: EntityMarformer, types: Dict[str, EntityType], device: torch.device) -> torch.Tensor:
    reg_loss = torch.zeros((), device=device)
    for type_name, t in types.items():
        if not t.variation.enabled or t.variation.reg_weight <= 0.0:
            continue
        table = model.deviation_tables.get(type_name, None)
        if table is None:
            continue
        reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()
    return reg_loss


def _guided_init_attention(
    model: EntityMarformer,
    p2c_index: int,
    param_branch_scale: float = 3.0,
) -> None:
    """
    Heuristic attention initialization to encourage:
      - strong positive bias on P2C edges via Q_rel
      - V to propagate param-slice information
      - out to behave approximately like identity

    This is a soft inductive bias, not an exact analytic solution.
    Handles both per-head and shared-bias relational attention layouts.

    param_branch_scale: diagonal scale on V for param dims (use ~max out-degree
    so uniform attention over children approximates summing child params).
    """
    D = model.model_dim
    F_dim = model.feature_dim
    P_dim = model.param_dim

    with torch.no_grad():
        for block in model.blocks:
            attn = block["attn"]
            H = attn.num_heads
            hd = attn.head_dim
            R = attn.num_relationships

            attn.Q.weight.zero_()
            attn.Q.bias.zero_()
            attn.K.weight.zero_()
            attn.K.bias.zero_()

            if 0 <= p2c_index < R:
                if attn.use_per_head_rel:
                    kc = hd - R
                    for h in range(H):
                        attn.Q.bias[h * hd + kc + p2c_index] = 5.0
                else:
                    attn.Q.bias[D + p2c_index] = 5.0

            attn.V.weight.zero_()
            attn.V.bias.zero_()
            for d in range(P_dim):
                attn.V.weight[F_dim + d, F_dim + d] = param_branch_scale

            attn.out.weight.zero_()
            attn.out.bias.zero_()
            for d in range(D):
                attn.out.weight[d, d] = 1.0

            ff = block["ff"]
            for name, param in ff.named_parameters():
                if "weight" in name or "bias" in name:
                    param.zero_()

def run_entity_mf_random_trees(
    device: torch.device | None = None,
    shuffle_nodes: bool = True,
    freeze_variation: bool = False,
    guide_attention: bool = False,
    max_depth: int = 4,
    max_degree: int = 3,
    max_nodes: int = 32,
    num_train: int = 30,
    num_test: int = 10,
    scalar_low: float = -10.0,
    scalar_high: float = 10.0,
    # Architecture knobs (mirror EntityMarformerConfig)
    use_per_head_rel: bool = True,
    use_rel_value: bool = False,
    use_addone_attn: bool = False,
    use_feature_only_norm: bool = False,
    scale_shared_rel: bool = False,
    type_embedding_init: str = "normal",
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(42)

    # Build train/test samples (types registry is identical for all).
    train_samples: List[TreeSample] = []
    for _ in range(num_train):
        train_samples.append(
            build_random_bounded_tree(
                device=device,
                rng=rng,
                max_depth=max_depth,
                max_degree=max_degree,
                max_nodes=max_nodes,
                shuffle_nodes=shuffle_nodes,
                scalar_low=scalar_low,
                scalar_high=scalar_high,
            )
        )

    test_samples: List[TreeSample] = []
    for _ in range(num_test):
        test_samples.append(
            build_random_bounded_tree(
                device=device,
                rng=rng,
                max_depth=max_depth,
                max_degree=max_degree,
                max_nodes=max_nodes,
                shuffle_nodes=shuffle_nodes,
                scalar_low=scalar_low,
                scalar_high=scalar_high,
            )
        )

    # Use types and relationships from the first train sample to build the model.
    ref_graph = train_samples[0].graph
    types = ref_graph.types

    cfg = EntityMarformerConfig(
        embedding_dim=8,
        num_layers=1,
        attention_heads=1,
        dropout=0.0,
        d_ff=32,
        num_ffn_layers=1,
        use_per_head_rel=use_per_head_rel,
        use_rel_value=use_rel_value,
        use_addone_attn=use_addone_attn,
        use_feature_only_norm=use_feature_only_norm,
        scale_shared_rel=scale_shared_rel,
        type_embedding_init=type_embedding_init,
    )

    model = EntityMarformer(
        config=cfg,
        types=types,
        num_relationships=ref_graph.num_relationships,
    ).to(device)

    if freeze_variation:
        # Randomly initialize deviation tables once and freeze them so they act
        # as fixed per-entity embeddings (symmetry breaking) without being
        # further optimized.
        for type_name, t in types.items():
            if not t.variation.enabled or t.variation.num_entities <= 0:
                continue
            table = model.deviation_tables.get(type_name, None)
            if table is None:
                continue
            with torch.no_grad():
                table.normal_(mean=0.0, std=0.1)
            table.requires_grad_(False)

    if guide_attention:
        # Use relationships on the reference graph to locate the P2C relation.
        rel_index = {rel.name: i for i, rel in enumerate(ref_graph.relationships)}
        p2c_idx = rel_index.get("P2C", None)
        if p2c_idx is not None:
            _guided_init_attention(
                model,
                p2c_index=p2c_idx,
                param_branch_scale=float(max_degree),
            )

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    print(
        "Starting EntityMarformer subtree-AVERAGE task (bounded random trees)\n"
        f"  max_depth={max_depth}, max_degree={max_degree}, max_nodes={max_nodes}, "
        f"num_train={num_train}, num_test={num_test}\n"
        f"  node scalars ~ U({scalar_low}, {scalar_high}), "
        f"target = mean(scalars in subtree)\n"
        f"  shuffle_nodes={shuffle_nodes}, freeze_variation={freeze_variation}, guide_attention={guide_attention}\n"
        f"  use_per_head_rel={use_per_head_rel}, use_rel_value={use_rel_value}, "
        f"use_addone_attn={use_addone_attn}\n"
        f"  use_feature_only_norm={use_feature_only_norm}, scale_shared_rel={scale_shared_rel}, "
        f"type_embedding_init={type_embedding_init}"
    )
    for step in range(1000):
        model.train()
        opt.zero_grad()

        # Average MSE over all train samples each step (small dataset).
        mse_sum = torch.zeros((), device=device)
        for sample in train_samples:
            mse_sum = mse_sum + _compute_mse_on_sample(model, sample, device=device)
        train_mse = mse_sum / float(len(train_samples))

        reg_loss = _compute_deviation_reg_loss(model, types, device=device)
        loss = train_mse + reg_loss

        loss.backward()
        opt.step()

        if (step + 1) % 10 == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                # Aggregate test MSE.
                mse_sum_test = torch.zeros((), device=device)
                for sample in test_samples:
                    mse_sum_test = mse_sum_test + _compute_mse_on_sample(model, sample, device=device)
                test_mse = mse_sum_test / float(len(test_samples))

                print(
                    f"step {step+1:4d} | "
                    f"train_mse={float(train_mse.item()):.4f} "
                    f"test_mse={float(test_mse.item()):.4f} "
                    f"reg={float(reg_loss.item()):.4f} "
                    f"total={float(loss.item()):.4f}"
                )

                # Inspect predictions vs targets for a few train/test samples.
                def _print_sample(prefix: str, sample: TreeSample) -> None:
                    graph = sample.graph
                    params = model(graph, device=device)  # [1, L, P]
                    node_type = graph.types["tree_node"]
                    assert isinstance(node_type, SyntheticRegressionType)
                    out_slice = node_type.slices.output_slice()
                    entity_ids = [tok.entity_id for tok in graph.tokens]
                    preds = []
                    tgts = []
                    for idx, tok in enumerate(graph.tokens):
                        p = params[0, idx, out_slice]
                        preds.append(float(p.item()))
                        tgts.append(float(sample.targets[idx]))
                    ent_str = " ".join(f"{eid:4d}" for eid in entity_ids)
                    preds_str = " ".join(f"{round(v, 3):4.2f}" for v in preds)
                    tgts_str = " ".join(f"{round(v, 3):4.2f}" for v in tgts)
                    print(f"  {prefix} ents:  {ent_str}")
                    print(f"  {prefix} preds: {preds_str}")
                    print(f"  {prefix} tgts : {tgts_str}")
                    print()

                # Pick two random train/test samples to inspect.
                if train_samples:
                    train_idxs = rng.sample(range(len(train_samples)), k=min(2, len(train_samples)))
                    for i in train_idxs:
                        _print_sample(prefix=f"train[{i}]", sample=train_samples[i])
                if test_samples:
                    test_idxs = rng.sample(range(len(test_samples)), k=min(2, len(test_samples)))
                    for i in test_idxs:
                        _print_sample(prefix=f"test[{i}]", sample=test_samples[i])

    # Final summary on test set.
    model.eval()
    with torch.no_grad():
        mse_sum_test = torch.zeros((), device=device)
        for sample in test_samples:
            mse_sum_test = mse_sum_test + _compute_mse_on_sample(model, sample, device=device)
        test_mse = mse_sum_test / float(len(test_samples))
    print("\nFinal test MSE:", float(test_mse.item()))


if __name__ == "__main__":
    # Default: bounded random trees, node shuffling. Toggle knobs like the two-layer toy:
    #   run_entity_mf_random_trees(use_per_head_rel=False, scale_shared_rel=True, guide_attention=True)
    #   run_entity_mf_random_trees(use_rel_value=True, guide_attention=True)
    run_entity_mf_random_trees(shuffle_nodes=True, 
                               freeze_variation=False,
                               guide_attention=False,
                               use_per_head_rel=False,
                               use_rel_value=True,
                               use_addone_attn=False,
                               use_feature_only_norm=True,
                               scale_shared_rel=True,
                               type_embedding_init="normal",
                               scalar_low=-10.0,
                               scalar_high=10.0)



