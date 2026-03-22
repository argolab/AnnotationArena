from __future__ import annotations

"""
Toy generalization experiment for EntityMarformer on random bounded-depth trees.

Key differences from the two-layer script:
- We generate **random rooted trees** with:
  - depth \(\leq max_depth\)
  - out-degree at each node \(\leq max_degree\)
  - total number of nodes \(\leq max_nodes\)
- The actual depth and total size **vary per sample**.
- We optionally **shuffle node indices** so the model cannot rely on fixed
  positions for any structural role (e.g., the root is not always index 0).

Targets are subtree sizes (inclusive):
- For each node \(u\), target is the number of nodes in the subtree rooted at \(u\).

We use:
- A single SyntheticRegressionType "tree_node" with (input_dim=1, output_dim=1)
- EntityMarformer with a small configuration.

Run from repo root:

  python toy_final.py
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
    targets: List[float]  # subtree sizes per node index (aligned with tokens)


def build_random_bounded_tree(
    device: torch.device,
    rng: random.Random,
    max_depth: int,
    max_degree: int,
    max_nodes: int,
    shuffle_nodes: bool = True,
) -> TreeSample:
    """
    Build a random rooted tree with constraints:
      - depth <= max_depth (root at depth 0)
      - out-degree at each node <= max_degree
      - total number of nodes <= max_nodes
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

    # Subtree sizes in canonical index space.
    subtree_sizes: List[float] = [1.0 for _ in range(N)]
    # Accumulate child subtree sizes into parents in reverse topological order.
    for node in reversed(range(1, N)):
        p = parents[node]
        assert 0 <= p < N
        subtree_sizes[p] += subtree_sizes[node]

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

    # Tokens: one type "tree_node" with scalar input=1.0 and target=subtree size.
    # We assign both token index and entity_id in the *new* index space, while
    # targets come from the corresponding canonical node (old_idx).
    tokens: List[Token] = []
    targets_shuffled: List[float] = []
    for new_idx in range(N):
        old_idx = perm[new_idx]
        raw = {
            "input_value": [1.0],
            "target_value": [subtree_sizes[old_idx]],
        }
        tokens.append(Token(type_name="tree_node", entity_id=new_idx, status=2, raw_data=raw))
        targets_shuffled.append(subtree_sizes[old_idx])

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


def _guided_init_attention(model: EntityMarformer, p2c_index: int) -> None:
    """
    Heuristic attention initialization to encourage:
      - strong positive bias on P2C edges via Q_rel
      - V to propagate param-slice information
      - out to behave approximately like identity

    This is a soft inductive bias, not an exact analytic solution.
    """
    D = model.model_dim
    F_dim = model.feature_dim
    P_dim = model.param_dim

    with torch.no_grad():
        for block in model.blocks:
            attn = block["attn"]

            # Zero Q/K weights and biases first.
            attn.Q.weight.zero_()
            attn.Q.bias.zero_()
            attn.K.weight.zero_()
            attn.K.bias.zero_()

            # Relational part of Q bias: add positive bias on the P2C relation dim.
            # Q_full outputs [D + R]; relation dims are indices D..D+R-1.
            if 0 <= p2c_index < attn.num_relationships:
                attn.Q.bias[D + p2c_index] = 5.0

            # V: copy param slice of combined [features, params] into the param slice,
            # scaled by the branching factor (3) so that, when attention uses a
            # roughly uniform weight over 3 children, the aggregate approximates
            # a sum rather than an average.
            attn.V.weight.zero_()
            attn.V.bias.zero_()
            for d in range(P_dim):
                attn.V.weight[F_dim + d, F_dim + d] = 3.0

            # out: make it close to identity on the whole model_dim.
            attn.out.weight.zero_()
            attn.out.bias.zero_()
            for d in range(D):
                attn.out.weight[d, d] = 1.0

            # Optionally, make FFN initially close to zero so it does not disturb
            # the initial attention behavior.
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
            _guided_init_attention(model, p2c_index=p2c_idx)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    print(
        "Starting EntityMarformer random-tree generalization experiment "
        f"(shuffle_nodes={shuffle_nodes}, freeze_variation={freeze_variation}, guide_attention={guide_attention})..."
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
                    tgts_str = " ".join(f"{round(v, 3):4.1f}" for v in tgts)
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
    # By default, run with node shuffling enabled and trainable variation.
    # To test fixed-per-entity embeddings, call:
    #   run_entity_mf_random_trees(shuffle_nodes=True, freeze_variation=True)
    # To test attention-guided initialization, pass guide_attention=True.
    run_entity_mf_random_trees(shuffle_nodes=True, freeze_variation=False, guide_attention=False)



