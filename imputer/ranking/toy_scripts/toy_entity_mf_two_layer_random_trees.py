from __future__ import annotations

"""
Toy generalization experiment for EntityMarformer on random depth-2 trees.

Setup:
- 13 nodes per graph: 0=root, 1..3=children, 4..12=grandchildren.
- Always a 2-layer tree (root -> children -> grandchildren), but the assignment
  of grandchildren to children is randomized per graph.
- Targets: subtree sizes (inclusive of the node itself).
  - root: 13
  - child j: 1 + (#grandchildren assigned to j)
  - grandchild: 1

We use:
- A single SyntheticRegressionType "tree_node" with (input_dim=1, output_dim=1)
- VariationConfig(enabled=False, num_entities=13, reg_weight=1.0).
- EntityMarformer with 3 layers, 1 head, small embedding dimension.
- Architecture knobs (use_per_head_rel, use_rel_value, use_addone_attn,
  use_feature_only_norm, scale_shared_rel, type_embedding_init) are exposed
  as function parameters for easy comparison.

We generate multiple random trees, split into train/test, train the model, and
monitor whether it generalizes the subtree counting behavior to held-out trees,
not just memorizing a single structure.

Run from repo root:

  python toy_entity_mf_two_layer_random_trees.py
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


@dataclass
class TreeSample:
    graph: EntityGraph
    targets: List[float]  # subtree sizes per node index


def build_random_depth2_tree(
    device: torch.device,
    rng: random.Random,
    shuffle_nodes: bool = False,
) -> TreeSample:
    """
    Build a random depth-2 tree on 13 nodes:
      - node 0: root
      - nodes 1..3: children
      - nodes 4..12: grandchildren
    Each grandchild is assigned uniformly at random to one of the 3 children.
    """
    N = 13
    root = 0
    children = [1, 2, 3]
    grandchildren = list(range(4, N))

    # Randomly assign each grandchild to a child.
    assignments: Dict[int, List[int]] = {c: [] for c in children}
    for g in grandchildren:
        parent = rng.choice(children)
        assignments[parent].append(g)

    # Relationships: P2C and C2P for tree_node.
    relationships = [
        Relationship(name="P2C", source_type="tree_node", target_type="tree_node", inverse="C2P"),
        Relationship(name="C2P", source_type="tree_node", target_type="tree_node", inverse="P2C"),
    ]

    # Subtree sizes:
    # - each grandchild: 1
    # - each child j: 1 + len(assignments[j])
    # - root: 1 + sum_j subtree_size(child_j)
    subtree_sizes: List[float] = [0.0 for _ in range(N)]
    for g in grandchildren:
        subtree_sizes[g] = 1.0
    for c in children:
        subtree_sizes[c] = 1.0 + float(len(assignments[c]))
    subtree_sizes[root] = 1.0 + sum(subtree_sizes[c] for c in children)

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
    for c in children:
        rc = inv_perm[root], inv_perm[c]
        edges.append((rc[0], rc[1], "P2C"))
        edges.append((rc[1], rc[0], "C2P"))
        for g in assignments[c]:
            cg = inv_perm[c], inv_perm[g]
            edges.append((cg[0], cg[1], "P2C"))
            edges.append((cg[1], cg[0], "C2P"))

    # Types registry (shared across all graphs)
    slices = RegressionSlices(input_dim=1, output_dim=1)
    var_cfg = VariationConfig(enabled=False, num_entities=N, reg_weight=1.0)
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
    Handles both per-head and shared-bias relational attention layouts.
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
                    # Per-head: Q outputs [D]. Reshaped to [H, hd] per token.
                    # Within each head, last R dims are relational:
                    #   head h -> flat index h*hd + (hd - R) + p2c_index
                    kc = hd - R
                    for h in range(H):
                        attn.Q.bias[h * hd + kc + p2c_index] = 5.0
                else:
                    # Shared-bias: Q outputs [D + R]. Relational dims at D..D+R-1.
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

            ff = block["ff"]
            for name, param in ff.named_parameters():
                if "weight" in name or "bias" in name:
                    param.zero_()

def run_entity_mf_random_trees(
    device: torch.device | None = None,
    shuffle_nodes: bool = True,
    freeze_variation: bool = False,
    guide_attention: bool = False,
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

    num_train = 10
    num_test = 10

    train_samples: List[TreeSample] = []
    for _ in range(num_train):
        train_samples.append(build_random_depth2_tree(device=device, rng=rng, shuffle_nodes=shuffle_nodes))

    test_samples: List[TreeSample] = []
    for _ in range(num_test):
        test_samples.append(build_random_depth2_tree(device=device, rng=rng, shuffle_nodes=shuffle_nodes))

    ref_graph = train_samples[0].graph
    types = ref_graph.types

    cfg = EntityMarformerConfig(
        embedding_dim=8,
        num_layers=3,
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
            _guided_init_attention(model, p2c_index=p2c_idx)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    print(
        "Starting EntityMarformer random-tree generalization experiment\n"
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
                    ent_str = " ".join(f"{eid:4.0f}" for eid in entity_ids)
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
    # Default: per-head relational attention, node shuffling, guided init.
    # Toggle architecture knobs to compare designs:
    #   run_entity_mf_random_trees(use_per_head_rel=False, scale_shared_rel=True)  # shared-bias
    #   run_entity_mf_random_trees(use_rel_value=True)     # relational value augmentation
    #   run_entity_mf_random_trees(use_addone_attn=True)   # add-one attention
    #   run_entity_mf_random_trees(use_feature_only_norm=True)  # norm feature stream only
    # run_entity_mf_random_trees(shuffle_nodes=True, freeze_variation=False, guide_attention=False)
    run_entity_mf_random_trees(shuffle_nodes=True, 
                               freeze_variation=False, 
                               guide_attention=False,
                               use_per_head_rel=False,
                               use_rel_value=False,
                               use_addone_attn=True,
                               use_feature_only_norm=False,
                               scale_shared_rel=True,
                               type_embedding_init="normal")
        



