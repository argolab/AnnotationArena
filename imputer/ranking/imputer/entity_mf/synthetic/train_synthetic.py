from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import torch

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.model import EntityMarformer

from .datagen_tree import TreeTaskConfig, generate_tree_graph
from tqdm.auto import tqdm


TaskName = Literal["tree"]


@dataclass
class TrainCurves:
    train_loss: List[float]
    test_loss: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_graph_loss(
    model: EntityMarformer,
    graph,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute supervised MSE loss on all tokens that have target_value in raw_data.
    """
    params = model(graph, device=device)  # [1, L, P]
    total = torch.zeros((), device=device)
    total_n = 0

    for type_name, t in model.types.items():
        type_mask = torch.tensor(
            [tok.type_name == type_name for tok in graph.tokens],
            device=device,
            dtype=torch.bool,
        ).unsqueeze(0)
        loss_t = t.compute_loss(params, graph.tokens, type_mask, model.global_param_dim)
        n_t = int(type_mask[0].sum().item())
        if n_t > 0:
            total = total + loss_t * n_t
            total_n += n_t

    if total_n == 0:
        return torch.zeros((), device=device)

    return total / total_n


def build_tree_datasets(cfg: TreeTaskConfig, num_train: int, num_test: int, seed: int) -> Tuple[List[Any], List[Any]]:
    train_graphs = [generate_tree_graph(cfg, seed=seed + i) for i in range(num_train)]
    test_graphs = [generate_tree_graph(cfg, seed=seed + 10_000 + i) for i in range(num_test)]
    return train_graphs, test_graphs


def main() -> None:
    parser = argparse.ArgumentParser("Synthetic supervised trainer for EntityMarformer sanity checks.")
    parser.add_argument("--task", type=str, choices=["tree"], required=True)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-graphs", type=int, default=200)
    parser.add_argument("--num-test-graphs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="OUTPUT/SYNTHETIC/tmp")

    # Model
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--num-ffn-layers", type=int, default=1)

    # Architecture knobs (EntityMarformerConfig) — defaults match synthetic sweep baseline
    parser.add_argument("--use-per-head-rel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-pointer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-rel-value", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-addone-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--type-embedding-init",
        type=str,
        default="normal",
        choices=["normal", "scaled_normal", "kaiming"],
    )
    parser.add_argument("--use-deviation-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale-shared-rel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-feature-only-norm", action=argparse.BooleanOptionalAction, default=True)

    # Tree task args
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--tree-width", type=int, default=3)
    parser.add_argument("--num-trees", type=int, default=1)
    parser.add_argument("--randomize-tree", action="store_true")
    parser.add_argument("--aggregate", type=str, choices=["count", "sum"], default="count")
    parser.add_argument("--leaf-only", action="store_true")
    parser.add_argument("--empty-param", action="store_true")
    parser.add_argument("--param-dim", type=int, default=1)
    parser.add_argument("--edge-direction", type=str, choices=["both", "p2c", "c2p"], default="both")
    parser.add_argument(
        "--shuffle-nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Permute node indices after building the tree so position does not encode depth/role (default: on).",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build task config + datasets
    if args.task == "tree":
        task_cfg = TreeTaskConfig(
            tree_depth=int(args.tree_depth),
            tree_width=int(args.tree_width),
            num_trees=int(args.num_trees),
            randomize_tree=bool(args.randomize_tree),
            aggregate=str(args.aggregate),  # type: ignore[assignment]
            leaf_only=bool(args.leaf_only),
            empty_param=bool(args.empty_param),
            param_dim=int(args.param_dim),
            edge_direction=str(args.edge_direction),  # type: ignore[assignment]
            shuffle_nodes=bool(args.shuffle_nodes),
        )
        train_graphs, test_graphs = build_tree_datasets(
            task_cfg, num_train=int(args.num_train_graphs), num_test=int(args.num_test_graphs), seed=int(args.seed)
        )
    else:
        raise ValueError(f"Unknown task {args.task}")

    # Initialize model based on one graph (for relationship count + types registry).
    graph0 = train_graphs[0]
    cfg = EntityMarformerConfig(
        embedding_dim=int(args.embedding_dim),
        num_layers=int(args.num_layers),
        attention_heads=int(args.attention_heads),
        dropout=float(args.dropout),
        d_ff=int(args.d_ff),
        num_ffn_layers=int(args.num_ffn_layers),
        use_per_head_rel=bool(args.use_per_head_rel),
        use_pointer=bool(args.use_pointer),
        use_rel_value=bool(args.use_rel_value),
        use_addone_attn=bool(args.use_addone_attn),
        type_embedding_init=str(args.type_embedding_init),
        use_deviation_norm=bool(args.use_deviation_norm),
        scale_shared_rel=bool(args.scale_shared_rel),
        use_feature_only_norm=bool(args.use_feature_only_norm),
    )
    model = EntityMarformer(config=cfg, types=graph0.types, num_relationships=graph0.num_relationships).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    curves = TrainCurves(train_loss=[], test_loss=[])

    # Save config snapshot
    snapshot: Dict[str, Any] = {
        "task": args.task,
        "task_config": asdict(task_cfg),
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_train_graphs": args.num_train_graphs,
            "num_test_graphs": args.num_test_graphs,
            "seed": args.seed,
            "device": str(device),
        },
        "model": {
            "embedding_dim": cfg.embedding_dim,
            "num_layers": cfg.num_layers,
            "attention_heads": cfg.attention_heads,
            "dropout": cfg.dropout,
            "d_ff": cfg.d_ff,
            "num_ffn_layers": cfg.num_ffn_layers,
            "global_param_dim": model.global_param_dim,
            "use_per_head_rel": cfg.use_per_head_rel,
            "use_pointer": cfg.use_pointer,
            "use_rel_value": cfg.use_rel_value,
            "use_addone_attn": cfg.use_addone_attn,
            "type_embedding_init": cfg.type_embedding_init,
            "use_deviation_norm": cfg.use_deviation_norm,
            "scale_shared_rel": cfg.scale_shared_rel,
            "use_feature_only_norm": cfg.use_feature_only_norm,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(snapshot, indent=2))

    # Short human-readable description of this run (helps when scanning logs).
    print(
        "Synthetic run:"
        f" task={args.task}"
        f" | depth={task_cfg.tree_depth}"
        f" width={task_cfg.tree_width}"
        f" num_trees={task_cfg.num_trees}"
        f" randomize_tree={task_cfg.randomize_tree}"
        f" aggregate={task_cfg.aggregate}"
        f" leaf_only={task_cfg.leaf_only}"
        f" empty_param={task_cfg.empty_param}"
        f" param_dim={task_cfg.param_dim}"
        f" edge_direction={task_cfg.edge_direction}"
        f" shuffle_nodes={task_cfg.shuffle_nodes}"
        f" | layers={cfg.num_layers}"
        f" emb_dim={cfg.embedding_dim}"
        f" heads={cfg.attention_heads}"
        f" per_head_rel={cfg.use_per_head_rel}"
        f" pointer={cfg.use_pointer}"
        f" rel_value={cfg.use_rel_value}"
        f" addone_attn={cfg.use_addone_attn}"
        f" type_emb_init={cfg.type_embedding_init}"
        f" dev_norm={cfg.use_deviation_norm}"
        f" scale_shared_rel={cfg.scale_shared_rel}"
        f" feat_only_norm={cfg.use_feature_only_norm}"
        f" | epochs={args.epochs}"
        f" lr={args.lr}"
        f" num_train_graphs={args.num_train_graphs}"
        f" num_test_graphs={args.num_test_graphs}"
    )

    num_epochs = int(args.epochs)
    epochs_run = 0
    progress = tqdm(range(num_epochs), desc="epoch", total=num_epochs)
    for epoch in progress:
        model.train()
        order = list(range(len(train_graphs)))
        random.shuffle(order)
        train_sum = 0.0
        for i in order:
            g = train_graphs[i]
            opt.zero_grad(set_to_none=True)
            loss = _compute_graph_loss(model, g, device=device)
            loss.backward()
            opt.step()
            train_sum += float(loss.detach().cpu().item())
        train_mean = train_sum / max(1, len(train_graphs))

        model.eval()
        with torch.no_grad():
            test_sum = 0.0
            for g in test_graphs:
                loss = _compute_graph_loss(model, g, device=device)
                test_sum += float(loss.detach().cpu().item())
            test_mean = test_sum / max(1, len(test_graphs))

        curves.train_loss.append(train_mean)
        curves.test_loss.append(test_mean)
        epochs_run = epoch + 1

        # Update progress bar display.
        progress.set_postfix(
            train_mse=f"{train_mean:.4f}",
            test_mse=f"{test_mean:.4f}",
        )

        # No early stopping; run for full num_epochs so sweeps are comparable.

    (out_dir / "training_curves.json").write_text(json.dumps(asdict(curves), indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Lightweight summary for sweep scripts / post-hoc analysis.
    if curves.train_loss:
        final_train = curves.train_loss[-1]
        final_test = curves.test_loss[-1]
        min_train = min(curves.train_loss)
        min_test = min(curves.test_loss)
        best_test_epoch = int(min(range(len(curves.test_loss)), key=lambda i: curves.test_loss[i])) + 1
    else:
        final_train = final_test = min_train = min_test = None
        best_test_epoch = None

    summary = {
        "final": {"train_mse": final_train, "test_mse": final_test},
        "min": {"train_mse": min_train, "test_mse": min_test, "best_test_epoch": best_test_epoch},
        "epochs_requested": num_epochs,
        "epochs_run": epochs_run,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

