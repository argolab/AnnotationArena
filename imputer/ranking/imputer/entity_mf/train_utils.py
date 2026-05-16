from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from imputer.data import DataConverter
from imputer.utils import sizes_from_configs


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)


def new_run_dir(output_root: Path, run_name: str | None = None) -> Path:
    if run_name:
        run_dir = output_root / run_name
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        return run_dir
    from datetime import datetime

    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_bundle_and_converter(data_dir: Path) -> tuple[Any, DataConverter, Dict[str, int]]:
    bundle_path = data_dir / "data_bundle.json"
    configs_path = data_dir / "configs.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"data_bundle.json not found in {data_dir}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found in {data_dir}")

    with open(configs_path, "r") as f:
        configs = json.load(f)

    sizes = sizes_from_configs(configs)
    model_cfg = configs.get("model_config") or {}
    max_rank_size = int(model_cfg.get("max_rank_size", 2))

    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=max_rank_size,
    )
    bundle = converter.load_bundle_data(bundle_path)
    sizes["max_rank_size"] = max_rank_size
    return bundle, converter, sizes


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by flat and recurrent Marformer trainers."""
    parser.add_argument("--data-dir", required=True, help="Directory with data_bundle.json and configs.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--masking-rate",
        type=float,
        default=0.15,
        help="Fraction of observed vars to mask each step.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-root",
        type=str,
        default="OUTPUT/ENTITY_MF",
        help="Root directory for run outputs.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name under output-root.")
    parser.add_argument(
        "--transductive-learning",
        action="store_true",
        help="Include test_observed tokens in training.",
    )
    parser.add_argument(
        "--transductive-valtest-mask",
        action="store_true",
        help="In transductive mode, mask only val/test observed.",
    )
    parser.add_argument("--llm-annotator-id", type=int, default=None)
    parser.add_argument("--human-observed-rate", type=float, default=0.0)
    parser.add_argument(
        "--always-observed-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--max-item", type=int, default=None, help="Max items per forward pass (chunking).")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-ffn-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use-per-head-rel", action="store_true", default=True)
    parser.add_argument("--no-per-head-rel", dest="use_per_head_rel", action="store_false")
    parser.add_argument("--use-pointer", action="store_true")
    parser.add_argument("--use-rel-value", action="store_true")
    parser.add_argument("--use-addone-attn", action="store_true")
    parser.add_argument("--scale-shared-rel", action="store_true", default=False)
    parser.add_argument("--use-graph-mask", action="store_true")
    parser.add_argument(
        "--type-embedding-init",
        type=str,
        default="normal",
        choices=["normal", "scaled_normal", "kaiming"],
    )
    parser.add_argument("--overwrite-existing-data", action="store_true")
    parser.add_argument("--annotator-reg-weight", type=float, default=0.0)
    parser.add_argument("--mask-augmentations", type=int, default=5)
    parser.add_argument("--masked-loss-weight", type=float, default=15.0)
    parser.add_argument("--observed-loss-weight", type=float, default=1.0)
    parser.add_argument("--llm-input-dist", action="store_true")
    parser.add_argument("--item-dropout-rate", type=float, default=1.0)
    parser.add_argument("--annotator-dropout-rate", type=float, default=0.0)
    parser.add_argument("--item-reg-weight", type=float, default=0.0)
    parser.add_argument("--attribute-reg-weight", type=float, default=0.0)
    parser.add_argument("--use-deviation-norm", action="store_true")
    parser.add_argument("--use-param-output-head", action="store_true")
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-step-epoch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-item-chunks", action="store_true")


def apply_common_model_config(config: Any, args: argparse.Namespace) -> None:
    if args.embedding_dim is not None:
        config.embedding_dim = args.embedding_dim
    if args.attention_heads is not None:
        config.attention_heads = args.attention_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.num_ffn_layers is not None:
        config.num_ffn_layers = args.num_ffn_layers
    if args.dropout is not None:
        config.dropout = args.dropout
    config.use_per_head_rel = args.use_per_head_rel
    config.use_pointer = args.use_pointer
    config.use_rel_value = args.use_rel_value
    config.use_addone_attn = args.use_addone_attn
    config.type_embedding_init = args.type_embedding_init
    config.use_deviation_norm = args.use_deviation_norm
    config.scale_shared_rel = args.scale_shared_rel
    config.use_graph_mask = args.use_graph_mask
    config.use_param_output_head = args.use_param_output_head


def shared_model_config_dict(config: Any, global_param_dim: int) -> Dict[str, Any]:
    return {
        "embedding_dim": config.embedding_dim,
        "attention_heads": config.attention_heads,
        "d_ff": config.d_ff,
        "num_ffn_layers": config.num_ffn_layers,
        "dropout": config.dropout,
        "use_per_head_rel": config.use_per_head_rel,
        "use_pointer": config.use_pointer,
        "use_rel_value": config.use_rel_value,
        "use_addone_attn": config.use_addone_attn,
        "type_embedding_init": config.type_embedding_init,
        "use_deviation_norm": config.use_deviation_norm,
        "scale_shared_rel": config.scale_shared_rel,
        "use_graph_mask": config.use_graph_mask,
        "use_param_output_head": config.use_param_output_head,
        "logit_high": config.logit_high,
        "temperature": config.temperature,
        "global_param_dim": global_param_dim,
    }


def shared_training_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "masking_rate": args.masking_rate,
        "transductive_learning": bool(args.transductive_learning),
        "transductive_valtest_mask": bool(args.transductive_valtest_mask),
        "llm_annotator_id": args.llm_annotator_id,
        "human_observed_rate": args.human_observed_rate,
        "always_observed_ids": args.always_observed_ids,
        "max_item": args.max_item,
        "annotator_reg_weight": args.annotator_reg_weight,
        "item_reg_weight": args.item_reg_weight,
        "attribute_reg_weight": args.attribute_reg_weight,
        "lr_schedule": args.lr_schedule,
        "lr_min": args.lr_min,
        "lr_step_epoch": args.lr_step_epoch,
        "random_item_chunks": bool(args.random_item_chunks),
        "mask_augmentations": args.mask_augmentations,
        "masked_loss_weight": args.masked_loss_weight,
        "observed_loss_weight": args.observed_loss_weight,
        "llm_input_dist": args.llm_input_dist,
        "item_dropout_rate": args.item_dropout_rate,
        "annotator_dropout_rate": args.annotator_dropout_rate,
        "device": args.device,
    }
