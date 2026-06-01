"""
Test evaluation for a trained Recurrent Entity Marformer run.

Usage:
    python -m imputer.entity_mf.recurrent.test --run-dir RESULTS/RECURRENT_MARFORMER/...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats

from imputer.data import DataConverter, RankingData
from imputer.entity_mf.types import build_default_domain3_types
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.eval import evaluate_entity_marformer_split, EntityEvalResults
from imputer.entity_mf.backbone import MarformerBackbone

from .config import RecurrentMarformerConfig
from .model import RecurrentEntityMarformer


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _reconstruct(run_dir: Path) -> tuple[RecurrentEntityMarformer, List[RankingData], Dict[str, Any]]:
    with open(run_dir / "train_config.json") as f:
        train_config = json.load(f)

    data_dir = Path(train_config["data"]["data_dir"])
    sizes = train_config["resolved_sizes"]
    mcfg = train_config["model"]
    train_cfg = train_config["training"]

    config = RecurrentMarformerConfig()
    config.embedding_dim = mcfg["embedding_dim"]
    config.attention_heads = mcfg["attention_heads"]
    config.d_ff = mcfg["d_ff"]
    config.num_ffn_layers = mcfg["num_ffn_layers"]
    config.dropout = mcfg["dropout"]
    config.use_per_head_rel = mcfg["use_per_head_rel"]
    config.use_pointer = mcfg["use_pointer"]
    config.use_rel_value = mcfg["use_rel_value"]
    config.use_addone_attn = mcfg["use_addone_attn"]
    config.type_embedding_init = mcfg["type_embedding_init"]
    config.use_deviation_norm = mcfg["use_deviation_norm"]
    config.scale_shared_rel = mcfg["scale_shared_rel"]
    config.use_graph_mask = mcfg["use_graph_mask"]
    config.use_param_output_head = mcfg.get("use_param_output_head", False)
    config.prelude_depth = mcfg["prelude_depth"]
    config.num_core_layers = mcfg["num_core_layers"]
    config.num_recurrence = mcfg["num_recurrence"]
    config.coda_depth = mcfg["coda_depth"]

    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", 2),
    )
    bundle = converter.load_bundle_data(data_dir / "data_bundle.json")

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", 2),
        logit_high=config.logit_high,
        annotator_reg_weight=train_cfg.get("annotator_reg_weight", 0.0),
        item_reg_weight=train_cfg.get("item_reg_weight", 0.0),
        attribute_reg_weight=train_cfg.get("attribute_reg_weight", 0.0),
        llm_input_dist=train_cfg.get("llm_input_dist", False),
        item_dropout_rate=train_cfg.get("item_dropout_rate", 1.0),
    )

    train_obs = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_miss = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    graph0 = variable_list_to_entity_graph(train_obs + train_miss, types)
    model = RecurrentEntityMarformer(
        config=config,
        types=types,
        num_relationships=graph0.num_relationships,
    )

    test_obs = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_miss = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    if bool(train_cfg.get("transductive_learning", False)):
        eval_vars = train_obs + test_obs + test_miss + train_miss
    else:
        eval_vars = test_obs + test_miss + train_obs + train_miss

    return model, eval_vars, train_cfg


def _find_checkpoint(ckpt_dir: Path, which: str) -> Path:
    if which in ("last", "latest"):
        from imputer.entity_mf.recurrent.train import _find_latest_numbered_checkpoint

        if which == "last":
            print("Note: checkpoint 'last' uses latest numbered periodic/best save.")
        return _find_latest_numbered_checkpoint(ckpt_dir)
    if which == "best":
        candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")
        return candidates[0]
    p = ckpt_dir / which
    if not p.exists():
        raise FileNotFoundError(f"{which} not found in {ckpt_dir}")
    return p


def _load_checkpoint(model: MarformerBackbone, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model.eval()


def _compute_metrics(result: EntityEvalResults) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for status in ("missing", "observed"):
        preds = result.missing_preds if status == "missing" else result.observed_preds
        trues = result.missing_true if status == "missing" else result.observed_true
        rating_metrics = result.metrics.get(status, {}).get("rating", {})
        log_loss = rating_metrics.get("xent", None)
        accuracy = rating_metrics.get("acc", None)
        if len(preds) == 0:
            out[status] = {"log_loss": log_loss, "accuracy": accuracy, "n": 0}
            continue
        p = np.array(preds) + 1.0
        t = np.array(trues, dtype=float) + 1.0
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        sp_r, sp_p = stats.spearmanr(p, t)
        kt_tau, kt_p = stats.kendalltau(p, t)
        pe_r, pe_p = stats.pearsonr(p, t)
        out[status] = {
            "log_loss": log_loss,
            "accuracy": accuracy,
            "rmse": rmse,
            "n": len(preds),
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "kendall_tau": float(kt_tau),
            "kendall_p": float(kt_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
        }
    return out


def _evaluate_checkpoint(
    run_dir: Path,
    which: str,
    model: RecurrentEntityMarformer,
    eval_vars: List[RankingData],
    device: torch.device,
    max_item: int | None = None,
) -> Dict[str, Any]:
    ckpt_path = _find_checkpoint(run_dir / "checkpoints", which)
    print(f"  checkpoint : {ckpt_path.name}")
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)
    result = evaluate_entity_marformer_split(
        model=model,
        split="test",
        variables=eval_vars,
        types=model.types,
        global_param_dim=model.global_param_dim,
        device=device,
        max_item=max_item,
    )
    metrics = _compute_metrics(result)
    return {
        "checkpoint": ckpt_path.name,
        "checkpoint_type": which,
        "split": "test",
        "max_item": max_item,
        "eval_max_item": max_item,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test evaluation for Recurrent Entity Marformer.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="both",
        choices=["best", "last", "both", "all"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-item",
        type=int,
        default=None,
        help="Chunk eval by item count (default: train_config training.max_item).",
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Evaluate on the full transductive graph (max_item=None), matching recurrence_scaling_eval.",
    )
    parser.add_argument(
        "--num-recurrence",
        type=int,
        default=None,
        help="Override num_recurrence at eval time (default: trained value from train_config.json).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for JSON outputs (default: <run-dir>/TEST_RESULTS).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "TEST_RESULTS"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir  : {run_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Device   : {device}")

    model, eval_vars, train_cfg = _reconstruct(run_dir)
    trained_r = int(model.recurrent_config.num_recurrence)
    if args.num_recurrence is not None:
        model.recurrent_config.num_recurrence = int(args.num_recurrence)
        print(
            f"num_recurrence: {args.num_recurrence} at eval "
            f"(trained={trained_r}, actual_depth={model.effective_depth})"
        )
    if args.full_graph:
        max_item = None
    elif args.max_item is not None:
        max_item = args.max_item
    else:
        max_item = train_cfg.get("max_item")
    print(f"Eval vars: {len(eval_vars)}")
    print(f"max_item : {max_item}")

    if args.checkpoint == "all":
        ckpt_dir = run_dir / "checkpoints"
        best = sorted(ckpt_dir.glob("best-*.ckpt"))
        periodic = sorted(ckpt_dir.glob("periodic-*.ckpt"))
        which_list = [p.name for p in best] + [p.name for p in periodic]
    elif args.checkpoint == "both":
        which_list = ["best", "last"]
    else:
        which_list = [args.checkpoint]

    for which in which_list:
        print(f"\n--- {which} ---")
        try:
            result_dict = _evaluate_checkpoint(
                run_dir=run_dir,
                which=which,
                model=model,
                eval_vars=eval_vars,
                device=device,
                max_item=max_item,
            )
            if args.num_recurrence is not None:
                result_dict["num_recurrence_at_eval"] = int(args.num_recurrence)
                result_dict["trained_num_recurrence"] = trained_r
                result_dict["actual_depth_at_eval"] = int(model.effective_depth)
            result_dict["train_max_item"] = train_cfg.get("max_item")
            result_dict["eval_out_dir"] = str(out_dir)
            stem = Path(which).stem if which.endswith(".ckpt") else which
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2, default=_json_default)
            miss = result_dict.get("missing", {})
            ll = miss.get("log_loss")
            rm = miss.get("rmse")
            sp = miss.get("spearman_r")
            print(
                f"  missing → log_loss={ll:.4f}  rmse={rm:.4f}  spearman={sp:.4f}"
                if ll is not None
                else "  missing → no missing tokens"
            )
            print(f"  saved  → {out_path}")
        except FileNotFoundError as e:
            print(f"  skipped ({e})")


if __name__ == "__main__":
    main()
