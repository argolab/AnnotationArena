"""
Test evaluation script for Entity Marformer.

Reconstructs model from train_config.json, loads a checkpoint (best, last, or
both), evaluates on the test split using the same forward pass as epoch-end
validation, and saves metrics to TEST_RESULTS/ inside the run directory.

Metrics saved per checkpoint:
  missing / observed:
    log_loss    — mean cross-entropy (from LossBreakdown; handles soft targets)
    rmse        — root-mean-square error on expected-value predictions (1-indexed)
    n           — number of tokens evaluated
    spearman_r  — Spearman rank correlation
    kendall_tau — Kendall tau
    pearson_r   — Pearson correlation

Usage:
    python -m imputer.entity_mf.test --run-dir RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_10
    python -m imputer.entity_mf.test --run-dir ... --checkpoint best
    python -m imputer.entity_mf.test --run-dir ... --checkpoint last
    python -m imputer.entity_mf.test --run-dir ... --checkpoint both   (default)
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
from imputer.utils import sizes_from_configs

from .config import EntityMarformerConfig
from .types import build_default_domain3_types
from .data import variable_list_to_entity_graph
from .model import EntityMarformer
from .eval import evaluate_entity_marformer_split, EntityEvalResults


# ── JSON serialisation ────────────────────────────────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ── Reconstruction ────────────────────────────────────────────────────────────

def _reconstruct(run_dir: Path) -> tuple:
    """
    Read train_config.json and reconstruct:
      model, eval_vars (List[RankingData]), train_cfg dict.
    """
    with open(run_dir / "train_config.json") as f:
        train_config = json.load(f)

    data_dir   = Path(train_config["data"]["data_dir"])
    print(data_dir)
    sizes      = train_config["resolved_sizes"]
    mcfg       = train_config["model"]
    train_cfg  = train_config["training"]

    # ── EntityMarformerConfig ─────────────────────────────────────────────────
    config = EntityMarformerConfig()
    config.embedding_dim        = mcfg["embedding_dim"]
    config.num_layers           = mcfg["num_layers"]
    config.attention_heads      = mcfg["attention_heads"]
    config.d_ff                 = mcfg["d_ff"]
    config.num_ffn_layers       = mcfg["num_ffn_layers"]
    config.dropout              = mcfg["dropout"]
    config.use_per_head_rel     = mcfg["use_per_head_rel"]
    config.use_pointer          = mcfg["use_pointer"]
    config.use_rel_value        = mcfg["use_rel_value"]
    config.use_addone_attn      = mcfg["use_addone_attn"]
    config.type_embedding_init  = mcfg["type_embedding_init"]
    config.use_deviation_norm   = mcfg["use_deviation_norm"]
    config.scale_shared_rel     = mcfg["scale_shared_rel"]
    config.use_graph_mask       = mcfg["use_graph_mask"]
    config.use_param_output_head = mcfg.get("use_param_output_head", False)

    # ── DataConverter + bundle ────────────────────────────────────────────────
    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", 2),
    )
    bundle = converter.load_bundle_data(data_dir / "data_bundle.json")

    # ── Types ─────────────────────────────────────────────────────────────────
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

    # ── Model (num_relationships from train graph) ────────────────────────────
    train_obs  = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_miss = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    graph0 = variable_list_to_entity_graph(train_obs + train_miss, types)
    model = EntityMarformer(
        config=config,
        types=types,
        num_relationships=graph0.num_relationships,
    )

    # ── Evaluation variables ─────────────────────────────────────────────────
    # For transductive runs, include train_observed as extra context at test time
    # while still evaluating only on test_missing. Never include train_missing.
    test_obs  = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_miss = converter.create_variables_from_bundle(bundle, partition="test", status="missing")

    if bool(train_cfg.get("transductive_learning", False)):
        eval_vars = train_obs + test_obs + test_miss + train_miss

    else:
        eval_vars = test_obs + test_miss + train_miss

    return model, eval_vars, train_cfg


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _find_checkpoint(ckpt_dir: Path, which: str) -> Path:
    if which == "last":
        p = ckpt_dir / "last.ckpt"
        if not p.exists():
            raise FileNotFoundError(f"last.ckpt not found in {ckpt_dir}")
        return p
    if which == "best":
        candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")
        return candidates[0]
    # Direct filename (e.g. "periodic-epoch=0024.ckpt")
    p = ckpt_dir / which
    if not p.exists():
        raise FileNotFoundError(f"{which} not found in {ckpt_dir}")
    return p


def _load_checkpoint(model: EntityMarformer, ckpt_path: Path, device: torch.device) -> None:
    """Load a Lightning checkpoint's state dict into a bare EntityMarformer."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # Lightning prefixes all keys with "model." (from self.model = model in LightningModule)
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model.eval()


# ── Metrics computation ───────────────────────────────────────────────────────

def _compute_metrics(result: EntityEvalResults) -> Dict[str, Any]:
    """
    Derive RMSE + correlations from the expected-value predictions stored in
    result.  Log-loss comes from the metrics tree (handles soft targets).
    All values converted to 1-indexed scale before RMSE/correlation.
    """
    out: Dict[str, Any] = {}
    for status in ("missing", "observed"):
        preds = result.missing_preds  if status == "missing" else result.observed_preds
        trues = result.missing_true   if status == "missing" else result.observed_true

        log_loss = result.metrics.get(status, {}).get("rating", {}).get("xent", None)

        if len(preds) == 0:
            out[status] = {"log_loss": log_loss, "n": 0}
            continue

        # 0-indexed internally → convert to 1-indexed for interpretable RMSE scale
        p = np.array(preds) + 1.0
        t = np.array(trues,  dtype=float) + 1.0

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        sp_r,  sp_p  = stats.spearmanr(p, t)
        kt_tau, kt_p = stats.kendalltau(p, t)
        pe_r,  pe_p  = stats.pearsonr(p, t)

        out[status] = {
            "log_loss":    log_loss,
            "rmse":        rmse,
            "n":           len(preds),
            "spearman_r":  float(sp_r),
            "spearman_p":  float(sp_p),
            "kendall_tau": float(kt_tau),
            "kendall_p":   float(kt_p),
            "pearson_r":   float(pe_r),
            "pearson_p":   float(pe_p),
        }
    return out


# ── Per-checkpoint evaluation ─────────────────────────────────────────────────

def _evaluate_checkpoint(
    run_dir: Path,
    which: str,
    model: EntityMarformer,
    test_all: List[RankingData],
    train_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    ckpt_path = _find_checkpoint(run_dir / "checkpoints", which)
    print(f"  checkpoint : {ckpt_path.name}")
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)

    result = evaluate_entity_marformer_split(
        model=model,
        split="test",
        variables=test_all,
        types=model.types,
        global_param_dim=model.global_param_dim,
        device=device,
        max_item=None
    )

    metrics = _compute_metrics(result)
    return {
        "checkpoint":      ckpt_path.name,
        "checkpoint_type": which,
        "split":           "test",
        **metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test evaluation for a trained Entity Marformer run.")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to run directory (contains train_config.json and checkpoints/).",
    )
    parser.add_argument(
        "--checkpoint", default="both", choices=["best", "last", "both", "all"],
        help="Checkpoint(s) to evaluate (default: both). 'all' runs every ckpt except last.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device  = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    out_dir = run_dir / "TEST_RESULTS"
    out_dir.mkdir(exist_ok=True)

    print(f"Run dir  : {run_dir}")
    print(f"Device   : {device}")

    model, eval_vars, train_cfg = _reconstruct(run_dir)
    print(f"Eval vars: {len(eval_vars)}")

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
                test_all=eval_vars,
                train_cfg=train_cfg,
                device=device,
            )
            stem = Path(which).stem if which.endswith(".ckpt") else which
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2, default=_json_default)

            miss = result_dict.get("missing", {})
            ll   = miss.get("log_loss")
            rm   = miss.get("rmse")
            sp   = miss.get("spearman_r")
            print(
                f"  missing → log_loss={ll:.4f}  rmse={rm:.4f}  spearman={sp:.4f}"
                if ll is not None else "  missing → no missing tokens"
            )
            print(f"  saved  → {out_path}")
        except FileNotFoundError as e:
            print(f"  skipped ({e})")


if __name__ == "__main__":
    main()
