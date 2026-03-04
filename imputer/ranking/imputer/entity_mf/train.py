"""
Entity Marformer training module.

Usage:
python -m imputer.entity_mf.train   --data-dir path/to/your/bundle_dir   --epochs 50   --lr 1e-4   --weight-decay 0.01   --device cuda   --masking-rate 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from imputer.data import DataConverter, RankingData
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_json
from imputer.utils import sizes_from_configs

from .config import EntityMarformerConfig
from .types import build_default_domain3_types
from .data import bundle_to_entity_graph
from .model import EntityMarformer
from .eval import LossStat, compute_loss_stat, evaluate_entity_marformer_split
from .masking import MaskingStrategy, build_default_masking_strategy


class EntityMarformerLightningModule(pl.LightningModule):
    """
    Minimal Lightning wrapper around EntityMarformer.

    Each step we apply random training masking to observed vars, build the graph,
    run forward, and compute per-type loss + deviation reg.
    """

    def __init__(
        self,
        model: EntityMarformer,
        train_observed: List[RankingData],
        train_missing: List[RankingData],
        bundle: GroundTruthBundle,
        types: Dict[str, Any],
        test_missing: List[RankingData] | None = None,
        masking_rate: float = 0.15,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        llm_annotator_id: int | None = None,
        human_observed_rate: float = 0.0,
        max_item: int | None = None,
        run_dir: Path | None = None,
    ):
        super().__init__()
        self.model = model
        self.train_observed = train_observed
        self.train_missing = train_missing
        self.bundle = bundle
        self.types = types
        self.test_missing = test_missing
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_item = max_item
        self.run_dir = run_dir
        # Masking strategy can be MCAR or structured (LLM vs human) depending on args.
        self.masking_strategy: MaskingStrategy = build_default_masking_strategy(
            masking_rate=masking_rate,
            llm_annotator_id=llm_annotator_id,
            human_observed_rate=human_observed_rate,
        )
        self.training_history: List[Dict[str, Any]] = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_dataloader(self):
        """Dummy dataloader; data lives in train_observed + train_missing, graph built each step."""
        ds = torch.utils.data.TensorDataset(torch.zeros(1))
        return torch.utils.data.DataLoader(ds, batch_size=1)

    def training_step(self, batch, batch_idx):
        """
        Single training step: apply random masking, build graph (full instance), forward,
        then loss breakdown: only masked loss for backprop; observed/missing for metrics.
        """
        device = self.device

        # Collect all unique items across all observed and missing variables
        all_items: set[int] = set()
        for var in self.train_observed:
            all_items.update(var.item_ids)
        for var in self.train_missing:
            all_items.update(var.item_ids)

        all_items_list = sorted(all_items)
        num_items = len(all_items_list)

        # Split into item chunks if max_item is set (mirrors ImputerLightningModule logic).
        if self.max_item is not None and num_items > self.max_item:
            item_chunks: List[set[int]] = []
            for i in range(0, num_items, self.max_item):
                chunk = set(all_items_list[i : i + self.max_item])
                item_chunks.append(chunk)
        else:
            item_chunks = [all_items]

        # Accumulate loss tensors (for a single scalar loss) and statistics for logging.
        loss_tensors: List[torch.Tensor] = []
        loss_weights: List[int] = []  # number of masked tokens per chunk

        masked_loss_sum = 0.0
        observed_loss_sum = 0.0
        missing_loss_sum = 0.0

        # Token/status accounting (based on RankingData statuses, to match Imputer).
        n_masked_total = 0
        n_observed_total = 0
        n_missing_total = 0

        for available_items in item_chunks:
            # Filter variables to those whose items all lie in this chunk.
            chunk_observed: List[RankingData] = []
            for v in self.train_observed:
                if all(item_id in available_items for item_id in v.item_ids):
                    chunk_observed.append(v)

            chunk_missing: List[RankingData] = []
            for v in self.train_missing:
                if all(item_id in available_items for item_id in v.item_ids):
                    chunk_missing.append(v)

            if not chunk_observed and not chunk_missing:
                continue

            # Apply training mask to observed vars in this chunk.
            masked_or_observed = self.masking_strategy.mask(chunk_observed)
            train_vars = masked_or_observed + chunk_missing

            graph = bundle_to_entity_graph(self.bundle, train_vars, self.types)

            expected_var_tokens = len(train_vars)
            num_var_tokens = sum(
                1 for t in graph.tokens if t.type_name in ("rating", "ranking_pairwise")
            )
            assert num_var_tokens == expected_var_tokens, (
                f"Graph must contain all variable tokens: got {num_var_tokens}, expected {expected_var_tokens}"
            )

            params = self.model(graph, device=device)  # [1, L, P]

            loss_stat = compute_loss_stat(
                params, graph, self.model.types, self.model.global_param_dim, device
            ) # gather loss over entities

            # Deviation regularization (per-entity deviations)
            reg_loss = torch.zeros((), device=device)
            for type_name, t in self.model.types.items():
                if not t.variation.enabled or t.variation.reg_weight <= 0.0:
                    continue
                table = self.model.deviation_tables.get(type_name, None)
                if table is None:
                    continue
                reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()

            chunk_loss = loss_stat.trainable_loss + reg_loss

            # Weight by number of masked tokens; if none, fall back to 1.
            weight = loss_stat.n_masked if loss_stat.n_masked > 0 else 1
            loss_tensors.append(chunk_loss)
            loss_weights.append(weight)

            # Aggregate loss stats for logging.
            if loss_stat.n_masked > 0:
                masked_loss_sum += loss_stat.loss_masked * loss_stat.n_masked
            if loss_stat.n_observed > 0:
                observed_loss_sum += loss_stat.loss_observed * loss_stat.n_observed
            if loss_stat.n_missing > 0:
                missing_loss_sum += loss_stat.loss_missing * loss_stat.n_missing

            # Token/status counts based on RankingData (to line up with Imputer logging).
            n_masked_chunk = sum(1 for v in masked_or_observed if v.is_masked)
            n_observed_chunk = sum(1 for v in masked_or_observed if v.is_observed)
            n_missing_chunk = sum(1 for v in chunk_missing if v.is_missing)
            n_masked_total += n_masked_chunk
            n_observed_total += n_observed_chunk
            n_missing_total += n_missing_chunk

            if reg_loss.requires_grad and reg_loss.item() != 0:
                self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True)

        # Combine chunk losses into a single scalar.
        if loss_tensors:
            total_weight = sum(loss_weights)
            loss = sum(t * (w / total_weight) for t, w in zip(loss_tensors, loss_weights))
        else:
            # Fallback: no valid chunks; zero loss with grad.
            loss = torch.zeros((), device=device, requires_grad=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Compute global mean losses for logging.
        loss_masked = (
            masked_loss_sum / n_masked_total if n_masked_total > 0 else 0.0
        )
        loss_observed = (
            observed_loss_sum / n_observed_total if n_observed_total > 0 else 0.0
        )
        loss_missing = (
            missing_loss_sum / n_missing_total if n_missing_total > 0 else 0.0
        )

        self.log("train/trainable_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_masked", loss_masked, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_observed", loss_observed, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_missing", loss_missing, prog_bar=True, on_step=False, on_epoch=True)

        # Token counts for sanity-checking vs. Imputer.
        tokens_total = float(n_masked_total + n_observed_total + n_missing_total)
        tokens_masked = float(n_masked_total)
        tokens_observed = float(n_observed_total)
        tokens_missing = float(n_missing_total)
        self.log("train/tokens_total", tokens_total, on_step=False, on_epoch=True)
        self.log("train/tokens_masked", tokens_masked, on_step=False, on_epoch=True)
        self.log("train/tokens_observed", tokens_observed, on_step=False, on_epoch=True)
        self.log("train/tokens_missing", tokens_missing, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Epoch-end summary print, similar to the standard Marformer Lightning trainer.
        """
        metrics = self.trainer.callback_metrics if hasattr(self, "trainer") else {}
        total = metrics.get("train/loss", torch.tensor(0.0, device=self.device))
        loss_masked = metrics.get("train/loss_masked", torch.tensor(0.0, device=self.device))
        loss_observed = metrics.get("train/loss_observed", torch.tensor(0.0, device=self.device))

        # Convert to floats for pretty printing and history
        total_f = float(total.detach().cpu()) if isinstance(total, torch.Tensor) else float(total)
        masked_f = float(loss_masked.detach().cpu()) if isinstance(loss_masked, torch.Tensor) else float(loss_masked)
        observed_f = float(loss_observed.detach().cpu()) if isinstance(loss_observed, torch.Tensor) else float(loss_observed)

        print(
            f"[Epoch {self.current_epoch}] "
            f"total={total_f:.4f}  "
            f"masked_rating={masked_f:.4f}  "
            f"observed_rating={observed_f:.4f}"
        )

        # Print token counts, similar to Imputer's epoch summary.
        tokens_total = metrics.get("train/tokens_total", "?")
        tokens_masked = metrics.get("train/tokens_masked", "?")
        tokens_observed = metrics.get("train/tokens_observed", "?")
        tokens_missing = metrics.get("train/tokens_missing", "?")
        print(
            f"  | tokens: total={tokens_total} masked={tokens_masked} observed={tokens_observed} missing={tokens_missing}"
        )

        # Record training history for later plotting.
        epoch_metrics = {
            "epoch": int(self.current_epoch),
            "total_loss": total_f,
            "masked_rating_loss": masked_f,
            "observed_rating_loss": observed_f,
        }
        self.training_history.append(epoch_metrics)

        # Optional test-missing evaluation (mirrors Marformer [test_missing] print).
        if self.test_missing:
            try:
                eval_res = evaluate_entity_marformer_split(
                    model=self.model,
                    bundle=self.bundle,
                    variables=self.test_missing,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                )
                acc_str = f"{eval_res.missing_accuracy:.4f}"
                xent_str = f"{eval_res.missing_xent:.4f}"
                print(f"  [test_missing] acc={acc_str}  xent={xent_str}")
            except Exception as e:
                # Avoid breaking training if evaluation fails.
                print(f"  [test_missing] evaluation failed: {e}")

    def on_train_end(self) -> None:
        """Save training history to run_dir/training_history.json for post-hoc plotting."""
        if self.run_dir is not None and self.training_history:
            try:
                save_json(self.training_history, Path(self.run_dir) / "training_history.json")
            except Exception as e:
                print(f"Warning: failed to save training history to {self.run_dir}: {e}")


def load_bundle_and_converter(data_dir: Path) -> tuple[GroundTruthBundle, DataConverter, Dict[str, int]]:
    bundle_path = data_dir / "data_bundle.json"
    configs_path = data_dir / "configs.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"data_bundle.json not found in {data_dir}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found in {data_dir}")

    import json

    with open(bundle_path, "r") as f:
        bundle_dict = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_dict)

    with open(configs_path, "r") as f:
        configs = json.load(f)

    sizes = sizes_from_configs(configs)

    # max_rank_size: use model_config if present (e.g. from run_imputer); else default 2 (pairwise)
    model_cfg = configs.get("model_config") or {}
    max_rank_size = int(model_cfg.get("max_rank_size", 2))

    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=max_rank_size,
    )

    sizes["max_rank_size"] = max_rank_size
    return bundle, converter, sizes


def build_entity_marformer_from_bundle(
    bundle: GroundTruthBundle,
    converter: DataConverter,
    sizes: Dict[str, int],
    config: EntityMarformerConfig,
) -> tuple[EntityMarformer, Any]:
    # Reuse DataConverter to create RankingData variables for train partition.
    train_observed: List[RankingData] = converter.create_variables_from_bundle(
        bundle, partition="train", status="observed"
    )
    train_missing: List[RankingData] = converter.create_variables_from_bundle(
        bundle, partition="train", status="missing"
    )

    # Combine for now; masking behavior can be refined later.
    train_all = train_observed + train_missing

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=converter.max_rank_size,
        logit_high=config.logit_high,
    )

    # Global param dimension: 1 + max(C, R), matching existing design.
    global_param_dim = 1 + max(sizes["num_likert_classes"], converter.max_rank_size)

    graph = bundle_to_entity_graph(bundle, train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
        global_param_dim=global_param_dim,
        num_relationships=graph.num_relationships,
    )
    return model, graph


def main():
    parser = argparse.ArgumentParser(description="Minimal trainer for Entity Marformer (entity_mf).")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--masking-rate", type=float, default=0.15, help="Fraction of observed vars to mask each step (like imputer)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-root",
        type=str,
        default="OUTPUT/ENTITY_MF",
        help="Root directory for Entity Marformer runs (mirrors OUTPUT/IMPUTER for imputer).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name under output-root (default: timestamped).",
    )
    parser.add_argument(
        "--transductive-learning",
        action="store_true",
        help="Include test_observed tokens in training (like run_imputer.py).",
    )
    parser.add_argument(
        "--llm-annotator-id",
        type=int,
        default=None,
        help="0-indexed annotator ID of the LLM (for structured masking on real data).",
    )
    parser.add_argument(
        "--human-observed-rate",
        type=float,
        default=0.0,
        help="Fraction of human annotations to keep observed when LLM annotator is set.",
    )
    parser.add_argument(
        "--max-item",
        type=int,
        default=None,
        help="Experimental: max number of items per Entity Marformer forward pass (item-chunking).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    train_observed = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_missing = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    test_observed = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_missing = converter.create_variables_from_bundle(bundle, partition="test", status="missing")

    config = EntityMarformerConfig()
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=config.logit_high,
    )
    global_param_dim = 1 + max(sizes["num_likert_classes"], converter.max_rank_size)
    # Build one graph to get num_relationships and init model
    train_all = train_observed + train_missing
    graph0 = bundle_to_entity_graph(bundle, train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
        global_param_dim=global_param_dim,
        num_relationships=graph0.num_relationships,
    )

    # Transductive learning: optionally include test_observed in training vars (like run_imputer.py).
    train_observed_for_trainer = train_observed
    if args.transductive_learning:
        print("\033[93mTransductive mode: including test_observed in training.\033[0m")
        print(f"  +{len(test_observed)} test_observed tokens added to train_observed.")
        train_observed_for_trainer = train_observed_for_trainer + test_observed

    # Create run directory for this Entity Marformer run.
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = new_run_dir(output_root, run_name=args.run_name)
    except FileExistsError as e:
        # If the user reuses a run-name, fail loudly to avoid overwriting.
        raise RuntimeError(f"Run directory already exists for Entity Marformer: {e}") from e

    # Save a minimal train configuration snapshot.
    train_config = {
        "data": {
            "data_dir": str(data_dir),
        },
        "resolved_sizes": sizes,
        "model": {
            "embedding_dim": config.embedding_dim,
            "num_layers": config.num_layers,
            "attention_heads": config.attention_heads,
            "d_ff": config.d_ff,
            "num_ffn_layers": config.num_ffn_layers,
            "logit_high": config.logit_high,
            "temperature": config.temperature,
            "global_param_dim": global_param_dim,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "masking_rate": args.masking_rate,
            "transductive_learning": bool(args.transductive_learning),
            "llm_annotator_id": args.llm_annotator_id,
            "human_observed_rate": args.human_observed_rate,
            "max_item": args.max_item,
            "device": args.device,
        },
        "run": {
            "run_dir": str(run_dir),
        },
    }
    save_json(train_config, run_dir / "train_config.json")

    lightning_module = EntityMarformerLightningModule(
        model=model,
        train_observed=train_observed_for_trainer,
        train_missing=train_missing,
        bundle=bundle,
        types=types,
        test_missing=test_missing,
        masking_rate=args.masking_rate,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        llm_annotator_id=args.llm_annotator_id,
        human_observed_rate=args.human_observed_rate,
        max_item=args.max_item,
        run_dir=run_dir,
    )

    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs")
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=accelerator, devices=1, logger=logger)
    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()

