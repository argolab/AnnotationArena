"""
Entity Marformer training module.

Usage:
python -m imputer.entity_mf.train   --data-dir path/to/your/bundle_dir   --epochs 50   --lr 1e-4   --weight-decay 0.01   --device cuda   --masking-rate 0.15
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from imputer.data import DataConverter, RankingData
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_json
from imputer.utils import sizes_from_configs

from .config import EntityMarformerConfig
from .types import build_default_domain3_types
from .data import variable_list_to_entity_graph
from .model import EntityMarformer
from .eval import compute_trainable_loss, evaluate_entity_marformer_split, EntityEvalResults
from .masking import MaskingStrategy, build_default_masking_strategy


class EntityMarformerLightningModule(pl.LightningModule):
    """
    Minimal Lightning wrapper around EntityMarformer.

    Each step we 
    1. apply random training masking to observed vars, 
    2. build the graph,
    3. run forward, 
    4. compute per-type loss + deviation reg.
    5. log metrics.
    """

    def __init__(
        self,
        model: EntityMarformer,
        bundle: GroundTruthBundle,
        converter: DataConverter,
        types: Dict[str, Any],
        masking_rate: float = 0.15,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        llm_annotator_id: int | None = None,
        human_observed_rate: float = 0.0,
        always_observed_ids: List[int] | None = None,
        max_item: int | None = None,
        run_dir: Path | None = None,
        transductive: bool = False,
        mask_augmentations: int = 5,
        masked_loss_weight: float = 15.0,
        observed_loss_weight: float = 1.0,
        lr_schedule: str = "none",
        lr_min: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.bundle = bundle
        self.converter = converter
        self.types = types
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_item = max_item
        self.run_dir = run_dir
        self.transductive = bool(transductive)
        self.mask_augmentations = mask_augmentations
        self.masked_loss_weight = masked_loss_weight
        self.observed_loss_weight = observed_loss_weight
        self.lr_schedule = lr_schedule
        self.lr_min = lr_min

        # Build persistent splits from the bundle. Training-time graphs may merge
        # train/test variables when transductive is enabled, but these remain as
        # clean splits for evaluation and bookkeeping.
        self.train_observed: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="train", status="observed"
        )
        self.train_missing: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="train", status="missing"
        )
        self.test_observed: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="test", status="observed"
        )
        self.test_missing: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="test", status="missing"
        )
        # Full per-partition variable lists for evaluation (observed + missing).
        self.train_all: List[RankingData] = self.train_observed + self.train_missing
        self.test_all: List[RankingData] = self.test_observed + self.test_missing
        # Masking strategy can be MCAR or structured (LLM vs human) depending on args.
        self.masking_strategy: MaskingStrategy = build_default_masking_strategy(
            masking_rate=masking_rate,
            llm_annotator_id=llm_annotator_id,
            human_observed_rate=human_observed_rate,
            always_observed_ids=always_observed_ids,
        )
        self.training_history: List[Dict[str, Any]] = []
        # Pre-build per-chunk EntityGraphs for non-transductive mode.
        # Graphs are reused every step; only variable token statuses are updated in-place.
        # edge_mask and K_aug are cached on each graph object after the first forward pass.
        self._cached_chunks: list | None = self._build_training_chunks()

    def _print_var_count(
        self,
        graph,
        masked_or_observed: List[RankingData],
        chunk_missing: List[RankingData],
    ) -> None:
        """
        Sanity-check that the graph contains the expected number of variable tokens
        and print a short summary of:
          - variable vs entity token counts
          - how many variables are observed, masked, missing in this chunk.
        """
        num_var_tokens = sum(
            1 for t in graph.tokens if t.type_name in ("rating", "ranking_pairwise")
        )
        num_entity_tokens = len(graph.tokens) - num_var_tokens

        n_masked = sum(1 for v in masked_or_observed if v.is_masked)
        n_observed = sum(1 for v in masked_or_observed if v.is_observed)
        n_missing = sum(1 for v in chunk_missing if v.is_missing)

        print(
            f"[EntityMarformer] graph tokens: variables={num_var_tokens}, "
            f"entities={num_entity_tokens} | masked={n_masked}, observed={n_observed}, missing={n_missing}"
        )

    def _build_training_chunks(self) -> list | None:
        """
        Pre-build one EntityGraph per item chunk for non-transductive training.

        The graphs are reused across all training steps; only variable token
        statuses (observed vs masked) are updated in-place each step.
        Returns None for transductive mode (graphs include test data, rebuilt fresh).
        """
        if self.transductive:
            return None
        observed_sources = list(self.train_observed)
        missing_sources = list(self.train_missing)
        all_items: set = set()
        for var in observed_sources + missing_sources:
            all_items.update(var.item_ids)
        all_items_list = sorted(all_items)
        num_items = len(all_items_list)
        if self.max_item is not None and num_items > self.max_item:
            item_chunks = [
                set(all_items_list[i : i + self.max_item])
                for i in range(0, num_items, self.max_item)
            ]
        else:
            item_chunks = [all_items]
        chunks = []
        for available_items in item_chunks:
            chunk_observed = [v for v in observed_sources if all(iid in available_items for iid in v.item_ids)]
            chunk_missing  = [v for v in missing_sources  if all(iid in available_items for iid in v.item_ids)]
            if not chunk_observed and not chunk_missing:
                continue
            graph = variable_list_to_entity_graph(chunk_observed + chunk_missing, self.types)
            chunks.append({"chunk_observed": chunk_observed, "chunk_missing": chunk_missing, "graph": graph})
        return chunks

    def _compute_fresh_chunks(self) -> list:
        """Build chunk list on-the-fly for transductive mode (includes test data)."""
        observed_sources = list(self.train_observed) + list(self.test_observed)
        missing_sources  = list(self.train_missing)  + list(self.test_missing)
        all_items: set = set()
        for var in observed_sources + missing_sources:
            all_items.update(var.item_ids)
        all_items_list = sorted(all_items)
        num_items = len(all_items_list)
        if self.max_item is not None and num_items > self.max_item:
            item_chunks = [
                set(all_items_list[i : i + self.max_item])
                for i in range(0, num_items, self.max_item)
            ]
        else:
            item_chunks = [all_items]
        chunks = []
        for available_items in item_chunks:
            chunk_observed = [v for v in observed_sources if all(iid in available_items for iid in v.item_ids)]
            chunk_missing  = [v for v in missing_sources  if all(iid in available_items for iid in v.item_ids)]
            if chunk_observed or chunk_missing:
                chunks.append({"chunk_observed": chunk_observed, "chunk_missing": chunk_missing})
        return chunks

    @staticmethod
    def _refresh_variable_tokens(graph, masked_or_observed: List[RankingData]) -> None:
        """
        Update the first len(masked_or_observed) variable token statuses in-place.

        Only status and is_masked change between steps; all other token fields
        (entity ids, edges, rating_value, etc.) are identical to the template graph.
        """
        for i, var in enumerate(masked_or_observed):
            tok = graph.tokens[i]
            tok.status = var.status
            tok.raw_data["is_masked"] = var.is_masked

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_min
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def train_dataloader(self):
        """Dummy dataloader with mask_augmentations entries per epoch.
        Each entry triggers a training_step with fresh random masking."""
        ds = torch.utils.data.TensorDataset(torch.zeros(self.mask_augmentations))
        return torch.utils.data.DataLoader(ds, batch_size=1)

    def training_step(self, batch, batch_idx):
        """
        Single training step: apply random masking, build graph (full instance), forward,
        then loss breakdown: only masked loss for backprop; observed/missing for metrics.
        """
        device = self.device

        # Use pre-built chunk graphs (non-transductive) or build fresh (transductive).
        if self._cached_chunks is not None:
            active_chunks = self._cached_chunks
            use_graph_cache = True
        else:
            active_chunks = self._compute_fresh_chunks()
            use_graph_cache = False

        # Accumulate loss tensors (for a single scalar loss).
        loss_tensors: List[torch.Tensor] = []
        loss_weights: List[int] = []  # number of masked tokens per chunk
        # Accumulate raw (unweighted) CEs for readable logging.
        masked_ce_accum: float = 0.0
        masked_ce_count: int = 0
        observed_ce_accum: float = 0.0
        observed_ce_count: int = 0

        for chunk_data in active_chunks:
            chunk_observed = chunk_data["chunk_observed"]
            chunk_missing  = chunk_data["chunk_missing"]

            # Apply training mask to observed vars in this chunk.
            masked_or_observed = self.masking_strategy.mask(chunk_observed)

            if use_graph_cache:
                graph = chunk_data["graph"]
                self._refresh_variable_tokens(graph, masked_or_observed)
            else:
                train_vars = masked_or_observed + chunk_missing
                graph = variable_list_to_entity_graph(train_vars, self.types)

            if self.current_epoch == 0:
                self._print_var_count(graph, masked_or_observed, chunk_missing)

            ################## Forward pass ########################
            # print(f"\nINPUT: {train_vars[300]}\n")
            # print(f"OUTPUT: {params[0]}")
            # sys.exit()
            params = self.model(graph, device=device)  # [1, L, P]
            ########################################################

            trainable_loss, chunk_masked_ce, chunk_observed_ce = compute_trainable_loss(
                params, graph, self.model.types, self.model.global_param_dim, device,
                masked_loss_weight=self.masked_loss_weight,
                observed_loss_weight=self.observed_loss_weight,
            )

            # Deviation regularization (per-entity deviations)
            reg_loss = torch.zeros((), device=device)
            for type_name, t in self.model.types.items():
                if not t.variation.enabled or t.variation.reg_weight <= 0.0:
                    continue
                table = self.model.deviation_tables.get(type_name, None)
                if table is None:
                    continue
                reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()

            chunk_loss = trainable_loss + reg_loss
            
            # Weight by number of masked tokens; if none, fall back to 1.
            # We reuse the masked count from the loss aggregation helper.
            n_masked = sum(
                1 for t in graph.tokens
                if t.type_name in ("rating", "ranking_pairwise") and t.status == 1
            )
            n_observed_chunk = sum(
                1 for t in graph.tokens
                if t.type_name in ("rating", "ranking_pairwise") and t.status == 2
            )
            weight = n_masked if n_masked > 0 else 1
            loss_tensors.append(chunk_loss)
            loss_weights.append(weight)
            # Accumulate raw CEs weighted by token counts for averaging across chunks.
            masked_ce_accum += chunk_masked_ce * n_masked
            masked_ce_count += n_masked
            observed_ce_accum += chunk_observed_ce * n_observed_chunk
            observed_ce_count += n_observed_chunk

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
        # Raw unweighted CEs — comparable scale to test xent (~1.3 for random on 4-class).
        if masked_ce_count > 0:
            self.log("train/masked_ce", masked_ce_accum / masked_ce_count, prog_bar=True, on_step=True, on_epoch=True)
        if observed_ce_count > 0:
            self.log("train/observed_ce", observed_ce_accum / observed_ce_count, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Epoch-end summary + evaluation.

        Training_step focuses on optimizing the masked loss; here we run the
        full evaluation path on train/test splits, goal is to get the loss on missing tokens
        imputer behavior.
        """
        metrics = self.trainer.callback_metrics if hasattr(self, "trainer") else {}
        total = metrics.get("train/loss", torch.tensor(0.0, device=self.device))
        total_f = float(total.detach().cpu()) if isinstance(total, torch.Tensor) else float(total)

        print(f"[Epoch {self.current_epoch}] total_train_loss={total_f:.4f}")

        # Record full evaluation results for later plotting (including observed/masked/missing).
        epoch_metrics: Dict[str, Any] = {
            "epoch": int(self.current_epoch),
            "total_loss": total_f,
        }

        if self.transductive:
            # In transductive mode, run a single evaluation on the combined split
            # (train_all + test_all), plus a test-only evaluation for reporting.
            combined_vars = self.train_all + self.test_all
            if combined_vars:
                combined_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="combined",
                    variables=combined_vars,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    combined_eval.metrics.get("missing", {}).get("rating", {})
                    if combined_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [combined_missing] acc={acc_str}  xent={xent_str}")
                epoch_metrics["combined_eval"] = {
                    "split": combined_eval.split,
                    "metrics": combined_eval.metrics,
                }
            else:
                print("No combined variables to evaluate on")

            # Test-only metrics (primary reported numbers).
            if self.test_all:
                test_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="test",
                    variables=self.test_all,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    test_eval.metrics.get("missing", {}).get("rating", {})
                    if test_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [test_missing] acc={acc_str}  xent={xent_str}")
                epoch_metrics["test_eval"] = {
                    "split": test_eval.split,
                    "metrics": test_eval.metrics,
                }
            else:
                print("No test variables to evaluate on")
        else:
            # Non-transductive: evaluate train and test splits separately.
            if self.train_all:
                train_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="train",
                    variables=self.train_all,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    train_eval.metrics.get("missing", {}).get("rating", {})
                    if train_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [train_missing] acc={acc_str}  xent={xent_str}")
                epoch_metrics["train_eval"] = {
                    "split": train_eval.split,
                    "metrics": train_eval.metrics,
                }
            else:
                print("No train variables to evaluate on")

            if self.test_all:
                test_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="test",
                    variables=self.test_all,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    test_eval.metrics.get("missing", {}).get("rating", {})
                    if test_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [test_missing] acc={acc_str}  xent={xent_str}")
                epoch_metrics["test_eval"] = {
                    "split": test_eval.split,
                    "metrics": test_eval.metrics,
                }
            else:
                print("No test variables to evaluate on")

        self.training_history.append(epoch_metrics)


    def on_train_end(self) -> None:
        """Save training history to run_dir/training_history.json for post-hoc plotting."""
        if self.run_dir is not None and self.training_history:
            try:
                save_json(self.training_history, Path(self.run_dir) / "training_history.json")
            except Exception as e:
                print(f"Warning: failed to save training history to {self.run_dir}: {e}")


########################################################
# Helper functions for loading bundle and converter
########################################################

def load_bundle_and_converter(data_dir: Path) -> tuple[GroundTruthBundle, DataConverter, Dict[str, int]]:
    bundle_path = data_dir / "data_bundle.json"
    configs_path = data_dir / "configs.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"data_bundle.json not found in {data_dir}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found in {data_dir}")

    import json
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

    bundle = converter.load_bundle_data(bundle_path)

    sizes["max_rank_size"] = max_rank_size
    return bundle, converter, sizes


def build_entity_marformer_from_bundle(
    bundle: GroundTruthBundle,
    converter: DataConverter,
    sizes: Dict[str, int],
    config: EntityMarformerConfig,
    annotator_reg_weight: float = 0.0,
    llm_input_dist: bool = False,
    item_dropout_rate: float = 1.0,
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
        annotator_reg_weight=annotator_reg_weight,
        llm_input_dist=llm_input_dist,
        item_dropout_rate=item_dropout_rate,
    )

    graph = variable_list_to_entity_graph(train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
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
        "--always-observed-ids",
        type=int,
        nargs="+",
        default=None,
        help="One or more annotator IDs that are always kept observed during training "
             "(e.g. --always-observed-ids 4 5 6 7 8 for SummEval turker slots). "
             "Takes priority over --llm-annotator-id when set.",
    )
    parser.add_argument(
        "--max-item",
        type=int,
        default=None,
        help="Experimental: max number of items per Entity Marformer forward pass (item-chunking).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override EntityMarformerConfig.embedding_dim (total model dim).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override EntityMarformerConfig.num_layers (depth of Entity Marformer).",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=None,
        help="Override EntityMarformerConfig.attention_heads.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="Override EntityMarformerConfig.d_ff (feed-forward hidden dim).",
    )
    parser.add_argument(
        "--num-ffn-layers",
        type=int,
        default=None,
        help="Override EntityMarformerConfig.num_ffn_layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override EntityMarformerConfig.dropout.",
    )
    parser.add_argument(
        "--use-per-head-rel",
        action="store_true",
        default=True,
        help="Per-head relational bias: each head learns its own R-dim relation weights (default: True).",
    )
    parser.add_argument(
        "--no-per-head-rel",
        dest="use_per_head_rel",
        action="store_false",
        help="Use old shared-bias relational design: single shared R-dim bias added to all heads identically.",
    )
    parser.add_argument(
        "--use-pointer",
        action="store_true",
        help="Enable K_aug obs-obs shared-identity pointer bias (like old Marformer).",
    )
    parser.add_argument(
        "--use-rel-value",
        action="store_true",
        help="Enable relation-specific value augmentation V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r].",
    )
    parser.add_argument(
        "--use-addone-attn",
        action="store_true",
        help="Enable add-one attention: attn = exp(s) / (1 + sum(exp(s))), allowing sum < 1.",
    )
    parser.add_argument(
        "--scale-shared-rel",
        action="store_true",
        default=False,
        help="In shared-bias mode (--no-per-head-rel): scale rel_scores by 1/sqrt(head_dim), "
             "matching the joint normalization used in per-head mode.",
    )
    parser.add_argument(
        "--use-graph-mask",
        action="store_true",
        help="Hard graph attention mask: allow attention only where edge_mask or K_aug pointer "
             "exists (+ self-attention). All other pairs are masked to -inf.",
    )
    parser.add_argument(
        "--type-embedding-init",
        type=str,
        default="normal",
        choices=["normal", "scaled_normal", "kaiming"],
        help="Initialization for type centroid embeddings: 'normal' (BERT-style std=0.02), "
             "'scaled_normal' (std=1/sqrt(feature_dim)), 'kaiming' (legacy).",
    )
    parser.add_argument(
        "--overwrite-existing-data",
        action="store_true",
        help="If set and --run-name is used, (re)use that run directory instead of failing when it exists.",
    )
    parser.add_argument(
        "--annotator-reg-weight",
        type=float,
        default=0.0,
        help="L2 regularization weight for annotator deviation embeddings (AnnotatorEntityType).",
    )
    parser.add_argument(
        "--mask-augmentations",
        type=int,
        default=5,
        help="Number of independent masking draws per epoch (training steps per epoch).",
    )
    parser.add_argument(
        "--masked-loss-weight",
        type=float,
        default=15.0,
        help="Weight for masked loss in training objective.",
    )
    parser.add_argument(
        "--observed-loss-weight",
        type=float,
        default=1.0,
        help="Weight for observed loss in training objective.",
    )
    parser.add_argument(
        "--llm-input-dist",
        action="store_true",
        help="Encode observed ratings as log-probability distributions (for soft LLM labels).",
    )
    parser.add_argument(
        "--item-dropout-rate",
        type=float,
        default=1.0,
        help="Probability of dropping item deviation embedding during training (1.0 = always drop).",
    )
    parser.add_argument(
        "--item-reg-weight",
        type=float,
        default=0.0,
        help="L2 regularization weight for item deviation embeddings.",
    )
    parser.add_argument(
        "--attribute-reg-weight",
        type=float,
        default=0.0,
        help="L2 regularization weight for attribute deviation embeddings.",
    )
    parser.add_argument(
        "--use-deviation-norm",
        action="store_true",
        help="Apply LayerNorm to each deviation before adding to its type centroid (bounds deviation scale).",
    )
    parser.add_argument(
        "--use-learned-embedding",
        action="store_true",
        help="Replace fixed-scale build_param with learned Ax+b embedding; feature_dim=model_dim; unembed at top layer.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="LR schedule: 'none' = constant LR, 'cosine' = CosineAnnealingLR from --lr down to --lr-min.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum LR for cosine schedule (eta_min). Only used when --lr-schedule cosine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (controls init, dropout, masking, data ordering).",
    )
    args = parser.parse_args()

    # Seed everything before any model init or data loading.
    pl.seed_everything(args.seed, workers=True)

    data_dir = Path(args.data_dir)

    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    config = EntityMarformerConfig()
    if args.embedding_dim is not None:
        config.embedding_dim = args.embedding_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.attention_heads is not None:
        config.attention_heads = args.attention_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.num_ffn_layers is not None:
        config.num_ffn_layers = args.num_ffn_layers
    if args.dropout is not None:
        config.dropout = args.dropout
    config.use_per_head_rel      = args.use_per_head_rel
    config.use_pointer           = args.use_pointer
    config.use_rel_value         = args.use_rel_value
    config.use_addone_attn       = args.use_addone_attn
    config.type_embedding_init   = args.type_embedding_init
    config.use_deviation_norm    = args.use_deviation_norm
    config.scale_shared_rel      = args.scale_shared_rel
    config.use_learned_embedding = args.use_learned_embedding
    config.use_graph_mask        = args.use_graph_mask
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=config.logit_high,
        annotator_reg_weight=args.annotator_reg_weight,
        item_reg_weight=args.item_reg_weight,
        attribute_reg_weight=args.attribute_reg_weight,
        llm_input_dist=args.llm_input_dist,
        item_dropout_rate=args.item_dropout_rate,
    )
    # Build one graph (train partition) to get num_relationships and init model.
    # EntityMarformerLightningModule will build its own train/test splits from
    # the bundle and converter.
    train_observed_tmp = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_missing_tmp = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    train_all = train_observed_tmp + train_missing_tmp
    graph0 = variable_list_to_entity_graph(train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
        num_relationships=graph0.num_relationships,
    )

    # Create run directory for this Entity Marformer run.
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite_existing_data and args.run_name:
        run_dir = output_root / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            run_dir = new_run_dir(output_root, run_name=args.run_name)
        except FileExistsError as e:
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
            "dropout": config.dropout,
            "use_per_head_rel": config.use_per_head_rel,
            "use_pointer": config.use_pointer,
            "use_rel_value": config.use_rel_value,
            "use_addone_attn": config.use_addone_attn,
            "type_embedding_init": config.type_embedding_init,
            "use_deviation_norm": config.use_deviation_norm,
            "scale_shared_rel": config.scale_shared_rel,
            "use_learned_embedding": config.use_learned_embedding,
            "use_graph_mask": config.use_graph_mask,
            "logit_high": config.logit_high,
            "temperature": config.temperature,
            "global_param_dim": model.global_param_dim,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "masking_rate": args.masking_rate,
            "transductive_learning": bool(args.transductive_learning),
            "llm_annotator_id": args.llm_annotator_id,
            "human_observed_rate": args.human_observed_rate,
            "always_observed_ids": args.always_observed_ids,
            "max_item": args.max_item,
            "annotator_reg_weight": args.annotator_reg_weight,
            "item_reg_weight": args.item_reg_weight,
            "attribute_reg_weight": args.attribute_reg_weight,
            "lr_schedule": args.lr_schedule,
            "lr_min": args.lr_min,
            "mask_augmentations": args.mask_augmentations,
            "masked_loss_weight": args.masked_loss_weight,
            "observed_loss_weight": args.observed_loss_weight,
            "llm_input_dist": args.llm_input_dist,
            "item_dropout_rate": args.item_dropout_rate,
            "device": args.device,
        },
        "run": {
            "run_dir": str(run_dir),
            "seed": args.seed,
        },
    }
    save_json(train_config, run_dir / "train_config.json")

    lightning_module = EntityMarformerLightningModule(
        model=model,
        bundle=bundle,
        converter=converter,
        types=types,
        masking_rate=args.masking_rate,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        llm_annotator_id=args.llm_annotator_id,
        human_observed_rate=args.human_observed_rate,
        always_observed_ids=args.always_observed_ids,
        max_item=args.max_item,
        run_dir=run_dir,
        transductive=bool(args.transductive_learning),
        mask_augmentations=args.mask_augmentations,
        masked_loss_weight=args.masked_loss_weight,
        observed_loss_weight=args.observed_loss_weight,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
    )

    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs")
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=accelerator, devices=1, logger=logger)
    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()

