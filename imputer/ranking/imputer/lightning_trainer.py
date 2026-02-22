"""
PyTorch Lightning implementation of the Imputer trainer.

This provides a cleaner, more maintainable training loop with built-in:
- TensorBoard logging
- Checkpoint saving/loading
- Early stopping
- Device management
- Distributed training support (future)

Usage:
    from imputer.lightning_trainer import ImputerLightningModule
    import pytorch_lightning as pl
    
    model = ImputerLightningModule(...)
    trainer = pl.Trainer(max_epochs=50, ...)
    trainer.fit(model)
"""

from typing import List, Dict, Any, Optional, Set
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import copy
import random
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from imputer.losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from imputer.data import RankingData
from imputer.eval import EvaluationEngine
from imputer.metrics import (
    log_instance_metrics_to_tb,
    compute_metrics_for_subset,
    partition_variables_by_status,
    compute_rating_entropy_metrics,
)
from imputer.utils import convert_to_json_serializable


class ImputerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for the ranking imputer.
    
    This simplifies the training loop and provides automatic:
    - TensorBoard logging
    - Checkpoint management
    - Early stopping
    - Device management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_observed_vars: List[RankingData],
        train_missing_vars: List[RankingData],
        test_variables: Optional[List[RankingData]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        masking_rate: float = 0.15,
        masked_loss_weight: float = 8.0,
        observed_loss_weight: float = 1.0,
        mask_augmentations: int = 1,
        decay_observed_weight: bool = False,
        decay_observed_epochs: int = 20,
        eval_engine: Optional[EvaluationEngine] = None,
        converter=None,
        build_predictives_fn=None,
        run_dir: Optional[str] = None,
        early_stopping_metric: str = "loss",
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        log_optimizer_stats: bool = True,
        log_attention_stats: bool = True,
        log_every_n_steps: int = 1,
        log_update_every_n_steps: int = 1,
        use_cosine_schedule: bool = False,
        warmup_steps: int = 100,
        max_epochs: int = 50,
        train_instance_callback=None,
        test_instance_callback=None,
        selective_masking=False,
        max_item=30,
        llm_annotator_id: Optional[int] = None,
        human_observed_rate: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'train_observed_vars', 'train_missing_vars',
                                          'test_variables', 'eval_engine', 'converter',
                                          'build_predictives_fn', 'train_instance_callback', 'test_instance_callback'])
        
        self.model = model
        self.train_observed_vars = train_observed_vars
        self.train_missing_vars = train_missing_vars
        self.test_variables = test_variables
        self.train_instance_callback = train_instance_callback
        self.test_instance_callback = test_instance_callback
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.masking_rate = masking_rate
        self.mask_augmentations = mask_augmentations
        self.decay_observed_weight = decay_observed_weight
        self.decay_observed_epochs = decay_observed_epochs
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        
        self.loss_strategy = DefaultLossStrategy(
            masked_loss_weight=masked_loss_weight,
            observed_loss_weight=observed_loss_weight
        )
        self.initial_observed_weight = observed_loss_weight
        self.initial_masked_weight = masked_loss_weight
        
        self.eval_engine = eval_engine
        self.converter = converter
        self.build_predictives_fn = build_predictives_fn
        self.run_dir = run_dir
        self.early_stopping_metric = early_stopping_metric

        self.log_optimizer_stats = log_optimizer_stats
        self.log_attention_stats = log_attention_stats
        self.log_every_n_steps = int(log_every_n_steps)
        self.log_update_every_n_steps = int(log_update_every_n_steps)

        self._log_optimizer_this_step = False
        self._log_attention_this_step = False
        self._log_update_this_step = False
        self._param_snapshot: dict[str, torch.Tensor] | None = None
        
        # Track training history
        self.training_history = []
        self.callback_history = []

        self.selective_masking = selective_masking
        self.max_item = max_item
        self.llm_annotator_id = llm_annotator_id
        self.human_observed_rate = human_observed_rate

    #########################################################
    # Helper functions for logging and tracking
    #########################################################

    @staticmethod
    def _group_key(param_name: str) -> str:
        parts = param_name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit():
            return f"block_{parts[1]}.{parts[2]}"
        if len(parts) >= 2 and parts[0] == "embedding_provider":
            return f"embedding.{parts[1]}"
        if len(parts) >= 2 and parts[0] == "final_norm":
            return f"final_norm.{parts[1]}"
        return parts[0]

    def _is_log_step(self, every_n_steps: int) -> bool:
        if every_n_steps <= 0:
            return False
        return int(self.global_step) % every_n_steps == 0

    def _current_lr(self) -> float:
        if getattr(self, "trainer", None) is not None and getattr(self.trainer, "optimizers", None):
            opt = self.trainer.optimizers[0]
            if opt.param_groups:
                return float(opt.param_groups[0].get("lr", self.learning_rate))
        return float(self.learning_rate)

    def _compute_param_norms(self) -> dict[str, torch.Tensor]:
        sum_sq: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            group = self._group_key(name)
            t = p.detach()
            sum_sq[group] = sum_sq.get(group, torch.zeros((), device=t.device, dtype=torch.float32)) + t.float().pow(2).sum()
            sum_sq["global"] = sum_sq.get("global", torch.zeros((), device=t.device, dtype=torch.float32)) + t.float().pow(2).sum()
        return {k: torch.sqrt(v) for k, v in sum_sq.items()}

    def _compute_grad_norms(self) -> dict[str, torch.Tensor]:
        sum_sq: dict[str, torch.Tensor] = {}
        max_abs: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            group = self._group_key(name)
            g = p.grad.detach()
            g2 = g.float().pow(2).sum()
            sum_sq[group] = sum_sq.get(group, torch.zeros((), device=g.device, dtype=torch.float32)) + g2
            sum_sq["global"] = sum_sq.get("global", torch.zeros((), device=g.device, dtype=torch.float32)) + g2

            gmax = g.abs().max()
            max_abs[group] = gmax if group not in max_abs else torch.maximum(max_abs[group], gmax)
            max_abs["global"] = gmax if "global" not in max_abs else torch.maximum(max_abs["global"], gmax)

        out: dict[str, torch.Tensor] = {}
        for k, v in sum_sq.items():
            out[f"{k}.l2"] = torch.sqrt(v)
        for k, v in max_abs.items():
            out[f"{k}.linf"] = v
        return out

    def _take_param_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad}

    def _compute_update_norms(self, snapshot: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sum_sq: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            before = snapshot.get(name, None)
            if before is None:
                continue
            group = self._group_key(name)
            delta = (p.detach() - before).float()
            d2 = delta.pow(2).sum()
            sum_sq[group] = sum_sq.get(group, torch.zeros((), device=delta.device, dtype=torch.float32)) + d2
            sum_sq["global"] = sum_sq.get("global", torch.zeros((), device=delta.device, dtype=torch.float32)) + d2
        return {k: torch.sqrt(v) for k, v in sum_sq.items()}

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx: int = 0) -> None:
        """ logging optimizer and attention stats """
        self._log_optimizer_this_step = self.log_optimizer_stats and self._is_log_step(self.log_every_n_steps)
        self._log_attention_this_step = self.log_attention_stats and self._is_log_step(self.log_every_n_steps)
        self._log_update_this_step = self.log_optimizer_stats and self._is_log_step(self.log_update_every_n_steps)

        # Toggle attention stat collection on transformer blocks (if supported).
        blocks = getattr(self.model, "blocks", None)
        if blocks is not None:
            for block in blocks:
                if hasattr(block, "collect_attention_stats"):
                    block.collect_attention_stats = bool(self._log_attention_this_step)

        if self._log_update_this_step:
            self._param_snapshot = self._take_param_snapshot()
        else:
            self._param_snapshot = None

    def on_after_backward(self) -> None:
        """ logging optimizer stats """
        if not self._log_optimizer_this_step:
            return

        self.log("opt/lr", self._current_lr(), on_step=True, on_epoch=False)

        grad_norms = self._compute_grad_norms()
        for k, v in grad_norms.items():
            # Example keys: global.l2, block_0.Q.l2, embedding.attribute_embedding.linf, ...
            self.log(f"opt/grad_norm/{k}", v, on_step=True, on_epoch=False)

        param_norms = self._compute_param_norms()
        for k, v in param_norms.items():
            self.log(f"opt/param_norm/{k}", v, on_step=True, on_epoch=False)

        # Relative grad scale (global).
        if "global.l2" in grad_norms and "global" in param_norms:
            rel = grad_norms["global.l2"] / (param_norms["global"] + 1e-12)
            self.log("opt/grad_to_param_ratio/global", rel, on_step=True, on_epoch=False)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns a dummy dataloader.
        We handle data directly in training_step, but Lightning requires a dataloader.
        The number of batches equals mask_augmentations (one per augmentation per epoch).
        """
        # Create a dummy dataset with mask_augmentations items
        # Each item represents one masking augmentation
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.zeros(self.mask_augmentations, dtype=torch.float32)
        )
        return torch.utils.data.DataLoader(
            dummy_dataset, 
            batch_size=1,
            shuffle=False,  # We handle randomness in training_step
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
    
    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if not self.use_cosine_schedule:
            return optimizer
        
        # Setup warmup + cosine scheduler
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        total_steps = self.max_epochs * self.mask_augmentations
        
        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Cosine annealing phase
            progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step (not epoch)
                "frequency": 1,
            },
        }
    
    #########################################################
    # Helper functions for training
    #########################################################

    def _apply_training_mask(self, observed_vars: List[RankingData], masking_rate: float, available_items: Set[int]) -> List[RankingData]:
        """Return a new list where a random subset of observed vars are marked masked (status=1).
        
        Only includes variables whose items are entirely within the provided available_items set.
        
        Args:
            observed_vars: List of observed variables to process
            masking_rate: Fraction of variables to mask
            available_items: Set of item IDs to include in this batch
        """
        if not observed_vars:
            return []
        
        # Filter to only include variables whose items are all in the available set
        filtered_vars: List[RankingData] = []
        for var in observed_vars:
            if all(item_id in available_items for item_id in var.item_ids):
                filtered_vars.append(var)
        
        if not filtered_vars:
            return []

        out: List[RankingData] = []
        if self.llm_annotator_id is not None:
            # Structured masking: LLM always observed; humans masked except random human_observed_rate fraction
            human_vars = [v for v in filtered_vars if v.annotator_id != self.llm_annotator_id]
            num_human_observed = int(len(human_vars) * self.human_observed_rate)
            human_observed_indices = set(
                random.sample(range(len(human_vars)), num_human_observed)
            ) if num_human_observed > 0 else set()

            human_idx = 0
            for var in filtered_vars:
                if var.annotator_id == self.llm_annotator_id:
                    status = 2
                else:
                    status = 2 if human_idx in human_observed_indices else 1
                    human_idx += 1
                out.append(RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    status=status,
                    instance=var.instance,
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order,
                    rating_dist=var.rating_dist,
                ))
            return out

        # Random masking (default for synthetic data)
        num_to_mask = int(len(filtered_vars) * masking_rate)
        num_to_mask = max(0, min(len(filtered_vars), num_to_mask))
        masked_indices = set(random.sample(list(range(len(filtered_vars))), num_to_mask)) if num_to_mask > 0 else set()

        for idx, var in enumerate(filtered_vars):
            status = 1 if idx in masked_indices else 2  # 1=masked, 2=observed
            out.append(RankingData(
                annotator_id=var.annotator_id,
                attribute_id=var.attribute_id,
                is_listwise=var.is_listwise,
                item_ids=var.item_ids,
                status=status,
                instance=var.instance,
                rating_value=var.rating_value,
                ranking_order=var.ranking_order,
            ))
        return out

    
    def _apply_selective_training_mask(self, observed_vars: List[RankingData], masking_rate: float) -> List[RankingData]:
        """Return a new list where variables are masked by (annotator, item) pairs.
        
        Instead of randomly selecting individual variables, this selects random
        (annotator_id, item_id) pairs and masks ALL variables for each selected pair
        until the target masking count is reached.
        """
        if not observed_vars:
            return []
        
        num_to_mask = int(len(observed_vars) * masking_rate)
        num_to_mask = max(0, min(len(observed_vars), num_to_mask))
        
        if num_to_mask == 0:
            # No masking needed, return all as observed
            return [
                RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    status=2,
                    instance=var.instance,
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order,
                )
                for var in observed_vars
            ]
        
        # Build mapping from (annotator_id, item_id) pairs to variable indices
        pair_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx, var in enumerate(observed_vars):
            # For ratings, item_ids has one item; for rankings, iterate all items
            for item_id in var.item_ids:
                pair = (var.annotator_id, item_id)
                if pair not in pair_to_indices:
                    pair_to_indices[pair] = []
                pair_to_indices[pair].append(idx)
        
        # Randomly select pairs until we reach the target mask count
        all_pairs = list(pair_to_indices.keys())
        random.shuffle(all_pairs)
        
        masked_indices: Set[int] = set()
        for pair in all_pairs:
            if len(masked_indices) >= num_to_mask:
                break
            # Add all indices for this pair
            for idx in pair_to_indices[pair]:
                masked_indices.add(idx)
        
        # Build output list
        out: List[RankingData] = []
        for idx, var in enumerate(observed_vars):
            status = 1 if idx in masked_indices else 2  # 1=masked, 2=observed
            out.append(RankingData(
                annotator_id=var.annotator_id,
                attribute_id=var.attribute_id,
                is_listwise=var.is_listwise,
                item_ids=var.item_ids,
                status=status,
                instance=var.instance,
                rating_value=var.rating_value,
                ranking_order=var.ranking_order,
            ))
        return out

    def _compute_supervised_prediction_metrics(
        self, model_output: Dict[str, torch.Tensor], supervised_refs: List[RankingData]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute lightweight **per-step** training metrics over supervised refs (masked+observed).
        
        Uses shared metric computation logic from imputer.metrics (no extra forward pass).
        For comprehensive epoch-end evaluation with loss, see EvaluationCallback.on_epoch_end().
        """
        with torch.no_grad():
            rating_logits = model_output["rating"][0]
            ranking_logits = model_output["ranking"][0]
            n = len(supervised_refs)
            rating_logits = rating_logits[:n]
            ranking_logits = ranking_logits[:n]
            device = rating_logits.device

            metrics: Dict[str, torch.Tensor] = {}

            # Partition by status using shared logic
            rating_observed, _, rating_masked = partition_variables_by_status(supervised_refs, is_rating=True)
            ranking_observed, _, ranking_masked = partition_variables_by_status(supervised_refs, is_rating=False)
            
            rating_all = rating_observed + rating_masked
            ranking_all = ranking_observed + ranking_masked

            # Compute metrics for each subset using shared logic
            def _compute_and_log_metrics(indices: List[int], prefix: str) -> None:
                if not indices:
                    return
                subset_metrics = compute_metrics_for_subset(
                    rating_logits, ranking_logits, indices, supervised_refs, device, include_entropy=False
                )
                if subset_metrics.get("rating_accuracy") is not None:
                    metrics[f"{prefix}/accuracy"] = torch.tensor(subset_metrics["rating_accuracy"], device=device)
                if subset_metrics.get("rating_rmse") is not None:
                    metrics[f"{prefix}/rmse"] = torch.tensor(subset_metrics["rating_rmse"], device=device)
                if subset_metrics.get("ranking_accuracy") is not None:
                    metrics[f"{prefix}/accuracy"] = torch.tensor(subset_metrics["ranking_accuracy"], device=device)

            # Overall metrics (with entropy for ratings) - step-level
            if rating_all:
                overall_metrics = compute_metrics_for_subset(
                    rating_logits, ranking_logits, rating_all, supervised_refs, device, include_entropy=True
                )
                if overall_metrics.get("rating_accuracy") is not None:
                    metrics["train_step/rating/accuracy"] = torch.tensor(overall_metrics["rating_accuracy"], device=device)
                if overall_metrics.get("rating_rmse") is not None:
                    metrics["train_step/rating/rmse"] = torch.tensor(overall_metrics["rating_rmse"], device=device)
                if overall_metrics.get("pred_entropy") is not None:
                    metrics["train_step/rating/pred_entropy"] = overall_metrics["pred_entropy"]
                if overall_metrics.get("pred_max_prob") is not None:
                    metrics["train_step/rating/pred_max_prob"] = overall_metrics["pred_max_prob"]
            
            if ranking_all:
                overall_metrics = compute_metrics_for_subset(
                    rating_logits, ranking_logits, ranking_all, supervised_refs, device, include_entropy=False
                )
                if overall_metrics.get("ranking_accuracy") is not None:
                    metrics["train_step/ranking/accuracy"] = torch.tensor(overall_metrics["ranking_accuracy"], device=device)

            # Breakdown by status - step-level
            _compute_and_log_metrics(rating_masked, "train_step/rating_masked")
            _compute_and_log_metrics(rating_observed, "train_step/rating_observed")
            _compute_and_log_metrics(ranking_masked, "train_step/ranking_masked")
            _compute_and_log_metrics(ranking_observed, "train_step/ranking_observed")

            # Counts - step-level
            metrics["train_step/count_supervised"] = torch.tensor(float(n), device=device)
            metrics["train_step/count_supervised_rating"] = torch.tensor(float(len(rating_all)), device=device)
            metrics["train_step/count_supervised_ranking_pairwise"] = torch.tensor(float(len(ranking_all)), device=device)

            return metrics

    def _compute_regularizer_loss(self) -> torch.Tensor:
        """L2 regularizer (for logging only) based on current parameters."""
        reg = torch.zeros((), device=next(self.model.parameters()).device)
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            reg = reg + p.float().pow(2).sum()
        return reg * float(self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        """Single training step with item batching."""
        self.model.train()
        
        # Enable full_dropout for training (if embedding provider supports it)
        embed = getattr(self.model, "embedding_provider", None)
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(True)
        
        # Collect all unique items across all observed and missing variables
        all_items: Set[int] = set()
        for var in self.train_observed_vars:
            all_items.update(var.item_ids)
        for var in self.train_missing_vars:
            all_items.update(var.item_ids)
        
        # Create ordered list and split into chunks
        all_items_list = sorted(all_items)
        num_items = len(all_items_list)
        
        if self.max_item is not None and num_items > self.max_item:
            # Split into chunks of max_item size
            item_chunks = []
            for i in range(0, num_items, self.max_item):
                chunk = set(all_items_list[i:i + self.max_item])
                item_chunks.append(chunk)
        else:
            # All items fit in one batch
            item_chunks = [all_items]
        
        # Accumulate loss tensors (connected to computation graph) and counts
        loss_tensors: List[torch.Tensor] = []
        loss_weights: List[int] = []  # Number of supervised tokens per chunk
        
        # For logging (float values)
        total_loss_accum = 0.0
        total_rating_loss_accum = 0.0
        total_ranking_loss_accum = 0.0
        total_supervised_count = 0
        chunk_losses: List[Dict] = []
        
        # For logging (aggregate across chunks)
        all_masked_or_observed: List[RankingData] = []
        all_masked_or_observed_per_chunk: List[List[RankingData]] = []
        all_batch_list: List[RankingData] = []
        all_outputs: List = []
        
        for chunk_idx, available_items in enumerate(item_chunks):
            # Apply masking to observed vars for this chunk
            if self.selective_masking:
                masked_or_observed = self._apply_selective_training_mask(self.train_observed_vars, self.masking_rate, available_items)
            else:
                masked_or_observed = self._apply_training_mask(self.train_observed_vars, self.masking_rate, available_items)
            
            if not masked_or_observed:
                continue
            
            # Build batch list for this chunk
            batch_list: List[RankingData] = []
            batch_list.extend(masked_or_observed)
            
            for var in self.train_missing_vars:
                if not var.is_missing:
                    raise ValueError("train_missing_vars contains an entry that is not missing")
                if all(item_id in available_items for item_id in var.item_ids):
                    batch_list.append(var)
            
            if not batch_list:
                continue
            
            # Forward pass for this chunk
            out = self.model(batch_list, verbose=False)
            
            # Compute loss for this chunk
            losses = self._compute_loss_for_batch(out, masked_or_observed)
            
            # Get the actual loss tensor (connected to computation graph)
            chunk_loss_tensor = losses.get('_total_loss_tensor')
            if chunk_loss_tensor is not None:
                loss_tensors.append(chunk_loss_tensor)
                loss_weights.append(len(masked_or_observed))
            
            chunk_total = losses.get('total_loss', 0.0)
            chunk_rating = losses.get('rating_loss', 0.0)
            chunk_ranking = losses.get('ranking_loss', 0.0)
            chunk_count = len(masked_or_observed)
            
            # Accumulate float values for logging
            total_loss_accum += chunk_total * chunk_count
            total_rating_loss_accum += chunk_rating * chunk_count
            total_ranking_loss_accum += chunk_ranking * chunk_count
            total_supervised_count += chunk_count
            
            # Collect for logging
            all_masked_or_observed.extend(masked_or_observed)
            all_masked_or_observed_per_chunk.append(masked_or_observed)
            all_batch_list.extend(batch_list)
            all_outputs.append(out)
            chunk_losses.append(losses)
        
        # Compute weighted average of loss tensors (maintains gradient connection)
        if loss_tensors and sum(loss_weights) > 0:
            total_weight = sum(loss_weights)
            # Weighted sum: sum(loss_tensor * weight) / total_weight
            total_loss_tensor = sum(
                loss_tensor * (weight / total_weight)
                for loss_tensor, weight in zip(loss_tensors, loss_weights)
            )
        else:
            # Fallback: no valid chunks, return zero loss with grad
            device = next(self.model.parameters()).device
            total_loss_tensor = torch.zeros((), device=device, requires_grad=True)
        
        # Compute float values for logging
        if total_supervised_count > 0:
            final_total_loss = total_loss_accum / total_supervised_count
            final_rating_loss = total_rating_loss_accum / total_supervised_count
            final_ranking_loss = total_ranking_loss_accum / total_supervised_count
        else:
            final_total_loss = 0.0
            final_rating_loss = 0.0
            final_ranking_loss = 0.0
        
        # Logging
        if self._log_attention_this_step:
            layer_stats: list[dict[str, torch.Tensor]] = []
            blocks = getattr(self.model, "blocks", None)
            if blocks is not None:
                for layer_idx, block in enumerate(blocks):
                    stats = getattr(block, "last_attention_stats", None)
                    if not stats:
                        continue
                    layer_stats.append(stats)
                    for k, v in stats.items():
                        self.log(f"attn/layer{layer_idx}/{k}", v, on_step=True, on_epoch=True)
                    if hasattr(block, "param_scale"):
                        self.log(f"model/layer{layer_idx}/param_scale", block.param_scale.detach(), on_step=True, on_epoch=True)

            if layer_stats:
                keys = sorted(layer_stats[0].keys())
                for k in keys:
                    vals = torch.stack([s[k] for s in layer_stats if k in s])
                    self.log(f"attn/{k}", vals.mean(), on_step=True, on_epoch=True)

        if self._is_log_step(self.log_every_n_steps):
            # Token/status accounting
            masked_count = sum(1 for v in all_masked_or_observed if v.is_masked)
            observed_count = sum(1 for v in all_masked_or_observed if v.is_observed)
            missing_count = sum(1 for v in all_batch_list if v.is_missing)
            total_count = len(all_batch_list)
            self.log("train/tokens_total", float(total_count), on_step=True, on_epoch=True)
            self.log("train/tokens_supervised", float(len(all_masked_or_observed)), on_step=True, on_epoch=True)
            self.log("train/tokens_missing", float(missing_count), on_step=True, on_epoch=True)
            self.log("train/tokens_masked", float(masked_count), on_step=True, on_epoch=True)
            self.log("train/tokens_observed", float(observed_count), on_step=True, on_epoch=True)
            self.log("train/num_item_chunks", float(len(item_chunks)), on_step=True, on_epoch=True)
            if len(all_masked_or_observed) > 0:
                self.log("train/masked_fraction", float(masked_count) / float(len(all_masked_or_observed)), on_step=True, on_epoch=True)

            # Prediction quality on supervised tokens - aggregate across chunks
            aggregated_metrics: Dict[str, List[torch.Tensor]] = {}
            for out, masked_or_observed in zip(all_outputs, all_masked_or_observed_per_chunk):
                if not masked_or_observed:
                    continue
                supervised_metrics = self._compute_supervised_prediction_metrics(out, masked_or_observed)
                for k, v in supervised_metrics.items():
                    if k not in aggregated_metrics:
                        aggregated_metrics[k] = []
                    aggregated_metrics[k].append(v)
            
            # Log mean of each metric across chunks
            for k, values in aggregated_metrics.items():
                if values:
                    mean_value = torch.stack(values).mean()
                    self.log(k, mean_value, on_step=True, on_epoch=True)

        # Disable full_dropout after training step
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(False)
        
        # Log objective losses
        self.log('obj/total_loss', final_total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('obj/rating_loss', final_rating_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('obj/ranking_loss', final_ranking_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if self._is_log_step(self.log_every_n_steps):
            reg_loss = self._compute_regularizer_loss()
            base_total = torch.as_tensor(final_total_loss, device=reg_loss.device)
            self.log('obj/regularizer_loss', reg_loss, on_step=True, on_epoch=True)
            self.log('obj/total_loss_with_reg', base_total + reg_loss, on_step=True, on_epoch=True)
        
        # Aggregate and log per-status losses across chunks
        for k in [
            "masked_total_loss",
            "observed_total_loss",
            "masked_rating_loss",
            "observed_rating_loss",
            "masked_ranking_loss",
            "observed_ranking_loss",
        ]:
            values = [losses[k] for losses in chunk_losses if k in losses]
            if values:
                mean_value = sum(values) / len(values)
                self.log(f"obj/{k}", mean_value, on_step=True, on_epoch=True)
        
        # Update loss weights if decay is enabled
        if self.decay_observed_weight:
            current_epoch = self.current_epoch
            if current_epoch < self.decay_observed_epochs:
                current_observed_weight = self.initial_observed_weight * (1.0 - current_epoch / self.decay_observed_epochs)
            else:
                current_observed_weight = 0.0
            self.loss_strategy.update_weights(
                masked_loss_weight=self.initial_masked_weight,
                observed_loss_weight=current_observed_weight
            )
            self.log('train/observed_loss_weight', current_observed_weight, on_epoch=True)
            self.log('train/masked_loss_weight', self.initial_masked_weight, on_epoch=True)
        
        return total_loss_tensor

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not self._log_update_this_step or self._param_snapshot is None:
            return

        update_norms = self._compute_update_norms(self._param_snapshot)
        for k, v in update_norms.items():
            self.log(f"opt/update_norm/{k}", v, on_step=True, on_epoch=False)

        # Relative update scale (global).
        param_norms = self._compute_param_norms()
        if "global" in update_norms and "global" in param_norms:
            rel = update_norms["global"] / (param_norms["global"] + 1e-12)
            self.log("opt/update_to_param_ratio/global", rel, on_step=True, on_epoch=False)

        self._param_snapshot = None
    
    def _compute_loss_for_batch(self, model_output, supervised_refs: List[RankingData]):
        """Compute loss for supervised refs (observed+masked); ignores missing."""
        rating_logits = model_output['rating']
        ranking_logits = model_output['ranking']

        predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
        num_supervised = len(supervised_refs)
        predictions = predictions_full[:num_supervised]
        references = supervised_refs

        losses = self.loss_strategy.compute(predictions, references)

        # Create total loss tensor for backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            device = next(self.model.parameters()).device
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=device)
        
        losses['_total_loss_tensor'] = total_loss_tensor
        return losses
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        masked_rating_loss = self.trainer.callback_metrics.get("obj/masked_rating_loss", 0.0)
        observed_rating_loss = self.trainer.callback_metrics.get("obj/observed_rating_loss", 0.0)
        epoch_metrics = {
            "epoch": self.current_epoch,
            "total_loss": self.trainer.callback_metrics.get("obj/total_loss", 0.0),
            "rating_loss": self.trainer.callback_metrics.get("obj/rating_loss", 0.0),
            "ranking_loss": self.trainer.callback_metrics.get("obj/ranking_loss", 0.0),
            "masked_rating_loss": float(masked_rating_loss),
            "observed_rating_loss": float(observed_rating_loss),
        }
        total_tokens = self.trainer.callback_metrics.get('train/tokens_total', '?')
        masked_tokens = self.trainer.callback_metrics.get('train/tokens_masked', '?')
        observed_tokens = self.trainer.callback_metrics.get('train/tokens_observed', '?')
        print(
            f"[Epoch {self.current_epoch}] "
            f"total={epoch_metrics['total_loss']:.4f}  "
            f"masked_rating={epoch_metrics['masked_rating_loss']:.4f}  "
            f"observed_rating={epoch_metrics['observed_rating_loss']:.4f}  "
            f"| tokens: total={total_tokens} masked={masked_tokens} observed={observed_tokens}"
        )
        self.training_history.append(epoch_metrics)

        test_cb_result = None
        for cb in [self.train_instance_callback, self.test_instance_callback]:
            if cb is None:
                continue
            result = cb.on_epoch_end(self.model, self.current_epoch)
            instance_name = result.get("instance", cb.instance_name)
            log_instance_metrics_to_tb(self, result, instance_name)
            self.callback_history.append(result)
            if cb is self.test_instance_callback:
                test_cb_result = result

        if test_cb_result is not None:
            mm = test_cb_result.get("missing_metrics") or {}
            miss_acc  = mm.get("rating_accuracy")
            miss_loss = mm.get("rating_loss")
            acc_str  = f"{miss_acc:.4f}"  if miss_acc  is not None else "N/A"
            loss_str = f"{miss_loss:.4f}" if miss_loss is not None else "N/A"
            print(f"  [test_missing] acc={acc_str}  xent={loss_str}")

        if self.run_dir is not None:
            self._save_epoch_artifacts()

    
    def _save_epoch_artifacts(self) -> None:
        """Save obj training_loss_history and per-instance training histories."""
        run_path = Path(self.run_dir)
        if self.training_history:
            buf = convert_to_json_serializable(self.training_history)
            with open(run_path / "training_loss_history.json", "w") as f:
                json.dump(buf, f, indent=2)
        for instance_name in ("train_instance", "test_instance"):
            hist = [convert_to_json_serializable(e) for e in self.callback_history if e.get("instance") == instance_name]
            if not hist:
                continue
            with open(run_path / f"{instance_name}_training_history.json", "w") as f:
                json.dump(hist, f, indent=2)

    def on_train_end(self) -> None:
        """Save final training history and instance histories."""
        if self.run_dir is not None:
            self._save_epoch_artifacts()


def create_lightning_trainer(
    run_dir: Optional[str] = None,
    max_epochs: int = 50,
    checkpoint_every: Optional[int] = 5,
    gradient_clip_val: Optional[float] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with ModelCheckpoint (no early stopping).
    
    Args:
        run_dir: Directory for saving checkpoints and logs
        max_epochs: Maximum number of epochs
        checkpoint_every: Save checkpoint every N epochs (None = disabled)
        gradient_clip_val: Gradient clipping value (None = no clipping)
        **trainer_kwargs: Additional arguments for pl.Trainer
    """
    callbacks = []
    
    if run_dir is not None and checkpoint_every is not None and int(checkpoint_every) > 0:
        checkpoint_dir = Path(run_dir) / "lightning_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="checkpoint-{epoch:04d}",
                monitor="test_instance/total/xent_loss/total",
                mode="min",
                save_top_k=0,
                every_n_epochs=int(checkpoint_every),
                save_last=True,
            )
        )
    
    # TensorBoard logger
    loggers = []
    if run_dir is not None:
        log_dir = Path(run_dir) / "lightning_logs"
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs"))
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else True,  # Use default logger if none specified
        enable_progress_bar=True,
        gradient_clip_val=gradient_clip_val,  # None means no clipping
        **trainer_kwargs
    )
    
    return trainer
