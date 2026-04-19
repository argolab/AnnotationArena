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

import math
import random
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment

from imputer.data import DataConverter, RankingData
from imputer.utils import sizes_from_configs

from .config import EntityMarformerConfig
from .types import build_default_domain3_types
from .data import variable_list_to_entity_graph
from .model import EntityMarformer
from .eval import compute_trainable_loss, evaluate_entity_marformer_split, EntityEvalResults
from .masking import MaskingStrategy, build_default_masking_strategy


def _total_grad_l2_norm(module: torch.nn.Module) -> float:
    """L2 norm of all gradients flattened (sqrt of sum of per-parameter grad norms squared)."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq += float(torch.sum(g * g).item())
    return math.sqrt(sq) if sq > 0.0 else 0.0


def _save_json(data: Any, path: Path) -> None:
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


def _new_run_dir(output_root: Path, run_name: str | None = None) -> Path:
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
        bundle: Any,
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
        transductive_valtest_mask: bool = False,
        mask_augmentations: int = 5,
        masked_loss_weight: float = 15.0,
        observed_loss_weight: float = 1.0,
        lr_schedule: str = "none",
        lr_min: float = 1e-5,
        lr_step_epoch: int = 40,
        random_item_chunks: bool = False,
        grad_norm_print_interval: int = 100,
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
        self.transductive_valtest_mask = bool(transductive_valtest_mask)
        self.mask_augmentations = mask_augmentations
        self.masked_loss_weight = masked_loss_weight
        self.observed_loss_weight = observed_loss_weight
        self.lr_schedule = lr_schedule
        self.lr_min = lr_min
        self.lr_step_epoch = lr_step_epoch
        self.random_item_chunks = bool(random_item_chunks)
        self._last_logged_lr = float(learning_rate)
        # Gradient-norm print cadence in on_after_backward (0 = no prints; TB log always).
        self.grad_norm_print_interval = int(grad_norm_print_interval)

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
        self.val_observed: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="val", status="observed"
        )
        self.val_missing: List[RankingData] = converter.create_variables_from_bundle(
            bundle, partition="val", status="missing"
        )
        # Full per-partition variable lists for evaluation (observed + missing).
        self.train_all: List[RankingData] = self.train_observed + self.train_missing
        self.val_all: List[RankingData] = self.val_observed + self.val_missing
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
        self._cached_chunks: list | None = None if self.random_item_chunks else self._build_training_chunks()

    def _print_var_count(
        self,
        graph,
        masked_or_observed: List[RankingData],
        chunk_missing: List[RankingData],
        chunk_fixed: List[RankingData] | None = None,
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
        n_fixed = sum(1 for v in (chunk_fixed or []) if v.is_observed)

        print(
            f"[EntityMarformer] graph tokens: variables={num_var_tokens}, "
            f"entities={num_entity_tokens} | masked={n_masked}, observed={n_observed}, "
            f"fixed={n_fixed}, missing={n_missing}"
        )

    def _build_training_chunks(self, randomize: bool = False) -> list:
        """
        Pre-build one EntityGraph per item chunk.

        The graphs are reused across all training steps; only variable token
        statuses (observed vs masked) are updated in-place each step via
        _refresh_variable_tokens.

        Transductive mode: val_observed AND test_observed are all in the maskable
        pool alongside train_observed.
          - maskable_sources (train + val + test observed): randomly masked each step;
            occupy token indices 0..len(maskable)-1, refreshed by _refresh_variable_tokens.
            Crucially, test_observed is maskable so the model receives direct gradient
            signal about test annotators — masking a test_obs token and predicting it
            from partial context directly simulates the test-time prediction task.
          - fixed_sources: empty in transductive mode (no always-observed annotator).
          - missing_sources (train + val missing): always status=0, no loss.
            test_missing is held out entirely — not in the training graph.
        Graph topology is identical every step, so caching applies here too.
        """
        if self.transductive:
            if self.transductive_valtest_mask:
                # Mask only val/test observed; train observed is always visible as context.
                # MASKING_RATE controls what fraction of val+test observed is masked each step,
                # directly simulating the test-time task (some val/test observed as context,
                # the rest predicted). Train observed is never a prediction target.
                maskable_sources = list(self.val_observed) + list(self.test_observed)
                fixed_sources    = list(self.train_observed)
            else:
                maskable_sources = (list(self.train_observed) + list(self.val_observed)
                                    + list(self.test_observed))
                fixed_sources    = []
            missing_sources  = list(self.train_missing)  + list(self.val_missing)
        else:
            maskable_sources = list(self.train_observed)
            fixed_sources    = []
            missing_sources  = list(self.train_missing)

        # Special handling for transductive + val/test masking with item chunking:
        # ensure every chunk contains maskable (val/test observed) tokens and
        # includes train-observed context items. This avoids many zero-signal
        # chunks when item ids are contiguous by split.
        if (
            self.transductive
            and self.transductive_valtest_mask
            and self.max_item is not None
            and self.max_item > 0
        ):
            maskable_items = sorted({iid for v in maskable_sources for iid in v.item_ids})
            train_items = sorted({iid for v in fixed_sources for iid in v.item_ids})
            if randomize:
                random.shuffle(maskable_items)
                random.shuffle(train_items)
            item_chunks = [
                set(maskable_items[i : i + self.max_item])
                for i in range(0, len(maskable_items), self.max_item)
            ]
            # Add a deterministic train-context slice to each maskable chunk.
            context_cap = min(self.max_item, len(train_items))
            mixed_chunks: List[set] = []
            for ci, item_set in enumerate(item_chunks):
                mixed = set(item_set)
                if context_cap > 0:
                    start = (ci * context_cap) % len(train_items)
                    for j in range(context_cap):
                        mixed.add(train_items[(start + j) % len(train_items)])
                mixed_chunks.append(mixed)
            item_chunks = mixed_chunks if mixed_chunks else [set(maskable_items)]
        else:
            all_items: set = set()
            for var in maskable_sources + fixed_sources + missing_sources:
                all_items.update(var.item_ids)
            all_items_list = sorted(all_items)
            if randomize:
                random.shuffle(all_items_list)
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
            chunk_maskable = [v for v in maskable_sources if all(iid in available_items for iid in v.item_ids)]
            chunk_fixed    = [v for v in fixed_sources    if all(iid in available_items for iid in v.item_ids)]
            chunk_missing  = [v for v in missing_sources  if all(iid in available_items for iid in v.item_ids)]
            # In transductive + val/test mask mode, skip chunks without any
            # maskable tokens to guarantee a learning signal every step.
            if self.transductive and self.transductive_valtest_mask and not chunk_maskable:
                continue
            if not chunk_maskable and not chunk_fixed and not chunk_missing:
                continue
            # maskable tokens must be first so _refresh_variable_tokens indexes them correctly.
            graph = variable_list_to_entity_graph(chunk_maskable + chunk_fixed + chunk_missing, self.types)
            chunks.append(
                {
                    "chunk_observed": chunk_maskable,
                    "chunk_fixed": chunk_fixed,
                    "chunk_missing": chunk_missing,
                    "graph": graph,
                }
            )
        return chunks

    def _compute_fresh_chunks(self) -> list:
        """Build chunk list on-the-fly using the same logic as cached chunks, but with randomized item grouping."""
        return self._build_training_chunks(randomize=True)

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
        if self.lr_schedule == "step":
            gamma = self.lr_min / self.learning_rate if self.learning_rate > 0 else 1.0
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[max(1, int(self.lr_step_epoch))],
                gamma=gamma,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def train_dataloader(self):
        """Dummy dataloader with mask_augmentations entries per epoch.
        Each entry triggers a training_step with fresh random masking."""
        ds = torch.utils.data.TensorDataset(torch.zeros(self.mask_augmentations))
        return torch.utils.data.DataLoader(ds, batch_size=1)

    def on_train_epoch_start(self) -> None:
        """
        Print when step LR schedule changes the optimizer LR.
        Lightning applies epoch schedulers between epochs; this hook reports
        the new LR at the start of the epoch where it becomes active.

        Also resets triplet-rating diagnostic state so EntityMarformer can print
        stats on epochs 0, 10, 20, ... (first training chunk of that epoch).
        """
        model = self.model
        if getattr(model, "use_triplet_rating_base", False):
            model._triplet_diag_epoch = int(self.current_epoch)
            model._triplet_diag_logged_this_epoch = False

        if self.lr_schedule != "step":
            return
        if not hasattr(self, "trainer") or not self.trainer.optimizers:
            return
        current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        if abs(current_lr - self._last_logged_lr) > 1e-15:
            print(
                f"[EntityMarformer] LR changed at epoch {self.current_epoch}: "
                f"{self._last_logged_lr:.6g} -> {current_lr:.6g}"
            )
            self._last_logged_lr = current_lr

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
            chunk_fixed    = chunk_data.get("chunk_fixed", [])
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
                self._print_var_count(graph, masked_or_observed, chunk_missing, chunk_fixed=chunk_fixed)

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

    def on_after_backward(self) -> None:
        """Log total L2 grad norm on `self.model` (helps spot vanishing / dead grads)."""
        gn = _total_grad_l2_norm(self.model)
        self.log("train/grad_norm_l2", gn, prog_bar=False, on_step=True, on_epoch=False)
        interval = getattr(self, "grad_norm_print_interval", 0) or 0
        if interval > 0 and self.global_step % interval == 0:
            print(f"[EntityMarformer] grad_norm_l2(model)={gn:.6g}  global_step={self.global_step}")

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
            # (train_all + val_all), plus a val-only evaluation for reporting.
            combined_vars = self.train_all + self.val_all
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

            # Val-only metrics (used for checkpointing).
            if self.val_all:
                val_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="val",
                    variables=self.val_all,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    val_eval.metrics.get("missing", {}).get("rating", {})
                    if val_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [val_missing] acc={acc_str}  xent={xent_str}")
                if xent_val is not None:
                    self.log("val/missing_ce", xent_val, prog_bar=True, on_epoch=True, on_step=False)
                epoch_metrics["val_eval"] = {
                    "split": val_eval.split,
                    "metrics": val_eval.metrics,
                }
            else:
                print("No val variables to evaluate on")
        else:
            # Non-transductive: evaluate train and val splits separately.
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

            if self.val_all:
                val_eval: EntityEvalResults = evaluate_entity_marformer_split(
                    model=self.model,
                    split="val",
                    variables=self.val_all,
                    types=self.model.types,
                    global_param_dim=self.model.global_param_dim,
                    device=self.device,
                    max_item=self.max_item,
                )
                rating_missing = (
                    val_eval.metrics.get("missing", {}).get("rating", {})
                    if val_eval.metrics
                    else {}
                )
                acc_val = rating_missing.get("acc", None)
                xent_val = rating_missing.get("xent", None)
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                xent_str = f"{xent_val:.4f}" if xent_val is not None else "N/A"
                print(f"  [val_missing] acc={acc_str}  xent={xent_str}")
                if xent_val is not None:
                    self.log("val/missing_ce", xent_val, prog_bar=True, on_epoch=True, on_step=False)
                epoch_metrics["val_eval"] = {
                    "split": val_eval.split,
                    "metrics": val_eval.metrics,
                }
            else:
                print("No val variables to evaluate on")

        self.training_history.append(epoch_metrics)


    def on_train_end(self) -> None:
        """Save training history to run_dir/training_history.json for post-hoc plotting."""
        if self.run_dir is not None and self.training_history:
            try:
                _save_json(self.training_history, Path(self.run_dir) / "training_history.json")
            except Exception as e:
                print(f"Warning: failed to save training history to {self.run_dir}: {e}")


########################################################
# Helper functions for loading bundle and converter
########################################################

def load_bundle_and_converter(data_dir: Path) -> tuple[Any, DataConverter, Dict[str, int]]:
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
    bundle: Any,
    converter: DataConverter,
    sizes: Dict[str, int],
    config: EntityMarformerConfig,
    annotator_reg_weight: float = 0.0,
    llm_input_dist: bool = False,
    item_dropout_rate: float = 1.0,
    annotator_dropout_rate: float = 0.0,
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
        annotator_dropout_rate=annotator_dropout_rate,
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
        "--transductive-valtest-mask",
        action="store_true",
        help="In transductive mode, mask only val/test observed (train observed always "
             "visible as context). MASKING_RATE controls the fraction of val+test "
             "observed masked each step.",
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
        "--annotator-dropout-rate",
        type=float,
        default=0.0,
        help="Probability of dropping annotator deviation embedding during training (0.0 = off; symmetric to item).",
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
        "--use-param-output-head",
        action="store_true",
        help="Predict final params from the last combined hidden state instead of reading them directly from the residual param stream.",
    )
    parser.add_argument(
        "--use-triplet-rating-base",
        action="store_true",
        help="For rating tokens, add learnable_scale * tri_norm to mu_raw: L2-normalize attr/annot/item entity features, "
             "tri_norm = sum_d (ha*hb*hc) with no extra /D scaling (first item id).",
    )
    parser.add_argument(
        "--triplet-rating-tanh",
        action="store_true",
        help="With --use-triplet-rating-base, apply tanh after L2-per-entity normalization (optional extra squashing).",
    )
    parser.add_argument(
        "--triplet-initial-scale",
        type=float,
        default=10.0,
        help="Initial value of learnable triplet prior scale (triplet_rating_logit_scale).",
    )
    parser.add_argument(
        "--triplet-mix-mode",
        type=str,
        default="add",
        choices=["add", "prior_only", "anneal_to_average"],
        help="How to combine transformer mu_raw and triplet prior at rating head.",
    )
    parser.add_argument(
        "--triplet-anneal-start-epoch",
        type=int,
        default=0,
        help="For --triplet-mix-mode anneal_to_average: epoch where annealing starts (prior-only before this).",
    )
    parser.add_argument(
        "--triplet-anneal-end-epoch",
        type=int,
        default=200,
        help="For --triplet-mix-mode anneal_to_average: epoch where final mix weights are reached.",
    )
    parser.add_argument(
        "--triplet-transformer-final-weight",
        type=float,
        default=0.5,
        help="Final transformer weight at/after triplet-anneal-end-epoch (anneal_to_average mode).",
    )
    parser.add_argument(
        "--triplet-prior-final-weight",
        type=float,
        default=0.5,
        help="Final prior weight at/after triplet-anneal-end-epoch (anneal_to_average mode).",
    )
    parser.add_argument(
        "--grad-norm-print-interval",
        type=int,
        default=100,
        help="Print model L2 grad norm every N global_steps after backward (0 = never print; still logs train/grad_norm_l2 to TensorBoard).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="LR schedule: 'none' = constant LR, 'cosine' = CosineAnnealingLR from --lr down to --lr-min, "
             "'step' = single drop to --lr-min at --lr-step-epoch.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Target minimum LR used by cosine schedule and as post-drop LR for step schedule.",
    )
    parser.add_argument(
        "--lr-step-epoch",
        type=int,
        default=40,
        help="Epoch at which to apply one LR drop when --lr-schedule step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (controls init, dropout, masking, data ordering).",
    )
    parser.add_argument(
        "--random-item-chunks",
        action="store_true",
        help="Re-sample item chunk groupings every training step instead of using fixed item chunks.",
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
    config.use_graph_mask        = args.use_graph_mask
    config.use_param_output_head = args.use_param_output_head
    config.use_triplet_rating_base = bool(args.use_triplet_rating_base)
    config.triplet_rating_tanh = bool(args.triplet_rating_tanh)
    config.triplet_initial_scale = float(args.triplet_initial_scale)
    config.triplet_mix_mode = str(args.triplet_mix_mode)
    config.triplet_anneal_start_epoch = int(args.triplet_anneal_start_epoch)
    config.triplet_anneal_end_epoch = int(args.triplet_anneal_end_epoch)
    config.triplet_transformer_final_weight = float(args.triplet_transformer_final_weight)
    config.triplet_prior_final_weight = float(args.triplet_prior_final_weight)
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
        annotator_dropout_rate=args.annotator_dropout_rate,
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
            run_dir = _new_run_dir(output_root, run_name=args.run_name)
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
            "use_graph_mask": config.use_graph_mask,
            "use_param_output_head": config.use_param_output_head,
            "use_triplet_rating_base": config.use_triplet_rating_base,
            "triplet_rating_tanh": config.triplet_rating_tanh,
            "triplet_initial_scale": config.triplet_initial_scale,
            "triplet_mix_mode": config.triplet_mix_mode,
            "triplet_anneal_start_epoch": config.triplet_anneal_start_epoch,
            "triplet_anneal_end_epoch": config.triplet_anneal_end_epoch,
            "triplet_transformer_final_weight": config.triplet_transformer_final_weight,
            "triplet_prior_final_weight": config.triplet_prior_final_weight,
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
            "grad_norm_print_interval": int(args.grad_norm_print_interval),
        },
        "run": {
            "run_dir": str(run_dir),
            "seed": args.seed,
        },
    }
    _save_json(train_config, run_dir / "train_config.json")

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
        transductive_valtest_mask=bool(args.transductive_valtest_mask),
        mask_augmentations=args.mask_augmentations,
        masked_loss_weight=args.masked_loss_weight,
        observed_loss_weight=args.observed_loss_weight,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_step_epoch=args.lr_step_epoch,
        random_item_chunks=bool(args.random_item_chunks),
        grad_norm_print_interval=int(args.grad_norm_print_interval),
    )

    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs")

    checkpoint_best = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:04d}",
        monitor="val/missing_ce",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    checkpoint_periodic = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="periodic-{epoch:04d}",
        every_n_epochs=25,
        save_top_k=-1,
    )

    # Single-GPU training under sbatch: avoid Lightning's SlurmEnvironment, which validates
    # SLURM_NTASKS / --ntasks and can raise if the submit script uses --ntasks (not ntasks-per-node).
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_best, checkpoint_periodic],
        plugins=[LightningEnvironment()],
    )
    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()
