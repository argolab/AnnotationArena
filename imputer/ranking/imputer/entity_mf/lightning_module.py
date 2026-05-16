from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import pytorch_lightning as pl

from imputer.data import DataConverter, RankingData

from .backbone import MarformerBackbone
from .data import variable_list_to_entity_graph
from .eval import compute_trainable_loss, evaluate_entity_marformer_split, EntityEvalResults
from .masking import MaskingStrategy, build_default_masking_strategy
from .train_utils import save_json


class EntityMarformerLightningModule(pl.LightningModule):
    """
    Lightning wrapper for any MarformerBackbone model (flat or recurrent).
    """

    def __init__(
        self,
        model: MarformerBackbone,
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
        log_prefix: str = "EntityMarformer",
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
        self.log_prefix = log_prefix

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
        self.train_all: List[RankingData] = self.train_observed + self.train_missing
        self.val_all: List[RankingData] = self.val_observed + self.val_missing
        self.test_all: List[RankingData] = self.test_observed + self.test_missing
        self.masking_strategy: MaskingStrategy = build_default_masking_strategy(
            masking_rate=masking_rate,
            llm_annotator_id=llm_annotator_id,
            human_observed_rate=human_observed_rate,
            always_observed_ids=always_observed_ids,
        )
        self.training_history: List[Dict[str, Any]] = []
        self._cached_chunks: list | None = (
            None if self.random_item_chunks else self._build_training_chunks()
        )

    def _print_var_count(
        self,
        graph,
        masked_or_observed: List[RankingData],
        chunk_missing: List[RankingData],
        chunk_fixed: List[RankingData] | None = None,
    ) -> None:
        num_var_tokens = sum(
            1 for t in graph.tokens if t.type_name in ("rating", "ranking_pairwise")
        )
        num_entity_tokens = len(graph.tokens) - num_var_tokens
        n_masked = sum(1 for v in masked_or_observed if v.is_masked)
        n_observed = sum(1 for v in masked_or_observed if v.is_observed)
        n_missing = sum(1 for v in chunk_missing if v.is_missing)
        n_fixed = sum(1 for v in (chunk_fixed or []) if v.is_observed)
        print(
            f"[{self.log_prefix}] graph tokens: variables={num_var_tokens}, "
            f"entities={num_entity_tokens} | masked={n_masked}, observed={n_observed}, "
            f"fixed={n_fixed}, missing={n_missing}"
        )

    def _build_training_chunks(self, randomize: bool = False) -> list:
        if self.transductive:
            if self.transductive_valtest_mask:
                maskable_sources = list(self.val_observed) + list(self.test_observed)
                fixed_sources = list(self.train_observed)
            else:
                maskable_sources = (
                    list(self.train_observed) + list(self.val_observed) + list(self.test_observed)
                )
                fixed_sources = []
            missing_sources = list(self.train_missing) + list(self.val_missing)
        else:
            maskable_sources = list(self.train_observed)
            fixed_sources = []
            missing_sources = list(self.train_missing)

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
            chunk_maskable = [
                v for v in maskable_sources if all(iid in available_items for iid in v.item_ids)
            ]
            chunk_fixed = [
                v for v in fixed_sources if all(iid in available_items for iid in v.item_ids)
            ]
            chunk_missing = [
                v for v in missing_sources if all(iid in available_items for iid in v.item_ids)
            ]
            if self.transductive and self.transductive_valtest_mask and not chunk_maskable:
                continue
            if not chunk_maskable and not chunk_fixed and not chunk_missing:
                continue
            graph = variable_list_to_entity_graph(
                chunk_maskable + chunk_fixed + chunk_missing, self.types
            )
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
        return self._build_training_chunks(randomize=True)

    @staticmethod
    def _refresh_variable_tokens(graph, masked_or_observed: List[RankingData]) -> None:
        for i, var in enumerate(masked_or_observed):
            tok = graph.tokens[i]
            tok.status = var.status
            tok.raw_data["is_masked"] = var.is_masked

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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
        ds = torch.utils.data.TensorDataset(torch.zeros(self.mask_augmentations))
        return torch.utils.data.DataLoader(ds, batch_size=1)

    def on_train_epoch_start(self) -> None:
        if self.lr_schedule != "step":
            return
        if not hasattr(self, "trainer") or not self.trainer.optimizers:
            return
        current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        if abs(current_lr - self._last_logged_lr) > 1e-15:
            print(
                f"[{self.log_prefix}] LR changed at epoch {self.current_epoch}: "
                f"{self._last_logged_lr:.6g} -> {current_lr:.6g}"
            )
            self._last_logged_lr = current_lr

    def training_step(self, batch, batch_idx):
        device = self.device
        if self._cached_chunks is not None:
            active_chunks = self._cached_chunks
            use_graph_cache = True
        else:
            active_chunks = self._compute_fresh_chunks()
            use_graph_cache = False

        loss_tensors: List[torch.Tensor] = []
        loss_weights: List[int] = []
        masked_ce_accum = 0.0
        masked_ce_count = 0
        observed_ce_accum = 0.0
        observed_ce_count = 0

        for chunk_data in active_chunks:
            chunk_observed = chunk_data["chunk_observed"]
            chunk_fixed = chunk_data.get("chunk_fixed", [])
            chunk_missing = chunk_data["chunk_missing"]
            masked_or_observed = self.masking_strategy.mask(chunk_observed)

            if use_graph_cache:
                graph = chunk_data["graph"]
                self._refresh_variable_tokens(graph, masked_or_observed)
            else:
                train_vars = masked_or_observed + chunk_missing
                graph = variable_list_to_entity_graph(train_vars, self.types)

            if self.current_epoch == 0:
                self._print_var_count(
                    graph, masked_or_observed, chunk_missing, chunk_fixed=chunk_fixed
                )

            params = self.model(graph, device=device)
            trainable_loss, chunk_masked_ce, chunk_observed_ce = compute_trainable_loss(
                params,
                graph,
                self.model.types,
                self.model.global_param_dim,
                device,
                masked_loss_weight=self.masked_loss_weight,
                observed_loss_weight=self.observed_loss_weight,
            )

            reg_loss = torch.zeros((), device=device)
            for type_name, t in self.model.types.items():
                if not t.variation.enabled or t.variation.reg_weight <= 0.0:
                    continue
                table = self.model.deviation_tables.get(type_name, None)
                if table is None:
                    continue
                reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()

            chunk_loss = trainable_loss + reg_loss
            n_masked = sum(
                1
                for t in graph.tokens
                if t.type_name in ("rating", "ranking_pairwise") and t.status == 1
            )
            n_observed_chunk = sum(
                1
                for t in graph.tokens
                if t.type_name in ("rating", "ranking_pairwise") and t.status == 2
            )
            weight = n_masked if n_masked > 0 else 1
            loss_tensors.append(chunk_loss)
            loss_weights.append(weight)
            masked_ce_accum += chunk_masked_ce * n_masked
            masked_ce_count += n_masked
            observed_ce_accum += chunk_observed_ce * n_observed_chunk
            observed_ce_count += n_observed_chunk

            if reg_loss.requires_grad and reg_loss.item() != 0:
                self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True)

        if loss_tensors:
            total_weight = sum(loss_weights)
            loss = sum(t * (w / total_weight) for t, w in zip(loss_tensors, loss_weights))
        else:
            loss = torch.zeros((), device=device, requires_grad=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if masked_ce_count > 0:
            self.log(
                "train/masked_ce",
                masked_ce_accum / masked_ce_count,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        if observed_ce_count > 0:
            self.log(
                "train/observed_ce",
                observed_ce_accum / observed_ce_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics if hasattr(self, "trainer") else {}
        total = metrics.get("train/loss", torch.tensor(0.0, device=self.device))
        total_f = float(total.detach().cpu()) if isinstance(total, torch.Tensor) else float(total)
        print(f"[Epoch {self.current_epoch}] total_train_loss={total_f:.4f}")

        epoch_metrics: Dict[str, Any] = {
            "epoch": int(self.current_epoch),
            "total_loss": total_f,
        }

        if self.transductive:
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
        if self.run_dir is not None and self.training_history:
            try:
                save_json(self.training_history, Path(self.run_dir) / "training_history.json")
            except Exception as e:
                print(f"Warning: failed to save training history to {self.run_dir}: {e}")
