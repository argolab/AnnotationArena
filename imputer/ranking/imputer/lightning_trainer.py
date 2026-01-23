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

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import copy
import random
import json

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping as LSEarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None
    LSEarlyStopping = None
    ModelCheckpoint = None
    TensorBoardLogger = None

from imputer.losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from imputer.data import RankingData
from imputer.eval import EvaluationEngine


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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'train_observed_vars', 'train_missing_vars', 
                                          'test_variables', 'eval_engine', 'converter', 
                                          'build_predictives_fn'])
        
        self.model = model
        self.train_observed_vars = train_observed_vars
        self.train_missing_vars = train_missing_vars
        self.test_variables = test_variables
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
    
    def _apply_training_mask(self, observed_vars: List[RankingData], masking_rate: float) -> List[RankingData]:
        """Return a new list where a random subset of observed vars are marked masked (status=1)."""
        if not observed_vars:
            return []
        num_to_mask = int(len(observed_vars) * masking_rate)
        num_to_mask = max(0, min(len(observed_vars), num_to_mask))
        masked_indices = set(random.sample(list(range(len(observed_vars))), num_to_mask)) if num_to_mask > 0 else set()

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
        """Compute lightweight training metrics over supervised refs (masked+observed)."""
        with torch.no_grad():
            rating_logits = model_output["rating"][0]
            ranking_logits = model_output["ranking"][0]
            n = len(supervised_refs)
            rating_logits = rating_logits[:n]
            ranking_logits = ranking_logits[:n]

            metrics: Dict[str, torch.Tensor] = {}

            # Rating metrics
            rating_all = [i for i, v in enumerate(supervised_refs) if (not v.is_listwise) and (v.rating_value is not None)]
            rating_masked = [i for i, v in enumerate(supervised_refs) if (not v.is_listwise) and v.is_masked and (v.rating_value is not None)]
            rating_observed = [i for i, v in enumerate(supervised_refs) if (not v.is_listwise) and v.is_observed and (v.rating_value is not None)]

            def _rating_metrics(idx: List[int], prefix: str) -> None:
                if not idx:
                    return
                logits = rating_logits[idx]
                targets = torch.tensor([supervised_refs[i].rating_value for i in idx], device=logits.device, dtype=torch.long)
                preds = torch.argmax(logits, dim=-1)
                metrics[f"{prefix}/accuracy"] = (preds == targets).float().mean()
                metrics[f"{prefix}/rmse"] = torch.sqrt(((preds.float() - targets.float()) ** 2).mean())
                probs = torch.softmax(logits, dim=-1)
                metrics[f"{prefix}/pred_entropy"] = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
                metrics[f"{prefix}/pred_max_prob"] = probs.max(dim=-1).values.mean()

            _rating_metrics(rating_all, "train/rating")
            _rating_metrics(rating_masked, "train/rating_masked")
            _rating_metrics(rating_observed, "train/rating_observed")

            # Pairwise ranking metrics (only when we have a 2-way ranking)
            ranking_all = [
                i for i, v in enumerate(supervised_refs)
                if v.is_listwise and (v.ranking_order is not None) and (len(v.ranking_order) == 2)
            ]
            ranking_masked = [
                i for i, v in enumerate(supervised_refs)
                if v.is_listwise and v.is_masked and (v.ranking_order is not None) and (len(v.ranking_order) == 2)
            ]
            ranking_observed = [
                i for i, v in enumerate(supervised_refs)
                if v.is_listwise and v.is_observed and (v.ranking_order is not None) and (len(v.ranking_order) == 2)
            ]

            def _ranking_acc(idx: List[int], prefix: str) -> None:
                if not idx:
                    return
                logits = ranking_logits[idx, :2]
                pred_first_wins = logits[:, 0] > logits[:, 1]
                true_first_wins = torch.tensor(
                    [bool(supervised_refs[i].ranking_order[0] < supervised_refs[i].ranking_order[1]) for i in idx],
                    device=logits.device,
                    dtype=torch.bool,
                )
                metrics[f"{prefix}/accuracy"] = (pred_first_wins == true_first_wins).float().mean()

            _ranking_acc(ranking_all, "train/ranking")
            _ranking_acc(ranking_masked, "train/ranking_masked")
            _ranking_acc(ranking_observed, "train/ranking_observed")

            metrics["train/count_supervised"] = torch.tensor(float(n), device=rating_logits.device)
            metrics["train/count_supervised_rating"] = torch.tensor(float(len(rating_all)), device=rating_logits.device)
            metrics["train/count_supervised_ranking_pairwise"] = torch.tensor(float(len(ranking_all)), device=rating_logits.device)

            return metrics
    
    def training_step(self, batch, batch_idx):
        """Single training step."""
        self.model.train()
        
        # Enable full_dropout for training (if embedding provider supports it)
        embed = getattr(self.model, "embedding_provider", None)
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(True)
        
        # Apply masking to observed
        masked_or_observed = self._apply_training_mask(self.train_observed_vars, self.masking_rate)
        
        # Append missing as-is (status=0)
        batch_list: List[RankingData] = []
        batch_list.extend(masked_or_observed)
        for var in self.train_missing_vars:
            if not var.is_missing and not var.is_masked and not var.is_observed:
                raise ValueError("train_missing_vars contains an entry that is not missing")
            batch_list.append(var)
        
        # Forward
        out = self.model(batch_list)

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
            masked_count = sum(1 for v in masked_or_observed if v.is_masked)
            observed_count = sum(1 for v in masked_or_observed if v.is_observed)
            missing_count = len(self.train_missing_vars)
            total_count = len(batch_list)
            self.log("train/tokens_total", float(total_count), on_step=True, on_epoch=True)
            self.log("train/tokens_supervised", float(len(masked_or_observed)), on_step=True, on_epoch=True)
            self.log("train/tokens_missing", float(missing_count), on_step=True, on_epoch=True)
            self.log("train/tokens_masked", float(masked_count), on_step=True, on_epoch=True)
            self.log("train/tokens_observed", float(observed_count), on_step=True, on_epoch=True)
            if len(masked_or_observed) > 0:
                self.log("train/masked_fraction", float(masked_count) / float(len(masked_or_observed)), on_step=True, on_epoch=True)

            # Prediction quality on supervised tokens (cheap train-time accuracy/RMSE/entropy).
            supervised_metrics = self._compute_supervised_prediction_metrics(out, masked_or_observed)
            for k, v in supervised_metrics.items():
                self.log(k, v, on_step=True, on_epoch=True)
        
        # Compute loss: only over non-missing (observed+masked)
        losses = self._compute_loss_for_batch(out, masked_or_observed)
        total_loss = losses.get('_total_loss_tensor')
        
        # Disable full_dropout after training step
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(False)
        
        # Log metrics
        self.log('train/total_loss', losses.get('total_loss', 0.0), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/rating_loss', losses.get('rating_loss', 0.0), on_step=True, on_epoch=True)
        self.log('train/ranking_loss', losses.get('ranking_loss', 0.0), on_step=True, on_epoch=True)
        for k in [
            "masked_total_loss",
            "observed_total_loss",
            "masked_rating_loss",
            "observed_rating_loss",
            "masked_ranking_loss",
            "observed_ranking_loss",
        ]:
            if k in losses:
                self.log(f"train/{k}", losses[k], on_step=True, on_epoch=True)
        
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
        
        return total_loss

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
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Store training history
        epoch_metrics = {
            'epoch': self.current_epoch,
            'total_loss': self.trainer.callback_metrics.get('train/total_loss_epoch', 0.0),
            'rating_loss': self.trainer.callback_metrics.get('train/rating_loss_epoch', 0.0),
            'ranking_loss': self.trainer.callback_metrics.get('train/ranking_loss_epoch', 0.0),
        }
        self.training_history.append(epoch_metrics)
        
        # Run evaluation if test variables are provided
        if self.test_variables is not None and self.eval_engine is not None and self.converter is not None:
            self._run_evaluation()
    
    def _run_evaluation(self):
        """Run evaluation on test variables."""
        try:
            # Get device from Lightning
            device = next(self.model.parameters()).device
            results = self.eval_engine.evaluate_model(
                model=self.model,
                variables=self.test_variables,
                converter=self.converter,
                device=device
            )
            
            # Log metrics
            self.log('test/total_loss', results.total_loss, on_epoch=True)
            self.log('test/rating_loss', results.rating_loss, on_epoch=True)
            self.log('test/ranking_loss', results.ranking_loss, on_epoch=True)
            
            if results.rating_accuracy is not None:
                self.log('test/rating_accuracy', results.rating_accuracy, on_epoch=True, prog_bar=True)
            if results.ranking_accuracy is not None:
                self.log('test/ranking_accuracy', results.ranking_accuracy, on_epoch=True)
            if results.rating_rmse is not None:
                self.log('test/rating_rmse', results.rating_rmse, on_epoch=True)
            
            # Log breakdown metrics
            if results.observed_metrics:
                self._log_breakdown_metrics('test/observed', results.observed_metrics)
            if results.missing_metrics:
                self._log_breakdown_metrics('test/missing', results.missing_metrics)
            if results.masked_metrics:
                self._log_breakdown_metrics('test/masked', results.masked_metrics)
            
            # Store callback history
            callback_result = {
                'epoch': self.current_epoch,
                'name': 'test_all_evaluation',
                'total_loss': results.total_loss,
                'rating_loss': results.rating_loss,
                'ranking_loss': results.ranking_loss,
                'rating_accuracy': results.rating_accuracy,
                'rating_rmse': results.rating_rmse,
                'ranking_accuracy': results.ranking_accuracy,
                'num_rating_evaluations': results.num_rating_evaluations,
                'num_ranking_evaluations': results.num_ranking_evaluations,
                'masked_metrics': results.masked_metrics,
                'observed_metrics': results.observed_metrics,
                'missing_metrics': results.missing_metrics,
            }
            self.callback_history.append(callback_result)
            
            # Save metrics to file
            if self.run_dir is not None:
                self._save_metrics(callback_result)
                
        except Exception as e:
            print(f"Warning: Evaluation failed at epoch {self.current_epoch}: {e}")
    
    def _log_breakdown_metrics(self, prefix: str, metrics: Dict[str, Any]):
        """Log breakdown metrics for a specific status."""
        if 'rating_loss' in metrics:
            self.log(f'{prefix}/rating_loss', metrics['rating_loss'], on_epoch=True)
        if 'rating_accuracy' in metrics and metrics['rating_accuracy'] is not None:
            self.log(f'{prefix}/rating_accuracy', metrics['rating_accuracy'], on_epoch=True)
        if 'ranking_accuracy' in metrics and metrics['ranking_accuracy'] is not None:
            self.log(f'{prefix}/ranking_accuracy', metrics['ranking_accuracy'], on_epoch=True)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert PyTorch tensors and other non-serializable objects to native Python types."""
        if isinstance(obj, torch.Tensor):
            # Convert tensor to Python scalar or list
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif obj is None:
            return None
        else:
            # Try to convert to native type if possible
            try:
                if isinstance(obj, (int, float, str, bool)):
                    return obj
                # Try to get item() if it has it (like numpy scalars)
                if hasattr(obj, 'item'):
                    return obj.item()
            except:
                pass
            return obj
    
    def _save_metrics(self, metrics_obj: Dict[str, Any]):
        """Save metrics to JSON file."""
        try:
            run_path = Path(self.run_dir)
            epoch = metrics_obj['epoch']
            
            # Convert all tensors to native Python types before saving
            serializable_metrics = self._convert_to_json_serializable(metrics_obj)
            
            # Save epoch-specific metrics
            with open(run_path / f"test_metrics_epoch_{epoch:04d}.json", "w") as f:
                json.dump(serializable_metrics, f, indent=2)
            
            # Update latest test_metrics.json
            with open(run_path / "test_metrics.json", "w") as f:
                json.dump(serializable_metrics, f, indent=2)
            
            # Save training history (also convert to serializable)
            test_history = [self._convert_to_json_serializable(entry) 
                          for entry in self.callback_history 
                          if entry.get('name') == 'test_all_evaluation']
            with open(run_path / "test_training_history.json", "w") as f:
                json.dump(test_history, f, indent=2)
            
            # Save training loss history (also convert to serializable)
            if self.training_history:
                serializable_training_history = self._convert_to_json_serializable(self.training_history)
                with open(run_path / "training_loss_history.json", "w") as f:
                    json.dump(serializable_training_history, f, indent=2)
                    
        except Exception as e:
            print(f"Warning: Failed to save metrics at epoch {self.current_epoch}: {e}")
    
    def on_train_end(self):
        """Called when training ends."""
        # Save final training history
        if self.run_dir is not None:
            try:
                run_path = Path(self.run_dir)
                if self.training_history:
                    serializable_training_history = self._convert_to_json_serializable(self.training_history)
                    with open(run_path / "training_loss_history.json", "w") as f:
                        json.dump(serializable_training_history, f, indent=2)
                if self.callback_history:
                    test_history = [self._convert_to_json_serializable(entry) 
                                  for entry in self.callback_history 
                                  if entry.get('name') == 'test_all_evaluation']
                    with open(run_path / "test_training_history.json", "w") as f:
                        json.dump(test_history, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save final training history: {e}")


def create_lightning_trainer(
    run_dir: Optional[str] = None,
    max_epochs: int = 50,
    early_stopping: bool = False,
    early_stopping_metric: str = "loss",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    checkpoint_every: Optional[int] = 5,
    save_top_k: int = 1,
    monitor_metric: Optional[str] = None,
    gradient_clip_val: Optional[float] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with appropriate callbacks.
    
    Args:
        run_dir: Directory for saving checkpoints and logs
        max_epochs: Maximum number of epochs
        early_stopping: Whether to enable early stopping
        early_stopping_metric: Metric to monitor ("loss" or "accuracy")
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for early stopping
        checkpoint_every: Save checkpoint every N epochs
        save_top_k: Number of best models to keep
        monitor_metric: Metric to monitor for checkpointing (e.g., "test/missing_rating_loss")
        gradient_clip_val: Gradient clipping value (None = no clipping)
        **trainer_kwargs: Additional arguments for pl.Trainer
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning is not installed. Install with: pip install pytorch-lightning")
    
    callbacks = []
    
    # Early stopping callback
    if early_stopping:
        if early_stopping_metric == "loss":
            monitor = "test/missing_rating_loss" if monitor_metric is None else monitor_metric
            mode = "min"
        elif early_stopping_metric == "accuracy":
            monitor = "test/missing_rating_accuracy" if monitor_metric is None else monitor_metric
            mode = "max"
        else:
            raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")
        
        callbacks.append(
            LSEarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode=mode,
                verbose=True
            )
        )
    
    # Model checkpoint callback
    if run_dir is not None and checkpoint_every is not None and int(checkpoint_every) > 0:
        checkpoint_dir = Path(run_dir) / "lightning_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine monitor metric for checkpointing
        if monitor_metric is None:
            if early_stopping:
                monitor = "test/missing_rating_loss" if early_stopping_metric == "loss" else "test/missing_rating_accuracy"
                mode = "min" if early_stopping_metric == "loss" else "max"
            else:
                monitor = "test/total_loss"
                mode = "min"
        else:
            monitor = monitor_metric
            mode = "min" if "loss" in monitor.lower() else "max"
        
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename='checkpoint-{epoch:04d}-{test/total_loss:.4f}',
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
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
