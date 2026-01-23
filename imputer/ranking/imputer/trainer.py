from typing import List, Dict, Any, Optional
import torch
import numpy as np
import torch.optim as optim
import copy
import random
from pathlib import Path

from imputer.losses import DefaultLossStrategy, adapt_batched_logits_to_predictions, TopLayerPredictionResult
from imputer.data import RankingData
from tqdm import tqdm

class EvaluationCallback:
    """Callback for evaluation during training (no masking during eval)."""

    def __init__(self, eval_engine, test_variables, converter, device='cuda', name='EvaluationCallback'):
        self.eval_engine = eval_engine
        self.test_variables = test_variables
        self.converter = converter
        self.device = device
        self.name = name

    def on_epoch_end(self, model, epoch):
        try:
            results = self.eval_engine.evaluate_model(
                model=model,
                variables=self.test_variables,
                converter=self.converter,
                device=self.device
            )
            return {
                'epoch': epoch,
                'name': self.name,
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
        except Exception as e:
            print(f"Warning: Evaluation callback failed at epoch {epoch}: {e}")
            return {'epoch': epoch, 'error': str(e)}


class EarlyStopping:
    """Early stopping utility for training.

    Supports both minimization (e.g., loss) and maximization (e.g., accuracy) modes.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss (lower is better), "max" for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        if mode == "min":
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - self.min_delta
        elif mode == "max":
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.early_stopped = False

    def should_stop(self, current_score: float, model) -> bool:
        """Check if training should stop and save best model state."""
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.epochs_without_improvement = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.early_stopped = True
                return True
            return False

    def restore_best_model(self, model):
        """Restore the best model state."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def calculate_rmse(predictions: List[int], targets: List[int]) -> float:
    """Calculate RMSE for rating predictions (on 1-5 scale)."""
    if len(predictions) == 0:
        return 0.0

    # Convert from 0-4 internal representation to 1-5 rating scale
    pred_ratings = [p + 1 for p in predictions]
    true_ratings = [t + 1 for t in targets]

    mse = np.mean([(p - t)**2 for p, t in zip(pred_ratings, true_ratings)])
    return np.sqrt(mse)


class ImputerTrainer:
    """Trainer that masks a subset of training observed variables and appends missing ones."""

    def __init__(self, model, learning_rate=1e-3, device='cuda', embedding_anchor_reg: float = 0.0, callbacks=None,
                 masked_loss_weight: float = 8.0, observed_loss_weight: float = 1.0, 
                 checkpoint_dir: Optional[str] = None, save_checkpoints: bool = False,
                 model_save_dir: Optional[str] = None, weight_decay: float = 0.0,
                 run_dir: Optional[str] = None, eval_engine=None, test_variables=None, converter=None,
                 build_predictives_fn=None, gradient_clip_val: float = 0.0, 
                 use_cosine_schedule: bool = False, warmup_steps: int = 100, max_epochs: int = 50):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_strategy = DefaultLossStrategy(masked_loss_weight=masked_loss_weight,
                                               observed_loss_weight=observed_loss_weight)
        # Callback system
        self.callbacks = callbacks or []
        
        # Gradient clipping
        self.gradient_clip_val = gradient_clip_val
        
        # Learning rate scheduler
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.scheduler = None
        self.current_step = 0
        
        # Checkpoint saving configuration
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        if self.save_checkpoints and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided when save_checkpoints=True")
        
        # Model saving configuration
        self.model_save_dir = model_save_dir
        
        # History and predictives saving configuration
        self.run_dir = run_dir
        self.eval_engine = eval_engine
        self.test_variables = test_variables
        self.converter = converter
        self.build_predictives_fn = build_predictives_fn

    def register_callback(self, callback):
        """Register an evaluation callback."""
        self.callbacks.append(callback)
    
    def _setup_scheduler(self, total_steps: int):
        """Setup warmup + cosine scheduler."""
        if not self.use_cosine_schedule:
            return
        
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Cosine annealing phase
            progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(f"Setup warmup+cosine scheduler: warmup_steps={self.warmup_steps}, total_steps={total_steps}")
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract model configuration for checkpoint saving."""
        # Get basic model dimensions
        config = {
            'num_attributes': self.model.num_attributes,
            'num_annotators': self.model.num_annotators,
            'num_items': self.model.num_items,
            'num_likert_classes': self.model.num_likert_classes,
            'max_rank_size': self.model.max_rank_size,
            'encoder_layers_num': len(self.model.blocks),
            'embedding_dim': self.model.embedding_dim,
            'embedding_type': self.model.embedding_type,
            'device': str(self.device)
        }
        
        # Get attention heads - fail fast if missing
        config['attention_heads'] = self.model.blocks[0].attention_heads
            
        # Get dropout - fail fast if missing
        config['dropout'] = self.model.blocks[0].dropout_1.p
            
        # Debug: print what we're saving
        print(f"Saving model_config: {config}")
            
        return config
    
    def _save_checkpoint(self, epoch: int, loss_dict: Dict[str, float]):
        """Save model checkpoint if checkpoint saving is enabled."""
        if not self.save_checkpoints:
            return
            
        import os
        import json
        from pathlib import Path
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_dict": loss_dict,
            "model_config": self._extract_model_config(),
        }, checkpoint_file)
        
        # Also save as latest checkpoint
        latest_file = checkpoint_path / "checkpoint_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_dict": loss_dict,
            "model_config": self._extract_model_config(),
        }, latest_file)
        
        # Save predictives if function is provided
        if self.build_predictives_fn is not None and self.test_variables is not None:
            try:
                predictives = self.build_predictives_fn(self.model, self.test_variables)
                predictives_file = checkpoint_path / f"predictives_epoch_{epoch:04d}.json"
                with open(predictives_file, "w") as f:
                    json.dump(predictives, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save predictives at epoch {epoch}: {e}")
    
    def _save_model(self, epoch: int, suffix: str = ""):
        """Save model.pt file (final model format) if model saving is enabled."""
        if self.model_save_dir is None:
            return
        
        from pathlib import Path
        
        # Create model save directory if it doesn't exist
        model_path = Path(self.model_save_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model in the same format as final model.pt
        if suffix:
            model_file = model_path / f"model_epoch_{epoch:04d}{suffix}.pt"
        else:
            model_file = model_path / f"model_epoch_{epoch:04d}.pt"
        
        torch.save({
            "state_dict": self.model.state_dict(),
            "model_config": self._extract_model_config(),
            "epoch": epoch
        }, model_file)
        
        print(f"Saved model to {model_file}")

    def _call_epoch_end_callbacks(self, epoch):
        """Call all registered callbacks at epoch end."""
        callback_results = []
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                try:
                    result = callback.on_epoch_end(self.model, epoch)
                    callback_results.append(result)
                except Exception as e:
                    print(f"Warning: Callback failed at epoch {epoch}: {e}")
                    callback_results.append({'epoch': epoch, 'error': str(e)})
        return callback_results
    
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

    def train_step(self,
                   train_observed_vars: List[RankingData],
                   train_missing_vars: List[RankingData],
                   masking_rate: float) -> Dict[str, float]:
        """Single training step: mask subset of observed, append missing, compute loss on non-missing only."""
        self.model.train()
        self.optimizer.zero_grad()

        # Enable full_dropout for training (if embedding provider supports it)
        embed = getattr(self.model, "embedding_provider", None)
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(True)

        # Validate inputs
        if train_observed_vars is None or train_missing_vars is None:
            raise ValueError("train_observed_vars and train_missing_vars must be provided")

        # Apply masking to observed
        masked_or_observed = self._apply_training_mask(train_observed_vars, masking_rate)
        # print(f"[DEBUG] Number of masked or observed training variables: {len(masked_or_observed)}")

        # Append missing as-is (status=0)
        batch_list: List[RankingData] = []
        batch_list.extend(masked_or_observed)
        for var in train_missing_vars:
            if not var.is_missing and not var.is_masked and not var.is_observed:
                # Defensive: enforce valid status
                raise ValueError("train_missing_vars contains an entry that is not missing")
            batch_list.append(var)

        # print(f"[DEBUG] Number of training variables in batch (now include missing variables): {len(batch_list)}")

        # Forward
        out = self.model(batch_list)

        # Compute loss: only over non-missing (observed+masked)
        total_losses = self._compute_loss_for_batch(out, masked_or_observed)
        total_loss_tensor = total_losses.get('_total_loss_tensor')

        if total_loss_tensor is not None:
            # Snapshot params for delta tracking
            embed = getattr(self.model, "embedding_provider", None)
            attr_before = None
            anno_before = None
            params_before = None
            if embed is not None:
                attr_before = embed.attribute_embedding.detach().clone()
                anno_before = embed.annotator_embedding_learnable.detach().clone()
            params_before = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if p.requires_grad
            }

            total_loss_tensor.backward()
            
            # Apply gradient clipping if enabled
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            # Debug: gradient norms for embedding params to verify backprop flow
            try:
                if embed is not None:
                    attr_grad = getattr(embed.attribute_embedding, "grad", None)
                    anno_grad = getattr(embed.annotator_embedding_learnable, "grad", None)
                    def _grad_norm(g):
                        return float(g.norm().item()) if g is not None else None
                    print(
                        f"[DEBUG] grad_norms: attribute={_grad_norm(attr_grad)}, "
                        f"annotator={_grad_norm(anno_grad)}"
                    )
            except Exception as e:
                print(f"[DEBUG] grad_norms: failed to read embedding grads: {e}")
            
            self.optimizer.step()
            
            # Step scheduler if enabled
            if self.scheduler is not None:
                self.scheduler.step()
                self.current_step += 1

            # Debug: parameter delta norms after optimizer step
            try:
                if embed is not None and attr_before is not None and anno_before is not None:
                    attr_delta = embed.attribute_embedding.detach() - attr_before
                    anno_delta = embed.annotator_embedding_learnable.detach() - anno_before
                    attr_norm = float(embed.attribute_embedding.detach().norm().item())
                    anno_norm = float(embed.annotator_embedding_learnable.detach().norm().item())
                    print(
                        f"[DEBUG] embed_delta_norms: attribute={float(attr_delta.norm().item())}, "
                        f"annotator={float(anno_delta.norm().item())}"
                    )
                    print(
                        f"[DEBUG] embed_norms: attribute={attr_norm}, annotator={anno_norm}"
                    )
                    attr_rel = float(attr_delta.norm().item()) / attr_norm if attr_norm > 0 else None
                    anno_rel = float(anno_delta.norm().item()) / anno_norm if anno_norm > 0 else None
                    print(
                        f"[DEBUG] embed_rel_update: attribute={attr_rel}, annotator={anno_rel}"
                    )
            except Exception as e:
                print(f"[DEBUG] embed_delta_norms: failed to compute deltas: {e}")

            # Debug: full parameter delta norms and relative updates (all trainable params)
            try:
                if params_before is not None:
                    for name, p in self.model.named_parameters():
                        if not p.requires_grad:
                            continue
                        before = params_before.get(name)
                        if before is None:
                            continue
                        delta = p.detach() - before
                        delta_norm = float(delta.norm().item())
                        param_norm = float(p.detach().norm().item())
                        rel_update = (delta_norm / param_norm) if param_norm > 0 else None
                        print(f"[DEBUG] param_delta_norm: {name}={delta_norm}")
                        print(f"[DEBUG] param_rel_update: {name}={rel_update}")
            except Exception as e:
                print(f"[DEBUG] param_delta_norm: failed to compute deltas: {e}")

        # Disable full_dropout after training step
        if embed is not None and hasattr(embed, "set_full_dropout"):
            embed.set_full_dropout(False)

        return {k: v for k, v in total_losses.items() if not k.startswith('_')}

    def _compute_loss_for_batch(self, model_output, supervised_refs: List[RankingData]):
        """Compute loss for supervised refs (observed+masked); ignores missing."""
        rating_logits = model_output['rating']
        ranking_logits = model_output['ranking']

        predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
        # Only take predictions corresponding to supervised refs (the first segment of the batch)
        num_supervised = len(supervised_refs)
        predictions = predictions_full[:num_supervised]
        references = supervised_refs

        losses = self.loss_strategy.compute(predictions, references)

        '''# Embedding anchor regularization: keep embeddings close to their random initialization
        reg_scaled = torch.tensor(0.0, device=self.device)
        if self.embedding_anchor_reg > 0.0 and self._embedding_initial_params:
            reg = torch.tensor(0.0, device=self.device)
            for name, p in self.model.named_parameters():
                if p.requires_grad and self._is_embedding_param_name(name):
                    init = self._embedding_initial_params.get(name)
                    if init is not None:
                        reg = reg + (p - init).pow(2).sum()
            reg_scaled = self.embedding_anchor_reg * reg
            # Log metric (float only in returned dict)
            losses['embedding_reg'] = float(reg_scaled.detach().item())
            # Ensure total_loss reflects the regularizer for logging
            if 'total_loss' in losses:
                losses['total_loss'] = float(losses['total_loss'] + losses['embedding_reg'])'''

        # Create total loss tensor for backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=self.device)
        # Add regularization term to tensor used for backprop
        total_loss_tensor = total_loss_tensor #+ reg_scaled

        # Store tensor for backward pass
        losses['_total_loss_tensor'] = total_loss_tensor

        return losses

    def train(self,
              train_observed_vars: List[RankingData],
              train_missing_vars: List[RankingData],
              masking_rate: float,
              epochs: int = 10,
              call_callbacks_every: int = 1,
              save_checkpoints_every: int = 10,
              save_model_every: Optional[int] = None,
              save_best_model: bool = False,
              verbose: bool = True,
              mask_augmentations: int = 1,
              early_stopping: Optional[EarlyStopping] = None,
              early_stopping_metric: str = "loss",
              decay_observed_weight: bool = False,
              decay_observed_epochs: int = 20):
        """Simple training loop using the new API.

        Args:
            mask_augmentations: Number of different masking patterns to generate per epoch.
                               If > 1, creates data augmentation by training on multiple random
                               masking patterns per epoch. Default: 1 (no augmentation).
            early_stopping: EarlyStopping object for early termination based on validation metrics.
            early_stopping_metric: Metric to monitor: "loss" (rating_loss) or "accuracy" (rating_accuracy).
            decay_observed_weight: If True, linearly decay observed_loss_weight from initial value to 0.
            decay_observed_epochs: Number of epochs over which to decay observed weight (default: 20).
        """
        training_history = []
        callback_history = []

        # Setup scheduler if enabled
        total_steps = epochs * mask_augmentations
        if self.use_cosine_schedule:
            self._setup_scheduler(total_steps)
        
        # Store initial weights for decay schedule
        initial_observed_weight = self.loss_strategy.observed_loss_weight
        initial_masked_weight = self.loss_strategy.masked_loss_weight

        for epoch in tqdm(range(epochs)):
            # Update loss weights if decay is enabled
            if decay_observed_weight:
                # Linear decay from initial_observed_weight to 0 over decay_observed_epochs
                if epoch < decay_observed_epochs:
                    current_observed_weight = initial_observed_weight * (1.0 - epoch / decay_observed_epochs)
                else:
                    current_observed_weight = 0.0

                # Update the loss strategy weights
                self.loss_strategy.update_weights(
                    masked_loss_weight=initial_masked_weight,
                    observed_loss_weight=current_observed_weight
                )

                if verbose and epoch % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch}: observed_weight={current_observed_weight:.4f}, masked_weight={initial_masked_weight:.4f}")
            # Train with multiple masking patterns per epoch (data augmentation)
            epoch_losses = []
            for aug_idx in range(mask_augmentations):
                # Each augmentation generates a fresh random masking pattern
                loss_dict = self.train_step(train_observed_vars, train_missing_vars, masking_rate)
                epoch_losses.append(loss_dict)

            # Average losses across augmentations for logging
            if mask_augmentations > 1:
                loss_dict = {
                    key: sum(d[key] for d in epoch_losses) / mask_augmentations
                    for key in epoch_losses[0].keys()
                }
            else:
                loss_dict = epoch_losses[0]

            training_history.append({'epoch': epoch, **loss_dict})

            # Save checkpoint if enabled and at specified intervals
            if self.save_checkpoints and (epoch + 1) % save_checkpoints_every == 0:
                self._save_checkpoint(epoch, loss_dict)
            
            # Save model.pt if enabled and at specified intervals
            if save_model_every is not None and (epoch + 1) % save_model_every == 0:
                self._save_model(epoch + 1)

            if (epoch + 1) % call_callbacks_every == 0:
                callback_results = self._call_epoch_end_callbacks(epoch)
                if callback_results:
                    print("Callback results:")
                    import json
                    print(json.dumps(callback_results, indent=2, sort_keys=True))
                    callback_history.extend(callback_results)
                    
                    # Save training history and test metrics at every epoch
                    if self.run_dir is not None:
                        try:
                            from pathlib import Path
                            run_path = Path(self.run_dir)
                            
                            # Separate test and train callback results
                            test_history = [entry for entry in callback_history if entry.get('name') == 'test_all_evaluation']
                            train_history = [entry for entry in callback_history if entry.get('name') == 'train_all_evaluation']
                            
                            # Save test history
                            if test_history:
                                with open(run_path / "test_training_history.json", "w") as f:
                                    json.dump(test_history, f, indent=2)
                            
                            # Save train history
                            if train_history:
                                with open(run_path / "train_training_history.json", "w") as f:
                                    json.dump(train_history, f, indent=2)
                            
                            # Save test metrics from latest test evaluation
                            if test_history and self.eval_engine is not None and self.test_variables is not None:
                                try:
                                    # Re-evaluate to get full metrics
                                    results = self.eval_engine.evaluate_model(
                                        model=self.model,
                                        variables=self.test_variables,
                                        converter=self.converter,
                                        device=self.device
                                    )
                                    metrics_obj = {
                                        "epoch": epoch,
                                        "total_loss": results.total_loss,
                                        "rating_loss": results.rating_loss,
                                        "ranking_loss": results.ranking_loss,
                                        "num_rating_evaluations": results.num_rating_evaluations,
                                        "num_ranking_evaluations": results.num_ranking_evaluations,
                                        "observed_metrics": results.observed_metrics,
                                        "missing_metrics": results.missing_metrics,
                                        "masked_metrics": results.masked_metrics,
                                    }
                                    # Save as epoch-specific metrics file
                                    with open(run_path / f"test_metrics_epoch_{epoch:04d}.json", "w") as f:
                                        json.dump(metrics_obj, f, indent=2)
                                    # Also update latest test_metrics.json
                                    with open(run_path / "test_metrics.json", "w") as f:
                                        json.dump(metrics_obj, f, indent=2)
                                except Exception as e:
                                    print(f"Warning: Failed to save test metrics at epoch {epoch}: {e}")
                        except Exception as e:
                            print(f"Warning: Failed to save training history at epoch {epoch}: {e}")

                    # Early stopping based on test missing metrics
                    if early_stopping is not None:
                        # Look for test_all_evaluation callback results
                        test_results = [r for r in callback_results if r.get('name') == 'test_all_evaluation']
                        if test_results:
                            missing_metrics = test_results[0].get('missing_metrics', {})

                            # Extract the monitored metric
                            if early_stopping_metric == "loss":
                                metric_value = missing_metrics.get('rating_loss', float('inf'))
                                metric_name = "missing_rating_loss"
                            elif early_stopping_metric == "accuracy":
                                metric_value = missing_metrics.get('rating_accuracy', 0.0)
                                # Handle None case
                                if metric_value is None:
                                    metric_value = 0.0
                                metric_name = "missing_rating_accuracy"
                            else:
                                raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")

                            # Check if this is a better model before calling should_stop
                            is_better = early_stopping.is_better(metric_value, early_stopping.best_score)
                            
                            # Check if we should stop
                            if early_stopping.should_stop(metric_value, self.model):
                                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                                print(f"Best {metric_name}: {early_stopping.best_score:.4f}")
                                print(f"Restoring best model weights...")
                                early_stopping.restore_best_model(self.model)
                                # Save best model if enabled
                                if save_best_model:
                                    self._save_model(epoch + 1, suffix="_best")
                                break
                            elif is_better and save_best_model:
                                # Save best model whenever we find a better one
                                self._save_model(epoch + 1, suffix="_best")

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                total_loss = loss_dict.get('total_loss', 0.0)
                rating_loss = loss_dict.get('rating_loss', 0.0)
                ranking_loss = loss_dict.get('ranking_loss', 0.0)
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Total Loss: {total_loss:.4f}, "
                      f"Rating Loss: {rating_loss:.4f}, "
                      f"Ranking Loss: {ranking_loss:.4f}")

        return {
            'training_history': training_history,
            'callback_history': callback_history
        }
