from typing import List, Dict, Any, Optional, Callable
import torch
import numpy as np
import torch.optim as optim
import copy

from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from .data import RankingData
import random
import sys


class EvaluationCallback:
    """Callback for evaluation during training."""

    def __init__(self, eval_engine, test_variables, test_data, converter, masking_rate=0.5, device='cpu'):
        """
        Initialize evaluation callback.

        Args:
            eval_engine: EvaluationEngine instance
            test_variables: List of test variables for evaluation
            test_data: Dictionary with test rating_data and ranking_data
            converter: DataConverter instance
            masking_rate: Masking rate for evaluation
            device: Device for computation
        """
        self.eval_engine = eval_engine
        self.test_variables = test_variables
        self.test_data = test_data
        self.converter = converter
        self.masking_rate = masking_rate
        self.device = device

    def on_epoch_end(self, model, epoch):
        """
        Called at the end of each epoch.

        Args:
            model: The model being trained
            epoch: Current epoch number

        Returns:
            Dictionary with evaluation results
        """
        try:
            results = self.eval_engine.evaluate_model(
                model=model,
                variables=self.test_variables,
                masking_rate=self.masking_rate,
                converter=self.converter,
                device=self.device
            )
            return {
                'epoch': epoch,
                'total_loss': results.total_loss,
                'rating_loss': results.rating_loss,
                'ranking_loss': results.ranking_loss,
                'rating_accuracy': results.rating_accuracy,
                'rating_rmse': results.rating_rmse,
                'ranking_accuracy': results.ranking_accuracy,
                'num_rating_evaluations': results.num_rating_evaluations,
                'num_ranking_evaluations': results.num_ranking_evaluations,
                'masked_metrics': results.masked_metrics,
                'observed_metrics': results.observed_metrics
            }
        except Exception as e:
            print(f"Warning: Evaluation callback failed at epoch {epoch}: {e}")
            return {'epoch': epoch, 'error': str(e)}


class EarlyStopping:
    """Early stopping utility for training."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.early_stopped = False

    def should_stop(self, current_loss: float, model) -> bool:
        """Check if training should stop and save best model state."""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
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
    def __init__(self, model, learning_rate=1e-3, device='cpu', embedding_anchor_reg: float = 0.0, callbacks=None,
                 masked_loss_weight: float = 1.0, observed_loss_weight: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy(masked_loss_weight=masked_loss_weight,
                                               observed_loss_weight=observed_loss_weight)
        # Callback system
        self.callbacks = callbacks or []

        # Regularize embedding parameters towards their random initialization
        self.embedding_anchor_reg = float(embedding_anchor_reg)
        self._embedding_initial_params = {}
        if self.embedding_anchor_reg > 0.0:
            # Snapshot initial embedding parameters after model is moved to device
            for name, param in self.model.named_parameters():
                if param.requires_grad and self._is_embedding_param_name(name):
                    self._embedding_initial_params[name] = param.detach().clone()

    def _is_embedding_param_name(self, name: str) -> bool:
        # FIXME: this function seems not robust. Be careful.
        n = name.lower()
        return ("embedding" in n) or ("embed" in n)

    def register_callback(self, callback):
        """Register an evaluation callback."""
        self.callbacks.append(callback)

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
    


    def train_step(self, batch_of_masked_versions):
        """Single training step with batch of multiple masked versions."""
        self.optimizer.zero_grad()

        # Handle both old format (List[RankingData]) and new format (List[List[RankingData]])
        if len(batch_of_masked_versions) > 0 and isinstance(batch_of_masked_versions[0], list):
            # New format: List[List[RankingData]] - batch of masked versions
            # For now, process each masked version separately and accumulate gradients
            total_loss_tensor = None
            total_losses = {}

            for masked_version in batch_of_masked_versions:
                if len(masked_version) == 0:
                    continue

                # Forward pass on this masked version
                reference_data_list = copy.deepcopy(masked_version)
                out = self.model(masked_version)

                # Compute loss for this version
                version_losses = self._compute_loss_for_version(out, reference_data_list)

                # Accumulate losses
                if total_loss_tensor is None:
                    total_loss_tensor = version_losses.get('_total_loss_tensor')
                    total_losses = {k: v for k, v in version_losses.items()}
                else:
                    if version_losses.get('_total_loss_tensor') is not None:
                        total_loss_tensor = total_loss_tensor + version_losses.get('_total_loss_tensor')
                    for k, v in version_losses.items():
                        if not k.startswith('_'):
                            total_losses[k] = total_losses.get(k, 0.0) + v

            # Average losses by number of versions
            num_versions = len([v for v in batch_of_masked_versions if len(v) > 0])
            if num_versions > 0:
                if total_loss_tensor is not None:
                    total_loss_tensor = total_loss_tensor / num_versions
                for k in total_losses:
                    if not k.startswith('_'):
                        total_losses[k] = total_losses[k] / num_versions

        else:
            # Old format: List[RankingData] - single masked version
            reference_data_list = copy.deepcopy(batch_of_masked_versions)
            out = self.model(batch_of_masked_versions)
            total_losses = self._compute_loss_for_version(out, reference_data_list)
            total_loss_tensor = total_losses.get('_total_loss_tensor')

        # Backprop and step
        if total_loss_tensor is not None:
            total_loss_tensor.backward()
            self.optimizer.step()

        # Return only float metrics
        return {k: v for k, v in total_losses.items() if not k.startswith('_')}

    def _compute_loss_for_version(self, model_output, reference_data_list):
        """Compute loss for a single masked version."""
        rating_logits = model_output['rating']
        ranking_logits = model_output['ranking']

        # Structured predictions and references for loss computation
        predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
        predictions: List["TopLayerPredictionResult"] = []
        references: List[RankingData] = []

        # Reconstruct references from batch tensors (0-indexed) - for ALL training variables with ground truth
        for i, var in enumerate(reference_data_list):
            if not var.is_listwise:
                predictions.append(predictions_full[i])
                references.append(RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=False,
                    item_ids=var.item_ids,
                    rating_value=var.rating_value,
                    is_masked=var.is_masked,
                ))
            elif var.is_listwise:
                predictions.append(predictions_full[i])
                references.append(RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=True,
                    item_ids=[it for it in var.item_ids[: self.model.max_rank_size]],
                    ranking_order=var.ranking_order,
                    is_masked=var.is_masked,
                ))
            else:
                raise ValueError("Shouldn't be here")
            
        losses = self.loss_strategy.compute(predictions, references)

        # Embedding anchor regularization: keep embeddings close to their random initialization
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
                losses['total_loss'] = float(losses['total_loss'] + losses['embedding_reg'])

        # Create total loss tensor for backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=self.device)
        # Add regularization term to tensor used for backprop
        total_loss_tensor = total_loss_tensor + reg_scaled

        # Store tensor for backward pass
        losses['_total_loss_tensor'] = total_loss_tensor

        return losses

    def train(self, train_batches, epochs=10, call_callbacks_every=1, verbose=True):
        """
        Training loop with callback support.

        Args:
            train_batches: List of training batches or single batch to repeat
            epochs: Number of epochs to train
            call_callbacks_every: Call callbacks every N epochs
            verbose: Print training progress

        Returns:
            Dictionary with training history and callback results
        """
        training_history = []
        callback_history = []

        # Handle single batch case
        if isinstance(train_batches, dict):
            train_batches = [train_batches]

        for epoch in range(epochs):
            epoch_losses = []

            # Training on all batches
            for batch in train_batches:
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)

            # Average losses for this epoch
            avg_losses = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    avg_losses[key] = np.mean([losses[key] for losses in epoch_losses])

            training_history.append({'epoch': epoch, **avg_losses})

            # Call callbacks
            if (epoch + 1) % call_callbacks_every == 0:
                callback_results = self._call_epoch_end_callbacks(epoch)
                if callback_results:
                    callback_history.extend(callback_results)

            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                total_loss = avg_losses.get('total_loss', 0.0)
                rating_loss = avg_losses.get('rating_loss', 0.0)
                ranking_loss = avg_losses.get('ranking_loss', 0.0)
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Total Loss: {total_loss:.4f}, "
                      f"Rating Loss: {rating_loss:.4f}, "
                      f"Ranking Loss: {ranking_loss:.4f}")

        return {
            'training_history': training_history,
            'callback_history': callback_history
        }