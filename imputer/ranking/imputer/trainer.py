from typing import List, Dict, Any, Optional
import torch
import numpy as np
import torch.optim as optim
import copy
import random

from losses import DefaultLossStrategy, adapt_batched_logits_to_predictions, TopLayerPredictionResult
from data import RankingData
from tqdm import tqdm

class EvaluationCallback:
    """Callback for evaluation during training (no masking during eval)."""

    def __init__(self, eval_engine, test_variables, converter, device='cuda'):
        self.eval_engine = eval_engine
        self.test_variables = test_variables
        self.converter = converter
        self.device = device

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
    """Trainer that masks a subset of training observed variables and appends missing ones."""

    def __init__(self, model, learning_rate=1e-3, device='cuda', embedding_anchor_reg: float = 0.0, callbacks=None,
                 masked_loss_weight: float = 3.0, observed_loss_weight: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy(masked_loss_weight=masked_loss_weight,
                                               observed_loss_weight=observed_loss_weight)
        # Callback system
        self.callbacks = callbacks or []

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
        self.optimizer.zero_grad()

        # Validate inputs
        if train_observed_vars is None or train_missing_vars is None:
            raise ValueError("train_observed_vars and train_missing_vars must be provided")

        # Apply masking to observed
        masked_or_observed = self._apply_training_mask(train_observed_vars, masking_rate)

        # Append missing as-is (status=0)
        batch_list: List[RankingData] = []
        batch_list.extend(masked_or_observed)
        for var in train_missing_vars:
            if not var.is_missing and not var.is_masked and not var.is_observed:
                # Defensive: enforce valid status
                raise ValueError("train_missing_vars contains an entry that is not missing")
            batch_list.append(var)

        # Forward
        out = self.model(batch_list)

        # Compute loss: only over non-missing (observed+masked)
        total_losses = self._compute_loss_for_batch(out, masked_or_observed)
        total_loss_tensor = total_losses.get('_total_loss_tensor')

        if total_loss_tensor is not None:
            total_loss_tensor.backward()
            self.optimizer.step()

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
              verbose: bool = True):
        """Simple training loop using the new API."""
        training_history = []
        callback_history = []

        for epoch in tqdm(range(epochs)):
            loss_dict = self.train_step(train_observed_vars, train_missing_vars, masking_rate)

            training_history.append({'epoch': epoch, **loss_dict})

            if (epoch + 1) % call_callbacks_every == 0:
                callback_results = self._call_epoch_end_callbacks(epoch)
                if callback_results:
                    print("Callback results:")
                    import json
                    print(json.dumps(callback_results, indent=2, sort_keys=True))
                    callback_history.extend(callback_results)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                total_loss = loss_dict.get('total_loss', 0.0)
                rating_loss = loss_dict.get('rating_loss', 0.0)
                ranking_loss = loss_dict.get('ranking_loss', 0.0)
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Total Loss: {total_loss:.4f}, "
                      f"Rating Loss: {rating_loss:.4f}, "
                      f"Ranking Loss: {ranking_loss:.4f}")
                
            model_path = '/export/fs06/psingh54/StanExps/imputer/ranking/OUTPUT/IMPUTER/models' / f"model_{str(epoch)}.pt"
            torch.save({
                "state_dict": self.model.state_dict(),
                "max_rank_size": 2
            }, model_path)

        return {
            'training_history': training_history,
            'callback_history': callback_history
        }