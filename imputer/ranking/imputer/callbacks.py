"""Callback classes for training evaluation."""

from typing import Dict, Any
from imputer.eval import EvaluationEngine
from imputer.data import DataConverter, RankingData


class EvaluationCallback:
    """Callback for evaluation during training (no masking during eval)."""

    def __init__(self, eval_engine, test_variables, converter, device='cuda', name='EvaluationCallback', instance_name=None, max_item=30):
        self.eval_engine = eval_engine
        self.test_variables = test_variables
        self.converter = converter
        self.device = device
        self.name = name
        self.instance_name = instance_name if instance_name is not None else name
        self.max_item = max_item

    def on_epoch_end(self, model, epoch):
        results = self.eval_engine.evaluate_model(
            model=model,
            variables=self.test_variables,
            converter=self.converter,
            device=self.device,
            max_item=self.max_item
        )
        return {
            'epoch': epoch,
            'name': self.name,
            'instance': self.instance_name,
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
        import copy
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
