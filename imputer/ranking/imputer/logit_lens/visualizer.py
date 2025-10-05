"""Minimal visualization utilities for logit lens analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .analyzer import LogitLensResults


class LogitLensVisualizer:
    """Minimal visualizer for logit lens analysis results."""
    
    def __init__(self, results: LogitLensResults):
        self.results = results
        self.num_layers = len(self.results.all_variables[0].layer_analyses) if self.results.all_variables else 0
        self.layer_indices = list(range(self.num_layers - 1))  # Exclude last layer
        
    def _safe_mean(self, values: List[float]) -> float:
        """Compute mean of values, handling None and NaN."""
        arr = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.mean(arr)) if arr else 0.0
    
    def _get_metric_by_layer(self, variables: List, metric: str) -> List[float]:
        """Extract metric values across layers for given variables."""
        return [self._safe_mean([var.layer_analyses[i].metrics.get(metric, 0.0) for var in variables]) 
                for i in self.layer_indices]
    
    def plot_train_performance(self, save_path: Optional[str] = None) -> None:
        """Plot train performance: observed vs missing for all metrics."""
        
        # Get train variables
        train_observed = self.results.filter_variables(is_train=True, is_observed=True)
        train_missing = self.results.filter_variables(is_train=True, is_missing=True)
        
        # Get rating and ranking subsets
        train_observed_rating = self.results.filter_variables(is_train=True, is_rating=True, is_observed=True)
        train_missing_rating = self.results.filter_variables(is_train=True, is_rating=True, is_missing=True)
        train_observed_ranking = self.results.filter_variables(is_train=True, is_ranking=True, is_observed=True)
        train_missing_ranking = self.results.filter_variables(is_train=True, is_ranking=True, is_missing=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Train Performance: Observed vs Missing', fontsize=16)
        
        # Rating metrics
        metrics_data = [
            ('accuracy', train_observed_rating, train_missing_rating, 'Rating Accuracy'),
            ('rmse', train_observed_rating, train_missing_rating, 'Rating RMSE'),
            ('l2_loss', train_observed_rating, train_missing_rating, 'Rating L2 Loss'),
            ('accuracy', train_observed_ranking, train_missing_ranking, 'Ranking Accuracy'),
            ('bt_loss', train_observed_ranking, train_missing_ranking, 'Ranking Log Loss')
        ]
        
        for i, (metric, observed_vars, missing_vars, title) in enumerate(metrics_data):
            row, col = i // 3, i % 3
            
            observed_values = self._get_metric_by_layer(observed_vars, metric)
            missing_values = self._get_metric_by_layer(missing_vars, metric)
            
            axes[row, col].plot(self.layer_indices, observed_values, '--', label='Observed', linewidth=2, color='blue')
            axes[row, col].plot(self.layer_indices, missing_values, '-', label='Missing', linewidth=2, color='red')
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Layer')
            axes[row, col].set_ylabel(metric.title())
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_test_performance(self, save_path: Optional[str] = None) -> None:
        """Plot test performance: observed vs missing for all metrics."""
        
        # Get test variables
        test_observed_rating = self.results.filter_variables(is_test=True, is_rating=True, is_observed=True)
        test_missing_rating = self.results.filter_variables(is_test=True, is_rating=True, is_missing=True)
        test_observed_ranking = self.results.filter_variables(is_test=True, is_ranking=True, is_observed=True)
        test_missing_ranking = self.results.filter_variables(is_test=True, is_ranking=True, is_missing=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Test Performance: Observed vs Missing', fontsize=16)
        
        # Rating metrics
        metrics_data = [
            ('accuracy', test_observed_rating, test_missing_rating, 'Rating Accuracy'),
            ('rmse', test_observed_rating, test_missing_rating, 'Rating RMSE'),
            ('l2_loss', test_observed_rating, test_missing_rating, 'Rating L2 Loss'),
            ('accuracy', test_observed_ranking, test_missing_ranking, 'Ranking Accuracy'),
            ('bt_loss', test_observed_ranking, test_missing_ranking, 'Ranking Log Loss')
        ]
        
        for i, (metric, observed_vars, missing_vars, title) in enumerate(metrics_data):
            row, col = i // 3, i % 3
            
            observed_values = self._get_metric_by_layer(observed_vars, metric)
            missing_values = self._get_metric_by_layer(missing_vars, metric)
            
            axes[row, col].plot(self.layer_indices, observed_values, '--', label='Observed', linewidth=2, color='blue')
            axes[row, col].plot(self.layer_indices, missing_values, '-', label='Missing', linewidth=2, color='red')
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Layer')
            axes[row, col].set_ylabel(metric.title())
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_performance(self, save_path: Optional[str] = None) -> None:
        """Plot all performance metrics in a single comprehensive view."""
        
        # Get all variable subsets
        train_observed_rating = self.results.filter_variables(is_train=True, is_rating=True, is_observed=True)
        train_missing_rating = self.results.filter_variables(is_train=True, is_rating=True, is_missing=True)
        train_observed_ranking = self.results.filter_variables(is_train=True, is_ranking=True, is_observed=True)
        train_missing_ranking = self.results.filter_variables(is_train=True, is_ranking=True, is_missing=True)
        test_observed_rating = self.results.filter_variables(is_test=True, is_rating=True, is_observed=True)
        test_missing_rating = self.results.filter_variables(is_test=True, is_rating=True, is_missing=True)
        test_observed_ranking = self.results.filter_variables(is_test=True, is_ranking=True, is_observed=True)
        test_missing_ranking = self.results.filter_variables(is_test=True, is_ranking=True, is_missing=True)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('All Performance Metrics: Observed vs Missing', fontsize=16)
        
        # Define all metrics and their data
        metrics_data = [
            ('accuracy', train_observed_rating, train_missing_rating, 'Train Rating Accuracy'),
            ('rmse', train_observed_rating, train_missing_rating, 'Train Rating RMSE'),
            ('l2_loss', train_observed_rating, train_missing_rating, 'Train Rating L2 Loss'),
            ('accuracy', train_observed_ranking, train_missing_ranking, 'Train Ranking Accuracy'),
            ('bt_loss', train_observed_ranking, train_missing_ranking, 'Train Ranking Log Loss'),
            ('accuracy', test_observed_rating, test_missing_rating, 'Test Rating Accuracy'),
            ('rmse', test_observed_rating, test_missing_rating, 'Test Rating RMSE'),
            ('l2_loss', test_observed_rating, test_missing_rating, 'Test Rating L2 Loss'),
            ('accuracy', test_observed_ranking, test_missing_ranking, 'Test Ranking Accuracy'),
            ('bt_loss', test_observed_ranking, test_missing_ranking, 'Test Ranking Log Loss')
        ]
        
        for i, (metric, observed_vars, missing_vars, title) in enumerate(metrics_data):
            row, col = i // 5, i % 5
            
            observed_values = self._get_metric_by_layer(observed_vars, metric)
            missing_values = self._get_metric_by_layer(missing_vars, metric)
            
            axes[row, col].plot(self.layer_indices, observed_values, '--', label='Observed', linewidth=2, color='blue')
            axes[row, col].plot(self.layer_indices, missing_values, '-', label='Missing', linewidth=2, color='red')
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xlabel('Layer')
            axes[row, col].set_ylabel(metric.title())
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_translator_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot translator training curves if this is a tuned lens analysis.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'translator_training_history' not in self.results.data_config:
            print("No training history found. This appears to be a logit lens analysis, not tuned lens.")
            return
        
        training_history = self.results.data_config['translator_training_history']
        
        if not training_history.get('epochs'):
            print("No training epochs found in training history.")
            return
        
        epochs = training_history['epochs']  # These are now actual epoch numbers (1, 2, 3, ...)
        num_layers = len(training_history['train_losses'])
        
        # Create subplots - 3 rows: total, rating, ranking
        fig, axes = plt.subplots(3, num_layers, figsize=(4*num_layers, 12))
        if num_layers == 1:
            axes = axes.reshape(3, 1)
        
        fig.suptitle('Translator Training Curves: Per-Layer Losses (Total, Rating, Ranking)', fontsize=16)
        
        for layer_idx in range(num_layers):
            # Total losses
            train_total = training_history['train_losses'][layer_idx]
            eval_total = training_history['eval_losses'][layer_idx]
            
            axes[0, layer_idx].plot(epochs, train_total, 'b-', label='Train', linewidth=2)
            axes[0, layer_idx].plot(epochs, eval_total, 'r-', label='Eval', linewidth=2)
            axes[0, layer_idx].set_title(f'Layer {layer_idx} - Total Loss')
            axes[0, layer_idx].set_xlabel('Epoch')
            axes[0, layer_idx].set_ylabel('Loss')
            axes[0, layer_idx].legend()
            axes[0, layer_idx].grid(True, alpha=0.3)
            
            # Rating losses
            train_rating = training_history['train_rating_losses'][layer_idx]
            eval_rating = training_history['eval_rating_losses'][layer_idx]
            
            axes[1, layer_idx].plot(epochs, train_rating, 'b-', label='Train', linewidth=2)
            axes[1, layer_idx].plot(epochs, eval_rating, 'r-', label='Eval', linewidth=2)
            axes[1, layer_idx].set_title(f'Layer {layer_idx} - Rating Loss')
            axes[1, layer_idx].set_xlabel('Epoch')
            axes[1, layer_idx].set_ylabel('Loss')
            axes[1, layer_idx].legend()
            axes[1, layer_idx].grid(True, alpha=0.3)
            
            # Ranking losses
            train_ranking = training_history['train_ranking_losses'][layer_idx]
            eval_ranking = training_history['eval_ranking_losses'][layer_idx]
            
            axes[2, layer_idx].plot(epochs, train_ranking, 'b-', label='Train', linewidth=2)
            axes[2, layer_idx].plot(epochs, eval_ranking, 'r-', label='Eval', linewidth=2)
            axes[2, layer_idx].set_title(f'Layer {layer_idx} - Ranking Loss')
            axes[2, layer_idx].set_xlabel('Epoch')
            axes[2, layer_idx].set_ylabel('Loss')
            axes[2, layer_idx].legend()
            axes[2, layer_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def save_results(self, save_path: str) -> None:
        """Save analysis results to JSON file."""
        serializable_results = {
            'model_config': self.results.model_config,
            'data_config': self.results.data_config,
            'all_variables': [
                {
                    'variable_id': f"{var.variable.attribute_id}_{var.variable.annotator_id}_{var.variable.item_ids}",
                    'instance': var.variable.instance,
                    'is_listwise': var.variable.is_listwise,
                    'is_observed': var.variable.is_observed,
                    'is_masked': var.variable.is_masked,
                    'is_missing': var.variable.is_missing,
                    'layer_analyses': [
                        {
                            'layer_idx': layer.layer_idx,
                            'metrics': layer.metrics
                        } for layer in var.layer_analyses
                    ]
                } for var in self.results.all_variables
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        print("=" * 60)
        print("LOGIT LENS ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"Model Configuration:")
        for key, value in self.results.model_config.items():
            print(f"  {key}: {value}")
        
        print(f"\nData Configuration:")
        for key, value in self.results.data_config.items():
            print(f"  {key}: {value}")
        
        print(f"\nPerformance Summary:")
        
        # Find best performing layer for each condition
        conditions = [
            ('Train Observed', self.results.filter_variables(is_train=True, is_observed=True)),
            ('Train Missing', self.results.filter_variables(is_train=True, is_missing=True)),
            ('Test Observed', self.results.filter_variables(is_test=True, is_observed=True)),
            ('Test Missing', self.results.filter_variables(is_test=True, is_missing=True))
        ]
        
        for condition_name, variables in conditions:
            if not variables:
                continue
            
            # Calculate average accuracy across all variables for each layer
            layer_accuracies = []
            for layer_idx in self.layer_indices:
                accuracies = [var.layer_analyses[layer_idx].metrics.get('accuracy', 0.0) for var in variables]
                layer_accuracies.append(np.mean(accuracies))
            
            if layer_accuracies:
                best_layer = np.argmax(layer_accuracies)
                best_acc = layer_accuracies[best_layer]
                print(f"  {condition_name}:")
                print(f"    Best Layer: {best_layer} (Accuracy: {best_acc:.4f})")
        
        print("=" * 60)