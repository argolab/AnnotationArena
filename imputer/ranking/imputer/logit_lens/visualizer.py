"""Visualization utilities for logit lens analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .analyzer import LogitLensResults, LayerAnalysis


class LogitLensVisualizer:
    """Creates visualizations for logit lens analysis results."""
    
    def __init__(self, results: LogitLensResults):
        self.results = results
        # Get number of layers from first variable (all should have same number)
        self.num_layers = len(self.results.all_variables[0].layer_analyses) if self.results.all_variables else 0
        
    def plot_performance_by_layer(self, 
                                 save_path: Optional[str] = None,
                                 figsize: tuple = (15, 10)) -> None:
        """Create comprehensive performance plots by layer."""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Logit Lens Analysis: Performance by Layer', fontsize=16)
        
        # Extract layer indices and metrics
        layer_indices = list(range(self.num_layers))
        
        # Train vs Test accuracy
        train_vars = self.results.get_train_variables()
        test_vars = self.results.get_test_variables()
        
        train_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in train_vars]) for i in layer_indices]
        test_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_vars]) for i in layer_indices]
        
        axes[0, 0].plot(layer_indices, train_acc, 'o-', label='Train', linewidth=2)
        axes[0, 0].plot(layer_indices, test_acc, 's-', label='Test', linewidth=2)
        axes[0, 0].set_title('Train vs Test Accuracy')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rating vs Ranking accuracy
        rating_vars = self.results.get_rating_variables()
        ranking_vars = self.results.get_ranking_variables()
        
        rating_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in rating_vars]) for i in layer_indices]
        ranking_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in ranking_vars]) for i in layer_indices]
        
        axes[0, 1].plot(layer_indices, rating_acc, 'o-', label='Rating', linewidth=2)
        axes[0, 1].plot(layer_indices, ranking_acc, 's-', label='Ranking', linewidth=2)
        axes[0, 1].set_title('Rating vs Ranking Accuracy')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Observed vs Masked accuracy (training instance only)
        observed_vars = self.results.filter_variables(is_train=True, is_observed=True)
        masked_vars = self.results.filter_variables(is_train=True, is_masked=True)
        
        observed_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in observed_vars]) for i in layer_indices]
        masked_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in masked_vars]) for i in layer_indices]
        
        axes[0, 2].plot(layer_indices, observed_acc, 'o-', label='Observed (Train)', linewidth=2)
        axes[0, 2].plot(layer_indices, masked_acc, 's-', label='Masked (Train)', linewidth=2)
        axes[0, 2].set_title('Training Instance: Observed vs Masked')
        axes[0, 2].set_xlabel('Layer Index')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # RMSE plots (for rating tasks)
        train_rmse = [np.mean([var.layer_analyses[i].metrics.get('rmse', 0.0) for var in train_vars]) for i in layer_indices]
        test_rmse = [np.mean([var.layer_analyses[i].metrics.get('rmse', 0.0) for var in test_vars]) for i in layer_indices]
        
        axes[1, 0].plot(layer_indices, train_rmse, 'o-', label='Train', linewidth=2)
        axes[1, 0].plot(layer_indices, test_rmse, 's-', label='Test', linewidth=2)
        axes[1, 0].set_title('Train vs Test RMSE')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of evaluations
        train_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in train_vars]) for i in layer_indices]
        test_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in test_vars]) for i in layer_indices]
        
        axes[1, 1].bar([x - 0.2 for x in layer_indices], train_eval, 0.4, label='Train', alpha=0.7)
        axes[1, 1].bar([x + 0.2 for x in layer_indices], test_eval, 0.4, label='Test', alpha=0.7)
        axes[1, 1].set_title('Number of Evaluations')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance improvement
        baseline_acc = train_acc[0] if train_acc else 0.0
        improvement = [acc - baseline_acc for acc in train_acc]
        
        axes[1, 2].plot(layer_indices, improvement, 'o-', linewidth=2, color='green')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].set_title('Performance Improvement from Baseline')
        axes[1, 2].set_xlabel('Layer Index')
        axes[1, 2].set_ylabel('Accuracy Improvement')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heatmap(self, 
                    metric: str = 'accuracy',
                    save_path: Optional[str] = None,
                    figsize: tuple = (12, 8)) -> None:
        """Create a heatmap showing performance across different conditions."""
        
        # Prepare data matrix
        conditions = ['Train', 'Test', 'Rating', 'Ranking', 'Observed (Train)', 'Masked (Train)']
        results_list = [
            self.results.get_train_variables(),
            self.results.get_test_variables(),
            self.results.get_rating_variables(),
            self.results.get_ranking_variables(),
            self.results.filter_variables(is_train=True, is_observed=True),
            self.results.filter_variables(is_train=True, is_masked=True)
        ]
        
        data_matrix = []
        for results in results_list:
            if results:  # Check if results list is not empty
                layer_metrics = [np.mean([var.layer_analyses[i].metrics.get(metric, 0.0) for var in results]) for i in range(self.num_layers)]
                data_matrix.append(layer_metrics)
            else:
                data_matrix.append([0.0] * self.num_layers)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        plt.figure(figsize=figsize)
        if HAS_SEABORN:
            sns.heatmap(data_matrix, 
                        xticklabels=[f'Layer {i}' for i in range(self.num_layers)],
                        yticklabels=conditions,
                        annot=True, 
                        fmt='.3f',
                        cmap='viridis',
                        cbar_kws={'label': metric.title()})
        else:
            # Fallback to matplotlib
            plt.imshow(data_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label=metric.title())
            plt.xticks(range(self.num_layers), [f'Layer {i}' for i in range(self.num_layers)])
            plt.yticks(range(len(conditions)), conditions)
            
            # Add text annotations
            for i in range(len(conditions)):
                for j in range(self.num_layers):
                    plt.text(j, i, f'{data_matrix[i, j]:.3f}', 
                            ha='center', va='center', color='white')
        
        plt.title(f'Logit Lens Analysis: {metric.title()} Heatmap')
        plt.xlabel('Layer Index')
        plt.ylabel('Condition')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_layer_comparison(self, 
                            layers_to_compare: List[int],
                            save_path: Optional[str] = None,
                            figsize: tuple = (15, 5)) -> None:
        """Compare performance across specific layers."""
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Layer Comparison: {layers_to_compare}', fontsize=16)
        
        conditions = ['Train', 'Test', 'Rating', 'Ranking', 'Observed (Train)', 'Masked (Train)']
        results_list = [
            self.results.get_train_variables(),
            self.results.get_test_variables(),
            self.results.get_rating_variables(),
            self.results.get_ranking_variables(),
            self.results.filter_variables(is_train=True, is_observed=True),
            self.results.filter_variables(is_train=True, is_masked=True)
        ]
        
        # Accuracy comparison
        acc_data = []
        for results in results_list:
            if results:  # Check if results list is not empty
                layer_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in results]) for i in layers_to_compare]
                acc_data.append(layer_acc)
            else:
                acc_data.append([0.0] * len(layers_to_compare))
        
        x = np.arange(len(conditions))
        width = 0.25
        
        for i, layer_idx in enumerate(layers_to_compare):
            layer_acc = [acc_data[j][i] for j in range(len(conditions))]
            axes[0].bar(x + i * width, layer_acc, width, label=f'Layer {layer_idx}', alpha=0.7)
        
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xlabel('Condition')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(conditions, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE comparison (for rating tasks)
        rmse_data = []
        for results in results_list:
            layer_rmse = [results[i].metrics.get('rmse', 0.0) for i in layers_to_compare]
            rmse_data.append(layer_rmse)
        
        for i, layer_idx in enumerate(layers_to_compare):
            layer_rmse = [rmse_data[j][i] for j in range(len(conditions))]
            axes[1].bar(x + i * width, layer_rmse, width, label=f'Layer {layer_idx}', alpha=0.7)
        
        axes[1].set_title('RMSE Comparison')
        axes[1].set_xlabel('Condition')
        axes[1].set_ylabel('RMSE')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(conditions, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Evaluation count comparison
        eval_data = []
        for results in results_list:
            layer_eval = [results[i].metrics.get('num_evaluations', 0) for i in layers_to_compare]
            eval_data.append(layer_eval)
        
        for i, layer_idx in enumerate(layers_to_compare):
            layer_eval = [eval_data[j][i] for j in range(len(conditions))]
            axes[2].bar(x + i * width, layer_eval, width, label=f'Layer {layer_idx}', alpha=0.7)
        
        axes[2].set_title('Evaluation Count Comparison')
        axes[2].set_xlabel('Condition')
        axes[2].set_ylabel('Count')
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(conditions, rotation=45)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, save_path: str) -> None:
        """Save analysis results to JSON file."""
        
        # Convert results to serializable format
        serializable_results = {
            'model_config': self.results.model_config,
            'data_config': self.results.data_config,
            'train_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.train_results
            ],
            'test_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.test_results
            ],
            'rating_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.rating_results
            ],
            'ranking_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.ranking_results
            ],
            'observed_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.observed_results
            ],
            'masked_results': [
                {
                    'layer_idx': r.layer_idx,
                    'metrics': r.metrics
                } for r in self.results.masked_results
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
        
        print(f"\nVariable Status Breakdown:")
        print(f"  Training Instance:")
        print(f"    Observed: {self.results.data_config.get('num_train_observed', 'N/A')}")
        print(f"    Masked: {self.results.data_config.get('num_train_masked', 'N/A')}")
        print(f"  Test Instance:")
        print(f"    Observed: {self.results.data_config.get('num_test_observed', 'N/A')}")
        print(f"    Missing: {self.results.data_config.get('num_test_missing', 'N/A')}")
        
        print(f"\nPerformance Summary:")
        
        # Find best performing layer for each condition
        conditions = [
            ('Train', self.results.train_results),
            ('Test', self.results.test_results),
            ('Rating', self.results.rating_results),
            ('Ranking', self.results.ranking_results),
            ('Observed', self.results.observed_results),
            ('Masked', self.results.masked_results)
        ]
        
        for condition_name, results in conditions:
            if not results:
                continue
                
            accuracies = [r.metrics.get('accuracy', 0.0) for r in results]
            best_layer = np.argmax(accuracies)
            best_acc = accuracies[best_layer]
            
            print(f"  {condition_name}:")
            print(f"    Best Layer: {best_layer} (Accuracy: {best_acc:.4f})")
            
            if 'rmse' in results[0].metrics:
                rmses = [r.metrics.get('rmse', 0.0) for r in results]
                best_rmse_layer = np.argmin(rmses)
                best_rmse = rmses[best_rmse_layer]
                print(f"    Best RMSE Layer: {best_rmse_layer} (RMSE: {best_rmse:.4f})")
        
        print("=" * 60)
