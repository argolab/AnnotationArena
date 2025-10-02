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
    """Creates visualizations for logit lens analysis results.
    
    This visualizer generates four main types of plots:
    
    1. Rating Performance by Layer (plot_rating_performance_by_layer):
       - Train vs Test accuracy for rating tasks
       - Total rating performance across all data
       - Observed vs Masked accuracy (training instance only)
       - Observed vs Missing accuracy (test instance only)
       - Train vs Test RMSE for rating tasks
       - Number of evaluations per layer
    
    2. Ranking Performance by Layer (plot_ranking_performance_by_layer):
       - Train vs Test accuracy for ranking tasks
       - Total ranking performance across all data
       - Observed vs Masked accuracy (training instance only)
       - Observed vs Missing accuracy (test instance only)
       - Train vs Test Bradley-Terry loss for ranking tasks
       - Number of evaluations per layer
    
    3. Heatmap (plot_heatmap):
       - Performance matrix across conditions (Train/Test/Rating/Ranking/Observed/Masked)
       - Shows how different conditions perform at each layer
       - Supports accuracy, RMSE, or other metrics
    
    4. Layer Comparison (plot_layer_comparison):
       - Direct comparison of specific layers side-by-side
       - Bar charts for accuracy, RMSE, and evaluation counts
       - Useful for identifying optimal layer depths
    """
    
    def __init__(self, results: LogitLensResults):
        self.results = results
        # Get number of layers from first variable (all should have same number)
        self.num_layers = len(self.results.all_variables[0].layer_analyses) if self.results.all_variables else 0
        
    def plot_rating_performance_by_layer(self, 
                                        save_path: Optional[str] = None,
                                        figsize: tuple = (15, 10)) -> None:
        """Create rating-specific performance plots by layer.
        
        Question: How does the model's rating prediction performance evolve across layers,
        and how do different data conditions (train/test, observed/masked) affect this evolution?
        """
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Logit Lens Analysis: Rating Performance by Layer', fontsize=16)
        
        # Extract layer indices and metrics
        layer_indices = list(range(self.num_layers))
        
        # Get rating variables only
        rating_vars = self.results.get_rating_variables()
        train_rating_vars = self.results.filter_variables(is_train=True, is_rating=True)
        test_rating_vars = self.results.filter_variables(is_test=True, is_rating=True)
        observed_rating_vars = self.results.filter_variables(is_train=True, is_rating=True, is_observed=True)
        masked_rating_vars = self.results.filter_variables(is_train=True, is_rating=True, is_masked=True)
        test_observed_rating_vars = self.results.filter_variables(is_test=True, is_rating=True, is_observed=True)
        test_missing_rating_vars = self.results.filter_variables(is_test=True, is_rating=True, is_missing=True)
        
        # 1. Train vs Test accuracy for ratings
        train_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in train_rating_vars]) for i in layer_indices]
        test_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_rating_vars]) for i in layer_indices]
        
        axes[0, 0].plot(layer_indices, train_acc, 'o-', label='Train', linewidth=2)
        axes[0, 0].plot(layer_indices, test_acc, 's-', label='Test', linewidth=2)
        axes[0, 0].set_title('Rating: Train vs Test Accuracy')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total rating performance across all data
        total_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in rating_vars]) for i in layer_indices]
        
        axes[0, 1].plot(layer_indices, total_acc, 'o-', label='Total Rating', linewidth=2, color='purple')
        axes[0, 1].set_title('Rating: Total Performance')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Observed vs Masked accuracy (training instance only)
        observed_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in observed_rating_vars]) for i in layer_indices]
        masked_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in masked_rating_vars]) for i in layer_indices]
        
        axes[0, 2].plot(layer_indices, observed_acc, 'o-', label='Observed (Train)', linewidth=2)
        axes[0, 2].plot(layer_indices, masked_acc, 's-', label='Masked (Train)', linewidth=2)
        axes[0, 2].set_title('Rating: Observed vs Masked (Train)')
        axes[0, 2].set_xlabel('Layer Index')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Observed vs Missing accuracy (test instance only)
        test_observed_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_observed_rating_vars]) for i in layer_indices]
        test_missing_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_missing_rating_vars]) for i in layer_indices]
        
        axes[1, 0].plot(layer_indices, test_observed_acc, 'o-', label='Observed (Test)', linewidth=2)
        axes[1, 0].plot(layer_indices, test_missing_acc, 's-', label='Missing (Test)', linewidth=2)
        axes[1, 0].set_title('Rating: Observed vs Missing (Test)')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. RMSE plots (for rating tasks)
        train_rmse = [np.mean([var.layer_analyses[i].metrics.get('rmse', 0.0) for var in train_rating_vars]) for i in layer_indices]
        test_rmse = [np.mean([var.layer_analyses[i].metrics.get('rmse', 0.0) for var in test_rating_vars]) for i in layer_indices]
        
        axes[1, 1].plot(layer_indices, train_rmse, 'o-', label='Train', linewidth=2)
        axes[1, 1].plot(layer_indices, test_rmse, 's-', label='Test', linewidth=2)
        axes[1, 1].set_title('Rating: Train vs Test RMSE')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Number of evaluations
        train_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in train_rating_vars]) for i in layer_indices]
        test_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in test_rating_vars]) for i in layer_indices]
        
        axes[1, 2].bar([x - 0.2 for x in layer_indices], train_eval, 0.4, label='Train', alpha=0.7)
        axes[1, 2].bar([x + 0.2 for x in layer_indices], test_eval, 0.4, label='Test', alpha=0.7)
        axes[1, 2].set_title('Rating: Number of Evaluations')
        axes[1, 2].set_xlabel('Layer Index')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ranking_performance_by_layer(self, 
                                         save_path: Optional[str] = None,
                                         figsize: tuple = (15, 10)) -> None:
        """Create ranking-specific performance plots by layer.
        
        Question: How does the model's ranking prediction performance evolve across layers,
        and how do different data conditions (train/test, observed/masked) affect this evolution?
        """
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Logit Lens Analysis: Ranking Performance by Layer', fontsize=16)
        
        # Extract layer indices and metrics
        layer_indices = list(range(self.num_layers))
        
        # Get ranking variables only
        ranking_vars = self.results.get_ranking_variables()
        train_ranking_vars = self.results.filter_variables(is_train=True, is_ranking=True)
        test_ranking_vars = self.results.filter_variables(is_test=True, is_ranking=True)
        observed_ranking_vars = self.results.filter_variables(is_train=True, is_ranking=True, is_observed=True)
        masked_ranking_vars = self.results.filter_variables(is_train=True, is_ranking=True, is_masked=True)
        test_observed_ranking_vars = self.results.filter_variables(is_test=True, is_ranking=True, is_observed=True)
        test_missing_ranking_vars = self.results.filter_variables(is_test=True, is_ranking=True, is_missing=True)
        
        # 1. Train vs Test accuracy for rankings
        train_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in train_ranking_vars]) for i in layer_indices]
        test_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_ranking_vars]) for i in layer_indices]
        
        axes[0, 0].plot(layer_indices, train_acc, 'o-', label='Train', linewidth=2)
        axes[0, 0].plot(layer_indices, test_acc, 's-', label='Test', linewidth=2)
        axes[0, 0].set_title('Ranking: Train vs Test Accuracy')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total ranking performance across all data
        total_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in ranking_vars]) for i in layer_indices]
        
        axes[0, 1].plot(layer_indices, total_acc, 'o-', label='Total Ranking', linewidth=2, color='purple')
        axes[0, 1].set_title('Ranking: Total Performance')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Observed vs Masked accuracy (training instance only)
        observed_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in observed_ranking_vars]) for i in layer_indices]
        masked_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in masked_ranking_vars]) for i in layer_indices]
        
        axes[0, 2].plot(layer_indices, observed_acc, 'o-', label='Observed (Train)', linewidth=2)
        axes[0, 2].plot(layer_indices, masked_acc, 's-', label='Masked (Train)', linewidth=2)
        axes[0, 2].set_title('Ranking: Observed vs Masked (Train)')
        axes[0, 2].set_xlabel('Layer Index')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Observed vs Missing accuracy (test instance only)
        test_observed_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_observed_ranking_vars]) for i in layer_indices]
        test_missing_acc = [np.mean([var.layer_analyses[i].metrics.get('accuracy', 0.0) for var in test_missing_ranking_vars]) for i in layer_indices]
        
        axes[1, 0].plot(layer_indices, test_observed_acc, 'o-', label='Observed (Test)', linewidth=2)
        axes[1, 0].plot(layer_indices, test_missing_acc, 's-', label='Missing (Test)', linewidth=2)
        axes[1, 0].set_title('Ranking: Observed vs Missing (Test)')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Bradley-Terry loss plots (for ranking tasks)
        train_bt_loss = [np.mean([var.layer_analyses[i].metrics.get('bt_loss', 0.0) for var in train_ranking_vars]) for i in layer_indices]
        test_bt_loss = [np.mean([var.layer_analyses[i].metrics.get('bt_loss', 0.0) for var in test_ranking_vars]) for i in layer_indices]
        
        axes[1, 1].plot(layer_indices, train_bt_loss, 'o-', label='Train', linewidth=2)
        axes[1, 1].plot(layer_indices, test_bt_loss, 's-', label='Test', linewidth=2)
        axes[1, 1].set_title('Ranking: Train vs Test BT Loss')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Bradley-Terry Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Number of evaluations
        train_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in train_ranking_vars]) for i in layer_indices]
        test_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in test_ranking_vars]) for i in layer_indices]
        
        axes[1, 2].bar([x - 0.2 for x in layer_indices], train_eval, 0.4, label='Train', alpha=0.7)
        axes[1, 2].bar([x + 0.2 for x in layer_indices], test_eval, 0.4, label='Test', alpha=0.7)
        axes[1, 2].set_title('Ranking: Number of Evaluations')
        axes[1, 2].set_xlabel('Layer Index')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_all_performance_by_layer(self, 
                                    save_dir: Optional[str] = None,
                                    figsize: tuple = (15, 10)) -> None:
        """Create both rating and ranking performance plots by layer.
        
        Question: How do rating and ranking tasks differ in their performance evolution
        across layers, and what insights can we gain from comparing them?
        """
        
        if save_dir:
            rating_path = f"{save_dir}/rating_performance_by_layer.png"
            ranking_path = f"{save_dir}/ranking_performance_by_layer.png"
        else:
            rating_path = None
            ranking_path = None
        
        self.plot_rating_performance_by_layer(save_path=rating_path, figsize=figsize)
        self.plot_ranking_performance_by_layer(save_path=ranking_path, figsize=figsize)
    
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
            if results:  # Check if results list is not empty
                layer_rmse = [np.mean([var.layer_analyses[i].metrics.get('rmse', 0.0) for var in results]) for i in layers_to_compare]
                rmse_data.append(layer_rmse)
            else:
                rmse_data.append([0.0] * len(layers_to_compare))
        
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
            if results:  # Check if results list is not empty
                layer_eval = [np.mean([var.layer_analyses[i].metrics.get('num_evaluations', 0) for var in results]) for i in layers_to_compare]
                eval_data.append(layer_eval)
            else:
                eval_data.append([0.0] * len(layers_to_compare))
        
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
        
        print(f"\nVariable Status Breakdown:")
        print(f"  Training Instance:")
        print(f"    Observed: {self.results.data_config.get('num_observed_variables', 'N/A')}")
        print(f"    Masked: {self.results.data_config.get('num_masked_variables', 'N/A')}")
        print(f"  Test Instance:")
        print(f"    Observed: {self.results.data_config.get('num_observed_variables', 'N/A')}")
        print(f"    Missing: {self.results.data_config.get('num_missing_variables', 'N/A')}")
        
        print(f"\nPerformance Summary:")
        
        # Find best performing layer for each condition
        conditions = [
            ('Train', self.results.get_train_variables()),
            ('Test', self.results.get_test_variables()),
            ('Rating', self.results.get_rating_variables()),
            ('Ranking', self.results.get_ranking_variables()),
            ('Observed', self.results.get_observed_variables()),
            ('Masked', self.results.get_masked_variables())
        ]
        
        for condition_name, variables in conditions:
            if not variables:
                continue
            
            # Calculate average accuracy across all variables for each layer
            layer_accuracies = []
            for layer_idx in range(self.num_layers):
                accuracies = [var.layer_analyses[layer_idx].metrics.get('accuracy', 0.0) for var in variables]
                layer_accuracies.append(np.mean(accuracies))
            
            best_layer = np.argmax(layer_accuracies)
            best_acc = layer_accuracies[best_layer]
            
            print(f"  {condition_name}:")
            print(f"    Best Layer: {best_layer} (Accuracy: {best_acc:.4f})")
            
            # Check if RMSE is available (for rating tasks)
            if variables and 'rmse' in variables[0].layer_analyses[0].metrics:
                layer_rmses = []
                for layer_idx in range(self.num_layers):
                    rmses = [var.layer_analyses[layer_idx].metrics.get('rmse', 0.0) for var in variables]
                    layer_rmses.append(np.mean(rmses))
                
                best_rmse_layer = np.argmin(layer_rmses)
                best_rmse = layer_rmses[best_rmse_layer]
                print(f"    Best RMSE Layer: {best_rmse_layer} (RMSE: {best_rmse:.4f})")
        
        print("=" * 60)
