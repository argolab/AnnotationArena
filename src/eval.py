"""
Evaluation {Local and WandB} for Active Learner framework.

Author: Prabhav Singh / Haojun Shi
"""

import os
import json
import logging
import numpy as np
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from utils import compute_metrics, AnnotationDataset
from config import Config
from annotationArena import AnnotationArena
from selection import SelectionFactory

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Wandb not available, evaluation logging disabled")

class ModelEvaluator:
    """Comprehensive model evaluation with logging and metrics tracking."""
    
    def __init__(self, config: Config, use_wandb: bool = False):
        """Initialize evaluator with config and optional wandb."""
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Track evaluation history
        self.evaluation_history = []
        
        # Setup WandB metric organization
        if self.use_wandb and wandb.run is not None:
            wandb.define_metric("cycle")
            wandb.define_metric("training/*", step_metric="cycle")
            wandb.define_metric("validation/*", step_metric="cycle") 
            wandb.define_metric("test/*", step_metric="cycle")
            wandb.define_metric("questions/*", step_metric="cycle")
            wandb.define_metric("test_incremental/*", step_metric="features_observed")
        
        logger.info(f"ModelEvaluator initialized - Wandb: {self.use_wandb}")
    
    def evaluate_model(self, model, dataset: AnnotationDataset, dataset_name: str = "unknown", 
                      target_questions: Optional[List[int]] = None, split_type: str = "test") -> Dict[str, Any]:
        """Comprehensive model evaluation on a dataset."""
        
        logger.info(f"Evaluating model on {dataset_name} {split_type} set ({len(dataset)} examples)")
        
        if target_questions is None:
            target_questions = list(range(7))  # Default for HANNA
        
        model.eval()
        
        # Overall metrics
        all_predictions = []
        all_true_values = []
        all_losses = []
        
        # Question-wise metrics
        question_predictions = {q: [] for q in target_questions}
        question_true_values = {q: [] for q in target_questions}
        question_losses = {q: [] for q in target_questions}
        
        # Annotator-wise metrics
        annotator_predictions = {}
        annotator_true_values = {}
        
        total_examples = 0
        processed_examples = 0
        
        with torch.no_grad():
            for example_idx in range(len(dataset)):
                try:
                    data_entry = dataset.get_data_entry(example_idx)
                    known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
                    
                    inputs = inputs.unsqueeze(0).to(self.device)
                    annotators_tensor = annotators.unsqueeze(0).to(self.device)
                    questions_tensor = questions.unsqueeze(0).to(self.device)
                    
                    if embeddings is not None:
                        embeddings = embeddings.unsqueeze(0).to(self.device)
                    else:
                        seq_len = inputs.shape[1]
                        embeddings = torch.zeros(1, seq_len, 384).to(self.device)
                    
                    # Get model predictions
                    outputs = model(inputs, annotators_tensor, questions_tensor, embeddings)
                    
                    # Process each position
                    for pos in range(len(data_entry['questions'])):
                        question_idx = data_entry['questions'][pos]
                        annotator_idx = data_entry['annotators'][pos]
                        
                        # Skip if not target question
                        if question_idx not in target_questions:
                            continue
                        
                        # Get prediction and true value
                        pred_probs = F.softmax(outputs[0, pos], dim=0)
                        pred_class = torch.argmax(pred_probs).item()
                        pred_score = pred_class + 1  # Convert to 1-5 scale
                        
                        # Get true value
                        if 'true_answers' in data_entry and data_entry['true_answers']:
                            true_class = torch.argmax(torch.tensor(data_entry['true_answers'][pos])).item()
                        else:
                            true_class = torch.argmax(torch.tensor(data_entry['answers'][pos])).item()
                        true_score = true_class + 1
                        
                        # Compute loss
                        loss = F.cross_entropy(
                            outputs[0:1, pos], 
                            torch.tensor([true_class], device=self.device)
                        ).item()
                        
                        # Add to overall metrics
                        all_predictions.append(pred_score)
                        all_true_values.append(true_score)
                        all_losses.append(loss)
                        
                        # Add to question-wise metrics
                        question_predictions[question_idx].append(pred_score)
                        question_true_values[question_idx].append(true_score)
                        question_losses[question_idx].append(loss)
                        
                        # Add to annotator-wise metrics
                        if annotator_idx not in annotator_predictions:
                            annotator_predictions[annotator_idx] = []
                            annotator_true_values[annotator_idx] = []
                        annotator_predictions[annotator_idx].append(pred_score)
                        annotator_true_values[annotator_idx].append(true_score)
                    
                    processed_examples += 1
                    total_examples += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing example {example_idx}: {e}")
                    total_examples += 1
                    continue
        
        # Compute overall metrics
        if len(all_predictions) == 0:
            logger.error("No valid predictions found")
            return self._empty_evaluation_result(dataset_name, split_type)
        
        overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_true_values))
        overall_metrics['avg_expected_loss'] = np.mean(all_losses)
        overall_metrics['total_predictions'] = len(all_predictions)
        
        # Compute question-wise metrics
        question_metrics = {}
        for q_idx in target_questions:
            if len(question_predictions[q_idx]) > 0:
                q_metrics = compute_metrics(
                    np.array(question_predictions[q_idx]), 
                    np.array(question_true_values[q_idx])
                )
                q_metrics['avg_expected_loss'] = np.mean(question_losses[q_idx])
                q_metrics['count'] = len(question_predictions[q_idx])
                question_metrics[f'Q{q_idx}'] = q_metrics
            else:
                question_metrics[f'Q{q_idx}'] = self._empty_question_metrics()
        
        # Compute annotator-wise metrics
        annotator_metrics = {}
        for ann_idx in annotator_predictions:
            if len(annotator_predictions[ann_idx]) > 0:
                ann_metrics = compute_metrics(
                    np.array(annotator_predictions[ann_idx]),
                    np.array(annotator_true_values[ann_idx])
                )
                ann_metrics['count'] = len(annotator_predictions[ann_idx])
                
                # Determine annotator type
                if ann_idx == -1:
                    annotator_metrics['LLM'] = ann_metrics
                else:
                    annotator_metrics[f'Human_{ann_idx}'] = ann_metrics
        
        # Compile results
        evaluation_result = {
            'dataset_name': dataset_name,
            'split_type': split_type,
            'timestamp': len(self.evaluation_history),
            'total_examples': total_examples,
            'processed_examples': processed_examples,
            'overall': overall_metrics,
            'by_question': question_metrics,
            'by_annotator': annotator_metrics,
            'target_questions': target_questions
        }
        
        # Log results
        self._log_evaluation_results(evaluation_result)
        
        # Store in history
        self.evaluation_history.append(evaluation_result)
        
        logger.info(f"Evaluation completed - RMSE: {overall_metrics['rmse']:.4f}, "
                   f"Pearson: {overall_metrics['pearson']:.4f}, "
                   f"Predictions: {overall_metrics['total_predictions']}")
        
        return evaluation_result
    
    def evaluate_model_test(self, model, dataset: AnnotationDataset, dataset_name: str = "unknown", target_questions: Optional[List[int]] = None, split_type: str = "test", feature_selection_type: str = "voi") -> Dict[str, Any]:
        """Comprehensive model evaluation on a dataset using active learning with feature selection."""
        
        logger.info(f"\n-- Evaluating model on {dataset_name} {split_type} set ({len(dataset)} examples) with {feature_selection_type} feature selection --")
        
        if target_questions is None:
            target_questions = list(range(7))
        all_results = []
        model.eval()
        
        # Create deep copy of dataset to avoid state persistence between cycles
        dataset_copy = copy.deepcopy(dataset)
        
        # Initialize arena and feature selector
        arena = AnnotationArena(model, self.device)
        arena.set_dataset(dataset_copy)
        feature_selector = SelectionFactory.create_feature_strategy(feature_selection_type, model, self.device)
        
        # Initialize metrics tracking
        metrics_trends = {
            'rmse': [],
            'pearson': [],
            'spearman': [],
            'kendall': [],
            'accuracy': [],
            'mae': [],
            'avg_expected_loss': []
        }
        
        # Count total features across all examples
        total_features = 0
        for example_idx in range(len(dataset_copy)):
            data_entry = dataset_copy.get_data_entry(example_idx)
            total_features += len(data_entry['questions'])
        
        logger.info(f"Starting evaluation with {total_features} total features to collect")
        
        # Initial evaluation with no features observed (all positions unknown)
        initial_eval = self.evaluate_model(model, dataset_copy, dataset_name, target_questions, f"{split_type}_initial")
        all_results.append(initial_eval)
        for metric_name in metrics_trends.keys():
            if metric_name in initial_eval['overall']:
                metrics_trends[metric_name].append(initial_eval['overall'][metric_name])
            else:
                metrics_trends[metric_name].append(0.0)
        
        logger.info(f"Initial evaluation (0 features): RMSE={initial_eval['overall']['rmse']:.4f}, Pearson={initial_eval['overall']['pearson']:.4f}")
        
        # Iteratively select and observe features
        features_collected = 0
        
        while features_collected < total_features:

            # For each example, select one feature if available
            features_selected_this_round = 0
            
            for example_idx in tqdm(range(len(dataset_copy))):
                # Select features for this example (limit to 1 per round)
                selected_features = feature_selector.select_features(
                    example_idx, dataset_copy, 
                    num_to_select=1,
                    loss_type="cross_entropy",
                    target_questions=[0,1,2,3,4,5]
                )
                
                # Observe selected features
                for feature_info in selected_features:
                    pos = feature_info[0]  # Position index
                    success_criteria = arena.observe_position(example_idx, pos)
                    features_selected_this_round += 1
                    features_collected += 1
                    
                    logger.debug(f"Observed feature at example {example_idx}, position {pos} (total collected: {features_collected}). Success - {success_criteria}")
                    
                    # Break after selecting one feature per example per round
                    break
            
            # If no features were selected this round, break
            if features_selected_this_round == 0:
                logger.info("No more features available for selection")
                break
            
            # Evaluate model with newly observed features
            current_eval = self.evaluate_model(model, dataset_copy, dataset_name, target_questions, f"{split_type}_step_{features_collected/len(dataset_copy)}")
            all_results.append(current_eval)
            # Track metrics
            for metric_name in metrics_trends.keys():
                if metric_name in current_eval['overall']:
                    metrics_trends[metric_name].append(current_eval['overall'][metric_name])
                else:
                    metrics_trends[metric_name].append(0.0)
            
            logger.info(f"After {features_collected} features: RMSE={current_eval['overall']['rmse']:.4f}, "
                    f"Pearson={current_eval['overall']['pearson']:.4f}, "
                    f"Features selected this round: {features_selected_this_round}")
            
            # Early termination if all features have been collected
            if features_collected >= total_features:
                break
        
        # Final evaluation summary
        final_metrics = {metric: values[-1] for metric, values in metrics_trends.items() if values}
        logger.info(f"Final evaluation after {features_collected} features: RMSE={final_metrics.get('rmse', 0):.4f}, "
                f"Pearson={final_metrics.get('pearson', 0):.4f}")
        
        # Return results with trends
        result = {
            'dataset_name': dataset_name,
            'split_type': split_type,
            'feature_selection_type': feature_selection_type,
            'total_features_collected': features_collected,
            'total_features_available': total_features,
            'metrics_trends': metrics_trends,
            'final_metrics': final_metrics,
            'evaluation_steps': len(metrics_trends['rmse'])
        }
        
        logger.info(f"Evaluation completed: {len(metrics_trends['rmse'])} evaluation steps from 0 to {features_collected} features")
        
        return result, all_results[len(all_results) // 2] if all_results else initial_eval
    
    def evaluate_active_learning_cycle(self, model, datasets: Dict[str, AnnotationDataset], 
                                 cycle_num: int, additional_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate model at the end of an active learning cycle."""
        
        logger.info(f"Evaluating active learning cycle {cycle_num}")
        
        cycle_results = {
            'cycle': cycle_num,
            'timestamp': len(self.evaluation_history),
            'evaluations': {}
        }
        
        # Evaluate on all provided datasets
        for dataset_name, dataset in datasets.items():
            if not dataset_name == "test":
                eval_result = self.evaluate_model(model, dataset, dataset_name, split_type=dataset_name)
                cycle_results['evaluations'][dataset_name] = eval_result
            else:
                test_trend, eval_result = self.evaluate_model_test(model, dataset)
                cycle_results["evaluations"]["test"] = eval_result
                cycle_results["test_trend"] = test_trend
        
        # Add additional metrics if provided
        if additional_metrics:
            cycle_results['additional_metrics'] = additional_metrics
            logger.debug(f"Added {len(additional_metrics)} additional metrics")
        
        # Log cycle summary with organized WandB metrics
        if self.use_wandb and wandb.run is not None:
            wandb_data = {"cycle": cycle_num}
            
            # Organize metrics by dataset type
            for dataset_name, eval_result in cycle_results['evaluations'].items():
                metrics = eval_result['overall']
                
                # Core metrics for each dataset
                wandb_data.update({
                    f"{dataset_name}/rmse": metrics['rmse'],
                    f"{dataset_name}/pearson": metrics['pearson'],
                    f"{dataset_name}/spearman": metrics['spearman'],
                    f"{dataset_name}/kendall": metrics['kendall'],
                    f"{dataset_name}/accuracy": metrics['accuracy'],
                    f"{dataset_name}/expected_loss": metrics['avg_expected_loss'],
                    f"{dataset_name}/total_predictions": metrics['total_predictions']
                })
                
                # Question-wise metrics
                for q_name, q_metrics in eval_result['by_question'].items():
                    if q_metrics['count'] > 0:
                        wandb_data[f"questions/{dataset_name}_{q_name}_rmse"] = q_metrics['rmse']
                        wandb_data[f"questions/{dataset_name}_{q_name}_pearson"] = q_metrics['pearson']
                        wandb_data[f"questions/{dataset_name}_{q_name}_count"] = q_metrics['count']
        
            # Test incremental metrics
            if 'test_trend' in cycle_results:
                test_trend = cycle_results['test_trend']
                metrics_trends = test_trend.get('metrics_trends', {})
                
                if metrics_trends:
                    # Get the number of steps in this cycle
                    num_steps = len(next(iter(metrics_trends.values())))
                    
                    # For each step, log all metrics together with features_observed as step
                    for step_idx in range(num_steps):
                        features_observed = step_idx
                        
                        # 1. Log current values for each metric at this step
                        step_data = {
                            "features_observed": features_observed,
                            "cycle": cycle_num
                        }
                        
                        for metric_name, values in metrics_trends.items():
                            if step_idx < len(values):
                                # Log raw value with cycle in name for cycle-specific plots
                                step_data[f"test_incremental/cycle_{cycle_num}_{metric_name}"] = values[step_idx]
                        
                        wandb.log(step_data)
                    
                    # 2. Log final values separately for cycle summary
                    cycle_data = {"cycle": cycle_num}
                    for metric_name, values in metrics_trends.items():
                        if values:
                            final_value = values[-1]
                            # Add to cycle summary
                            cycle_data[f"test_incremental/final_{metric_name}_by_cycle"] = final_value
                    
                    # Log cycle summary
                    wandb.log(cycle_data)
            
            # Additional metrics from training
            if additional_metrics:
                for key, value in additional_metrics.items():
                    wandb_data[f"training/{key}"] = value
            
            wandb.log(wandb_data)
            
            # Create summary table for final cycle
            if cycle_num > 0 and cycle_num % 5 == 0:
                self._log_summary_table(cycle_results, cycle_num)
        
        self.evaluation_history.append(cycle_results)
        
        return cycle_results
    
    def _log_summary_table(self, cycle_results: Dict[str, Any], cycle_num: int):
        """Log summary table to WandB."""
        
        if not self.use_wandb or wandb.run is None:
            return
        
        table_data = []
        
        for dataset_name, eval_result in cycle_results['evaluations'].items():
            metrics = eval_result['overall']
            table_data.append([
                dataset_name,
                cycle_num,
                f"{metrics['rmse']:.4f}",
                f"{metrics['pearson']:.4f}",
                f"{metrics['spearman']:.4f}",
                f"{metrics['kendall']:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['avg_expected_loss']:.4f}",
                metrics['total_predictions']
            ])
        
        table = wandb.Table(
            data=table_data,
            columns=["Dataset", "Cycle", "RMSE", "Pearson", "Spearman", "Kendall", "Accuracy", "Expected Loss", "Predictions"]
        )
        
        wandb.log({f"summary_table_cycle_{cycle_num}": table})
    
    def compare_models(self, models: Dict[str, Any], dataset: AnnotationDataset, 
                      dataset_name: str = "comparison") -> Dict[str, Any]:
        """Compare multiple models on the same dataset."""
        
        logger.info(f"Comparing {len(models)} models on {dataset_name}")
        
        comparison_results = {
            'dataset_name': dataset_name,
            'timestamp': len(self.evaluation_history),
            'models': {},
            'summary': {}
        }
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            eval_result = self.evaluate_model(model, dataset, dataset_name, split_type="comparison")
            comparison_results['models'][model_name] = eval_result
        
        # Generate comparison summary
        if len(models) > 1:
            model_names = list(models.keys())
            metrics = ['rmse', 'pearson', 'avg_expected_loss']
            
            for metric in metrics:
                values = [comparison_results['models'][name]['overall'][metric] for name in model_names]
                comparison_results['summary'][metric] = {
                    'best_model': model_names[np.argmin(values) if metric in ['rmse', 'avg_expected_loss'] else np.argmax(values)],
                    'values': dict(zip(model_names, values))
                }
        
        self.evaluation_history.append(comparison_results)
        
        logger.info(f"Model comparison completed")
        return comparison_results
    
    def save_evaluation_history(self, experiment_name: str) -> str:
        """Save evaluation history to file."""
        
        if not self.evaluation_history:
            logger.warning("No evaluation history to save")
            return ""
        
        exp_paths = self.config.get_experiment_paths(experiment_name)
        eval_file = os.path.join(exp_paths['results_dir'], "evaluation_history.json")
        
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation history to {eval_file}")
        return eval_file
    
    def generate_evaluation_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Generate summary of all evaluations."""
        
        if not self.evaluation_history:
            return {}
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'experiment_name': experiment_name,
            'best_performance': {},
            'performance_trends': {}
        }
        
        # Find best performance across all evaluations
        all_rmse = []
        all_pearson = []
        
        for eval_entry in self.evaluation_history:
            if 'overall' in eval_entry:
                all_rmse.append(eval_entry['overall']['rmse'])
                all_pearson.append(eval_entry['overall']['pearson'])
            elif 'evaluations' in eval_entry:
                for dataset_name, eval_result in eval_entry['evaluations'].items():
                    all_rmse.append(eval_result['overall']['rmse'])
                    all_pearson.append(eval_result['overall']['pearson'])
        
        if all_rmse:
            summary['best_performance'] = {
                'best_rmse': min(all_rmse),
                'best_pearson': max(all_pearson),
                'avg_rmse': np.mean(all_rmse),
                'avg_pearson': np.mean(all_pearson)
            }
        
        return summary
    
    def _log_evaluation_results(self, eval_result: Dict[str, Any]):
        """Log evaluation results to console and wandb."""
        
        overall = eval_result['overall']
        
        # Console logging
        logger.info(f"=== {eval_result['dataset_name']} {eval_result['split_type']} Evaluation ===")
        logger.info(f"RMSE: {overall['rmse']:.4f}")
        logger.info(f"Pearson: {overall['pearson']:.4f}")
        logger.info(f"Expected Loss: {overall['avg_expected_loss']:.4f}")
        logger.info(f"Total Predictions: {overall['total_predictions']}")
        
        # Question-wise logging
        for q_name, q_metrics in eval_result['by_question'].items():
            if q_metrics['count'] > 0:
                logger.debug(f"{q_name}: RMSE={q_metrics['rmse']:.4f}, Count={q_metrics['count']}")
    
    def _empty_evaluation_result(self, dataset_name: str, split_type: str) -> Dict[str, Any]:
        """Return empty evaluation result structure."""
        return {
            'dataset_name': dataset_name,
            'split_type': split_type,
            'timestamp': len(self.evaluation_history),
            'total_examples': 0,
            'processed_examples': 0,
            'overall': self._empty_metrics(),
            'by_question': {},
            'by_annotator': {},
            'target_questions': []
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            'rmse': 0.0,
            'pearson': 0.0,
            'spearman': 0.0,
            'kendall': 0.0,
            'accuracy': 0.0,
            'mae': 0.0,
            'avg_expected_loss': 0.0,
            'total_predictions': 0
        }
    
    def _empty_question_metrics(self) -> Dict[str, float]:
        """Return empty question metrics structure."""
        metrics = self._empty_metrics()
        metrics['count'] = 0
        return metrics

class TrainingMetricsTracker:
    """Track training metrics throughout active learning process."""
    
    def __init__(self, config: Config, use_wandb: bool = False):
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.metrics_history = []
        
        logger.info(f"TrainingMetricsTracker initialized - Wandb: {self.use_wandb}")
    
    def log_training_metrics(self, cycle: int, epoch: int, metrics: Dict[str, Any]):
        """Log training metrics for a specific cycle and epoch."""
        
        entry = {
            'cycle': cycle,
            'epoch': epoch,
            'timestamp': len(self.metrics_history),
            'metrics': metrics
        }
        
        self.metrics_history.append(entry)
        
        # Console logging
        logger.info(f"Cycle {cycle}, Epoch {epoch}: Loss={metrics.get('loss', 0):.4f}")
        
        # Wandb logging
        if self.use_wandb and wandb.run is not None:
            wandb_data = {
                'cycle': cycle,
                'training/epoch': epoch,
                'training/loss': metrics.get('loss', 0),
                'training/examples_trained': metrics.get('examples_trained', 0)
            }
            wandb.log(wandb_data)
    
    def log_selection_metrics(self, cycle: int, selection_metrics: Dict[str, Any]):
        """Log active learning selection metrics."""
        
        logger.info(f"Cycle {cycle} Selection: "
                   f"Examples={selection_metrics.get('examples_selected', 0)}, "
                   f"Features={selection_metrics.get('features_selected', 0)}")
        
        if self.use_wandb and wandb.run is not None:
            wandb_data = {
                'cycle': cycle,
                'training/examples_selected': selection_metrics.get('examples_selected', 0),
                'training/features_selected': selection_metrics.get('features_selected', 0),
                'training/pool_size_remaining': selection_metrics.get('pool_size_remaining', 0)
            }
            if 'benefit_cost_ratio' in selection_metrics:
                wandb_data['training/avg_benefit_cost_ratio'] = selection_metrics['benefit_cost_ratio']
            
            wandb.log(wandb_data)
    
    def save_metrics_history(self, experiment_name: str) -> str:
        """Save metrics history to file."""
        
        exp_paths = self.config.get_experiment_paths(experiment_name)
        metrics_file = os.path.join(exp_paths['results_dir'], "training_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        logger.info(f"Saved training metrics to {metrics_file}")
        return metrics_file

# Convenience functions for common evaluation patterns
def quick_evaluate(model, dataset: AnnotationDataset, config: Config, 
                  dataset_name: str = "dataset", use_wandb: bool = False) -> Dict[str, Any]:
    """Quick evaluation of a model on a dataset."""
    evaluator = ModelEvaluator(config, use_wandb)
    return evaluator.evaluate_model(model, dataset, dataset_name)

def evaluate_training_progress(model, train_dataset: AnnotationDataset, val_dataset: AnnotationDataset, 
                             test_dataset: AnnotationDataset, config: Config, cycle: int, 
                             use_wandb: bool = False) -> Dict[str, Any]:
    """Evaluate model on train/val/test datasets for training progress tracking."""
    evaluator = ModelEvaluator(config, use_wandb)
    
    datasets = {
        'train': train_dataset,
        'validation': val_dataset, 
        'test': test_dataset
    }
    
    return evaluator.evaluate_active_learning_cycle(model, datasets, cycle)