"""
Evaluation {Local and WandB} for Active Learner framework.

Author: Prabhav Singh / Haojun Shi
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from utils import compute_metrics, AnnotationDataset
from config import Config

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
            eval_result = self.evaluate_model(model, dataset, dataset_name, split_type=dataset_name)
            cycle_results['evaluations'][dataset_name] = eval_result
        
        # Add additional metrics if provided
        if additional_metrics:
            cycle_results['additional_metrics'] = additional_metrics
            logger.debug(f"Added {len(additional_metrics)} additional metrics")
        
        # Log cycle summary to wandb
        if self.use_wandb and wandb.run is not None:
            wandb_metrics = {f"cycle_{cycle_num}": cycle_num}
            
            for dataset_name, eval_result in cycle_results['evaluations'].items():
                prefix = f"{dataset_name}_"
                wandb_metrics.update({
                    f"{prefix}rmse": eval_result['overall']['rmse'],
                    f"{prefix}pearson": eval_result['overall']['pearson'],
                    f"{prefix}expected_loss": eval_result['overall']['avg_expected_loss'],
                    f"{prefix}predictions": eval_result['overall']['total_predictions']
                })
            
            if additional_metrics:
                wandb_metrics.update({f"cycle_{k}": v for k, v in additional_metrics.items()})
            
            wandb.log(wandb_metrics)
        
        self.evaluation_history.append(cycle_results)
        
        return cycle_results
    
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
        
        # Wandb logging
        if self.use_wandb and wandb.run is not None:
            wandb_data = {
                f"{eval_result['dataset_name']}_rmse": overall['rmse'],
                f"{eval_result['dataset_name']}_pearson": overall['pearson'],
                f"{eval_result['dataset_name']}_expected_loss": overall['avg_expected_loss'],
                f"{eval_result['dataset_name']}_predictions": overall['total_predictions']
            }
            
            # Add question-wise metrics
            for q_name, q_metrics in eval_result['by_question'].items():
                if q_metrics['count'] > 0:
                    wandb_data[f"{eval_result['dataset_name']}_{q_name}_rmse"] = q_metrics['rmse']
                    wandb_data[f"{eval_result['dataset_name']}_{q_name}_count"] = q_metrics['count']
            
            wandb.log(wandb_data)
    
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
                'epoch': epoch,
                'training_loss': metrics.get('loss', 0),
                'examples_trained': metrics.get('examples_trained', 0)
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
                'examples_selected': selection_metrics.get('examples_selected', 0),
                'features_selected': selection_metrics.get('features_selected', 0),
                'pool_size_remaining': selection_metrics.get('pool_size_remaining', 0)
            }
            if 'benefit_cost_ratio' in selection_metrics:
                wandb_data['avg_benefit_cost_ratio'] = selection_metrics['benefit_cost_ratio']
            
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