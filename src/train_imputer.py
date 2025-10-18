"""
Standalone training script for ImputerEmbedding without active learning.
Trains the model using dynamic masking on a fixed dataset.

Author: Based on Prabhav Singh / Haojun Shi's active learning framework
"""

import os
import argparse
import torch
import json
import random
import numpy as np
import logging
import time
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Import your existing modules
from config import Config, ModelConfig, DefaultHyperparams
from utils import AnnotationDataset, DataManager, compute_metrics, minimum_bayes_risk_l2, minimum_bayes_risk_ce
from imputer import ImputerEmbedding

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set random seeds for reproducibility
random.seed(90)
torch.manual_seed(90)
np.random.seed(90)
import ipdb


def collate_batch(batch):
    """Collate function for DataLoader."""
    batch_size = len(batch)
    
    # Extract components from batch
    known_questions_list = []
    inputs_list = []
    answers_list = []
    annotators_list = []
    questions_list = []
    embeddings_list = []
    
    for item in batch:
        if len(item) == 6:  # With embeddings
            known_questions, inputs, answers, annotators, questions, embeddings = item
        else:  # Without embeddings
            known_questions, inputs, answers, annotators, questions = item
            embeddings = None
            
        known_questions_list.append(known_questions)
        inputs_list.append(inputs)
        answers_list.append(answers)
        annotators_list.append(annotators)
        questions_list.append(questions)
        embeddings_list.append(embeddings)
    
    # Stack tensors
    batch_dict = {
        'known_questions': torch.stack(known_questions_list),
        'inputs': torch.stack(inputs_list),
        'answers': torch.stack(answers_list),
        'annotators': torch.stack(annotators_list),
        'questions': torch.stack(questions_list)
    }
    
    # Handle embeddings (might be None)
    if embeddings_list[0] is not None:
        batch_dict['embeddings'] = torch.stack(embeddings_list)
    else:
        batch_dict['embeddings'] = None
        
    return batch_dict


def evaluate_model(model, dataset, device, batch_size=32, loss_type="cross_entropy"):
    """Evaluate model performance on a dataset using comprehensive metrics."""
    model.eval()
    
    eval_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = batch['inputs'].to(device)
            annotators = batch['annotators'].to(device)
            questions = batch['questions'].to(device)
            embeddings = batch['embeddings'].to(device) if batch['embeddings'] is not None else None
            targets = batch['answers'].to(device)
            # Forward pass
            outputs = model(inputs, annotators, questions, embeddings)
            # FIXED: Only evaluate on masked positions (7-12 for test data)
            mask = (inputs[:, :, 0] == 1).bool()  # Positions where mask bit is 1
            
            # Flatten and select only masked positions
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes)
            targets_flat = targets.view(-1, num_classes)
            mask_flat = mask.view(-1)
            
            # Select only masked positions
            outputs_masked = outputs_flat[mask_flat]
            targets_masked = targets_flat[mask_flat]
            
            if outputs_masked.size(0) > 0:
                # Compute loss only on masked positions
                loss = F.cross_entropy(
                    outputs_masked,
                    torch.argmax(targets_masked, dim=-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_samples += outputs_masked.size(0)
                
                # Collect predictions for metrics
                pred_probs = torch.softmax(outputs_masked, dim=-1)
                for i in range(pred_probs.size(0)):
                    pred_dist = pred_probs[i].cpu().numpy()
                    true_dist = targets_masked[i].cpu().numpy()
                    
                    pred_scalar = sum((j + 1) * pred_dist[j] for j in range(len(pred_dist)))
                    true_scalar = np.argmax(true_dist) + 1
                    
                    all_predictions.append(pred_scalar)
                    all_targets.append(true_scalar)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    # Compute comprehensive metrics
    if all_predictions and all_targets:
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        metrics = compute_metrics(predictions_array, targets_array)
    else:
        metrics = {
            "rmse": 0.0, "mae": 0.0, "pearson": 0.0, 
            "spearman": 0.0, "kendall": 0.0, "accuracy": 0.0
        }
    
    metrics.update({
        'avg_expected_loss': avg_loss,
        'total_samples': total_samples,
        'total_predictions': len(all_predictions)
    })
    
    return metrics


def train_with_dynamic_masking(model, train_dataset, val_dataset=None,
                             epochs=10, batch_size=16, lr=1e-4,
                             num_patterns_per_example=5, visible_ratio=0.5,
                             device=None, use_wandb=False, save_path=None,
                             eval_every=1, patience=5, loss_type="cross_entropy"):
    """
    Train the model using dynamic masking strategy.
    
    Args:
        model: ImputerEmbedding model to train
        train_dataset: Training dataset (AnnotationDataset)
        val_dataset: Validation dataset (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        num_patterns_per_example: Number of masking patterns per example
        visible_ratio: Ratio of observed positions to keep visible
        device: Training device
        use_wandb: Whether to use wandb for logging
        save_path: Path to save the best model
        eval_every: Evaluate every N epochs
        patience: Early stopping patience
        loss_type: Loss type for evaluation ("cross_entropy" or "l2")
    
    Returns:
        Dictionary containing training history and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.set_dataset(train_dataset)  # Required for the training methods
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with dynamic masking on device: {device}")
    logger.info(f"Training set size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Training history
    history = {
        'train_losses': [],
        'val_metrics': [],
        'epochs': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Early stopping
    patience_counter = 0
    
    # Create a simple training queue with all examples
    model.training_queue = []
    model.unique_examples = []
    model.recent_indicators = []
    
    for idx in range(len(train_dataset)):
        queue_entry = {
            'example_idx': idx,
            'positions': list(range(len(train_dataset.get_data_entry(idx)['input']))),
            'weight': 1.0,
            'timestamp': idx,
            'needs_revisit': False
        }
        model.training_queue.append(queue_entry)
        model.unique_examples.append(idx)
        model.recent_indicators.append(True)
    
    logger.info(f"Created training queue with {len(model.training_queue)} entries")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch using the model's dynamic masking method
        train_losses = model.train_on_examples_dynamic_masking(
            examples_indices=None,  # Use all examples
            epochs=1,  # Single epoch per call
            batch_size=batch_size,
            lr=lr,
            num_patterns_per_example=1,
            visible_ratio=visible_ratio
        )
        
        avg_train_loss = train_losses[0] if train_losses else 0.0
        history['train_losses'].append(avg_train_loss)
        
        # Validation evaluation
        if val_dataset and epoch % eval_every == 0:
            val_metrics = evaluate_model(model, val_dataset, device, batch_size, loss_type)
            val_loss = val_metrics['avg_expected_loss']
            
            history['val_metrics'].append(val_metrics)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                       f"Pearson: {val_metrics['pearson']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping and model saving
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'val_metrics': val_metrics,
                        'history': history
                    }, save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb_data = {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_pearson': val_metrics['pearson'],
                    'val_spearman': val_metrics['spearman'],
                    'val_kendall': val_metrics['kendall'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': lr
                }
                wandb.log(wandb_data)
        else:
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'learning_rate': lr
                })
        
        history['epochs'].append(epoch)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']+1}")
    
    return history


def plot_training_history(history, save_path=None):
    """Plot comprehensive training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['epochs'], history['train_losses'], label='Train Loss', marker='o')
    if history['val_metrics']:
        # Val losses might be recorded less frequently
        val_epochs = [i * len(history['epochs']) // len(history['val_metrics']) for i in range(len(history['val_metrics']))]
        val_losses = [m['avg_expected_loss'] for m in history['val_metrics']]
        axes[0, 0].plot(val_epochs, val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # RMSE and Accuracy
    if history['val_metrics']:
        val_epochs = [i * len(history['epochs']) // len(history['val_metrics']) for i in range(len(history['val_metrics']))]
        rmse_values = [m['rmse'] for m in history['val_metrics']]
        accuracy_values = [m['accuracy'] for m in history['val_metrics']]
        
        axes[0, 1].plot(val_epochs, rmse_values, label='RMSE', marker='s', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(val_epochs, accuracy_values, label='Accuracy', marker='s', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[0, 1].text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Validation RMSE (No Data)')
        axes[1, 0].text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Validation Accuracy (No Data)')
    
    # Correlation metrics
    if history['val_metrics']:
        pearson_values = [m['pearson'] for m in history['val_metrics']]
        spearman_values = [m['spearman'] for m in history['val_metrics']]
        
        axes[1, 1].plot(val_epochs, pearson_values, label='Pearson', marker='s', color='blue')
        axes[1, 1].plot(val_epochs, spearman_values, label='Spearman', marker='^', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('Validation Correlations')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Validation Correlations (No Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Standalone training for ImputerEmbedding")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="hanna", 
                       choices=["hanna", "llm_rubric"],
                       help="Dataset to use")
    parser.add_argument("--runner", type=str, default="local", 
                       help="Runner identifier for config")
    parser.add_argument("--use_embedding", action="store_true", default=False,
                       help="Use text embeddings")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--num_patterns_per_example", type=int, default=5,
                       help="Number of masking patterns per example")
    parser.add_argument("--visible_ratio", type=float, default=0.5,
                       help="Ratio of observed positions to keep visible")
    parser.add_argument("--eval_every", type=int, default=1,
                       help="Evaluate every N epochs")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    parser.add_argument("--loss_type", type=str, default="cross_entropy",
                       choices=["cross_entropy", "l2"],
                       help="Loss type for evaluation metrics")
    
    # Data preparation arguments
    parser.add_argument("--num_partition", type=int, default=1200,
                       help="Number of examples to use (hanna) or use default for llm_rubric")
    parser.add_argument("--initial_train_ratio", type=float, default=0.9,
                       help="Ratio of data to use for training (rest goes to active pool)")
    
    # Logging and saving
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="imputer-training",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str,
                       help="Wandb entity name")
    parser.add_argument("--experiment_name", type=str,
                       help="Experiment name for logging and file naming")
    parser.add_argument("--save_model", action="store_true", default=True,
                       help="Save the best model")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"imputer_training_{args.dataset}"
    
    # Initialize config
    config = Config(args.runner, args.dataset)
    config.ensure_directories()
    
    # Setup logging
    exp_paths = config.get_experiment_paths(args.experiment_name)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_paths['log_file']),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_config = vars(args).copy()
        wandb_config.update({
            'config_timestamp': config.timestamp,
            'base_path': config.BASE_PATH,
        })
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.experiment_name}_{config.timestamp}",
            config=wandb_config
        )
        logger.info("Wandb initialized")
    elif args.use_wandb:
        logger.warning("Wandb requested but not available")
    
    # Prepare data
    data_manager = DataManager(config)
    
    # Use different partition sizes for different datasets
    if args.dataset == "hanna":
        num_partition = args.num_partition
    elif args.dataset == "llm_rubric":
        num_partition = 225
    
    data_manager.prepare_data(
        num_partition=num_partition, 
        initial_train_ratio=0.9, 
        dataset=args.dataset,
        cold_start=False,  # We want observed data for training
        use_embedding=False
    )
    # Load datasets
    train_dataset = AnnotationDataset(data_manager.paths['train'])
    val_dataset = AnnotationDataset(data_manager.paths['validation'])
    test_dataset = AnnotationDataset(data_manager.paths['test'])
    
    logger.info(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Initialize model
    model_config = ModelConfig.get_config(args.dataset, training_buffer_size=None)
    model = ImputerEmbedding(**model_config).to(device)
    logger.info(f"Model initialized with config: {model_config}")
    
    # Train the model
    save_path = exp_paths['model_file'] if args.save_model else None
    
    for i in range(args.epochs):
        history = train_with_dynamic_masking(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            num_patterns_per_example=args.num_patterns_per_example,
            visible_ratio=args.visible_ratio,
            device=device,
            use_wandb=args.use_wandb,
            save_path=save_path,
            eval_every=args.eval_every,
            patience=args.patience,
            loss_type=args.loss_type
        )
        
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, test_dataset, device, args.batch_size, args.loss_type)
        logger.info(f"Test Results: Loss={test_metrics['avg_expected_loss']:.4f}, "
                f"RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}, "
                f"Accuracy={test_metrics['accuracy']:.4f}")
    
    # Save training history
    history_file = os.path.join(exp_paths['results_dir'], "training_history.json")
    
    # Convert numpy values to regular Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    history_serializable = convert_for_json(history)
    test_metrics_serializable = convert_for_json(test_metrics)
    
    with open(history_file, "w") as f:
        json.dump(history_serializable, f, indent=4)
    logger.info(f"Training history saved to {history_file}")
    
    # Plot training history
    plot_path = os.path.join(exp_paths['results_dir'], "training_plot.png")
    plot_training_history(history, save_path=plot_path)
    
    # Final summary
    final_val_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}
    results_summary = {
        'experiment_name': args.experiment_name,
        'dataset': args.dataset,
        'loss_type': args.loss_type,
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else None,
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'],
        'final_val_metrics': final_val_metrics,
        'test_metrics': test_metrics_serializable,
        'total_epochs': len(history['epochs']),
        'model_config': model_config,
        'training_args': vars(args)
    }
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_file}")
    
    # Plot training history
    plot_path = os.path.join(exp_paths['results_dir'], "training_plot.png")
    plot_training_history(history, save_path=plot_path)
    
    # Final summary
    results_summary = {
        'experiment_name': args.experiment_name,
        'dataset': args.dataset,
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else None,
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'],
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'total_epochs': len(history['epochs']),
        'model_config': model_config,
        'training_args': vars(args)
    }
    
    summary_file = os.path.join(exp_paths['results_dir'], "experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Experiment summary saved to {summary_file}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'final_test_loss': test_metrics['loss'],
            'final_test_accuracy': test_metrics['accuracy']
        })
        wandb.finish()
    
    logger.info("Training completed successfully!")
    return results_summary


if __name__ == "__main__":
    main()