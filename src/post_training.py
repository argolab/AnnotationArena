"""
Post-training experiment that loads saved active learning results and continues training
the imputer model with all annotated features to evaluate the effect on test loss.
"""

import os
import argparse
import torch
import json
import numpy as np
from tqdm.auto import tqdm
import copy

from annotationArena import *
from utils_prabhav import AnnotationDataset, DataManager
from imputer import Imputer
from imputer_embedding import ImputerEmbedding


def load_experiment_results(results_path):
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def extract_observation_history(results):
    """Extract all observations made during the active learning process."""
    observations = []
    
    if 'observation_history' in results:
        # Parse the variable_id format: "example_{idx}_position_{pos}"
        for obs in results['observation_history']:
            if isinstance(obs, dict) and 'variable_id' in obs:
                variable_id = obs['variable_id']
                
                # Parse variable_id format: "example_310_position_10"
                if variable_id.startswith('example_') and '_position_' in variable_id:
                    try:
                        # Split by underscore and extract indices
                        parts = variable_id.split('_')
                        if len(parts) >= 4 and parts[0] == 'example' and parts[2] == 'position':
                            example_idx = int(parts[1])
                            position = int(parts[3])
                            observations.append((example_idx, position))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse variable_id '{variable_id}': {e}")
                        continue
            
            # Fallback: check for direct format
            elif isinstance(obs, dict) and 'example_idx' in obs and 'position' in obs:
                observations.append((obs['example_idx'], obs['position']))
            elif isinstance(obs, (list, tuple)) and len(obs) >= 2:
                observations.append((obs[0], obs[1]))
    
    # Fallback: try to reconstruct from other data if available
    if not observations and 'arena_training_losses' in results:
        print("Warning: No direct observation history found, attempting reconstruction...")
        # This would require more complex logic based on your specific data format
        
    return observations


def apply_annotations_from_history(arena, dataset, observations, results=None):
    """Apply all observations from the experiment history to the arena."""
    arena.set_dataset(dataset)
    
    applied_count = 0
    failed_count = 0
    
    # If we have the full observation history with values, we could potentially use them
    # For now, we'll just apply the observations and let the arena use the dataset values
    
    for example_idx, position in tqdm(observations, desc="Applying saved annotations"):
        try:
            # Register example if not already registered
            arena.register_example(example_idx, add_all_positions=False)
            
            # Apply the observation
            if arena.observe_position(example_idx, position):
                applied_count += 1
                
                # Make prediction for this variable
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"Warning: Failed to apply observation (example {example_idx}, position {position}): {e}")
            failed_count += 1
            continue
    
    print(f"Applied {applied_count} annotations from history")
    if failed_count > 0:
        print(f"Failed to apply {failed_count} annotations")
    
    return applied_count


def run_post_training_experiment(
    model_path,
    results_path,
    dataset_train,
    dataset_val, 
    dataset_test,
    epochs=10,
    batch_size=8,
    lr=1e-4,
    device=None,
    training_type='basic',
    plot_trends=True
):
    """
    Run post-training experiment using saved model and annotation history.
    
    Args:
        model_path: Path to saved model state dict
        results_path: Path to JSON file with experiment results
        dataset_train: Training dataset
        dataset_val: Validation dataset  
        dataset_test: Test dataset
        epochs: Number of additional training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: PyTorch device
        training_type: Type of training ('basic' or 'random_masking')
    
    Returns:
        Dictionary with post-training metrics
    """
    
    # Load experiment results
    print(f"Loading experiment results from {results_path}")
    results = load_experiment_results(results_path)
    
    # Extract observation history
    observations = extract_observation_history(results)
    print(f"Found {len(observations)} observations in experiment history")
    
    if len(observations) > 0:
        # Show some sample observations for verification
        print("Sample observations:")
        for i, (example_idx, position) in enumerate(observations[:5]):
            print(f"  {i+1}. Example {example_idx}, Position {position}")
        if len(observations) > 5:
            print(f"  ... and {len(observations) - 5} more")
    
    if not observations:
        print("No observations found in results file!")
        return None
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    
    # Determine model type based on saved results or model file
    use_embedding = True
    
    # Initialize model (you'll need to adjust these parameters based on your dataset)
    if use_embedding:
        ModelClass = ImputerEmbedding
    else:
        ModelClass = Imputer
    
    # You may need to adjust these parameters based on your specific dataset
    if 'hanna' in results_path or 'hanna' in model_path:
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    elif 'llm_rubric' in results_path or 'llm_rubric' in model_path:
        model = ModelClass(
            question_num=9, max_choices=4, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=24, 
            annotator_embedding_dim=24, dropout=0.1
        ).to(device)
    else:
        # Default parameters - you should adjust these
        print("Warning: Using default model parameters. Please verify these are correct.")
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create arena and apply all annotations
    arena = AnnotationArena(model, device)
    applied_count = apply_annotations_from_history(arena, dataset_train, observations, results)
    
    # Evaluate before additional training
    print("\nEvaluating before additional training...")
    
    arena.set_dataset(dataset_val)
    val_metrics_before = arena.evaluate(list(range(len(dataset_val))))
    
    arena.set_dataset(dataset_test)
    test_metrics_before = arena.evaluate(list(range(len(dataset_test))))
    
    print(f"Before additional training:")
    print(f"  Validation Loss: {val_metrics_before['avg_expected_loss']:.4f}")
    print(f"  Test Loss: {test_metrics_before['avg_expected_loss']:.4f}")
    
    # Perform additional training epoch by epoch and track metrics
    print(f"\nPerforming incremental training for {epochs} epochs...")
    arena.set_dataset(dataset_train)
    
    # Initialize tracking lists
    epoch_numbers = [0]  # Start with epoch 0 (before training)
    val_losses = [val_metrics_before['avg_expected_loss']]
    val_rmses = [val_metrics_before['rmse']]
    test_losses = [test_metrics_before['avg_expected_loss']]
    test_rmses = [test_metrics_before['rmse']]
    
    # Train one epoch at a time and evaluate
    for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
        # Train for 1 epoch
        epoch_history = arena.train(
            epochs=1, 
            batch_size=batch_size, 
            lr=lr, 
            training_type=training_type
        )
        
        # Evaluate on validation set
        arena.set_dataset(dataset_val)
        val_metrics_current = arena.evaluate(list(range(len(dataset_val))))
        
        # Evaluate on test set
        arena.set_dataset(dataset_test)
        test_metrics_current = arena.evaluate(list(range(len(dataset_test))))
        
        # Store metrics
        epoch_numbers.append(epoch)
        val_losses.append(val_metrics_current['avg_expected_loss'])
        val_rmses.append(val_metrics_current['rmse'])
        test_losses.append(test_metrics_current['avg_expected_loss'])
        test_rmses.append(test_metrics_current['rmse'])
        
        # Print progress every few epochs
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Val RMSE={val_metrics_current['rmse']:.4f}, "
                  f"Test RMSE={test_metrics_current['rmse']:.4f}")
        
        # Reset dataset back to training for next iteration
        arena.set_dataset(dataset_train)
    
    # Final evaluation metrics
    val_metrics_after = {'avg_expected_loss': val_losses[-1], 'rmse': val_rmses[-1]}
    test_metrics_after = {'avg_expected_loss': test_losses[-1], 'rmse': test_rmses[-1]}
    
    # Get final complete metrics
    arena.set_dataset(dataset_val)
    val_metrics_after_complete = arena.evaluate(list(range(len(dataset_val))))
    
    arena.set_dataset(dataset_test)
    test_metrics_after_complete = arena.evaluate(list(range(len(dataset_test))))
    
    print(f"After additional training:")
    print(f"  Validation Loss: {val_metrics_after_complete['avg_expected_loss']:.4f}")
    print(f"  Test Loss: {test_metrics_after_complete['avg_expected_loss']:.4f}")
    print(f"  Validation RMSE: {val_metrics_after_complete['rmse']:.4f}")
    print(f"  Test RMSE: {test_metrics_after_complete['rmse']:.4f}")
    
    # Calculate improvements
    val_improvement = val_metrics_before['avg_expected_loss'] - val_metrics_after_complete['avg_expected_loss']
    test_improvement = test_metrics_before['avg_expected_loss'] - test_metrics_after_complete['avg_expected_loss']
    val_rmse_improvement = val_metrics_before['rmse'] - val_metrics_after_complete['rmse']
    test_rmse_improvement = test_metrics_before['rmse'] - test_metrics_after_complete['rmse']
    
    print(f"\nImprovements:")
    print(f"  Validation Loss: {val_improvement:.4f} ({'improved' if val_improvement > 0 else 'degraded'})")
    print(f"  Test Loss: {test_improvement:.4f} ({'improved' if test_improvement > 0 else 'degraded'})")
    print(f"  Validation RMSE: {val_rmse_improvement:.4f} ({'improved' if val_rmse_improvement > 0 else 'degraded'})")
    print(f"  Test RMSE: {test_rmse_improvement:.4f} ({'improved' if test_rmse_improvement > 0 else 'degraded'})")
    
    # Plot trends if requested
    if plot_trends:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot Test RMSE (main focus)
        ax1.plot(epoch_numbers, test_rmses, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Test RMSE')
        ax1.set_title('Test RMSE vs Training Epochs')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot Validation RMSE
        ax2.plot(epoch_numbers, val_rmses, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation RMSE')
        ax2.set_title('Validation RMSE vs Training Epochs')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # Plot Test Loss
        ax3.plot(epoch_numbers, test_losses, 'r-', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Loss')
        ax3.set_title('Test Loss vs Training Epochs')
        ax3.grid(True, alpha=0.3)
        
        # Plot Validation Loss
        ax4.plot(epoch_numbers, val_losses, 'm-', linewidth=2, marker='d', markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss')
        ax4.set_title('Validation Loss vs Training Epochs')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"post_training_trends_{os.path.basename(results_path).replace('.json', '')}.png"
        plot_path = os.path.join(os.path.dirname(results_path), plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nTrend plots saved to: {plot_path}")
        
        # Also create a focused plot just for Test RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_numbers, test_rmses, 'b-', linewidth=3, marker='o', markersize=6)
        plt.xlabel('Training Epoch', fontsize=12)
        plt.ylabel('Test RMSE', fontsize=12)
        plt.title('Test RMSE Trend During Post-Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        # Add improvement annotation
        if len(test_rmses) > 1:
            improvement_pct = ((test_rmses[0] - test_rmses[-1]) / test_rmses[0]) * 100
            plt.text(0.02, 0.98, f'Total RMSE Change: {test_rmse_improvement:.4f}\n({improvement_pct:+.2f}%)', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save focused plot
        focused_plot_filename = f"test_rmse_trend_{os.path.basename(results_path).replace('.json', '')}.png"
        focused_plot_path = os.path.join(os.path.dirname(results_path), focused_plot_filename)
        plt.savefig(focused_plot_path, dpi=300, bbox_inches='tight')
        print(f"Test RMSE trend plot saved to: {focused_plot_path}")
        
        plt.show()
    
    # Compile results
    post_training_results = {
        'experiment_info': {
            'model_path': model_path,
            'results_path': results_path,
            'annotations_applied': applied_count,
            'additional_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'training_type': training_type
        },
        'metrics_before': {
            'validation': val_metrics_before,
            'test': test_metrics_before
        },
        'metrics_after': {
            'validation': val_metrics_after_complete,
            'test': test_metrics_after_complete
        },
        'epoch_by_epoch_metrics': {
            'epochs': epoch_numbers,
            'val_losses': val_losses,
            'val_rmses': val_rmses,
            'test_losses': test_losses,
            'test_rmses': test_rmses
        },
        'improvements': {
            'validation_loss': val_improvement,
            'test_loss': test_improvement,
            'validation_rmse': val_rmse_improvement,
            'test_rmse': test_rmse_improvement
        },
        'original_results_summary': {
            'final_val_loss': results.get('val_losses', [])[-1] if results.get('val_losses') else None,
            'final_test_loss': results.get('test_expected_losses', [])[-1] if results.get('test_expected_losses') else None,
            'total_examples_annotated': sum(results.get('examples_annotated', [])),
            'total_features_annotated': sum(results.get('features_annotated', []))
        }
    }
    
    return post_training_results


def main():
    parser = argparse.ArgumentParser(description="Run post-training experiment with saved active learning results.")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model state dict (.pth file)")
    parser.add_argument("--results_path", type=str, required=True,
                       help="Path to JSON file with experiment results")
    parser.add_argument("--dataset", type=str, default="hanna",
                       help="Dataset name (hanna or llm_rubric)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of additional training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--training_type", type=str, default="basic",
                       help="Training type (basic or random_masking)")
    parser.add_argument("--output_dir", type=str, default="outputs/post_training",
                       help="Directory to save post-training results")
    parser.add_argument("--runner", type=str, default="local",
                       help="Runner identifier")
    parser.add_argument("--use_embedding", action="store_true",
                       help="Use embedding model")
    parser.add_argument("--plot_trends", action="store_true", default=True,
                       help="Plot training trends")
    
    args = parser.parse_args()
    
    # Setup paths
    if args.runner == 'prabhav':
        base_path = '/export/fs06/psingh54/ActiveRubric-Internal/outputs'
    else:
        base_path = "outputs"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    if args.runner == "prabhav":
        data_manager = DataManager(base_path + '/data/')
    else:
        data_manager = DataManager(base_path + f'/data_{args.dataset}/')

    if args.dataset == "hanna":
        data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, 
                                dataset=args.dataset, cold_start=True, 
                                use_embedding=args.use_embedding)
    elif args.dataset == "llm_rubric":
        data_manager.prepare_data(num_partition=1000, initial_train_ratio=0.0, 
                                dataset=args.dataset, cold_start=True, 
                                use_embedding=args.use_embedding)

    train_dataset = AnnotationDataset(data_manager.paths['train'])
    val_dataset = AnnotationDataset(data_manager.paths['validation'])
    test_dataset = AnnotationDataset(data_manager.paths['test'])
    active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
    
    # Use active pool as training dataset if train_dataset is empty
    if len(train_dataset) == 0:
        train_dataset = active_pool_dataset
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Run post-training experiment
    results = run_post_training_experiment(
        model_path=args.model_path,
        results_path=args.results_path,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        training_type=args.training_type,
        plot_trends=args.plot_trends
    )
    
    if results:
        # Save results
        output_filename = f"post_training_results_{os.path.basename(args.results_path).replace('.json', '')}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nPost-training results saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"POST-TRAINING EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Annotations applied: {results['experiment_info']['annotations_applied']}")
        print(f"Additional epochs: {results['experiment_info']['additional_epochs']}")
        print(f"Validation loss improvement: {results['improvements']['validation_loss']:.4f}")
        print(f"Test loss improvement: {results['improvements']['test_loss']:.4f}")
        print(f"Validation RMSE improvement: {results['improvements']['validation_rmse']:.4f}")
        print(f"Test RMSE improvement: {results['improvements']['test_rmse']:.4f}")
        
        # Print epoch-by-epoch summary
        if 'epoch_by_epoch_metrics' in results:
            test_rmses = results['epoch_by_epoch_metrics']['test_rmses']
            print(f"Test RMSE progression: {test_rmses[0]:.4f} -> {test_rmses[-1]:.4f}")
            
            # Find best epoch
            best_epoch = np.argmin(test_rmses)
            best_rmse = test_rmses[best_epoch]
            print(f"Best Test RMSE: {best_rmse:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()