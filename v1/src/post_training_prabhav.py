"""
Direct model comparison script with comprehensive evaluation and visualization.
Loads observation history and trains fresh models with different training methods.
"""

import os
import argparse
import torch
import json
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from annotationArena import AnnotationArena
from utils_prabhav import AnnotationDataset, DataManager, compute_metrics
from imputer import Imputer
from imputer_embedding import ImputerEmbedding

kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

def set_seeds(seed=43):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_experiment_results(results_path):
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def extract_observation_history(results):
    """Extract all observations made during the active learning process."""
    observations = []
    
    if 'observation_history' in results:
        for obs in results['observation_history']:
            if isinstance(obs, dict) and 'variable_id' in obs:
                variable_id = obs['variable_id']
                
                if variable_id.startswith('example_') and '_position_' in variable_id:
                    try:
                        parts = variable_id.split('_')
                        if len(parts) >= 4 and parts[0] == 'example' and parts[2] == 'position':
                            example_idx = int(parts[1])
                            position = int(parts[3])
                            observations.append((example_idx, position))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse variable_id '{variable_id}': {e}")
                        continue
    
    return observations


def create_fresh_model(dataset_name, use_embedding, device, seed=43):
    """Create a fresh model with fixed random seed."""
    set_seeds(seed)
    
    if use_embedding:
        ModelClass = ImputerEmbedding
    else:
        ModelClass = Imputer
    
    if dataset_name == "hanna":
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    elif dataset_name == "llm_rubric":
        model = ModelClass(
            question_num=9, max_choices=4, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=24, 
            annotator_embedding_dim=24, dropout=0.1
        ).to(device)
    else:
        print("Warning: Using default model parameters")
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    
    return model


def apply_observations_to_model(model, dataset, observations, device):
    """Apply observations to model using Arena for consistency with training data."""
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset)
    
    applied_count = 0
    for example_idx, position in tqdm(observations, desc="Applying observations"):
        try:
            arena.register_example(example_idx, add_all_positions=False)
            
            if arena.observe_position(example_idx, position):
                applied_count += 1
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        except Exception as e:
            print(f"Warning: Failed to apply observation (example {example_idx}, position {position}): {e}")
            continue
    
    print(f"Applied {applied_count} observations")
    return arena, applied_count


def evaluate_model_directly(model, dataset, device, max_positions=14):
    """
    Directly evaluate model on dataset without Arena overhead.
    
    Args:
        model: The imputer model
        dataset: Dataset to evaluate on
        device: Device to use
        max_positions: Maximum number of positions (14 for HANNA)
        
    Returns:
        dict: Comprehensive evaluation metrics
    """
    model.eval()
    
    # Overall metrics (all positions combined)
    all_preds = []
    all_true = []
    
    # Per-position metrics 
    position_preds = [[] for _ in range(max_positions)]
    position_true = [[] for _ in range(max_positions)]
    
    # Per-question metrics (questions 0-6)
    question_preds = [[] for _ in range(7)]
    question_true = [[] for _ in range(7)]
    
    with torch.no_grad():
        for example_idx in range(len(dataset)):
            # Get example data
            known_questions, inputs, answers, annotators, questions, embeddings = dataset[example_idx]
            
            # Move to device
            inputs = inputs.unsqueeze(0).to(device)
            annotators = annotators.unsqueeze(0).to(device) 
            questions = questions.unsqueeze(0).to(device)
            if embeddings is not None:
                embeddings = embeddings.unsqueeze(0).to(device)
            
            # Get model predictions for ALL positions at once
            outputs = model(inputs, annotators, questions, embeddings)  # [1, max_positions, num_classes]
            outputs = outputs.squeeze(0)  # [max_positions, num_classes]
            questions_squeezed = questions.squeeze(0)  # [max_positions]
            
            # Process each position
            for pos in range(min(max_positions, len(answers))):
                if pos < len(answers) and pos < len(questions_squeezed):
                    # Convert output to prediction (expected value for L2 loss)
                    probs = torch.softmax(outputs[pos], dim=0)
                    pred_score = torch.sum(probs * torch.arange(1, 6, device=device).float()).item()
                    
                    # Get ground truth
                    true_label = torch.argmax(answers[pos]).item()
                    true_score = true_label + 1  # Convert to 1-5 scale
                    
                    # Add to overall lists
                    all_preds.append(pred_score)
                    all_true.append(true_score)
                    
                    # Add to position-specific lists
                    position_preds[pos].append(pred_score)
                    position_true[pos].append(true_score)
                    
                    # Add to question-specific lists
                    question_idx = questions_squeezed[pos].item() if pos < len(questions_squeezed) else pos % 7
                    if 0 <= question_idx < 7:
                        question_preds[question_idx].append(pred_score)
                        question_true[question_idx].append(true_score)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(np.array(all_preds), np.array(all_true))
    overall_metrics['total_predictions'] = len(all_preds)
    
    # Compute per-position metrics
    position_metrics = {}
    for pos in range(max_positions):
        if len(position_preds[pos]) > 0:
            pos_metrics = compute_metrics(np.array(position_preds[pos]), np.array(position_true[pos]))
            pos_metrics['count'] = len(position_preds[pos])
            position_metrics[f'position_{pos}'] = pos_metrics
        else:
            position_metrics[f'position_{pos}'] = {'count': 0, 'rmse': float('nan')}
    
    # Compute per-question metrics
    question_metrics = {}
    for q_idx in range(7):
        if len(question_preds[q_idx]) > 0:
            q_metrics = compute_metrics(np.array(question_preds[q_idx]), np.array(question_true[q_idx]))
            q_metrics['count'] = len(question_preds[q_idx])
            question_metrics[f'Q{q_idx}'] = q_metrics
        else:
            question_metrics[f'Q{q_idx}'] = {'count': 0, 'rmse': float('nan')}
    
    return {
        'overall': overall_metrics,
        'by_position': position_metrics,
        'by_question': question_metrics,
        'raw_predictions': {'all_preds': all_preds, 'all_true': all_true}
    }


def train_model_with_method(arena, method, epochs, batch_size, lr, num_patterns_per_example=5, visible_ratio=0.5):
    """Train model with specified method and track progress."""
    if method == 'dynamic_masking':
        arena.set_dynamic_masking_params(num_patterns_per_example, visible_ratio)
    
    # Track training progress
    training_losses = []
    
    for epoch in range(epochs):
        training_metrics = arena.train(
            epochs=1, 
            batch_size=batch_size, 
            lr=lr, 
            training_type=method
        )
        
        avg_loss = training_metrics.get('avg_loss', 0.0)
        training_losses.append(avg_loss)
    
    return training_losses


def create_overall_plots(results, save_path):
    """Create plots for overall metrics comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'random_masking': 'red', 'dynamic_masking': 'green'}
    markers = {'random_masking': 's', 'dynamic_masking': '^'}
    
    methods = list(results.keys())
    
    # Test RMSE over epochs
    for method in methods:
        data = results[method]
        epochs = list(range(len(data['test_rmse_progress'])))
        ax1.plot(epochs, data['test_rmse_progress'], 
                color=colors[method], marker=markers[method], 
                linewidth=2, markersize=4, label=method.replace('_', ' ').title())
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test RMSE')
    ax1.set_title('Test RMSE Progress (Overall)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation RMSE over epochs
    for method in methods:
        data = results[method]
        epochs = list(range(len(data['val_rmse_progress'])))
        ax2.plot(epochs, data['val_rmse_progress'], 
                color=colors[method], marker=markers[method], 
                linewidth=2, markersize=4, label=method.replace('_', ' ').title())
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation RMSE')
    ax2.set_title('Validation RMSE Progress (Overall)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Final metrics comparison
    final_test_rmse = [results[method]['final_metrics']['test']['overall']['rmse'] for method in methods]
    final_val_rmse = [results[method]['final_metrics']['val']['overall']['rmse'] for method in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax3.bar(x_pos - width/2, final_test_rmse, width, label='Test RMSE', 
            color=[colors[method] for method in methods], alpha=0.7)
    ax3.bar(x_pos + width/2, final_val_rmse, width, label='Validation RMSE',
            color=[colors[method] for method in methods], alpha=0.4)
    
    ax3.set_xlabel('Training Method')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Final RMSE Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([method.replace('_', ' ').title() for method in methods])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Correlation comparison
    final_test_pearson = [results[method]['final_metrics']['test']['overall']['pearson'] for method in methods]
    final_val_pearson = [results[method]['final_metrics']['val']['overall']['pearson'] for method in methods]
    
    ax4.bar(x_pos - width/2, final_test_pearson, width, label='Test Pearson', 
            color=[colors[method] for method in methods], alpha=0.7)
    ax4.bar(x_pos + width/2, final_val_pearson, width, label='Validation Pearson',
            color=[colors[method] for method in methods], alpha=0.4)
    
    ax4.set_xlabel('Training Method')
    ax4.set_ylabel('Pearson Correlation')
    ax4.set_title('Final Correlation Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([method.replace('_', ' ').title() for method in methods])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_position_plots(results, save_path):
    """Create plots for position-specific metrics."""
    methods = list(results.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'random_masking': 'red', 'dynamic_masking': 'green'}
    
    # Position-wise RMSE comparison for test set
    positions = list(range(14))
    
    for method in methods:
        test_rmse_by_pos = []
        for pos in positions:
            pos_key = f'position_{pos}'
            rmse = results[method]['final_metrics']['test']['by_position'].get(pos_key, {}).get('rmse', float('nan'))
            test_rmse_by_pos.append(rmse)
        
        ax1.plot(positions, test_rmse_by_pos, 'o-', color=colors[method], 
                linewidth=2, markersize=6, label=method.replace('_', ' ').title())
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Test RMSE')
    ax1.set_title('Test RMSE by Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(positions)
    
    # Position-wise correlation comparison for test set
    for method in methods:
        test_pearson_by_pos = []
        for pos in positions:
            pos_key = f'position_{pos}'
            pearson = results[method]['final_metrics']['test']['by_position'].get(pos_key, {}).get('pearson', float('nan'))
            test_pearson_by_pos.append(pearson)
        
        ax2.plot(positions, test_pearson_by_pos, 'o-', color=colors[method], 
                linewidth=2, markersize=6, label=method.replace('_', ' ').title())
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Test Pearson Correlation')
    ax2.set_title('Test Correlation by Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(positions)
    
    # Position counts
    for method in methods:
        test_counts_by_pos = []
        for pos in positions:
            pos_key = f'position_{pos}'
            count = results[method]['final_metrics']['test']['by_position'].get(pos_key, {}).get('count', 0)
            test_counts_by_pos.append(count)
        
        ax3.bar([p + 0.2*list(methods).index(method) for p in positions], test_counts_by_pos, 
               width=0.2, color=colors[method], alpha=0.7, label=method.replace('_', ' ').title())
    
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Number of Examples')
    ax3.set_title('Test Examples per Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(positions)
    
    # RMSE improvement by position
    if len(methods) == 2:
        method1, method2 = methods
        rmse_improvements = []
        for pos in positions:
            pos_key = f'position_{pos}'
            rmse1 = results[method1]['final_metrics']['test']['by_position'].get(pos_key, {}).get('rmse', float('nan'))
            rmse2 = results[method2]['final_metrics']['test']['by_position'].get(pos_key, {}).get('rmse', float('nan'))
            improvement = rmse1 - rmse2  # Positive means method2 is better
            rmse_improvements.append(improvement)
        
        colors_imp = ['green' if x > 0 else 'red' for x in rmse_improvements]
        ax4.bar(positions, rmse_improvements, color=colors_imp, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Position')
        ax4.set_ylabel(f'RMSE Improvement\n({method2.replace("_", " ").title()} vs {method1.replace("_", " ").title()})')
        ax4.set_title('RMSE Improvement by Position\n(Positive = Better)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(positions)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_question_plots(results, save_path):
    """Create plots for question-specific metrics."""
    methods = list(results.keys())
    questions = [f'Q{i}' for i in range(7)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'random_masking': 'red', 'dynamic_masking': 'green'}
    
    # Question-wise RMSE comparison
    x_pos = np.arange(len(questions))
    width = 0.35
    
    test_rmse_by_q = {}
    for method in methods:
        rmse_values = []
        for q in questions:
            rmse = results[method]['final_metrics']['test']['by_question'].get(q, {}).get('rmse', float('nan'))
            rmse_values.append(rmse)
        test_rmse_by_q[method] = rmse_values
    
    for i, method in enumerate(methods):
        ax1.bar(x_pos + i*width - width/2, test_rmse_by_q[method], width, 
               color=colors[method], alpha=0.7, label=method.replace('_', ' ').title())
    
    ax1.set_xlabel('Question')
    ax1.set_ylabel('Test RMSE')
    ax1.set_title('Test RMSE by Question')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(questions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Question-wise correlation comparison
    test_pearson_by_q = {}
    for method in methods:
        pearson_values = []
        for q in questions:
            pearson = results[method]['final_metrics']['test']['by_question'].get(q, {}).get('pearson', float('nan'))
            pearson_values.append(pearson)
        test_pearson_by_q[method] = pearson_values
    
    for i, method in enumerate(methods):
        ax2.bar(x_pos + i*width - width/2, test_pearson_by_q[method], width, 
               color=colors[method], alpha=0.7, label=method.replace('_', ' ').title())
    
    ax2.set_xlabel('Question')
    ax2.set_ylabel('Test Pearson Correlation')
    ax2.set_title('Test Correlation by Question')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(questions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Question counts
    for i, method in enumerate(methods):
        count_values = []
        for q in questions:
            count = results[method]['final_metrics']['test']['by_question'].get(q, {}).get('count', 0)
            count_values.append(count)
        
        ax3.bar(x_pos + i*width - width/2, count_values, width, 
               color=colors[method], alpha=0.7, label=method.replace('_', ' ').title())
    
    ax3.set_xlabel('Question')
    ax3.set_ylabel('Number of Examples')
    ax3.set_title('Test Examples per Question')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(questions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # RMSE improvement by question
    if len(methods) == 2:
        method1, method2 = methods
        rmse_improvements = []
        for q in questions:
            rmse1 = results[method1]['final_metrics']['test']['by_question'].get(q, {}).get('rmse', float('nan'))
            rmse2 = results[method2]['final_metrics']['test']['by_question'].get(q, {}).get('rmse', float('nan'))
            improvement = rmse1 - rmse2
            rmse_improvements.append(improvement)
        
        colors_imp = ['green' if x > 0 else 'red' for x in rmse_improvements]
        ax4.bar(x_pos, rmse_improvements, color=colors_imp, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Question')
        ax4.set_ylabel(f'RMSE Improvement\n({method2.replace("_", " ").title()} vs {method1.replace("_", " ").title()})')
        ax4.set_title('RMSE Improvement by Question\n(Positive = Better)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(questions)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def run_comprehensive_comparison(
    results_path,
    dataset_train,
    dataset_val, 
    dataset_test,
    dataset_name,
    epochs=15,
    batch_size=8,
    lr=1e-4,
    device=None,
    use_embedding=True,
    num_patterns_per_example=5,
    visible_ratio=0.5
):
    """Run comprehensive training method comparison."""
    
    print(f"Loading experiment results from {results_path}")
    results = load_experiment_results(results_path)
    
    observations = extract_observation_history(results)
    print(f"Found {len(observations)} observations in experiment history")
    
    if not observations:
        print("No observations found in results file!")
        return None
    
    training_methods = ['random_masking', 'dynamic_masking']
    all_results = {}
    
    for method in training_methods:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        # Create fresh model with same seed for fair comparison
        model = create_fresh_model(dataset_name, use_embedding, device, seed=43)
        
        # Apply observations and get training data ready
        arena, applied_count = apply_observations_to_model(model, dataset_train, observations, device)
        
        # Evaluate before training
        print("Evaluating before training...")
        val_metrics_before = evaluate_model_directly(model, dataset_val, device)
        test_metrics_before = evaluate_model_directly(model, dataset_test, device)
        
        print(f"Before training:")
        print(f"  Validation RMSE: {val_metrics_before['overall']['rmse']:.4f}")
        print(f"  Test RMSE: {test_metrics_before['overall']['rmse']:.4f}")
        print(f"  Test predictions: {test_metrics_before['overall']['total_predictions']}")
        
        # Train and track progress
        print(f"Training with {method} for {epochs} epochs...")
        
        val_rmse_progress = [val_metrics_before['overall']['rmse']]
        test_rmse_progress = [test_metrics_before['overall']['rmse']]
        training_losses = []
        
        for epoch in tqdm(range(epochs), desc=f"Training {method}"):
            # Train one epoch
            losses = train_model_with_method(
                arena, method, epochs=1, batch_size=batch_size, lr=lr,
                num_patterns_per_example=num_patterns_per_example,
                visible_ratio=visible_ratio
            )
            training_losses.extend(losses)
            
            # Evaluate after this epoch
            val_metrics_current = evaluate_model_directly(model, dataset_val, device)
            test_metrics_current = evaluate_model_directly(model, dataset_test, device)
            
            val_rmse_progress.append(val_metrics_current['overall']['rmse'])
            test_rmse_progress.append(test_metrics_current['overall']['rmse'])
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}: Val RMSE={val_metrics_current['overall']['rmse']:.4f}, "
                      f"Test RMSE={test_metrics_current['overall']['rmse']:.4f}")
        
        # Final comprehensive evaluation
        print("Final evaluation...")
        val_metrics_final = evaluate_model_directly(model, dataset_val, device)
        test_metrics_final = evaluate_model_directly(model, dataset_test, device)
        
        print(f"Final results:")
        print(f"  Validation RMSE: {val_metrics_final['overall']['rmse']:.4f}")
        print(f"  Test RMSE: {test_metrics_final['overall']['rmse']:.4f}")
        print(f"  Validation Pearson: {val_metrics_final['overall']['pearson']:.4f}")
        print(f"  Test Pearson: {test_metrics_final['overall']['pearson']:.4f}")
        
        # Store comprehensive results
        all_results[method] = {
            'initial_metrics': {
                'val': val_metrics_before,
                'test': test_metrics_before
            },
            'final_metrics': {
                'val': val_metrics_final,
                'test': test_metrics_final
            },
            'val_rmse_progress': val_rmse_progress,
            'test_rmse_progress': test_rmse_progress,
            'training_losses': training_losses,
            'applied_observations': applied_count,
            'improvements': {
                'val_rmse': val_metrics_before['overall']['rmse'] - val_metrics_final['overall']['rmse'],
                'test_rmse': test_metrics_before['overall']['rmse'] - test_metrics_final['overall']['rmse'],
                'val_pearson': val_metrics_final['overall']['pearson'] - val_metrics_before['overall']['pearson'],
                'test_pearson': test_metrics_final['overall']['pearson'] - test_metrics_before['overall']['pearson']
            }
        }
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive direct model comparison.")
    parser.add_argument("--results_path", type=str, 
                       default="/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_enhanced_hanna/DynamicMasking_NewSelectionLoss/TEST1/enhanced_gradient_voi_q0_human_with_embedding.json",
                       help="Path to JSON file with experiment results")
    parser.add_argument("--dataset", type=str, default="hanna", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, 
                       default="/export/fs06/psingh54/ActiveRubric-Internal/outputs/results_enhanced_hanna/PostTrainingAnalysis",
                       help="Directory to save results")
    parser.add_argument("--runner", type=str, default="prabhav", help="Runner identifier")
    parser.add_argument("--use_embedding", action="store_true", default=True, help="Use embedding model")
    parser.add_argument("--num_patterns_per_example", type=int, default=2, help="Dynamic masking patterns")
    parser.add_argument("--visible_ratio", type=float, default=0.7, help="Dynamic masking visible ratio")
    
    args = parser.parse_args()
    
    # Setup
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
                                dataset=args.dataset, cold_start=False, 
                                use_embedding=args.use_embedding)
    
    train_dataset = AnnotationDataset(data_manager.paths['train'])
    val_dataset = AnnotationDataset(data_manager.paths['validation'])
    test_dataset = AnnotationDataset(data_manager.paths['test'])
    active_pool_dataset = AnnotationDataset(data_manager.paths['active_pool'])
    
    if len(train_dataset) == 0:
        train_dataset = active_pool_dataset
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison(
        results_path=args.results_path,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_embedding=args.use_embedding,
        num_patterns_per_example=args.num_patterns_per_example,
        visible_ratio=args.visible_ratio
    )
    
    if results:
        # Save comprehensive results
        output_filename = f"comprehensive_direct_comparison_{os.path.basename(args.results_path).replace('.json', '')}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        final_results = {
            'experiment_info': {
                'results_path': args.results_path,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'dataset': args.dataset,
                'use_embedding': args.use_embedding,
                'evaluation_method': 'direct_model_evaluation_all_14_positions'
            },
            'method_comparisons': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        # Create and save plots
        base_name = os.path.basename(args.results_path).replace('.json', '')
        
        # Overall metrics plots
        overall_plot_path = os.path.join(args.output_dir, f"overall_comparison_{base_name}.png")
        fig1 = create_overall_plots(results, overall_plot_path)
        
        # Position-specific plots
        position_plot_path = os.path.join(args.output_dir, f"position_comparison_{base_name}.png")
        fig2 = create_position_plots(results, position_plot_path)
        
        # Question-specific plots  
        question_plot_path = os.path.join(args.output_dir, f"question_comparison_{base_name}.png")
        fig3 = create_question_plots(results, question_plot_path)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Overall plots: {overall_plot_path}")
        print(f"Position plots: {position_plot_path}")
        print(f"Question plots: {question_plot_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        methods = list(results.keys())
        print(f"{'Method':<15} {'Test RMSE':<12} {'Test Pearson':<12} {'Val RMSE':<12} {'Val Pearson':<12}")
        print(f"{'-'*80}")
        
        for method in methods:
            test_rmse = results[method]['final_metrics']['test']['overall']['rmse']
            test_pearson = results[method]['final_metrics']['test']['overall']['pearson']
            val_rmse = results[method]['final_metrics']['val']['overall']['rmse']
            val_pearson = results[method]['final_metrics']['val']['overall']['pearson']
            
            print(f"{method:<15} {test_rmse:<12.4f} {test_pearson:<12.4f} {val_rmse:<12.4f} {val_pearson:<12.4f}")
        
        print(f"\n{'='*80}")
        print(f"IMPROVEMENTS")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'RMSE Δ':<12} {'Pearson Δ':<12} {'Obs Applied':<12}")
        print(f"{'-'*60}")
        
        for method in methods:
            rmse_imp = results[method]['improvements']['test_rmse']
            pearson_imp = results[method]['improvements']['test_pearson']
            obs_count = results[method]['applied_observations']
            
            print(f"{method:<15} {rmse_imp:<12.4f} {pearson_imp:<12.4f} {obs_count:<12}")
        
        plt.show()


if __name__ == "__main__":
    main()