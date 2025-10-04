"""
Integration script to add lens analysis to your model evaluation pipeline.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from lens_analysis import LogitLens, TunedLens
import logging

logger = logging.getLogger(__name__)


def analyze_trained_model_layers(model, train_dataset, eval_dataset, 
                                 save_dir='lens_results', num_examples=5):
    """
    Analyze a trained model using lens techniques on multiple examples.
    Analyzes ALL masked positions for each example.
    
    Args:
        model: Trained ImputerEmbedding model
        train_dataset: Training dataset (for tuned lens training)
        eval_dataset: Evaluation dataset
        save_dir: Directory to save results
        num_examples: Number of examples to analyze
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Analyzing model with lens techniques on {num_examples} examples")
    
    # Initialize lenses
    logit_lens = LogitLens(model)
    tuned_lens = TunedLens(model, model.max_choices)
    tuned_lens = tuned_lens.to(model.device)
    
    # Train tuned lens once on training data
    logger.info("Training Tuned Lens...")
    tuned_lens.train_lens(train_dataset, epochs=60, lr=1e-3, batch_size=32)
    
    # Analyze multiple examples
    all_logit_results = []
    all_tuned_results = []
    
    for example_idx in range(min(num_examples, len(eval_dataset))):
        logger.info(f"Analyzing example {example_idx + 1}/{num_examples}")
        
        # Get example
        known_q, inputs, answers, annotators, questions = eval_dataset[example_idx]
        inputs_tensor = inputs.unsqueeze(0).to(model.device)
        annotators_tensor = annotators.unsqueeze(0).to(model.device)
        questions_tensor = questions.unsqueeze(0).to(model.device)
        
        # Find ALL masked positions to analyze
        masked_positions = [i for i, inp in enumerate(inputs) if inp[0] == 1]
        if not masked_positions:
            logger.info(f"No masked positions in example {example_idx}, skipping")
            continue
        
        logger.info(f"  Found {len(masked_positions)} masked positions: {masked_positions}")
        
        # Analyze each masked position
        for target_position in masked_positions:
            true_label = torch.argmax(answers[target_position]).item()
            true_probs = answers[target_position].to(model.device)
            
            # Run lens analysis with true probabilities
            logit_results = logit_lens.analyze_layers(
                inputs_tensor, annotators_tensor, questions_tensor, target_position, true_probs
            )
            tuned_results = tuned_lens.analyze_layers(
                inputs_tensor, annotators_tensor, questions_tensor, target_position, true_probs
            )
            
            all_logit_results.append(logit_results)
            all_tuned_results.append(tuned_results)
    
    # Create aggregate analysis
    if all_logit_results:
        logger.info(f"Creating aggregate plot from {len(all_logit_results)} total masked positions")
        plot_aggregate_lens_analysis(all_logit_results, all_tuned_results, 
                                     f"{save_dir}/lens_analysis_aggregate.png")
    
    return all_logit_results, all_tuned_results


def plot_aggregate_lens_analysis(all_logit_results, all_tuned_results, save_path):
    """
    Plot aggregate statistics across multiple examples.
    """
    num_layers = all_logit_results[0]['num_layers']
    
    # Aggregate entropy and confidence
    logit_entropies_per_layer = [[] for _ in range(num_layers)]
    tuned_entropies_per_layer = [[] for _ in range(num_layers)]
    logit_confidences_per_layer = [[] for _ in range(num_layers)]
    tuned_confidences_per_layer = [[] for _ in range(num_layers)]
    logit_kl_per_layer = [[] for _ in range(num_layers)]
    tuned_kl_per_layer = [[] for _ in range(num_layers)]
    
    for logit_res, tuned_res in zip(all_logit_results, all_tuned_results):
        for i in range(num_layers):
            logit_entropies_per_layer[i].append(logit_res['entropies'][i].mean().item())
            tuned_entropies_per_layer[i].append(tuned_res['entropies'][i].mean().item())
            logit_confidences_per_layer[i].append(logit_res['confidences'][i].mean().item())
            tuned_confidences_per_layer[i].append(tuned_res['confidences'][i].mean().item())
            
            # Add KL divergence if available
            if 'kl_divergences' in logit_res:
                logit_kl_per_layer[i].append(logit_res['kl_divergences'][i].item())
                tuned_kl_per_layer[i].append(tuned_res['kl_divergences'][i].item())
    
    # Compute statistics
    logit_entropy_mean = [np.mean(vals) for vals in logit_entropies_per_layer]
    logit_entropy_std = [np.std(vals) for vals in logit_entropies_per_layer]
    tuned_entropy_mean = [np.mean(vals) for vals in tuned_entropies_per_layer]
    tuned_entropy_std = [np.std(vals) for vals in tuned_entropies_per_layer]
    
    logit_conf_mean = [np.mean(vals) for vals in logit_confidences_per_layer]
    logit_conf_std = [np.std(vals) for vals in logit_confidences_per_layer]
    tuned_conf_mean = [np.mean(vals) for vals in tuned_confidences_per_layer]
    tuned_conf_std = [np.std(vals) for vals in tuned_confidences_per_layer]
    
    # KL divergence statistics
    has_kl = len(logit_kl_per_layer[0]) > 0
    if has_kl:
        logit_kl_mean = [np.mean(vals) for vals in logit_kl_per_layer]
        logit_kl_std = [np.std(vals) for vals in logit_kl_per_layer]
        tuned_kl_mean = [np.mean(vals) for vals in tuned_kl_per_layer]
        tuned_kl_std = [np.std(vals) for vals in tuned_kl_per_layer]

    print(logit_kl_mean)
    
    layers = list(range(num_layers))
    
    # Create plot - 2x3 if KL available, 2x2 otherwise
    if has_kl:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Entropy with error bars
    axes[0, 0].errorbar(layers, logit_entropy_mean, yerr=logit_entropy_std, 
                        fmt='o-', label='Logit Lens', capsize=5, linewidth=2)
    axes[0, 0].errorbar(layers, tuned_entropy_mean, yerr=tuned_entropy_std,
                        fmt='s-', label='Tuned Lens', capsize=5, linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Average Entropy Across Examples')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence with error bars
    axes[0, 1].errorbar(layers, logit_conf_mean, yerr=logit_conf_std,
                        fmt='o-', label='Logit Lens', capsize=5, linewidth=2)
    axes[0, 1].errorbar(layers, tuned_conf_mean, yerr=tuned_conf_std,
                        fmt='s-', label='Tuned Lens', capsize=5, linewidth=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Confidence')
    axes[0, 1].set_title('Average Confidence Across Examples')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence (if available)
    if has_kl:
        axes[0, 2].errorbar(layers, logit_kl_mean, yerr=logit_kl_std,
                            fmt='o-', label='Logit Lens', capsize=5, linewidth=2)
        axes[0, 2].errorbar(layers, tuned_kl_mean, yerr=tuned_kl_std,
                            fmt='s-', label='Tuned Lens', capsize=5, linewidth=2)
        axes[0, 2].set_xlabel('Layer')
        axes[0, 2].set_ylabel('KL Divergence from True')
        axes[0, 2].set_title('KL Divergence vs True Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Entropy reduction rate
    logit_entropy_reduction = [logit_entropy_mean[i] - logit_entropy_mean[i+1] 
                               if i < len(logit_entropy_mean)-1 else 0
                               for i in range(num_layers)]
    tuned_entropy_reduction = [tuned_entropy_mean[i] - tuned_entropy_mean[i+1]
                               if i < len(tuned_entropy_mean)-1 else 0
                               for i in range(num_layers)]
    
    axes[1, 0].bar(np.array(layers) - 0.2, logit_entropy_reduction, 0.4, 
                   label='Logit Lens', alpha=0.7)
    axes[1, 0].bar(np.array(layers) + 0.2, tuned_entropy_reduction, 0.4,
                   label='Tuned Lens', alpha=0.7)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Entropy Reduction')
    axes[1, 0].set_title('Per-Layer Entropy Reduction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Confidence gain rate
    logit_conf_gain = [logit_conf_mean[i+1] - logit_conf_mean[i]
                       if i < len(logit_conf_mean)-1 else 0
                       for i in range(num_layers)]
    tuned_conf_gain = [tuned_conf_mean[i+1] - tuned_conf_mean[i]
                       if i < len(tuned_conf_mean)-1 else 0
                       for i in range(num_layers)]
    
    axes[1, 1].bar(np.array(layers) - 0.2, logit_conf_gain, 0.4,
                   label='Logit Lens', alpha=0.7)
    axes[1, 1].bar(np.array(layers) + 0.2, tuned_conf_gain, 0.4,
                   label='Tuned Lens', alpha=0.7)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Confidence Gain')
    axes[1, 1].set_title('Per-Layer Confidence Gain')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # KL reduction rate (if available)
    if has_kl:
        logit_kl_reduction = [logit_kl_mean[i] - logit_kl_mean[i+1]
                             if i < len(logit_kl_mean)-1 else 0
                             for i in range(num_layers)]
        tuned_kl_reduction = [tuned_kl_mean[i] - tuned_kl_mean[i+1]
                             if i < len(tuned_kl_mean)-1 else 0
                             for i in range(num_layers)]
        
        axes[1, 2].bar(np.array(layers) - 0.2, logit_kl_reduction, 0.4,
                       label='Logit Lens', alpha=0.7)
        axes[1, 2].bar(np.array(layers) + 0.2, tuned_kl_reduction, 0.4,
                       label='Tuned Lens', alpha=0.7)
        axes[1, 2].set_xlabel('Layer')
        axes[1, 2].set_ylabel('KL Divergence Reduction')
        axes[1, 2].set_title('Per-Layer KL Divergence Reduction')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Aggregate lens analysis saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example standalone usage
    import argparse
    from imputer_gaussian import ImputerEmbedding
    from runner import GaussianDataset
    
    parser = argparse.ArgumentParser(description='Run lens analysis on trained model')
    parser.add_argument('--model_path', required=True, help='Path to trained model .pth file')
    parser.add_argument('--train_file', required=True, help='Training data JSON file')
    parser.add_argument('--eval_file', required=True, help='Evaluation data JSON file')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of examples to analyze')
    parser.add_argument('--save_dir', default='lens_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load trained model
    logger.info(f"Loading model from {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, weights_only=False)
    
    model_config = checkpoint['model_config']
    model = ImputerEmbedding(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Load datasets
    train_dataset = GaussianDataset(args.train_file, is_training=True)
    eval_dataset = GaussianDataset(args.eval_file, is_training=False)
    
    logger.info(f"Training dataset: {len(train_dataset)} examples")
    logger.info(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Run lens analysis
    logit_results, tuned_results = analyze_trained_model_layers(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        save_dir=args.save_dir,
        num_examples=args.num_examples
    )
    
    logger.info(f"Lens analysis complete. Results saved to {args.save_dir}")