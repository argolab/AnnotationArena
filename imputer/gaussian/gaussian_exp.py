"""
Training script for Gaussian data using AnnotationArena and ImputerEmbedding.

This script:
1. Loads the Gaussian training and dev data
2. Creates a custom dataset wrapper
3. Sets up AnnotationArena with ImputerEmbedding model
4. Trains on training data using dynamic masking
5. Evaluates on dev data with KL divergence
6. Plots histogram of KL divergence values across dev examples
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
import json
import numpy as np
import logging
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any
import time
import matplotlib.pyplot as plt
import seaborn as sns
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import random
random.seed(91)
torch.manual_seed(91)
np.random.seed(91)
class GaussianDataset(Dataset):
    """Custom dataset wrapper for Gaussian data with marginal distributions."""
    
    def __init__(self, data_path, is_training=False, data_num=None):
        """
        Initialize dataset from JSON file.
        
        Args:
            data_path: Path to JSON data file
            is_training: Whether this is training data (has marginal_distributions)
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)["data"]
        if data_num is not None:
            if is_training:
                self.data = self.data[:data_num]
            else:
                self.data = self.data[-data_num:]
        self.is_training = is_training
        logger.info(f"Loaded dataset with {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get example data in the format expected by ImputerEmbedding.
        
        Returns:
            tuple: (known_questions, inputs, answers, annotators, questions)
        """
        entry = self.data[idx]
        
        # Convert to tensors
        known_questions = torch.tensor(entry['known_questions'], dtype=torch.float32)
        inputs = torch.tensor(entry['input'], dtype=torch.float32)
        answers = torch.tensor(entry['answers'], dtype=torch.float32)
        annotators = torch.tensor(entry['annotators'], dtype=torch.long)
        questions = torch.tensor(entry['questions'], dtype=torch.long)
        
        return known_questions, inputs, answers, annotators, questions
    
    def get_data_entry(self, idx):
        """Get raw data entry for arena interface."""
        entry = self.data[idx]
        
        return {
            'known_questions': entry['known_questions'],
            'input': entry['input'],
            'answers': entry['answers'],
            'true_answers': entry['answers'],  # For evaluation
            'annotators': entry['annotators'],
            'questions': entry['questions'],
            'marginal_distributions': entry.get('marginal_distributions', {})
        }
    
    def get_masked_positions(self, idx):
        """Get positions that are masked (input[i][0] == 1)."""
        entry = self.data[idx]
        masked_positions = []
        for i, input_vec in enumerate(entry['input']):
            if input_vec[0] == 1:  # Masked position
                masked_positions.append(i)
        return masked_positions
    
    def observe_position(self, idx, position):
        """Mark a position as observed (for interface compatibility)."""
        # In this simulation, we don't actually modify the data
        pass

def compute_kl_divergence_loss(predicted_probs, target_probs, epsilon=1e-10):
    """
    Compute KL divergence between predicted and target probability distributions.
    
    Args:
        predicted_probs: Predicted probability distributions [batch_size, seq_len, num_classes]
        target_probs: Target probability distributions [batch_size, seq_len, num_classes]
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence loss
    """
    # Add epsilon to avoid log(0)
    predicted_probs = predicted_probs + epsilon
    target_probs = target_probs + epsilon
    
    # Normalize to ensure they sum to 1
    predicted_probs = predicted_probs / predicted_probs.sum(dim=-1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence: KL(target || predicted)
    kl_loss = F.kl_div(
        torch.log(predicted_probs).to("cpu"), 
        target_probs.to("cpu"), 
        reduction='batchmean'
    )
    
    return kl_loss

def evaluate_with_kl_divergence(model, dataset, device, num_examples=None):
    """
    Evaluate model on dataset using KL divergence.
    
    Args:
        model: Trained ImputerEmbedding model
        dataset: GaussianDataset to evaluate on
        device: Device to use
        num_examples: Number of examples to evaluate (None for all)
        
    Returns:
        dict: Evaluation metrics including per-example KL divergences
    """
    model.eval()
    total_kl_loss = 0.0
    total_positions = 0
    example_losses = []
    example_kl_divergences = []  # Store KL divergence for each example
    
    if num_examples is None:
        num_examples = len(dataset)
    
    with torch.no_grad():
        for idx in range(min(num_examples, len(dataset))):
            known_questions, inputs, answers, annotators, questions = dataset[idx]
            # Move to device
            inputs = inputs.unsqueeze(0).to(device)
            annotators = annotators.unsqueeze(0).to(device)
            questions = questions.unsqueeze(0).to(device)
            answers = answers.unsqueeze(0).to(device)
            
            # Get model predictions
            predictions, _ = model(inputs, annotators, questions)
            
            # Convert to probabilities
            predicted_probs = F.softmax(predictions, dim=-1)
            
            # Compute KL divergence for each position
            example_kl_loss = 0.0
            example_positions = 0
            example_kls = []
            for pos in range(inputs.shape[1]):
                # Only evaluate on positions that have probabilistic targets (not one-hot)
                target_dist = answers[0, pos]
                if not torch.allclose(target_dist.sum(), torch.tensor(1.0), atol=1e-6):
                    continue  # Skip positions without valid probability distributions
                
                # Check if this is a probabilistic target (not one-hot)
                if (target_dist > 0).sum() > 1:  # More than one non-zero entry
                    pos_kl_loss = compute_kl_divergence_loss(
                        predicted_probs[0:1, pos:pos+1, :],
                        target_dist.unsqueeze(0).unsqueeze(0)
                    )
                    example_kl_loss += pos_kl_loss.item()
                    example_positions += 1
                    example_kls.append(pos_kl_loss.item())
            
            if example_positions > 0:
                example_avg_loss = example_kl_loss / example_positions
                example_losses.append(example_avg_loss)
                example_kl_divergences.extend(example_kls)  # Store for histogram
                total_kl_loss += example_kl_loss
                total_positions += example_positions
    
    # Compute overall metrics
    avg_kl_loss = total_kl_loss / max(1, total_positions)
    std_kl_loss = np.std(example_losses) if example_losses else 0.0
    
    return {
        'avg_kl_divergence': avg_kl_loss,
        'std_kl_divergence': std_kl_loss,
        'total_positions_evaluated': total_positions,
        'examples_evaluated': len(example_losses),
        'example_kl_divergences': example_kl_divergences  # Add this for plotting
    }

def plot_kl_divergence_histogram(kl_divergences, save_path="kl_divergence_histogram.png"):
    """
    Plot histogram of KL divergence values across dev examples.
    
    Args:
        kl_divergences: List of KL divergence values for each example
        save_path: Path to save the histogram plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n_bins = min(30, len(kl_divergences) // 2)  # Adaptive number of bins
    plt.hist(kl_divergences, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics as text
    mean_kl = np.mean(kl_divergences)
    std_kl = np.std(kl_divergences)
    median_kl = np.median(kl_divergences)
    
    plt.axvline(mean_kl, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_kl:.4f}')
    plt.axvline(median_kl, color='green', linestyle='--', linewidth=2, label=f'Median: {median_kl:.4f}')
    
    plt.xlabel('KL Divergence')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of KL Divergence Across Dev Examples')
    plt.legend()
    plt.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"KL divergence histogram saved to {save_path}")
    
    # Print additional statistics
    logger.info(f"KL Divergence Statistics:")
    logger.info(f"  Mean: {mean_kl:.4f}")
    logger.info(f"  Std: {std_kl:.4f}")
    logger.info(f"  Median: {median_kl:.4f}")
    logger.info(f"  Min: {min(kl_divergences):.4f}")
    logger.info(f"  Max: {max(kl_divergences):.4f}")
    logger.info(f"  25th percentile: {np.percentile(kl_divergences, 25):.4f}")
    logger.info(f"  75th percentile: {np.percentile(kl_divergences, 75):.4f}")

def main():
    """Main training and evaluation loop."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_sizes = [100, 200, 500, 800, 1000, 1200, 1600, 2000, 2400]
    dict = {}

    for train_size in train_sizes:
    
        # Load datasets
        logger.info(f"Loading datasets with {train_size} training data")
        train_dataset = GaussianDataset("gaussian_train_10_new.json", is_training=True, data_num = train_size)
        dev_dataset = GaussianDataset("gaussian_dev_10_new.json", is_training=False, data_num = 100)
        
        # Initialize model
        logger.info("Initializing model...")
        from imputer_gaussian import ImputerEmbedding  # Import your model class
        
        model = ImputerEmbedding(
            question_num=10,  # 5 variables (questions 0-4)
            max_choices=5,   # 5 categories per variable
            encoder_layers_num=4,
            attention_heads=5,
            hidden_dim=64,
            num_annotator=1,  # Only one "annotator" (the data generation process)
            annotator_embedding_dim=30,
            dropout=0.1,
        ).to(device)
        
        # Create AnnotationArena
        logger.info("Creating AnnotationArena...")
        from annotationArena import AnnotationArena  # Import your arena class
        
        arena = AnnotationArena(model, device=device)
        arena.set_dataset(train_dataset)
        
        # Set dynamic masking parameters
        arena.set_dynamic_masking_params(
            num_patterns_per_example=3,  # Use fewer patterns for efficiency
            visible_ratio=0.6,           # Keep 60% of observed variables visible
            masking_lambda=0.1
        )
        
        # Register training examples in the arena
        logger.info("Registering training examples...")
        for example_idx in range(len(train_dataset)):
            variable_ids = arena.register_example(example_idx, add_all_positions=False)
            
            # Make predictions on masked positions to populate training queue
            for variable_id in variable_ids:
                arena.predict(variable_id, train=True, weight=1.0)
            
            if (example_idx + 1) % 100 == 0:
                logger.info(f"Registered {example_idx + 1} examples")
        
        logger.info(f"Total variables registered: {arena.get_variable_count()}")
        logger.info(f"Training queue size: {len(model.training_queue)}")
        
        # Training loop
        logger.info("Starting training...")
        training_epochs = 20
        batch_size = 16
        learning_rate = 1e-4
        
        training_metrics = arena.train(
            training_type='dynamic_masking',
            epochs=training_epochs,
            batch_size=batch_size,
            lr=learning_rate
        )
        
        logger.info(f"Training completed. Average loss: {training_metrics['avg_loss']:.4f}")
        
        # Evaluation on dev set
        logger.info("Evaluating on dev set...")
        arena.set_dataset(dev_dataset)  # Switch to dev dataset
        
        # Register dev examples
        dev_variable_ids = []
        for example_idx in range(len(dev_dataset)):
            variable_ids = arena.register_example(example_idx, add_all_positions=True)
            dev_variable_ids.extend(variable_ids)
        
        # Evaluate with KL divergence
        eval_metrics = evaluate_with_kl_divergence(model, dev_dataset, device)
        
        logger.info("Evaluation Results:")
        logger.info(f"  Average KL Divergence: {eval_metrics['avg_kl_divergence']:.4f}")
        logger.info(f"  Std KL Divergence: {eval_metrics['std_kl_divergence']:.4f}")
        logger.info(f"  Positions Evaluated: {eval_metrics['total_positions_evaluated']}")
        logger.info(f"  Examples Evaluated: {eval_metrics['examples_evaluated']}")
        
        # Plot KL divergence histogram
        logger.info("Plotting KL divergence histogram...")
        plot_kl_divergence_histogram(
            eval_metrics['example_kl_divergences'], 
            save_path=f"kl_divergence_histogram_{train_size}_num_train.png"
        )
        dict[train_size] = eval_metrics['avg_kl_divergence']
    print(dict)
    

if __name__ == "__main__":
    main()