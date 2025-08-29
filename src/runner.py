"""
Comprehensive evaluation script comparing three models:
1. Ground Truth Model (using true parameters from data generation)
2. Domain-Specific Model (MAP estimation on training data)
3. Neural Model (ImputerEmbedding trained on training data)

This script ensures fair comparison by:
- Using the same training and evaluation data
- Computing KL divergence consistently across all models
- Training both domain-specific and neural models on the specified training size
- Recording training times for each model type
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import json
import numpy as np
import logging
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional
import time
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from multi_gaussian_new import GaussianBinningWithLinGauss
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def record_training_time(train_file: str, model_type: str, training_size: int, 
                        training_time: float, timing_file: str = "training_times.json"):
    """
    Record training time to a JSON file.
    
    Args:
        train_file: Name of the training file
        model_type: Type of model ('MAP', 'tiny', 'small', 'large')
        training_size: Number of training examples used
        training_time: Training time in seconds
        timing_file: Path to the timing JSON file
    """
    # Create the key
    key = f"{train_file}_{model_type}_{training_size}"
    
    # Load existing data or create empty dict
    try:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        timing_data = {}
    
    # Add new timing data
    timing_data[key] = training_time
    
    # Save back to file
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2, sort_keys=True)
    
    logger.info(f"Recorded training time for {key}: {training_time:.2f} seconds")

class GaussianDataset(Dataset):
    """Custom dataset wrapper for Gaussian data with marginal distributions."""
    
    def __init__(self, data_path, is_training=False, data_num=None):
        """Initialize dataset from JSON file."""
        with open(data_path, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                self.data = content['data']
                self.metadata = content.get('metadata', {})
            else:
                self.data = content
                self.metadata = {}
        
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
        """Get example data in the format expected by ImputerEmbedding."""
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
            'true_answers': entry['answers'],
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


class DomainSpecificEvaluator:
    """Handles evaluation using the domain-specific Gaussian model."""
    
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        
        # Import the domain-specific model classes
          # Assuming the first file is saved as paste.py
        self.GaussianBinningWithLinGauss = GaussianBinningWithLinGauss
    
    def setup_model_from_metadata(self, metadata: dict):
        """Setup domain model using metadata from training data."""
        # Set up model parameters (should match data generation)
        alpha = 2.0
        nu = self.n + 2
        V = np.eye(self.n)
        penalty_strength = 100.0
        
        sparsity_pattern = np.zeros((self.n, self.n))
        for i in range(self.n):
            sparsity_pattern[i, i] = 1
            if i > 0:
                sparsity_pattern[i, i-1] = 1
                sparsity_pattern[i-1, i] = 1
        if "sparsity_pattern" in metadata:
            sparsity_pattern = np.array(metadata['sparsity_pattern'])
            print(f"Loaded sparsity pattern: {np.sum(sparsity_pattern == 0)} zero elements")
        penalty_strength = metadata.get('penalty_strength', penalty_strength)
        
        # Initialize domain model
        self.model = GaussianBinningWithLinGauss(
            self.n, self.k, sparsity_pattern, penalty_strength, alpha, nu, V
        )
        
        return metadata
    
    def extract_training_patterns(self, train_data: List[Dict]) -> Tuple[List, List]:
        """Extract training patterns from training data."""
        train_x_obs_batch = []
        train_obs_idx_batch = []
        
        for entry in train_data:
            known_questions = entry['known_questions']
            input_vecs = entry['input']
            
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i + 1)  # Convert to 1-based
                    value = np.argmax(input_vec[1:]) + 1
                    obs_values.append(value)
            
            if len(obs_indices) > 0:
                train_x_obs_batch.append(np.array(obs_values))
                train_obs_idx_batch.append(np.array(obs_indices))
        
        return train_x_obs_batch, train_obs_idx_batch
    
    def extract_dev_patterns(self, dev_data: List[Dict]) -> List[Tuple]:
        """Extract dev patterns for evaluation."""
        dev_patterns = []
        
        for entry in dev_data:
            known_questions = entry['known_questions']
            input_vecs = entry['input']
            
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(known_questions, input_vecs)):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i + 1)  # Convert to 1-based
                    value = np.argmax(input_vec[1:]) + 1
                    obs_values.append(value)
            
            dev_patterns.append((np.array(obs_indices), np.array(obs_values)))
        
        return dev_patterns
    
    def extract_true_values(self, dev_data: List[Dict]) -> np.ndarray:
        """Extract true values from dev data entries."""
        true_values = []
        
        for entry in dev_data:
            true_x = []
            for answer_vec in entry['answers']:
                if max(answer_vec) == 1.0:  # One-hot encoded
                    true_x.append(np.argmax(answer_vec) + 1)
                else:  # Probabilistic - use mode
                    true_x.append(np.argmax(answer_vec) + 1)
            true_values.append(true_x)
        
        return np.array(true_values)
    
    def evaluate_ground_truth_model(self, metadata: dict, dev_patterns: List, 
                                  dev_data: List[Dict]) -> Dict:
        """Evaluate using ground truth parameters."""
        boundaries = np.array(metadata['boundaries'])
        gt_omega = np.array(metadata['true_parameters']['Omega'])
        gt_sigma = np.array(metadata['true_parameters']['Sigma'])
        
        gt_params = {'Omega': gt_omega}
        
        eval_data = {
            'boundaries': boundaries,
            'x': self.extract_true_values(dev_data),
            'true_parameters': {
                'Omega': gt_omega,
                'Sigma': gt_sigma
            }
        }
        
        return self.model.evaluate_with_kl_divergence(
            eval_data, dev_patterns, 'known_params', gt_params, dev_data
        )
    
    def train_and_evaluate_map_model(self, train_x_obs_batch: List, train_obs_idx_batch: List,
                               metadata: dict, dev_patterns: List, dev_data: List[Dict],
                               pytorch_epochs: int = 500, save_path: str = None,
                               train_file: str = "", training_size: int = 0,
                               init_with_metadata: bool = False, num_restart=5) -> Tuple[Dict, Dict]:
        """Train MAP model and evaluate with timing."""
        
        # Check if save path exists and skip training if so
        if save_path and os.path.exists(save_path):
            logger.info(f"Model already exists at {save_path}, loading instead of training...")
            with open(save_path, 'r') as f:
                map_params = json.load(f)
            # Convert back to numpy arrays
            map_params['Omega'] = np.array(map_params['Omega'])
            if 'boundaries' in map_params:
                map_params['boundaries'] = np.array(map_params['boundaries'])
            if 'p_mat' in map_params and map_params['p_mat'] is not None:
                map_params['p_mat'] = np.array(map_params['p_mat'])
            return map_params
        
        boundaries = np.array(metadata['boundaries'])
        
        # Start timing
        start_time = time.time()
        
        # Train MAP model with optional metadata initialization
        if init_with_metadata:
            logger.info("Training MAP model with metadata initialization...")
            map_params = self.model.fit_map_with_pytorch(
                train_x_obs_batch, train_obs_idx_batch, boundaries,
                epochs=pytorch_epochs, lr=0.01, lambda_sparsity=100.0,
                device='cuda' if torch.cuda.is_available() else 'cpu', init_metadata=metadata
            )
        else:
            logger.info("Training MAP model with random initialization...")
            map_params = {"losses": [10000000]}
            for i in range(num_restart):
                new_map_params = self.model.fit_map_with_pytorch(
                    train_x_obs_batch, train_obs_idx_batch, boundaries,
                    epochs=pytorch_epochs, lr=0.01, lambda_sparsity=100.0,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                if new_map_params["losses"][-1] < map_params["losses"][-1]:
                    print("Restart have better results")
                    map_params = new_map_params

        training_time = time.time() - start_time

        
        # Record training time with appropriate model type
        if train_file and training_size > 0:
            model_type = "MAPINIT" if init_with_metadata else "MAP"
            record_training_time(train_file, model_type, training_size, training_time)
        
        logger.info(f"MAP model training completed in {training_time:.2f} seconds")
        
        # Save MAP parameters if path provided
        if save_path:
            map_save_data = {
                'Omega': map_params['Omega'].tolist(),
                'boundaries': map_params['boundaries'].tolist() if 'boundaries' in map_params else boundaries.tolist(),
                'p_mat': map_params['p_mat'].tolist() if 'p_mat' in map_params else None,
                'losses': map_params.get('losses', []),
                'training_info': {
                    'epochs': pytorch_epochs,
                    'lr': 0.02,
                    'lambda_sparsity': 100.0,
                    'n_variables': self.n,
                    'k_categories': self.k,
                    'training_patterns': len(train_x_obs_batch),
                    'training_time_seconds': training_time,
                    'initialized_with_metadata': init_with_metadata
                },
                'metadata': metadata
            }
            
            with open(save_path, 'w') as f:
                json.dump(map_save_data, f, indent=2)
            logger.info(f"MAP model parameters saved to: {save_path}")
        
        return map_params


class NeuralModelEvaluator:
    """Handles training and evaluation of the neural model."""
    
    def __init__(self, n: int, k: int, device: str):
        self.n = n
        self.k = k
        self.device = device
    
    def train_neural_model(self, train_dataset: GaussianDataset, epochs: int = 20, 
                      save_path: str = None, model_scale="large",
                      train_file: str = "", training_size: int = 0) -> Any:
        """Train the neural model on training data and optionally save weights with timing."""
        
        # Check if save path exists and skip training if so
        if save_path and os.path.exists(save_path):
            logger.info(f"Neural model already exists at {save_path}, loading instead of training...")
            checkpoint = torch.load(save_path, map_location=self.device)
            
            # Import neural model classes
            from imputer_gaussian import ImputerEmbedding
            
            model_config = checkpoint['model_config']
            model = ImputerEmbedding(**model_config).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        
        # Rest of the training code remains the same...
        # Import neural model classes
        from imputer_gaussian import ImputerEmbedding
        from annotationArena import AnnotationArena
        
        # Start timing
        start_time = time.time()
        
        # Determine number of restarts based on model scale
        num_restarts = 5 if model_scale == "tiny" else 1
        best_model = None
        best_loss = float('inf')
        best_training_metrics = None
        
        logger.info(f"Training {model_scale} neural model with {num_restarts} restart(s)...")
        
        for restart_idx in range(num_restarts):
            if num_restarts > 1:
                logger.info(f"Starting restart {restart_idx + 1}/{num_restarts}")
            
            # Initialize model
            if model_scale == "large":
                model = ImputerEmbedding(
                    question_num=self.n,
                    max_choices=self.k,
                    encoder_layers_num=6,
                    attention_heads=4,
                    hidden_dim=64,
                    num_annotator=1,
                    annotator_embedding_dim=128,
                    dropout=0.1,
                ).to(self.device)
            elif model_scale == "small":
                model = ImputerEmbedding(
                    question_num=self.n,
                    max_choices=self.k,
                    encoder_layers_num=4,
                    attention_heads=4,
                    hidden_dim=64,
                    num_annotator=1,
                    annotator_embedding_dim=64,
                    dropout=0.1,
                ).to(self.device)
            elif model_scale == "tiny":
                model = ImputerEmbedding(
                    question_num=self.n,
                    max_choices=self.k,
                    encoder_layers_num=2,
                    attention_heads=4,
                    hidden_dim=64,
                    num_annotator=1,
                    annotator_embedding_dim=32,
                    dropout=0.1,
                ).to(self.device)
            
            # Create AnnotationArena
            arena = AnnotationArena(model, device=self.device)
            arena.set_dataset(train_dataset)
            
            # Set dynamic masking parameters
            arena.set_dynamic_masking_params(
                num_patterns_per_example=3,
                visible_ratio=0.6,
                masking_lambda=0.1
            )
            
            # Register training examples
            if num_restarts > 1:
                logger.info(f"Restart {restart_idx + 1}: Registering training examples...")
            else:
                logger.info("Registering training examples for neural model...")
            
            for example_idx in range(len(train_dataset)):
                variable_ids = arena.register_example(example_idx, add_all_positions=False)
                
                # Make predictions on masked positions
                for variable_id in variable_ids:
                    arena.predict(variable_id, train=True, weight=1.0)
                
                if (example_idx + 1) % 100 == 0 and (restart_idx == 0 or num_restarts == 1):
                    logger.info(f"Registered {example_idx + 1} training examples")
            
            # Training
            if num_restarts > 1:
                logger.info(f"Restart {restart_idx + 1}: Training...")
            
            training_metrics = arena.train(
                training_type='dynamic_masking',
                epochs=epochs,
                batch_size=16,
                lr=1e-4
            )
            
            current_loss = training_metrics['avg_loss']
            
            if num_restarts > 1:
                logger.info(f"Restart {restart_idx + 1}: Final loss = {current_loss:.4f}")
            
            # Check if this is the best model so far
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = model
                best_training_metrics = training_metrics
                if num_restarts > 1:
                    logger.info(f"New best model found at restart {restart_idx + 1} with loss {best_loss:.4f}")
        
        # End timing
        training_time = time.time() - start_time
        
        # Record training time
        if train_file and training_size > 0:
            record_training_time(train_file, model_scale, training_size, training_time)
        
        if num_restarts > 1:
            logger.info(f"{model_scale.capitalize()} neural training completed with {num_restarts} restarts in {training_time:.2f} seconds. Best loss: {best_loss:.4f}")
        else:
            logger.info(f"{model_scale.capitalize()} neural training completed in {training_time:.2f} seconds. Average loss: {best_training_metrics['avg_loss']:.4f}")
        
        if save_path:
            # Get the correct model config based on model_scale
            if model_scale == "large":
                model_config = {
                    'question_num': self.n,
                    'max_choices': self.k,
                    'encoder_layers_num': 6,
                    'attention_heads': 4,
                    'hidden_dim': 64,
                    'num_annotator': 1,
                    'annotator_embedding_dim': 128,
                    'dropout': 0.1,
                }
            elif model_scale == "small":
                model_config = {
                    'question_num': self.n,
                    'max_choices': self.k,
                    'encoder_layers_num': 4,
                    'attention_heads': 4,
                    'hidden_dim': 64,
                    'num_annotator': 1,
                    'annotator_embedding_dim': 64,
                    'dropout': 0.1,
                }
            elif model_scale == "tiny":
                model_config = {
                    'question_num': self.n,
                    'max_choices': self.k,
                    'encoder_layers_num': 2,
                    'attention_heads': 4,
                    'hidden_dim': 64,
                    'num_annotator': 1,
                    'annotator_embedding_dim': 32,
                    'dropout': 0.1,
                }
            
            # Save both the model state dict and training info
            save_data = {
                'model_state_dict': best_model.state_dict(),
                'model_config': model_config,
                'training_info': {
                    'epochs': epochs,
                    'batch_size': 16,
                    'lr': 1e-4,
                    'final_loss': best_training_metrics['avg_loss'],
                    'training_examples': len(train_dataset),
                    'device': str(self.device),
                    'training_time_seconds': training_time,
                    'model_scale': model_scale,
                    'num_restarts': num_restarts,
                    'best_loss': best_loss
                },
                'arena_config': {
                    'num_patterns_per_example': 3,
                    'visible_ratio': 0.6,
                    'masking_lambda': 0.1
                }
            }
            
            torch.save(save_data, save_path)
            logger.info(f"Neural model weights saved to: {save_path}")
        
        return best_model
    
    def evaluate_neural_model(self, model: Any, dev_dataset: GaussianDataset) -> Dict:
        """Evaluate neural model with KL divergence."""
        model.eval()
        total_kl_loss = 0.0
        total_positions = 0
        example_losses = []
        all_kl_divergences = []
        
        with torch.no_grad():
            for idx in range(len(dev_dataset)):
                known_questions, inputs, answers, annotators, questions = dev_dataset[idx]
                
                # Move to device
                inputs = inputs.unsqueeze(0).to(self.device)
                annotators = annotators.unsqueeze(0).to(self.device)
                questions = questions.unsqueeze(0).to(self.device)
                answers = answers.unsqueeze(0).to(self.device)
                
                # Get model predictions
                predictions, _ = model(inputs, annotators, questions)
                predicted_probs = F.softmax(predictions, dim=-1)
                
                # Compute KL divergence for each position
                example_kl_loss = 0.0
                example_positions = 0
                
                for pos in range(inputs.shape[1]):
                    target_dist = answers[0, pos]
                    if not torch.allclose(target_dist.sum(), torch.tensor(1.0), atol=1e-6):
                        continue
                    
                    # Check if this is a probabilistic target (not one-hot)
                    if (target_dist > 0).sum() > 1:
                        epsilon = 1e-10
                        pred_prob = predicted_probs[0, pos] + epsilon
                        target_prob = target_dist + epsilon
                        
                        pred_prob = pred_prob / pred_prob.sum()
                        target_prob = target_prob / target_prob.sum()
                        
                        kl_div = F.kl_div(
                            torch.log(pred_prob).to("cpu"),
                            target_prob.to("cpu"),
                            reduction='sum'
                        ).item()
                        
                        example_kl_loss += kl_div
                        example_positions += 1
                        all_kl_divergences.append(kl_div)
                
                if example_positions > 0:
                    example_avg_loss = example_kl_loss / example_positions
                    example_losses.append(example_avg_loss)
                    total_kl_loss += example_kl_loss
                    total_positions += example_positions
        
        avg_kl_loss = total_kl_loss / max(1, total_positions)
        std_kl_loss = np.std(example_losses) if example_losses else 0.0
        
        return {
            'avg_kl_divergence': avg_kl_loss,
            'std_kl_divergence': std_kl_loss,
            'total_positions_evaluated': total_positions,
            'examples_evaluated': len(example_losses),
            'all_kl_divergences': all_kl_divergences
        }


def plot_comparison_results(results: Dict, save_path: str = "model_comparison.png"):
    """Plot comparison of KL divergences across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of average KL divergences
    models = list(results.keys())
    avg_kls = [results[model]['avg_kl_divergence'] for model in models]
    std_kls = [results[model]['std_kl_divergence'] for model in models]
    
    bars = ax1.bar(models, avg_kls, yerr=std_kls, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average KL Divergence')
    ax1.set_title('Model Comparison: Average KL Divergence')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, avg_kl in zip(bars, avg_kls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{avg_kl:.4f}', ha='center', va='bottom')
    
    # Histogram comparison (if available)
    colors = ['blue', 'orange', 'green']
    for i, (model, color) in enumerate(zip(models, colors)):
        if 'all_kl_divergences' in results[model]:
            kl_divs = results[model]['all_kl_divergences']
            if kl_divs:  # Only plot if there are values
                ax2.hist(kl_divs, alpha=0.6, label=model, color=color, bins=20)
    
    ax2.set_xlabel('KL Divergence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of KL Divergences')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {save_path}")
    
    return fig


def load_saved_models(neural_model_path: str = None, map_model_path: str = None, 
                     device: str = 'cpu'):
    """
    Load previously saved neural and MAP models.
    
    Args:
        neural_model_path: Path to saved neural model (.pth file)
        map_model_path: Path to saved MAP model (.json file)
        device: Device to load neural model on
        
    Returns:
        Dictionary with loaded models and their configurations
    """
    loaded_models = {}
    
    # Load neural model
    if neural_model_path and os.path.exists(neural_model_path):
        logger.info(f"Loading neural model from: {neural_model_path}")
        
        checkpoint = torch.load(neural_model_path, map_location=device)
        model_config = checkpoint['model_config']
        
        # Import and recreate model
        from imputer_gaussian import ImputerEmbedding
        
        neural_model = ImputerEmbedding(**model_config).to(device)
        neural_model.load_state_dict(checkpoint['model_state_dict'])
        neural_model.eval()
        
        loaded_models['neural_model'] = {
            'model': neural_model,
            'config': model_config,
            'training_info': checkpoint['training_info'],
            'arena_config': checkpoint['arena_config']
        }
        
        logger.info("Neural model loaded successfully")
    
    # Load MAP model
    if map_model_path and os.path.exists(map_model_path):
        logger.info(f"Loading MAP model from: {map_model_path}")
        
        with open(map_model_path, 'r') as f:
            map_data = json.load(f)
        
        loaded_models['map_model'] = {
            'Omega': np.array(map_data['Omega']),
            'boundaries': np.array(map_data['boundaries']),
            'p_mat': np.array(map_data['p_mat']) if map_data['p_mat'] else None,
            'losses': map_data['losses'],
            'training_info': map_data['training_info'],
            'metadata': map_data['metadata']
        }
        
        logger.info("MAP model loaded successfully")
    
    return loaded_models


def evaluate_all_models(train_file: str, dev_file: str, training_size: int, 
                       n: int = 10, k: int = 5, neural_epochs: int = 20,
                       domain_epochs: int = 500, save_models: bool = True) -> Dict:
    """
    Evaluate all models on the same data and return KL divergences.
    
    Args:
        train_file: Path to training data JSON file
        dev_file: Path to dev data JSON file  
        training_size: Number of training examples to use
        n: Number of variables
        k: Number of categories
        neural_epochs: Number of epochs for neural model training
        domain_epochs: Number of epochs for domain model training
        save_models: Whether to save model weights and parameters
        
    Returns:
        Dictionary with KL divergence results for all models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info(f"Loading datasets with training_size={training_size}")
    train_dataset = GaussianDataset(train_file, is_training=True, data_num=training_size)
    dev_dataset = GaussianDataset(dev_file, is_training=False, data_num=100)  # Fixed dev size
    
    # Get metadata
    metadata = train_dataset.metadata
    if not metadata:
        raise ValueError("Training data must include metadata with true parameters")
    
    results = {}
    
    # Setup save paths if requested
    if save_models:
        train_file_name = train_file
        
        # Save paths for both MAP models
        map_save_path = f"models/map_model_train_{training_size}_{train_file_name}.json"
        mapinit_save_path = f"models/mapinit_model_train_{training_size}_{train_file_name}.json"
    else:
        map_save_path = None
        mapinit_save_path = None
    
    # Extract train file name without path and extension for timing records
    train_file_name = train_file
    
    domain_evaluator = DomainSpecificEvaluator(n, k)
    domain_evaluator.setup_model_from_metadata(metadata)
    
    dev_patterns = domain_evaluator.extract_dev_patterns(dev_dataset.data)
    train_x_obs_batch, train_obs_idx_batch = domain_evaluator.extract_training_patterns(
        train_dataset.data
    )
    
    # Train regular MAP model (random initialization)
    logger.info("\n" + "="*60)
    logger.info("TRAINING DOMAIN-SPECIFIC MODEL WITH RANDOM INITIALIZATION")
    logger.info("="*60)

    print(map_save_path)
    
    if os.path.exists(map_save_path):
        print(f"Skip training {map_save_path}")
    else:
        map_params = domain_evaluator.train_and_evaluate_map_model(
            train_x_obs_batch, train_obs_idx_batch, metadata, dev_patterns, 
            dev_dataset.data, pytorch_epochs=domain_epochs, save_path=map_save_path,
            train_file=train_file_name, training_size=training_size,
            init_with_metadata=False
        )
    
    # Train MAP model with metadata initialization
    logger.info("\n" + "="*60)
    logger.info("TRAINING DOMAIN-SPECIFIC MODEL WITH METADATA INITIALIZATION")
    logger.info("="*60)

    if os.path.exists(mapinit_save_path):
        print(f"Skip training {mapinit_save_path}")
    else:
        mapinit_params = domain_evaluator.train_and_evaluate_map_model(
            train_x_obs_batch, train_obs_idx_batch, metadata, dev_patterns, 
            dev_dataset.data, pytorch_epochs=domain_epochs, save_path=mapinit_save_path,
            train_file=train_file_name, training_size=training_size,
            init_with_metadata=True
        )
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING NEURAL MODELS")
    logger.info("="*60)
    
    # Train neural models of different sizes
    for size in ["large", "small", "tiny"]:
        if save_models:
            neural_save_path = f"models/neural_model_train_{training_size}_{train_file_name}_{size}.pth"
        else:
            neural_save_path = None

        print(neural_save_path)
        
        if os.path.exists(neural_save_path):
            print(f"Skip training {neural_save_path}")
            continue
            
        neural_evaluator = NeuralModelEvaluator(n, k, device)
        neural_model = neural_evaluator.train_neural_model(
            train_dataset, epochs=neural_epochs, save_path=neural_save_path, 
            model_scale=size, train_file=train_file_name, training_size=training_size
        )
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED - Check training_times.json for timing records")
    logger.info("="*60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate all models on Gaussian data')
    parser.add_argument('--train_file', default='gaussian_dev_10_obs50_new.json', 
                       help='Training data file')
    parser.add_argument('--dev_file', default='gaussian_train_10_obs50_new.json',
                       help='Dev data file')
    parser.add_argument('--training_size', type=int, default=100,
                       help='Number of training examples to use')
    parser.add_argument('--n', type=int, default=10, help='Number of variables')
    parser.add_argument('--k', type=int, default=5, help='Number of categories')
    parser.add_argument('--neural_epochs', type=int, default=20,
                       help='Number of epochs for neural model')
    parser.add_argument('--domain_epochs', type=int, default=40,
                       help='Number of epochs for domain model')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save trained model weights and parameters')
    parser.add_argument('--no_save_models', action='store_false', dest='save_models',
                       help='Do not save trained model weights and parameters')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_all_models(
        train_file=args.train_file,
        dev_file=args.dev_file,
        training_size=args.training_size,
        n=args.n,
        k=args.k,
        neural_epochs=args.neural_epochs,
        domain_epochs=args.domain_epochs,
        save_models=args.save_models
    )
    


if __name__ == "__main__":
    main()