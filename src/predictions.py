"""
Standalone script to load a single trained model and extract raw predictions.
Saves predictions in JSON format with keys as "example_position" and values as raw predictions.
Also evaluates true model if not already done and logs inference times.

Usage:
    python extract_predictions.py --data_file data.json --model_path model.pth --model_type neural
    python extract_predictions.py --data_file data.json --model_path model.json --model_type map
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def record_inference_time(data_file: str, model_type: str, inference_time: float, 
                         timing_file: str = "inference_times.json", train_size=None):
    """
    Record inference time to a JSON file.
    
    Args:
        data_file: Name of the data file
        model_type: Type of model ('MAP', 'tiny', 'small', 'large', 'true')
        inference_time: Inference time in seconds
        timing_file: Path to the timing JSON file
    """
    # Create the key
    key = f"{data_file}_{model_type}"
    if train_size is not None:
        key = f'{data_file}_{model_type}_{train_size}'
    
    # Load existing data or create empty dict
    try:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        timing_data = {}
    
    # Add new timing data
    timing_data[key] = inference_time
    
    # Save back to file
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2, sort_keys=True)
    
    logger.info(f"Recorded inference time for {key}: {inference_time:.2f} seconds")


class GaussianDataset:
    """Dataset wrapper for loading Gaussian data."""
    
    def __init__(self, data_path):
        """Initialize dataset from JSON file."""
        with open(data_path, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                self.data = content['data'][:100]
                self.metadata = content.get('metadata', {})
            else:
                self.data = content
                self.metadata = {}
        
        logger.info(f"Loaded dataset with {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def get_data_entry(self, idx):
        """Get raw data entry."""
        return self.data[idx]
    
    def get_neural_input(self, idx):
        """Get data in format expected by neural model."""
        entry = self.data[idx]
        
        known_questions = torch.tensor(entry['known_questions'], dtype=torch.float32)
        inputs = torch.tensor(entry['input'], dtype=torch.float32)
        answers = torch.tensor(entry['answers'], dtype=torch.float32)
        annotators = torch.tensor(entry['annotators'], dtype=torch.long)
        questions = torch.tensor(entry['questions'], dtype=torch.long)
        
        return known_questions, inputs, answers, annotators, questions


class TrueModelPredictor:
    """Handles predictions from true model using ground truth parameters."""
    
    def __init__(self):
        self.model = None
        self.true_params = None
        self.metadata = None
    
    def load_from_dataset(self, dataset: GaussianDataset):
        """Load true model parameters from dataset metadata."""
        try:
            from multi_gaussian_new import GaussianBinningWithLinGauss
        except ImportError:
            logger.error("Could not import GaussianBinningWithLinGauss. Make sure multi_gaussian_new.py is available.")
            raise
        
        if not dataset.metadata or 'true_parameters' not in dataset.metadata:
            raise ValueError("Dataset must contain metadata with true_parameters")
        
        self.metadata = dataset.metadata
        self.true_params = {
            'Omega': np.array(self.metadata['true_parameters']['Omega']),
            'boundaries': np.array(self.metadata['boundaries'])
        }
        
        self.n = self.metadata.get('n_variables', len(self.true_params['Omega']))
        self.k = self.metadata.get('k_categories', len(self.true_params['boundaries'][0]) + 1)
        
        # Setup model with same configuration as training
        alpha = 2.0
        nu = self.n + 2
        V = np.eye(self.n)
        penalty_strength = self.metadata.get('penalty_strength', 100.0)
        
        # Set up sparsity pattern
        sparsity_pattern = np.zeros((self.n, self.n))
        for i in range(self.n):
            sparsity_pattern[i, i] = 1
            if i > 0:
                sparsity_pattern[i, i-1] = 1
                sparsity_pattern[i-1, i] = 1
        
        if 'sparsity_pattern' in self.metadata:
            sparsity_pattern = np.array(self.metadata['sparsity_pattern'])
        
        self.model = GaussianBinningWithLinGauss(
            self.n, self.k, sparsity_pattern, penalty_strength, alpha, nu, V
        )
        
        logger.info("True model loaded successfully")
        logger.info(f"Model configuration: {self.n} variables, {self.k} categories")
    
    def predict_all(self, dataset: GaussianDataset) -> Dict[str, List[float]]:
        """Generate predictions for all examples and positions using true parameters."""
        predictions = {}
        
        for idx in tqdm(range(len(dataset)), desc="Generating true model predictions"):
            entry = dataset.get_data_entry(idx)
            
            # Extract observed variables
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(entry['known_questions'], entry['input'])):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i)  # 0-based indexing for internal use
                    value = np.argmax(input_vec[1:]) + 1  # Convert to 1-based for model
                    obs_values.append(value)
            
            # Convert to model format (1-based indexing)
            obs_idx_1 = np.array([i + 1 for i in obs_indices]) if obs_indices else np.array([])
            obs_vals = np.array(obs_values) if obs_values else np.array([])
            
            # Get missing variable indices (0-based)
            missing_idx_0 = [i for i in range(self.n) if i not in obs_indices]
            
            if len(missing_idx_0) > 0:
                # Compute marginal distributions for missing variables using true parameters
                marginals = self.model._compute_marginal_distributions_batch(
                    obs_idx_1 - 1 if len(obs_idx_1) > 0 else np.array([]),  # Convert back to 0-based
                    obs_vals,
                    self.true_params['Omega'],  # Use true Omega
                    self.true_params['boundaries'],  # Use true boundaries
                    missing_idx_0,
                    n_samples=1000
                )
                
                # Store predictions for missing positions
                for var_idx in missing_idx_0:
                    key = f"{idx}_{var_idx}"
                    predictions[key] = marginals[var_idx].tolist()
            
            # For observed positions, create one-hot distributions
            for i, obs_idx in enumerate(obs_indices):
                key = f"{idx}_{obs_idx}"
                one_hot = np.zeros(self.k)
                one_hot[obs_values[i] - 1] = 1.0  # Convert to 0-based indexing
                predictions[key] = one_hot.tolist()
        
        return predictions


class NeuralModelPredictor:
    """Handles predictions from neural model."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.model_scale = None
    
    def load_model(self, model_path: str):
        """Load neural model from .pth file."""
        try:
            from imputer_gaussian import ImputerEmbedding
        except ImportError:
            logger.error("Could not import ImputerEmbedding. Make sure imputer_gaussian.py is available.")
            raise
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['model_config']
        
        # Determine model scale from config or filename
        self.model_scale = self._determine_model_scale(model_config, model_path)
        
        # Create model with appropriate configuration
        if self.model_scale == "large":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=6,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=128,
                dropout=0.1,
            ).to(self.device)
        elif self.model_scale == "small":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=4,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=64,
                dropout=0.1,
            ).to(self.device)
        elif self.model_scale == "tiny":
            self.model = ImputerEmbedding(
                question_num=model_config['question_num'],
                max_choices=model_config['max_choices'],
                encoder_layers_num=2,
                attention_heads=4,
                hidden_dim=64,
                num_annotator=1,
                annotator_embedding_dim=32,
                dropout=0.1,
            ).to(self.device)
        else:
            # Use saved config if scale not determined
            self.model = ImputerEmbedding(**model_config).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.n = model_config['question_num']
        self.k = model_config['max_choices']
        
        logger.info(f"Neural model ({self.model_scale}) loaded successfully")
        logger.info(f"Model configuration: {self.n} variables, {self.k} categories")
    
    def _determine_model_scale(self, config: dict, model_path: str) -> str:
        """Determine model scale from config or filename."""
        # Check filename for scale indicators
        filename = os.path.basename(model_path).lower()
        if 'large' in filename:
            return 'large'
        elif 'small' in filename:
            return 'small'
        elif 'tiny' in filename:
            return 'tiny'
        
        # Check config for scale indicators
        if 'encoder_layers_num' in config:
            layers = config['encoder_layers_num']
            if layers >= 6:
                return 'large'
            elif layers >= 4:
                return 'small'
            else:
                return 'tiny'
        
        # Default to small if unclear
        return 'small'
    
    def predict_all(self, dataset: GaussianDataset) -> Dict[str, List[float]]:
        """Generate predictions for all examples and positions."""
        predictions = {}
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Generating neural predictions"):
                known_questions, inputs, answers, annotators, questions = dataset.get_neural_input(idx)
                
                # Move to device and add batch dimension
                inputs = inputs.unsqueeze(0).to(self.device)
                annotators = annotators.unsqueeze(0).to(self.device)
                questions = questions.unsqueeze(0).to(self.device)
                
                # Get model predictions
                logits, _ = self.model(inputs, annotators, questions)
                predicted_probs = F.softmax(logits, dim=-1)
                
                # Store predictions for each position
                for pos in range(inputs.shape[1]):
                    key = f"{idx}_{pos}"
                    predictions[key] = predicted_probs[0, pos].cpu().tolist()
        
        return predictions


class DomainModelPredictor:
    """Handles predictions from domain-specific MAP model."""
    
    def __init__(self):
        self.model = None
        self.map_params = None
        self.metadata = None
    
    def load_model(self, model_path: str):
        """Load MAP model from .json file."""
        try:
            from multi_gaussian_new import GaussianBinningWithLinGauss
        except ImportError:
            logger.error("Could not import GaussianBinningWithLinGauss. Make sure multi_gaussian_new.py is available.")
            raise
        
        with open(model_path, 'r') as f:
            map_data = json.load(f)
        
        self.map_params = {
            'Omega': np.array(map_data['Omega']),
            'boundaries': np.array(map_data['boundaries']),
            'p_mat': np.array(map_data['p_mat']) if map_data['p_mat'] else None
        }
        
        self.metadata = map_data['metadata']
        self.n = self.metadata.get('n_variables', len(self.map_params['Omega']))
        self.k = self.metadata.get('k_categories', len(self.map_params['boundaries'][0]) - 1)
        
        # Setup model with same configuration as training
        alpha = 2.0
        nu = self.n + 2
        V = np.eye(self.n)
        penalty_strength = self.metadata.get('penalty_strength', 100.0)
        
        # Set up sparsity pattern
        sparsity_pattern = np.zeros((self.n, self.n))
        for i in range(self.n):
            sparsity_pattern[i, i] = 1
            if i > 0:
                sparsity_pattern[i, i-1] = 1
                sparsity_pattern[i-1, i] = 1
        
        if 'sparsity_pattern' in self.metadata:
            sparsity_pattern = np.array(self.metadata['sparsity_pattern'])
        
        self.model = GaussianBinningWithLinGauss(
            self.n, self.k, sparsity_pattern, penalty_strength, alpha, nu, V
        )
        
        logger.info("MAP model loaded successfully")
        logger.info(f"Model configuration: {self.n} variables, {self.k} categories")
    
    def predict_all(self, dataset: GaussianDataset) -> Dict[str, List[float]]:
        """Generate predictions for all examples and positions."""
        predictions = {}
        
        for idx in tqdm(range(len(dataset)), desc="Generating MAP predictions"):
            entry = dataset.get_data_entry(idx)
            
            # Extract observed variables
            obs_indices = []
            obs_values = []
            
            for i, (known, input_vec) in enumerate(zip(entry['known_questions'], entry['input'])):
                if known == 1.0:  # Observed variable
                    obs_indices.append(i)  # 0-based indexing for internal use
                    value = np.argmax(input_vec[1:]) + 1  # Convert to 1-based for model
                    obs_values.append(value)
            
            # Convert to model format (1-based indexing)
            obs_idx_1 = np.array([i + 1 for i in obs_indices]) if obs_indices else np.array([])
            obs_vals = np.array(obs_values) if obs_values else np.array([])
            
            # Get missing variable indices (0-based)
            missing_idx_0 = [i for i in range(self.n) if i not in obs_indices]
            
            if len(missing_idx_0) > 0:
                # Compute marginal distributions for missing variables
                marginals = self.model._compute_marginal_distributions_batch(
                    obs_idx_1 - 1 if len(obs_idx_1) > 0 else np.array([]),  # Convert back to 0-based
                    obs_vals,
                    self.map_params['Omega'],
                    self.map_params['boundaries'],
                    missing_idx_0,
                    n_samples=3000
                )
                
                # Store predictions for missing positions
                for var_idx in missing_idx_0:
                    key = f"{idx}_{var_idx}"
                    predictions[key] = marginals[var_idx].tolist()
            
            # For observed positions, create one-hot distributions
            for i, obs_idx in enumerate(obs_indices):
                key = f"{idx}_{obs_idx}"
                one_hot = np.zeros(self.k)
                one_hot[obs_values[i] - 1] = 1.0  # Convert to 0-based indexing
                predictions[key] = one_hot.tolist()
        
        return predictions


def check_and_evaluate_true_model(data_file: str) -> str:
    """
    Check if true model has been evaluated. If not, evaluate it.
    
    Args:
        data_file: Path to the data file
        
    Returns:
        Path to the true model predictions file
    """
    data_file_name = os.path.splitext(os.path.basename(data_file))[0]
    true_model_file = f"predictions/predictions_true_model_{data_file_name}.json"
    
    # Check if true model predictions already exist
    if os.path.exists(true_model_file):
        logger.info(f"True model predictions already exist: {true_model_file}")
        return true_model_file
    
    logger.info(f"True model not evaluated yet. Evaluating for {data_file}")
    
    # Load dataset
    dataset = GaussianDataset(data_file)
    
    # Check if dataset has true parameters
    if not dataset.metadata or 'true_parameters' not in dataset.metadata:
        logger.warning(f"Dataset {data_file} does not contain true parameters. Skipping true model evaluation.")
        return None
    
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Evaluate true model with timing
    start_time = time.time()
    
    predictor = TrueModelPredictor()
    predictor.load_from_dataset(dataset)
    predictions = predictor.predict_all(dataset)
    
    inference_time = time.time() - start_time
    
    # Record inference time
    record_inference_time(data_file_name, "true", inference_time)
    
    # Save predictions
    with open(true_model_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"True model predictions saved to: {true_model_file}")
    logger.info(f"True model inference completed in {inference_time:.2f} seconds")
    
    return true_model_file


def extract_predictions(data_file: str, model_path: str, model_type: str, 
                       device: str = 'cpu', output_file: str = None) -> Dict[str, List[float]]:
    """
    Extract raw predictions from a trained model.
    
    Args:
        data_file: Path to data JSON file
        model_path: Path to model file (.pth for neural, .json for MAP)
        model_type: Type of model ('neural' or 'map')
        device: Device for neural model
        output_file: Optional output file path
    
    Returns:
        Dictionary with predictions
    """
    # Load dataset
    dataset = GaussianDataset(data_file)
    
    # Record timing
    start_time = time.time()
    
    # Load model and generate predictions
    if model_type.lower() == 'neural':
        predictor = NeuralModelPredictor(device)
        predictor.load_model(model_path)
        predictions = predictor.predict_all(dataset)
        model_scale = predictor.model_scale
        
    elif model_type.lower() == 'map':
        predictor = DomainModelPredictor()
        predictor.load_model(model_path)
        predictions = predictor.predict_all(dataset)
        model_scale = "MAP"
        
    else:
        raise ValueError("model_type must be 'neural' or 'map'")
    
    inference_time = time.time() - start_time
    train_size = model_path.split("_")[3]
    
    # Record inference time
    data_file_name = os.path.splitext(os.path.basename(data_file))[0]
    record_inference_time(data_file_name, model_scale, inference_time, train_size=train_size)
    
    logger.info(f"Generated predictions for {len(predictions)} (example, position) pairs")
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    # Save predictions if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to: {output_file}")
    
    return predictions


def main():
    """Main function with argument parsing."""
    file_list = [f for f in os.listdir("models") if "map_model" in f or "mapinit_model" in f or ("neural_model" in f and ("tiny" in f or "small" in f or "large" in f))]
    file_list.sort()
    
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Keep track of processed data files to avoid evaluating true model multiple times
    processed_data_files = set()
    
    for file in tqdm(file_list):
        print(f"Processing {file}")
        if "map_model" in file or "mapinit_model" in file:
            model_type = "map"
            device = "cpu"
        else:
            model_type = "neural"
            device = "cuda"

        obs_type = file.split("_")[7]
        prefix = file.split("_")[4]
        data_file = f"{prefix}_dev_10_{obs_type}_new.json"

        # Check and evaluate true model if needed (only once per data file)
        if data_file not in processed_data_files:
            true_model_file = check_and_evaluate_true_model(data_file)
            processed_data_files.add(data_file)
        else:
            data_file_name = os.path.splitext(os.path.basename(data_file))[0]
            true_model_file = f"predictions/predictions_true_model_{data_file_name}.json"

        output_file = f"predictions/predictions_{file.replace('.json', '').replace('.pth', '')}.json"
        if os.path.exists(output_file):
            print(f"File {file} already processed")
            continue
    
        # Extract predictions
        predictions = extract_predictions(
            data_file=data_file,
            model_path=f"models/{file}",
            model_type=model_type,
            device=device,
            output_file=output_file
        )
        
        # Print summary
        print(f"\nExtracted {len(predictions)} predictions")
        print(f"Model type: {model_type}")
        print(f"Model path: {file}")
        print(f"Data file: {data_file}")
        print(f"Output file: {output_file}")
        if true_model_file and os.path.exists(true_model_file):
            print(f"True model file: {true_model_file}")
        
        # Show example predictions
        if predictions:
            example_key = list(predictions.keys())[0]
            example_pred = predictions[example_key]
            print(f"\nExample prediction for {example_key}:")
            print(f"  Shape: {len(example_pred)} categories")
            print(f"  Values: {[f'{v:.4f}' for v in example_pred]}")


if __name__ == "__main__":
    main()