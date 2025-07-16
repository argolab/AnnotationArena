import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import copy
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConformalPredictor:
    """
    Conformal prediction for test-time annotation stopping.
    Uses high-probability score method for discrete ordinal classification.
    """
    
    def __init__(self, model, calibration_dataset, alpha=0.1, device=None):
        """
        Initialize conformal predictor.
        
        Args:
            model: Trained imputer model
            calibration_dataset: AnnotationDataset with fully annotated examples
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            device: Device for model inference
        """
        self.model = model
        self.calibration_dataset = calibration_dataset
        self.alpha = alpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store position-specific thresholds: thresholds[num_observed][position] = threshold_value
        self.thresholds = {}
        
        # Initialize thresholds dict structure
        for num_observed in range(1, 14):  # 1 to 13 observed positions
            self.thresholds[num_observed] = {}
            for position in range(14):  # 14 positions total
                self.thresholds[num_observed][position] = float('inf')
                
        logger.info(f"ConformalPredictor initialized: alpha={alpha}, device={device}")
    
    def create_partial_observation(self, example_data, target_position, observed_positions):
        """
        Create partially observed version of calibration example.
        
        Args:
            example_data: Full calibration example dict
            target_position: Position to keep masked (target for prediction)
            observed_positions: List of positions to unmask/observe
            
        Returns:
            dict: Modified example with partial observations
        """
        # Deep copy to avoid modifying original
        partial_example = copy.deepcopy(example_data)
        
        # Step 1: Mask all positions 
        for i in range(14):
            partial_example['input'][i][0] = 1  # Set mask bit to 1 (masked)
            partial_example['known_questions'][i] = 0
            # Zero out the probability values (keep only mask bit)
            for j in range(1, 6):
                partial_example['input'][i][j] = 0.0
        
        # Step 2: Unmask observed positions with true values
        for pos in observed_positions:
            if pos != target_position:  # Don't unmask target
                partial_example['input'][pos][0] = 0  # Set mask bit to 0 (unmasked)
                partial_example['known_questions'][pos] = 1
                # Copy true probability values from answers
                for j in range(5):
                    partial_example['input'][pos][j+1] = example_data['answers'][pos][j]
        
        return partial_example
    
    def get_model_prediction(self, example_data):
        """
        Get model predictions for all positions.
        
        Args:
            example_data: Example dict with partial observations
            
        Returns:
            torch.Tensor: Probability distributions [14, 5] 
        """
        # Convert to model input format
        known_questions = torch.tensor(example_data['known_questions'], dtype=torch.int64).unsqueeze(0)
        inputs = torch.tensor(example_data['input'], dtype=torch.float32).unsqueeze(0)
        annotators = torch.tensor(example_data['annotators'], dtype=torch.int64).unsqueeze(0)
        questions = torch.tensor(example_data['questions'], dtype=torch.int64).unsqueeze(0)
        
        # Handle embeddings
        if 'text_embedding' in example_data and example_data['text_embedding'] is not None:
            embeddings = torch.tensor(example_data['text_embedding'], dtype=torch.float32).unsqueeze(0)
        else:
            embeddings = None
        
        # Move to device
        inputs = inputs.to(self.device)
        annotators = annotators.to(self.device) 
        questions = questions.to(self.device)
        known_questions = known_questions.to(self.device)
        if embeddings is not None:
            embeddings = embeddings.to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs, annotators, questions, embeddings)  # [1, 14, 5]
            probabilities = F.softmax(logits, dim=-1)  # [1, 14, 5]
        
        return probabilities.squeeze(0)  # [14, 5]
    
    def compute_conformal_score(self, predicted_probs, true_answer):
        """
        Compute high-probability conformal score.
        
        Args:
            predicted_probs: Model probability distribution [5]
            true_answer: True probability distribution [5]
            
        Returns:
            float: Conformal score (higher = worse prediction)
        """
        # Get true class via argmax
        true_class = np.argmax(true_answer)
        
        # High-probability score: negative log probability of true class
        true_class_prob = predicted_probs[true_class].item()
        
        # Avoid log(0) by adding small epsilon
        score = -np.log(max(true_class_prob, 1e-10))
        
        return score
    
    def calibrate(self, num_patterns_per_example=2):
        """
        Perform conformal calibration.
        
        Args:
            num_patterns_per_example: Number of random observation patterns per example
        """
        logger.info(f"Starting conformal calibration with {len(self.calibration_dataset)} examples")
        
        # For each (num_observed, position) pair, collect scores
        all_scores = {}
        for num_observed in range(1, 14):
            all_scores[num_observed] = {}
            for position in range(14):
                all_scores[num_observed][position] = []
        
        # Main calibration loop
        for cal_idx in range(len(self.calibration_dataset)):
            example_data = self.calibration_dataset.get_data_entry(cal_idx)
            
            # For each number of observed positions
            for num_observed in range(1, 14):
                # For each target position
                for target_position in range(14):
                    # Generate multiple random observation patterns
                    for pattern_idx in range(num_patterns_per_example):
                        # Select random positions to observe (excluding target)
                        available_positions = [p for p in range(14) if p != target_position]
                        if len(available_positions) < num_observed:
                            continue
                            
                        observed_positions = random.sample(available_positions, num_observed)
                        
                        try:
                            # Create partial observation
                            partial_example = self.create_partial_observation(
                                example_data, target_position, observed_positions
                            )
                            
                            # Get model prediction
                            predicted_probs = self.get_model_prediction(partial_example)
                            target_probs = predicted_probs[target_position]
                            
                            # Compute conformal score
                            true_answer = example_data['answers'][target_position]
                            score = self.compute_conformal_score(target_probs, true_answer)
                            
                            # Store score
                            all_scores[num_observed][target_position].append(score)
                            
                        except Exception as e:
                            logger.warning(f"Error in calibration: {e}")
                            continue
            
            if (cal_idx + 1) % 20 == 0:
                logger.info(f"Processed {cal_idx + 1}/{len(self.calibration_dataset)} calibration examples")
        
        # Compute quantile thresholds
        logger.info("Computing quantile thresholds...")
        for num_observed in range(1, 14):
            for position in range(14):
                scores = all_scores[num_observed][position]
                if len(scores) > 0:
                    # Conformal quantile with finite sample correction
                    n = len(scores)
                    quantile_level = (1 - self.alpha) * (1 + 1/n)
                    threshold = np.quantile(scores, quantile_level)
                    self.thresholds[num_observed][position] = threshold
                    
                    logger.debug(f"num_obs={num_observed}, pos={position}: "
                               f"{len(scores)} scores, threshold={threshold:.4f}")
                else:
                    logger.warning(f"No scores for num_observed={num_observed}, position={position}")
                    self.thresholds[num_observed][position] = float('inf')
        
        logger.info("Conformal calibration completed")
    
    def should_stop_annotating(self, example_data, current_observations, width_threshold=2):
        """
        Decide whether to stop annotating based on conformal prediction sets.
        
        Args:
            example_data: Test example (partially annotated)
            current_observations: Set of currently observed positions  
            width_threshold: Maximum allowed prediction set width
            
        Returns:
            bool: True if should stop annotating, False if should continue
        """
        num_observed = len(current_observations)
        
        # Edge cases
        if num_observed == 0:
            return False  # Need at least one observation
        if num_observed >= 13:
            return True  # Almost everything is observed
        if num_observed not in self.thresholds:
            return False  # No calibration data for this observation count
        
        # Get unobserved positions
        unobserved_positions = [p for p in range(14) if p not in current_observations]
        
        try:
            # Get model predictions for current state
            predicted_probs = self.get_model_prediction(example_data)
            
            # Check prediction set width for each unobserved position
            for position in unobserved_positions:
                threshold = self.thresholds[num_observed][position]
                position_probs = predicted_probs[position]
                
                # Compute conformal prediction set
                prediction_set = []
                for class_idx in range(5):
                    # Include class if prob >= exp(-threshold)
                    if position_probs[class_idx].item() >= np.exp(-threshold):
                        prediction_set.append(class_idx)
                
                # Check if prediction set is too wide
                if len(prediction_set) > width_threshold:
                    return False  # Continue annotating
            
            return True  # All prediction sets are narrow enough
            
        except Exception as e:
            logger.warning(f"Error in stopping decision: {e}")
            return False  # Conservative: continue annotating
    
    def save_calibration(self, path):
        """Save calibration thresholds to file."""
        calibration_data = {
            'alpha': self.alpha,
            'thresholds': self.thresholds,
            'num_positions': 14
        }
        
        with open(path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Conformal calibration saved to {path}")
    
    def load_calibration(self, path):
        """Load calibration thresholds from file."""
        with open(path, 'r') as f:
            calibration_data = json.load(f)
        
        self.alpha = calibration_data['alpha']
        self.thresholds = {}
        
        # Convert string keys back to integers
        for num_obs_str, pos_dict in calibration_data['thresholds'].items():
            num_obs_int = int(num_obs_str)
            self.thresholds[num_obs_int] = {}
            
            for pos_str, threshold in pos_dict.items():
                pos_int = int(pos_str)
                self.thresholds[num_obs_int][pos_int] = threshold
        
        logger.info(f"Conformal calibration loaded from {path}")