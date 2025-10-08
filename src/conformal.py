"""
Conformal Prediction for Test-Time Annotation Stopping using CQR approach.
Extends trained Variable Gradient model with quantile regression heads.

Author: Based on your requirements
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import copy
import logging
from tqdm.auto import tqdm
import math
import random

from config import Config, ModelConfig
from utils import AnnotationDataset
from annotationArena import AnnotationArena
from imputerExpandedAblation import ImputerEmbedding
from selection import VOISelectionStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantileRegressionHead(nn.Module):
    """Quantile regression head for CQR."""
    
    def __init__(self, input_dim, num_positions=14):
        super().__init__()
        self.num_positions = num_positions
        # Separate heads for each position
        self.lower_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_positions)
        ])
        self.upper_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_positions)
        ])
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, num_positions, feature_dim]
        Returns:
            lower_quantiles: [batch_size, num_positions, 1]
            upper_quantiles: [batch_size, num_positions, 1]
        """
        batch_size, num_pos, _ = features.shape
        
        lower_quantiles = []
        upper_quantiles = []
        
        for i in range(min(num_pos, self.num_positions)):
            lower_q = self.lower_heads[i](features[:, i, :])
            upper_q = self.upper_heads[i](features[:, i, :])
            lower_quantiles.append(lower_q)
            upper_quantiles.append(upper_q)
        
        return torch.stack(lower_quantiles, dim=1), torch.stack(upper_quantiles, dim=1)

class CQRExtendedModel(nn.Module):
    """Extended model with frozen encoder and quantile regression heads."""
    
    def __init__(self, trained_model, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.device = trained_model.device
        
        # Freeze the entire trained model
        self.encoder = trained_model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Add quantile regression heads
        feature_dim = self.encoder.feature_dim
        self.quantile_heads = QuantileRegressionHead(feature_dim, num_positions=14)
        
        # Keep reference to original model for compatibility
        self.trained_model = trained_model
        
        logger.info(f"CQR model initialized with α={alpha}, feature_dim={feature_dim}")
    
    def forward(self, x, annotators, questions, embeddings):
        """Forward pass through frozen encoder + quantile heads."""
        with torch.no_grad():
            # Get features from frozen encoder
            features, _ = self.encoder(x, annotators, questions, embeddings)
        
        # Pass through quantile heads
        lower_quantiles, upper_quantiles = self.quantile_heads(features)
        
        return lower_quantiles, upper_quantiles
    
    def predict_original(self, inputs, annotators, questions, embeddings, positions=None, train=False, weight=1.0, example_idx=None):
        """Use original model for predictions during VOI selection."""
        return self.trained_model.predict(inputs, annotators, questions, embeddings, positions, train, weight, example_idx)

def create_random_mask_pattern(num_positions, mask_probability=0.5):
    """Create random masking pattern with given probability."""
    mask_pattern = []
    for _ in range(num_positions):
        if random.random() < mask_probability:
            mask_pattern.append(1)  # Masked
        else:
            mask_pattern.append(0)  # Observed
    return mask_pattern

def apply_mask_to_data_entry(data_entry, mask_pattern):
    """Apply masking pattern to a data entry and return masked positions."""
    masked_data = copy.deepcopy(data_entry)
    masked_positions = []
    
    for i, mask_bit in enumerate(mask_pattern):
        if i < len(masked_data['input']):
            if mask_bit == 1:  # Mask this position
                masked_data['input'][i][0] = 1  # Set mask bit
                # Zero out the 5 probability values (positions 1-5)
                masked_data['input'][i][1:6] = [0.0, 0.0, 0.0, 0.0, 0.0]
                masked_positions.append(i)
            else:  # Keep this position observed
                masked_data['input'][i][0] = 0  # Keep unmasked
                # Keep original 5 probabilities intact
    
    return masked_data, masked_positions

def quantile_loss(predictions, targets, quantile):
    """Quantile loss function."""
    errors = targets - predictions
    loss = torch.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return loss.mean()

def train_quantile_heads(cqr_model, calibration_dataset, alpha=0.1, epochs=100, batch_size=8, lr=1e-3, mask_probability=0.5):
    """Train quantile heads using randomly masked calibration data."""
    
    logger.info(f"Training quantile heads on {len(calibration_dataset)} calibration examples")
    logger.info(f"Using mask probability: {mask_probability}, alpha: {alpha}")
    
    # Only optimize quantile heads parameters
    optimizer = optim.Adam(cqr_model.quantile_heads.parameters(), lr=lr)
    
    cqr_model.train()
    cqr_model.encoder.eval()  # Keep encoder frozen
    
    total_losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle indices for each epoch
        indices = list(range(len(calibration_dataset)))
        random.shuffle(indices)
        
        # Create batches
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            batch_losses = []
            
            for idx in batch_indices:
                # Get original calibration data (fully observed)
                original_data = calibration_dataset.get_data_entry(idx)
                
                # Create random mask pattern
                num_positions = len(original_data['input'])
                mask_pattern = create_random_mask_pattern(num_positions, mask_probability)
                
                # Apply mask to create training example
                masked_data, masked_positions = apply_mask_to_data_entry(original_data, mask_pattern)
                
                # Skip if no positions are masked
                if not masked_positions:
                    continue

                # # Create temporary dataset with masked data and use its tensor conversion
                # temp_dataset = AnnotationDataset([masked_data])
                # known_questions, inputs, answers, annotators, questions, embeddings = temp_dataset[0]

                # # Move to device
                # inputs = inputs.unsqueeze(0).to(cqr_model.device)
                # answers = answers.unsqueeze(0).to(cqr_model.device)
                # annotators = annotators.unsqueeze(0).to(cqr_model.device)
                # questions = questions.unsqueeze(0).to(cqr_model.device)
                # if embeddings is not None:
                #     embeddings = embeddings.unsqueeze(0).to(cqr_model.device)
                
                # Convert to tensors
                inputs = torch.tensor(masked_data['input'], dtype=torch.float32).unsqueeze(0).to(cqr_model.device)
                answers = torch.tensor(original_data['answers'], dtype=torch.float32).unsqueeze(0).to(cqr_model.device)
                annotators = torch.tensor(masked_data['annotators'], dtype=torch.int64).unsqueeze(0).to(cqr_model.device)
                questions = torch.tensor(masked_data['questions'], dtype=torch.int64).unsqueeze(0).to(cqr_model.device)
                
                embeddings = None
                if 'text_embedding' in masked_data:
                    embeddings = torch.tensor(masked_data['text_embedding'], dtype=torch.float32).unsqueeze(0).to(cqr_model.device)
                
                # Forward pass through CQR model
                lower_quantiles, upper_quantiles = cqr_model(inputs, annotators, questions, embeddings)
                
                # Compute losses only for masked positions
                position_losses = []
                
                for pos in masked_positions:
                    if pos < lower_quantiles.shape[1]:
                        target = answers[0, pos].unsqueeze(0)  # [1]
                        
                        # Lower quantile loss (α/2)
                        lower_pred = lower_quantiles[0, pos, 0]  # [1]
                        lower_loss = quantile_loss(lower_pred.unsqueeze(0), target, alpha/2)
                        
                        # Upper quantile loss (1-α/2)
                        upper_pred = upper_quantiles[0, pos, 0]  # [1]
                        upper_loss = quantile_loss(upper_pred.unsqueeze(0), target, 1-alpha/2)
                        
                        position_losses.append(lower_loss + upper_loss)
                
                if position_losses:
                    example_loss = torch.stack(position_losses).mean()
                    batch_losses.append(example_loss)
            
            # Optimize on batch
            if batch_losses:
                batch_loss = torch.stack(batch_losses).mean()
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                epoch_losses.append(batch_loss.item())
        
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        total_losses.append(avg_epoch_loss)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Average Loss: {avg_epoch_loss:.6f}")
    
    logger.info(f"Quantile head training completed. Final loss: {total_losses[-1]:.6f}")
    return total_losses

def compute_cqr_score(lower_quantile, upper_quantile, true_value):
    """Compute CQR non-conformity score."""
    score = max(lower_quantile - true_value, true_value - upper_quantile)
    return score

def simulate_sequential_voi_for_cqr(cqr_model, calibration_dataset, example_idx, voi_strategy):
    """Simulate sequential VOI observation process for CQR calibration."""
    
    all_cqr_scores = []
    
    # Create arena with original model for VOI compatibility
    arena = AnnotationArena(cqr_model.trained_model, cqr_model.device)
    arena.set_dataset(calibration_dataset)
    
    # Get original data
    original_data = calibration_dataset.get_data_entry(example_idx)
    
    # Create artificially masked version (start with all positions masked)
    masked_data = copy.deepcopy(original_data)
    for i in range(len(masked_data['input'])):
        masked_data['input'][i][0] = 1  # Mask all positions
        masked_data['input'][i][1:] = [0.0] * 5  # Zero out answers
    
    # Create temporary dataset with masked example
    temp_dataset = AnnotationDataset([masked_data])
    arena.set_dataset(temp_dataset)
    arena.register_example(0)
    
    observed_positions = set()
    
    while len(observed_positions) < 13:  # Leave at least 1 for scoring
        # Get VOI recommendation
        try:
            voi_results = voi_strategy.select_features(
                example_idx=0,
                dataset=temp_dataset,
                num_to_select=1,
                target_questions=[0,1,2,3,4,5,6]
            )
            
            if not voi_results:
                break
                
            next_position = voi_results[0][0]
        except Exception as e:
            logger.warning(f"VOI selection failed: {e}, breaking")
            break
        
        # Before observing: get CQR predictions for all unobserved positions
        unobserved_positions = temp_dataset.get_masked_positions(0)
        
        if unobserved_positions:
            # Get current state data
            known_questions, inputs, answers, annotators, questions, embeddings = temp_dataset[0]
            inputs = inputs.unsqueeze(0).to(cqr_model.device)
            annotators = annotators.unsqueeze(0).to(cqr_model.device)
            questions = questions.unsqueeze(0).to(cqr_model.device)
            if embeddings is not None:
                embeddings = embeddings.unsqueeze(0).to(cqr_model.device)
            
            # Get quantile predictions
            with torch.no_grad():
                lower_quantiles, upper_quantiles = cqr_model(inputs, annotators, questions, embeddings)
            
            # Compute CQR scores for unobserved positions
            for pos in unobserved_positions:
                if pos < lower_quantiles.shape[1]:
                    lower_q = lower_quantiles[0, pos, 0].item()
                    upper_q = upper_quantiles[0, pos, 0].item()
                    
                    # Convert probability distribution to expected Likert value
                    true_probs = original_data['answers'][pos]
                    true_value = sum((i+1) * prob for i, prob in enumerate(true_probs))  # 1*p1 + 2*p2 + 3*p3 + 4*p4 + 5*p5
                    
                    cqr_score = compute_cqr_score(lower_q, upper_q, true_value)
                    all_cqr_scores.append(cqr_score)
        
        # Observe the VOI-selected position
        try:
            arena.observe_position(0, next_position)
            observed_positions.add(next_position)
        except Exception as e:
            logger.warning(f"Failed to observe position {next_position}: {e}")
            break
    
    return all_cqr_scores

def calibrate_cqr_threshold(cqr_model, calibration_dataset, voi_strategy, alpha=0.1):
    """Calibrate CQR threshold using full sequential simulation."""
    
    logger.info(f"Starting CQR calibration on {len(calibration_dataset)} examples")
    
    all_cqr_scores = []
    
    for example_idx in tqdm(range(len(calibration_dataset)), desc="CQR Calibration"):
        try:
            example_scores = simulate_sequential_voi_for_cqr(
                cqr_model, calibration_dataset, example_idx, voi_strategy
            )
            all_cqr_scores.extend(example_scores)
        except Exception as e:
            logger.warning(f"Failed to process example {example_idx}: {e}")
            continue
    
    if not all_cqr_scores:
        raise ValueError("No CQR scores collected during calibration")
    
    # Compute threshold as (1-α) quantile
    threshold = np.quantile(all_cqr_scores, 1 - alpha)
    
    logger.info(f"CQR calibration completed. Collected {len(all_cqr_scores)} scores")
    logger.info(f"Calibrated threshold (α={alpha}): {threshold:.6f}")
    
    return threshold, all_cqr_scores

def test_time_conformal_stopping(cqr_model, test_dataset, test_example_idx, voi_strategy, threshold, rmse_budget=0.5):
    """Test-time stopping algorithm using CQR intervals."""
    
    # Create arena with original model for VOI compatibility
    arena = AnnotationArena(cqr_model.trained_model, cqr_model.device)
    arena.set_dataset(test_dataset)
    arena.register_example(test_example_idx)
    
    observed_positions = set()
    decisions = []
    
    while len(observed_positions) < 14:
        unobserved_positions = test_dataset.get_masked_positions(test_example_idx)
        if not unobserved_positions:
            break
        
        # Get current state data
        known_questions, inputs, answers, annotators, questions, embeddings = test_dataset[test_example_idx]
        inputs = inputs.unsqueeze(0).to(cqr_model.device)
        annotators = annotators.unsqueeze(0).to(cqr_model.device)
        questions = questions.unsqueeze(0).to(cqr_model.device)
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(0).to(cqr_model.device)
        
        # Get quantile predictions
        with torch.no_grad():
            lower_quantiles, upper_quantiles = cqr_model(inputs, annotators, questions, embeddings)
        
        # Compute prediction intervals and estimate RMSE
        total_estimated_rmse_squared = 0
        position_intervals = {}
        
        for pos in unobserved_positions:
            if pos < lower_quantiles.shape[1]:
                lower_q = lower_quantiles[0, pos, 0].item()
                upper_q = upper_quantiles[0, pos, 0].item()
                
                # Conformal prediction interval
                interval_lower = lower_q - threshold
                interval_upper = upper_q + threshold
                interval_width = interval_upper - interval_lower
                
                position_intervals[pos] = (interval_lower, interval_upper, interval_width)
                
                # Estimate RMSE contribution: interval width approximates uncertainty
                total_estimated_rmse_squared += (interval_width / 2) ** 2
        
        estimated_rmse = math.sqrt(total_estimated_rmse_squared)
        
        decision_info = {
            'observed_positions': list(observed_positions),
            'unobserved_positions': unobserved_positions,
            'estimated_rmse': estimated_rmse,
            'position_intervals': position_intervals
        }
        decisions.append(decision_info)
        
        # Stopping decision
        if estimated_rmse <= rmse_budget:
            logger.info(f"Stopping: estimated RMSE {estimated_rmse:.4f} <= budget {rmse_budget}")
            decision_info['decision'] = 'STOP'
            break
        
        # Continue: get VOI recommendation
        try:
            voi_results = voi_strategy.select_features(
                example_idx=test_example_idx,
                dataset=test_dataset,
                num_to_select=1,
                target_questions=[0,1,2,3,4,5,6]
            )
            
            if not voi_results:
                logger.warning("No VOI results, stopping")
                decision_info['decision'] = 'NO_VOI'
                break
            
            next_position = voi_results[0][0]
            arena.observe_position(test_example_idx, next_position)
            observed_positions.add(next_position)
            
            decision_info['decision'] = 'CONTINUE'
            decision_info['selected_position'] = next_position
            
        except Exception as e:
            logger.warning(f"Failed to continue: {e}")
            decision_info['decision'] = 'ERROR'
            break
    
    final_result = {
        'observed_positions': list(observed_positions),
        'total_observations': len(observed_positions),
        'decisions': decisions,
        'final_decision': decisions[-1]['decision'] if decisions else 'UNKNOWN'
    }
    
    return final_result

def main():
    """Main function to run CQR conformal prediction experiment."""
    
    # Configuration
    model_path = "/export/fs06/psingh54/AnnotationArena/src/output/models/AblationStudy_VarGrad_3Patterns_0.5_Ratio_var_grad_dynamic_masking_hist_only_20250710_112144.pth"
    calibration_data_path = "/export/fs06/psingh54/AnnotationArena/src/input/data/calibration_holdout.json"
    alpha = 0.1  # 90% coverage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load trained model
    logger.info(f"Loading trained model from {model_path}")
    model_config = ModelConfig.HANNA
    trained_model = ImputerEmbedding(**model_config).to(device)
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model.eval()
    
    # Create CQR extended model
    cqr_model = CQRExtendedModel(trained_model, alpha=alpha).to(device)
    
    # Load calibration dataset
    logger.info(f"Loading calibration dataset from {calibration_data_path}")
    calibration_dataset = AnnotationDataset(calibration_data_path)
    
    # Train quantile heads
    logger.info("Training quantile regression heads")
    train_losses = train_quantile_heads(
        cqr_model, 
        calibration_dataset, 
        alpha=alpha, 
        epochs=50, 
        batch_size=8, 
        lr=1e-3, 
        mask_probability=0.5
    )
    
    # Initialize VOI strategy
    voi_strategy = VOISelectionStrategy(trained_model, device)
    
    # Calibrate CQR threshold
    logger.info("Calibrating CQR threshold")
    threshold, all_scores = calibrate_cqr_threshold(
        cqr_model, 
        calibration_dataset, 
        voi_strategy, 
        alpha=alpha
    )
    
    # Save results
    results = {
        'model_path': model_path,
        'alpha': alpha,
        'threshold': threshold,
        'num_calibration_examples': len(calibration_dataset),
        'num_cqr_scores': len(all_scores),
        'train_losses': train_losses,
        'cqr_scores_stats': {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'median': np.median(all_scores)
        }
    }
    
    # Save calibration results
    output_file = "cqr_calibration_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"CQR calibration completed. Results saved to {output_file}")
    logger.info(f"Calibrated threshold: {threshold:.6f}")
    logger.info(f"Ready for test-time conformal stopping with RMSE budget")

main()