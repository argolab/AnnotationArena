"""Tuned lens analyzer for examining intermediate representations with learned translators."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
import logging
from tqdm import tqdm

from imputer.data import RankingData, DataConverter
from imputer.ranking_imputer import MultiVariableImputer
from .analyzer import LogitLensResults, LayerAnalysis, VariableAnalysis, LogitLensAnalyzer

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class TunedLensConfig:
    """Configuration for tuned lens analysis."""
    learning_rate: float = 1e-3
    epochs: int = 50
    target_type: str = 'ground_truth'  # 'ground_truth' or 'final_logits'
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    train_split_ratio: float = 0.8  # Ratio for train/eval split
    random_seed: int = 42


def split_variables_for_translator_training(variables: List[RankingData], 
                                         config: TunedLensConfig) -> Tuple[List[bool], List[bool]]:
    """
    Create train/eval masks for translator training.
    
    Args:
        variables: All variables to create masks for
        config: Configuration for splitting
        
    Returns:
        Tuple of (train_mask, eval_mask) where each is a list of booleans
    """
    
    # Filter valid variables (same as before)
    valid_variables = []
    for var in variables:
        if not var.is_listwise:  # Rating variable
            if var.rating_value is not None:
                valid_variables.append(var)
        else:  # Ranking variable
            if var.ranking_order is not None and len(var.ranking_order) == 2:
                valid_variables.append(var)
    
    logger.info(f"Found {len(valid_variables)} valid variables out of {len(variables)} total")
    
    if len(valid_variables) < 2:
        raise ValueError(f"Need at least 2 valid variables for train/eval split, got {len(valid_variables)}")
    
    # Create mapping from original variables to valid variables
    valid_indices = []
    for i, var in enumerate(variables):
        if var in valid_variables:
            valid_indices.append(i)
    
    # Split valid variables by type to maintain distribution
    rating_vars = [var for var in valid_variables if not var.is_listwise]
    ranking_vars = [var for var in valid_variables if var.is_listwise]
    
    # Initialize masks
    train_mask = [False] * len(variables)
    eval_mask = [False] * len(variables)
    
    # Shuffle and split rating variables
    if rating_vars:
        random.seed(config.random_seed)
        rating_vars_shuffled = rating_vars.copy()
        random.shuffle(rating_vars_shuffled)
        
        split_idx = max(1, int(len(rating_vars_shuffled) * config.train_split_ratio))
        
        for var in rating_vars_shuffled[:split_idx]:
            idx = variables.index(var)
            train_mask[idx] = True
        for var in rating_vars_shuffled[split_idx:]:
            idx = variables.index(var)
            eval_mask[idx] = True
    
    # Shuffle and split ranking variables
    if ranking_vars:
        random.seed(config.random_seed)
        ranking_vars_shuffled = ranking_vars.copy()
        random.shuffle(ranking_vars_shuffled)
        
        split_idx = max(1, int(len(ranking_vars_shuffled) * config.train_split_ratio))
        
        for var in ranking_vars_shuffled[:split_idx]:
            idx = variables.index(var)
            train_mask[idx] = True
        for var in ranking_vars_shuffled[split_idx:]:
            idx = variables.index(var)
            eval_mask[idx] = True
    
    # Log statistics
    train_count = sum(train_mask)
    eval_count = sum(eval_mask)
    rating_train_count = sum(1 for i, var in enumerate(variables) if train_mask[i] and not var.is_listwise)
    rating_eval_count = sum(1 for i, var in enumerate(variables) if eval_mask[i] and not var.is_listwise)
    ranking_train_count = sum(1 for i, var in enumerate(variables) if train_mask[i] and var.is_listwise)
    ranking_eval_count = sum(1 for i, var in enumerate(variables) if eval_mask[i] and var.is_listwise)
    
    logger.info(f"Split: {train_count} train, {eval_count} eval")
    logger.info(f"Rating vars: {rating_train_count} train, {rating_eval_count} eval")
    logger.info(f"Ranking vars: {ranking_train_count} train, {ranking_eval_count} eval")
    
    return train_mask, eval_mask


class LayerTranslator(nn.Module):
    """Learnable translator for a single layer: [features + params] -> params."""
    
    def __init__(self, feature_dim: int, param_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.param_dim = param_dim
        
        # Linear transformation: [features + params] -> params
        self.translator = nn.Linear(feature_dim + param_dim, param_dim)
        
    def forward(self, features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, D_features] - feature stream
            params: [B, N, D_params] - param stream
        Returns:
            translated_params: [B, N, D_params] - translated param stream
        """
        # Concatenate features and params along last dimension
        combined = torch.cat([features, params], dim=-1)  # [B, N, D_features + D_params]
        
        # Apply linear transformation
        translated_params = self.translator(combined)  # [B, N, D_params]
        
        return translated_params


class TunedLensAnalyzer(LogitLensAnalyzer):
    """Analyzes intermediate representations using tuned lens technique."""
    
    def __init__(self, model: MultiVariableImputer, converter: DataConverter, 
                 device: str = 'cuda', config: Optional[TunedLensConfig] = None):
        # Initialize base class
        super().__init__(model, converter, device)
        
        self.config = config or TunedLensConfig()
        
        # Freeze the base model (override base class behavior)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get dimensions from model
        self.feature_dim = self.model.embedding_dim
        self.param_dim = self.model.embedding_provider.parameter_dimension - 1 #TODO: minus the masking bit, because we don't give it to the model.
        self.num_layers = len(self.model.blocks) + 1  # +1 for final layer after norm
        
        # Create translators for each layer
        self.translators = nn.ModuleList([
            LayerTranslator(self.feature_dim, self.param_dim)
            for _ in range(self.num_layers)
        ]).to(device)
        
        # Optimizer for translators only
        self.optimizer = torch.optim.AdamW(
            self.translators.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Loss tracking for visualization
        self.training_history = {
            'train_losses': [[] for _ in range(self.num_layers)],
            'eval_losses': [[] for _ in range(self.num_layers)],
            'train_rating_losses': [[] for _ in range(self.num_layers)],
            'train_ranking_losses': [[] for _ in range(self.num_layers)],
            'eval_rating_losses': [[] for _ in range(self.num_layers)],
            'eval_ranking_losses': [[] for _ in range(self.num_layers)],
            'epochs': []
        }
        
        # Train/eval masks for loss calculation
        self.train_mask = None
        self.eval_mask = None
        
    def _get_targets(self, variables: List[RankingData], 
                     final_logits: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Get target labels/logits for training."""
        
        if self.config.target_type == 'ground_truth':
            return self._get_ground_truth_targets(variables)
        elif self.config.target_type == 'final_logits':
            if final_logits is None:
                raise ValueError("final_logits required when target_type='final_logits'")
            return self._get_final_logits_targets(variables, final_logits)
        else:
            raise ValueError(f"Unknown target_type: {self.config.target_type}")
    
    def _get_final_logits_targets(self, variables: List[RankingData], 
                                 final_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract final logits targets with proper masking."""
        
        num_vars = len(variables)
        
        # Create masks to indicate valid targets
        rating_mask = [False] * num_vars
        ranking_mask = [False] * num_vars
        
        for i, var in enumerate(variables):
            if not var.is_listwise:  # Rating variable
                assert var.rating_value is not None, "rating value cannot be None for rating variable"
                rating_mask[i] = True
            else:  # Ranking variable
                assert var.ranking_order is not None and len(var.ranking_order) == 2, "ranking order cannot be None for ranking variable"
                ranking_mask[i] = True
        
        # Convert masks to tensors [N]
        rating_mask_tensor = torch.tensor(rating_mask, device=self.device, dtype=torch.bool)
        ranking_mask_tensor = torch.tensor(ranking_mask, device=self.device, dtype=torch.bool)
        
        # Assert mutual exclusivity
        combined_mask = rating_mask_tensor | ranking_mask_tensor
        assert combined_mask.all(), "All variables must have either rating or ranking targets"
        assert not (rating_mask_tensor & ranking_mask_tensor).any(), "Variables cannot have both rating and ranking targets"
        
        # Expand masks to batch dimension [B, N]
        batch_size = final_logits['rating'].shape[0]
        assert batch_size == 1, "batching is not supported for final logits targets"
        rating_mask_batch = rating_mask_tensor.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        ranking_mask_batch = ranking_mask_tensor.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        
        return {
            'rating': final_logits['rating'],  # [B, N, num_classes] - soft targets
            'ranking': final_logits['ranking'],  # [B, N, max_rank_size] - soft targets
            'rating_mask': rating_mask_batch,  # [B, N] - boolean mask
            'ranking_mask': ranking_mask_batch  # [B, N] - boolean mask
        }
    
    def _get_ground_truth_targets(self, variables: List[RankingData]) -> Dict[str, torch.Tensor]:
        """Extract ground truth targets and convert to soft target distributions."""
        
        num_vars = len(variables)
        
        # Initialize all targets with invalid values (-1)
        rating_targets = [-1] * num_vars
        ranking_targets = [-1] * num_vars
        
        # Create masks to indicate valid targets
        rating_mask = [False] * num_vars
        ranking_mask = [False] * num_vars
        
        for i, var in enumerate(variables):
            if not var.is_listwise:  # Rating variable
                assert var.rating_value is not None, "rating value cannot be None for rating variable"
                rating_targets[i] = var.rating_value
                rating_mask[i] = True
            else:  # Ranking variable
                assert var.ranking_order is not None and len(var.ranking_order) == 2, "ranking order cannot be None for ranking variable"
                # Convert ranking to binary: 0 if first item wins, 1 if second wins
                ranking_targets[i] = 0 if var.ranking_order[0] < var.ranking_order[1] else 1
                ranking_mask[i] = True
        
        # Convert masks to tensors [N]
        rating_mask_tensor = torch.tensor(rating_mask, device=self.device, dtype=torch.bool)
        ranking_mask_tensor = torch.tensor(ranking_mask, device=self.device, dtype=torch.bool)
        
        # Assert mutual exclusivity: each position should have exactly one mask True
        combined_mask = rating_mask_tensor | ranking_mask_tensor
        assert combined_mask.all(), "All variables must have either rating or ranking targets"
        assert not (rating_mask_tensor & ranking_mask_tensor).any(), "Variables cannot have both rating and ranking targets"
        
        # Convert hard targets to soft target distributions (logits)
        rating_soft_targets = self._hard_to_soft_targets(
            rating_targets, rating_mask_tensor, self.model.num_likert_classes
        )
        ranking_soft_targets = self._hard_to_soft_targets(
            ranking_targets, ranking_mask_tensor, self.model.max_rank_size
        )
        
        # Expand masks to batch dimension [B, N]
        batch_size = rating_soft_targets.shape[0]
        assert batch_size == 1, "batching is not supported for ground truth targets"
        rating_mask_batch = rating_mask_tensor.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        ranking_mask_batch = ranking_mask_tensor.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        
        return {
            'rating': rating_soft_targets,  # [B, N, num_classes] - soft targets
            'ranking': ranking_soft_targets,  # [B, N, max_rank_size] - soft targets
            'rating_mask': rating_mask_batch,  # [B, N] - boolean mask
            'ranking_mask': ranking_mask_batch  # [B, N] - boolean mask
        }
    
    def _hard_to_soft_targets(self, hard_targets: List[int], mask: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert hard targets to soft target distributions (logits).
        
        Args:
            hard_targets: List of hard target indices (invalid entries = -1)
            mask: Boolean mask indicating valid targets
            num_classes: Number of classes for the soft target distribution
            
        Returns:
            soft_targets: [B, N, num_classes] tensor with soft target distributions
        """
        
        num_vars = len(hard_targets)
        
        # Initialize with uniform logits (log(1/num_classes))
        uniform_logit = torch.log(torch.tensor(1.0 / num_classes, device=self.device))
        soft_targets = torch.full((1, num_vars, num_classes), uniform_logit, device=self.device)
        
        # Set high logits for correct classes (valid targets only)
        for i, target in enumerate(hard_targets):
            if mask[i] and target >= 0:
                # Set high logit for the correct class
                soft_targets[0, i, target] = 10.0  # High confidence
                # Set lower logits for other classes
                other_indices = torch.arange(num_classes, device=self.device) != target
                soft_targets[0, i, other_indices] = -10.0  # Low confidence
        
        return soft_targets
    
    def _compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute KL divergence loss directly in logit space. (for a specific layer)
        
        Args:
            predictions: Dict with keys 'rating', 'ranking'
                - 'rating': [B, N, num_classes] - predicted logits
                - 'ranking': [B, N, max_rank_size] - predicted logits
            targets: Dict with keys 'rating', 'ranking', 'rating_mask', 'ranking_mask'
                - 'rating': [B, N, num_classes] - target logits
                - 'ranking': [B, N, max_rank_size] - target logits
                - 'rating_mask': [B, N] - boolean mask for valid rating targets
                - 'ranking_mask': [B, N] - boolean mask for valid ranking targets
        
        Returns:
            Dict with keys 'total', 'rating', 'ranking' - separate losses for debugging
        """
        
        # Initialize losses
        rating_loss = torch.tensor(0.0, device=self.device)
        ranking_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        
        # Rating loss
        rating_preds = predictions['rating']  # [B, N, num_classes]
        rating_targets = targets['rating']     # [B, N, num_classes]
        rating_mask = targets['rating_mask']   # [B, N]
        
        if rating_mask.any():
            # Apply mask to all batch elements
            valid_preds = rating_preds[rating_mask]      # [valid_count, num_classes] Note pytorch flatten the masked dimensions here
            valid_targets = rating_targets[rating_mask]  # [valid_count, num_classes]
            
            # KL divergence in logit space using PyTorch's built-in function
            # Convert predictions to log-probabilities, keep targets as logits
            target_log_probs = F.log_softmax(valid_targets, dim=-1)  # [valid_count, num_classes]
            
            # KL divergence: KL(target || pred) using log_target=True for efficiency
            rating_loss = F.kl_div(target_log_probs, valid_preds, reduction='batchmean', log_target=True)
            num_valid += 1
        else:
            logger.debug("no valid rating targets")

        
        # Ranking loss
        ranking_preds = predictions['ranking']  # [B, N, max_rank_size]
        ranking_targets = targets['ranking']     # [B, N, max_rank_size]
        ranking_mask = targets['ranking_mask']    # [B, N]
        
        if ranking_mask.any():
            # Apply mask to all batch elements
            valid_preds = ranking_preds[ranking_mask]      # [valid_count, max_rank_size]
            valid_targets = ranking_targets[ranking_mask]  # [valid_count, max_rank_size]
            
            # KL divergence in logit space using PyTorch's built-in function
            # Convert predictions to log-probabilities, keep targets as logits
            target_log_probs = F.log_softmax(valid_targets, dim=-1)  # [valid_count, max_rank_size]
            
            # KL divergence: KL(target || pred) using log_target=True for efficiency
            ranking_loss = F.kl_div(target_log_probs, valid_preds, reduction='batchmean', log_target=True)
            num_valid += 1
        else:
            logger.debug("no valid ranking targets")
        
        # Compute total loss
        total_loss = rating_loss + ranking_loss
        
        return {
            'total': total_loss,
            'rating': rating_loss,
            'ranking': ranking_loss
        }
    
    
    def train_translators(self, instance_variable_lists: List[List[RankingData]]) -> Dict[str, List[float]]:
        """
        Train translators using multiple instances of variables.
        
        Args:
            instance_variable_lists: List of variable lists, each representing a different instance
                                   (e.g., [train_instance_variables, test_instance_variables])
            
        Returns:
            Training history dictionary
        """
        
        logger.info(f"Training tuned lens translators for {self.num_layers} layers...")
        logger.info(f"Target type: {self.config.target_type}")
        logger.info(f"Processing {len(instance_variable_lists)} instances...")
        
        # Combine all variables across instances for train/eval split
        all_variables = []
        for instance_vars in instance_variable_lists:
            all_variables.extend(instance_vars)
        
        logger.info(f"Total variables across all instances: {len(all_variables)}")
        
        # Create train/eval masks
        self.train_mask, self.eval_mask = split_variables_for_translator_training(all_variables, self.config)
        
        # Training loop
        best_eval_loss = float('inf')
        patience_counter = 0
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.config.epochs), desc="Training Translators", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training phase - process each instance separately
            train_losses = self._train_epoch(instance_variable_lists)
            
            # Evaluation phase - process each instance separately
            eval_losses = self._eval_epoch(instance_variable_lists)
            
            # Track losses (use epoch+1 for actual epoch number)
            self.training_history['epochs'].append(epoch + 1)
            for layer_idx in range(self.num_layers):
                self.training_history['train_losses'][layer_idx].append(train_losses['total'][layer_idx])
                self.training_history['eval_losses'][layer_idx].append(eval_losses['total'][layer_idx])
                self.training_history['train_rating_losses'][layer_idx].append(train_losses['rating'][layer_idx])
                self.training_history['train_ranking_losses'][layer_idx].append(train_losses['ranking'][layer_idx])
                self.training_history['eval_rating_losses'][layer_idx].append(eval_losses['rating'][layer_idx])
                self.training_history['eval_ranking_losses'][layer_idx].append(eval_losses['ranking'][layer_idx])
            
            # Early stopping based on average eval loss across layers
            avg_eval_loss = np.mean(eval_losses['total'])
            avg_train_loss = np.mean(train_losses['total'])
            
            # Update progress bar with loss information
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.2f}',
                'eval_loss': f'{avg_eval_loss:.2f}',
                'patience': patience_counter
            })
            
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training complete after {len(self.training_history['epochs'])} epochs")
        return self.training_history
    
    def _train_epoch(self, instance_variable_lists: List[List[RankingData]]) -> Dict[str, List[float]]:
        """Train for one epoch processing each instance separately."""
        
        self.translators.train()
        self.optimizer.zero_grad()
        
        # Process each instance separately
        layer_losses = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in range(self.num_layers)]
        layer_rating_losses = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in range(self.num_layers)]
        layer_ranking_losses = [torch.tensor(0.0, device=self.device, requires_grad=True) for _ in range(self.num_layers)]
        num_instances = len(instance_variable_lists)
        
        for instance_variables in instance_variable_lists:
            # Forward pass on this instance to get proper embeddings and attention
            with torch.no_grad():
                logits_final, hidden_intermediates = self.model(instance_variables, return_intermediate=True)
            
            # Get targets for this instance
            targets = self._get_targets(instance_variables, logits_final)
            
            # Compute loss for each layer
            for layer_idx, (features, params) in enumerate(hidden_intermediates):
                # Apply translator
                translated_params = self.translators[layer_idx](features, params)
                
                # Apply heads to translated params
                rating_logits = self.model.apply_head('rating', translated_params)
                ranking_logits = self.model.apply_head('ranking', translated_params)
                
                predictions = {
                    'rating': rating_logits,
                    'ranking': ranking_logits
                }
                
                # Compute loss for this layer
                loss_dict = self._compute_loss(predictions, targets)
                
                # Accumulate losses (avoid in-place operations)
                layer_losses[layer_idx] = layer_losses[layer_idx] + loss_dict['total']
                layer_rating_losses[layer_idx] = layer_rating_losses[layer_idx] + loss_dict['rating']
                layer_ranking_losses[layer_idx] = layer_ranking_losses[layer_idx] + loss_dict['ranking']
        
        # Average losses across instances
        for layer_idx in range(self.num_layers):
            layer_losses[layer_idx] = layer_losses[layer_idx] / num_instances
            layer_rating_losses[layer_idx] = layer_rating_losses[layer_idx] / num_instances
            layer_ranking_losses[layer_idx] = layer_ranking_losses[layer_idx] / num_instances
        
        # Backward pass and optimization
        total_loss = sum(layer_losses)
        total_loss.backward()
        self.optimizer.step()
        
        # Convert to Python floats for return
        return {
            'total': [loss.item() for loss in layer_losses],
            'rating': [loss.item() for loss in layer_rating_losses],
            'ranking': [loss.item() for loss in layer_ranking_losses]
        }
    
    def _eval_epoch(self, instance_variable_lists: List[List[RankingData]]) -> Dict[str, List[float]]:
        """Evaluate for one epoch processing each instance separately."""
        
        self.translators.eval()
        
        with torch.no_grad():
            # Process each instance separately
            layer_losses = [0.0] * self.num_layers
            layer_rating_losses = [0.0] * self.num_layers
            layer_ranking_losses = [0.0] * self.num_layers
            num_instances = len(instance_variable_lists)
            
            for instance_variables in instance_variable_lists:
                # Forward pass on this instance to get proper embeddings and attention
                logits_final, hidden_intermediates = self.model(instance_variables, return_intermediate=True)
                
                # Get targets for this instance
                targets = self._get_targets(instance_variables, logits_final)
            
                # Compute loss for each layer
                for layer_idx, (features, params) in enumerate(hidden_intermediates):
                    # Apply translator
                    translated_params = self.translators[layer_idx](features, params)
                    
                    # Apply heads to translated params
                    rating_logits = self.model.apply_head('rating', translated_params)
                    ranking_logits = self.model.apply_head('ranking', translated_params)
                    
                    predictions = {
                        'rating': rating_logits,
                        'ranking': ranking_logits
                    }
                    
                    # Compute loss for this layer
                    loss_dict = self._compute_loss(predictions, targets)
                    
                    # Accumulate losses
                    layer_losses[layer_idx] += loss_dict['total'].item()
                    layer_rating_losses[layer_idx] += loss_dict['rating'].item()
                    layer_ranking_losses[layer_idx] += loss_dict['ranking'].item()
            
            # Average losses across instances
            for layer_idx in range(self.num_layers):
                layer_losses[layer_idx] /= num_instances
                layer_rating_losses[layer_idx] /= num_instances
                layer_ranking_losses[layer_idx] /= num_instances
            
            return {
                'total': layer_losses,
                'rating': layer_rating_losses,
                'ranking': layer_ranking_losses
            }
    
    
    ###########################################################################
    # POST-TRAINING ANALYSIS METHODS (Override LogitLensAnalyzer)
    ###########################################################################
    
    def analyze_all_variables_across_layers(self, 
                                          all_variables: List[RankingData]) -> List[VariableAnalysis]:
        """Override base class to use trained translators instead of direct head application."""
        
        with torch.no_grad():
            # Run a single forward pass with intermediates captured
            logits_final, hidden_intermediates = self.model(all_variables, return_intermediate=True)

            layer_analyses: List[LayerAnalysis] = []
            for layer_idx, (features_snapshot, params_snapshot) in enumerate(hidden_intermediates):
                # Apply trained translator instead of direct head application
                translated_params = self.translators[layer_idx](features_snapshot, params_snapshot)
                
                # Compute head logits from the translated params
                rating_logits = self.model.apply_head('rating', translated_params)
                ranking_logits = self.model.apply_head('ranking', translated_params)

                layer_analyses.append(
                    LayerAnalysis(
                        layer_idx=layer_idx,
                        hidden_states=features_snapshot,
                        logits={'rating': rating_logits, 'ranking': ranking_logits},
                        metrics={}
                    )
                )
            
            # Create VariableAnalysis for each variable (reuse base class logic)
            variable_analyses = []
            for i, var in enumerate(all_variables):
                # Determine head type based on variable type
                head_type = 'ranking' if var.is_listwise else 'rating'
                
                # Extract metrics for this variable across all layers
                var_layer_analyses = []
                for layer_analysis in layer_analyses:
                    logits_full = layer_analysis.logits[head_type]
                    # Take per-variable slice to avoid storing logits for all variables
                    logits_slice = logits_full[0, i]
                    metrics = self._compute_single_variable_metrics(var, logits_slice, head_type)
                    
                    # Store only this variable's hidden state vector for the layer
                    if layer_analysis.hidden_states is not None:
                        # hidden_states shape expected [B, N, D]; take [0, i, :]
                        try:
                            hidden_slice = layer_analysis.hidden_states[0, i]
                        except Exception:
                            hidden_slice = layer_analysis.hidden_states
                    else:
                        hidden_slice = None
                    
                    var_layer_analysis = LayerAnalysis(
                        layer_idx=layer_analysis.layer_idx,
                        hidden_states=hidden_slice,
                        logits={head_type: logits_slice},
                        metrics=metrics
                    )
                    var_layer_analyses.append(var_layer_analysis)
                
                variable_analysis = VariableAnalysis(
                    variable=var,
                    layer_analyses=var_layer_analyses
                )
                variable_analyses.append(variable_analysis)
            
            return variable_analyses
    
    def analyze_all_layers(self, train_variables: List[RankingData],
                          test_variables: List[RankingData]) -> LogitLensResults:
        """Override base class to include translator training."""
        
        logger.info("Running tuned lens analysis...")
        
        # Combine all variables for translator training
        all_variables = train_variables + test_variables
        
        # Train translators - pass as list of instance lists
        training_history = self.train_translators([train_variables, test_variables])
        
        # Use base class method but with trained translators
        results = super().analyze_all_layers(train_variables, test_variables)
        
        # Add tuned lens specific config
        results.model_config['tuned_lens_config'] = {
            'learning_rate': self.config.learning_rate,
            'epochs': self.config.epochs,
            'target_type': self.config.target_type,
            'weight_decay': self.config.weight_decay,
            'train_split_ratio': self.config.train_split_ratio,
            'random_seed': self.config.random_seed
        }
        
        # Add training history to data config
        results.data_config['translator_training_history'] = training_history
        
        logger.info(f"Tuned lens analysis complete: {len(results.all_variables)} variables analyzed")
        
        return results
    
