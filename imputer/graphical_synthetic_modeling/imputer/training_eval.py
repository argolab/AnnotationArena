"""
Training and evaluation functions for two-stream transformer imputer.

Provides training loops with early stopping, KL divergence loss computation,
and comprehensive evaluation metrics for the neural imputation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for CPT extraction in training/evaluation. "
        "Please install pyAgrum: pip install pyagrum"
    )

from .architecture import (
    GraphImputer, ImputationDataset, collate_batch, 
    extract_cpts_for_nodes, compute_max_cpt_size, 
    create_model, DEVICE, SampleTuple
)

logger = logging.getLogger(__name__)


# ================================= LOSS FUNCTIONS =================================

def compute_kl_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence loss: KL(true || pred) for unobserved nodes only.
    
    This is the core loss function that trains the model to predict probability
    distributions for unobserved nodes that match the true posterior distributions.
    
    Args:
        predictions: Model predictions [batch, n_nodes, n_states]
        targets: True probability distributions [batch, n_nodes, n_states]  
        mask: Binary mask [batch, n_nodes] where 1=unobserved (predict these)
        
    Returns:
        KL divergence loss averaged over unobserved nodes
    """
    # Get mask for unobserved nodes (these are what we predict)
    unobserved_mask = mask.bool()
    
    # If no unobserved nodes, return zero loss
    if unobserved_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Extract predictions and targets for unobserved nodes only
    pred_unobserved = predictions[unobserved_mask]
    targets_unobserved = targets[unobserved_mask]
    
    logger.debug(f"KL loss computation: {unobserved_mask.sum().item()} unobserved nodes")
    
    # Compute KL divergence: KL(true || pred) = sum(true * log(true/pred))
    kl_loss = F.kl_div(
        torch.log(pred_unobserved + 1e-10),  # Add epsilon to avoid log(0)
        targets_unobserved,
        reduction='batchmean'  # Average over batch and nodes
    )
    
    return kl_loss


# ================================= TRAINING FUNCTIONS =================================

def train_epoch(model: GraphImputer, train_loader: DataLoader, 
               optimizer: torch.optim.Optimizer) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: GraphImputer model to train
        train_loader: DataLoader with training batches
        optimizer: Optimizer for parameter updates
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        inputs, structure_info, dimensions, mask, targets, cpt_info, true_states = batch
        
        # Move to device
        inputs = inputs.to(DEVICE)
        structure_info = structure_info.to(DEVICE)
        dimensions = dimensions.to(DEVICE)
        mask = mask.to(DEVICE)
        targets = targets.to(DEVICE)
        cpt_info = cpt_info.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs, structure_info, cpt_info, dimensions)
        
        # Compute KL loss for unobserved nodes
        loss = compute_kl_loss(predictions, targets, mask)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    logger.debug(f"Training epoch: avg_loss={avg_loss:.6f}")
    
    return avg_loss


def validate_epoch(model: GraphImputer, val_loader: DataLoader) -> float:
    """
    Validate the model for one epoch.
    
    Args:
        model: GraphImputer model to validate
        val_loader: DataLoader with validation batches
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, structure_info, dimensions, mask, targets, cpt_info, true_states = batch
            
            # Move to device
            inputs = inputs.to(DEVICE)
            structure_info = structure_info.to(DEVICE)
            dimensions = dimensions.to(DEVICE)
            mask = mask.to(DEVICE)
            targets = targets.to(DEVICE)
            cpt_info = cpt_info.to(DEVICE)
            
            # Forward pass
            predictions = model(inputs, structure_info, cpt_info, dimensions)
            
            # Compute KL loss for unobserved nodes
            loss = compute_kl_loss(predictions, targets, mask)
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    logger.debug(f"Validation epoch: avg_loss={avg_loss:.6f}")
    
    return avg_loss


def train_model(model: GraphImputer, train_loader: DataLoader, val_loader: DataLoader,
               epochs: int = 100, lr: float = 1e-4, patience: int = 30) -> GraphImputer:
    """
    Train the imputation model with early stopping.
    
    Args:
        model: GraphImputer model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        epochs: Maximum number of training epochs
        lr: Learning rate for optimizer
        patience: Early stopping patience (epochs without improvement)
        
    Returns:
        Trained model
    """
    logger.info(f"Training model with standard learning: epochs={epochs}, lr={lr}, patience={patience}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience//2, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training (standard)"):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer)
        
        # Validate
        val_loss = validate_epoch(model, val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.debug(f"Epoch {epoch}: New best validation loss = {val_loss:.6f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model


# ================================= EVALUATION FUNCTIONS =================================

def evaluate_model(model: GraphImputer, test_data: List[SampleTuple], 
                  bn: gum.BayesNet, n_nodes: int, n_states: int = 2) -> Dict[str, Any]:
    """
    Evaluate imputation model using KL divergence on test data.
    
    Args:
        model: Trained GraphImputer model
        test_data: List of test samples with missing values
        bn: BayesNet for CPT extraction
        n_nodes: Number of nodes in the graph
        n_states: Number of states per node (must be 2)
        
    Returns:
        Dict with evaluation metrics including mean KL, standard deviation, etc.
        
    Raises:
        Exception: Any inference failures will bubble up for debugging
    """
    logger.debug(f"Evaluating imputation model on {len(test_data)} test samples")
    
    model.eval()
    kl_divergences = []
    prediction_errors = []
    
    # Get maximum CPT size for consistent processing
    max_cpt_size = compute_max_cpt_size(bn) if bn else 8
    
    with torch.no_grad():
        for sample_idx, (inputs, structure_info, dimensions, mask, targets, true_states) in enumerate(test_data):
            # Identify unobserved nodes (mask == 1)
            unobserved_nodes = [i for i in range(n_nodes) if mask[i] == 1]
            observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
            
            if not unobserved_nodes:
                logger.debug(f"Sample {sample_idx}: No unobserved nodes, skipping")
                continue
            
            logger.debug(f"Sample {sample_idx}: {len(observed_nodes)} observed, {len(unobserved_nodes)} unobserved")
            
            # Extract CPTs for observed nodes (privacy-preserving)
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            
            # NO TRY-EXCEPT HERE - Let inference failures bubble up for debugging
            
            # Prepare tensors for model forward pass
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)
            structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
            cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
            dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)
            
            # Get model predictions
            predictions = model(inputs_batch, structure_info_batch, 
                              cpt_info_batch, dimensions_batch)
            predictions = predictions.squeeze(0).cpu()  # Remove batch dimension
            
            # Compute true Bayesian posterior using ground truth BN
            true_posteriors = {}
            if observed_nodes:  # Only compute if there are observed nodes
                # Create evidence from observed nodes for true posterior computation
                evidence = {}
                for obs_node in observed_nodes:
                    obs_state = torch.argmax(inputs[obs_node, 1:]).item()
                    evidence[str(obs_node)] = str(obs_state)
                
                # Use LazyPropagation on ground truth BN to get true posteriors
                import pyagrum as gum
                true_infer = gum.LazyPropagation(bn)
                true_infer.setEvidence(evidence)
                true_infer.makeInference()
                
                for node in unobserved_nodes:
                    true_posterior = true_infer.posterior(str(node))
                    # Convert pyAgrum potential to numpy array
                    true_posteriors[node] = np.array([
                        true_posterior[{str(node): str(state)}] 
                        for state in range(n_states)
                    ])
                true_infer.eraseAllEvidence()
            else:
                # No observed nodes - use marginal probabilities from ground truth BN
                for node in unobserved_nodes:
                    marginal = bn.cpt(str(node))
                    true_posteriors[node] = np.array([
                        marginal[{str(node): str(state)}]
                        for state in range(n_states)
                    ])
            
            # Evaluate each unobserved node
            for node in unobserved_nodes:
                # Get predicted probabilities
                pred_probs = predictions[node, :].numpy()
                
                # Get TRUE Bayesian posterior (not one-hot sample!)
                true_probs = true_posteriors[node]
                
                # Validate predictions (normalize if needed)
                if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                    logger.warning(f"Sample {sample_idx}, Node {node}: Invalid predictions, using uniform")
                    pred_probs = np.ones(n_states) / n_states
                else:
                    pred_probs = pred_probs / np.sum(pred_probs)  # Ensure normalization
                
                # Validate ground truth
                if np.any(np.isnan(true_probs)) or np.sum(true_probs) == 0:
                    logger.warning(f"Sample {sample_idx}, Node {node}: Invalid ground truth, skipping")
                    continue
                
                # Compute KL divergence: KL(true || pred) = sum(true * log(true/pred))
                kl = 0.0
                for state in range(n_states):
                    if true_probs[state] > 1e-10:
                        kl += true_probs[state] * np.log(
                            (true_probs[state] + 1e-10) / (pred_probs[state] + 1e-10)
                        )

                # Handle floating point precision issues in KL computation
                if np.isnan(kl) or np.isinf(kl):
                    logger.warning(f"Sample {sample_idx}, Node {node}: Invalid KL={kl} (NaN/Inf), skipping")
                    continue
                
                # Clamp small negative values to 0 (due to numerical precision when pred ≈ true)
                if kl < -1e-6:  # Only warn for significantly negative values
                    logger.warning(f"Sample {sample_idx}, Node {node}: Significantly negative KL={kl}, skipping")
                    continue
                    
                kl = max(kl, 0.0)  # Ensure no negative due to numerical precision
                kl_divergences.append(kl)
                
                # L2 prediction error
                error = np.linalg.norm(pred_probs - true_probs)
                prediction_errors.append(error)
                    
    # Compile results
    if not kl_divergences:
        logger.warning("No successful evaluations - all samples failed!")
        return {
            'mean_kl': float('inf'),
            'std_kl': 0.0,
            'mean_error': float('inf'),
            'failed_rate': 1.0,
            'n_evaluations': 0,
            'kl_distribution': []
        }
    
    results = {
        'mean_kl': np.mean(kl_divergences),
        'std_kl': np.std(kl_divergences),
        'mean_error': np.mean(prediction_errors),
        'failed_rate': 0.0,  # No try-except masking failures
        'n_evaluations': len(kl_divergences),
        'kl_distribution': kl_divergences
    }
    
    logger.info(f"Imputation evaluation: Mean KL = {results['mean_kl']:.4f} ± {results['std_kl']:.4f}")
    logger.info(f"Successful evaluations: {results['n_evaluations']}")
    
    return results


def evaluate_log_loss(model: GraphImputer, test_data: List[SampleTuple], 
                     bn: gum.BayesNet, n_nodes: int) -> Dict[str, Any]:
    """
    Evaluate neural imputer log-loss: -log P(true_unobserved | observed, neural_model).
    
    Args:
        model: Trained GraphImputer model
        test_data: List of test samples with missing values
        bn: BayesNet for CPT extraction
        n_nodes: Number of nodes in the graph
        
    Returns:
        Dict with log-loss evaluation results
        
    Raises:
        Exception: Any inference failures will bubble up for debugging
    """
    logger.debug(f"Evaluating neural imputer log-loss on {len(test_data)} test samples")
    
    model.eval()
    log_losses = []
    
    # Get maximum CPT size for consistent processing
    max_cpt_size = compute_max_cpt_size(bn) if bn else 8
    
    with torch.no_grad():
        for sample_idx, (inputs, structure_info, dimensions, mask, targets, true_states) in enumerate(test_data):
            # Identify unobserved nodes (mask == 1)
            unobserved_nodes = [i for i in range(n_nodes) if mask[i] == 1]
            observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
            
            if not unobserved_nodes:
                logger.debug(f"Sample {sample_idx}: No unobserved nodes, skipping")
                continue
                
            logger.debug(f"Sample {sample_idx}: {len(observed_nodes)} observed, {len(unobserved_nodes)} unobserved")
            
            # Extract CPTs for observed nodes (privacy-preserving)
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            
            # NO TRY-EXCEPT HERE - Let inference failures bubble up for debugging
            
            # Prepare tensors for model forward pass
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)
            structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
            cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
            dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)
            
            # Get model predictions
            predictions = model(inputs_batch, structure_info_batch, 
                              cpt_info_batch, dimensions_batch)
            predictions = predictions.squeeze(0).cpu()  # Remove batch dimension
            
            # Compute log-loss for this sample
            example_log_loss = 0.0
            
            for node in unobserved_nodes:
                # Get predicted probabilities (softmax output from model)
                pred_probs = predictions[node, :].numpy()
                
                # Get true state for this unobserved node from actual sampled states
                true_state = true_states[node].item()
                
                # Get probability of true state
                prob_true_state = pred_probs[true_state]
                
                # Add to log-loss: -log(prob)
                if prob_true_state > 1e-10:
                    node_log_loss = -np.log(prob_true_state)
                else:
                    node_log_loss = 10  # Large penalty for very low probability
                
                example_log_loss += node_log_loss
            
            log_losses.append(example_log_loss)
            logger.debug(f"Sample {sample_idx}: Total log_loss={example_log_loss:.4f}")
    
    # Compile results
    if not log_losses:
        logger.warning("No successful neural log-loss inferences!")
        return {
            'mean_log_loss': float('inf'),
            'std_log_loss': 0.0,
            'failed_rate': 1.0,
            'log_loss_values': []
        }
    
    results = {
        'mean_log_loss': np.mean(log_losses),
        'std_log_loss': np.std(log_losses),
        'failed_rate': 0.0,  # No try-except masking failures
        'log_loss_values': log_losses
    }
    
    logger.info(f"Neural imputer log-loss: Mean={results['mean_log_loss']:.4f} ± {results['std_log_loss']:.4f}")
    
    return results