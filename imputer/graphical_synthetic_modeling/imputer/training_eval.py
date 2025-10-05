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
import gc
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Clear CUDA cache periodically for memory efficiency
        # if torch.cuda.is_available() and batch_idx % 10 == 0:
        #     torch.cuda.empty_cache()
        
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


def train_epoch_with_deep_supervision(
    model: GraphImputer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    final_weight: float = 3.0,
    layer_weight: float = 1.0
) -> float:
    """
    Train one epoch with deep supervision on intermediate layers.

    Supervises layer_0, layer_1, layer_2, layer_3 (NOT initial or final_norm).
    Uses shared output heads with final normalization applied to intermediate layers.

    Args:
        model: GraphImputer model to train
        train_loader: DataLoader with training batches
        optimizer: Optimizer for parameter updates
        final_weight: Weight for final layer loss (default: 3.0)
        layer_weight: Weight for intermediate layer losses (default: 1.0)

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_layers = len(model.transformer.layers)

    for batch in train_loader:
        inputs, structure_info, dimensions, mask, targets, cpt_info, true_states = batch

        # Move to device
        inputs = inputs.to(DEVICE)
        structure_info = structure_info.to(DEVICE)
        dimensions = dimensions.to(DEVICE)
        mask = mask.to(DEVICE)
        targets = targets.to(DEVICE)
        cpt_info = cpt_info.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass with layer capture
        final_predictions, layer_streams = forward_with_layer_capture(
            model, inputs, structure_info, cpt_info, dimensions
        )

        # Compute final layer loss
        final_loss = compute_kl_loss(final_predictions, targets, mask)

        # Compute intermediate layer losses
        # Supervise layers 1-4 (layer_0, layer_1, layer_2, layer_3)
        # Skip index 0 (initial) and index 5 (final_norm)
        layer_losses = []
        for layer_idx in range(1, n_layers + 1):  # Indices 1, 2, 3, 4 = transformer layers
            layer_stream = layer_streams[layer_idx]
            layer_predictions = get_predictions_from_layer_stream(model, layer_stream)
            layer_loss = compute_kl_loss(layer_predictions, targets, mask)
            layer_losses.append(layer_loss)

        # Weighted combination (normalized)
        mean_layer_loss = sum(layer_losses) / len(layer_losses)
        total_loss_value = (final_weight * final_loss + layer_weight * mean_layer_loss) / (final_weight + layer_weight)

        # Backward pass
        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_loss_value.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    logger.debug(f"Training epoch (deep supervision): avg_loss={avg_loss:.6f}")

    return avg_loss


def train_model_with_deep_supervision(
    model: GraphImputer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    patience: int = 30,
    final_weight: float = 3.0,
    layer_weight: float = 1.0
) -> GraphImputer:
    """
    Train the imputation model with deep supervision and early stopping.

    Deep supervision applies loss to intermediate transformer layers (layer_0 through layer_3),
    not to the initial embedding or final normalization.

    Args:
        model: GraphImputer model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of training epochs
        lr: Learning rate for optimizer
        patience: Early stopping patience (epochs without improvement)
        final_weight: Weight for final layer loss (default: 3.0)
        layer_weight: Weight for intermediate layer losses (default: 1.0)

    Returns:
        Trained model
    """
    logger.info(f"Training model with DEEP SUPERVISION: epochs={epochs}, lr={lr}, patience={patience}")
    logger.info(f"Deep supervision weights: final={final_weight}, layer={layer_weight}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience//2, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training (deep supervision)"):
        # Train for one epoch with deep supervision
        train_loss = train_epoch_with_deep_supervision(
            model, train_loader, optimizer, final_weight, layer_weight
        )

        # Validate (standard validation, no deep supervision)
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

    logger.info(f"Deep supervision training completed. Best validation loss: {best_val_loss:.4f}")
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
    
    # Create single inference engine for reuse throughout evaluation
    true_infer = None
    if bn:
        import pyagrum as gum
        true_infer = gum.LazyPropagation(bn)
    
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
            if observed_nodes and true_infer:  # Only compute if there are observed nodes and inference engine
                # Create evidence from observed nodes for true posterior computation
                evidence = {}
                for obs_node in observed_nodes:
                    obs_state = torch.argmax(inputs[obs_node, 1:]).item()
                    evidence[str(obs_node)] = str(obs_state)
                
                # Reuse the single inference engine
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
    
    # Clean up inference engine explicitly
    if true_infer:
        try:
            true_infer.eraseAllEvidence()
            del true_infer
        except:
            pass  # Ignore cleanup errors
    
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
    
    # Force garbage collection to clean up any residual PyAgrum objects
    gc.collect()
    
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
    
    # Force garbage collection to clean up any residual PyAgrum objects
    gc.collect()
    
    return results


def evaluate_cross_entropy(model: torch.nn.Module, test_data: List[SampleTuple],
                          bn: gum.BayesNet, n_nodes: int, **kwargs) -> Dict[str, Any]:
    """
    Evaluate neural model using cross-entropy H(p_true, q_neural).
    
    Args:
        model: Trained neural imputation model
        test_data: List of test samples with missing values
        bn: Ground truth BayesNet for computing true posteriors
        n_nodes: Number of nodes
        **kwargs: Additional evaluation parameters
        
    Returns:
        dict: Cross-entropy evaluation results
    """
    from imputer.architecture import compute_max_cpt_size
    from imputer.cross_entropy_eval import evaluate_neural_cross_entropy
    
    # Compute max CPT size for this BN
    max_cpt_size = compute_max_cpt_size(bn) if bn else 8
    
    return evaluate_neural_cross_entropy(model, test_data, bn, n_nodes, max_cpt_size)


# ================================= MECHANISTIC INTERPRETABILITY FUNCTIONS =================================

def forward_with_layer_capture(
    model,  # GraphImputer
    inputs: torch.Tensor,
    structure_info: torch.Tensor,
    cpt_info: torch.Tensor,
    dimensions: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass that captures parameter stream after each transformer layer.

    CRITICAL: This replicates the EXACT forward logic from TwoStreamTransformer.forward()
    and GraphImputer.forward() but stores intermediate parameter streams.

    Args:
        model: GraphImputer model
        inputs: [batch, n_nodes, input_dim] - observed states with mask
        structure_info: [batch, n_nodes, structure_dim] - adjacency matrix
        cpt_info: [batch, n_nodes, cpt_dim] - CPTs with privacy masking
        dimensions: [batch, n_nodes] - node indices

    Returns:
        final_predictions: [batch, n_nodes, n_states] - Final model predictions
        layer_parameter_streams: List of [batch, n_nodes, cpt_dim] tensors:
            - [0] = after pos_encoder (initial state, before any transformer layer)
            - [1] = after transformer layer 0
            - [2] = after transformer layer 1
            - ...
            - [n_layers] = after transformer layer (n_layers-1)
            - [n_layers+1] = after final_norm_parameter (what goes to output heads)
    """
    transformer = model.transformer

    # Step 1: Initialize both streams (EXACT copy from architecture.py:347-349)
    embedding_stream, parameter_stream = transformer.pos_encoder(
        inputs, structure_info, cpt_info, dimensions
    )

    # Store initial state (layer 0 - after positional encoder)
    layer_streams = [parameter_stream.clone()]

    # Step 2: Extract mask (EXACT copy from architecture.py:352)
    mask = inputs[:, :, 0].bool()

    # Step 3: Process through transformer layers (EXACT copy from architecture.py:355-358)
    for layer in transformer.layers:
        embedding_stream, parameter_stream = layer(
            embedding_stream, parameter_stream, mask
        )
        layer_streams.append(parameter_stream.clone())

    # Step 4: Apply final normalization (EXACT copy from architecture.py:361-362)
    embedding_stream = transformer.final_norm_embedding(embedding_stream)
    parameter_stream_normalized = transformer.final_norm_parameter(parameter_stream)
    layer_streams.append(parameter_stream_normalized.clone())

    # Step 5: Convert to predictions (EXACT copy from architecture.py:437-444)
    predictions = []
    for i in range(model.n_nodes):
        node_probs = model.output_heads[i](parameter_stream_normalized[:, i, :])
        predictions.append(node_probs)
    predictions = torch.stack(predictions, dim=1)

    return predictions, layer_streams


def get_predictions_from_layer_stream(
    model,  # GraphImputer
    layer_parameter_stream: torch.Tensor
) -> torch.Tensor:
    """
    Convert layer parameter stream to state probability predictions using output heads.

    CRITICAL: Applies final_norm_parameter BEFORE output heads, since output heads
    were trained on normalized parameter streams. This ensures intermediate layers
    are processed the same way as the final layer.

    Args:
        model: GraphImputer model
        layer_parameter_stream: [batch, n_nodes, cpt_dim] - RAW layer output

    Returns:
        predictions: [batch, n_nodes, n_states]
    """
    # Apply final normalization (output heads expect normalized inputs)
    normalized_stream = model.transformer.final_norm_parameter(layer_parameter_stream)

    # Apply output heads to normalized stream
    predictions = []
    for i in range(model.n_nodes):
        node_probs = model.output_heads[i](normalized_stream[:, i, :])
        predictions.append(node_probs)
    return torch.stack(predictions, dim=1)


def evaluate_model_with_layer_analysis(
    model,  # GraphImputer
    test_data: List[SampleTuple],
    bn: gum.BayesNet,
    n_nodes: int,
    n_states: int = 2
) -> Dict[str, Any]:
    """
    Evaluate model with layer-by-layer KL divergence analysis for mechanistic interpretability.

    This function performs standard evaluation AND captures intermediate layer representations
    to analyze how predictions are progressively refined through the transformer.

    Args:
        model: Trained GraphImputer model
        test_data: List of test samples with missing values
        bn: BayesNet for CPT extraction and true posterior computation
        n_nodes: Number of nodes in the graph
        n_states: Number of states per node (must be 2)

    Returns:
        Dictionary with:
            # Standard evaluation metrics (same as evaluate_model)
            'mean_kl': float - Final layer mean KL divergence
            'std_kl': float - Final layer std KL divergence
            'kl_distribution': List[float] - Final layer KL values per node
            'n_evaluations': int - Number of successful evaluations
            'failed_rate': float - Failure rate

            # Layer-wise analysis (NEW)
            'n_layers': int - Number of transformer layers
            'layer_kl_raw_data': Dict - Sparse storage: {layer_idx: {sample_idx: {node_idx: kl}}}
            'layer_kl_means': np.ndarray - [n_layers+2] mean KL per layer
            'layer_kl_stds': np.ndarray - [n_layers+2] std KL per layer
            'layer_descriptions': List[str] - ['initial', 'layer_0', ..., 'final_norm']
            'sample_metadata': List[Dict] - Per-sample observed/unobserved node info
    """
    logger.debug(f"Evaluating model with layer-wise analysis on {len(test_data)} test samples")

    model.eval()
    n_layers = len(model.transformer.layers)
    max_cpt_size = compute_max_cpt_size(bn) if bn else 8

    # Storage: {layer_idx: {sample_idx: {node_idx: kl_value}}}
    layer_kl_data = {layer_idx: {} for layer_idx in range(n_layers + 2)}
    sample_metadata = []

    # Create single inference engine for reuse
    true_infer = None
    if bn:
        true_infer = gum.LazyPropagation(bn)

    with torch.no_grad():
        for sample_idx, (inputs, structure_info, dimensions, mask, targets, true_states) in enumerate(test_data):
            unobserved_nodes = [i for i in range(n_nodes) if mask[i] == 1]
            observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]

            if not unobserved_nodes:
                logger.debug(f"Sample {sample_idx}: No unobserved nodes, skipping")
                continue

            # Store sample metadata
            sample_metadata.append({
                'sample_idx': sample_idx,
                'unobserved_nodes': unobserved_nodes,
                'observed_nodes': observed_nodes,
                'n_unobserved': len(unobserved_nodes),
                'n_observed': len(observed_nodes)
            })

            logger.debug(f"Sample {sample_idx}: {len(observed_nodes)} observed, {len(unobserved_nodes)} unobserved")

            # Prepare inputs (SAME as evaluate_model)
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)
            structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
            cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
            dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)

            # Get layer-wise parameter streams
            final_predictions, layer_streams = forward_with_layer_capture(
                model, inputs_batch, structure_info_batch,
                cpt_info_batch, dimensions_batch
            )

            # Compute true posteriors (SAME as evaluate_model:298-325)
            true_posteriors = {}
            if observed_nodes and true_infer:
                # Create evidence from observed nodes
                evidence = {}
                for obs_node in observed_nodes:
                    obs_state = torch.argmax(inputs[obs_node, 1:]).item()
                    evidence[str(obs_node)] = str(obs_state)

                # Compute posteriors
                true_infer.setEvidence(evidence)
                true_infer.makeInference()

                for node in unobserved_nodes:
                    posterior = true_infer.posterior(str(node))
                    true_posteriors[node] = np.array([
                        posterior[{str(node): str(state)}]
                        for state in range(n_states)
                    ])

                true_infer.eraseAllEvidence()
            else:
                # No observed nodes - use marginal probabilities
                for node in unobserved_nodes:
                    marginal = bn.cpt(str(node))
                    true_posteriors[node] = np.array([
                        marginal[{str(node): str(state)}]
                        for state in range(n_states)
                    ])

            # For each layer, compute KL divergences
            for layer_idx, layer_stream in enumerate(layer_streams):
                # Get predictions from this layer's parameter stream
                layer_predictions = get_predictions_from_layer_stream(
                    model, layer_stream
                ).squeeze(0).cpu()

                # Initialize storage for this layer/sample
                layer_kl_data[layer_idx][sample_idx] = {}

                # Compute KL for each unobserved node
                for node_idx in unobserved_nodes:
                    pred_probs = layer_predictions[node_idx, :].numpy()
                    true_probs = true_posteriors[node_idx]

                    # Normalize predictions (same as evaluate_model)
                    if np.any(np.isnan(pred_probs)) or np.sum(pred_probs) == 0:
                        pred_probs = np.ones(n_states) / n_states
                    else:
                        pred_probs = pred_probs / np.sum(pred_probs)

                    # Compute KL divergence: KL(true || pred)
                    kl = 0.0
                    for state in range(n_states):
                        if true_probs[state] > 1e-10:
                            kl += true_probs[state] * np.log(
                                (true_probs[state] + 1e-10) / (pred_probs[state] + 1e-10)
                            )

                    # Handle numerical precision (same as evaluate_model)
                    if np.isnan(kl) or np.isinf(kl):
                        logger.warning(f"Sample {sample_idx}, Layer {layer_idx}, Node {node_idx}: Invalid KL={kl}")
                        continue

                    if kl < -1e-6:
                        logger.warning(f"Sample {sample_idx}, Layer {layer_idx}, Node {node_idx}: Negative KL={kl}")
                        continue

                    kl = max(kl, 0.0)  # Clamp small negative values
                    layer_kl_data[layer_idx][sample_idx][node_idx] = kl

    # Clean up inference engine
    if true_infer:
        try:
            true_infer.eraseAllEvidence()
            del true_infer
        except:
            pass

    # Aggregate results
    results = _aggregate_layer_kl_data(layer_kl_data, sample_metadata, n_layers)

    logger.info(f"Layer-wise evaluation completed: {results['n_evaluations']} total evaluations, {n_layers} layers")
    logger.info(f"Final layer KL: {results['mean_kl']:.4f} ± {results['std_kl']:.4f}")

    return results


def _aggregate_layer_kl_data(
    layer_kl_data: Dict[int, Dict[int, Dict[int, float]]],
    sample_metadata: List[Dict],
    n_layers: int
) -> Dict[str, Any]:
    """
    Aggregate sparse layer KL data into arrays and statistics.

    Args:
        layer_kl_data: Sparse dict {layer_idx: {sample_idx: {node_idx: kl}}}
        sample_metadata: Per-sample metadata
        n_layers: Number of transformer layers

    Returns:
        Dictionary with aggregated statistics and layer-wise analysis
    """
    # Convert sparse dict to statistics
    all_kl_values = []  # For final layer only (standard evaluation)
    layer_kl_means = []
    layer_kl_stds = []
    layer_kl_counts = []

    for layer_idx in range(n_layers + 2):
        # Collect all KL values for this layer
        layer_kls = []
        for sample_idx, node_kls in layer_kl_data[layer_idx].items():
            layer_kls.extend(node_kls.values())

        if layer_kls:
            layer_kl_means.append(np.mean(layer_kls))
            layer_kl_stds.append(np.std(layer_kls))
            layer_kl_counts.append(len(layer_kls))

            # For final layer (after final_norm), store distribution
            if layer_idx == n_layers + 1:
                all_kl_values = layer_kls
        else:
            layer_kl_means.append(float('inf'))
            layer_kl_stds.append(0.0)
            layer_kl_counts.append(0)

    # Create layer descriptions
    layer_descriptions = ['initial'] + [f'layer_{i}' for i in range(n_layers)] + ['final_norm']

    logger.debug(f"Layer KL aggregation: {layer_kl_counts} evaluations per layer")

    return {
        # Standard evaluation metrics (compatible with existing code)
        'mean_kl': layer_kl_means[-1] if layer_kl_means else float('inf'),
        'std_kl': layer_kl_stds[-1] if layer_kl_stds else 0.0,
        'kl_distribution': all_kl_values,
        'n_evaluations': len(all_kl_values),
        'failed_rate': 0.0,

        # Layer-wise analysis (NEW)
        'n_layers': n_layers,
        'layer_kl_raw_data': layer_kl_data,  # Sparse dict for detailed analysis
        'layer_kl_means': np.array(layer_kl_means),
        'layer_kl_stds': np.array(layer_kl_stds),
        'layer_kl_counts': np.array(layer_kl_counts),
        'layer_descriptions': layer_descriptions,
        'sample_metadata': sample_metadata
    }