"""
Training and evaluation functions for tuned lens probes.

Provides training loops for individual probe calibration and comprehensive
evaluation with layer-wise analysis using the tuned transformations.

Reuses existing loss functions and evaluation logic from training_eval.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import gc
from tqdm import tqdm
from typing import List, Dict, Any

try:
    import pyagrum as gum
    PYAGRUM_AVAILABLE = True
except ImportError:
    PYAGRUM_AVAILABLE = False
    raise ImportError(
        "pyAgrum is required for tuned lens training/evaluation. "
        "Please install pyAgrum: pip install pyagrum"
    )

from .architecture import DEVICE, SampleTuple, compute_max_cpt_size, extract_cpts_for_nodes
from .training_eval import compute_kl_loss  # REUSE existing loss function
from .tuned_lens_probes import TunedLensGraphImputer

logger = logging.getLogger(__name__)


# ================================= TRAINING FUNCTIONS =================================

def train_single_probe(
    tuned_lens_model: TunedLensGraphImputer,
    layer_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 15
) -> None:
    """
    Train a single tuned lens probe for a specific layer.

    This function trains ONLY the probe at layer_idx while keeping the base
    model and all other probes frozen.

    Args:
        tuned_lens_model: TunedLensGraphImputer with probes
        layer_idx: Which layer to train (0=initial, ..., n_layers+1=final_norm)
        train_loader: Training data loader (calibration set)
        val_loader: Validation data loader
        epochs: Maximum training epochs
        lr: Learning rate for probe training
        patience: Early stopping patience

    Note:
        Modifies tuned_lens_model.probes[layer_idx] in place.
    """
    logger.info(f"Training tuned lens probe for layer {layer_idx}: "
               f"epochs={epochs}, lr={lr}, patience={patience}")

    # Freeze all probes except the one we're training
    for i, probe in enumerate(tuned_lens_model.probes):
        for param in probe.parameters():
            param.requires_grad = (i == layer_idx)

    # Setup optimizer (only for this probe's parameters)
    probe = tuned_lens_model.probes[layer_idx]
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience//2, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop (same structure as train_model in training_eval.py)
    for epoch in tqdm(range(epochs), desc=f"Training Layer {layer_idx} Probe"):
        # Train for one epoch
        probe.train()
        total_train_loss = 0.0
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

            # Forward pass through this layer with probe
            optimizer.zero_grad()
            predictions = tuned_lens_model.forward(
                inputs, structure_info, cpt_info, dimensions, layer_idx
            )

            # Compute KL loss (REUSE existing function from training_eval.py)
            loss = compute_kl_loss(predictions, targets, mask)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches if n_batches > 0 else 0.0

        # Validate
        probe.eval()
        total_val_loss = 0.0
        n_val_batches = 0

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
                predictions = tuned_lens_model.forward(
                    inputs, structure_info, cpt_info, dimensions, layer_idx
                )

                # Compute KL loss (REUSE existing function)
                loss = compute_kl_loss(predictions, targets, mask)

                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches if n_val_batches > 0 else 0.0

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.debug(f"Layer {layer_idx}, Epoch {epoch}: New best val loss = {avg_val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Layer {layer_idx} early stopping at epoch {epoch}")
            break

    logger.info(f"Layer {layer_idx} probe training completed. Best val loss: {best_val_loss:.4f}")


def train_all_probes(
    tuned_lens_model: TunedLensGraphImputer,
    calibration_loader: DataLoader,
    val_loader: DataLoader,
    epochs_per_probe: int = 50,
    lr: float = 1e-3,
    patience: int = 15
) -> TunedLensGraphImputer:
    """
    Train all tuned lens probes independently.

    Each probe is trained separately to align its layer's representation with
    the output heads for better prediction accuracy.

    Args:
        tuned_lens_model: TunedLensGraphImputer with initialized probes
        calibration_loader: Calibration dataset for probe training
        val_loader: Validation dataset
        epochs_per_probe: Epochs to train each probe
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        TunedLensGraphImputer with trained probes
    """
    logger.info(f"Training all {len(tuned_lens_model.probes)} tuned lens probes independently")

    for layer_idx in range(len(tuned_lens_model.probes)):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training probe {layer_idx + 1}/{len(tuned_lens_model.probes)}")
        logger.info(f"{'=' * 60}")

        train_single_probe(
            tuned_lens_model=tuned_lens_model,
            layer_idx=layer_idx,
            train_loader=calibration_loader,
            val_loader=val_loader,
            epochs=epochs_per_probe,
            lr=lr,
            patience=patience
        )

    logger.info("All probes trained successfully")
    return tuned_lens_model


# ================================= EVALUATION FUNCTIONS =================================

def evaluate_tuned_lens_model(
    tuned_lens_model: TunedLensGraphImputer,
    test_data: List[SampleTuple],
    bn: gum.BayesNet,
    n_nodes: int,
    n_states: int = 2
) -> Dict[str, Any]:
    """
    Evaluate tuned lens model with layer-wise analysis.

    This function mirrors evaluate_model_with_layer_analysis() from training_eval.py
    but uses tuned lens probes instead of direct output heads.

    Args:
        tuned_lens_model: TunedLensGraphImputer with trained probes
        test_data: List of test samples
        bn: BayesNet for true posterior computation
        n_nodes: Number of nodes
        n_states: Number of states per node

    Returns:
        Dictionary with same structure as evaluate_model_with_layer_analysis:
            'mean_kl': Final layer mean KL
            'std_kl': Final layer std KL
            'kl_distribution': Final layer KL values
            'n_evaluations': Number of evaluations
            'failed_rate': Failure rate
            'n_layers': Number of transformer layers
            'layer_kl_raw_data': Sparse dict {layer_idx: {sample_idx: {node_idx: kl}}}
            'layer_kl_means': Array of mean KL per layer
            'layer_kl_stds': Array of std KL per layer
            'layer_descriptions': Layer names
            'sample_metadata': Per-sample metadata
    """
    logger.debug(f"Evaluating tuned lens model on {len(test_data)} test samples")

    tuned_lens_model.eval()
    n_layers = tuned_lens_model.n_layers
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

            # Prepare inputs (same as evaluate_model_with_layer_analysis)
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)
            structure_info_batch = structure_info.unsqueeze(0).to(DEVICE)
            cpt_info_batch = torch.FloatTensor(cpt_info).unsqueeze(0).to(DEVICE)
            dimensions_batch = dimensions.unsqueeze(0).to(DEVICE)

            # Compute true posteriors (same logic as evaluate_model_with_layer_analysis:713-740)
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

            # For each layer, get tuned lens predictions and compute KL
            for layer_idx in range(n_layers + 2):
                # Get predictions from this layer WITH tuned probe
                layer_predictions = tuned_lens_model.forward(
                    inputs_batch, structure_info_batch,
                    cpt_info_batch, dimensions_batch,
                    layer_idx
                ).squeeze(0).cpu()

                # Initialize storage for this layer/sample
                layer_kl_data[layer_idx][sample_idx] = {}

                # Compute KL for each unobserved node (same as evaluate_model_with_layer_analysis:752-781)
                for node_idx in unobserved_nodes:
                    pred_probs = layer_predictions[node_idx, :].numpy()
                    true_probs = true_posteriors[node_idx]

                    # Normalize predictions
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

                    # Handle numerical precision
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

    # Aggregate results (REUSE existing aggregation function)
    from .training_eval import _aggregate_layer_kl_data
    results = _aggregate_layer_kl_data(layer_kl_data, sample_metadata, n_layers)

    logger.info(f"Tuned lens evaluation completed: {results['n_evaluations']} total evaluations")
    logger.info(f"Final layer KL: {results['mean_kl']:.4f} ± {results['std_kl']:.4f}")

    # Force garbage collection
    gc.collect()

    return results
