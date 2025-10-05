"""
Tuned Lens Probes for MARFORMER mechanistic interpretability.

Implements learned affine transformations (W*x + b) for each layer to probe
intermediate representations, similar to nostalgebraist's tuned lens approach.

This file extends the logit lens approach with trainable probes while
keeping the base MARFORMER architecture frozen.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import logging

from .architecture import GraphImputer, DEVICE

logger = logging.getLogger(__name__)


class TunedLensProbe(nn.Module):
    """
    Learned affine transformation probe for a specific transformer layer.

    Transforms parameter stream from layer L to better align with output heads:
    probe(x) = W * x + b

    Initialized to identity transformation to start from logit lens baseline.
    """

    def __init__(self, cpt_dim: int):
        """
        Initialize tuned lens probe with identity initialization.

        Args:
            cpt_dim: Dimension of parameter stream (CPT dimension)
        """
        super().__init__()
        self.cpt_dim = cpt_dim

        # Affine transformation
        self.linear = nn.Linear(cpt_dim, cpt_dim)

        # Identity initialization: start from logit lens
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(cpt_dim))
            self.linear.bias.zero_()

        logger.debug(f"TunedLensProbe initialized: cpt_dim={cpt_dim}, identity init")

    def forward(self, layer_stream: torch.Tensor) -> torch.Tensor:
        """
        Apply learned affine transformation.

        Args:
            layer_stream: [batch, n_nodes, cpt_dim] - parameter stream from layer L

        Returns:
            transformed: [batch, n_nodes, cpt_dim] - aligned representation
        """
        return self.linear(layer_stream)


class TunedLensGraphImputer(nn.Module):
    """
    Wrapper that adds tuned lens probes to a frozen MARFORMER model.

    The base GraphImputer is frozen, and only the layer-specific probes are trained.
    This allows us to learn optimal transformations for each layer's representation
    without modifying the base model.
    """

    def __init__(self, base_model: GraphImputer, n_layers: int):
        """
        Initialize tuned lens wrapper with frozen base model.

        Args:
            base_model: Trained GraphImputer to analyze
            n_layers: Number of transformer layers (for probe count)
        """
        super().__init__()

        # Freeze base model
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.n_nodes = base_model.n_nodes
        self.n_states = base_model.n_states
        self.cpt_dim = base_model.cpt_dim
        self.n_layers = n_layers

        # Create one probe per layer (n_layers + 2 total: initial + n_layers + final_norm)
        self.probes = nn.ModuleList([
            TunedLensProbe(self.cpt_dim)
            for _ in range(n_layers + 2)
        ])

        logger.info(f"TunedLensGraphImputer initialized: {n_layers + 2} probes, "
                   f"base model frozen, cpt_dim={self.cpt_dim}")

    def forward_with_layer_capture(
        self,
        inputs: torch.Tensor,
        structure_info: torch.Tensor,
        cpt_info: torch.Tensor,
        dimensions: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass capturing layer parameter streams (REUSES existing function).

        This delegates to the existing forward_with_layer_capture() function
        from training_eval.py to avoid code duplication.

        Args:
            inputs: [batch, n_nodes, input_dim]
            structure_info: [batch, n_nodes, structure_dim]
            cpt_info: [batch, n_nodes, cpt_dim]
            dimensions: [batch, n_nodes]

        Returns:
            final_predictions: [batch, n_nodes, n_states]
            layer_streams: List of [batch, n_nodes, cpt_dim] tensors (n_layers+2)
        """
        from imputer.training_eval import forward_with_layer_capture
        return forward_with_layer_capture(
            self.base_model, inputs, structure_info, cpt_info, dimensions
        )

    def get_predictions_from_layer_stream(
        self,
        layer_parameter_stream: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert layer stream to predictions (REUSES existing function).

        This delegates to the existing get_predictions_from_layer_stream()
        function from training_eval.py to avoid code duplication.

        Args:
            layer_parameter_stream: [batch, n_nodes, cpt_dim]

        Returns:
            predictions: [batch, n_nodes, n_states]
        """
        from imputer.training_eval import get_predictions_from_layer_stream
        return get_predictions_from_layer_stream(
            self.base_model, layer_parameter_stream
        )

    def forward_layer_with_probe(
        self,
        layer_idx: int,
        layer_stream: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply probe to layer stream and get predictions.

        This is the KEY function for tuned lens: apply learned transformation
        before normalizing and passing to output heads.

        Args:
            layer_idx: Which layer (0=initial, 1=layer_0, ..., n_layers+1=final_norm)
            layer_stream: [batch, n_nodes, cpt_dim] - raw layer stream

        Returns:
            predictions: [batch, n_nodes, n_states] - state probabilities
        """
        # Apply learned probe transformation to raw layer stream
        transformed_stream = self.probes[layer_idx](layer_stream)

        # Apply final normalization (output heads expect normalized inputs)
        normalized_stream = self.base_model.transformer.final_norm_parameter(transformed_stream)

        # Apply output heads to normalized, transformed stream
        predictions = []
        for i in range(self.n_nodes):
            node_probs = self.base_model.output_heads[i](normalized_stream[:, i, :])
            predictions.append(node_probs)
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        structure_info: torch.Tensor,
        cpt_info: torch.Tensor,
        dimensions: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Forward pass through specific layer with tuned lens probe.

        Args:
            inputs: [batch, n_nodes, input_dim]
            structure_info: [batch, n_nodes, structure_dim]
            cpt_info: [batch, n_nodes, cpt_dim]
            dimensions: [batch, n_nodes]
            layer_idx: Which layer to probe (0=initial, ..., n_layers+1=final_norm)

        Returns:
            predictions: [batch, n_nodes, n_states] - probed predictions
        """
        # Capture all layer streams
        _, layer_streams = self.forward_with_layer_capture(
            inputs, structure_info, cpt_info, dimensions
        )

        # Apply probe and get predictions
        return self.forward_layer_with_probe(layer_idx, layer_streams[layer_idx])
