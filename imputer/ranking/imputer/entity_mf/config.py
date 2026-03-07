from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityMarformerConfig:
    """
    Minimal configuration for the Entity Marformer.

    embedding_dim is the total base model dimension (feature_dim + param_dim).
    Must be divisible by attention_heads and greater than max type param_dim;
    global_param_dim is computed from types and feature_dim = embedding_dim - global_param_dim.
    """

    embedding_dim: int = 72
    num_layers: int = 4
    attention_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 128
    num_ffn_layers: int = 1
    logit_high: float = 20.0
    temperature: float = 1.0

