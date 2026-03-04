from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityMarformerConfig:
    """
    Minimal configuration for the Entity Marformer.

    This is intentionally small for fast iteration; additional knobs can be
    added later as needed.
    """

    embedding_dim: int = 67
    num_layers: int = 4
    attention_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 128
    num_ffn_layers: int = 1
    logit_high: float = 20.0
    temperature: float = 1.0

