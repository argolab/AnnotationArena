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
    use_per_head_rel: bool = True   # Per-head relational bias (each head learns own R weights); False = old shared-bias design (single shared R-dim bias added to all heads identically)
    use_pointer: bool = False       # K_aug obs-obs shared-identity pointer bias (3 extra shared Q dims, like old Marformer)
    use_rel_value: bool = False     # Relation-specific value augmentation V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r]
    use_addone_attn: bool = False   # Add-one (softmax-1) attention: attn = exp(s) / (1 + sum(exp(s)))
    type_embedding_init: str = "normal"  # Init for type centroid embeddings: "normal" (BERT-style std=0.02) | "scaled_normal" (std=1/sqrt(feature_dim)) | "kaiming" (legacy, not recommended)
    use_deviation_norm: bool = False     # Apply LayerNorm to deviation before adding to type centroid (bounds deviation scale)
    scale_shared_rel: bool = False       # In shared-bias mode: scale rel_scores by 1/sqrt(head_dim), matching per-head normalization
    use_learned_embedding: bool = False  # Replace fixed-scale build_param with learned Ax+b embedding; unembeds at top layer

