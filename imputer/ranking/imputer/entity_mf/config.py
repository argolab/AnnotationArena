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
    use_graph_mask: bool = False         # Hard graph attention mask: allow attention only where edge_mask or K_aug pointer exists (+ self)
    use_param_output_head: bool = False  # Predict final params from the last combined hidden state instead of using the residual param stream directly
    use_embed_unembed: bool = False      # Embedding/unembedding architecture: single residual stream; value embedded via learned A*x+b at layer 0, unembedded via per-type linear at output
    dual_outer_residual: str = "all"     # Dual-stream only. Which layers keep the OUTER residual add (features/params += proj(combined)).
                                         #   "all"      -> baseline: every layer adds (double residual stream: inner on `combined`, outer on features/params)
                                         #   "not_last" -> every layer but the last adds; the last layer emits only [proj_out; W_param](combined)
                                         #   "none"     -> no layer adds; the stream is carried purely by `combined`'s inner residual, rotated each layer
                                         #   "none_id" / "not_last_id" -> identity folded into the projection: h' = c + [W1;W2]c

    block_output_projection_freeze: bool = False
                                         # Hold every block's output projection at its random
                                         # initialisation. Tests whether the projection is a
                                         # LEARNED transform or a fixed random mixing: the trained
                                         # matrices keep their initialisation statistics almost
                                         # exactly (effective rank ~57/80, no identity component,
                                         # all four row/column blocks at the same RMS), so the
                                         # question is whether training it matters at all.
    block_output_projection: str = "full"
                                         # Dual-stream only. The trailing d -> d projection
                                         # [proj_out; W_param] applied to the block body.
                                         #   "full" -> shipped: the projection is learned
                                         #   "none" -> the projection is the identity, so the
                                         #             block is h' = h + attn + ff. Removes
                                         #             num_layers * d * d parameters. Exactly
                                         #             foldable into the feed-forward's output
                                         #             linear ONLY when inner_residual="none";
                                         #             with an inner residual it is a real
                                         #             change of function class (see
                                         #             RERUN/threads/block-projection/c0_fold.py).

    inner_residual: str = "both"          # Dual-stream only. The residual adds INSIDE the block.
                                         #   "both"    -> c = h + attn; c = c + ff   (shipped)
                                         #   "ff_only" -> c = attn;     c = c + ff
                                         #   "none"    -> c = attn;     c = ff        (outer add is the only skip)
