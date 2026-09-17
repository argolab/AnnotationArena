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
    value_codec: str = "dual"            # How a variable's value enters and leaves the residual stream.
                                         #   "dual"  -> shipped: a hand-built parameter stream carries the value,
                                         #              and its final value IS the prediction.
                                         #   "typed" -> a learned per-type encoder maps the value's binary
                                         #              encoding (wildcard bit + one-hot over C classes) into
                                         #              R^d, and a separate untied decoder maps R^d -> R^C,
                                         #              read out with a softmax. The block stack is identical.
    codec_init: str = "unit"             # "unit" -> Gaussian encoder at std 1.0. The default because it
                                         #           carries no constant at all: output norm sqrt(d)~=8.9,
                                         #           and B10 measured it at parity with the dual stream at
                                         #           K=2000 (-0.009 +/-0.003, n=6). The requirement is
                                         #           "loud enough", not "exactly logit_high".
                                         # "scaled" -> Gaussian encoder at std logit_high/sqrt(d), so the
                                         #             value arrives at the magnitude the stream expects.
                                         #             This is the default because "random" (std 0.02) is
                                         #             ~1000x too quiet and costs 0.09 nats; see
                                         #             threads/PLAN.md, thread 2.
                                         # "random" -> Gaussian encoder at std 0.02, default Linear decoder.
                                         #             Kept only to reproduce the regression.
                                         # "aligned" -> encoder writes into the last 1+C dimensions and the
                                         #              decoder reads the last C, reproducing the dual stream
                                         #              exactly (see threads/value-codec/b2_equivalence.py).
    codec_freeze: bool = False           # Hold the encoder and decoder fixed at their initial values.
    dual_outer_residual: str = "all"     # Dual-stream only. Which layers keep the OUTER residual add (features/params += proj(combined)).
                                         #   "all"      -> baseline: every layer adds (double residual stream: inner on `combined`, outer on features/params)
                                         #   "not_last" -> every layer but the last adds; the last layer emits only [proj_out; W_param](combined)
                                         #   "none"     -> no layer adds; the stream is carried purely by `combined`'s inner residual, rotated each layer
                                         #   "none_id" / "not_last_id" -> identity folded into the projection: h' = c + [W1;W2]c

    inner_residual: str = "both"          # Dual-stream only. The residual adds INSIDE the block.
                                         #   "both"    -> c = h + attn; c = c + ff   (shipped)
                                         #   "ff_only" -> c = attn;     c = c + ff
                                         #   "none"    -> c = attn;     c = ff        (outer add is the only skip)
