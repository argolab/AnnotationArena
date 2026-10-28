from __future__ import annotations

from dataclasses import dataclass, field


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
    rel_key: str = "bits"           # How a relation enters the ATTENTION SCORE.
                                    #   "bits"    -> shipped: rel_{ij} is a frozen {0,1}^R
                                    #                indicator standing in for the key's
                                    #                relational dimensions, so the score term is
                                    #                the query's own r-th relational coordinate.
                                    #   "learned" -> Shaw et al. (2018) S3.1: each relation gets a
                                    #                LEARNED per-head embedding a^K_rho in key
                                    #                space, added to the key. No dimensions are
                                    #                reserved and the query needs no extra outputs,
                                    #                so use_per_head_rel and scale_shared_rel are
                                    #                inert. NOT more expressive as a score term
                                    #                (q.a^K is still linear in x_i, and now has to
                                    #                factor through W_Q) -- the claim is about
                                    #                learning dynamics and parameter sharing.
    rel_value: str = "none"         # Whether a relation also enters the VALUE.
                                    #   "none"    -> shipped: it does not.
                                    #   "learned" -> Shaw's other half, per head and exact:
                                    #                z^h_i = sum_j alpha^h_ij (V^h(x_j) + a^V_rho[h]).
                                    #                This is the half that is genuinely a wider
                                    #                function class; use_rel_value is an older
                                    #                head-averaged approximation of it and the two
                                    #                are mutually exclusive.
    rel_init: str = "key_scale"     # Init for a^K. "key_scale" -> std 1/sqrt(model_dim), the scale
                                    # nn.Linear gives a key coordinate, so a relation starts as loud
                                    # as a key. "zero" -> relational attention starts switched off
                                    # and has to be grown, which is the first thing to suspect if
                                    # the learned arm trains more slowly. a^V is always zero-init.
    use_per_head_rel: bool = True   # Per-head relational bias (each head learns own R weights); False = old shared-bias design (single shared R-dim bias added to all heads identically)
    use_pointer: bool = False       # K_aug obs-obs shared-identity pointer bias (3 extra shared Q dims, like old Marformer)
    use_rel_value: bool = False     # Relation-specific value augmentation V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r]
    use_addone_attn: bool = False   # Add-one (softmax-1) attention: attn = exp(s) / (1 + sum(exp(s)))
    type_embedding_init: str = "normal"  # Init for type centroid embeddings: "normal" (BERT-style std=0.02) | "scaled_normal" (std=1/sqrt(feature_dim)) | "kaiming" (legacy, not recommended)
    use_deviation_norm: bool = False     # Apply LayerNorm to deviation before adding to type centroid (bounds deviation scale)
    scale_shared_rel: bool = False       # In shared-bias mode: scale rel_scores by 1/sqrt(head_dim), matching per-head normalization
    use_graph_mask: bool = False         # Hard graph attention mask: allow attention only where edge_mask or K_aug pointer exists (+ self)
    use_param_output_head: bool = False  # Legacy: equivalent to a decoder spec of "mlp:hidden=128"
                                         # (per type now, where it used to be one shared head).
    value_codec: str = "slice"           # How every variable type's value enters and leaves the
                                         # residual stream, as a codec spec -- see value_codecs.py
                                         # for the grammar. Encoder and decoder are chosen
                                         # independently ("enc/dec"); one half means both alike.
                                         #   "slice"           -> shipped: the hand-built row is
                                         #                        copied into the last
                                         #                        global_param_dim dimensions and
                                         #                        read back out of them (the dual
                                         #                        stream; "dual" is accepted).
                                         #   "linear[:init=X]" -> a learned projection into the
                                         #                        stream and a linear probe out of
                                         #                        it (the typed codec; "typed" is
                                         #                        accepted, and reads codec_init).
                                         #   "mlp[:hidden=H]"  -> the same with an MLP each way.
                                         #   ":width=T"        -> a learned codec confined to the
                                         #                        last T dimensions.
    type_codecs: dict = field(default_factory=dict)
                                         # Per-type overrides, {type_name: spec}.
    codec_init: str = "unit"             # Legacy: the init for value_codec="typed" only. A spec
                                         # carries its own (linear:init=X); the choices and what
                                         # they measured are documented on LinearEncoder.
    codec_freeze: bool = False           # Legacy: value_codec="typed" only; a spec says "freeze".
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
