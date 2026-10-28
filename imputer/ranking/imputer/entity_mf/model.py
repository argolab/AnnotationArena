from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward

from .config import EntityMarformerConfig
from .data import EntityGraph
from .fastgraph import graph_tensors
from .types import EntityType
from .value_codecs import build_codecs, planned_reserved_tail, resolve_specs


class RelationalAttentionBlock(nn.Module):
    """
    Relational self-attention with two selectable designs:

    use_per_head_rel=True  (default, new design):
      Q: D -> D  (each head gets k_content_dim content dims + R relational dims)
      K: D -> H * k_content_dim  (content only)
      scores = (Q_content @ K_content^T + einsum(Q_rel, edge_mask)) / sqrt(head_dim)
      Each head learns independent relational weights.
      Requires: head_dim > R  (i.e. model_dim // num_heads > num_relationships).

    use_per_head_rel=False  (old/shared-bias design):
      Q: D -> D + R  (D content dims shared across heads, R shared relational dims)
      K: D -> D      (content only)
      scores = (Q_content @ K_content^T) / sqrt(head_dim)  +  Q_rel @ edge_mask^T
      Single shared relational bias added identically to all heads.

    rel_key="learned" (Shaw et al. 2018, S3.1)
      Supersedes BOTH modes above, which differ only in where the frozen
      indicator bits live. Instead of a bit that j waves at i, each relation
      gets a LEARNED embedding in key space, added to the key when i attends
      to j:

        scores[h,i,j] = q^h_i . (k^h_j + sum_{rho: i -rho-> j} a^K_rho[h]) / sqrt(hd)

      The key keeps the full head_dim (no dims are reserved) and the query
      needs no extra relational outputs, so `use_per_head_rel` and
      `scale_shared_rel` no longer change anything: both collapse to this one
      path. Each head reads its own slice a^K_rho[h], which is a superset of
      Shaw's design (they share one embedding across heads per layer).

      Note what this does and does not buy. The bias is q^h_i . a^K_rho =
      x_i^T W_Q^T a^K_rho, still a LINEAR function of x_i -- so as a score
      term it is a rank-constrained version of the bits design, not a richer
      one: the bits give each head an unconstrained R-dim linear readout of x,
      while this one must factor through W_Q. What it does buy is the learning
      dynamics Shaw et al. rely on: a^K_rho lives in key space, so similar
      relations can take similar embeddings, an unimportant relation can go to
      zero in every head at once, and no fixed dimension budget is reserved.

    rel_value="learned" (Shaw et al. 2018, S3.1, the other half)
      The genuinely new function class: the relation also enters the VALUE,

        z^h_i = sum_j alpha^h_ij (V^h(x_j) + sum_{rho: i -rho-> j} a^V_rho[h])

      per head and exactly, computed from the sparse edge list. Zero-init, so
      it is a no-op at step 0. This is the piece the shipped architecture has
      no analogue for at all.

    Optional extensions (both modes):
      use_pointer:     K_aug obs-obs shared-identity bias (3 extra shared Q dims, like old Marformer).
      use_rel_value:   V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r]. The OLD
                       approximation: one embedding shared across heads, driven
                       by the head-AVERAGED attention, and it needs the dense
                       [L, L, R] mask. rel_value="learned" is the exact,
                       per-head, sparse version; the two are mutually exclusive.
      use_addone_attn: attn = exp(s) / (1 + sum_j exp(s_j)), sum <= 1.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_relationships: int,
        dropout: float,
        use_per_head_rel: bool = True,
        use_pointer: bool = False,
        use_rel_value: bool = False,
        use_addone_attn: bool = False,
        scale_shared_rel: bool = False,
        use_graph_mask: bool = False,
        rel_key: str = "bits",
        rel_value: str = "none",
        rel_init: str = "key_scale",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_relationships = num_relationships
        self.use_per_head_rel = use_per_head_rel
        self.use_pointer = use_pointer
        self.use_rel_value = use_rel_value
        self.use_addone_attn = use_addone_attn
        self.scale_shared_rel = scale_shared_rel
        self.use_graph_mask = use_graph_mask
        if rel_key not in ("bits", "learned"):
            raise ValueError(f"rel_key must be 'bits' or 'learned', got {rel_key!r}")
        if rel_value not in ("none", "learned"):
            raise ValueError(f"rel_value must be 'none' or 'learned', got {rel_value!r}")
        if rel_init not in ("key_scale", "zero"):
            raise ValueError(f"rel_init must be 'key_scale' or 'zero', got {rel_init!r}")
        if use_rel_value and rel_value == "learned":
            raise ValueError(
                "use_rel_value (old head-averaged approximation) and "
                "rel_value='learned' (Shaw's exact per-head term) both write the "
                "same bias; pick one."
            )
        self.rel_key = rel_key
        self.rel_value = rel_value
        self.rel_init = rel_init
        assert model_dim % num_heads == 0, (
            f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = model_dim // num_heads
        R = num_relationships

        if rel_key == "learned":
            # Shaw: the relation is a learned vector ADDED to the key, so the key
            # keeps every content dimension and the query needs no relational
            # outputs. use_per_head_rel / scale_shared_rel are inert here -- the
            # two bit designs differ only in where the reserved dims sit.
            self.k_content_dim = self.head_dim
            q_out_dim = model_dim + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, model_dim)
            self.key_rel = nn.Parameter(torch.zeros(R, num_heads, self.head_dim))
            if rel_init == "key_scale":
                # The same scale nn.Linear gives a key coordinate, so a relation
                # embedding starts out as loud as a key and no louder.
                nn.init.normal_(self.key_rel, std=model_dim ** -0.5)
        elif use_per_head_rel:
            # Per-head: each head has (head_dim - R) content dims + R relational dims
            self.k_content_dim = self.head_dim - R
            assert self.k_content_dim > 0, (
                f"head_dim {self.head_dim} must be > num_relationships {R}. "
                f"Increase model_dim or reduce num_heads."
            )
            q_out_dim = model_dim + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, num_heads * self.k_content_dim)
        else:
            # Shared-bias: full head_dim for content, separate R-dim shared relational vector
            self.k_content_dim = self.head_dim  # full head_dim used for content
            q_out_dim = model_dim + R + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, model_dim)  # content only

        self.V = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        # Relation-specific value embeddings: shared across heads, init to zero (no-op at start)
        if use_rel_value:
            self.value_rel = nn.Parameter(
                torch.zeros(R, model_dim)
            )

        # Shaw's a^V: per head, exact, and always zero-init -- this one adds
        # straight into the residual stream rather than passing through a
        # softmax, so a random init would inject a constant at step 0.
        if rel_value == "learned":
            self.value_rel_h = nn.Parameter(
                torch.zeros(R, num_heads, self.head_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def _rel_bias(self, Q_rel: torch.Tensor, edge_flat: torch.Tensor,
                  edge_rel: torch.Tensor, L: int) -> torch.Tensor:
        """
        Relational score bias from the sparse edge list.

        bias[..., i, j] = sum_r Q_rel[..., i, r] * edge_mask[i, j, r], computed
        by gathering one value per edge and scattering it into a [..., L*L]
        accumulator. Duplicate (i, j) pairs under different relations
        accumulate, which is what the dense contraction did too.

        Q_rel is [B, L, R] (shared bias) or [B, H, L, R] (per-head); the leading
        dimensions are flattened so one code path covers both.
        """
        lead = Q_rel.shape[:-2]
        q = Q_rel.reshape(-1, L, Q_rel.shape[-1])                  # [N, L, R]
        src = edge_flat // L
        vals = q[:, src, edge_rel]                                 # [N, nnz]
        out = torch.zeros(q.shape[0], L * L, device=q.device, dtype=q.dtype)
        out = out.index_add(1, edge_flat, vals)
        return out.view(*lead, L, L)

    def _rel_mass(self, attn: torch.Tensor, edge_flat: torch.Tensor,
                  edge_rel: torch.Tensor, L: int, R: int) -> torch.Tensor:
        """
        The attention mass each relation carries, for Shaw's value term.

        mass[..., i, r] = sum_j attn[..., i, j] * edge_mask[i, j, r] -- the
        transpose of _rel_bias: gather one attention weight per edge and
        scatter it into an [..., L*R] accumulator keyed by (i, r) instead of
        (i, j). Contracting j this way never builds the dense [L, L, R] mask,
        which is 365 MiB per cached graph at L=3900, R=6.

        Duplicate (i, j) pairs under different relations land in different r
        slots, so each relation gets its own share of the same weight.
        """
        lead = attn.shape[:-2]
        a = attn.reshape(-1, L * L)                                # [N, L*L]
        vals = a[:, edge_flat]                                     # [N, nnz]
        idx = (edge_flat // L) * R + edge_rel                      # [nnz] -> i*R + r
        out = torch.zeros(a.shape[0], L * R, device=a.device, dtype=a.dtype)
        out = out.index_add(1, idx, vals)
        return out.view(*lead, L, R)

    def forward(
        self,
        x: torch.Tensor,
        edge_flat: torch.Tensor,
        edge_rel: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, L, D] combined feature+param representation.
            edge_flat: [nnz] long, src * L + tgt for each edge.
            edge_rel:  [nnz] long, relation index for each edge.
            attn_mask: [B, L] bool mask for valid tokens, or None if nothing is padded.
            K_aug:     [3, L, L] obs-obs shared-identity indicators (optional, for pointer).
            edge_mask: dense [L, L, R], only needed by use_rel_value / use_graph_mask.
        """
        B, L, D = x.shape
        H = self.num_heads
        R = self.num_relationships
        hd = self.head_dim
        kc = self.k_content_dim
        if edge_mask is not None:
            edge_mask_f = edge_mask.to(x.dtype)                    # [L, L, R]

        V_base = self.V(x).view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]

        if self.rel_key == "learned":
            # ── Shaw et al. relational attention: learned key embeddings ──────
            # scores[h,i,j] = q^h_i . (k^h_j + sum_{rho: i->j} a^K_rho[h]) / sqrt(hd)
            # The second term is q . a^K contracted over the edge list, which is
            # the same sparse scatter the bit designs use -- only what gets
            # gathered changes.
            Q_proj = self.Q(x)                                     # [B, L, D] or [B, L, D+3]
            Qh = Q_proj[..., :D].view(B, L, H, hd).transpose(1, 2)  # [B, H, L, hd]
            if self.use_pointer:
                Q_ptr = Q_proj[..., D:]                            # [B, L, 3]
            Kh = self.K(x).view(B, L, H, hd).transpose(1, 2)        # [B, H, L, hd]
            # Scale Q, not the [B, H, L, L] scores: same reason as below.
            Qh = Qh * (1.0 / math.sqrt(hd))
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1))                  # [B, H, L, L]
            # [B, H, L, R]: how much head h at i would like each relation.
            rel_logits = torch.einsum("bhid,rhd->bhir", Qh, self.key_rel)
            rel_scores = self._rel_bias(rel_logits, edge_flat, edge_rel, L)          # [B, H, L, L]
            scores = content_scores + rel_scores
        elif self.use_per_head_rel:
            # ── Per-head relational attention ───────────────────────────────────
            # Q: D -> D (+ 3 for pointer); split per head into [content | rel]
            Q_proj = self.Q(x)                                     # [B, L, D] or [B, L, D+3]
            Q_full = Q_proj[..., :D].view(B, L, H, hd).transpose(1, 2)  # [B, H, L, hd]
            if self.use_pointer:
                Q_ptr = Q_proj[..., D:]                            # [B, L, 3]
            K_full = self.K(x).view(B, L, H, kc).transpose(1, 2) # [B, H, L, kc]
            # Scale Q rather than the scores: the scores are [B, H, L, L]
            # (135 MiB at B=5, L=1298), so dividing them is a full extra pass
            # over the largest tensor in the block for a factor that is linear
            # in Q anyway.
            inv_scale = 1.0 / math.sqrt(hd)
            Q_content = Q_full[..., :kc] * inv_scale              # [B, H, L, kc]
            Q_rel     = Q_full[..., kc:] * inv_scale              # [B, H, L, R]
            content_scores = torch.matmul(Q_content, K_full.transpose(-2, -1))       # [B, H, L, L]
            rel_scores     = self._rel_bias(Q_rel, edge_flat, edge_rel, L)           # [B, H, L, L]
            scores = content_scores + rel_scores
        else:
            # ── Shared-bias relational attention (old design) ───────────────────
            # Q: D -> D+R (+ 3 for pointer); K: D -> D; rel bias shared across heads
            Q_proj = self.Q(x)                                     # [B, L, D+R] or [B, L, D+R+3]
            Q_content = Q_proj[..., :D]                            # [B, L, D]
            Q_rel_shared = Q_proj[..., D:D+R]                     # [B, L, R]  (shared across heads)
            if self.use_pointer:
                Q_ptr = Q_proj[..., D+R:]                          # [B, L, 3]
            K_content = self.K(x)                                  # [B, L, D]
            # Same reason as the per-head branch: fold the score scale into Q.
            inv_scale = 1.0 / math.sqrt(hd)
            Qh = Q_content.view(B, L, H, hd).transpose(1, 2) * inv_scale   # [B, H, L, hd]
            Kh = K_content.view(B, L, H, hd).transpose(1, 2)               # [B, H, L, hd]
            if self.scale_shared_rel:
                Q_rel_shared = Q_rel_shared * inv_scale
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1))        # [B, H, L, L]
            # Relational bias: [B, L, L] -> broadcast over heads [B, 1, L, L].
            # einsum rather than broadcast-multiply-then-sum: the latter
            # materialises a [B, L, L, R] intermediate (and its gradient), which
            # at L=1300, R=6 is 40 MiB per layer for a result that is 6.5 MiB.
            rel_scores = self._rel_bias(Q_rel_shared, edge_flat, edge_rel, L).unsqueeze(1)
            scores = content_scores + rel_scores

        # ── Pointer bias (shared across heads) ─────────────────────────────────
        if self.use_pointer and K_aug is not None:
            # Q_ptr: [B, L, 3], K_aug: [L, L, 3]
            # Three scaled planes rather than a contraction: Q_ptr[..., p] is
            # [B, L, 1] and broadcasts against K_aug[p], which is [L, L].
            K = K_aug.to(Q_ptr.dtype)
            ptr_bias = (Q_ptr[..., 0:1] * K[0]
                        + Q_ptr[..., 1:2] * K[1]
                        + Q_ptr[..., 2:3] * K[2])                  # [B, L, L]
            scores = scores + ptr_bias.unsqueeze(1)                        # broadcast over H

        # ── Graph mask (hard): allow only edge_mask + K_aug + self ─────────────
        if self.use_graph_mask:
            graph_mask = edge_mask.any(dim=-1)                                   # [L, L]
            if K_aug is not None:
                graph_mask = graph_mask | K_aug.any(dim=0).bool()                # [L, L]
            graph_mask = graph_mask | torch.eye(L, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(~graph_mask[None, None], torch.finfo(scores.dtype).min)

        # ── Attention mask ──────────────────────────────────────────────────────
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            key_mask = attn_mask[:, None, None, :]                # [B, 1, 1, L]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        # ── Softmax or add-one attention ────────────────────────────────────────
        if self.use_addone_attn:
            # Numerically stable: shift by max before exp
            scores_shifted = scores - scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores_shifted)
            if attn_mask is not None:
                exp_s = exp_s.masked_fill(~key_mask, 0.0)
            attn = exp_s / (1.0 + exp_s.sum(dim=-1, keepdim=True))
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # ── Value aggregation + optional relation value augmentation ───────────
        out = torch.matmul(attn, V_base)   # [B, H, L, hd]

        if self.rel_value == "learned":
            # Shaw's value term, exactly: z^h_i = sum_j a^h_ij (V^h(x_j) + a^V_rho[h]).
            # sum_j a^h_ij a^V_rho[h] factors as (relation mass) x (embedding),
            # so this costs one sparse gather and one small einsum -- no dense
            # mask and no [L, L, D] intermediate.
            mass = self._rel_mass(attn, edge_flat, edge_rel, L, R)                   # [B, H, L, R]
            out = out + torch.einsum("bhir,rhd->bhid", mass, self.value_rel_h)

        if self.use_rel_value:
            # Contract j first to avoid materializing [L, L, D] intermediate (OOM for large graphs).
            # attn_r_mass[b, i, r] = sum_j attn_mean[b, i, j] * edge_mask[i, j, r]
            attn_mean = attn.mean(dim=1)                                                  # [B, L, L]
            attn_r_mass = torch.einsum("bij,ijr->bir", attn_mean, edge_mask_f)           # [B, L, R]
            # bias[b, i, d] = sum_r attn_r_mass[b, i, r] * value_rel[r, d]
            bias = attn_r_mass @ self.value_rel                                           # [B, L, D]
            bias_h = bias.view(B, L, H, hd).permute(0, 2, 1, 3).contiguous()            # [B, H, L, hd]
            out = out + bias_h

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


def _init_type_embedding(p: nn.Parameter, mode: str, feature_dim: int) -> None:
    """Initialize a type centroid embedding parameter."""
    if mode == "normal":
        nn.init.normal_(p, mean=0.0, std=0.02)
    elif mode == "scaled_normal":
        nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(feature_dim))
    elif mode == "kaiming":
        nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
    else:
        raise ValueError(
            f"Unknown type_embedding_init: {mode!r}. "
            f"Choose from 'normal', 'scaled_normal', 'kaiming'."
        )


def _rename_legacy_codec_keys(state_dict, prefix, *_args, **_kwargs) -> None:
    """Load a checkpoint written before codecs were modules of their own.

    The typed codec used to keep its halves in two flat dicts,
    value_encoders.<type> and value_decoders.<type>. The layout of the weights
    is unchanged -- T was already column 0 of the encoder -- so this is a rename.
    The old shared output head (use_param_output_head) has no per-type
    counterpart to rename into, and no checkpoint ever used it.
    """
    moves = {"value_encoders.": "codecs.{}.encoder.proj.",
             "value_decoders.": "codecs.{}.decoder.probe."}
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        for old, new in moves.items():
            if rest.startswith(old):
                type_name, _, leaf = rest[len(old):].partition(".")
                state_dict[prefix + new.format(type_name) + leaf] = state_dict.pop(key)


class EntityMarformer(nn.Module):
    """
    Entity Marformer.

    Layer-0: x = type_centroid + entity_deviation + codec.encode(value), per token.
    Blocks:  attn+FFN on the residual stream, split at feature_dim for the
             optional outer residual / output projection.
    Output:  codec.decode(stream at layer L), per variable type.

    How each variable type's value enters and leaves the stream is its codec's
    business (value_codecs.py). The shipped dual stream is the slice codec: the
    value is copied into the last global_param_dim dimensions -- which the
    centroids are narrowed to stay out of -- and read back out of them. The
    typed codec is the linear one.
    """

    def __init__(
        self,
        config: EntityMarformerConfig,
        types: Dict[str, EntityType],
        num_relationships: int,
    ):
        super().__init__()
        self.config = config
        self.types = types
        self.use_per_head_rel = config.use_per_head_rel
        self.use_pointer = config.use_pointer
        self.use_rel_value = config.use_rel_value
        self.use_addone_attn = config.use_addone_attn
        # Set by enable_compile(); None means run the eager block stack.
        self._compiled_blocks = None
        # Set by enable_amp(); None means run the block stack in fp32.
        self._amp_dtype = None
        self.dual_outer_residual = getattr(config, "dual_outer_residual", "all")
        self.inner_residual = getattr(config, "inner_residual", "both")
        self.default_codec_spec, self.type_codec_specs = resolve_specs(config)
        self.block_output_projection = getattr(config, "block_output_projection", "full")
        if self.block_output_projection not in ("full", "none"):
            raise ValueError(
                f"Unknown block_output_projection: {self.block_output_projection!r}. "
                "Choose 'full' or 'none'."
            )
        if self.block_output_projection == "none" and self.dual_outer_residual != "none":
            # Removing the projection means the block body IS the new stream
            # value: h' = c = h + attn + ff, the textbook block. That is the
            # "replace" outer mode, i.e. dual_outer_residual='none'.
            #
            # Pairing it with 'all' instead gives h' = h + c = 2h + attn + ff,
            # which doubles the residual stream at every layer -- 2^8 = 256x over
            # the stack. That is not an ablation of the projection, it is a
            # broken model: it starts at a cross-entropy of 2067 against 5.7 for
            # the shipped architecture, and the first version of this experiment
            # measured its failure and called it the cost of removing the
            # projection.
            raise ValueError(
                "block_output_projection='none' requires dual_outer_residual='none' "
                "(h' = h + attn + ff). With 'all' the residual stream doubles every layer."
            )
        if self.inner_residual not in ("both", "ff_only", "none"):
            raise ValueError(
                f"Unknown inner_residual: {self.inner_residual!r}. "
                "Choose 'both', 'ff_only', or 'none'."
            )
        if self.inner_residual == "none" and self.dual_outer_residual == "none":
            raise ValueError(
                "inner_residual='none' with dual_outer_residual='none' leaves the "
                "block with no skip connection at all."
            )

        # Global param dim = max over types.
        self.global_param_dim = max(t.param_dim for t in types.values())
        self.param_dim = self.global_param_dim

        # Config embedding_dim = total model dim. Must be divisible by heads.
        self.model_dim = config.embedding_dim
        assert self.model_dim % config.attention_heads == 0, (
            f"embedding_dim {self.model_dim} must be divisible by attention_heads {config.attention_heads}"
        )

        assert self.model_dim > self.param_dim, (
            f"embedding_dim {self.model_dim} must be > global_param_dim {self.param_dim}"
        )
        self.feature_dim = self.model_dim - self.param_dim

        # Whatever trailing range a codec's encoder owns, no centroid or deviation
        # may write into at layer 0. The slice codec owns global_param_dim
        # dimensions, which is why the dual stream's centroids have always been
        # feature_dim wide; a learned codec owns none and the centroids are the
        # full model_dim. feature_dim itself is left alone either way: the block
        # stack still splits at the same place, which keeps the architectures
        # comparable layer for layer.
        self.reserved_tail = planned_reserved_tail(
            types, model_dim=self.model_dim, global_param_dim=self.global_param_dim,
            default_spec=self.default_codec_spec, type_specs=self.type_codec_specs)
        self.embed_dim = self.model_dim - self.reserved_tail

        self.type_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(1, self.embed_dim)) for name in types.keys()}
        )
        for p in self.type_embeddings.values():
            _init_type_embedding(p, config.type_embedding_init, self.embed_dim)

        self.deviation_tables = nn.ParameterDict()
        for name, t in types.items():
            if t.variation.enabled and t.variation.num_entities > 0:
                self.deviation_tables[name] = nn.Parameter(
                    torch.zeros(t.variation.num_entities, self.embed_dim)
                )

        self.deviation_norm = NormLayer(self.embed_dim)

        # One codec per variable type, built where the typed codec's modules
        # always were so a seed draws the same weights it always did.
        self.codecs = nn.ModuleDict(build_codecs(
            types, model_dim=self.model_dim, global_param_dim=self.global_param_dim,
            default_spec=self.default_codec_spec, type_specs=self.type_codec_specs,
            logit_high=config.logit_high))
        built = max((c.reserved_tail for c in self.codecs.values()), default=0)
        assert built == self.reserved_tail, (built, self.reserved_tail)
        self._register_load_state_dict_pre_hook(_rename_legacy_codec_keys)

        blocks: List[nn.Module] = []
        for _ in range(config.num_layers):
            blocks.append(nn.ModuleDict({
                "norm_1":    NormLayer(self.model_dim),
                "attn":      self._make_attn_block(config, num_relationships),
                "norm_2":    NormLayer(self.model_dim),
                "ff":        FeedForward(self.model_dim, d_ff=config.d_ff, dropout=config.dropout, num_layers=config.num_ffn_layers),
                "dropout_2": nn.Dropout(config.dropout),
            }))
            if self.block_output_projection == "full":
                blocks[-1]["proj_out"] = nn.Linear(self.model_dim, self.feature_dim)
                blocks[-1]["W_param"] = nn.Linear(self.model_dim, self.param_dim)
        self.blocks = nn.ModuleList(blocks)
        if getattr(config, "block_output_projection_freeze", False):
            if self.block_output_projection != "full":
                raise ValueError(
                    "block_output_projection_freeze needs a projection to freeze."
                )
            for b in self.blocks:
                for mod in (b["proj_out"], b["W_param"]):
                    for prm in mod.parameters():
                        prm.requires_grad_(False)


    @property
    def codec_summary(self) -> Dict[str, str]:
        """{type: canonical spec}, the form train_config.json records."""
        return {name: c.spec.render() for name, c in self.codecs.items()}

    def enable_compile(self, mode: str = "default") -> None:
        """
        Compile the block stack.

        Worth doing because the step is launch-bound, not math-bound: at L~1300
        the GPU does ~11 ms of work per step while the CPU spends ~27 ms queueing
        it. Chunks have different token counts, so the graph is compiled once per
        distinct L (a handful) -- dynamic=False keeps each specialisation fast
        rather than tracing one dynamic kernel for all of them.
        """
        if mode in ("", "none"):
            self._compiled_blocks = None
            return
        self._compiled_blocks = torch.compile(self._blocks, mode=mode, dynamic=False)

    def enable_amp(self, dtype: str = "none") -> None:
        """
        Run the block stack under autocast.

        The block is dominated by traffic over the [B, H, L, L] score tensor, so
        halving its width is worth more here than the matmul speedup. Note the
        param stream carries raw logits (logit_high = 20), so this is a real
        numerical change, not just a speed knob -- check the loss curve.
        """
        self._amp_dtype = {"none": None, "bf16": torch.bfloat16,
                           "fp16": torch.float16}[dtype]

    def _run_blocks(self, *core_args) -> torch.Tensor:
        core = self._compiled_blocks if self._compiled_blocks is not None else self._blocks
        if self._amp_dtype is None:
            return core(*core_args)
        with torch.autocast("cuda", dtype=self._amp_dtype):
            out = core(*core_args)
        return out.float()

    def _make_attn_block(self, config: EntityMarformerConfig, num_relationships: int) -> RelationalAttentionBlock:
        return RelationalAttentionBlock(
            model_dim=self.model_dim,
            num_heads=config.attention_heads,
            num_relationships=num_relationships,
            dropout=config.dropout,
            use_per_head_rel=config.use_per_head_rel,
            use_pointer=config.use_pointer,
            use_rel_value=config.use_rel_value,
            use_addone_attn=config.use_addone_attn,
            scale_shared_rel=config.scale_shared_rel,
            use_graph_mask=config.use_graph_mask,
            rel_key=config.rel_key,
            rel_value=config.rel_value,
            rel_init=config.rel_init,
        )

    def _dense_edge_mask(self, graph: EntityGraph, device: torch.device) -> "torch.Tensor | None":
        """
        The [L, L, R] mask, built only for the features that still need it.

        The relational bias now reads the sparse edge list, so with rel_value
        and graph_mask off -- the shipped Domain 3 and LLM-Rubric settings --
        this tensor is never touched, and it is 365 MiB per cached graph at
        L=3900, R=6.
        """
        if self.use_rel_value or self.config.use_graph_mask:
            return graph.build_edge_masks(device=device)
        return None

    def _get_k_aug(self, graph: EntityGraph, device: torch.device) -> "torch.Tensor | None":
        """Build (and cache) the [3, L, L] obs-obs shared-identity K_aug tensor."""
        if not self.use_pointer:
            return None
        dev_key = str(device)
        if dev_key not in graph._k_aug_cache:
            L = graph.num_tokens
            attr_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            annot_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            item_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            for idx, token in enumerate(graph.tokens):
                if token.type_name in ("rating", "ranking_pairwise") and token.raw_data:
                    attr_ids[idx]  = token.raw_data.get("attribute_id", -1)
                    annot_ids[idx] = token.raw_data.get("annotator_id", -1)
                    iids = token.raw_data.get("item_ids", [])
                    item_ids[idx]  = iids[0] if iids else -1

            def _same(ids: torch.Tensor) -> torch.Tensor:
                eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
                valid = (ids >= 0).float()
                return eq * valid.unsqueeze(0) * valid.unsqueeze(1)

            # [3, L, L], not [L, L, 3]. Contracting the trailing 3 against
            # Q_ptr's trailing 3 lands in cuBLAS gemvx at ~0.9 ms a call (18% of
            # GPU time at L=3900) because the reduction axis has stride 3. With
            # the planes separated it is three broadcast multiplies instead.
            graph._k_aug_cache[dev_key] = torch.stack(
                [_same(attr_ids), _same(annot_ids), _same(item_ids)], dim=0
            )
        return graph._k_aug_cache[dev_key]

    def _build_initial_streams(
        self,
        graph: EntityGraph,
        device: torch.device,
        status: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Layer 0, returned split at feature_dim as the block stack expects.

            x = type_centroid + entity_deviation + codec.encode(value)

        The centroid and deviation are embed_dim wide and padded with zeros up to
        model_dim, so they stay out of whatever range the codecs own. Each codec
        then adds its encoding of its own tokens' values: T for a hidden token,
        the value for a revealed one, from whichever source the codec reads.

        Everything here except `status` is fixed by the graph topology, so it is
        precomputed once per graph by GraphTensors and this is a handful of
        gathers. See _build_initial_streams_ref for the original per-token loop;
        test_equiv.py asserts the two agree.

        status is [B, L] for the batched path and None to use the graph's own.

        Returns:
            features:  [B, L, feature_dim]
            params:    [B, L, param_dim]
            attn_mask: None -- every token is real, and returning None instead
                       of an all-true mask skips a masked_fill over the
                       [B, H, L, L] score tensor per layer.
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        if status is None:
            gt.refresh(graph)
        B = 1 if status is None else status.shape[0]

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, E]
        x = type_mat.index_select(0, gt.type_ids).unsqueeze(0)                         # [1, L, E]
        if B > 1:
            x = x.expand(B, -1, -1)

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, E]
            dev = dev.unsqueeze(0).expand(B, -1, -1)
            if self.training and variation.dropout_rate > 0:
                # One independent Bernoulli draw per token and per augmentation,
                # as a single kernel instead of M `.item()` syncs.
                keep = (torch.rand(B, dev.shape[1], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            x = x.index_add(1, pos, dev)

        if self.reserved_tail:
            x = F.pad(x, (0, self.reserved_tail))                                     # [B, L, D]

        rows: Dict[str, torch.Tensor] = {}
        for name, codec in self.codecs.items():
            pos = gt.slices[name].positions
            if not pos.numel():
                continue
            src = codec.value_source
            if src not in rows:
                rows[src] = gt.value_rows(src, status)                                 # [B, L, P]
            x = x.index_add(1, pos, codec.encode(rows[src].index_select(1, pos)))

        return x[..., : self.feature_dim], x[..., self.feature_dim :], None

    def _build_initial_streams_ref(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original per-token implementation. Kept as the equivalence reference.

        It is the dual stream's layer 0 written out by hand, so it is only a
        reference for the slice codec.
        """
        if any(c.spec.encoder.kind != "slice" or c.value_source != "hand_built"
               for c in self.codecs.values()):
            raise NotImplementedError("the reference loop only knows the slice codec")
        L = graph.num_tokens
        features = torch.zeros(1, L, self.feature_dim, device=device)
        params   = torch.zeros(1, L, self.param_dim,   device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]
            feat_vec = self.type_embeddings[token.type_name].expand(1, -1)
            if token.type_name in self.deviation_tables and token.entity_id >= 0:
                dev_table = self.deviation_tables[token.type_name]
                if 0 <= token.entity_id < dev_table.shape[0]:
                    dev = dev_table[token.entity_id].unsqueeze(0)
                    if self.training and t.variation.dropout_rate > 0:
                        if torch.rand(1).item() < t.variation.dropout_rate:
                            dev = torch.zeros_like(dev)
                    if self.config.use_deviation_norm:
                        dev = self.deviation_norm(dev)
                    feat_vec = feat_vec + dev
            features[0, idx] = feat_vec
            params[0, idx]   = t.build_param(token.raw_data or {}, device=device, global_param_dim=self.param_dim)

        return features, params, attn_mask

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over a single EntityGraph.

        Returns:
            params: [1, L, global_param_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        return self._forward(graph, device)

    def forward_batched(
        self,
        graph: EntityGraph,
        status: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over one EntityGraph under B different masks at once.

        The graph topology -- and therefore the edge mask, K_aug and every
        attention shape but the leading dimension -- is identical across the
        mask augmentations of a training epoch. Running them as a batch turns
        B forward/backward passes into one, which matters because the step is
        launch-bound: at L~1300 the GPU does ~11 ms of work per pass while the
        CPU spends ~27 ms queueing it.

        Args:
            graph:  the chunk graph.
            status: [B, L] long, per-augmentation token statuses
                    (0=missing, 1=masked, 2=observed), from
                    GraphTensors.sample_status.

        Returns:
            params: [B, L, global_param_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        return self._forward(graph, device, status)

    def _forward(self, graph: EntityGraph, device: torch.device,
                 status: torch.Tensor | None = None) -> torch.Tensor:
        features, params, attn_mask = self._build_initial_streams(graph, device, status)
        edge_flat, edge_rel = graph.build_edge_index(device)
        edge_mask = self._dense_edge_mask(graph, device)
        K_aug = self._get_k_aug(graph, device)
        combined = self._run_blocks(features, params, edge_flat, edge_rel,
                                    attn_mask, K_aug, edge_mask)
        return self._decode(combined, graph, device)

    def _decode(self, combined: torch.Tensor, graph: EntityGraph,
                device: torch.device) -> torch.Tensor:
        """Per-type decode of the final residual stream.

        Returns the [B, L, global_param_dim] carrier the loss and evaluation
        read, each type's C logits in slots 1..C (slot 0 is T's place in the
        extended domain and is never predicted). Entity tokens, which have no
        value, are left at zero.
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        B, L, _ = combined.shape
        out = combined.new_zeros(B, L, self.global_param_dim)
        for name, codec in self.codecs.items():
            pos = gt.slices[name].positions
            if not pos.numel():
                continue
            logits = codec.decode(combined.index_select(1, pos))                       # [B, M, C]
            carrier = logits.new_zeros(B, pos.shape[0], self.global_param_dim)
            carrier[..., 1 : 1 + logits.shape[-1]] = logits
            out = out.index_add(1, pos, carrier)
        return out

    def _outer_residual_mode(self, layer_idx: int, num_blocks: int) -> str:
        """How this layer combines its projection with the outer stream.

        Returns "add_h" (baseline outer residual), "replace" (rotated part only),
        or "add_c" (identity folded into the projection).
        """
        mode = self.dual_outer_residual
        is_last = layer_idx == num_blocks - 1
        if mode == "all":
            return "add_h"
        if mode == "none":
            return "replace"
        if mode == "not_last":
            return "replace" if is_last else "add_h"
        if mode == "none_id":
            return "add_c"
        if mode == "not_last_id":
            return "add_c" if is_last else "add_h"
        raise ValueError(
            f"Unknown dual_outer_residual: {mode!r}. Choose 'all', 'not_last', "
            "'none', 'none_id', or 'not_last_id'."
        )

    def _blocks(
        self,
        features: torch.Tensor,
        params: torch.Tensor,
        edge_flat: torch.Tensor,
        edge_rel: torch.Tensor,
        attn_mask: torch.Tensor | None,
        K_aug: torch.Tensor | None,
        edge_mask: torch.Tensor | None = None,
        trace: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        The block stack, as a pure tensor -> tensor function. Returns the
        residual stream at layer L, [B, L, model_dim].

        Split out of _forward so it can be handed to torch.compile: the
        graph-dependent work (which is Python, and cannot be traced) all happens
        before this point, and everything from here is fixed-shape tensor code.

        `trace`, if given, receives the stream at every depth 0..L. It is how the
        logit lens reads intermediate layers through the very same arithmetic,
        rather than through a copy of it that can drift.
        """
        combined = torch.cat([features, params], dim=-1)  # [B, L, model_dim]
        if trace is not None:
            trace.append(combined)
        num_blocks = len(self.blocks)
        for layer_idx, block in enumerate(self.blocks):
            # Inner residual stream. "both" is the shipped architecture; dropping
            # these adds turns the block into a single residual unit whose body is
            # a serial attention -> FFN composition, leaving the outer add as the
            # only skip connection.
            normed_attn = block["norm_1"](combined)
            attn_out    = block["attn"](normed_attn, edge_flat=edge_flat, edge_rel=edge_rel,
                                        attn_mask=attn_mask, K_aug=K_aug, edge_mask=edge_mask)
            combined    = combined + attn_out if self.inner_residual == "both" else attn_out

            normed_ff = block["norm_2"](combined)
            z_ff      = block["ff"](normed_ff)
            combined  = combined + z_ff if self.inner_residual in ("both", "ff_only") else z_ff

            # Outer residual. `combined` already carries an inner residual stream, so
            # adding proj_out/W_param back onto features/params gives a second, parallel
            # stream -- the "double residual stream". dual_outer_residual selects which
            # layers keep that outer add; dropping it leaves only the rotated part.
            if self.block_output_projection == "full":
                feat_update  = block["dropout_2"](block["proj_out"](combined))
                param_update = block["dropout_2"](block["W_param"](combined))
            else:
                # Identity in place of [proj_out; W_param]: the block body reaches
                # the outer stream unrotated, giving h' = h + attn + ff.
                upd          = block["dropout_2"](combined)
                feat_update  = upd[..., : self.feature_dim]
                param_update = upd[..., self.feature_dim :]
            mode = self._outer_residual_mode(layer_idx, num_blocks)
            if mode == "add_h":
                # Baseline: add onto the stream as it entered the layer.
                features = features + feat_update
                params   = params   + param_update
            elif mode == "replace":
                # Only the rotated part survives; no identity path across layers.
                features, params = feat_update, param_update
            else:  # "add_c" -- identity folded into [W1; W2]
                # h' = c + [W1; W2] c, i.e. the stacked projection is (I + M')
                # rather than M'. Same single-stream shape as "replace", but the
                # identity path is restored -- and unlike "add_h" the attention
                # and FFN output reaches the next layer unmultiplied. Weight
                # decay now pulls M' toward 0, which means toward the identity
                # layer, instead of toward the zero map.
                features = combined[..., : self.feature_dim] + feat_update
                params   = combined[..., self.feature_dim :] + param_update
            combined = torch.cat([features, params], dim=-1)
            if trace is not None:
                trace.append(combined)

        return combined
