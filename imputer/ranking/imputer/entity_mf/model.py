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

    Optional extensions (both modes):
      use_pointer:     K_aug obs-obs shared-identity bias (3 extra shared Q dims, like old Marformer).
      use_rel_value:   V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r].
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
        assert model_dim % num_heads == 0, (
            f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = model_dim // num_heads
        R = num_relationships

        if use_per_head_rel:
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

        if self.use_per_head_rel:
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


class EntityMarformer(nn.Module):
    """
    Entity Marformer.

    Layer-0: [feature_centroid + deviation | value] (feature_dim + param_dim = model_dim),
             where the value is either the hand-built parameter row (value_codec="dual")
             or a learned per-type encoding of it (value_codec="typed").
    Blocks:  attn+FFN on combined; explicit proj_out and W_param residuals back to each stream.
    Output:  the param stream at layer L, or the typed decoder applied to the whole stream.
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
        self.use_param_output_head = config.use_param_output_head
        # Set by enable_compile(); None means run the eager block stack.
        self._compiled_blocks = None
        # Set by enable_amp(); None means run the block stack in fp32.
        self._amp_dtype = None
        self.dual_outer_residual = getattr(config, "dual_outer_residual", "all")
        self.inner_residual = getattr(config, "inner_residual", "both")
        self.value_codec = getattr(config, "value_codec", "dual")
        self.codec_init = getattr(config, "codec_init", "random")
        self.codec_freeze = bool(getattr(config, "codec_freeze", False))
        if self.value_codec not in ("dual", "typed"):
            raise ValueError(
                f"Unknown value_codec: {self.value_codec!r}. Choose 'dual' or 'typed'."
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

        # With the typed codec the whole vector is one stream, so a type
        # centroid and an entity deviation are model_dim wide rather than
        # feature_dim wide. feature_dim itself is left alone: the block stack
        # still splits at the same place, which is what keeps the two
        # architectures comparable layer for layer.
        self.embed_dim = self.model_dim if self.value_codec == "typed" else self.feature_dim

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

        if self.value_codec == "typed":
            # One encoder and one decoder per VARIABLE type. The encoder has
            # no bias: the type centroid above is the bias, shared with the
            # entity types so every token is "centroid + something".
            # The decoder is untied from the encoder and never predicts the
            # wildcard, so it emits C outputs for a type of param_dim 1 + C.
            self.value_encoders = nn.ModuleDict()
            self.value_decoders = nn.ModuleDict()
            for name, t in types.items():
                if not getattr(t, "is_variable", False) or t.param_dim <= 1:
                    continue
                self.value_encoders[name] = nn.Linear(t.param_dim, self.model_dim, bias=False)
                self.value_decoders[name] = nn.Linear(self.model_dim, t.param_dim - 1)
            self._init_value_codec(types)

        if self.use_param_output_head:
            self.param_output_head = nn.Sequential(
                nn.Linear(self.model_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.param_dim),
            )

        blocks: List[nn.Module] = []
        for _ in range(config.num_layers):
            blocks.append(nn.ModuleDict({
                "norm_1":    NormLayer(self.model_dim),
                "attn":      self._make_attn_block(config, num_relationships),
                "norm_2":    NormLayer(self.model_dim),
                "ff":        FeedForward(self.model_dim, d_ff=config.d_ff, dropout=config.dropout, num_layers=config.num_ffn_layers),
                "proj_out":  nn.Linear(self.model_dim, self.feature_dim),
                "W_param":   nn.Linear(self.model_dim, self.param_dim),
                "dropout_2": nn.Dropout(config.dropout),
            }))
        self.blocks = nn.ModuleList(blocks)


    def _init_value_codec(self, types: Dict[str, EntityType]) -> None:
        """Initialise the per-type encoders and decoders.

        "random" is the real configuration: a Gaussian encoder into R^d and a
        default-initialised decoder out of it.

        "aligned" is the diagnostic. It places the encoder in the LAST 1 + C
        dimensions of R^d and has the decoder read the last C, which is exactly
        where the dual stream keeps its value slice -- so with the codec frozen
        the two architectures are the same function. The scale factor matters
        and is easy to get wrong: the dual stream's layer 0 writes 1.0 in the
        wildcard slot but `logit_high` in the observed class's slot, so the
        aligned encoder is diag(1, logit_high, ..., logit_high), not the
        identity.
        """
        for name, enc in self.value_encoders.items():
            dec = self.value_decoders[name]
            P = types[name].param_dim
            C = P - 1
            logit_high = float(getattr(types[name], "logit_high", self.config.logit_high))
            if self.codec_init == "random":
                nn.init.normal_(enc.weight.data, mean=0.0, std=0.02)
                continue
            if self.codec_init == "scaled":
                # Same random code as "random", but at the magnitude the residual
                # stream actually expects. The hand-built stream writes
                # logit_high (20.0) into an observed value's slot, so the value
                # arrives with L2 norm 20; a N(0, 0.02) encoder delivers ~0.02 --
                # three orders of magnitude quieter. The debug ladder showed that
                # is the whole problem: an aligned encoder matches the shipped
                # model to within noise, a random one is 0.087 worse, and a random
                # one that cannot adapt is 0.274 worse. std = logit_high /
                # sqrt(model_dim) gives an output norm of about logit_high, so the
                # code is random but arrives at the right volume.
                nn.init.normal_(enc.weight.data, mean=0.0,
                                std=logit_high / math.sqrt(self.model_dim))
                continue
            if self.codec_init == "orthogonal":
                # Loud AND well separated. "scaled" fixed the magnitude and
                # removed the systematic penalty, but left roughly one run in six
                # at small K landing somewhere training never leaves -- a
                # Gaussian draw can start with two rating levels pointing in
                # nearly the same direction, which is a bad code to be stuck
                # with when there is little data to correct it. Orthonormal
                # columns scaled to logit_high give every value its own
                # direction at the magnitude the stream expects, with no
                # hand-written layout: which direction each value gets is still
                # arbitrary.
                nn.init.orthogonal_(enc.weight.data)
                enc.weight.data.mul_(logit_high)
                continue
            if self.codec_init != "aligned":
                raise ValueError(
                    f"Unknown codec_init: {self.codec_init!r}. "
                    "Choose 'random', 'scaled', 'orthogonal' or 'aligned'."
                )
            enc.weight.data.zero_()
            base = self.model_dim - P
            enc.weight.data[base, 0] = 1.0                       # wildcard bit, unscaled
            for j in range(C):
                enc.weight.data[base + 1 + j, 1 + j] = logit_high
            dec.weight.data.zero_()
            for j in range(C):
                dec.weight.data[j, base + 1 + j] = 1.0
            dec.bias.data.zero_()

        if self.codec_init == "aligned":
            # The codec owns the last global_param_dim dimensions of the stream,
            # so nothing else may write there at layer 0 -- including the
            # centroids of the ENTITY types, which have no encoder of their own.
            # Leaving those random was the one thing standing between this
            # configuration and exact agreement with the dual stream.
            for p in self.type_embeddings.values():
                p.data[:, self.model_dim - self.param_dim :] = 0.0
            for p in self.deviation_tables.values():
                p.data[:, self.model_dim - self.param_dim :] = 0.0

        if self.codec_freeze:
            for mod in list(self.value_encoders.values()) + list(self.value_decoders.values()):
                for p in mod.parameters():
                    p.requires_grad_(False)

        if self.codec_init == "aligned" and self.codec_freeze:
            # Freezing the codec makes the FORWARD pass identical to the dual
            # stream, but not training. The typed architecture is strictly more
            # permissive at layer 0: its centroids and entity deviations are
            # model_dim wide, so gradient can push them into the value slice --
            # somewhere the dual stream's narrower centroids cannot reach. Left
            # alone they drift by ~1e-4 per step and the two architectures
            # separate. Pinning the tail restores an exact identity, which is
            # what makes this configuration usable as a control.
            tail = self.model_dim - self.param_dim

            def _pin_tail(grad: torch.Tensor) -> torch.Tensor:
                g = grad.clone()
                g[..., tail:] = 0.0
                return g

            # Handles kept so the equivalence test can lift the pin and measure
            # what it is worth.
            self._tail_pin_hooks = [
                p.register_hook(_pin_tail)
                for p in list(self.type_embeddings.values())
                + list(self.deviation_tables.values())
            ]

    def _build_initial_typed_stream(
        self,
        graph: EntityGraph,
        device: torch.device,
        status: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Layer 0 for the typed codec, single stream, returned pre-split.

        x = type_centroid + entity_deviation + encoder(value encoding)

        The split into (features, params) at feature_dim is bookkeeping only --
        the block stack immediately concatenates them again. Keeping the same
        signature as the dual builder is what lets the two share every block.

        status is [B, L] for the batched path and None for the single-mask path.
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        gt.refresh(graph)
        B = 1 if status is None else status.shape[0]

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, D]
        x = type_mat.index_select(0, gt.type_ids).unsqueeze(0)                         # [1, L, D]
        if B > 1:
            x = x.expand(B, -1, -1)

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, D]
            dev = dev.unsqueeze(0).expand(B, -1, -1)
            if self.training and variation.dropout_rate > 0:
                keep = (torch.rand(B, dev.shape[1], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            x = x.index_add(1, pos, dev)

        # Hidden tokens present their wildcard encoding, observed ones their
        # one-hot. Same selection rule as the dual stream's value row.
        hidden = (gt.hidden.unsqueeze(0) if status is None
                  else (gt.is_variable.unsqueeze(0) & (status != 2)))
        value = torch.where(hidden.unsqueeze(-1), gt.mask_row, gt.simple_obs)          # [B, L, P]

        for name, enc in self.value_encoders.items():
            sel = (gt.type_ids == gt.type_names.index(name))
            if not bool(sel.any()):
                continue
            pos = sel.nonzero(as_tuple=False).squeeze(-1)
            P = self.types[name].param_dim
            contrib = enc(value.index_select(1, pos)[..., :P])                         # [B, M, D]
            x = x.index_add(1, pos, contrib)

        return x[..., : self.feature_dim], x[..., self.feature_dim :], None

    def _decode_typed(self, combined: torch.Tensor, graph: EntityGraph,
                      device: torch.device) -> torch.Tensor:
        """Per-type decode of the final residual stream.

        Returns the same [B, L, global_param_dim] carrier every other part of the
        codebase expects, with each type's C logits written into slots 1..C --
        the slice the loss and the evaluation path already read. The decoder
        itself is R^d -> R^C; the carrier is only where its output is parked.
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        B, L, _ = combined.shape
        out = torch.zeros(B, L, self.global_param_dim, device=device, dtype=combined.dtype)
        for name, dec in self.value_decoders.items():
            sel = (gt.type_ids == gt.type_names.index(name))
            if not bool(sel.any()):
                continue
            pos = sel.nonzero(as_tuple=False).squeeze(-1)
            logits = dec(combined.index_select(1, pos))                                # [B, M, C]
            carrier = torch.zeros(B, pos.shape[0], self.global_param_dim,
                                  device=device, dtype=combined.dtype)
            carrier[..., 1 : 1 + logits.shape[-1]] = logits
            out = out.index_add(1, pos, carrier)
        return out

    def _forward_typed(self, graph: EntityGraph, device: torch.device,
                       status: torch.Tensor | None = None) -> torch.Tensor:
        features, params, attn_mask = self._build_initial_typed_stream(graph, device, status)
        edge_flat, edge_rel = graph.build_edge_index(device)
        edge_mask = self._dense_edge_mask(graph, device)
        K_aug = self._get_k_aug(graph, device)
        combined = self._run_blocks(features, params, edge_flat, edge_rel,
                                    attn_mask, K_aug, edge_mask)
        return self._decode_typed(combined, graph, device)

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
        self._compiled_blocks = torch.compile(self._dual_blocks, mode=mode, dynamic=False)

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
        core = self._compiled_blocks if self._compiled_blocks is not None else self._dual_blocks
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial feature and param streams from the EntityGraph (dual-stream path).

        Everything here except `status` is fixed by the graph topology, so it is
        precomputed once per graph by GraphTensors and this becomes a handful of
        gathers. See _build_initial_streams_ref for the original per-token loop;
        test_equiv.py asserts the two agree.

        Returns:
            features:  [1, L, feature_dim]
            params:    [1, L, param_dim]
            attn_mask: [1, L]
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        gt.refresh(graph)

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, F]
        features = type_mat.index_select(0, gt.type_ids)                               # [L, F]

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, F]
            if self.training and variation.dropout_rate > 0:
                # One independent Bernoulli draw per token, as in the reference
                # loop -- but as a single kernel instead of M `.item()` syncs.
                keep = (torch.rand(dev.shape[0], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            features = features.index_add(0, pos, dev)

        params = torch.where(gt.hidden.unsqueeze(-1), gt.mask_row, gt.param_obs)       # [L, P]
        # No padding: every token is real. Returning None instead of an all-true
        # mask skips a masked_fill over the [B, H, L, L] score tensor per layer.
        return features.unsqueeze(0), params.unsqueeze(0), None

    def _build_initial_streams_ref(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original per-token implementation. Kept as the equivalence reference."""
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

        if self.value_codec == "typed":
            return self._forward_typed(graph, device)
        return self._forward_dual(graph, device)

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
        if self.value_codec == "typed":
            return self._forward_typed(graph, device, status)

        features, params, attn_mask = self._build_initial_streams_batched(graph, status, device)
        edge_flat, edge_rel = graph.build_edge_index(device)
        edge_mask = self._dense_edge_mask(graph, device)
        K_aug = self._get_k_aug(graph, device)
        return self._run_blocks(features, params, edge_flat, edge_rel, attn_mask, K_aug, edge_mask)

    def _build_initial_streams_batched(
        self,
        graph: EntityGraph,
        status: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched counterpart of _build_initial_streams; see forward_batched."""
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        B, L = status.shape

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, F]
        features = type_mat.index_select(0, gt.type_ids).unsqueeze(0).expand(B, -1, -1)

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, F]
            dev = dev.unsqueeze(0).expand(B, -1, -1)
            if self.training and variation.dropout_rate > 0:
                # Independent draws per augmentation as well as per token.
                keep = (torch.rand(B, dev.shape[1], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            features = features.index_add(1, pos, dev)

        hidden = gt.is_variable.unsqueeze(0) & (status != 2)                           # [B, L]
        params = torch.where(hidden.unsqueeze(-1), gt.mask_row, gt.param_obs)          # [B, L, P]
        return features, params, None                                                  # no padding

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

    def _forward_dual(self, graph: EntityGraph, device: torch.device) -> torch.Tensor:
        """Dual-stream forward pass (original architecture)."""
        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_flat, edge_rel = graph.build_edge_index(device)
        edge_mask = self._dense_edge_mask(graph, device)
        K_aug = self._get_k_aug(graph, device)
        return self._run_blocks(features, params, edge_flat, edge_rel, attn_mask, K_aug, edge_mask)

    def _dual_blocks(
        self,
        features: torch.Tensor,
        params: torch.Tensor,
        edge_flat: torch.Tensor,
        edge_rel: torch.Tensor,
        attn_mask: torch.Tensor | None,
        K_aug: torch.Tensor | None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The block stack, as a pure tensor -> tensor function.

        Split out of _forward_dual so it can be handed to torch.compile: the
        graph-dependent work (which is Python, and cannot be traced) all happens
        before this point, and everything from here is fixed-shape tensor code.
        """
        combined = torch.cat([features, params], dim=-1)  # [1, L, model_dim]
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
            feat_update  = block["dropout_2"](block["proj_out"](combined))
            param_update = block["dropout_2"](block["W_param"](combined))
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

        if self.use_param_output_head:
            return self.param_output_head(combined)
        if self.value_codec == "typed":
            # The decoder reads the whole residual vector, not just the slice the
            # dual stream calls "params".
            return combined
        return params
