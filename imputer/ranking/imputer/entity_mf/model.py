from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward

from .config import EntityMarformerConfig
from .data import EntityGraph
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
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_relationships = num_relationships
        self.use_per_head_rel = use_per_head_rel
        self.use_pointer = use_pointer
        self.use_rel_value = use_rel_value
        self.use_addone_attn = use_addone_attn
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

        # Relation-specific value embeddings: e_r per head, init to zero (no-op at start)
        if use_rel_value:
            self.rel_value_emb = nn.Parameter(
                torch.zeros(num_heads, R, self.head_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, L, D] combined feature+param representation.
            edge_mask: [L, L, R] binary edge indicators.
            attn_mask: [B, L] bool mask for valid tokens.
            K_aug:     [L, L, 3] obs-obs shared-identity indicators (optional, for pointer).
        """
        B, L, D = x.shape
        H = self.num_heads
        R = self.num_relationships
        hd = self.head_dim
        kc = self.k_content_dim
        edge_mask_f = edge_mask.to(x.dtype)                        # [L, L, R]

        V_base = self.V(x).view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]

        if self.use_per_head_rel:
            # ── Per-head relational attention ───────────────────────────────────
            # Q: D -> D (+ 3 for pointer); split per head into [content | rel]
            Q_proj = self.Q(x)                                     # [B, L, D] or [B, L, D+3]
            Q_full = Q_proj[..., :D].view(B, L, H, hd).transpose(1, 2)  # [B, H, L, hd]
            if self.use_pointer:
                Q_ptr = Q_proj[..., D:]                            # [B, L, 3]
            K_full = self.K(x).view(B, L, H, kc).transpose(1, 2) # [B, H, L, kc]
            Q_content = Q_full[..., :kc]                          # [B, H, L, kc]
            Q_rel     = Q_full[..., kc:]                          # [B, H, L, R]
            content_scores = torch.matmul(Q_content, K_full.transpose(-2, -1))       # [B, H, L, L]
            rel_scores     = torch.einsum("bhir,ijr->bhij", Q_rel, edge_mask_f)      # [B, H, L, L]
            scores = (content_scores + rel_scores) / math.sqrt(hd)
        else:
            # ── Shared-bias relational attention (old design) ───────────────────
            # Q: D -> D+R (+ 3 for pointer); K: D -> D; rel bias shared across heads
            Q_proj = self.Q(x)                                     # [B, L, D+R] or [B, L, D+R+3]
            Q_content = Q_proj[..., :D]                            # [B, L, D]
            Q_rel_shared = Q_proj[..., D:D+R]                     # [B, L, R]  (shared across heads)
            if self.use_pointer:
                Q_ptr = Q_proj[..., D+R:]                          # [B, L, 3]
            K_content = self.K(x)                                  # [B, L, D]
            Qh = Q_content.view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]
            Kh = K_content.view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(hd) # [B, H, L, L]
            # Relational bias: [B, L, L] -> broadcast over heads [B, 1, L, L]
            rel_scores = (Q_rel_shared.unsqueeze(2) * edge_mask_f.unsqueeze(0)).sum(-1).unsqueeze(1)
            scores = content_scores + rel_scores

        # ── Pointer bias (shared across heads) ─────────────────────────────────
        if self.use_pointer and K_aug is not None:
            # Q_ptr: [B, L, 3], K_aug: [L, L, 3]
            ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)  # [B, L, L]
            scores = scores + ptr_bias.unsqueeze(1)                        # broadcast over H

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
            # attn_r_mass[b, h, i, r] = sum_j attn[b,h,i,j] * edge_mask[i,j,r]
            attn_r_mass = torch.einsum("bhij, ijr -> bhir", attn, edge_mask_f)  # [B, H, L, R]
            # rel_aug[b, h, i, :] = sum_r attn_r_mass[b,h,i,r] * rel_value_emb[h, r, :]
            rel_aug = torch.einsum("bhir, hrd -> bhid", attn_r_mass, self.rel_value_emb)  # [B, H, L, hd]
            out = out + rel_aug

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
    Minimal viable Entity Marformer with:
      - feature stream updated by one FFN
      - param stream updated by a separate FFN
      - relational attention over a unified token sequence.
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

        # Global param dim = max over types (no longer passed in).
        self.global_param_dim = max(t.param_dim for t in types.values())
        self.param_dim = self.global_param_dim

        # Config embedding_dim = total model dim (feature + param). Must be divisible by heads.
        self.model_dim = config.embedding_dim
        assert self.model_dim % config.attention_heads == 0, (
            f"embedding_dim {self.model_dim} must be divisible by attention_heads {config.attention_heads}"
        )
        assert self.model_dim > self.param_dim, (
            f"embedding_dim {self.model_dim} must be > global_param_dim {self.param_dim}"
        )
        self.feature_dim = self.model_dim - self.param_dim

        # Per-type base embeddings (centroids)
        self.type_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, self.feature_dim))
                for name in types.keys()
            }
        )
        for p in self.type_embeddings.values():
            _init_type_embedding(p, config.type_embedding_init, self.feature_dim)

        # Per-entity deviations/variations (where enabled)
        self.deviation_tables = nn.ParameterDict()
        for name, t in types.items():
            if t.variation.enabled and t.variation.num_entities > 0:
                table = nn.Parameter(torch.zeros(t.variation.num_entities, self.feature_dim))
                self.deviation_tables[name] = table

        self.deviation_norm = NormLayer(self.feature_dim)

        # Transformer blocks: one FFN on combined stream, then project back to feature/param
        # (same pattern as current imputer so information flows between streams in the FFN)
        blocks: List[nn.Module] = []
        for _ in range(config.num_layers):
            attn = RelationalAttentionBlock(
                model_dim=self.model_dim,
                num_heads=config.attention_heads,
                num_relationships=num_relationships,
                dropout=config.dropout,
                use_per_head_rel=config.use_per_head_rel,
                use_pointer=config.use_pointer,
                use_rel_value=config.use_rel_value,
                use_addone_attn=config.use_addone_attn,
            )
            ff = FeedForward(
                self.model_dim,
                d_ff=config.d_ff,
                dropout=config.dropout,
                num_layers=config.num_ffn_layers,
            )
            proj_out = nn.Linear(self.model_dim, self.feature_dim)
            W_param = nn.Linear(self.model_dim, self.param_dim)
            blocks.append(
                nn.ModuleDict(
                    {
                        "norm_1": NormLayer(self.model_dim),
                        "attn": attn,
                        "norm_2": NormLayer(self.model_dim),
                        "ff": ff,
                        "proj_out": proj_out,
                        "W_param": W_param,
                        "dropout_2": nn.Dropout(config.dropout),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def _build_initial_streams(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial feature and param streams from the EntityGraph.

        Returns:
            features: [1, L, D_feat]
            params: [1, L, D_param]
            attn_mask: [1, L] (True for valid tokens)
        """
        L = graph.num_tokens
        features = torch.zeros(1, L, self.feature_dim, device=device)
        params = torch.zeros(1, L, self.param_dim, device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]

            base = self.type_embeddings[token.type_name]  # [1, D]
            feat_vec = base.expand(1, -1)  # [1, D]
            if token.type_name in self.deviation_tables and token.entity_id >= 0:
                dev_table = self.deviation_tables[token.type_name]
                if 0 <= token.entity_id < dev_table.shape[0]:
                    dev = dev_table[token.entity_id].unsqueeze(0)
                    # Deviation dropout: drop with probability dropout_rate during training
                    if self.training and t.variation.dropout_rate > 0:
                        if torch.rand(1).item() < t.variation.dropout_rate:
                            dev = torch.zeros_like(dev)
                    feat_vec = feat_vec + dev
            features[0, idx] = feat_vec

            p = t.build_param(token.raw_data or {}, device=device, global_param_dim=self.param_dim)
            params[0, idx] = p

            # All tokens are currently valid for attention; we may later mask permanently-missing ones.
            attn_mask[0, idx] = True

        return features, params, attn_mask

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over a single EntityGraph.

        Returns:
            params: [1, L, D_param] final parameter stream.
        """
        if device is None:
            device = next(self.parameters()).device

        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_mask = graph.build_edge_masks(device=device)  # [L, L, R]

        # Build K_aug for pointer mechanism: [L, L, 3] obs-obs shared-identity indicators
        K_aug: torch.Tensor | None = None
        if self.use_pointer:
            L = graph.num_tokens
            attr_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            annot_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            item_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            for idx, token in enumerate(graph.tokens):
                if token.type_name in ("rating", "ranking_pairwise") and token.raw_data:
                    attr_ids[idx]  = token.raw_data.get("attribute_id", -1)
                    annot_ids[idx] = token.raw_data.get("annotator_id", -1)
                    # Use first item_id (like Marformer pointer)
                    iids = token.raw_data.get("item_ids", [])
                    item_ids[idx]  = iids[0] if iids else -1
            # [L, L] indicator: 1 if same id AND both valid (id >= 0)
            def _same(ids: torch.Tensor) -> torch.Tensor:
                eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()  # [L, L]
                valid = (ids >= 0).float()
                return eq * valid.unsqueeze(0) * valid.unsqueeze(1)
            K_aug = torch.stack([_same(attr_ids), _same(annot_ids), _same(item_ids)], dim=-1)  # [L, L, 3]

        for block in self.blocks:
            combined = torch.cat([features, params], dim=-1)  # [1, L, model_dim]
            # Pre-LN: norm before attention
            attn_out = block["attn"](
                block["norm_1"](combined),
                edge_mask=edge_mask,
                attn_mask=attn_mask,
                K_aug=K_aug,
            )  # [1, L, model_dim]

            combined = combined + attn_out

            # Single FFN on combined stream
            z_ff = block["ff"](block["norm_2"](combined))
            combined = combined + z_ff

            # Project back to the two streams and add as residuals
            back_feat = block["proj_out"](combined)   # [1, L, feature_dim]
            back_param = block["W_param"](combined)   # [1, L, param_dim]
            features = features + block["dropout_2"](back_feat)
            params = params + block["dropout_2"](back_param)

        return params

