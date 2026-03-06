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
    Relational self-attention where the last R dimensions of the query
    act as relationship-specific weights.

    For each query position i and key position j, we:
      - take the last R dims of Q_i as a length-R vector of relationship logits
      - take edge_mask[i, j, :] as a multi-hot vector over R relationships
      - compute a scalar relational bias by dotting these two vectors
      - add that bias to the base attention score for (i, j) across all heads
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_relationships: int,
        dropout: float,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_relationships = num_relationships

        self.Q = nn.Linear(model_dim, model_dim)
        self.K = nn.Linear(model_dim, model_dim)
        self.V = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] combined feature+param representation.
            edge_mask: [L, L, R] binary edge indicators (batch size 1 assumption).
            attn_mask: [B, L] bool mask for valid tokens.
        """
        B, L, D = x.shape
        H = self.num_heads

        # Simple equal head split; require divisibility.
        head_dim = D // H
        assert head_dim * H == D, f"model_dim {D} must be divisible by num_heads {H}"

        Q = self.Q(x)  # [B, L, D]
        K = self.K(x)
        V = self.V(x)

        R = self.num_relationships
        assert D >= R, f"model_dim {D} must be >= num_relationships {R}"
        assert (D - R) % H == 0, f"(model_dim - num_relationships) must be divisible by num_heads"

        # Base scores use only the first (D - R) dims; last R dims reserved for relational bias.
        Q_base = Q[..., :-R]   # [B, L, D-R]
        K_base = K[..., :-R]   # [B, L, D-R]
        base_head_dim = (D - R) // H
        Qh = Q_base.view(B, L, H, base_head_dim).transpose(1, 2)  # [B, H, L, base_head_dim]
        Kh = K_base.view(B, L, H, base_head_dim).transpose(1, 2)
        Vh = V.view(B, L, H, head_dim).transpose(1, 2)  # V still uses full D

        # Base scores: [B, H, L, L]
        base_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(base_head_dim)

        # Relational bias using last R dims of the queries and a multi-hot edge mask.
        Q_rel = Q[..., -R:]  # [B, L, R]
        # Expand queries and edge mask so each query position i uses Q_rel[:, i, :]
        Q_rel_exp = Q_rel.unsqueeze(2)  # [B, L, 1, R]
        edge_mask_exp = edge_mask.to(Q_rel.dtype).unsqueeze(0)  # [1, L, L, R]
        # Sum over relationship dimension -> [B, L, L]
        rel_scores = (Q_rel_exp * edge_mask_exp).sum(-1)
        # Broadcast over heads: [B, 1, L, L]
        rel_scores = rel_scores.unsqueeze(1)

        scores = base_scores + rel_scores

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            key_mask = attn_mask[:, None, None, :]  # [B, 1, 1, L]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, Vh)  # [B, H, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


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
        global_param_dim: int,
        num_relationships: int,
    ):
        super().__init__()
        self.config = config
        self.types = types
        self.global_param_dim = global_param_dim

        self.feature_dim = config.embedding_dim
        self.param_dim = global_param_dim
        self.model_dim = self.feature_dim + self.param_dim

        # Per-type base embeddings (centroids)
        self.type_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, self.feature_dim))
                for name in types.keys()
            }
        )
        for p in self.type_embeddings.values():
            nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

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
            )
            ff = FeedForward(self.model_dim, d_ff=config.d_ff, dropout=config.dropout, num_layers=config.num_ffn_layers)
            proj_out = nn.Linear(self.model_dim, self.feature_dim)
            W_param = nn.Linear(self.model_dim, self.param_dim)
            blocks.append(
                nn.ModuleDict(
                    {
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
                    feat_vec = feat_vec + dev_table[token.entity_id].unsqueeze(0)
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

        for block in self.blocks:
            combined = torch.cat([features, params], dim=-1)  # [1, L, model_dim]
            attn_out = block["attn"](combined, edge_mask=edge_mask, attn_mask=attn_mask)  # [1, L, model_dim]

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

