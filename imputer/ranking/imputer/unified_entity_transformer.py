"""
Unified Entity Transformer — entities as tokens in self-attention sequence.

Jason's proposal: entity tokens participate directly in self-attention alongside
observation tokens, using per-pair edge indicators instead of per-token type flags.

Key differences from EntityBankBlock:
  1. No separate cross-attention — entities are tokens in the sequence
  2. Per-pair edge indicators (6 dims: ATTR, ANNOT, ITEM + reverses) modulate attention
  3. Observations attend to entities, entities attend to observations
  4. No re-composition each layer — tokens evolve through self-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward


class UnifiedEntityBlock(nn.Module):
    """Transformer block for unified entity+observation sequence.

    Sequence structure: [observations (N) | attributes (I) | annotators (J) | items (K)]
    Total length: N + I + J + K

    Per-pair edge indicators (Jason's proposal):
      - 6 binary indicators per (query, key) pair: ATTR, ANNOT, ITEM + reverses
      - Set to 1 iff there is an edge with that label from query to key
      - Q_edge learns how much to weight each edge type in attention scores

    Args:
        feature_dim: Token feature dimension.
        param_dim: Parameter stream dimension (observations only; entities have dummy params).
        attention_heads: Number of self-attention heads.
        dropout: Dropout rate.
        use_gelu_after_attention: Apply GELU after attention (before residual).
        normalize_parameter: If True, normalize entire stream; else normalize feature only.
        num_ffn_layers: Number of FFN layers.
        d_ff: FFN hidden dimension.
    """

    def __init__(
        self,
        feature_dim: int,
        param_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        use_gelu_after_attention: bool = False,
        normalize_parameter: bool = False,
        num_ffn_layers: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.param_dim = param_dim
        self.total_dim = feature_dim + param_dim
        self.attention_heads = attention_heads
        self.use_gelu_after_attention = use_gelu_after_attention
        self.normalize_parameter = normalize_parameter

        # Model dimension for attention (must be divisible by heads)
        if normalize_parameter:
            self.model_dim = int(math.ceil(self.total_dim / attention_heads) * attention_heads)
        else:
            self.model_dim = self.total_dim

        # Projections to/from attention space
        self.proj_in = (nn.Identity() if self.model_dim == self.total_dim
                        else nn.Linear(self.total_dim, self.model_dim))
        self.proj_out = (nn.Identity() if self.model_dim == self.total_dim
                         else nn.Linear(self.model_dim, self.total_dim))

        # Q, K, V projections
        self.Q = nn.Linear(self.model_dim, self.model_dim)
        self.K = nn.Linear(self.model_dim, self.model_dim)
        self.V = nn.Linear(self.model_dim, self.model_dim)
        self.out = nn.Linear(self.model_dim, self.model_dim)

        # Per-pair edge indicator mechanism (6 edge types)
        # Learns per-query weights for each edge type: ATTR, ANNOT, ITEM, REV_ATTR, REV_ANNOT, REV_ITEM
        self.Q_edge = nn.Linear(self.model_dim, 6, bias=False)

        # Normalization
        if normalize_parameter:
            self.norm_1 = NormLayer(self.model_dim)
        else:
            self.norm_1 = NormLayer(self.feature_dim)
            self.param_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.norm_2 = NormLayer(self.model_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        # FFN
        self.ff = FeedForward(self.model_dim, d_ff=d_ff, dropout=dropout, num_layers=num_ffn_layers)

        # Attention stats collection (for debugging)
        self.collect_attention_stats: bool = False
        self.last_attention_stats: dict[str, torch.Tensor] | None = None

    def _multihead_attention(
        self,
        x: torch.Tensor,
        edge_indicators: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with per-pair edge indicators.

        Args:
            x: [B, L, model_dim] — input tokens (L = N + I + J + K)
            edge_indicators: [B, L, L, 6] — binary per-pair edge indicators
            attn_mask: [B, L] — True for valid tokens

        Returns:
            [B, L, model_dim] — attended output
        """
        B, L, _ = x.shape
        H = self.attention_heads

        # Head dimension (allowing for remainder)
        base_head_dim = self.model_dim // H
        remainder = self.model_dim % H
        head_dims = [base_head_dim + (1 if i < remainder else 0) for i in range(H)]

        # Compute Q, K, V
        Q_base = self.Q(x)  # [B, L, model_dim]
        K_base = self.K(x)  # [B, L, model_dim]
        V = self.V(x)       # [B, L, model_dim]

        # Edge contribution from per-pair indicators
        # Q_edge_weights: [B, L, 6] — learned weight per query per edge type
        # edge_indicators: [B, L, L, 6] — binary indicators
        # contribution[b,q,k] = sum_e Q_edge_weights[b,q,e] * edge_indicators[b,q,k,e]
        Q_edge_weights = self.Q_edge(x)  # [B, L, 6]
        edge_contribution = torch.einsum('bqe,bqke->bqk', Q_edge_weights, edge_indicators)

        # Process each head
        attention_outputs = []
        start_idx = 0

        for h in range(H):
            head_dim = head_dims[h]

            # Extract head-specific portions
            Q_h = Q_base[:, :, start_idx:start_idx + head_dim]  # [B, L, head_dim]
            K_h = K_base[:, :, start_idx:start_idx + head_dim]  # [B, L, head_dim]
            V_h = V[:, :, start_idx:start_idx + head_dim]       # [B, L, head_dim]

            # Base attention scores + edge contribution
            scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, L, L]
            scores = scores + edge_contribution

            # Apply attention mask
            if attn_mask is not None:
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.bool()
                key_mask = attn_mask[:, None, :]  # [B, 1, L]
                scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout_1(attn_probs)

            # Apply attention to values
            attended = torch.matmul(attn_probs, V_h)  # [B, L, head_dim]
            attention_outputs.append(attended)

            start_idx += head_dim

        # Concatenate heads
        out = torch.cat(attention_outputs, dim=-1)  # [B, L, model_dim]
        return self.out(out)

    def forward(
        self,
        x: torch.Tensor,
        edge_indicators: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, total_dim] — concatenated [features | params]
            edge_indicators: [B, L, L, 6] — per-pair edge indicators
            attn_mask: [B, L] — True for valid tokens

        Returns:
            [B, L, total_dim] — updated tokens
        """
        B, L, _ = x.shape

        # Project to model_dim if needed
        z = self.proj_in(x)

        # Normalize
        if self.normalize_parameter:
            z_norm = self.norm_1(z)
        else:
            z_norm = torch.cat([
                self.norm_1(z[:, :, :self.feature_dim]),
                self.param_scale * z[:, :, self.feature_dim:]
            ], dim=-1)

        # Attention
        attn_out = self._multihead_attention(z_norm, edge_indicators, attn_mask)

        if self.use_gelu_after_attention:
            attn_out = F.gelu(attn_out)

        # Unscale param part if not normalizing
        if not self.normalize_parameter:
            attn_out_feat = attn_out[:, :, :self.feature_dim]
            attn_out_param = attn_out[:, :, self.feature_dim:] / (self.param_scale + 1e-8)
            attn_out = torch.cat([attn_out_feat, attn_out_param], dim=-1)

        z = z + self.dropout_1(attn_out)

        # FFN
        z_ff = self.ff(self.norm_2(z))
        z = z + z_ff

        # Project back
        out = self.proj_out(z)
        # NOTE: Do NOT zero outputs based on attn_mask. The mask is only for
        # key masking in attention. Missing tokens need their outputs to flow
        # so their params can evolve and produce predictions.

        x = x + self.dropout_2(out)
        return x
