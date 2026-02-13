"""
Unified Entity Transformer — entities as tokens in self-attention sequence.

Jason's proposal: entity tokens participate directly in self-attention alongside
observation tokens, using edge-label key dimensions instead of K_aug.

Key differences from EntityBankBlock:
  1. No separate cross-attention — entities are tokens in the sequence
  2. Edge-label key dimensions (3 dims: IS_ATTR, IS_ANNOT, IS_ITEM) in keys
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

    Edge-label key dimensions (Jason's proposal):
      - 3 extra dimensions in keys: IS_ATTR, IS_ANNOT, IS_ITEM
      - These are fixed features based on key position type (not query-dependent)
      - Q learns to attend to these entity type flags

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
        # Q gets +3 extra dimensions for edge queries (IS_ATTR, IS_ANNOT, IS_ITEM)
        # K also gets +3 dimensions to receive fixed edge features from key positions
        self.Q = nn.Linear(self.model_dim, self.model_dim + 3)
        self.K = nn.Linear(self.model_dim, self.model_dim + 3)
        self.V = nn.Linear(self.model_dim, self.model_dim)
        self.out = nn.Linear(self.model_dim, self.model_dim)

        # Pointer mechanism for SAME_ATTR/SAME_ANNOT/SAME_ITEM between observations
        self.Q_ptr = nn.Linear(self.model_dim, 3, bias=False)

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
        key_edge_features: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
        num_obs: int = 0,
    ) -> torch.Tensor:
        """Multi-head attention with edge-label key dimensions and pointer mechanism.

        Args:
            x: [B, L, model_dim] — input tokens (L = N + I + J + K)
            key_edge_features: [B, L, 3] — fixed edge features for keys (IS_ATTR, IS_ANNOT, IS_ITEM)
            attn_mask: [B, L] — True for valid tokens
            K_aug: [B, N, N, 3] — pairwise SAME_ATTR/SAME_ANNOT/SAME_ITEM indicators (obs-to-obs)
            num_obs: number of observation tokens (first num_obs positions in sequence)

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
        Q_full = self.Q(x)  # [B, L, model_dim + 3]
        K_full = self.K(x)  # [B, L, model_dim + 3]
        V = self.V(x)  # [B, L, model_dim]

        # Split Q into base and edge parts
        Q_base = Q_full[:, :, :self.model_dim]  # [B, L, model_dim]
        Q_edge = Q_full[:, :, self.model_dim:]  # [B, L, 3] - learned edge query

        # Split K into base and edge parts
        K_base = K_full[:, :, :self.model_dim]  # [B, L, model_dim]
        # Replace learned K edge with fixed key edge features (IS_ATTR, IS_ANNOT, IS_ITEM)
        K_edge = key_edge_features  # [B, L, 3]

        # Compute edge contribution: Q_edge @ K_edge^T
        # Q_edge: [B, L, 3], K_edge: [B, L, 3]
        # Result: [B, L, L] where [b,i,j] = Q_edge[b,i,:] @ K_edge[b,j,:]
        edge_contribution = torch.einsum('bqe,bke->bqk', Q_edge, K_edge)

        # Pointer mechanism: SAME_ATTR/SAME_ANNOT/SAME_ITEM between observations
        if K_aug is not None and num_obs > 0:
            Q_ptr = self.Q_ptr(Q_base[:, :num_obs, :])  # [B, N, 3]
            ptr_additions = (Q_ptr.unsqueeze(2) * K_aug).sum(dim=-1)  # [B, N, N]
        else:
            ptr_additions = None

        # Process each head
        attention_outputs = []
        start_idx = 0

        for h in range(H):
            head_dim = head_dims[h]

            # Extract head-specific portions
            Q_h = Q_base[:, :, start_idx:start_idx + head_dim]  # [B, L, head_dim]
            K_h = K_base[:, :, start_idx:start_idx + head_dim]  # [B, L, head_dim]
            V_h = V[:, :, start_idx:start_idx + head_dim]  # [B, L, head_dim]

            # Base attention scores + edge contribution
            scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, L, L]
            scores = scores + edge_contribution

            # Add pointer mechanism contribution for observation-to-observation pairs
            if ptr_additions is not None:
                scores[:, :num_obs, :num_obs] = scores[:, :num_obs, :num_obs] + ptr_additions

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
        key_edge_features: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
        num_obs: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, total_dim] — concatenated [features | params]
            key_edge_features: [B, L, 3] — fixed edge features for keys (IS_ATTR, IS_ANNOT, IS_ITEM)
            attn_mask: [B, L] — True for valid tokens
            K_aug: [B, N, N, 3] — pairwise obs-to-obs indicators (passed to attention)
            num_obs: number of observation tokens

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
        attn_out = self._multihead_attention(z_norm, key_edge_features, attn_mask, K_aug=K_aug, num_obs=num_obs)

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
