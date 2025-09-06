import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class TransformerBlock(nn.Module):
    """Single-stream transformer block over feature embeddings only with mask support.

    attn_mask: optional bool tensor `[B, N]` where True marks valid tokens.
    Padded tokens neither attend to others nor contribute to outputs.
    """

    def __init__(self, feature_dim: int, attention_heads: int, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads

        # Feature stream attention
        self.Q = nn.Linear(feature_dim, feature_dim)
        self.K = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

        self.norm_1 = NormLayer(feature_dim)
        self.norm_2 = NormLayer(feature_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.ff = FeedForward(feature_dim, dropout=dropout)

    def _multihead_attention(self, feature_x: torch.Tensor, batch_size: int, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        H = self.attention_heads
        D = self.feature_dim // H
        Q = self.Q(feature_x).view(batch_size, -1, H, D).transpose(1, 2)
        K = self.K(feature_x).view(batch_size, -1, H, D).transpose(1, 2)
        V = self.V(feature_x).view(batch_size, -1, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            # Mask keys: broadcast [B, N] -> [B, 1, 1, N]
            key_mask = attn_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        scores = torch.matmul(scores, V)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        return self.out(scores)

    def forward(self, feature_x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = feature_x.shape[0]

        feature_x_norm = self.norm_1(feature_x)
        attn_out = self._multihead_attention(feature_x_norm, batch_size, attn_mask)
        if attn_mask is not None:
            attn_out = attn_out * attn_mask.unsqueeze(-1).to(attn_out.dtype)
        feature_x = feature_x + self.dropout_1(attn_out)

        feature_x_ff = self.norm_2(feature_x)
        ff_out = self.ff(feature_x_ff)
        if attn_mask is not None:
            ff_out = ff_out * attn_mask.unsqueeze(-1).to(ff_out.dtype)
        feature_x = self.dropout_2(ff_out) + feature_x
        return feature_x
