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
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1, output_dim: int = None):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model if output_dim is None else output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class TransformerBlock(nn.Module):
    """Unified-stream transformer block over concatenated feature+param with mask support.

    - Multi-head attention operates on the concatenated stream `[features | params]`.
    - No separate feature/param updates; a single FFN processes the unified stream.
    - `attn_mask`: optional bool tensor `[B, N]` where True marks valid tokens.
      Padded tokens neither attend to others nor contribute to outputs.
    """

    def __init__(self, feature_dim: int, param_dim: int, attention_heads: int, dropout: float = 0.3, use_gelu_after_attention: bool = False, normalize_parameter: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.param_dim = param_dim
        self.total_dim = feature_dim + param_dim
        self.attention_heads = attention_heads
        self.use_gelu_after_attention = use_gelu_after_attention
        self.normalize_parameter = normalize_parameter
        # Define an internal model dim that is a multiple of heads for MHAttention

        if normalize_parameter:
            self.model_dim = int(math.ceil(self.total_dim / attention_heads) * attention_heads)
        else:
            self.model_dim = self.total_dim

        # Projections to/from attention space when needed
        self.proj_in = (nn.Identity() if self.model_dim == self.total_dim
                        else nn.Linear(self.total_dim, self.model_dim))
        self.proj_out = (nn.Identity() if self.model_dim == self.total_dim
                         else nn.Linear(self.model_dim, self.total_dim))

        # Unified stream attention over concatenated [feature | param] in model_dim space
        self.Q = nn.Linear(self.model_dim, self.model_dim)
        self.K = nn.Linear(self.model_dim, self.model_dim)
        self.V = nn.Linear(self.model_dim, self.model_dim)
        self.out = nn.Linear(self.model_dim, self.model_dim)
        if normalize_parameter:
            self.norm_1 = NormLayer(self.model_dim)
        else:
            self.norm_1 = NormLayer(self.feature_dim)
        self.norm_2 = NormLayer(self.model_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        # Full FFN on unified stream in model_dim space
        self.ff = FeedForward(self.model_dim, dropout=dropout)

    def _multihead_attention(self, combined_x: torch.Tensor, batch_size: int, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        H = self.attention_heads
        
        # Calculate head dimension, allowing for remainder
        base_head_dim = self.model_dim // H
        remainder = self.model_dim % H
        
        # Create head dimensions - distribute remainder across first few heads
        head_dims = [base_head_dim + (1 if i < remainder else 0) for i in range(H)]
        
        # Split the feature dimension for Q, K, V
        Q = self.Q(combined_x)
        K = self.K(combined_x)
        V = self.V(combined_x)
        
        # Process each head separately since they have different dimensions
        attention_outputs = []
        start_idx = 0
        
        for i in range(H):
            head_dim = head_dims[i]
            
            # Extract head-specific portions
            Q_head = Q[:, :, start_idx:start_idx + head_dim]
            K_head = K[:, :, start_idx:start_idx + head_dim]
            V_head = V[:, :, start_idx:start_idx + head_dim]
            
            # Compute attention scores for this head
            scores = torch.matmul(Q_head, K_head.transpose(-2, -1)) / math.sqrt(head_dim)
            if attn_mask is not None:
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.bool()
                # Mask keys: broadcast [B, N] -> [B, 1, N]
                key_mask = attn_mask[:, None, :]
                scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout_1(scores)
            
            # Apply attention to values
            attended = torch.matmul(scores, V_head)
            attention_outputs.append(attended)
            
            start_idx += head_dim
        
        # Concatenate all head outputs back together
        scores = torch.cat(attention_outputs, dim=-1)
        
        # Apply output projection
        return self.out(scores)

    def forward(self, feature_x: torch.Tensor, param_x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = feature_x.shape[0]

        # Concatenate streams and apply unified attention + FFN with residuals
        combined_total = torch.cat([feature_x, param_x], dim=-1)
        z = self.proj_in(combined_total)

        if self.normalize_parameter:
            z_norm = self.norm_1(z)
        else:
            z_norm = torch.cat((self.norm_1(z[:, :, :self.feature_dim]), z[:, :, self.feature_dim:]), dim=-1)
        attn_out = self._multihead_attention(z_norm, batch_size, attn_mask)

        # Optional GeLU activation after attention (before residual)
        if self.use_gelu_after_attention:
            attn_out = F.gelu(attn_out)

        z = z + self.dropout_1(attn_out)

        z_ff_in = self.norm_2(z)  # pre-norm before feed-forward
        z_ff = self.ff(z_ff_in)
        z = z + z_ff

        back = self.proj_out(z)
        if attn_mask is not None:
            back = back * attn_mask.unsqueeze(-1).to(back.dtype)
        combined_total = combined_total + self.dropout_2(back)

        # Split back to feature and param streams for compatibility
        feature_x = combined_total[:, :, :self.feature_dim]
        param_x = combined_total[:, :, self.feature_dim:]
        return feature_x, param_x
