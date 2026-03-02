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
    def __init__(
        self,
        d_model: int,
        d_ff: int = 512,
        dropout: float = 0.1,
        output_dim: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()

        # Build a deeper MLP with ReLU non-linearities.
        # Structure: [d_model -> d_ff] + (num_layers-2) x [d_ff -> d_ff] + [d_ff -> out_dim]
        out_dim = d_model if output_dim is None else output_dim
        hidden_layers = max(1, num_layers - 1)

        layers: list[nn.Module] = []

        # First layer projects from input to hidden width
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))

        # Middle hidden layers (if any)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # Final projection to output dimension (no activation)
        layers.append(nn.Linear(d_ff, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Dual-stream transformer block: features projected into attention space, combined with params.

    Architecture per block:
      1. proj_in: Linear(feature_dim, feat_proj_dim)  — project features only
      2. z = cat([proj_in(features), params])          — combined at model_dim
      3. Attention + FFN on z (combined-over-combined)
      4. proj_out: Linear(model_dim, feature_dim)      — explicit feature outer residual
      5. W_param:  Linear(model_dim, param_dim)        — explicit param outer residual

    feat_proj_dim = model_dim - param_dim, so combined = model_dim (divisible by heads).
    `attn_mask`: optional bool tensor `[B, N]` where True marks valid tokens.
    """

    def __init__(self, feature_dim: int, param_dim: int, attention_heads: int, dropout: float = 0.3, use_gelu_after_attention: bool = False,
                 normalize_parameter: bool = False, num_ffn_layers: int = 4, enable_pointer_mechanism: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.param_dim = param_dim
        self.total_dim = feature_dim + param_dim
        self.attention_heads = attention_heads
        self.use_gelu_after_attention = use_gelu_after_attention
        self.normalize_parameter = normalize_parameter
        self.num_ffn_layers = num_ffn_layers
        self.enable_pointer_mechanism = enable_pointer_mechanism
        # model_dim must be divisible by heads; combined = cat([proj_features, params]) = model_dim

        if normalize_parameter:
            self.model_dim = int(math.ceil(self.total_dim / attention_heads) * attention_heads)
        else:
            self.model_dim = self.total_dim

        # feat_proj_dim: feature projection target so that cat([proj_features, params]) = model_dim
        # When normalize_parameter=False: model_dim = total_dim, feat_proj_dim = feature_dim (square proj)
        # When normalize_parameter=True:  model_dim > total_dim, feat_proj_dim = model_dim - param_dim
        self.feat_proj_dim = self.model_dim - self.param_dim

        # proj_in: project features only into attention space
        self.proj_in = nn.Linear(self.feature_dim, self.feat_proj_dim)
        # proj_out: feature stream outer residual (model_dim → feature_dim)
        self.proj_out = nn.Linear(self.model_dim, self.feature_dim)
        # W_param: explicit param stream outer residual (model_dim → param_dim)
        self.W_param = nn.Linear(self.model_dim, self.param_dim)

        # Attention on combined [proj_features | params] at model_dim
        # Q gets +3 dimensions for pointer mechanism, K and V stay same size
        if enable_pointer_mechanism:
            self.Q = nn.Linear(self.model_dim, self.model_dim + 3)
        else:
            self.Q = nn.Linear(self.model_dim, self.model_dim)
        self.K = nn.Linear(self.model_dim, self.model_dim)
        self.V = nn.Linear(self.model_dim, self.model_dim)
        self.out = nn.Linear(self.model_dim, self.model_dim)
        if normalize_parameter:
            self.norm_1 = NormLayer(self.model_dim)
        else:
            # Normalize projected features only; scale param stream separately
            self.norm_1 = NormLayer(self.feat_proj_dim)
            self.param_scale = nn.Parameter(torch.ones(1) * 0.01)  # Small initial scale
        self.norm_2 = NormLayer(self.model_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        # Full FFN on unified stream in model_dim space
        self.ff = FeedForward(self.model_dim, dropout=dropout, num_layers=self.num_ffn_layers)

        # Optional: collect attention distribution statistics for logging/debugging.
        # When enabled, the latest forward pass stats are stored in `last_attention_stats`.
        self.collect_attention_stats: bool = False
        self.last_attention_stats: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        return (x * mask_f).sum() / denom

    def _attention_stats(self, attn_probs: torch.Tensor, attn_mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Compute summary stats over attention probabilities [B, N, N]."""
        eps = 1e-12
        B, N, _ = attn_probs.shape
        if attn_mask is None:
            query_mask = torch.ones((B, N), dtype=torch.bool, device=attn_probs.device)
            valid_fraction = torch.tensor(1.0, device=attn_probs.device)
            valid_tokens_mean = torch.tensor(float(N), device=attn_probs.device)
        else:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            query_mask = attn_mask
            valid_fraction = attn_mask.to(dtype=attn_probs.dtype).mean()
            valid_tokens_mean = attn_mask.to(dtype=attn_probs.dtype).sum(dim=-1).mean()

        # Query-level distributions over keys.
        entropy = -(attn_probs * torch.log(attn_probs.clamp_min(eps))).sum(dim=-1)  # [B, N]
        max_prob = attn_probs.max(dim=-1).values  # [B, N]
        k = min(5, N)
        topk_mass = attn_probs.topk(k, dim=-1).values.sum(dim=-1)  # [B, N]
        effective_n = 1.0 / attn_probs.pow(2).sum(dim=-1).clamp_min(eps)  # [B, N]
        frac_gt_1pct = (attn_probs > 0.01).to(dtype=attn_probs.dtype).mean(dim=-1)  # [B, N]

        return {
            "valid_fraction": valid_fraction.detach(),
            "valid_tokens_mean": valid_tokens_mean.detach(),
            "entropy_mean": self._masked_mean(entropy, query_mask).detach(),
            "max_prob_mean": self._masked_mean(max_prob, query_mask).detach(),
            "top5_mass_mean": self._masked_mean(topk_mass, query_mask).detach(),
            "effective_n_mean": self._masked_mean(effective_n, query_mask).detach(),
            "frac_gt_0p01_mean": self._masked_mean(frac_gt_1pct, query_mask).detach(),
        }

    def _multihead_attention(self, combined_x: torch.Tensor, batch_size: int, attn_mask: torch.Tensor | None = None, K_aug: torch.Tensor | None = None) -> torch.Tensor:
        H = self.attention_heads
        
        # Calculate head dimension, allowing for remainder
        base_head_dim = self.model_dim // H
        remainder = self.model_dim % H
        
        # Create head dimensions - distribute remainder across first few heads
        head_dims = [base_head_dim + (1 if i < remainder else 0) for i in range(H)]
        
        # Compute Q, K, V projections
        Q_full = self.Q(combined_x)  # [B, N, model_dim + 3] if pointer enabled, else [B, N, model_dim]
        K = self.K(combined_x)  # [B, N, model_dim]
        V = self.V(combined_x)  # [B, N, model_dim]
        
        # Split Q into base and pointer parts if pointer mechanism is enabled
        if self.enable_pointer_mechanism:
            Q_base = Q_full[:, :, :self.model_dim]  # [B, N, model_dim]
            Q_ptr = Q_full[:, :, self.model_dim:]  # [B, N, 3]
            assert Q_ptr.shape[-1] == 3, "Q_ptr should have 3 dimensions"
        else:
            Q_base = Q_full
            Q_ptr = None
        
        # Process each head separately since they have different dimensions
        attention_outputs = []
        start_idx = 0
        attn_stats_sum: dict[str, torch.Tensor] | None = {} if self.collect_attention_stats else None
        
        for i in range(H):
            head_dim = head_dims[i]
            
            # Extract head-specific portions
            Q_head = Q_base[:, :, start_idx:start_idx + head_dim]  # [B, N, head_dim]
            K_head = K[:, :, start_idx:start_idx + head_dim]  # [B, N, head_dim]
            V_head = V[:, :, start_idx:start_idx + head_dim]  # [B, N, head_dim]
            
            # Compute base attention scores for this head
            raw_scores = torch.matmul(Q_head, K_head.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, N, N]
            
            # Add pointer mechanism if enabled
            if self.enable_pointer_mechanism and K_aug is not None:
                # K_aug is [B, N, N, 3] where [b, i, j, :] = [attr_match, annot_match, item_match] for query i and key j
                # Q_ptr is [B, N, 3]
                # We want: ptr_additions[b, i, j] = Q_ptr[b, i] @ K_aug[b, i, j]
                # This is: (Q_ptr.unsqueeze(2) * K_aug).sum(dim=-1) where Q_ptr is [B, N, 1, 3] and K_aug is [B, N, N, 3]
                ptr_additions = (Q_ptr.unsqueeze(2) * K_aug).sum(dim=-1)  # [B, N, N]
                raw_scores = raw_scores + ptr_additions
            
            if attn_mask is not None:
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.bool()
                # Mask keys: broadcast [B, N] -> [B, 1, N]
                key_mask = attn_mask[:, None, :]
                raw_scores = raw_scores.masked_fill(~key_mask, torch.finfo(raw_scores.dtype).min)
            attn_probs = F.softmax(raw_scores, dim=-1)

            if attn_stats_sum is not None:
                with torch.no_grad():
                    head_stats = self._attention_stats(attn_probs, attn_mask)
                    for k, v in head_stats.items():
                        attn_stats_sum[k] = attn_stats_sum.get(k, torch.zeros_like(v)) + v

            attn_probs = self.dropout_1(attn_probs)
            
            # Apply attention to values
            attended = torch.matmul(attn_probs, V_head)
            attention_outputs.append(attended)
            
            start_idx += head_dim
        
        # Concatenate all head outputs back together
        scores = torch.cat(attention_outputs, dim=-1)

        # Apply output projection
        if attn_stats_sum is not None:
            self.last_attention_stats = {k: (v / float(H)).detach() for k, v in attn_stats_sum.items()}
        return self.out(scores)

    def forward(self, feature_x: torch.Tensor, param_x: torch.Tensor, attn_mask: torch.Tensor | None = None, K_aug: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = feature_x.shape[0]

        # Project features only, then combine with params to form model_dim combined representation
        z = torch.cat([self.proj_in(feature_x), param_x], dim=-1)  # [B, N, model_dim]

        if self.normalize_parameter:
            z_norm = self.norm_1(z)
        else:
            # Normalize projected feature part; scale raw param stream to balance magnitudes
            z_norm = torch.cat((self.norm_1(z[:, :, :self.feat_proj_dim]),
                                self.param_scale * z[:, :, self.feat_proj_dim:]), dim=-1)
        attn_out = self._multihead_attention(z_norm, batch_size, attn_mask, K_aug=K_aug)

        # Unscale parameter portion of attention output before adding residual to z
        if not self.normalize_parameter:
            attn_out_feat = attn_out[:, :, :self.feat_proj_dim]
            attn_out_param = attn_out[:, :, self.feat_proj_dim:]
            attn_out_param_unscaled = attn_out_param / (self.param_scale + 1e-8)
            attn_out = torch.cat([attn_out_feat, attn_out_param_unscaled], dim=-1)

        z = z + self.dropout_1(attn_out)

        z_ff_in = self.norm_2(z)  # pre-norm before feed-forward
        z_ff = self.ff(z_ff_in)
        z = z + z_ff

        # Explicit outer residuals: dedicated projections for each stream
        back_feat = self.proj_out(z)   # [B, N, feature_dim]
        back_param = self.W_param(z)   # [B, N, param_dim]
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).to(back_feat.dtype)
            back_feat = back_feat * mask
            back_param = back_param * mask
        feature_x = feature_x + self.dropout_2(back_feat)
        param_x = param_x + self.dropout_2(back_param)
        return feature_x, param_x
