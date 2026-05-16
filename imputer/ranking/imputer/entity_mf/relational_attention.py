from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.k_content_dim = self.head_dim - R
            assert self.k_content_dim > 0, (
                f"head_dim {self.head_dim} must be > num_relationships {R}. "
                f"Increase model_dim or reduce num_heads."
            )
            q_out_dim = model_dim + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, num_heads * self.k_content_dim)
        else:
            self.k_content_dim = self.head_dim
            q_out_dim = model_dim + R + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, model_dim)

        self.V = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        if use_rel_value:
            self.value_rel = nn.Parameter(torch.zeros(R, model_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        R = self.num_relationships
        hd = self.head_dim
        kc = self.k_content_dim
        edge_mask_f = edge_mask.to(x.dtype)

        V_base = self.V(x).view(B, L, H, hd).transpose(1, 2)

        if self.use_per_head_rel:
            Q_proj = self.Q(x)
            Q_full = Q_proj[..., :D].view(B, L, H, hd).transpose(1, 2)
            if self.use_pointer:
                Q_ptr = Q_proj[..., D:]
            K_full = self.K(x).view(B, L, H, kc).transpose(1, 2)
            Q_content = Q_full[..., :kc]
            Q_rel = Q_full[..., kc:]
            content_scores = torch.matmul(Q_content, K_full.transpose(-2, -1))
            rel_scores = torch.einsum("bhir,ijr->bhij", Q_rel, edge_mask_f)
            scores = (content_scores + rel_scores) / math.sqrt(hd)
        else:
            Q_proj = self.Q(x)
            Q_content = Q_proj[..., :D]
            Q_rel_shared = Q_proj[..., D : D + R]
            if self.use_pointer:
                Q_ptr = Q_proj[..., D + R :]
            K_content = self.K(x)
            Qh = Q_content.view(B, L, H, hd).transpose(1, 2)
            Kh = K_content.view(B, L, H, hd).transpose(1, 2)
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(hd)
            rel_scores = (Q_rel_shared.unsqueeze(2) * edge_mask_f.unsqueeze(0)).sum(-1).unsqueeze(1)
            if self.scale_shared_rel:
                rel_scores = rel_scores / math.sqrt(hd)
            scores = content_scores + rel_scores

        if self.use_pointer and K_aug is not None:
            ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)
            scores = scores + ptr_bias.unsqueeze(1)

        if self.use_graph_mask:
            graph_mask = edge_mask.any(dim=-1)
            if K_aug is not None:
                graph_mask = graph_mask | K_aug.any(dim=-1)
            graph_mask = graph_mask | torch.eye(L, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(~graph_mask[None, None], torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            key_mask = attn_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        if self.use_addone_attn:
            scores_shifted = scores - scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores_shifted)
            if attn_mask is not None:
                exp_s = exp_s.masked_fill(~key_mask, 0.0)
            attn = exp_s / (1.0 + exp_s.sum(dim=-1, keepdim=True))
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, V_base)

        if self.use_rel_value:
            attn_mean = attn.mean(dim=1)
            attn_r_mass = torch.einsum("bij,ijr->bir", attn_mean, edge_mask_f)
            bias = attn_r_mass @ self.value_rel
            bias_h = bias.view(B, L, H, hd).permute(0, 2, 1, 3).contiguous()
            out = out + bias_h

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)
