from __future__ import annotations

import math
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward

from .relational_attention import RelationalAttentionBlock


def build_marformer_block(
    config: Any,
    *,
    model_dim: int,
    feature_dim: int,
    param_dim: int,
    num_relationships: int,
) -> nn.ModuleDict:
    """Build one Pre-LN relational transformer block (attention + FFN + stream residuals)."""
    attn = RelationalAttentionBlock(
        model_dim=model_dim,
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
    ff = FeedForward(
        model_dim,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_layers=config.num_ffn_layers,
    )
    return nn.ModuleDict(
        {
            "norm_1": NormLayer(model_dim),
            "attn": attn,
            "norm_2": NormLayer(model_dim),
            "ff": ff,
            "proj_out": nn.Linear(model_dim, feature_dim),
            "W_param": nn.Linear(model_dim, param_dim),
            "dropout_2": nn.Dropout(config.dropout),
        }
    )


def forward_marformer_block(
    block: nn.ModuleDict,
    combined: torch.Tensor,
    features: torch.Tensor,
    params: torch.Tensor,
    *,
    edge_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    K_aug: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one block; update combined, feature, and param streams."""
    normed_attn = block["norm_1"](combined)
    attn_out = block["attn"](
        normed_attn,
        edge_mask=edge_mask,
        attn_mask=attn_mask,
        K_aug=K_aug,
    )
    combined = combined + attn_out

    normed_ff = block["norm_2"](combined)
    z_ff = block["ff"](normed_ff)
    combined = combined + z_ff

    back_feat = block["proj_out"](combined)
    features = features + block["dropout_2"](back_feat)
    back_param = block["W_param"](combined)
    params = params + block["dropout_2"](back_param)
    combined = torch.cat([features, params], dim=-1)
    return combined, features, params


def run_marformer_blocks(
    blocks: Iterable[nn.ModuleDict],
    combined: torch.Tensor,
    features: torch.Tensor,
    params: torch.Tensor,
    *,
    edge_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    K_aug: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for block in blocks:
        combined, features, params = forward_marformer_block(
            block,
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )
    return combined, features, params


def build_marformer_block_stack(
    config: Any,
    *,
    depth: int,
    model_dim: int,
    feature_dim: int,
    param_dim: int,
    num_relationships: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            build_marformer_block(
                config,
                model_dim=model_dim,
                feature_dim=feature_dim,
                param_dim=param_dim,
                num_relationships=num_relationships,
            )
            for _ in range(depth)
        ]
    )
