from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from imputer.transformer import NormLayer

from .blocks import run_marformer_blocks
from .data import EntityGraph
from .types import EntityType


def init_type_embedding(p: nn.Parameter, mode: str, feature_dim: int) -> None:
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


class MarformerBackbone(nn.Module):
    """
    Shared embeddings, initial streams, pointer cache, and forward orchestration.
    Subclasses implement _transform() over transformer blocks.
    """

    def __init__(
        self,
        config: Any,
        types: Dict[str, EntityType],
        num_relationships: int,
    ):
        super().__init__()
        self.config = config
        self.types = types
        self.num_relationships = num_relationships
        self.use_per_head_rel = config.use_per_head_rel
        self.use_pointer = config.use_pointer
        self.use_rel_value = config.use_rel_value
        self.use_addone_attn = config.use_addone_attn
        self.use_param_output_head = config.use_param_output_head

        self.global_param_dim = max(t.param_dim for t in types.values())
        self.param_dim = self.global_param_dim
        self.model_dim = config.embedding_dim
        assert self.model_dim % config.attention_heads == 0, (
            f"embedding_dim {self.model_dim} must be divisible by "
            f"attention_heads {config.attention_heads}"
        )
        assert self.model_dim > self.param_dim, (
            f"embedding_dim {self.model_dim} must be > global_param_dim {self.param_dim}"
        )
        self.feature_dim = self.model_dim - self.param_dim

        self.type_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, self.feature_dim))
                for name in types.keys()
            }
        )
        for p in self.type_embeddings.values():
            init_type_embedding(p, config.type_embedding_init, self.feature_dim)

        self.deviation_tables = nn.ParameterDict()
        for name, t in types.items():
            if t.variation.enabled and t.variation.num_entities > 0:
                table = nn.Parameter(torch.zeros(t.variation.num_entities, self.feature_dim))
                self.deviation_tables[name] = table

        self.deviation_norm = NormLayer(self.feature_dim)
        if self.use_param_output_head:
            self.param_output_head = nn.Sequential(
                nn.Linear(self.model_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.param_dim),
            )

    def _build_initial_streams(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = graph.num_tokens
        features = torch.zeros(1, L, self.feature_dim, device=device)
        params = torch.zeros(1, L, self.param_dim, device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]
            base = self.type_embeddings[token.type_name]
            feat_vec = base.expand(1, -1)
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
            p = t.build_param(token.raw_data or {}, device=device, global_param_dim=self.param_dim)
            params[0, idx] = p
            attn_mask[0, idx] = True

        return features, params, attn_mask

    def _build_k_aug(self, graph: EntityGraph, device: torch.device) -> torch.Tensor | None:
        if not self.use_pointer:
            return None
        dev_key = str(device)
        if dev_key not in graph._k_aug_cache:
            L = graph.num_tokens
            attr_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            annot_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            item_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            for idx, token in enumerate(graph.tokens):
                if token.type_name in ("rating", "ranking_pairwise") and token.raw_data:
                    attr_ids[idx] = token.raw_data.get("attribute_id", -1)
                    annot_ids[idx] = token.raw_data.get("annotator_id", -1)
                    iids = token.raw_data.get("item_ids", [])
                    item_ids[idx] = iids[0] if iids else -1

            def _same(ids: torch.Tensor) -> torch.Tensor:
                eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
                valid = (ids >= 0).float()
                return eq * valid.unsqueeze(0) * valid.unsqueeze(1)

            graph._k_aug_cache[dev_key] = torch.stack(
                [_same(attr_ids), _same(annot_ids), _same(item_ids)], dim=-1
            )
        return graph._k_aug_cache[dev_key]

    def _transform(
        self,
        combined: torch.Tensor,
        features: torch.Tensor,
        params: torch.Tensor,
        *,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        K_aug: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_mask = graph.build_edge_masks(device=device)
        K_aug = self._build_k_aug(graph, device)

        combined = torch.cat([features, params], dim=-1)
        combined, features, params = self._transform(
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )

        if self.use_param_output_head:
            return self.param_output_head(combined)
        return params
