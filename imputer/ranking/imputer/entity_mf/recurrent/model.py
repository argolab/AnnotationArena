from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..backbone import MarformerBackbone
from ..blocks import build_marformer_block_stack, run_marformer_blocks
from ..data import EntityGraph
from ..types import EntityType
from .config import RecurrentMarformerConfig


class RecurrentEntityMarformer(MarformerBackbone):
    """
    Recurrent Entity Marformer: prelude (unique) -> core (weight-shared, unrolled) -> coda (unique).
    """

    def __init__(
        self,
        config: RecurrentMarformerConfig,
        types: Dict[str, EntityType],
        num_relationships: int,
    ):
        config.validate()
        super().__init__(config, types, num_relationships)
        self.recurrent_config = config

        block_kwargs = dict(
            model_dim=self.model_dim,
            feature_dim=self.feature_dim,
            param_dim=self.param_dim,
            num_relationships=num_relationships,
        )
        self.prelude_blocks = build_marformer_block_stack(
            config, depth=config.prelude_depth, **block_kwargs
        )
        self.core_blocks = build_marformer_block_stack(
            config, depth=config.num_core_layers, **block_kwargs
        )
        self.coda_blocks = build_marformer_block_stack(
            config, depth=config.coda_depth, **block_kwargs
        )
        self._num_recurrence_override: Optional[int] = None

    @property
    def effective_depth(self) -> int:
        return self.recurrent_config.effective_depth

    def _effective_num_recurrence(self) -> int:
        if self._num_recurrence_override is not None:
            return int(self._num_recurrence_override)
        return int(self.recurrent_config.num_recurrence)

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
        num_recurrence: int | None = None,
        *,
        deep_supervision: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._num_recurrence_override = num_recurrence
        try:
            if deep_supervision:
                return self.forward_deep_supervision(graph, device=device)
            return super().forward(graph, device=device)
        finally:
            self._num_recurrence_override = None

    def _params_from_streams(
        self, combined: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        if self.use_param_output_head:
            return self.param_output_head(combined)
        return params

    def forward_deep_supervision(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> List[torch.Tensor]:
        """
        Full unroll; one prediction head per core step (requires coda_depth=0).
        """
        if self.recurrent_config.coda_depth != 0:
            raise ValueError(
                f"forward_deep_supervision requires coda_depth=0, "
                f"got {self.recurrent_config.coda_depth}"
            )
        if device is None:
            device = next(self.parameters()).device

        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_mask = graph.build_edge_masks(device=device)
        K_aug = self._build_k_aug(graph, device)

        combined = torch.cat([features, params], dim=-1)
        combined, features, params = run_marformer_blocks(
            self.prelude_blocks,
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )

        heads: List[torch.Tensor] = []
        for _ in range(self._effective_num_recurrence()):
            combined, features, params = run_marformer_blocks(
                self.core_blocks,
                combined,
                features,
                params,
                edge_mask=edge_mask,
                attn_mask=attn_mask,
                K_aug=K_aug,
            )
            heads.append(self._params_from_streams(combined, params))

        return heads

    def _transform(
        self,
        combined,
        features,
        params,
        *,
        edge_mask,
        attn_mask,
        K_aug,
    ):
        combined, features, params = run_marformer_blocks(
            self.prelude_blocks,
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )
        for _ in range(self._effective_num_recurrence()):
            combined, features, params = run_marformer_blocks(
                self.core_blocks,
                combined,
                features,
                params,
                edge_mask=edge_mask,
                attn_mask=attn_mask,
                K_aug=K_aug,
            )
        combined, features, params = run_marformer_blocks(
            self.coda_blocks,
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )
        return combined, features, params
