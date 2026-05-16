from __future__ import annotations

from typing import Dict

from ..backbone import MarformerBackbone
from ..blocks import build_marformer_block_stack, run_marformer_blocks
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

    @property
    def effective_depth(self) -> int:
        return self.recurrent_config.effective_depth

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
        for _ in range(self.recurrent_config.num_recurrence):
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
