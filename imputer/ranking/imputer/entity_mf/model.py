from __future__ import annotations

from typing import Dict

from .backbone import MarformerBackbone
from .blocks import build_marformer_block_stack, run_marformer_blocks
from .config import EntityMarformerConfig
from .types import EntityType


class EntityMarformer(MarformerBackbone):
    """
    Flat-stack Entity Marformer: num_layers unique transformer blocks.
    """

    def __init__(
        self,
        config: EntityMarformerConfig,
        types: Dict[str, EntityType],
        num_relationships: int,
    ):
        super().__init__(config, types, num_relationships)
        self.blocks = build_marformer_block_stack(
            config,
            depth=config.num_layers,
            model_dim=self.model_dim,
            feature_dim=self.feature_dim,
            param_dim=self.param_dim,
            num_relationships=num_relationships,
        )

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
        return run_marformer_blocks(
            self.blocks,
            combined,
            features,
            params,
            edge_mask=edge_mask,
            attn_mask=attn_mask,
            K_aug=K_aug,
        )


# Re-export for backward compatibility.
from .relational_attention import RelationalAttentionBlock  # noqa: E402,F401
