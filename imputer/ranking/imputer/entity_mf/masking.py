from __future__ import annotations

from dataclasses import dataclass
from typing import List

import random

from imputer.data import RankingData


@dataclass
class MCARConfig:
    """Configuration for MCAR masking (Missing Completely At Random)."""

    masking_rate: float = 0.15


class MaskingStrategy:
    """
    Abstract masking strategy interface.

    For now, only MCAR is implemented; this class mainly serves as a
    lightweight hook so we can plug in alternative strategies later.
    """

    def mask(self, observed_vars: List[RankingData]) -> List[RankingData]:
        raise NotImplementedError


class MCARMasking(MaskingStrategy):
    """MCAR masking: independently mask a random subset of observed variables."""

    def __init__(self, config: MCARConfig):
        self.config = config

    def mask(self, observed_vars: List[RankingData]) -> List[RankingData]:
        if not observed_vars:
            return []

        masking_rate = self.config.masking_rate
        num_to_mask = int(len(observed_vars) * masking_rate)
        num_to_mask = max(0, min(len(observed_vars), num_to_mask))
        masked_indices = set(random.sample(range(len(observed_vars)), num_to_mask)) if num_to_mask > 0 else set()

        out: List[RankingData] = []
        for idx, var in enumerate(observed_vars):
            status = 1 if idx in masked_indices else 2  # 1=masked, 2=observed
            out.append(
                RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    status=status,
                    instance=var.instance,
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order,
                    rating_dist=var.rating_dist,
                )
            )
        return out


def build_default_masking_strategy(masking_rate: float) -> MaskingStrategy:
    """
    Factory for the default masking strategy.

    Right now it's MCAR with a given masking_rate, but this can later branch
    on a config enum to support other strategies.
    """
    return MCARMasking(MCARConfig(masking_rate=masking_rate))

