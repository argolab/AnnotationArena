from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import random

from imputer.data import RankingData


@dataclass
class MCARConfig:
    """Configuration for MCAR masking (Missing Completely At Random)."""

    masking_rate: float = 0.15


@dataclass
class StructuredMaskConfig:
    """
    Configuration for structured masking on real data.

    always_observed_ids: set of annotator IDs that are never masked (e.g. all turker slots).
      Takes priority over llm_annotator_id when set.
    llm_annotator_id: single always-observed annotator ID (legacy; use always_observed_ids
      for multiple IDs).
    human_observed_rate: fraction of non-always-observed annotations to keep observed.
    """

    masking_rate: float = 0.15
    llm_annotator_id: Optional[int] = None
    human_observed_rate: float = 0.0
    always_observed_ids: Optional[List[int]] = None


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


class StructuredMasking(MaskingStrategy):
    """Structured masking: optionally keep LLM annotations observed and mask humans."""

    def __init__(self, config: StructuredMaskConfig):
        self.config = config

    def mask(self, observed_vars: List[RankingData]) -> List[RankingData]:
        if not observed_vars:
            return []

        # Build the set of annotator IDs that are always kept observed.
        # always_observed_ids takes priority; fall back to single llm_annotator_id.
        if self.config.always_observed_ids is not None:
            always_obs: Set[int] = set(self.config.always_observed_ids)
        elif self.config.llm_annotator_id is not None:
            always_obs = {self.config.llm_annotator_id}
        else:
            # Fallback to MCAR when no always-observed annotator is specified.
            print("Fallback to MCAR masking when no always-observed annotator is specified.")
            return MCARMasking(MCARConfig(masking_rate=self.config.masking_rate)).mask(observed_vars)

        # Structured masking: always_obs IDs stay observed; others mostly masked.
        human_vars = [v for v in observed_vars if v.annotator_id not in always_obs]
        num_human_observed = int(len(human_vars) * self.config.human_observed_rate)
        human_observed_indices = (
            set(random.sample(range(len(human_vars)), num_human_observed))
            if num_human_observed > 0
            else set()
        )

        out: List[RankingData] = []
        human_idx = 0
        for var in observed_vars:
            if var.annotator_id in always_obs:
                status = 2  # always observed
            else:
                status = 2 if human_idx in human_observed_indices else 1
                human_idx += 1
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


def build_default_masking_strategy(
    masking_rate: float,
    llm_annotator_id: Optional[int] = None,
    human_observed_rate: float = 0.0,
    always_observed_ids: Optional[List[int]] = None,
) -> MaskingStrategy:
    """
    Factory for the default masking strategy.

    - If always_observed_ids is set: structured masking keeping those annotators observed.
    - Elif llm_annotator_id is set: structured masking keeping that one annotator observed.
    - Else: MCAR.
    """
    if always_observed_ids is not None or llm_annotator_id is not None:
        return StructuredMasking(
            StructuredMaskConfig(
                masking_rate=masking_rate,
                llm_annotator_id=llm_annotator_id,
                human_observed_rate=human_observed_rate,
                always_observed_ids=always_observed_ids,
            )
        )
    return MCARMasking(MCARConfig(masking_rate=masking_rate))

