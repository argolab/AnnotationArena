from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.losses import PlackettLuceLoss


@dataclass
class LossBreakdown:
    """Per-type loss breakdown by status. trainable_loss is masked-only (for backprop); others for metrics."""

    trainable_loss: torch.Tensor  # scalar, masked only
    loss_observed: float
    loss_masked: float
    loss_missing: float
    n_observed: int
    n_masked: int
    n_missing: int


@dataclass
class VariationConfig:
    """Configuration for per-entity deviation embeddings."""

    enabled: bool = False
    num_entities: int = 0  # number of entities of this type (define the size of the variational matrix)
    reg_weight: float = 0.0  # L2 regularization weight for deviations


class EntityType(ABC):
    """
    Abstract base class for all entity/variable types.

    Two major roles:
      1) Input adapter: encode raw observation into parameter-stream vector.
      2) Loss adapter: compute loss on predicted parameters for tokens of this type.
    """

    name: str
    is_variable: bool
    param_dim: int
    variation: VariationConfig

    def __init__(self, name: str, is_variable: bool, param_dim: int, variation: VariationConfig | None = None):
        self.name = name
        self.is_variable = is_variable
        self.param_dim = param_dim
        self.variation = variation or VariationConfig(enabled=False, num_entities=0, reg_weight=0.0)

    @abstractmethod
    def build_param(self, raw_data: Dict[str, Any], device: torch.device, global_param_dim: int) -> torch.Tensor:
        """
        Convert raw observation payload into a parameter-stream vector. (Single token)

        Args:
            raw_data: Dict payload stored on the token (type-specific).
            device: Torch device.
            global_param_dim: Global parameter dimension used by the model.

        Returns:
            Tensor of shape [global_param_dim].
        """

    @abstractmethod
    def compute_loss(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> torch.Tensor:
        """
        Compute loss for tokens of this type. (Full input batch)

        Args:
            predicted_params: [B, L, global_param_dim] tensor from the model.
            tokens: Flat list of all tokens (length L).
            type_mask: Bool tensor [B, L] marking which positions are this type.
            global_param_dim: Global parameter dimension.

        Returns:
            Scalar loss tensor (on same device as predicted_params).
        """

    def compute_loss_breakdown(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> LossBreakdown:
        """
        Compute loss breakdown by status (observed=2, masked=1, missing=0).
        trainable_loss = mean loss over masked only; others are for metrics (detached).
        Default for entity types: return zero breakdown.
        """
        device = predicted_params.device
        z = torch.zeros((), device=device)
        return LossBreakdown(
            trainable_loss=z,
            loss_observed=0.0,
            loss_masked=0.0,
            loss_missing=0.0,
            n_observed=0,
            n_masked=0,
            n_missing=0,
        )


class NullEntityType(EntityType):
    """
    Entity/variable type with no param content and no loss.

    Semantics: the random variable always evaluates to a single null value at the top,
    so there is never any prediction or loss associated with tokens of this type.
    """

    def __init__(self, name: str, variation: VariationConfig | None = None):
        super().__init__(
            name=name,
            is_variable=False,
            param_dim=0,
            variation=variation or VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )

    def build_param(self, raw_data: Dict[str, Any], device: torch.device, global_param_dim: int) -> torch.Tensor:
        # Always zero param stream for null variables/entities.
        return torch.zeros(global_param_dim, device=device)

    def compute_loss(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> torch.Tensor:
        # No direct loss.
        return torch.zeros((), device=predicted_params.device)


#########################################################
# Domain 3 Speicifc Entity Types
#########################################################

class AttributeEntityType(NullEntityType):
    """Attribute entity: closed class, no direct prediction."""

    def __init__(self, num_attributes: int):
        super().__init__(
            name="attribute",
            variation=VariationConfig(enabled=True, num_entities=num_attributes, reg_weight=0.0),
        )


class AnnotatorEntityType(NullEntityType):
    """Annotator entity: per-annotator deviation with regularization weight, no direct prediction."""

    def __init__(self, num_annotators: int, reg_weight: float = 0.0):
        super().__init__(
            name="annotator",
            variation=VariationConfig(enabled=True, num_entities=num_annotators, reg_weight=reg_weight),
        )

class ItemEntityType(NullEntityType):
    """Item entity: always new at test time, no deviation, no direct prediction."""

    def __init__(self, num_items: int):
        super().__init__(
            name="item",
            variation=VariationConfig(enabled=True, num_entities=num_items, reg_weight=0.0),
        )


class RatingVariableType(EntityType):
    """
    Rating variable: Likert rating with mask bit + class logits.

    Expects raw_data with keys:
      - is_missing: bool
      - is_masked: bool
      - rating_value: int | None
      - rating_dist: Optional[List[float]]
    """

    def __init__(self, num_classes: int, logit_high: float = 20.0):
        super().__init__(
            name="rating",
            is_variable=True,
            param_dim=1 + num_classes,
            variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )
        self.num_classes = num_classes
        self.logit_high = float(logit_high)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def build_param(self, raw_data: Dict[str, Any], device: torch.device, global_param_dim: int) -> torch.Tensor:
        # TODO: Input can be distribution, change this to accept "rating_dist" from the raw_data.
        p = torch.zeros(global_param_dim, device=device)
        is_missing = bool(raw_data.get("is_missing", False))
        is_masked = bool(raw_data.get("is_masked", False))

        if is_missing or is_masked:
            p[0] = 1.0
            return p

        rating_value = raw_data.get("rating_value", None)
        if rating_value is None:
            raise ValueError("rating_value must be provided for observed ratings")
        if not (0 <= rating_value < self.num_classes):
            raise ValueError(f"rating_value {rating_value} out of range [0, {self.num_classes})")

        p[0] = 0.0
        p[1 + rating_value] = self.logit_high
        return p

    def compute_loss(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> torch.Tensor:
        device = predicted_params.device
        B, L, _ = predicted_params.shape

        assert B == 1, "We assume B == 1 in this MVP."

        params = predicted_params[0]  # [L, P]
        mask_flat = type_mask[0]  # [L]
        if not mask_flat.any():
            return torch.zeros((), device=device)

        idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [M]
        params_sel = params[idx]  # [M, P]

        # Targets: int64 class indices in [0, C-1]
        targets: List[int] = []
        valid_mask = []
        for i in idx.tolist():
            t = tokens[i]
            raw = t.raw_data or {}
            is_missing = bool(raw.get("is_missing", False))
            if is_missing:
                # Never supervised on permanently missing.
                valid_mask.append(False)
                targets.append(0)
                continue

            rating_value = raw.get("rating_value", None)
            if rating_value is None:
                raise ValueError("rating_value must be present for non-missing rating tokens")

            valid_mask.append(True)
            targets.append(int(rating_value))

        valid_mask_tensor = torch.tensor(valid_mask, device=device, dtype=torch.bool)
        if not valid_mask_tensor.any():
            return torch.zeros((), device=device)

        params_valid = params_sel[valid_mask_tensor]  # [M_valid, P]
        targets_valid = torch.tensor([t for t, m in zip(targets, valid_mask) if m], device=device, dtype=torch.long)

        logits = params_valid[:, 1 : 1 + self.num_classes]

        #########################################################
        losses = self.ce_loss(logits, targets_valid)
        #########################################################
        return losses.mean()

    def compute_loss_breakdown(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> LossBreakdown:
        device = predicted_params.device
        B, L, _ = predicted_params.shape
        assert B == 1
        params = predicted_params[0]
        mask_flat = type_mask[0]
        if not mask_flat.any():
            return LossBreakdown(
                trainable_loss=torch.zeros((), device=device),
                loss_observed=0.0,
                loss_masked=0.0,
                loss_missing=0.0,
                n_observed=0,
                n_masked=0,
                n_missing=0,
            )
        idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # only masked indices  [M]
        params_sel = params[idx]  # [M, P] (P = global_param_dim = max dim of all entity types)
        logits = params_sel[:, 1 : 1 + self.num_classes]  # [M, C]
        # Build targets and status per token; require rating_value for all (observed, masked, missing)
        targets_list: List[int] = []
        status_list: List[int] = []
        for j, i in enumerate(idx.tolist()):
            t = tokens[i]
            raw = t.raw_data or {}
            status_list.append(t.status)
            rv = raw.get("rating_value", None)
            if rv is None:
                raise ValueError("rating_value must be present for rating tokens (observed/masked/missing)")
            targets_list.append(int(rv))
        targets = torch.tensor(targets_list, device=device, dtype=torch.long)
        losses = self.ce_loss(logits, targets)  # [M]
        n_observed = sum(1 for s in status_list if s == 2)
        n_masked = sum(1 for s in status_list if s == 1)
        n_missing = sum(1 for s in status_list if s == 0)
        masked_mask = torch.tensor([s == 1 for s in status_list], device=device, dtype=torch.bool)
        observed_mask = torch.tensor([s == 2 for s in status_list], device=device, dtype=torch.bool)
        missing_mask = torch.tensor([s == 0 for s in status_list], device=device, dtype=torch.bool)
        if masked_mask.any():
            trainable_loss = losses[masked_mask].mean()
            loss_masked_f = trainable_loss.detach().item()
        else:
            trainable_loss = torch.zeros((), device=device)
            loss_masked_f = 0.0
        loss_observed = float(losses[observed_mask].mean().item()) if observed_mask.any() else 0.0
        loss_missing = float(losses[missing_mask].mean().item()) if missing_mask.any() else 0.0
        return LossBreakdown(
            trainable_loss=trainable_loss,
            loss_observed=loss_observed,
            loss_masked=loss_masked_f,
            loss_missing=loss_missing,
            n_observed=n_observed,
            n_masked=n_masked,
            n_missing=n_missing,
        )


class PairwiseRankingVariableType(EntityType):
    """
    Pairwise ranking variable: two items with PL-style loss.

    Expects raw_data with keys:
      - is_missing: bool
      - is_masked: bool
      - ranking_order: List[int] of length 2 with positions (1=best).
    """

    def __init__(self, max_rank_size: int, logit_high: float = 20.0):
        super().__init__(
            name="ranking_pairwise",
            is_variable=True,
            param_dim=1 + max_rank_size,
            variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )
        self.max_rank_size = max_rank_size
        self.logit_high = float(logit_high)
        self.pl_loss = PlackettLuceLoss()

    def build_param(self, raw_data: Dict[str, Any], device: torch.device, global_param_dim: int) -> torch.Tensor:
        p = torch.zeros(global_param_dim, device=device)
        is_missing = bool(raw_data.get("is_missing", False))
        is_masked = bool(raw_data.get("is_masked", False))

        if is_missing or is_masked:
            p[0] = 1.0
            return p

        order = raw_data.get("ranking_order", None)
        if order is None or len(order) == 0:
            raise ValueError("ranking_order must be provided for observed rankings")

        p[0] = 0.0
        # For MVP we just set the winner's logit to 1.0 and loser to 0.0.
        # order[k] is rank (1=best). We keep only the first two.
        order = list(order)[: self.max_rank_size]
        if order[0] < order[1]:
            p[1] = self.logit_high
        else:
            p[2] = self.logit_high
        return p

    def compute_loss(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> torch.Tensor:
        device = predicted_params.device
        B, L, _ = predicted_params.shape

        params = predicted_params[0]  # [L, P]
        mask_flat = type_mask[0]
        if not mask_flat.any():
            return torch.zeros((), device=device)

        idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [M]
        params_sel = params[idx]  # [M, P]

        M = params_sel.shape[0]
        if M == 0:
            return torch.zeros((), device=device)

        logits = params_sel[:, 1 : 1 + self.max_rank_size].unsqueeze(0)  # [1, M, R]

        targets = torch.zeros(1, M, self.max_rank_size, device=device)
        mask = torch.zeros(1, M, dtype=torch.bool, device=device)
        for j, i in enumerate(idx.tolist()):
            raw = tokens[i].raw_data or {}
            order = raw.get("ranking_order", None)
            if order is None or len(order) == 0:
                raise ValueError("ranking_order must be present and non-empty for ranking tokens")
            order_t = torch.tensor(list(order)[: self.max_rank_size], dtype=targets.dtype, device=device)
            targets[0, j] = order_t
            mask[0, j] = True

        if not mask.any():
            return torch.zeros((), device=device)

        #########################################################
        loss = self.pl_loss(logits, targets, mask)
        #########################################################
        return loss

    def compute_loss_breakdown(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> LossBreakdown:
        device = predicted_params.device
        B, L, _ = predicted_params.shape
        assert B == 1
        params = predicted_params[0]
        mask_flat = type_mask[0]
        if not mask_flat.any():
            return LossBreakdown(
                trainable_loss=torch.zeros((), device=device),
                loss_observed=0.0,
                loss_masked=0.0,
                loss_missing=0.0,
                n_observed=0,
                n_masked=0,
                n_missing=0,
            )
        idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)
        params_sel = params[idx]
        M = params_sel.shape[0]
        logits = params_sel[:, 1 : 1 + self.max_rank_size].unsqueeze(0)  # [1, M, R]
        targets = torch.zeros(1, M, self.max_rank_size, device=device)
        pl_mask = torch.zeros(1, M, dtype=torch.bool, device=device)
        status_list: List[int] = []
        for j, i in enumerate(idx.tolist()):
            raw = tokens[i].raw_data or {}
            status_list.append(tokens[i].status)
            order = raw.get("ranking_order", None)
            if order is None or len(order) == 0:
                raise ValueError("ranking_order must be present for ranking tokens")
            order_t = torch.tensor(list(order)[: self.max_rank_size], dtype=targets.dtype, device=device)
            targets[0, j] = order_t
            pl_mask[0, j] = True
        n_observed = sum(1 for s in status_list if s == 2)
        n_masked = sum(1 for s in status_list if s == 1)
        n_missing = sum(1 for s in status_list if s == 0)

        def _pl_for_status(status_val: int) -> torch.Tensor:
            sel = [j for j in range(M) if status_list[j] == status_val]
            if not sel:
                return torch.tensor(0.0, device=device)
            logits_s = logits[:, sel, :]
            targets_s = targets[:, sel, :]
            mask_s = pl_mask[:, sel]
            return self.pl_loss(logits_s, targets_s, mask_s)

        trainable_loss = _pl_for_status(1) if n_masked > 0 else torch.zeros((), device=device)
        loss_observed = _pl_for_status(2).detach().item() if n_observed > 0 else 0.0
        loss_missing = _pl_for_status(0).detach().item() if n_missing > 0 else 0.0
        loss_masked_f = trainable_loss.detach().item() if n_masked > 0 else 0.0
        return LossBreakdown(
            trainable_loss=trainable_loss,
            loss_observed=loss_observed,
            loss_masked=loss_masked_f,
            loss_missing=loss_missing,
            n_observed=n_observed,
            n_masked=n_masked,
            n_missing=n_missing,
        )


def build_default_domain3_types(
    num_attributes: int,
    num_annotators: int,
    num_items: int,
    num_likert_classes: int,
    max_rank_size: int,
    logit_high: float = 20.0,
) -> Dict[str, EntityType]:
    """
    Convenience helper that builds the canonical domain-3 type registry.
    """
    return {
        "attribute": AttributeEntityType(num_attributes=num_attributes),
        "annotator": AnnotatorEntityType(num_annotators=num_annotators, reg_weight=0.0),
        "item": ItemEntityType(num_items=num_items),
        "rating": RatingVariableType(num_classes=num_likert_classes, logit_high=logit_high),
        "ranking_pairwise": PairwiseRankingVariableType(max_rank_size=max_rank_size, logit_high=logit_high),
    }

