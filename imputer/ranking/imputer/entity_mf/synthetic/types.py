from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from imputer.entity_mf.types import EntityType, LossBreakdown, VariationConfig


@dataclass
class RegressionSlices:
    """
    How we interpret a token's param stream for synthetic regression tasks.

    Layout within each token's param vector:
      - input:  [0 : input_dim)
      - output: [input_dim : input_dim + output_dim)  (supervised)
    """

    input_dim: int
    output_dim: int

    @property
    def param_dim(self) -> int:
        return self.input_dim + self.output_dim

    def output_slice(self) -> slice:
        return slice(self.input_dim, self.input_dim + self.output_dim)


class SyntheticRegressionType(EntityType):
    """
    A minimal EntityType for synthetic sanity-checking.

    - build_param encodes an optional input vector into the param stream.
    - compute_loss computes MSE between predicted output slice and target_value.

    Token raw_data conventions:
      - input_value:  List[float] length == input_dim (optional if input_dim == 0)
      - target_value: List[float] length == output_dim (required for supervised tokens)
    """

    def __init__(
        self,
        name: str,
        slices: RegressionSlices,
        has_target: bool = True,
        variation: VariationConfig | None = None,
    ):
        super().__init__(
            name=name,
            is_variable=bool(has_target),
            param_dim=int(slices.param_dim),
            variation=variation or VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )
        self.slices = slices
        self.has_target = bool(has_target)

    def build_param(self, raw_data: Dict[str, Any], device: torch.device, global_param_dim: int) -> torch.Tensor:
        p = torch.zeros(global_param_dim, device=device)
        if self.slices.input_dim <= 0:
            return p
        vals = raw_data.get("input_value", None)
        if vals is None:
            # Allow missing input_value for input_dim>0; treat as zeros (useful for internal nodes).
            return p
        if len(vals) != self.slices.input_dim:
            raise ValueError(f"{self.name}: input_value length {len(vals)} != input_dim {self.slices.input_dim}")
        p[: self.slices.input_dim] = torch.tensor(vals, device=device, dtype=p.dtype)
        return p

    def compute_loss(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> torch.Tensor:
        device = predicted_params.device
        if self.slices.output_dim <= 0 or not self.has_target:
            return torch.zeros((), device=device)

        assert predicted_params.shape[0] == 1, "Synthetic tasks assume batch size 1."
        mask_flat = type_mask[0]
        if not mask_flat.any():
            return torch.zeros((), device=device)

        idx = mask_flat.nonzero(as_tuple=False).squeeze(-1).tolist()
        preds: List[torch.Tensor] = []
        tgts: List[torch.Tensor] = []
        for i in idx:
            raw = getattr(tokens[i], "raw_data", None) or {}
            t = raw.get("target_value", None)
            if t is None:
                continue
            if len(t) != self.slices.output_dim:
                raise ValueError(f"{self.name}: target_value length {len(t)} != output_dim {self.slices.output_dim}")
            pred_i = predicted_params[0, i, self.slices.output_slice()]
            tgt_i = torch.tensor(t, device=device, dtype=pred_i.dtype)
            preds.append(pred_i)
            tgts.append(tgt_i)

        if not preds:
            return torch.zeros((), device=device)

        pred = torch.stack(preds, dim=0)
        tgt = torch.stack(tgts, dim=0)
        return F.mse_loss(pred, tgt, reduction="mean")

    def compute_loss_breakdown(
        self,
        predicted_params: torch.Tensor,
        tokens: Sequence[Any],
        type_mask: torch.Tensor,
        global_param_dim: int,
    ) -> LossBreakdown:
        """
        For synthetic tasks we treat all supervised tokens as observed.
        We still return a LossBreakdown so callers can reuse the same interface.
        """
        device = predicted_params.device
        loss = self.compute_loss(predicted_params, tokens, type_mask, global_param_dim)
        # For these tasks we optimize on all targets; use trainable_loss = loss.
        # Status-specific buckets are left as observed-only.
        n = int(type_mask[0].sum().item()) if predicted_params.shape[0] == 1 else 0
        return LossBreakdown(
            trainable_loss=loss,
            loss_observed=float(loss.detach().item()) if n > 0 else 0.0,
            loss_masked=0.0,
            loss_missing=0.0,
            n_observed=n,
            n_masked=0,
            n_missing=0,
        )

