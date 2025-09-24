"""Loss functions and strategies for ranking imputation models.

This module provides loss computation functionality for models that predict both
ratings and rankings. It includes:

1. PlackettLuceLoss: A loss function for ranking data based on the Plackett-Luce
   probability model over permutations.

2. LossStrategyBase: An abstract base class for loss computation strategies that
   operate independently of model architecture.

3. DefaultLossStrategy: A concrete implementation that combines cross-entropy
   loss for ratings with Plackett-Luce loss for rankings, with separate handling
   of observed and masked data.

"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from .data import RankingData


class PlackettLuceLoss(nn.Module):
    """Plackett-Luce loss for ranking data."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Plackett–Luce (PL) ranking loss.

        Args:
            logits (torch.Tensor): Predicted scores, shape [B, V, K].
            targets (torch.Tensor): Ground-truth ranks, same shape. Convention:
                - Rank 1 is best (smaller rank = better).
            mask (torch.Tensor): Boolean mask [B, V], True where loss should be computed.

        Returns:
            torch.Tensor: Scalar PL loss (negative log-likelihood of ground-truth rankings).

        Notes:
            The PL loss models the probability of generating the observed ranking
            item-by-item. At each step i, the correct item’s probability is:
                p_i = exp(score_i) / sum_j exp(score_j for j ≥ i)
            Loss = -∑ log(p_i)

        Example: (B=1, V=1, K=4, K_ranking=3)
            >>> scores = torch.tensor([[[2.0, 0.5, -1.0, 2.2]]])   # shape [1,1,4]
            >>> ranks  = torch.tensor([[[1, 2, 3, 0]]])             # 0 = ignored, 1 = best
            >>> mask   = torch.tensor([[True]])
            >>> loss = forward(scores, ranks, mask)
            >>> print(round(loss.item(), 2))
            0.44   # lower = better ranking fit
        """
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        loss = 0.0
        count = 0

        for b in range(logits.size(0)):  # batch 
            for v in range(logits.size(1)):  # v=(i,j, set of k)
                if mask[b, v]:
                    scores = logits[b, v]
                    target_ranks = targets[b, v]
                    valid_positions = target_ranks > 0  # valid k in the set of k's
                    if not valid_positions.any():
                        raise ValueError(f"No valid positions for voter {v}")  # should be masked in the first place

                    valid_scores = scores[valid_positions]
                    valid_targets = target_ranks[valid_positions]
                    # Sort by target rank (ascending: 1=best first, 2=second, etc.)
                    _, sort_indices = torch.sort(valid_targets, descending=False)
                    sorted_scores = valid_scores[sort_indices]

                    pl_loss = 0.0
                    for i in range(len(sorted_scores)):
                        remaining_scores = sorted_scores[i:]
                        log_prob = sorted_scores[i] - torch.logsumexp(remaining_scores, dim=0)
                        pl_loss -= log_prob
                    loss += pl_loss
                    count += 1

        return loss / max(count, 1)


class LossStrategyBase:
    """Base class for loss computation strategies using structured inputs."""

    def __init__(self):
        super().__init__()

    def compute(
        self,
        predictions: List["TopLayerPredictionResult"],
        references: List[RankingData],
        masked_flags: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """Compute loss from structured predictions and references.

        - predictions: per-variable model outputs (one per variable)
        - references: per-variable supervision (RatingData with targets present when available)
        - masked_flags: optional list aligned to predictions marking masked entries (True=masked)

        Returns keys consistent with trainer expectations. Implementations may return
        additional keys prefixed with '_' to pass tensors for backprop in orchestration layers.
        """
        raise NotImplementedError


class DefaultLossStrategy(LossStrategyBase):
    """Default loss strategy with cross-entropy for ratings and Plackett-Luce for rankings."""

    def __init__(self, masked_loss_weight: float = 1.0, observed_loss_weight: float = 1.0):
        """Initialize loss functions and weights."""
        super().__init__()
        self.rating_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.ranking_loss_fn = PlackettLuceLoss()
        self.masked_loss_weight = masked_loss_weight
        self.observed_loss_weight = observed_loss_weight

    def compute(
        self,
        predictions: List["TopLayerPredictionResult"],
        references: List[RankingData],
        masked_flags: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """Convert structured lists to tensors and compute vectorized losses (B=1)."""
        assert len(predictions) == len(references), "predictions and references length must match"
        N = len(predictions)
        device = predictions[0].device if predictions else torch.device('cpu')

        # Basic check: each prediction should provide at least one head
        for p in predictions:
            if p.rating_logits is None and p.ranking_logits is None:
                raise ValueError("Either rating or ranking logits must be provided")

        # Infer dimensions
        C = next((p.rating_logits.shape[-1] for p in predictions if p.rating_logits is not None), 0) # FIXME: should we use is_listwise for stricter checking? Implicitly checking either rating or ranking has to contain data.
        R = next((p.ranking_logits.shape[-1] for p in predictions if p.ranking_logits is not None), 0)

        # Stack logits to [1, N, C/R]
        rating_logits = torch.stack([
            p.rating_logits if p.rating_logits is not None else torch.zeros(C, device=device)
            for p in predictions
        ], dim=0).unsqueeze(0)

        ranking_logits = torch.stack([
            p.ranking_logits if p.ranking_logits is not None else torch.zeros(R, device=device)
            for p in predictions
        ], dim=0).unsqueeze(0)

        # Build targets and masks
        rating_targets = torch.zeros(1, N, C, device=device)
        ranking_targets = torch.zeros(1, N, R, device=device)
        rating_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        ranking_mask = torch.zeros(1, N, dtype=torch.bool, device=device)

        for i, ref in enumerate(references):
            if not ref.is_listwise:

                if ref.rating_value is not None and 0 <= ref.rating_value < C:
                    rating_targets[0, i, ref.rating_value] = 1.0  # one-hot distribution
                    rating_mask[0, i] = True
                else:
                    raise ValueError(f"Rating value not found or out of range for voter {i}")
            else:

                if ref.ranking_order is not None and len(ref.ranking_order) > 0:

                    assert len(ref.ranking_order) == R, f"Ranking order length must match R: {len(ref.ranking_order)} != {R}"
                    order_tensor = torch.tensor(ref.ranking_order, device=device, dtype=ranking_targets.dtype)
                    ranking_targets[0, i] = order_tensor
                    ranking_mask[0, i] = True
                else:
                    raise ValueError(f"Ranking order not found or empty for voter {i}")

        zero = torch.tensor(0.0, device=device)

        # Separate losses by masked status
        masked_rating_loss = zero
        observed_rating_loss = zero
        masked_ranking_loss = zero
        observed_ranking_loss = zero

        # Rating losses via CrossEntropy - vectorized with masked/observed separation
        if rating_mask.any():
            idx = rating_mask.nonzero(as_tuple=False)
            v_logits = rating_logits[idx[:, 0], idx[:, 1]]
            v_targets = rating_targets[idx[:, 0], idx[:, 1]]
            target_classes = torch.argmax(v_targets, dim=1)
            losses = self.rating_loss_fn(v_logits, target_classes)

            # Separate by masked status using boolean indexing
            valid_indices = idx[:, 1]  # Get variable indices
            masked_mask = torch.tensor([references[i].is_masked for i in valid_indices], device=device)

            masked_losses = losses[masked_mask]
            observed_losses = losses[~masked_mask]

            # Proper averaging
            if len(masked_losses) > 0:
                masked_rating_loss = masked_losses.mean()
            if len(observed_losses) > 0:
                observed_rating_loss = observed_losses.mean()

        # Ranking losses via Plackett-Luce - separated by masked status
        if ranking_mask.any():
            # Create separate masks for masked and observed rankings
            masked_ranking_mask = torch.zeros_like(ranking_mask)
            observed_ranking_mask = torch.zeros_like(ranking_mask)

            for i in range(N):
                if ranking_mask[0, i]:  # Has ranking data
                    if references[i].is_masked:
                        masked_ranking_mask[0, i] = True
                    else:
                        observed_ranking_mask[0, i] = True

            # Compute losses separately
            if masked_ranking_mask.any():
                masked_ranking_loss = self.ranking_loss_fn(ranking_logits, ranking_targets, masked_ranking_mask)
            if observed_ranking_mask.any():
                observed_ranking_loss = self.ranking_loss_fn(ranking_logits, ranking_targets, observed_ranking_mask)

        # Weighted combination of masked and observed losses (weights mostly relevant during training)
        masked_total_loss = masked_rating_loss + masked_ranking_loss
        observed_total_loss = observed_rating_loss + observed_ranking_loss

        total_loss = (self.masked_loss_weight * masked_total_loss +
                     self.observed_loss_weight * observed_total_loss)
        
        total_unweighted_loss = masked_total_loss + observed_total_loss

        # For backward compatibility, also compute unweighted component losses
        rating_loss = masked_rating_loss + observed_rating_loss
        ranking_loss = masked_ranking_loss + observed_ranking_loss

        return {
            'total_loss': float(total_unweighted_loss.item()),
            'rating_loss': float(rating_loss.item()),
            'ranking_loss': float(ranking_loss.item()),
            # Masked/observed breakdown
            'masked_total_loss': float(masked_total_loss.item()),
            'observed_total_loss': float(observed_total_loss.item()),
            'masked_rating_loss': float(masked_rating_loss.item()),
            'observed_rating_loss': float(observed_rating_loss.item()),
            'masked_ranking_loss': float(masked_ranking_loss.item()),
            'observed_ranking_loss': float(observed_ranking_loss.item()),
            # Return tensors for backprop in trainer
            '_total_loss_tensor': total_loss,
        }


# Structured adapters for predictions
@dataclass
class TopLayerPredictionResult:
    is_listwise: Optional[bool] = None
    rating_logits: Optional[torch.Tensor] = None  # [C]
    ranking_logits: Optional[torch.Tensor] = None  # [R]

    @property
    def device(self) -> torch.device:
        if self.rating_logits is not None:
            return self.rating_logits.device
        if self.ranking_logits is not None:
            return self.ranking_logits.device
        return torch.device('cpu')


def adapt_batched_logits_to_predictions(logits: Dict[str, torch.Tensor]) -> List[TopLayerPredictionResult]:
    """Convert batched model logits dict {'rating': [B,N,C], 'ranking': [B,N,R]} to per-variable predictions for B==1.

    This adapter pairs naturally with a List[RankingData] of length N used as references.
    """
    rating = logits['rating']  # [B,N,C]
    ranking = logits['ranking']  # [B,N,R]
    assert rating.shape[0] == 1 and ranking.shape[0] == 1, "Adapter expects batch size 1"
    N = rating.shape[1]
    out: List[TopLayerPredictionResult] = []
    for i in range(N):
        pr = TopLayerPredictionResult()
        pr.rating_logits = rating[0, i]
        pr.ranking_logits = ranking[0, i]
        out.append(pr)
    return out
