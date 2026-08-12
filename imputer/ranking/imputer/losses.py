"""Plackett-Luce ranking loss, used by EntityMarformer's RatingVariableType/
PairwiseRankingVariableType (see imputer/entity_mf/types.py)."""

import torch
import torch.nn as nn


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
