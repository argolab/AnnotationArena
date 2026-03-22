from __future__ import annotations

"""
Toy two-layer relational attention experiment for a single depth-1 tree.

Goal:
- Manually design fixed Q/K/V (one head, with base + relational scores)
- Freeze them, and only train a small FFN head
- Show that with sufficient inductive bias we can get closer to the true
  subtree counts [4,1,1,1] than the constant-baseline solution.

This is intentionally independent of the main EntityMarformer to make
REPL experimentation easier.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyTree:
    """
    Tiny depth-1 tree:
      - node 0: root
      - nodes 1,2,3: children
    """

    x: torch.Tensor          # [N, D_in] input features
    targets: torch.Tensor    # [N, 1] subtree counts
    edge_mask: torch.Tensor  # [N, N, R]


def build_toy_tree(device: torch.device) -> ToyTree:
    N = 4
    root = 0
    children = [1, 2, 3]

    # Input features: start with exact zeros for all nodes.
    # This is the maximally symmetric case; any learning signal must come
    # from how Q/K/V are initialized and how edge_mask is used.
    D_in = N
    x = torch.zeros(N, D_in, device=device)

    # Targets: subtree sizes
    targets = torch.tensor([[4.0], [1.0], [1.0], [1.0]], device=device)

    # Single relation R=1: P2C edges
    edge_mask = torch.zeros(N, N, 1, device=device)
    for c in children:
        edge_mask[root, c, 0] = 1.0  # P2C: 0 -> {1,2,3}

    return ToyTree(x=x, targets=targets, edge_mask=edge_mask)


class FixedRelationalAttention(nn.Module):
    """
    Single-head attention with manually designed Q/K/V:

    - Input:  [N, D_in]
    - Internal model_dim = D_in
    - Q_full, K_full: map D_in -> D_in+R, but we only care about:
        * base scores from Q_base, K_base (first D_in dims)
        * relational scores from Q_rel (last R dims) and edge_mask

    Design:
      - base_scores give a strong positive bias to self (so leaves attend to self)
      - rel_scores give strong positive bias on P2C edges (root attends to children)
    """

    def __init__(self, d_in: int, num_rels: int):
        super().__init__()
        self.d_in = d_in
        self.num_rels = num_rels

        self.Q = nn.Linear(d_in, d_in + num_rels)
        self.K = nn.Linear(d_in, d_in + num_rels)
        self.V = nn.Linear(d_in, d_in)

        # Careful initialization: mimic a simple relational counting bias.
        # - base_scores: small self-bias via Q_base/K_base
        # - rel_scores: large positive bias on P2C edges via Q_rel
        # - V: ignore x (which is zero) and use a constant value
        #   so attention pattern is visible in outputs.
        self._init_analytic()

    def _init_analytic(self) -> None:
        D = self.d_in
        R = self.num_rels
        with torch.no_grad():
            # Zero everything first.
            self.Q.weight.zero_(); self.Q.bias.zero_()
            self.K.weight.zero_(); self.K.bias.zero_()
            self.V.weight.zero_(); self.V.bias.zero_()

            # Base Q/K: give a small self-bias by making Q_base = K_base = e_0.
            # With x=0 this still collapses to constant base scores, but we
            # keep it simple here and let the relational term dominate.
            self.Q.bias[:D] = 0.0
            self.K.bias[:D] = 0.0

            # Relational part of Q: add +b on relation dim (last R entry).
            b = 5.0
            for r in range(R):
                self.Q.bias[D + r] = b

            # V: output a constant 1.0 in the first dimension for every token,
            # so attention rows show up clearly in the aggregated output.
            self.V.bias[0] = 1.0

    def forward(self, x: torch.Tensor, edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [N, D_in]
          edge_mask: [N, N, R]

        Returns:
          out:  [N, D_in]
          attn: [N, N]
        """
        N, D = x.shape
        assert D == self.d_in
        R = self.num_rels

        Q_full = self.Q(x)     # [N, D+R]
        K_full = self.K(x)     # [N, D+R]
        V = self.V(x)          # [N, D]

        Q_base = Q_full[:, :D]
        K_base = K_full[:, :D]

        # Base scores: [N,N]
        base_scores = (Q_base @ K_base.T) / math.sqrt(D)

        # Relational scores: use Q_rel * edge_mask
        Q_rel = Q_full[:, D:]  # [N,R]
        # rel_scores[i,j] = sum_r Q_rel[i,r] * edge_mask[i,j,r]
        rel_scores = torch.einsum("ir,ijr->ij", Q_rel, edge_mask)

        scores = base_scores + rel_scores
        attn = F.softmax(scores, dim=-1)  # [N,N]
        out = attn @ V                    # [N,D]
        return out, attn


class TwoLayerToyModel(nn.Module):
    """
    Two layers of fixed relational attention + trainable FFN head.
    """

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.attn1 = FixedRelationalAttention(d_in=d_in, num_rels=1)
        self.attn2 = FixedRelationalAttention(d_in=d_in, num_rels=1)

        # Trainable head that maps the aggregated representation to counts.
        self.ff = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1, attn1 = self.attn1(x, edge_mask)
        h2, attn2 = self.attn2(h1, edge_mask)
        y = self.ff(h2)
        return y, attn1, attn2


def run_toy_two_layer(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tree = build_toy_tree(device=device)
    N, D_in = tree.x.shape

    model = TwoLayerToyModel(d_in=D_in).to(device)

    # All params (Q/K/V + FFN) are now trainable, but start from the analytic init.
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable_params)

    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    print("Starting toy training...")
    for step in range(1000):
        opt.zero_grad()
        y_pred, attn1, attn2 = model(tree.x, tree.edge_mask)
        loss = F.mse_loss(y_pred, tree.targets)
        loss.backward()
        opt.step()

        if step == 0:
            # Print full attention matrices for the initial epoch.
            a1 = attn1.detach().cpu()
            a2 = attn2.detach().cpu()
            print("Initial Layer 1 attention matrix:")
            print(a1)
            print("Initial Layer 2 attention matrix:")
            print(a2)

        if (step + 1) % 50 == 0 or step == 0:
            print(f"step {step+1:4d} | loss={loss.item():.6f}")
            print("  preds:", [round(float(v), 4) for v in y_pred.detach().cpu().view(-1)])

    with torch.no_grad():
        y_pred, attn1, attn2 = model(tree.x, tree.edge_mask)
        print("\nFinal predictions:", [round(float(v), 4) for v in y_pred.detach().cpu().view(-1)])
        print("Targets:", tree.targets.view(-1).tolist())
        print("\nLayer 1 attention (root row):", [round(float(v), 4) for v in attn1[0].detach().cpu().tolist()])
        print("Layer 2 attention (root row):", [round(float(v), 4) for v in attn2[0].detach().cpu().tolist()])


if __name__ == "__main__":
    run_toy_two_layer()

