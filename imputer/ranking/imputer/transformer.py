"""NormLayer and FeedForward, used by imputer/entity_mf/model.py's
RelationalAttentionBlock (the current transformer block implementation)."""

import torch
import torch.nn as nn


class NormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = 512,
        dropout: float = 0.1,
        output_dim: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()

        # Build a deeper MLP with ReLU non-linearities.
        # Structure: [d_model -> d_ff] + (num_layers-2) x [d_ff -> d_ff] + [d_ff -> out_dim]
        out_dim = d_model if output_dim is None else output_dim
        hidden_layers = max(1, num_layers - 1)

        layers: list[nn.Module] = []

        # First layer projects from input to hidden width
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))

        # Middle hidden layers (if any)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # Final projection to output dimension (no activation)
        layers.append(nn.Linear(d_ff, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
