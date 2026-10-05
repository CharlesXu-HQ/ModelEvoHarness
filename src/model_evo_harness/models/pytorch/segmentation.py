"""Learned soft regions with independent local linear response models."""

import torch
from torch import nn


class MLR(nn.Module):
    """Mixture of logistic regressions for dense [batch, feature] input.

    Returns a probability, so the experiment should use a probability-aware
    binary loss. Region support should be inspected on held-out data.
    """

    def __init__(self, input_dim: int, num_regions: int):
        super().__init__()
        self.region = nn.Linear(input_dim, num_regions)
        self.local = nn.Linear(input_dim, num_regions)

    def region_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.region(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.region_weights(x) * torch.sigmoid(self.local(x))).sum(dim=-1)
