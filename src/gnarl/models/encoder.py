from __future__ import annotations

import torch
from torch import nn


class FeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1).float())


class MSCEncoder(nn.Module):
    """
    GNARL-style feature encoder for the MSC bipartite graph.

    Each distinct feature is encoded separately and features at the
    same graph location are summed.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Set input/state features.
        self.weight_encoder = FeatureEncoder(hidden_dim)
        self.selected_encoder = FeatureEncoder(hidden_dim)

        # Element state feature.
        self.covered_encoder = FeatureEncoder(hidden_dim)

    def forward(
        self,
        weights: torch.Tensor,
        selected: torch.Tensor,
        covered: torch.Tensor,
    ):
        set_h = (
            self.weight_encoder(weights)
            + self.selected_encoder(selected)
        )

        element_h = self.covered_encoder(covered)

        return set_h, element_h