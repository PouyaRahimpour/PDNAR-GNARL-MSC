"""Feature encoders for the PDNAR bipartite MSC state."""

from __future__ import annotations

import torch
from torch import nn


class MSCEncoder(nn.Module):
    """Encodes fixed weights plus the MDP's selected/covered state features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.set_encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.element_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, weights: torch.Tensor, selected: torch.Tensor, covered: torch.Tensor):
        # Relative scaling is invariant to a common rescaling of all costs.
        scale = weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
        set_features = torch.stack((weights / scale, selected.float()), dim=-1)
        return self.set_encoder(set_features), self.element_encoder(covered.float().unsqueeze(-1))
