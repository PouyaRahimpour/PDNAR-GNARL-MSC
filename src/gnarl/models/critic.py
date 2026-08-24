from __future__ import annotations

from torch import nn


class GraphCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, graph_h):
        return self.value(graph_h).squeeze(-1)
