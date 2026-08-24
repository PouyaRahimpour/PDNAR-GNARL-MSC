"""Message passing over PDNAR's element--set incidence graph."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter


class BipartiteMPNN(nn.Module):
    def __init__(self, hidden_dim: int, rounds: int = 2):
        super().__init__()
        self.rounds = rounds
        self.set_to_element = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(rounds)])
        self.element_update = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rounds)])
        self.element_to_set = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(rounds)])
        self.set_update = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rounds)])

    def forward(self, set_h: torch.Tensor, element_h: torch.Tensor, edge_index: torch.Tensor):
        element_index, set_index = edge_index
        for set_message, element_update, element_message, set_update in zip(
            self.set_to_element, self.element_update, self.element_to_set, self.set_update
        ):
            to_elements = scatter(set_message(set_h)[set_index], element_index, dim=0,
                                  dim_size=element_h.size(0), reduce="sum")
            element_h = element_update(to_elements, element_h)
            to_sets = scatter(element_message(element_h)[element_index], set_index, dim=0,
                              dim_size=set_h.size(0), reduce="sum")
            set_h = set_update(to_sets, set_h)
        return set_h, element_h
