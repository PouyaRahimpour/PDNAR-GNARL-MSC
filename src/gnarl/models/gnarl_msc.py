"""GNARL actor--critic for PDNAR's unchanged MSC bipartite representation."""

from __future__ import annotations

import torch
from torch import nn

from src.gnarl.models.actor import ProtoActionActor
from src.gnarl.models.critic import GraphCritic
from src.gnarl.models.encoder import MSCEncoder
from src.gnarl.models.mpnn import BipartiteMPNN


class GNARLMSC(nn.Module):
    def __init__(self, hidden_dim: int = 64, message_passing_rounds: int = 2):
        super().__init__()
        self.encoder = MSCEncoder(hidden_dim)
        self.processor = BipartiteMPNN(hidden_dim, message_passing_rounds)
        self.actor = ProtoActionActor(hidden_dim, graph_dim=2 * hidden_dim)
        self.critic = GraphCritic(2 * hidden_dim)

    def forward(self, env, state=None):
        state = env.state if state is None else state
        set_h, element_h = self.encoder(env.weights, state.selected, state.covered)
        set_h, element_h = self.processor(set_h, element_h, env.edge_index)
        graph_h = torch.cat((set_h.mean(dim=0), element_h.mean(dim=0)), dim=-1)
        return self.actor(set_h, graph_h), self.critic(graph_h)
