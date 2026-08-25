from __future__ import annotations

import torch
from torch import nn


class ProtoActionActor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        graph_dim: int | None = None,
    ):
        super().__init__()

        self.proto_action = nn.Linear(
            graph_dim or hidden_dim,
            hidden_dim,
        )

        self.log_temperature = nn.Parameter(
            torch.zeros(())
        )

    def forward(
        self,
        set_h: torch.Tensor,
        graph_h: torch.Tensor,
    ) -> torch.Tensor:

        proto = self.proto_action(graph_h)

        distance = torch.linalg.vector_norm(
            set_h - proto,
            ord=2,
            dim=-1,
        )

        temperature = (
            self.log_temperature.exp()
            .clamp_min(1e-3)
        )

        return -distance / temperature