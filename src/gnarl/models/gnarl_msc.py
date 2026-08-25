from __future__ import annotations

import torch
from torch import nn

from src.gnarl.models.actor import ProtoActionActor
from src.gnarl.models.critic import GraphCritic
from src.gnarl.models.encoder import MSCEncoder
from src.gnarl.models.mpnn import BipartiteMPNN


class GNARLMSC(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_rounds: int = 4,
        pooling: str = "mean",
    ):
        super().__init__()

        if pooling not in {"mean", "max"}:
            raise ValueError(
                f"Unknown pooling method: {pooling}"
            )

        self.pooling = pooling

        self.encoder = MSCEncoder(hidden_dim)

        self.processor = BipartiteMPNN(
            hidden_dim,
            message_passing_rounds,
        )

        self.actor = ProtoActionActor(
            hidden_dim,
            graph_dim=2 * hidden_dim,
        )

        self.critic = GraphCritic(
            2 * hidden_dim,
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return x.mean(dim=0)

        return x.max(dim=0).values

    def forward(self, env, state=None):
        state = env.state if state is None else state

        set_h, element_h = self.encoder(
            env.weights,
            state.selected,
            state.covered,
        )

        set_h, element_h = self.processor(
            set_h,
            element_h,
            env.edge_index,
        )

        graph_h = torch.cat(
            [
                self._pool(set_h),
                self._pool(element_h),
            ],
            dim=-1,
        )

        logits = self.actor(
            set_h,
            graph_h,
        )

        value = self.critic(
            graph_h,
        )

        return logits, value