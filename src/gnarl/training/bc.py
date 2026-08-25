from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.experts.ilp_expert import ILPExpert
from src.gnarl.policies.masked_policy import masked_categorical


@dataclass
class BCConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    episodes_per_graph: int = 1
    device: str = "cpu"


def train_bc(
    model,
    data: Iterable,
    config: BCConfig = BCConfig(),
):
    device = torch.device(config.device)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = []

    records = list(data)

    for _ in range(config.epochs):
        random.shuffle(records)

        losses = []

        model.train()

        for record in records:

            for _ in range(config.episodes_per_graph):

                env = MSCEnvironment(
                    record,
                    device,
                )

                expert = ILPExpert(
                    record,
                    device,
                )

                while not env.is_terminal():

                    logits, _ = model(env)

                    distribution = masked_categorical(
                        logits,
                        env.action_mask(),
                    )

                    target = expert.distribution(env)

                    # target is a probability distribution over actions.
                    log_probs = distribution.logits

                    loss = -(
                        target.to(log_probs.device)
                        * log_probs
                    ).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(
                        float(loss.detach().cpu())
                    )

                    env.step(
                        expert.sample_action(env)
                    )

        history.append(
            sum(losses) / max(len(losses), 1)
        )

    return history