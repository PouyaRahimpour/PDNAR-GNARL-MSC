from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.policies.masked_policy import masked_categorical


@dataclass
class PPOConfig:
    total_steps: int = 10_000_000

    # GNARL:
    rollout_steps: int = 1024
    update_epochs: int = 10

    # GNARL MVC:
    learning_rate: float = 1e-5

    # SB3 defaults:
    batch_size: int = 64
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    device: str = "cpu"


def _compute_gae(
    rewards,
    values,
    next_values,
    dones,
    gamma,
    gae_lambda,
):
    advantages = torch.zeros_like(values)

    gae = torch.tensor(
        0.0,
        device=values.device,
    )

    for t in reversed(range(len(rewards))):

        nonterminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_values[t] * nonterminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns


def train_ppo(
    model,
    data,
    config: PPOConfig = PPOConfig(),
):
    if not data:
        raise ValueError(
            "PPO requires at least one training graph."
        )

    device = torch.device(config.device)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    total_steps = 0
    history = []

    while total_steps < config.total_steps:

        rollout = []

        rewards = []
        dones = []
        values = []
        next_values = []
        old_log_probs = []

        model.eval()

        # ---------------------------------------------------------
        # Collect exactly one PPO rollout.
        # ---------------------------------------------------------
        while (
            len(rollout) < config.rollout_steps
            and total_steps < config.total_steps
        ):

            record = random.choice(data)

            env = MSCEnvironment(
                record,
                device,
            )

            while (
                not env.is_terminal()
                and len(rollout) < config.rollout_steps
                and total_steps < config.total_steps
            ):

                state = env.snapshot()

                with torch.no_grad():

                    logits, value = model(
                        env,
                        state,
                    )

                    distribution = masked_categorical(
                        logits,
                        env.action_mask(state),
                    )

                    action = distribution.sample()

                    old_log_prob = distribution.log_prob(
                        action
                    )

                _, reward, done, _ = env.step(
                    action
                )

                # Bootstrap from the state AFTER the action.
                if done:

                    next_value = torch.tensor(
                        0.0,
                        device=device,
                    )

                else:

                    with torch.no_grad():

                        _, next_value_tensor = model(
                            env
                        )

                        next_value = next_value_tensor.detach()

                rollout.append(
                    (
                        record,
                        state,
                        int(action.item()),
                    )
                )

                rewards.append(
                    float(reward)
                )

                dones.append(
                    float(done)
                )

                values.append(
                    value.detach().reshape(())
                )

                next_values.append(
                    next_value.reshape(())
                )

                old_log_probs.append(
                    old_log_prob.detach()
                )

                total_steps += 1

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device,
        )

        values = torch.stack(values)

        next_values = torch.stack(
            next_values
        )

        old_log_probs = torch.stack(
            old_log_probs
        )

        advantages, returns = _compute_gae(
            rewards,
            values,
            next_values,
            dones,
            config.gamma,
            config.gae_lambda,
        )

        # SB3-style advantage normalization.
        advantages = (
            advantages - advantages.mean()
        ) / advantages.std().clamp_min(1e-8)

        # ---------------------------------------------------------
        # PPO updates.
        # ---------------------------------------------------------
        model.train()

        num_samples = len(rollout)

        last_loss = None

        for _ in range(config.update_epochs):

            permutation = torch.randperm(
                num_samples,
                device=device,
            )

            for start in range(
                0,
                num_samples,
                config.batch_size,
            ):

                indices = permutation[
                    start:start + config.batch_size
                ]

                log_probs = []
                current_values = []
                entropies = []

                for index in indices.tolist():

                    record, state, action = rollout[index]

                    env = MSCEnvironment(
                        record,
                        device,
                    )

                    logits, value = model(
                        env,
                        state,
                    )

                    distribution = masked_categorical(
                        logits,
                        env.action_mask(state),
                    )

                    action_tensor = torch.tensor(
                        action,
                        device=device,
                    )

                    log_probs.append(
                        distribution.log_prob(
                            action_tensor
                        )
                    )

                    current_values.append(
                        value.reshape(())
                    )

                    entropies.append(
                        distribution.entropy()
                    )

                log_probs = torch.stack(
                    log_probs
                )

                current_values = torch.stack(
                    current_values
                )

                entropies = torch.stack(
                    entropies
                )

                batch_old_log_probs = (
                    old_log_probs[indices]
                )

                batch_advantages = (
                    advantages[indices]
                )

                batch_returns = (
                    returns[indices]
                )

                ratio = (
                    log_probs
                    - batch_old_log_probs
                ).exp()

                unclipped = (
                    ratio
                    * batch_advantages
                )

                clipped = (
                    ratio.clamp(
                        1.0 - config.clip_ratio,
                        1.0 + config.clip_ratio,
                    )
                    * batch_advantages
                )

                actor_loss = -torch.minimum(
                    unclipped,
                    clipped,
                ).mean()

                critic_loss = torch.nn.functional.mse_loss(
                    current_values,
                    batch_returns,
                )

                entropy_loss = entropies.mean()

                loss = (
                    actor_loss
                    + config.value_coefficient
                    * critic_loss
                    - config.entropy_coefficient
                    * entropy_loss
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )

                optimizer.step()

                last_loss = loss.detach()

        history.append(
            {
                "steps": total_steps,
                "mean_reward": float(
                    rewards.mean().cpu()
                ),
                "loss": float(
                    last_loss.cpu()
                ),
            }
        )

    return history