"""Small, dependency-free PPO loop for the sequential MSC environment."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from src.gnarl.envs.msc_env import MSCEnvironment, MSCState
from src.gnarl.policies.masked_policy import masked_categorical


@dataclass
class PPOConfig:
    total_steps: int = 10_000
    rollout_steps: int = 512
    update_epochs: int = 4
    learning_rate: float = 5e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gamma: float = 1.0
    device: str = "cpu"


def _discounted_returns(rewards, dones, gamma):
    returns, value = [], 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        value = reward + gamma * value * (1.0 - float(done))
        returns.append(value)
    return list(reversed(returns))


def train_ppo(model, data, config: PPOConfig = PPOConfig()):
    """Fine-tune a GNARL actor/critic from reward only; no expert is queried."""
    if not data:
        raise ValueError("PPO requires at least one training graph.")
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps, history = 0, []
    while total_steps < config.total_steps:
        rollout = []
        rewards, dones = [], []
        model.eval()
        while len(rollout) < config.rollout_steps and total_steps < config.total_steps:
            env = MSCEnvironment(random.choice(data), device)
            while not env.is_terminal() and len(rollout) < config.rollout_steps and total_steps < config.total_steps:
                with torch.no_grad():
                    logits, value = model(env)
                    distribution = masked_categorical(logits, env.action_mask())
                    action = distribution.sample()
                state = env.snapshot()
                _, reward, done, _ = env.step(action)
                rollout.append((env.data, state, int(action.item()), float(distribution.log_prob(action).item()), float(value.item())))
                rewards.append(reward)
                dones.append(done)
                total_steps += 1
        returns = torch.tensor(_discounted_returns(rewards, dones, config.gamma), device=device)
        old_log_probs = torch.tensor([item[3] for item in rollout], device=device)
        old_values = torch.tensor([item[4] for item in rollout], device=device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        model.train()
        for _ in range(config.update_epochs):
            log_probs, values, entropies = [], [], []
            for record, state, action, _, _ in rollout:
                env = MSCEnvironment(record, device)
                logits, value = model(env, state)
                distribution = masked_categorical(logits, env.action_mask(state))
                action = torch.tensor(action, device=device)
                log_probs.append(distribution.log_prob(action))
                values.append(value)
                entropies.append(distribution.entropy())
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            ratio = (log_probs - old_log_probs).exp()
            actor_loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages).mean()
            critic_loss = torch.nn.functional.mse_loss(values, returns)
            loss = actor_loss + config.value_coefficient * critic_loss - config.entropy_coefficient * torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        history.append({"steps": total_steps, "mean_reward": sum(rewards) / len(rewards), "loss": float(loss.detach().cpu())})
    return history
