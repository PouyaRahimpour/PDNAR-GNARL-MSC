from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.policies.masked_policy import masked_categorical


@dataclass
class PPOConfig:
    total_steps: int = 10_000_000
    rollout_steps: int = 1024
    update_epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 64
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    device: str = "cpu"

    # Logging
    log_every: int = 1
    output_dir: str = "runs/gnarl_msc_ppo"
    progress_width: int = 32


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


def _progress_bar(current, total, width=32):
    ratio = current / max(total, 1)
    filled = int(width * ratio)

    return (
        "["
        + "=" * filled
        + ">" * (filled < width)
        + " " * max(width - filled - (filled < width), 0)
        + "]"
        f" {100.0 * ratio:6.2f}%"
    )


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h {int(minutes)}m"


def _write_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_history.csv"

    if not history:
        return

    keys = list(history[0].keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _plot_history(history, output_dir):
    if not history:
        return

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [x["steps"] for x in history]
    rewards = [x["mean_reward"] for x in history]
    losses = [x["loss"] for x in history]
    rollout_times = [x["rollout_seconds"] for x in history]
    update_times = [x["update_seconds"] for x in history]
    throughput = [x["steps_per_second"] for x in history]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rewards)
    plt.xlabel("Environment steps")
    plt.ylabel("Mean rollout reward")
    plt.title("PPO training reward")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "reward.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses)
    plt.xlabel("Environment steps")
    plt.ylabel("Loss")
    plt.title("PPO training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rollout_times, label="Rollout")
    plt.plot(steps, update_times, label="Update")
    plt.xlabel("Environment steps")
    plt.ylabel("Time (seconds)")
    plt.title("PPO runtime per rollout")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "runtime.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(steps, throughput)
    plt.xlabel("Environment steps")
    plt.ylabel("Steps / second")
    plt.title("PPO training throughput")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.png", dpi=180)
    plt.close()


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
    rollout_id = 0
    history = []

    training_start = time.perf_counter()

    while total_steps < config.total_steps:
        rollout_start = time.perf_counter()

        rollout = []

        rewards = []
        dones = []
        values = []
        next_values = []
        old_log_probs = []

        model.eval()

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

                _, reward, done, _ = env.step(action)

                if done:
                    next_value = torch.tensor(
                        0.0,
                        device=device,
                    )
                else:
                    with torch.no_grad():
                        _, next_value_tensor = model(env)

                        next_value = (
                            next_value_tensor.detach()
                        )

                rollout.append(
                    (
                        record,
                        state,
                        int(action.item()),
                    )
                )

                rewards.append(float(reward))
                dones.append(float(done))
                values.append(value.detach().reshape(()))
                next_values.append(next_value.reshape(()))
                old_log_probs.append(old_log_prob.detach())

                total_steps += 1

        rollout_seconds = time.perf_counter() - rollout_start

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
        next_values = torch.stack(next_values)
        old_log_probs = torch.stack(old_log_probs)

        advantages, returns = _compute_gae(
            rewards,
            values,
            next_values,
            dones,
            config.gamma,
            config.gae_lambda,
        )

        advantages = (
            advantages - advantages.mean()
        ) / advantages.std().clamp_min(1e-8)

        model.train()

        num_samples = len(rollout)

        last_loss = None

        update_start = time.perf_counter()

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

                log_probs = torch.stack(log_probs)
                current_values = torch.stack(current_values)
                entropies = torch.stack(entropies)

                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                ratio = (
                    log_probs - batch_old_log_probs
                ).exp()

                unclipped = ratio * batch_advantages

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
                    + config.value_coefficient * critic_loss
                    - config.entropy_coefficient * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )

                optimizer.step()

                last_loss = loss.detach()

        update_seconds = time.perf_counter() - update_start

        rollout_id += 1

        elapsed = time.perf_counter() - training_start

        steps_per_second = (
            total_steps / max(elapsed, 1e-8)
        )

        history.append(
            {
                "rollout": rollout_id,
                "steps": total_steps,
                "mean_reward": float(
                    rewards.mean().cpu()
                ),
                "loss": float(
                    last_loss.cpu()
                ),
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_per_second,
            }
        )

        if rollout_id % config.log_every == 0:
            progress = _progress_bar(
                total_steps,
                config.total_steps,
                config.progress_width,
            )

            eta = (
                (config.total_steps - total_steps)
                / max(steps_per_second, 1e-8)
            )

            print(
                f"\r"
                f"PPO {progress} "
                f"| rollout {rollout_id:4d} "
                f"| reward {rewards.mean().item():8.4f} "
                f"| loss {last_loss.item():9.4f} "
                f"| {steps_per_second:7.1f} step/s "
                f"| ETA {_format_time(eta)}",
                end="",
                flush=True,
            )

    print()

    _write_history(
        history,
        config.output_dir,
    )

    _plot_history(
        history,
        config.output_dir,
    )

    return history