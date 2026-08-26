from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
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

    output_dir: str = "runs/gnarl_msc_bc"
    progress_width: int = 32


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


def _write_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        output_dir / "training_history.csv"
    ).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)


def _plot_history(history, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [x["epoch"] for x in history]
    losses = [x["loss"] for x in history]
    times = [x["epoch_seconds"] for x in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Behavioral cloning loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "loss.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, times)
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Behavioral cloning runtime")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "runtime.png",
        dpi=180,
    )
    plt.close()


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

    records = list(data)

    if not records:
        raise ValueError(
            "BC requires at least one training graph."
        )

    history = []

    total_episodes = (
        config.epochs
        * len(records)
        * config.episodes_per_graph
    )

    completed_episodes = 0

    training_start = time.perf_counter()

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        random.shuffle(records)

        losses = []
        steps = 0

        model.train()

        for record in records:
            for _ in range(
                config.episodes_per_graph
            ):
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

                    log_probs = distribution.logits

                    loss = -(
                        target.to(
                            log_probs.device
                        )
                        * log_probs
                    ).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(
                        float(
                            loss.detach().cpu()
                        )
                    )

                    steps += 1

                    env.step(
                        expert.sample_action(env)
                    )

                completed_episodes += 1

                progress = _progress_bar(
                    completed_episodes,
                    total_episodes,
                    config.progress_width,
                )

                print(
                    f"\rBC {progress} "
                    f"| epoch {epoch + 1:3d}/{config.epochs} "
                    f"| loss "
                    f"{sum(losses) / max(len(losses), 1):9.5f} "
                    f"| steps {steps:7d}",
                    end="",
                    flush=True,
                )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": (
                    sum(losses)
                    / max(len(losses), 1)
                ),
                "steps": steps,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": (
                    time.perf_counter()
                    - training_start
                ),
            }
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