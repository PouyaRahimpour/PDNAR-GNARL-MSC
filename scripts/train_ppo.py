#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.gnarl.models.gnarl_msc import GNARLMSC
from src.gnarl.training.data import load_msc_split
from src.gnarl.training.ppo import (
    PPOConfig,
    train_ppo,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/gnarl_msc_ppo.pt"
        ),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "runs/gnarl_msc_ppo"
        ),
    )

    args = parser.parse_args()

    model = GNARLMSC()

    if args.checkpoint:
        model.load_state_dict(
            torch.load(
                args.checkpoint,
                map_location=args.device,
            )["model"]
        )

    train = load_msc_split(
        "train",
        1000,
        16,
    )

    history = train_ppo(
        model,
        train,
        PPOConfig(
            total_steps=args.steps,
            device=args.device,
            output_dir=str(args.run_dir),
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
        },
        args.output,
    )

    print(
        f"Saved {args.output}"
    )

    print(
        f"Final reward: "
        f"{history[-1]['mean_reward']:.6f}"
    )

    print(
        f"Training time: "
        f"{history[-1]['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()