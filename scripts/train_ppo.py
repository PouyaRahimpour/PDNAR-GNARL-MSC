#!/usr/bin/env python3
"""Reward-only GNARL-MSC PPO training (optionally warm-started from BC)."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gnarl.models.gnarl_msc import GNARLMSC
from src.gnarl.training.data import load_msc_split
from src.gnarl.training.ppo import PPOConfig, train_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/gnarl_msc_ppo.pt"))
    args = parser.parse_args()
    model = GNARLMSC()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"])
    history = train_ppo(model, load_msc_split("train", 1000, 16), PPOConfig(total_steps=args.steps, device=args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, args.output)
    print(f"Saved {args.output}; final rollout reward={history[-1]['mean_reward']:.6f}")


if __name__ == "__main__":
    main()
