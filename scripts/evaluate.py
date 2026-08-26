#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.gnarl.evaluation.evaluate import evaluate_gnarl
from src.gnarl.models.gnarl_msc import GNARLMSC
from src.gnarl.training.data import load_msc_split

SIZES = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
]


def load_checkpoint(
    checkpoint,
    device,
):
    model = GNARLMSC(
        hidden_dim=64,
        message_passing_rounds=4,
        pooling="mean",
    )

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state["model"],
        strict=True,
    )

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint")

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--test-root",
        default="dataset/set_cover",
    )

    args = parser.parse_args()

    model = load_checkpoint(
        args.checkpoint,
        args.device,
    )

    for size in SIZES:
        data = load_msc_split(
            "test",
            100,
            size,
            args.test_root,
        )

        result = evaluate_gnarl(
            model,
            data,
            args.device,
        )

        print(
            f"n={size:4d} "
            f"opt={result.mean_optimal_ratio:.6f} "
            f"pd={result.mean_classical_pd_ratio:.6f}"
        )

if __name__ == "__main__":
    main()
