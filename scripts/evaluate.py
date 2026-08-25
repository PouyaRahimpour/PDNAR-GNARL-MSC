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

from src.gnarl.evaluation.evaluate import (
    evaluate_gnarl,
    aggregate_metrics,
)

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

    parser.add_argument(
        "checkpoint"
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--test-root",
        default="dataset/set_cover",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--b",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    model = load_checkpoint(
        args.checkpoint,
        args.device,
    )

    seeds = (
        [args.seed]
        if args.seed is not None
        else list(range(10))
    )

    for b in [args.b]:

        print(f"\n===== b={b} =====")

        for size in SIZES:

            metrics = []

            for seed in seeds:

                if b == 5:
                    path = (
                        Path(args.test_root)
                        / f"seed_{seed}"
                        / f"test_100_{size}_b5.pkl"
                    )
                else:
                    path = (
                        Path(args.test_root)
                        / f"seed_{seed}"
                        / f"test_100_{size}_b{b}.pkl"
                    )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing test split: {path}"
                    )

                import pickle

                with path.open("rb") as f:
                    data = pickle.load(f)

                result = evaluate_gnarl(
                    model,
                    data,
                    args.device,
                )

                metrics.append(result)

                print(
                    f"seed={seed:02d} "
                    f"n={size:4d} "
                    f"opt={result.mean_optimal_ratio:.6f} "
                    f"pd={result.mean_classical_pd_ratio:.6f}"
                )

            summary = aggregate_metrics(metrics)

            print(
                f"n={size:4d} "
                f"opt={summary['mean_optimal_ratio']:.4f}"
                f" ± {summary['std_optimal_ratio']:.4f} "
                f"pd={summary['mean_classical_pd_ratio']:.4f}"
                f" ± {summary['std_classical_pd_ratio']:.4f}"
            )

if __name__ == "__main__":
    main()
