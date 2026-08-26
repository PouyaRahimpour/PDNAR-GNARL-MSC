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

from src.gnarl.evaluation.evaluate import (
    evaluate_gnarl,
)
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
        "--output-dir",
        default="runs/evaluation",
    )

    args = parser.parse_args()

    model = load_checkpoint(
        args.checkpoint,
        args.device,
    )

    summary = []

    for size in SIZES:
        data = load_msc_split(
            "test",
            100,
            size,
            args.test_root,
        )

        output_dir = Path(
            args.output_dir
        ) / f"n_{size}"

        result = evaluate_gnarl(
            model,
            data,
            args.device,
            output_dir,
        )

        summary.append(
            {
                "n": size,
                "objective": result.mean_objective,
                "optimal_ratio": (
                    result.mean_optimal_ratio
                ),
                "classical_pd_ratio": (
                    result.mean_classical_pd_ratio
                ),
                "steps": result.mean_steps,
            }
        )

        print(
            f"\n"
            f"n={size:4d} "
            f"obj={result.mean_objective:.4f} "
            f"opt={result.mean_optimal_ratio:.6f} "
            f"pd={result.mean_classical_pd_ratio:.6f} "
            f"steps={result.mean_steps:.2f}"
        )

    import csv

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "summary.csv"
    ).open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    print(
        f"\nSaved evaluation results to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()