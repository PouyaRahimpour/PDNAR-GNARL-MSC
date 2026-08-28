#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import torch
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)
from src.gnarl.evaluation.evaluate import evaluate_gnarl
from src.gnarl.models.gnarl_msc import GNARLMSC


def load_checkpoint(
    checkpoint: str,
    device: str,
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

    if "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "checkpoint"
    )

    parser.add_argument(
        "--dataset-root",
        default="dataset/pdnar_ood",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/evaluation_ood",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--b",
        nargs="+",
        type=int,
        default=[3, 8],
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[
            16,
            128,
            512,
            1024,
        ],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    model = load_checkpoint(
        args.checkpoint,
        args.device,
    )

    dataset_root = Path(
        args.dataset_root
    )

    output_root = Path(
        args.output_dir
    )

    summary = []

    for b in args.b:

        for size in args.sizes:

            for seed in range(args.seeds):

                dataset_path = (
                    dataset_root
                    / f"b_{b}"
                    / f"n_{size}"
                    / f"test_50_seed_{seed}.pkl"
                )

                if not dataset_path.exists():
                    raise FileNotFoundError(
                        f"Missing OOD dataset:\n"
                        f"  {dataset_path}"
                    )

                print(
                    "\n"
                    + "=" * 72
                )

                print(
                    f"OOD evaluation | "
                    f"b={b} | "
                    f"n={size} | "
                    f"seed={seed}"
                )

                data = load_pickle(
                    dataset_path
                )

                result_dir = (
                    output_root
                    / f"b_{b}"
                    / f"n_{size}"
                    / f"seed_{seed}"
                )

                result = evaluate_gnarl(
                    model,
                    data,
                    args.device,
                    result_dir,
                )

                row = {
                    "b": b,
                    "n": size,
                    "seed": seed,
                    "instances": result.instances,
                    "objective": result.mean_objective,
                    "optimal_ratio": (
                        result.mean_optimal_ratio
                    ),
                    "classical_pd_ratio": (
                        result.mean_classical_pd_ratio
                    ),
                    "steps": result.mean_steps,
                }

                summary.append(row)

                print(
                    f"\n"
                    f"b={b:2d} "
                    f"n={size:4d} "
                    f"seed={seed:2d} | "
                    f"obj={row['objective']:.6f} | "
                    f"opt={row['optimal_ratio']:.6f} | "
                    f"pd={row['classical_pd_ratio']:.6f} | "
                    f"steps={row['steps']:.2f}"
                )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_root
        / "summary.csv"
    )

    with summary_path.open(
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
        f"\nSaved summary to "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()