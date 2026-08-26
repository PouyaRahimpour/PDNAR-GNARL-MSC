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
from src.gnarl.training.bc import (
    BCConfig,
    train_bc,
)
from src.gnarl.training.data import load_msc_split


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/gnarl_msc_bc.pt"
        ),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "runs/gnarl_msc_bc"
        ),
    )

    args = parser.parse_args()

    train = load_msc_split(
        "train",
        1000,
        16,
    )

    model = GNARLMSC()

    history = train_bc(
        model,
        train,
        BCConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
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
        f"Final BC loss: "
        f"{history[-1]['loss']:.6f}"
    )


if __name__ == "__main__":
    main()