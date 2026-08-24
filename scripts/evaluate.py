#!/usr/bin/env python3
"""Evaluate a GNARL-MSC checkpoint on PDNAR's in- and OOD test splits."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gnarl.evaluation.evaluate import evaluate_msc
from src.gnarl.models.gnarl_msc import GNARLMSC
from src.gnarl.training.data import load_msc_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = GNARLMSC()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"])
    for size in (16, 32, 64):
        metrics = evaluate_msc(model, load_msc_split("test", 100, size), args.device)
        print(f"n={size}: {metrics}")


if __name__ == "__main__":
    main()
