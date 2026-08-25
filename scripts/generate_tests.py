#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np

from src.dataset.graph import generate_bipartite_graphs
from src.dataset.algorithms.set_cover import set_cover


SIZES = [16, 32, 64, 128, 256, 512, 1024]
SEEDS = list(range(10))
B_VALUES = [5, 3, 8]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_split(
    *,
    seed: int,
    b: int,
    size: int,
    samples: int,
    output_dir: Path,
) -> None:
    set_seed(seed)

    graphs = generate_bipartite_graphs(
        num_graphs=samples,
        num_nodes=size,
        b=b,
    )

    data = [
        set_cover(graph)
        for graph in graphs
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / (
        f"test_{samples}_{size}_b{b}.pkl"
    )

    with path.open("wb") as f:
        pickle.dump(data, f)

    print(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="dataset/set_cover",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    root = Path(args.output)

    for seed in SEEDS:
        for b in B_VALUES:
            for size in SIZES:
                generate_split(
                    seed=seed,
                    b=b,
                    size=size,
                    samples=args.samples,
                    output_dir=root / f"seed_{seed}",
                )


if __name__ == "__main__":
    main()