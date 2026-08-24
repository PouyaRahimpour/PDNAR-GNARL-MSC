"""Load existing PDNAR MSC splits without changing their generation."""

from __future__ import annotations

import pickle
from pathlib import Path


def load_msc_split(split: str, samples: int, nodes: int, dataset_dir: str | Path = "dataset/set_cover"):
    path = Path(dataset_dir) / f"{split}_{samples}_{nodes}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"PDNAR MSC split not found: {path}. Generate it once with "
            "`python main.py data.algorithm=set_cover` before GNARL training."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)
