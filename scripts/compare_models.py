#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "objective": {
        "mean": "objective_mean",
        "std": "objective_std",
        "label": "Objective",
    },
    "optimal_ratio": {
        "mean": "optimal_ratio_mean",
        "std": "optimal_ratio_std",
        "label": "Optimal Ratio (Objective / Optimal)",
    },
    "classical_pd_ratio": {
        "mean": "classical_pd_ratio_mean",
        "std": "classical_pd_ratio_std",
        "label": "Classical Primal-Dual Ratio",
    },
    "steps": {
        "mean": "steps_mean",
        "std": "steps_std",
        "label": "Inference Steps",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple MSC models using evaluation summaries."
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=list(METRICS.keys()),
        help="Metric to compare.",
    )

    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="NAME=CSV",
        help=(
            "Model name and summary CSV. "
            "Example: GNARL_BC=runs/eval_bc/summary.csv"
        ),
    )

    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="For OOD results, restrict comparison to this b value.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/model_comparison",
        help="Directory for generated plots.",
    )

    return parser.parse_args()


def parse_results(items):
    results = {}

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid result specification: {item!r}. "
                "Expected NAME=CSV_PATH."
            )

        name, path = item.split("=", 1)

        if not name:
            raise ValueError(f"Empty model name in {item!r}.")

        if not path:
            raise ValueError(f"Empty CSV path in {item!r}.")

        results[name] = Path(path)

    return results


def load_summary(path: Path, metric: str, b: int | None):
    if not path.exists():
        raise FileNotFoundError(
            f"Result file does not exist:\n  {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Result file is empty:\n  {path}")

    spec = METRICS[metric]

    mean_col = spec["mean"]
    std_col = spec["std"]

    missing = [
        col
        for col in (mean_col, std_col)
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} does not contain the required columns: {missing}\n"
            f"Available columns:\n  {list(df.columns)}"
        )

    if b is not None:
        if "b" not in df.columns:
            raise ValueError(
                f"--b={b} was supplied, but {path} has no 'b' column."
            )

        df = df[df["b"] == b].copy()

        if df.empty:
            raise ValueError(
                f"No rows with b={b} found in {path}."
            )

    # The evaluation summaries use n as the x-axis.
    if "n" not in df.columns:
        raise ValueError(
            f"{path} has no 'n' column. "
            f"Available columns: {list(df.columns)}"
        )

    result = pd.DataFrame(
        {
            "n": pd.to_numeric(df["n"], errors="coerce"),
            "mean": pd.to_numeric(
                df[mean_col],
                errors="coerce",
            ),
            "std": pd.to_numeric(
                df[std_col],
                errors="coerce",
            ),
        }
    )

    result = result.dropna(
        subset=["n", "mean", "std"]
    )

    if result.empty:
        raise ValueError(
            f"No valid {metric} data found in {path}."
        )

    result = (
        result
        .sort_values("n")
        .reset_index(drop=True)
    )

    return result


def load_all(results, metric, b):
    loaded = {}

    for name, path in results.items():
        loaded[name] = load_summary(
            path=path,
            metric=metric,
            b=b,
        )

    return loaded


def print_results(results, metric):
    label = METRICS[metric]["label"]

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    for name, df in results.items():
        print(f"\n{name}")

        for _, row in df.iterrows():
            print(
                f"  n={int(row['n']):4d}  "
                f"{row['mean']:.6f} ± {row['std']:.6f}"
            )

    print("=" * 80)
    print()


def make_plot(
    results,
    metric,
    output_path,
    b=None,
):
    spec = METRICS[metric]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for name, df in results.items():
        x = df["n"].to_numpy()
        mean = df["mean"].to_numpy()
        std = df["std"].to_numpy()

        line = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=name,
        )[0]

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.15,
        )

    ax.set_xlabel("n")
    ax.set_ylabel(spec["label"])

    if b is None:
        ax.set_title(
            f"Model Comparison — {spec['label']}"
        )
    else:
        ax.set_title(
            f"Model Comparison — {spec['label']} (b={b})"
        )

    ax.set_xscale("log", base=2)
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    # For both ratios, 1 means "same objective as the reference".
    if metric in {
        "optimal_ratio",
        "classical_pd_ratio",
    }:
        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.5,
            label="Ratio = 1",
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    result_paths = parse_results(
        args.results
    )

    results = load_all(
        results=result_paths,
        metric=args.metric,
        b=args.b,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"compare_{args.metric}"
    )

    if args.b is not None:
        filename += f"_b{args.b}"

    output_path = (
        output_dir / f"{filename}.png"
    )

    make_plot(
        results=results,
        metric=args.metric,
        output_path=output_path,
        b=args.b,
    )

    print_results(
        results=results,
        metric=args.metric,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()