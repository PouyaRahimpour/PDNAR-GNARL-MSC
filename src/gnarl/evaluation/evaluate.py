from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.evaluation.metrics import MSCMetrics
from src.gnarl.policies.masked_policy import greedy_action


def aggregate_metrics(metrics):
    if not metrics:
        raise ValueError(
            "No metrics to aggregate."
        )

    return {
        "mean_objective": statistics.mean(
            m.mean_objective
            for m in metrics
        ),
        "std_objective": (
            statistics.stdev(
                m.mean_objective
                for m in metrics
            )
            if len(metrics) > 1
            else 0.0
        ),
        "mean_optimal_ratio": statistics.mean(
            m.mean_optimal_ratio
            for m in metrics
        ),
        "std_optimal_ratio": (
            statistics.stdev(
                m.mean_optimal_ratio
                for m in metrics
            )
            if len(metrics) > 1
            else 0.0
        ),
        "mean_classical_pd_ratio": statistics.mean(
            m.mean_classical_pd_ratio
            for m in metrics
        ),
        "std_classical_pd_ratio": (
            statistics.stdev(
                m.mean_classical_pd_ratio
                for m in metrics
            )
            if len(metrics) > 1
            else 0.0
        ),
    }


def classical_pd_cost(record) -> float:
    weights = (
        MSCEnvironment
        ._initial_weights(record)
        .float()
    )

    selection = (
        record.x_mask[:, -1]
        .flatten()
        .bool()
    )

    return float(
        weights[selection].sum().item()
    )


def _progress_bar(
    current,
    total,
    width=32,
):
    ratio = current / max(total, 1)
    filled = int(width * ratio)

    return (
        "["
        + "=" * filled
        + ">" * (filled < width)
        + " "
        * max(
            width
            - filled
            - (filled < width),
            0,
        )
        + "]"
        f" {100.0 * ratio:6.2f}%"
    )


def _write_results(
    results,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "instances.csv"

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


def _plot_results(
    results,
    output_dir,
):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    indices = [
        x["instance"]
        for x in results
    ]

    objectives = [
        x["objective"]
        for x in results
    ]

    optimal_ratios = [
        x["optimal_ratio"]
        for x in results
    ]

    classical_ratios = [
        x["classical_pd_ratio"]
        for x in results
    ]

    runtimes = [
        x["runtime_seconds"]
        for x in results
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(
        indices,
        objectives,
    )
    plt.xlabel("Test instance")
    plt.ylabel("Objective")
    plt.title("GNARL-MSC test objectives")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "objectives.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        indices,
        optimal_ratios,
        label="GNARL / optimum",
    )
    plt.plot(
        indices,
        classical_ratios,
        label="GNARL / classical PD",
    )
    plt.xlabel("Test instance")
    plt.ylabel("Ratio")
    plt.title("Solution quality ratios")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "ratios.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        indices,
        runtimes,
    )
    plt.xlabel("Test instance")
    plt.ylabel("Inference time (seconds)")
    plt.title("GNARL-MSC inference runtime")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "runtime.png",
        dpi=180,
    )
    plt.close()


@torch.no_grad()
def evaluate_gnarl(
    model,
    data,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> MSCMetrics:
    model.eval()
    model.to(device)

    objectives = []
    optimal_ratios = []
    classical_ratios = []
    steps = []

    instance_results = []

    total = len(data)

    evaluation_start = time.perf_counter()

    for instance_idx, record in enumerate(
        data,
        start=1,
    ):
        start = time.perf_counter()

        env = MSCEnvironment(
            record,
            device,
        )

        while not env.is_terminal():
            logits, _ = model(env)

            action = greedy_action(
                logits,
                env.action_mask(),
            )

            env.step(action)

        if not env.is_terminal():
            raise RuntimeError(
                "GNARL terminated without "
                "producing a valid set cover."
            )

        objective = float(env.objective)

        optimal = float(
            record.primal_optimal_weight
            .flatten()[0]
            .item()
        )

        classical = classical_pd_cost(
            record
        )

        runtime = (
            time.perf_counter()
            - start
        )

        optimal_ratio = (
            objective / optimal
        )

        classical_ratio = (
            objective / classical
        )

        objectives.append(objective)
        optimal_ratios.append(
            optimal_ratio
        )
        classical_ratios.append(
            classical_ratio
        )
        steps.append(
            env.state.step
        )

        instance_results.append(
            {
                "instance": instance_idx,
                "objective": objective,
                "optimal": optimal,
                "classical_pd": classical,
                "optimal_ratio": optimal_ratio,
                "classical_pd_ratio": classical_ratio,
                "steps": env.state.step,
                "runtime_seconds": runtime,
            }
        )

        elapsed = (
            time.perf_counter()
            - evaluation_start
        )

        avg_runtime = (
            elapsed / instance_idx
        )

        eta = (
            avg_runtime
            * (total - instance_idx)
        )

        print(
            f"\rTEST "
            f"{_progress_bar(instance_idx, total)} "
            f"| n={record.x_mask.shape[0]:4d} "
            f"| obj={objective:9.3f} "
            f"| opt={optimal_ratio:7.4f} "
            f"| pd={classical_ratio:7.4f} "
            f"| t={runtime:7.3f}s "
            f"| ETA={eta:7.1f}s",
            end="",
            flush=True,
        )

    print()

    n = len(objectives)

    if n == 0:
        raise ValueError(
            "Cannot evaluate an empty MSC split."
        )

    if output_dir is not None:
        _write_results(
            instance_results,
            output_dir,
        )

        _plot_results(
            instance_results,
            output_dir,
        )

    return MSCMetrics(
        instances=n,
        mean_objective=(
            sum(objectives) / n
        ),
        mean_optimal_ratio=(
            sum(optimal_ratios) / n
        ),
        mean_classical_pd_ratio=(
            sum(classical_ratios) / n
        ),
        mean_steps=(
            sum(steps) / n
        ),
    )