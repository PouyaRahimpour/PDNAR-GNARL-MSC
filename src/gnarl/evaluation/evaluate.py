from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.evaluation.metrics import MSCMetrics
from src.gnarl.policies.masked_policy import greedy_action


def _mean_std(values):
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_metrics(metrics):
    """Pool per-instance aggregates exactly using within/between-group variance."""
    if not metrics:
        raise ValueError("No metrics to aggregate.")
    fields = ("objective", "optimal_ratio", "classical_pd_ratio", "steps")
    result = {"instances": sum(m.instances for m in metrics)}
    for field in fields:
        ns = [m.instances for m in metrics]
        means = [getattr(m, f"mean_{field}") for m in metrics]
        total_n = sum(ns)
        mean = sum(n * x for n, x in zip(ns, means)) / total_n
        m2 = 0.0
        for metric, n, local_mean in zip(metrics, ns, means):
            local_std = getattr(metric, f"std_{field}")
            m2 += max(n - 1, 0) * local_std**2
            m2 += n * (local_mean - mean) ** 2
        std = (m2 / (total_n - 1)) ** 0.5 if total_n > 1 else 0.0
        result[f"mean_{field}"] = mean
        result[f"std_{field}"] = std
    return result


def classical_pd_cost(record) -> float:
    weights = MSCEnvironment._initial_weights(record).float()
    selection = record.x_mask[:, -1].flatten().bool()
    return float(weights[selection].sum().item())


def _progress_bar(current, total, width=32):
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    return "[" + "=" * filled + ">" * (filled < width) + " " * max(width - filled - (filled < width), 0) + "]" f" {100.0 * ratio:6.2f}%"


def write_results(results, output_dir, filename="instances.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return path


def plot_results(results, output_dir, title_prefix="GNARL-MSC"):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = [x["instance"] for x in results]
    objectives = [x["objective"] for x in results]
    optimal_ratios = [x["optimal_ratio"] for x in results]
    classical_ratios = [x["classical_pd_ratio"] for x in results]
    runtimes = [x["runtime_seconds"] for x in results]

    plt.figure(figsize=(10, 5)); plt.plot(indices, objectives)
    plt.xlabel("Test instance"); plt.ylabel("Objective"); plt.title(f"{title_prefix} test objectives")
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(output_dir / "objectives.png", dpi=180); plt.close()

    plt.figure(figsize=(10, 5)); plt.plot(indices, optimal_ratios, label="Model / optimum"); plt.plot(indices, classical_ratios, label="Model / classical PD")
    plt.xlabel("Test instance"); plt.ylabel("Ratio"); plt.title("Solution quality ratios")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(output_dir / "ratios.png", dpi=180); plt.close()

    plt.figure(figsize=(10, 5)); plt.plot(indices, runtimes)
    plt.xlabel("Test instance"); plt.ylabel("Inference time (seconds)"); plt.title(f"{title_prefix} inference runtime")
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(output_dir / "runtime.png", dpi=180); plt.close()


@torch.no_grad()
def evaluate_gnarl(model, data, device="cpu", output_dir=None, title_prefix="GNARL-MSC") -> MSCMetrics:
    model.eval(); model.to(device)
    objectives, optimal_ratios, classical_ratios, steps, instance_results = [], [], [], [], []
    total = len(data); evaluation_start = time.perf_counter()

    for instance_idx, record in enumerate(data, start=1):
        start = time.perf_counter(); env = MSCEnvironment(record, device)
        while not env.is_terminal():
            logits, _ = model(env)
            env.step(greedy_action(logits, env.action_mask()))

        objective = float(env.objective)
        optimal = float(record.primal_optimal_weight.flatten()[0].item())
        classical = classical_pd_cost(record)
        runtime = time.perf_counter() - start
        optimal_ratio = objective / optimal
        classical_ratio = objective / classical
        objectives.append(objective); optimal_ratios.append(optimal_ratio); classical_ratios.append(classical_ratio); steps.append(env.state.step)
        instance_results.append({"instance": instance_idx, "objective": objective, "optimal": optimal, "classical_pd": classical, "optimal_ratio": optimal_ratio, "classical_pd_ratio": classical_ratio, "steps": env.state.step, "runtime_seconds": runtime})

        elapsed = time.perf_counter() - evaluation_start; eta = elapsed / instance_idx * (total - instance_idx)
        print(f"\rTEST {_progress_bar(instance_idx, total)} | n={record.x_mask.shape[0]:4d} | obj={objective:9.3f} | opt={optimal_ratio:7.4f} | pd={classical_ratio:7.4f} | t={runtime:7.3f}s | ETA={eta:7.1f}s", end="", flush=True)
    print()
    if not objectives: raise ValueError("Cannot evaluate an empty MSC split.")

    if output_dir is not None:
        write_results(instance_results, output_dir)
        plot_results(instance_results, output_dir, title_prefix)

    values = {"objective": objectives, "optimal_ratio": optimal_ratios, "classical_pd_ratio": classical_ratios, "steps": steps}
    stats = {}
    for name, xs in values.items():
        stats[f"mean_{name}"], stats[f"std_{name}"] = _mean_std(xs)
    return MSCMetrics(instances=len(objectives), **stats)
