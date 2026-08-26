from __future__ import annotations

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.evaluation.metrics import MSCMetrics
from src.gnarl.policies.masked_policy import greedy_action

import statistics


def aggregate_metrics(metrics):
    if not metrics:
        raise ValueError("No metrics to aggregate.")

    return {
        "mean_objective": statistics.mean(
            m.mean_objective for m in metrics
        ),
        "std_objective": statistics.stdev(
            m.mean_objective for m in metrics
        ) if len(metrics) > 1 else 0.0,

        "mean_optimal_ratio": statistics.mean(
            m.mean_optimal_ratio for m in metrics
        ),
        "std_optimal_ratio": statistics.stdev(
            m.mean_optimal_ratio for m in metrics
        ) if len(metrics) > 1 else 0.0,

        "mean_classical_pd_ratio": statistics.mean(
            m.mean_classical_pd_ratio for m in metrics
        ),
        "std_classical_pd_ratio": statistics.stdev(
            m.mean_classical_pd_ratio for m in metrics
        ) if len(metrics) > 1 else 0.0,
    }

def classical_pd_cost(record) -> float:
    """
    Cost of the actual classical primal-dual approximation algorithm
    stored in the PDNAR dataset.
    """
    weights = MSCEnvironment._initial_weights(record).float()

    selection = record.x_mask[:, -1].flatten().bool()

    return float(weights[selection].sum().item())


@torch.no_grad()
def evaluate_gnarl(
    model,
    data,
    device: str = "cpu",
) -> MSCMetrics:
    model.eval()
    model.to(device)

    objectives = []
    optimal_ratios = []
    classical_ratios = []
    steps = []

    for record in data:
        env = MSCEnvironment(record, device)

        while not env.is_terminal():
            logits, _ = model(env)

            action = greedy_action(
                logits,
                env.action_mask(),
            )

            env.step(action)

        # This should be guaranteed by construction.
        if not env.is_terminal():
            raise RuntimeError(
                "GNARL terminated without producing a valid set cover."
            )

        objective = env.objective

        optimal = float(
            record.primal_optimal_weight.flatten()[0].item()
        )

        classical = classical_pd_cost(record)

        objectives.append(objective)
        optimal_ratios.append(objective / optimal)
        classical_ratios.append(objective / optimal)
        steps.append(env.state.step)

    n = len(objectives)

    if n == 0:
        raise ValueError("Cannot evaluate an empty MSC split.")

    return MSCMetrics(
        instances=n,
        mean_objective=sum(objectives) / n,
        mean_optimal_ratio=sum(optimal_ratios) / n,
        mean_classical_pd_ratio=sum(classical_ratios) / n,
        mean_steps=sum(steps) / n,
    )