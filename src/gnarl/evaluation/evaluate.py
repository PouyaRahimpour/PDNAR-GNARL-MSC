"""Matched MSC evaluation using PDNAR's stored optima and primal-dual output."""

from __future__ import annotations

import torch

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.evaluation.metrics import MSCMetrics
from src.gnarl.policies.masked_policy import greedy_action


def _pdnar_primal_dual_cost(record) -> float:
    weights = MSCEnvironment._initial_weights(record).float()
    # Final x_mask is the selected solution constructed by PDNAR's fixed
    # primal-dual algorithm; it is a baseline only, never model input here.
    selection = record.x_mask[:, -1].flatten().bool()
    return float(weights[selection].sum().item())


@torch.no_grad()
def evaluate_msc(model, data, device: str = "cpu") -> MSCMetrics:
    model.eval()
    model.to(device)
    objectives, ratios, baseline_ratios, steps, feasible = [], [], [], [], []
    for record in data:
        env = MSCEnvironment(record, device)
        while not env.is_terminal() and env.state.step < env.num_sets:
            logits, _ = model(env)
            env.step(greedy_action(logits, env.action_mask()))
        objective = env.objective
        optimal = float(record.primal_optimal_weight.flatten()[0].item())
        objectives.append(objective)
        ratios.append(objective / optimal)
        baseline_ratios.append(_pdnar_primal_dual_cost(record) / optimal)
        steps.append(env.state.step)
        feasible.append(float(env.is_terminal()))
    n = len(objectives)
    if n == 0:
        raise ValueError("Cannot evaluate an empty MSC split.")
    return MSCMetrics(
        instances=n,
        feasible_rate=sum(feasible) / n,
        mean_objective=sum(objectives) / n,
        mean_optimal_ratio=sum(ratios) / n,
        mean_primal_dual_ratio=sum(baseline_ratios) / n,
        mean_steps=sum(steps) / n,
    )
