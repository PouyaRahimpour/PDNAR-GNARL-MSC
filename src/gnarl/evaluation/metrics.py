from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MSCMetrics:
    instances: int
    feasible_rate: float
    mean_objective: float
    mean_optimal_ratio: float
    mean_primal_dual_ratio: float
    mean_steps: float
