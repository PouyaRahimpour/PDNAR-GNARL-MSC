from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MSCMetrics:
    instances: int
    mean_objective: float
    std_objective: float
    mean_optimal_ratio: float
    std_optimal_ratio: float
    mean_classical_pd_ratio: float
    std_classical_pd_ratio: float
    mean_steps: float
    std_steps: float
