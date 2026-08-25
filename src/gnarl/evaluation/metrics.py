from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MSCMetrics:
    instances: int
    mean_objective: float
    mean_optimal_ratio: float
    mean_classical_pd_ratio: float
    mean_steps: float