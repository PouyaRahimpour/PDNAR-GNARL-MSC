"""Sequential minimum-set-cover environment over PDNAR ``BipartiteData``.

The environment deliberately consumes the data emitted by
``src.dataset.algorithms.set_cover``.  It does not regenerate graphs, change
weights, or use PDNAR's primal-dual hint trajectory as a state representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class MSCState:
    """Markov state for a single PDNAR MSC instance."""

    selected: torch.Tensor
    covered: torch.Tensor
    step: int


class MSCEnvironment:
    """Choose one unselected set per step until every element is covered.

    Set nodes are PDNAR's ``x`` nodes and element nodes are its ``y`` nodes.
    The action mask excludes previously selected sets; therefore each episode
    is feasible by construction when the source instance is feasible.  The
    shaped reward is the negative incremental set cost, so cumulative reward
    equals the negative objective value.
    """

    def __init__(self, data, device: torch.device | str = "cpu"):
        self.data = data
        self.device = torch.device(device)
        self.weights = self._initial_weights(data).to(self.device).float()
        self.edge_index = data.edge_index.to(self.device).long()
        self.num_sets = self.weights.numel()
        self.num_elements = int(data.y.size(0))
        self._element_to_sets = self.edge_index[0]
        self._set_to_elements = self.edge_index[1]
        self.state = self.reset()

    @staticmethod
    def _initial_weights(data: object) -> torch.Tensor:
        x = data.x
        # PDNAR stores x as [num_sets, timesteps, 1].  Keeping this fallback
        # makes the environment usable with a minimal hand-built test record.
        if x.dim() == 3:
            return x[:, 0, 0]
        if x.dim() == 2:
            return x[:, 0]
        return x

    def reset(self) -> MSCState:
        self.state = MSCState(
            selected=torch.zeros(self.num_sets, dtype=torch.bool, device=self.device),
            covered=torch.zeros(self.num_elements, dtype=torch.bool, device=self.device),
            step=0,
        )
        return self.state

    def action_mask(self, state: MSCState | None = None) -> torch.Tensor:
        state = self.state if state is None else state
        return ~state.selected

    def is_terminal(self, state: MSCState | None = None) -> bool:
        state = self.state if state is None else state
        return bool(state.covered.all())

    def step(self, action: int | torch.Tensor) -> Tuple[MSCState, float, bool, Dict[str, float]]:
        action = int(action)
        if self.is_terminal():
            raise RuntimeError("Cannot step a terminated MSC episode; call reset().")
        if action < 0 or action >= self.num_sets or self.state.selected[action]:
            raise ValueError(f"Invalid MSC action {action}.")

        selected = self.state.selected.clone()
        covered = self.state.covered.clone()
        selected[action] = True
        covered[self._element_to_sets[self._set_to_elements == action]] = True
        self.state = MSCState(selected=selected, covered=covered, step=self.state.step + 1)
        reward = -float(self.weights[action].item())
        done = self.is_terminal()
        return self.state, reward, done, {"cost": -reward, "coverage": float(covered.float().mean())}

    @property
    def objective(self) -> float:
        return float(self.weights[self.state.selected].sum().item())

    def snapshot(self) -> MSCState:
        return MSCState(self.state.selected.clone(), self.state.covered.clone(), self.state.step)
