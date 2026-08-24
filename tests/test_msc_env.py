import torch
from torch_geometric.data import Data

from src.gnarl.envs.msc_env import MSCEnvironment


def record():
    # Sets: {e0}, {e0,e1}, {e1}; optimal cover is set 1 at cost 2.
    return Data(
        x=torch.tensor([[[1.0]], [[2.0]], [[1.0]]]),
        y=torch.zeros((2, 1, 1)),
        edge_index=torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]]),
        primal_optimal_solution=torch.tensor([0, 1, 0]),
        primal_optimal_weight=torch.tensor([2.0]),
        x_mask=torch.tensor([[[0.0]], [[1.0]], [[0.0]]]),
    )


def test_transition_covers_incident_elements_and_shapes_reward():
    env = MSCEnvironment(record())
    state, reward, done, info = env.step(1)
    assert state.selected.tolist() == [False, True, False]
    assert state.covered.tolist() == [True, True]
    assert reward == -2.0
    assert done and info["coverage"] == 1.0
