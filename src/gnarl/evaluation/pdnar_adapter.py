from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch


class PDNARModel:
    def __init__(
        self,
        pdnar_root: str | Path,
        checkpoint: str | Path,
        device: str = "cpu",
        hidden_dim: int = 32,
        eps: bool = False,
    ):
        self.device = torch.device(device)

        pdnar_root = Path(pdnar_root).resolve()
        sys.path.insert(0, str(pdnar_root))

        from src.model.graph_executor import GraphNeuralExecutor

        self.model = GraphNeuralExecutor(
            hidden_dim=hidden_dim,
            eps=eps,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint,
            map_location=self.device,
        )

        if "state_dict" in checkpoint:
            state_dict = {
                k.removeprefix("model."): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def solve(self, record):
        record = record.to(self.device)

        batch = Batch.from_data_list(
            [record],
            follow_batch=["x", "y"],
        )

        results = self.model.test(batch)

        pred_set = results["pred_set"].squeeze(-1).bool()

        weights = batch.x[:, 0]

        objective = float(
            weights[pred_set].sum().item()
        )

        optimal = float(
            batch.primal_optimal_weight.sum().item()
        )

        classical = float(
            weights[batch.x_mask[:, -1].bool()].sum().item()
        )

        return {
            "objective": objective,
            "optimal_ratio": objective / optimal,
            "classical_ratio": objective / classical,
            "selection": pred_set.cpu(),
        }