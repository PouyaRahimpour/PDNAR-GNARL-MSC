"""Numerically safe masked categorical policies for graph actions."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Return a categorical distribution supported only on ``mask`` entries."""
    if logits.shape != mask.shape:
        raise ValueError(f"Logits {logits.shape} and mask {mask.shape} must agree.")
    if not bool(mask.any()):
        raise ValueError("An action distribution requires at least one valid action.")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


def greedy_action(logits: torch.Tensor, mask: torch.Tensor) -> int:
    return int(masked_categorical(logits, mask).probs.argmax().item())
