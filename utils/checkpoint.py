import os
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    metrics: Dict[str, float],
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: str, model: nn.Module | None = None, optimizer: Optimizer | None = None
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state


