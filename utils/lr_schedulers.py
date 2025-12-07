import math
from typing import List


def lr_warmup_cosine(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    """
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    # cosine decay
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_param_group_lrs(param_groups, lrs: List[float]) -> None:
    """
    Assign learning rates to optimizer param groups.
    """
    assert len(param_groups) == len(lrs)
    for pg, lr in zip(param_groups, lrs):
        pg["lr"] = lr


