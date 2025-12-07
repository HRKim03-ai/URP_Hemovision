import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: Whether to set CuDNN to deterministic mode.
        benchmark: Whether to enable CuDNN benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


