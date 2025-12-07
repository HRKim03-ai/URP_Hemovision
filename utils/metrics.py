from typing import Dict

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2) + eps
    if denom <= eps:
        return 0.0
    return float(1.0 - ss_res / denom)


def acc_delta(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= delta))


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    # Check for NaN or empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "mae": float("nan"),
            "r2": float("nan"),
            "acc@0.5": 0.0,
            "acc@1.0": 0.0,
            "acc@2.0": 0.0,
        }
    
    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return {
            "mae": float("nan"),
            "r2": float("nan"),
            "acc@0.5": 0.0,
            "acc@1.0": 0.0,
            "acc@2.0": 0.0,
        }
    
    return {
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "acc@0.5": acc_delta(y_true, y_pred, 0.5),
        "acc@1.0": acc_delta(y_true, y_pred, 1.0),
        "acc@2.0": acc_delta(y_true, y_pred, 2.0),
    }


def tensor_to_numpy(t: torch.Tensor):
    """Convert tensor to numpy array, ensuring 1D output for regression predictions."""
    arr = t.detach().cpu().numpy()
    # Flatten to 1D if needed (for regression outputs)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


