import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from utils.logger import setup_logger
from utils.metrics import compute_regression_metrics


def load_predictions(pred_paths: Sequence[str]) -> List[np.ndarray]:
    """
    Load predictions stored as numpy .npy arrays or CSV.

    TODO: Adapt to your actual prediction storage format.
    """
    preds_list: List[np.ndarray] = []
    for path in pred_paths:
        if path.endswith(".npy"):
            preds = np.load(path)
        elif path.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(path)
            preds = df["pred"].values
        else:
            raise ValueError(f"Unsupported prediction file type: {path}")
        preds_list.append(preds)
    return preds_list


def tune_weights_grid_search(
    val_preds_list: List[np.ndarray],
    val_targets: np.ndarray,
    weight_ranges: List[Tuple[float, float]],
    n_steps: int = 5,
) -> np.ndarray:
    """
    Simple grid search over weights to minimize MAE on validation set.
    """
    num_models = len(val_preds_list)
    assert num_models == len(weight_ranges)
    best_mae = float("inf")
    best_weights = np.ones(num_models) / num_models

    grids = [np.linspace(lo, hi, n_steps) for (lo, hi) in weight_ranges]

    def recursive_search(idx: int, current_weights: List[float]):
        nonlocal best_mae, best_weights
        if idx == num_models:
            w = np.array(current_weights)
            w = w / (w.sum() + 1e-8)
            ensemble_pred = np.zeros_like(val_targets, dtype=float)
            for i in range(num_models):
                ensemble_pred += w[i] * val_preds_list[i]
            metrics = compute_regression_metrics(val_targets, ensemble_pred)
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_weights = w
            return
        for w_i in grids[idx]:
            recursive_search(idx + 1, current_weights + [float(w_i)])

    recursive_search(0, [])
    return best_weights


def ensemble_predict(config: Dict[str, Any]) -> None:
    logger = setup_logger("ensemble", config.get("log_file"))

    models_cfg = config["models"]
    val_targets = np.load(config["val_targets_path"])
    test_targets = np.load(config["test_targets_path"])

    val_pred_paths = [m["val_pred_path"] for m in models_cfg]
    test_pred_paths = [m["test_pred_path"] for m in models_cfg]
    val_preds_list = load_predictions(val_pred_paths)
    test_preds_list = load_predictions(test_pred_paths)

    weight_ranges = [tuple(m.get("weight_range", [0.0, 1.0])) for m in models_cfg]

    if config.get("weighting", "grid_search") == "grid_search":
        weights = tune_weights_grid_search(
            val_preds_list, val_targets, weight_ranges, n_steps=config.get("n_steps", 5)
        )
    else:
        maes = []
        for preds in val_preds_list:
            m = compute_regression_metrics(val_targets, preds)
            maes.append(m["mae"] + 1e-8)
        inv = 1.0 / np.array(maes)
        weights = inv / inv.sum()

    logger.info(f"Chosen ensemble weights: {weights.tolist()}")

    val_ensemble = np.zeros_like(val_targets, dtype=float)
    for i in range(len(val_preds_list)):
        val_ensemble += weights[i] * val_preds_list[i]
    val_metrics = compute_regression_metrics(val_targets, val_ensemble)
    logger.info(f"Validation ensemble metrics: {val_metrics}")

    test_ensemble = np.zeros_like(test_targets, dtype=float)
    for i in range(len(test_preds_list)):
        test_ensemble += weights[i] * test_preds_list[i]
    test_metrics = compute_regression_metrics(test_targets, test_ensemble)
    logger.info(f"Test ensemble metrics: {test_metrics}")

    out_dir = config.get("output_dir", "ensemble_results")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "val_ensemble_preds.npy"), val_ensemble)
    np.save(os.path.join(out_dir, "test_ensemble_preds.npy"), test_ensemble)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(
            {"val": val_metrics, "test": test_metrics, "weights": weights.tolist()},
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    ensemble_predict(config)


if __name__ == "__main__":
    main()


