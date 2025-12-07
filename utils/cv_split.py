from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class FusionSplits:
    """Container for fusion CV splits."""

    folds: List[Tuple[np.ndarray, np.ndarray]]
    external_test_idx: np.ndarray


def create_fusion_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    n_external_patients: int = 4,
    n_splits: int = 5,
    random_state: int = 42,
) -> FusionSplits:
    """
    Create external test set and 5-fold patient-level CV for remaining patients.

    Args:
        df: DataFrame with at least a patient_id column.
    """
    rng = np.random.default_rng(random_state)
    unique_patients = df[patient_col].unique()
    rng.shuffle(unique_patients)

    external_patients = unique_patients[:n_external_patients]
    remaining_patients = unique_patients[n_external_patients:]

    external_mask = df[patient_col].isin(external_patients)
    external_test_idx = np.where(external_mask.values)[0]

    remaining_mask = df[patient_col].isin(remaining_patients)
    remaining_idx = np.where(remaining_mask.values)[0]

    groups = df.loc[remaining_idx, patient_col].values
    gkf = GroupKFold(n_splits=n_splits)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    # X 자리에 remaining_idx 자체를 쓰고, groups 인자 한 번만 전달
    for train_idx, val_idx in gkf.split(remaining_idx, groups=groups):
        folds.append((remaining_idx[train_idx], remaining_idx[val_idx]))

    return FusionSplits(folds=folds, external_test_idx=external_test_idx)


