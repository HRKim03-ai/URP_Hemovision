from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ConjSample:
    image_path: str
    hb_value: float
    patient_id: str
    source: str  # e.g., "folder1", "folder2"


class ConjDataset(Dataset):
    def __init__(
        self,
        samples: List[ConjSample],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        hb = float(sample.hb_value)
        return img, hb, sample.patient_id, sample.source


def load_conj_metadata(
    csv_folder1: str,
    csv_folder2: str,
    hb_min: float = 8.0,
    hb_max: float = 16.0,
) -> List[ConjSample]:
    """
    Load and merge conjunctiva metadata from two folders, filter by Hb.

    Expected columns in both CSVs:
        image_path, hb_value, patient_id
    
    Note: CSV 파일의 image_path는 절대 경로 또는 프로젝트 루트 기준 상대 경로를 사용할 수 있습니다.
    상대 경로를 사용하는 경우, 프로젝트 루트를 기준으로 경로를 지정하세요.
    """
    df1 = pd.read_csv(csv_folder1)
    df1["source"] = "folder1"
    df2 = pd.read_csv(csv_folder2)
    df2["source"] = "folder2"

    df = pd.concat([df1, df2], ignore_index=True)
    
    # Check for NaN values in hb_value before filtering
    nan_count = df["hb_value"].isna().sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} rows with NaN hb_value. These will be filtered out.")
        df = df.dropna(subset=["hb_value"])
    
    # Convert relative paths to absolute paths if needed
    # If image_path is relative, it should be relative to the project root
    if "image_path" in df.columns:
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        def _normalize_path(p: str) -> str:
            if isinstance(p, str):
                # If path is already absolute, use it as is
                if os.path.isabs(p):
                    return p
                # If path is relative, make it relative to project root
                return os.path.join(project_root, p)
            return p
        
        df["image_path"] = df["image_path"].apply(_normalize_path)
    
    df = df[(df["hb_value"] >= hb_min) & (df["hb_value"] <= hb_max)]

    samples: List[ConjSample] = []
    for _, row in df.iterrows():
        hb_val = float(row["hb_value"])
        # Double check for NaN
        if np.isnan(hb_val):
            print(f"WARNING: Skipping row with NaN hb_value: {row.get('image_path', 'unknown')}")
            continue
        
        samples.append(
            ConjSample(
                image_path=row["image_path"],
                hb_value=hb_val,
                patient_id=str(row["patient_id"]),
                source=str(row["source"]),
            )
        )
    
    print(f"Loaded {len(samples)} valid samples from {csv_folder1} and {csv_folder2}")
    return samples


def split_conj_by_patient(
    samples: List[ConjSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[ConjSample], List[ConjSample], List[ConjSample]]:
    rng = np.random.default_rng(random_state)
    patient_ids = list({s.patient_id for s in samples})
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    train_samples = [s for s in samples if s.patient_id in train_ids]
    val_samples = [s for s in samples if s.patient_id in val_ids]
    test_samples = [s for s in samples if s.patient_id in test_ids]

    return train_samples, val_samples, test_samples


