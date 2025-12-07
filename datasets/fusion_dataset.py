from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class FusionSample:
    nail_image_path: str
    conj_image_path: str
    hb_value: float
    patient_id: str
    side: str  # e.g., "left", "right"
    age_z: float
    gender_binary: int  # 0/1


class FusionDataset(Dataset):
    def __init__(
        self,
        samples: List[FusionSample],
        nail_transform: Optional[Callable] = None,
        conj_transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.nail_transform = nail_transform
        self.conj_transform = conj_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        nail_img = Image.open(s.nail_image_path).convert("RGB")
        conj_img = Image.open(s.conj_image_path).convert("RGB")

        if self.nail_transform:
            nail_img = self.nail_transform(nail_img)
        if self.conj_transform:
            conj_img = self.conj_transform(conj_img)

        hb = float(s.hb_value)
        demo = np.array([s.age_z, float(s.gender_binary)], dtype=np.float32)
        return nail_img, conj_img, hb, s.patient_id, demo


def load_fusion_metadata(csv_path: str) -> List[FusionSample]:
    """
    Load ImageHB fusion metadata.

    Expected columns:
        nail_image_path, conj_image_path, hb_value, patient_id, side, age, gender

    Age will be z-scored across the dataset, gender encoded as 0/1.
    
    Note: CSV 파일의 nail_image_path와 conj_image_path는 절대 경로 또는 프로젝트 루트 기준 상대 경로를 사용할 수 있습니다.
    상대 경로를 사용하는 경우, 프로젝트 루트를 기준으로 경로를 지정하세요.
    """
    df = pd.read_csv(csv_path)

    # Convert relative paths to absolute paths if needed
    # If paths are relative, they should be relative to the project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if "nail_image_path" in df.columns:
        def _normalize_path(p: str) -> str:
            if isinstance(p, str):
                # If path is already absolute, use it as is
                if os.path.isabs(p):
                    return p
                # If path is relative, make it relative to project root
                return os.path.join(project_root, p)
            return p
        df["nail_image_path"] = df["nail_image_path"].apply(_normalize_path)
    
    if "conj_image_path" in df.columns:
        def _normalize_path(p: str) -> str:
            if isinstance(p, str):
                # If path is already absolute, use it as is
                if os.path.isabs(p):
                    return p
                # If path is relative, make it relative to project root
                return os.path.join(project_root, p)
            return p
        df["conj_image_path"] = df["conj_image_path"].apply(_normalize_path)

    # Encode gender
    # TODO: Adapt encoding if your dataset uses different labels.
    gender_map = {"M": 0, "F": 1}
    if "gender_binary" in df.columns:
        df["gender_binary"] = df["gender_binary"].astype(int)
    else:
        df["gender_binary"] = df["gender"].map(gender_map).fillna(0).astype(int)

    # Z-score age
    age_mean = df["age"].mean()
    age_std = df["age"].std() + 1e-8
    df["age_z"] = (df["age"] - age_mean) / age_std

    samples: List[FusionSample] = []
    for _, row in df.iterrows():
        samples.append(
            FusionSample(
                nail_image_path=row["nail_image_path"],
                conj_image_path=row["conj_image_path"],
                hb_value=float(row["hb_value"]),
                patient_id=str(row["patient_id"]),
                side=str(row["side"]),
                age_z=float(row["age_z"]),
                gender_binary=int(row["gender_binary"]),
            )
        )
    return samples


