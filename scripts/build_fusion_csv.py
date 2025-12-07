"""
Build fusion_meta.csv for the ImageHB multimodal dataset.

입력:
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/DATASAMPLE.csv
  * Image ID, Date Of Birth, Gender, Haemoglobin (gm/dL), ...
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/left_eye/{id}.jpeg
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/right_eye/{id}.jpeg
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/left_nail/{id}.jpeg
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/right_nail/{id}.jpeg

출력:
- /home/monetai/Desktop/URP(학룡)/multimodal_dataset/fusion_meta.csv

각 환자(Image ID)에 대해 4개의 pair를 만든다 (총 26 × 4 = 104 pairs):
- (left_nail,  left_eye)   -> side = "LL"
- (left_nail,  right_eye)  -> side = "LR"
- (right_nail, left_eye)   -> side = "RL"
- (right_nail, right_eye)  -> side = "RR"

각 행의 컬럼:
- nail_image_path, conj_image_path, hb_value, patient_id, side, age, gender
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/monetai/Desktop/URP(학룡)")
MM_ROOT = PROJECT_ROOT / "multimodal_dataset"


def compute_age_years(created_on: pd.Series, dob: pd.Series) -> pd.Series:
    """
    created_on, dob 문자열에서 나이를 (년 단위 float)로 계산.

    DATASAMPLE.csv 의 포맷이 살짝 복잡해서, 최대한 robust 하게 처리하고
    남는 NaN 은 전체 평균 나이로 채운다.
    """
    # Created On: "Tue Jun 21 2022 09:00:22 GMT+0530 (India Standard Time)" 형태
    created_dt = pd.to_datetime(created_on, errors="coerce", utc=True)
    # Date Of Birth: "09/12/2015" 형태
    dob_dt = pd.to_datetime(dob, errors="coerce", dayfirst=False)

    # 파싱 실패한 값들은 각각의 중앙값(또는 평균)으로 대체
    if created_dt.isna().all():
        # 전부 실패하면 임의 기준일 사용
        created_dt = pd.Series(
            pd.to_datetime("2022-01-01"), index=created_on.index
        )
    else:
        median_created = created_dt.dropna().median()
        created_dt = created_dt.fillna(median_created)

    if dob_dt.isna().all():
        # 전부 실패하면 임의 DOB 기준 (예: 2015-01-01)
        dob_dt = pd.Series(pd.to_datetime("2015-01-01"), index=dob.index)
    else:
        median_dob = dob_dt.dropna().median()
        dob_dt = dob_dt.fillna(median_dob)

    age_days = (created_dt - dob_dt).dt.days
    age_years = age_days / 365.25

    # 남은 NaN 이 있다면 전체 평균으로 채우기
    valid = age_years[~age_years.isna()]
    mean_age = valid.mean() if len(valid) > 0 else 0.0
    return age_years.fillna(mean_age)


def build_fusion_csv() -> None:
    src_csv = MM_ROOT / "DATASAMPLE.csv"
    out_csv = MM_ROOT / "fusion_meta.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Fusion metadata source not found: {src_csv}")

    df = pd.read_csv(src_csv)

    # 필수 컬럼 이름 가정:
    # Image ID, Date Of Birth, Gender, Haemoglobin (gm/dL)
    if "Image ID" not in df.columns or "Haemoglobin (gm/dL)" not in df.columns:
        raise RuntimeError("DATASAMPLE.csv 컬럼 이름이 예상과 다릅니다. (Image ID, Haemoglobin (gm/dL) 필요)")

    # patient_id 는 Image ID 를 문자열로 사용
    df["patient_id"] = df["Image ID"].astype(str)
    df["hb_value"] = df["Haemoglobin (gm/dL)"].astype(float)

    # 나이 계산
    if "Created On" in df.columns and "Date Of Birth" in df.columns:
        df["age_years"] = compute_age_years(df["Created On"], df["Date Of Birth"])
    else:
        # 정보가 없으면 일단 0으로 세팅
        df["age_years"] = 0.0

    # gender 컬럼 가정: "Gender" (M/F)
    if "Gender" in df.columns:
        df["gender_str"] = df["Gender"].astype(str).str.strip().str.upper()
    else:
        df["gender_str"] = "M"

    records = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        hb = row["hb_value"]
        age = float(row["age_years"])
        gender = row["gender_str"]

        # 각 view 의 이미지 경로 구성
        left_eye = (MM_ROOT / "left_eye" / f"{pid}.jpeg").resolve()
        right_eye = (MM_ROOT / "right_eye" / f"{pid}.jpeg").resolve()
        left_nail = (MM_ROOT / "left_nail" / f"{pid}.jpeg").resolve()
        right_nail = (MM_ROOT / "right_nail" / f"{pid}.jpeg").resolve()

        # 4개 pair 생성 (LL, LR, RL, RR)
        pairs = [
            (left_nail, left_eye, "LL"),
            (left_nail, right_eye, "LR"),
            (right_nail, left_eye, "RL"),
            (right_nail, right_eye, "RR"),
        ]

        for nail_path, conj_path, side in pairs:
            records.append(
                {
                    "nail_image_path": str(nail_path),
                    "conj_image_path": str(conj_path),
                    "hb_value": hb,
                    "patient_id": pid,
                    "side": side,
                    "age": age,
                    "gender": gender,
                }
            )

    out_df = pd.DataFrame.from_records(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[fusion] wrote {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    build_fusion_csv()


