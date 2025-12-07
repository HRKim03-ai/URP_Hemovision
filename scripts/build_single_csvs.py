"""
Build helper CSVs for single-modal training from the raw metadata files.

This script reads the existing metadata inside:

- /home/monetai/Desktop/URP/singlemodal_dataset/nail/2/0images_index_with_hb.csv
- /home/monetai/Desktop/URP/singlemodal_dataset/conj/2/0images_index_with_hb_c_2.csv

and writes simplified CSVs that the training code expects:

- /home/monetai/Desktop/URP/singlemodal_dataset/nail_meta.csv
- /home/monetai/Desktop/URP/singlemodal_dataset/conj_folder1.csv
- /home/monetai/Desktop/URP/singlemodal_dataset/conj_folder2.csv

You can run this once to generate the CSVs:

    python scripts/build_single_csvs.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/monetai/Desktop/URP")


def build_nail_csv() -> None:
    """
    Build nail_meta.csv from 0images_index_with_hb.csv.

    Output columns: image_path, hb_value, patient_id
    - image_path: absolute path under singlemodal_dataset/nail/2
    - hb_value: hb_value (G/dL)
    - patient_id: use subject_id (일단 subject_id를 환자 ID로 사용)
    """
    nail_root = PROJECT_ROOT / "singlemodal_dataset" / "nail" / "2"
    src_csv = nail_root / "0images_index_with_hb.csv"
    out_csv = PROJECT_ROOT / "singlemodal_dataset" / "nail_meta.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Nail metadata CSV not found: {src_csv}")

    df = pd.read_csv(src_csv)

    # Hb 값을 확실히 float 로 만들고, 숫자가 아닌 행은 제거
    df["hb_value"] = pd.to_numeric(df["hb_value (G/dL)"], errors="coerce")
    df = df.dropna(subset=["hb_value"])

    # 기존 Windows 경로 대신, 현재 폴더에 있는 파일명을 사용해서 절대경로 생성
    def make_local_path(old_path: str, subject_id: str) -> str:
        # 보통 old_path 끝에 파일명이 들어있으므로, 그걸 우선 사용
        fname = os.path.basename(str(old_path))
        if fname == "" or not (nail_root / fname).exists():
            # 혹시 모를 경우 subject_id 기반으로 fallback
            fname = f"{subject_id}.png"
        return str((nail_root / fname).resolve())

    df_out = pd.DataFrame()
    df_out["image_path"] = [
        make_local_path(p, sid) for p, sid in zip(df["image_path"], df["subject_id"])
    ]
    df_out["hb_value"] = df["hb_value"].astype(float)
    df_out["patient_id"] = df["subject_id"].astype(str)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[nail] wrote {len(df_out)} rows to {out_csv}")


def build_conj_csvs() -> None:
    """
    Build conj_folder1.csv and conj_folder2.csv from 0images_index_with_hb_c_2.csv.

    설명:
    - subject_id: c_2_1_xxx or c_2_2_xxx 형태
        - "_1_" 포함: Folder1 (예: whole eye + ROI)
        - "_2_" 포함: Folder2 (예: ROI-only)
    - image_path: 현재 폴더 구조에 맞춰 생성
        - conj/2/1/c_2_1_xxx.png
        - conj/2/2/c_2_2_xxx.png
    - hb_value: hb_value (G/dL)
    - patient_id: subject_num 사용 (동일 subject_num 이 같은 환자)
    - Hb 필터: [8, 16] g/dL 범위만 사용 (spec에 맞춤)
    """
    conj_root = PROJECT_ROOT / "singlemodal_dataset" / "conj" / "2"
    src_csv = conj_root / "0images_index_with_hb_c_2.csv"
    out1_csv = PROJECT_ROOT / "singlemodal_dataset" / "conj_folder1.csv"
    out2_csv = PROJECT_ROOT / "singlemodal_dataset" / "conj_folder2.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Conj metadata CSV not found: {src_csv}")

    df = pd.read_csv(src_csv)

    # Hb 필터: [8, 16]
    df = df[(df["hb_value (G/dL)"] >= 8.0) & (df["hb_value (G/dL)"] <= 16.0)].copy()

    def make_local_conj_path(row) -> str:
        sid = str(row["subject_id"])
        if "_1_" in sid:
            base = conj_root / "1"
        elif "_2_" in sid:
            base = conj_root / "2"
        else:
            # 예외적으로 다른 패턴이 있으면 1 폴더로 fallback
            base = conj_root / "1"
        fname = f"{sid}.png"
        return str((base / fname).resolve())

    df["image_path_local"] = df.apply(make_local_conj_path, axis=1)
    df["hb_value"] = df["hb_value (G/dL)"].astype(float)
    df["patient_id"] = df["subject_num"].astype(str)

    # Folder1 / Folder2 로 나눔
    df1 = df[df["subject_id"].str.contains("_1_")].copy()
    df2 = df[df["subject_id"].str.contains("_2_")].copy()

    cols = ["image_path_local", "hb_value", "patient_id"]
    df1_out = df1[cols].rename(columns={"image_path_local": "image_path"})
    df2_out = df2[cols].rename(columns={"image_path_local": "image_path"})

    out1_csv.parent.mkdir(parents=True, exist_ok=True)
    df1_out.to_csv(out1_csv, index=False)
    df2_out.to_csv(out2_csv, index=False)

    print(f"[conj folder1] wrote {len(df1_out)} rows to {out1_csv}")
    print(f"[conj folder2] wrote {len(df2_out)} rows to {out2_csv}")


if __name__ == "__main__":
    build_nail_csv()
    build_conj_csvs()


