#!/usr/bin/env python3
"""
Phase 2 결과에서 Phase 1처럼 명확한 기준(MAE 최소, R² 최대)으로 모델을 선택하는 스크립트.

Phase 1과 동일하게:
- w_demo MAE 최소 모델
- w_demo R² 최대 모델
- wo_demo MAE 최소 모델
- wo_demo R² 최대 모델

총 4개 모델을 선택합니다.
"""

import pandas as pd
import os
from pathlib import Path


def select_best_models(
    w_demo_csv: str,
    wo_demo_csv: str,
    output_dir: str = "logs/fusion_phase2",
):
    """
    Phase 2 결과 CSV에서 최고 성능 모델을 선택합니다.
    
    Args:
        w_demo_csv: w_demo 결과 CSV 파일 경로
        wo_demo_csv: wo_demo 결과 CSV 파일 경로
        output_dir: 출력 디렉토리
    """
    # CSV 파일 읽기
    df_w = pd.read_csv(w_demo_csv)
    df_wo = pd.read_csv(wo_demo_csv)
    
    # R² > 0 조건 필터링 (Phase 1과 동일한 기준)
    df_w_valid = df_w[df_w["r2"] > 0].copy()
    df_wo_valid = df_wo[df_wo["r2"] > 0].copy()
    
    print(f"w_demo: 총 {len(df_w)}개 모델 중 R² > 0인 모델 {len(df_w_valid)}개")
    print(f"wo_demo: 총 {len(df_wo)}개 모델 중 R² > 0인 모델 {len(df_wo_valid)}개")
    
    # w_demo: MAE 최소 모델
    if len(df_w_valid) > 0:
        w_mae_min = df_w_valid.loc[df_w_valid["mae"].idxmin()]
        print(f"\n[w_demo MAE 최소]")
        print(f"  Fold: {w_mae_min['fold']}")
        print(f"  Nail: {w_mae_min['nail_backbone']}")
        print(f"  Conj: {w_mae_min['conj_backbone']}")
        print(f"  MAE: {w_mae_min['mae']:.6f}")
        print(f"  R²: {w_mae_min['r2']:.6f}")
        print(f"  Best Epoch: {w_mae_min['best_epoch']}")
    else:
        print("\n[w_demo MAE 최소] R² > 0인 모델이 없습니다. R² 조건 없이 선택합니다.")
        w_mae_min = df_w.loc[df_w["mae"].idxmin()]
        print(f"  Fold: {w_mae_min['fold']}")
        print(f"  Nail: {w_mae_min['nail_backbone']}")
        print(f"  Conj: {w_mae_min['conj_backbone']}")
        print(f"  MAE: {w_mae_min['mae']:.6f}")
        print(f"  R²: {w_mae_min['r2']:.6f}")
        print(f"  Best Epoch: {w_mae_min['best_epoch']}")
    
    # w_demo: R² 최대 모델
    if len(df_w_valid) > 0:
        w_r2_max = df_w_valid.loc[df_w_valid["r2"].idxmax()]
        print(f"\n[w_demo R² 최대]")
        print(f"  Fold: {w_r2_max['fold']}")
        print(f"  Nail: {w_r2_max['nail_backbone']}")
        print(f"  Conj: {w_r2_max['conj_backbone']}")
        print(f"  MAE: {w_r2_max['mae']:.6f}")
        print(f"  R²: {w_r2_max['r2']:.6f}")
        print(f"  Best Epoch: {w_r2_max['best_epoch']}")
    else:
        print("\n[w_demo R² 최대] R² > 0인 모델이 없습니다. R² 조건 없이 선택합니다.")
        w_r2_max = df_w.loc[df_w["r2"].idxmax()]
        print(f"  Fold: {w_r2_max['fold']}")
        print(f"  Nail: {w_r2_max['nail_backbone']}")
        print(f"  Conj: {w_r2_max['conj_backbone']}")
        print(f"  MAE: {w_r2_max['mae']:.6f}")
        print(f"  R²: {w_r2_max['r2']:.6f}")
        print(f"  Best Epoch: {w_r2_max['best_epoch']}")
    
    # wo_demo: MAE 최소 모델
    if len(df_wo_valid) > 0:
        wo_mae_min = df_wo_valid.loc[df_wo_valid["mae"].idxmin()]
        print(f"\n[wo_demo MAE 최소]")
        print(f"  Fold: {wo_mae_min['fold']}")
        print(f"  Nail: {wo_mae_min['nail_backbone']}")
        print(f"  Conj: {wo_mae_min['conj_backbone']}")
        print(f"  MAE: {wo_mae_min['mae']:.6f}")
        print(f"  R²: {wo_mae_min['r2']:.6f}")
        print(f"  Best Epoch: {wo_mae_min['best_epoch']}")
    else:
        print("\n[wo_demo MAE 최소] R² > 0인 모델이 없습니다. R² 조건 없이 선택합니다.")
        wo_mae_min = df_wo.loc[df_wo["mae"].idxmin()]
        print(f"  Fold: {wo_mae_min['fold']}")
        print(f"  Nail: {wo_mae_min['nail_backbone']}")
        print(f"  Conj: {wo_mae_min['conj_backbone']}")
        print(f"  MAE: {wo_mae_min['mae']:.6f}")
        print(f"  R²: {wo_mae_min['r2']:.6f}")
        print(f"  Best Epoch: {wo_mae_min['best_epoch']}")
    
    # wo_demo: R² 최대 모델
    if len(df_wo_valid) > 0:
        wo_r2_max = df_wo_valid.loc[df_wo_valid["r2"].idxmax()]
        print(f"\n[wo_demo R² 최대]")
        print(f"  Fold: {wo_r2_max['fold']}")
        print(f"  Nail: {wo_r2_max['nail_backbone']}")
        print(f"  Conj: {wo_r2_max['conj_backbone']}")
        print(f"  MAE: {wo_r2_max['mae']:.6f}")
        print(f"  R²: {wo_r2_max['r2']:.6f}")
        print(f"  Best Epoch: {wo_r2_max['best_epoch']}")
    else:
        print("\n[wo_demo R² 최대] R² > 0인 모델이 없습니다. R² 조건 없이 선택합니다.")
        wo_r2_max = df_wo.loc[df_wo["r2"].idxmax()]
        print(f"  Fold: {wo_r2_max['fold']}")
        print(f"  Nail: {wo_r2_max['nail_backbone']}")
        print(f"  Conj: {wo_r2_max['conj_backbone']}")
        print(f"  MAE: {wo_r2_max['mae']:.6f}")
        print(f"  R²: {wo_r2_max['r2']:.6f}")
        print(f"  Best Epoch: {wo_r2_max['best_epoch']}")
    
    # 선택된 모델들을 DataFrame으로 정리
    selected_models = pd.DataFrame([
        {
            "version": "w_demo",
            "selection_criteria": "MAE 최소",
            "fold": w_mae_min["fold"],
            "nail_backbone": w_mae_min["nail_backbone"],
            "conj_backbone": w_mae_min["conj_backbone"],
            "best_epoch": w_mae_min["best_epoch"],
            "mae": w_mae_min["mae"],
            "r2": w_mae_min["r2"],
            "acc@0.5": w_mae_min["acc@0.5"],
            "acc@1.0": w_mae_min["acc@1.0"],
            "acc@2.0": w_mae_min["acc@2.0"],
        },
        {
            "version": "w_demo",
            "selection_criteria": "R² 최대",
            "fold": w_r2_max["fold"],
            "nail_backbone": w_r2_max["nail_backbone"],
            "conj_backbone": w_r2_max["conj_backbone"],
            "best_epoch": w_r2_max["best_epoch"],
            "mae": w_r2_max["mae"],
            "r2": w_r2_max["r2"],
            "acc@0.5": w_r2_max["acc@0.5"],
            "acc@1.0": w_r2_max["acc@1.0"],
            "acc@2.0": w_r2_max["acc@2.0"],
        },
        {
            "version": "wo_demo",
            "selection_criteria": "MAE 최소",
            "fold": wo_mae_min["fold"],
            "nail_backbone": wo_mae_min["nail_backbone"],
            "conj_backbone": wo_mae_min["conj_backbone"],
            "best_epoch": wo_mae_min["best_epoch"],
            "mae": wo_mae_min["mae"],
            "r2": wo_mae_min["r2"],
            "acc@0.5": wo_mae_min["acc@0.5"],
            "acc@1.0": wo_mae_min["acc@1.0"],
            "acc@2.0": wo_mae_min["acc@2.0"],
        },
        {
            "version": "wo_demo",
            "selection_criteria": "R² 최대",
            "fold": wo_r2_max["fold"],
            "nail_backbone": wo_r2_max["nail_backbone"],
            "conj_backbone": wo_r2_max["conj_backbone"],
            "best_epoch": wo_r2_max["best_epoch"],
            "mae": wo_r2_max["mae"],
            "r2": wo_r2_max["r2"],
            "acc@0.5": wo_r2_max["acc@0.5"],
            "acc@1.0": wo_r2_max["acc@1.0"],
            "acc@2.0": wo_r2_max["acc@2.0"],
        },
    ])
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일로 저장
    output_path = os.path.join(output_dir, "fusion_phase2_selected_models.csv")
    selected_models.to_csv(output_path, index=False)
    print(f"\n선택된 모델 정보가 저장되었습니다: {output_path}")
    
    return selected_models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 2 결과에서 최고 성능 모델 선택"
    )
    parser.add_argument(
        "--w_demo_csv",
        type=str,
        default="logs/fusion_phase2/fusion_phase2_results_w_demo_best.csv",
        help="w_demo 결과 CSV 파일 경로",
    )
    parser.add_argument(
        "--wo_demo_csv",
        type=str,
        default="logs/fusion_phase2/fusion_phase2_results_wo_demo_best.csv",
        help="wo_demo 결과 CSV 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/fusion_phase2",
        help="출력 디렉토리",
    )
    
    args = parser.parse_args()
    
    select_best_models(
        w_demo_csv=args.w_demo_csv,
        wo_demo_csv=args.wo_demo_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

