#!/usr/bin/env python3
"""
실제 존재하는 체크포인트를 기반으로 Phase 2 최고 성능 모델을 선택하는 스크립트.

실제 존재하는 체크포인트 파일에서 직접 메트릭을 읽어서,
MAE 최소, R² 최대 모델을 선택합니다.
"""

import argparse
import os
import pandas as pd
import torch
from pathlib import Path


def select_best_models_from_checkpoints(
    checkpoint_dir: str,
    output_csv: str,
):
    """
    실제 존재하는 체크포인트 파일에서 직접 메트릭을 읽어서 최고 성능 모델을 선택합니다.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # 모든 체크포인트 파일 찾기
    w_demo_checkpoints = list(checkpoint_dir.glob("p2_*_w_demo_best.pt"))
    wo_demo_checkpoints = list(checkpoint_dir.glob("p2_*_wo_demo_best.pt"))
    
    print(f"w_demo 체크포인트: {len(w_demo_checkpoints)}개")
    print(f"wo_demo 체크포인트: {len(wo_demo_checkpoints)}개")
    
    def parse_checkpoint_info(ckpt_path: Path):
        """체크포인트 파일명에서 정보 추출하고 메트릭 읽기"""
        name = ckpt_path.name
        # p2_fold0_nail-timm_convnext_small.in12k_conj-timm_convnext_small.in12k_w_demo_best.pt
        parts = name.replace("p2_fold", "").replace("_best.pt", "").split("_nail-")
        fold = int(parts[0])
        rest = parts[1]
        
        if "_w_demo" in rest:
            use_demographics = True
            rest = rest.replace("_w_demo", "")
        else:
            use_demographics = False
            rest = rest.replace("_wo_demo", "")
        
        nail_conj = rest.split("_conj-")
        nail = nail_conj[0]
        conj = nail_conj[1]
        
        # 체크포인트에서 메트릭 읽기
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            metrics = state.get("metrics", {})
            extra = state.get("extra", {})
            epoch = state.get("epoch", 0)
            
            return {
                "fold": fold,
                "nail_backbone": nail,
                "conj_backbone": conj,
                "use_demographics": use_demographics,
                "best_epoch": epoch,
                "mae": metrics.get("mae", float("inf")),
                "r2": metrics.get("r2", float("-inf")),
                "acc@0.5": metrics.get("acc@0.5", 0.0),
                "acc@1.0": metrics.get("acc@1.0", 0.0),
                "acc@2.0": metrics.get("acc@2.0", 0.0),
            }
        except Exception as e:
            print(f"Warning: Failed to load {ckpt_path}: {e}")
            return None
    
    # w_demo 체크포인트에서 정보 추출
    w_demo_models = []
    for ckpt in w_demo_checkpoints:
        info = parse_checkpoint_info(ckpt)
        if info:
            w_demo_models.append(info)
    
    # wo_demo 체크포인트에서 정보 추출
    wo_demo_models = []
    for ckpt in wo_demo_checkpoints:
        info = parse_checkpoint_info(ckpt)
        if info:
            wo_demo_models.append(info)
    
    if len(w_demo_models) == 0 and len(wo_demo_models) == 0:
        print("No valid checkpoints found!")
        return
    
    df_w = pd.DataFrame(w_demo_models) if w_demo_models else pd.DataFrame()
    df_wo = pd.DataFrame(wo_demo_models) if wo_demo_models else pd.DataFrame()
    
    print(f"\nw_demo: {len(df_w)}개 모델")
    print(f"wo_demo: {len(df_wo)}개 모델")
    
    # R² > 0 조건 필터링
    df_w_valid_r2 = df_w[df_w["r2"] > 0].copy() if len(df_w) > 0 else pd.DataFrame()
    df_wo_valid_r2 = df_wo[df_wo["r2"] > 0].copy() if len(df_wo) > 0 else pd.DataFrame()
    
    # w_demo: MAE 최소 모델
    if len(df_w_valid_r2) > 0:
        w_mae_min = df_w_valid_r2.loc[df_w_valid_r2["mae"].idxmin()]
    elif len(df_w) > 0:
        w_mae_min = df_w.loc[df_w["mae"].idxmin()]
    else:
        w_mae_min = None
    
    # w_demo: R² 최대 모델
    if len(df_w_valid_r2) > 0:
        w_r2_max = df_w_valid_r2.loc[df_w_valid_r2["r2"].idxmax()]
    elif len(df_w) > 0:
        w_r2_max = df_w.loc[df_w["r2"].idxmax()]
    else:
        w_r2_max = None
    
    # wo_demo: MAE 최소 모델
    if len(df_wo_valid_r2) > 0:
        wo_mae_min = df_wo_valid_r2.loc[df_wo_valid_r2["mae"].idxmin()]
    elif len(df_wo) > 0:
        wo_mae_min = df_wo.loc[df_wo["mae"].idxmin()]
    else:
        wo_mae_min = None
    
    # wo_demo: R² 최대 모델
    if len(df_wo_valid_r2) > 0:
        wo_r2_max = df_wo_valid_r2.loc[df_wo_valid_r2["r2"].idxmax()]
    elif len(df_wo) > 0:
        wo_r2_max = df_wo.loc[df_wo["r2"].idxmax()]
    else:
        wo_r2_max = None
    
    if w_mae_min is None and w_r2_max is None:
        print("\n[w_demo] 체크포인트가 존재하는 모델이 없습니다.")
        return
    
    if wo_mae_min is None and wo_r2_max is None:
        print("\n[wo_demo] 체크포인트가 존재하는 모델이 없습니다.")
        return
    
    # 선택된 모델들을 리스트로 구성
    selected_models_list = []
    
    if w_mae_min is not None:
        print(f"\n[w_demo MAE 최소]")
        print(f"  Fold: {w_mae_min['fold']}")
        print(f"  Nail: {w_mae_min['nail_backbone']}")
        print(f"  Conj: {w_mae_min['conj_backbone']}")
        print(f"  MAE: {w_mae_min['mae']:.6f}")
        print(f"  R²: {w_mae_min['r2']:.6f}")
        selected_models_list.append({
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
        })
    
    if w_r2_max is not None:
        # 동일한 모델이어도 MAE 최소와 R² 최대를 모두 포함
        print(f"\n[w_demo R² 최대]")
        print(f"  Fold: {w_r2_max['fold']}")
        print(f"  Nail: {w_r2_max['nail_backbone']}")
        print(f"  Conj: {w_r2_max['conj_backbone']}")
        print(f"  MAE: {w_r2_max['mae']:.6f}")
        print(f"  R²: {w_r2_max['r2']:.6f}")
        selected_models_list.append({
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
        })
    
    if wo_mae_min is not None:
        print(f"\n[wo_demo MAE 최소]")
        print(f"  Fold: {wo_mae_min['fold']}")
        print(f"  Nail: {wo_mae_min['nail_backbone']}")
        print(f"  Conj: {wo_mae_min['conj_backbone']}")
        print(f"  MAE: {wo_mae_min['mae']:.6f}")
        print(f"  R²: {wo_mae_min['r2']:.6f}")
        selected_models_list.append({
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
        })
    
    if wo_r2_max is not None:
        # 동일한 모델이어도 MAE 최소와 R² 최대를 모두 포함
        print(f"\n[wo_demo R² 최대]")
        print(f"  Fold: {wo_r2_max['fold']}")
        print(f"  Nail: {wo_r2_max['nail_backbone']}")
        print(f"  Conj: {wo_r2_max['conj_backbone']}")
        print(f"  MAE: {wo_r2_max['mae']:.6f}")
        print(f"  R²: {wo_r2_max['r2']:.6f}")
        selected_models_list.append({
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
        })
    
    if len(selected_models_list) == 0:
        print("\n선택된 모델이 없습니다.")
        return
    
    selected_models = pd.DataFrame(selected_models_list)
    
    # CSV 파일로 저장
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_models.to_csv(output_path, index=False)
    print(f"\n선택된 모델 정보가 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="실제 존재하는 체크포인트를 기반으로 Phase 2 최고 성능 모델 선택"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="logs/fusion_phase2",
        help="체크포인트 디렉토리",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="logs/fusion_phase2/fusion_phase2_selected_models.csv",
        help="출력 CSV 파일 경로",
    )
    
    args = parser.parse_args()
    
    select_best_models_from_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()

