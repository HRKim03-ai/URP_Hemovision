#!/usr/bin/env python3
"""
앙상블 config 파일을 자동으로 생성하는 스크립트.

Phase 1, Phase 2 선택된 모델 정보를 읽어서 w_demo와 wo_demo 각각의 앙상블 config를 생성합니다.
"""

import argparse
import os
import pandas as pd
import yaml


def create_ensemble_config(
    phase1_w_demo_csv: str,
    phase1_wo_demo_csv: str,
    phase2_selected_csv: str,
    output_dir: str = "config",
    predictions_dir: str = "ensemble_predictions",
):
    """
    앙상블 config 파일을 생성합니다.
    """
    # Phase 1 모델 선택 (MAE 최소, R² 최대)
    df_p1_w = pd.read_csv(phase1_w_demo_csv)
    df_p1_wo = pd.read_csv(phase1_wo_demo_csv)
    
    # R² > 0 필터링
    df_p1_w_valid = df_p1_w[df_p1_w["R²"] > 0] if "R²" in df_p1_w.columns else df_p1_w[df_p1_w["r2"] > 0]
    df_p1_wo_valid = df_p1_wo[df_p1_wo["R²"] > 0] if "R²" in df_p1_wo.columns else df_p1_wo[df_p1_wo["r2"] > 0]
    
    # MAE 컬럼명 정규화
    mae_col = "MAE" if "MAE" in df_p1_w.columns else "mae"
    r2_col = "R²" if "R²" in df_p1_w.columns else "r2"
    fold_col = "Fold" if "Fold" in df_p1_w.columns else "fold"
    nail_col = "Nail Model" if "Nail Model" in df_p1_w.columns else "nail_backbone"
    conj_col = "Conj Model" if "Conj Model" in df_p1_w.columns else "conj_backbone"
    
    # Phase 1 w_demo: MAE 최소, R² 최대
    p1_w_mae_min = df_p1_w_valid.loc[df_p1_w_valid[mae_col].idxmin()] if len(df_p1_w_valid) > 0 else df_p1_w.loc[df_p1_w[mae_col].idxmin()]
    p1_w_r2_max = df_p1_w_valid.loc[df_p1_w_valid[r2_col].idxmax()] if len(df_p1_w_valid) > 0 else df_p1_w.loc[df_p1_w[r2_col].idxmax()]
    
    # Phase 1 wo_demo: MAE 최소, R² 최대
    p1_wo_mae_min = df_p1_wo_valid.loc[df_p1_wo_valid[mae_col].idxmin()] if len(df_p1_wo_valid) > 0 else df_p1_wo.loc[df_p1_wo[mae_col].idxmin()]
    p1_wo_r2_max = df_p1_wo_valid.loc[df_p1_wo_valid[r2_col].idxmax()] if len(df_p1_wo_valid) > 0 else df_p1_wo.loc[df_p1_wo[r2_col].idxmax()]
    
    # Phase 2 모델 선택
    df_p2 = pd.read_csv(phase2_selected_csv)
    
    # w_demo 앙상블 config 생성
    w_demo_models = []
    
    # Phase 1 w_demo 모델들
    for model_info in [p1_w_mae_min, p1_w_r2_max]:
        fold = int(model_info[fold_col])
        nail = model_info[nail_col]
        conj = model_info[conj_col]
        
        # Normalize backbone names: remove timm/ prefix if present, add timm_ prefix if not
        def normalize_backbone_name(name: str) -> str:
            if name.startswith("timm/"):
                name = name[5:]  # Remove "timm/"
            if not name.startswith("timm_"):
                name = f"timm_{name}"
            return name
        
        nail = normalize_backbone_name(nail)
        conj = normalize_backbone_name(conj)
        pair_name = f"nail-{nail}_conj-{conj}"
        model_name = f"p1_fold{fold}_{pair_name}_w_demo"
        
        w_demo_models.append({
            "name": model_name,
            "val_pred_path": os.path.join(predictions_dir, f"{model_name}_val_preds.npy"),
            "test_pred_path": os.path.join(predictions_dir, f"{model_name}_test_preds.npy"),
            "weight_range": [0.0, 1.0],
        })
    
    # Phase 2 w_demo 모델들
    df_p2_w = df_p2[df_p2["version"] == "w_demo"]
    for _, row in df_p2_w.iterrows():
        fold = int(row["fold"])
        nail = row["nail_backbone"].replace("timm/", "timm_").replace("/", "_")
        conj = row["conj_backbone"].replace("timm/", "timm_").replace("/", "_")
        pair_name = f"nail-{nail}_conj-{conj}"
        model_name = f"p2_fold{fold}_{pair_name}_w_demo"
        
        w_demo_models.append({
            "name": model_name,
            "val_pred_path": os.path.join(predictions_dir, f"{model_name}_val_preds.npy"),
            "test_pred_path": os.path.join(predictions_dir, f"{model_name}_test_preds.npy"),
            "weight_range": [0.0, 1.0],
        })
    
    # wo_demo 앙상블 config 생성
    wo_demo_models = []
    
    # wo_demo는 fold 0 targets 사용 (Phase 2 fold 0 모델과 일치)
    wo_demo_target_fold = 0
    
    # Normalize backbone names 함수
    def normalize_backbone_name(name: str) -> str:
        if name.startswith("timm/"):
            name = name[5:]
        if not name.startswith("timm_"):
            name = f"timm_{name}"
        return name
    
    # Phase 1 wo_demo fold 0 모델 선택
    p1_wo_fold0 = df_p1_wo[df_p1_wo[fold_col] == 0]
    if len(p1_wo_fold0) > 0:
        p1_wo_fold0_valid = p1_wo_fold0[p1_wo_fold0[r2_col] > 0] if r2_col in p1_wo_fold0.columns and p1_wo_fold0[r2_col].dtype != 'object' else p1_wo_fold0
        
        if len(p1_wo_fold0_valid) > 0:
            p1_wo_fold0_mae_min = p1_wo_fold0_valid.loc[p1_wo_fold0_valid[mae_col].idxmin()]
            p1_wo_fold0_r2_max = p1_wo_fold0_valid.loc[p1_wo_fold0_valid[r2_col].idxmax()]
        else:
            p1_wo_fold0_mae_min = p1_wo_fold0.loc[p1_wo_fold0[mae_col].idxmin()]
            p1_wo_fold0_r2_max = p1_wo_fold0.loc[p1_wo_fold0[r2_col].idxmax()]
        
        seen_p1_wo = set()
        for model_info in [p1_wo_fold0_mae_min, p1_wo_fold0_r2_max]:
            fold = int(model_info[fold_col])
            nail = normalize_backbone_name(model_info[nail_col])
            conj = normalize_backbone_name(model_info[conj_col])
            model_key = (fold, nail, conj)
            
            if model_key not in seen_p1_wo:
                seen_p1_wo.add(model_key)
                pair_name = f"nail-{nail}_conj-{conj}"
                model_name = f"p1_fold{fold}_{pair_name}_wo_demo"
                
                wo_demo_models.append({
                    "name": model_name,
                    "val_pred_path": os.path.join(predictions_dir, f"{model_name}_val_preds.npy"),
                    "test_pred_path": os.path.join(predictions_dir, f"{model_name}_test_preds.npy"),
                    "weight_range": [0.0, 1.0],
                })
    
    # Phase 2 wo_demo fold 0 모델 추가
    df_p2_wo = df_p2[df_p2["version"] == "wo_demo"]
    seen_p2_wo = set()
    for _, row in df_p2_wo.iterrows():
        fold = int(row["fold"])
        if fold == wo_demo_target_fold:
            nail = row["nail_backbone"].replace("timm/", "timm_").replace("/", "_")
            conj = row["conj_backbone"].replace("timm/", "timm_").replace("/", "_")
            pair_name = f"nail-{nail}_conj-{conj}"
            model_key = (fold, nail, conj)
            
            if model_key not in seen_p2_wo:
                seen_p2_wo.add(model_key)
                model_name = f"p2_fold{fold}_{pair_name}_wo_demo"
                
                wo_demo_models.append({
                    "name": model_name,
                    "val_pred_path": os.path.join(predictions_dir, f"{model_name}_val_preds.npy"),
                    "test_pred_path": os.path.join(predictions_dir, f"{model_name}_test_preds.npy"),
                    "weight_range": [0.0, 1.0],
                })
    
    # Config 파일 생성
    from collections import Counter
    
    # w_demo fold 수집
    w_demo_folds = []
    for model_info in [p1_w_mae_min, p1_w_r2_max]:
        w_demo_folds.append(int(model_info[fold_col]))
    for _, row in df_p2[df_p2["version"] == "w_demo"].iterrows():
        w_demo_folds.append(int(row["fold"]))
    
    w_demo_fold_counts = Counter(w_demo_folds)
    w_demo_target_fold = w_demo_fold_counts.most_common(1)[0][0] if w_demo_fold_counts else 0
    
    w_demo_config = {
        "log_file": "logs/ensemble_w_demo.log",
        "output_dir": "ensemble_results/w_demo",
        "val_targets_path": os.path.join(predictions_dir, f"fold{w_demo_target_fold}_val_targets.npy"),
        "test_targets_path": os.path.join(predictions_dir, f"fold{w_demo_target_fold}_test_targets.npy"),
        "weighting": "grid_search",
        "n_steps": 5,
        "models": w_demo_models,
    }
    
    wo_demo_config = {
        "log_file": "logs/ensemble_wo_demo.log",
        "output_dir": "ensemble_results/wo_demo",
        "val_targets_path": os.path.join(predictions_dir, f"fold{wo_demo_target_fold}_val_targets.npy"),
        "test_targets_path": os.path.join(predictions_dir, f"fold{wo_demo_target_fold}_test_targets.npy"),
        "weighting": "grid_search",
        "n_steps": 5,
        "models": wo_demo_models,
    }
    
    # Save config files
    os.makedirs(output_dir, exist_ok=True)
    
    w_demo_path = os.path.join(output_dir, "ensemble_w_demo.yaml")
    wo_demo_path = os.path.join(output_dir, "ensemble_wo_demo.yaml")
    
    with open(w_demo_path, "w") as f:
        yaml.dump(w_demo_config, f, default_flow_style=False, allow_unicode=True)
    
    with open(wo_demo_path, "w") as f:
        yaml.dump(wo_demo_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Created ensemble config files:")
    print(f"  - {w_demo_path}")
    print(f"  - {wo_demo_path}")
    print(f"\nw_demo models: {len(w_demo_models)}")
    print(f"wo_demo models: {len(wo_demo_models)}")


def main():
    parser = argparse.ArgumentParser(description="앙상블 config 파일 생성")
    parser.add_argument(
        "--phase1_w_demo_csv",
        type=str,
        default="checkpoints/fusion_phase1/fusion_phase1_w_demo_results.csv",
        help="Phase 1 w_demo 결과 CSV",
    )
    parser.add_argument(
        "--phase1_wo_demo_csv",
        type=str,
        default="checkpoints/fusion_phase1/fusion_phase1_wo_demo_results.csv",
        help="Phase 1 wo_demo 결과 CSV",
    )
    parser.add_argument(
        "--phase2_selected_csv",
        type=str,
        default="logs/fusion_phase2/fusion_phase2_selected_models.csv",
        help="Phase 2 선택된 모델 CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="config",
        help="Config 파일 저장 디렉토리",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="ensemble_predictions",
        help="예측값 디렉토리",
    )
    
    args = parser.parse_args()
    
    create_ensemble_config(
        phase1_w_demo_csv=args.phase1_w_demo_csv,
        phase1_wo_demo_csv=args.phase1_wo_demo_csv,
        phase2_selected_csv=args.phase2_selected_csv,
        output_dir=args.output_dir,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()

