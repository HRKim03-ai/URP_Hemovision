#!/usr/bin/env python3
"""
앙상블을 위한 각 모델의 validation/test 예측값을 생성하는 스크립트.

각 fold별로:
- Validation set 예측값 저장
- Test set (external test) 예측값 저장
- Targets도 함께 저장 (첫 번째 모델에서만)

Phase 1, Phase 2, Single-modality 모델들의 예측값을 생성합니다.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fusion_dataset import FusionDataset, FusionSample, load_fusion_metadata
from datasets.transforms import build_transforms
from models.fusion_models import Phase1FusionModel, Phase2MultiLevelFusionModel
from utils.checkpoint import load_checkpoint
from utils.cv_split import create_fusion_splits
from utils.logger import setup_logger
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def generate_fusion_predictions(
    config: Dict[str, Any],
    fold_idx: int,
    nail_backbone: str,
    conj_backbone: str,
    phase: int,
    use_demographics: bool,
    checkpoint_path: str,
    output_dir: str,
    save_targets: bool = False,
    device_id: int | None = None,
) -> Dict[str, str]:
    """
    Fusion 모델의 validation/test 예측값을 생성합니다.
    
    Returns:
        dict: 예측값 파일 경로들 (val_pred_path, test_pred_path, val_targets_path, test_targets_path)
    """
    if device_id is not None:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))
    
    # Load data
    samples: List[FusionSample] = load_fusion_metadata(config["fusion_metadata_csv"])
    df = pd.DataFrame({"patient_id": [s.patient_id for s in samples]})
    splits = create_fusion_splits(df, patient_col="patient_id")
    
    # Get fold indices
    train_idx, val_idx = splits.folds[fold_idx]
    test_idx = splits.external_test_idx
    
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    # Build data loaders
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    
    nail_val_tf = build_transforms("val", image_size=image_size, modality="nail")
    conj_val_tf = build_transforms("val", image_size=image_size, modality="conj")
    nail_test_tf = build_transforms("test", image_size=image_size, modality="nail")
    conj_test_tf = build_transforms("test", image_size=image_size, modality="conj")
    
    val_ds = FusionDataset(val_samples, nail_transform=nail_val_tf, conj_transform=conj_val_tf)
    test_ds = FusionDataset(test_samples, nail_transform=nail_test_tf, conj_transform=conj_test_tf)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Build model
    demo_dim = 2 if use_demographics else 0
    if phase == 1:
        model = Phase1FusionModel(
            nail_backbone_name=nail_backbone,
            conj_backbone_name=conj_backbone,
            demo_dim=demo_dim,
        ).to(device)
    else:
        model = Phase2MultiLevelFusionModel(
            nail_backbone_name=nail_backbone,
            conj_backbone_name=conj_backbone,
            demo_dim=demo_dim,
        ).to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model=model)
    model.eval()
    
    # Generate predictions
    demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
    # Backbone names should already be normalized (timm_ prefix)
    pair_name = f"nail-{nail_backbone}_conj-{conj_backbone}"
    
    # Validation predictions
    val_preds_list: List[np.ndarray] = []
    val_targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val predictions"):
            nail_img, conj_img, hb, _, demo = batch
            nail_img = nail_img.to(device)
            conj_img = conj_img.to(device)
            hb = hb.to(device, dtype=torch.float32)
            demo = demo.to(device, dtype=torch.float32) if use_demographics else None
            preds = model(nail_img, conj_img, demo)
            val_preds_list.append(tensor_to_numpy(preds))
            val_targets_list.append(tensor_to_numpy(hb))
    
    val_preds = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)
    
    # Test predictions
    test_preds_list: List[np.ndarray] = []
    test_targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Test predictions"):
            nail_img, conj_img, hb, _, demo = batch
            nail_img = nail_img.to(device)
            conj_img = conj_img.to(device)
            hb = hb.to(device, dtype=torch.float32)
            demo = demo.to(device, dtype=torch.float32) if use_demographics else None
            preds = model(nail_img, conj_img, demo)
            test_preds_list.append(tensor_to_numpy(preds))
            test_targets_list.append(tensor_to_numpy(hb))
    
    test_preds = np.concatenate(test_preds_list)
    test_targets = np.concatenate(test_targets_list)
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    # For file naming, normalize backbone names (timm/xxx -> timm_xxx)
    def normalize_for_filename(name: str) -> str:
        if name.startswith("timm/"):
            name = name[5:]  # Remove "timm/"
        if not name.startswith("timm_"):
            name = f"timm_{name}"
        return name
    
    nail_for_filename = normalize_for_filename(nail_backbone)
    conj_for_filename = normalize_for_filename(conj_backbone)
    pair_name_for_file = f"nail-{nail_for_filename}_conj-{conj_for_filename}"
    model_name = f"p{phase}_fold{fold_idx}_{pair_name_for_file}{demo_suffix}"
    val_pred_path = os.path.join(output_dir, f"{model_name}_val_preds.npy")
    test_pred_path = os.path.join(output_dir, f"{model_name}_test_preds.npy")
    
    np.save(val_pred_path, val_preds)
    np.save(test_pred_path, test_preds)
    
    result = {
        "val_pred_path": val_pred_path,
        "test_pred_path": test_pred_path,
    }
    
    # Save targets (only once, for the first model)
    if save_targets:
        val_targets_path = os.path.join(output_dir, f"fold{fold_idx}_val_targets.npy")
        test_targets_path = os.path.join(output_dir, f"fold{fold_idx}_test_targets.npy")
        np.save(val_targets_path, val_targets)
        np.save(test_targets_path, test_targets)
        result["val_targets_path"] = val_targets_path
        result["test_targets_path"] = test_targets_path
    
    # Log metrics
    val_metrics = compute_regression_metrics(val_targets, val_preds)
    test_metrics = compute_regression_metrics(test_targets, test_preds)
    print(f"\n{model_name}:")
    print(f"  Val MAE: {val_metrics['mae']:.6f}, R²: {val_metrics['r2']:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}, R²: {test_metrics['r2']:.6f}")
    
    return result


def _generate_predictions_worker(config: Dict[str, Any]) -> None:
    """멀티 GPU 병렬 처리를 위한 worker 함수"""
    device_id = config.get("device_id", None)
    assigned_tasks = config.get("assigned_tasks", None)
    
    logger = setup_logger(
        "generate_predictions",
        config.get("log_file", "logs/generate_predictions.log"),
    )
    
    if device_id is not None:
        logger.info(f"GPU {device_id}: Processing {len(assigned_tasks) if assigned_tasks else 'all'} tasks")
    
    # Load selected models
    selected_models_csv = config.get("selected_models_csv")
    phase = config.get("phase")
    checkpoint_dir = config.get("checkpoint_dir")
    output_dir = config.get("output_dir", "ensemble_predictions")
    
    if selected_models_csv:
        df = pd.read_csv(selected_models_csv)
    else:
        logger.error("selected_models_csv is required")
        return
    
    # Process assigned tasks
    if assigned_tasks:
        # assigned_tasks는 (fold_idx, nail_backbone, conj_backbone, use_demographics) 튜플 리스트
        for task in assigned_tasks:
            fold_idx, nail_backbone, conj_backbone, use_demographics = task
            
            # Find checkpoint
            # Backbone names are already normalized in all_tasks (timm_ prefix added)
            demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
            pair_name = f"nail-{nail_backbone}_conj-{conj_backbone}"
            ckpt_filename = f"p{phase}_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt"
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)
            
            if not os.path.exists(checkpoint_path):
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"  This model may not have been trained yet. Skipping."
                )
                continue
            
            # Generate predictions (targets는 첫 번째 task에서만 저장)
            save_targets = (task == assigned_tasks[0])
            try:
                # For model creation, use original backbone names in timm/ format
                # The checkpoint path uses normalized names (timm_xxx), but model needs timm/xxx format
                def denormalize_backbone_name(name: str) -> str:
                    # Remove "timm_" prefix if present and convert to "timm/" format
                    if name.startswith("timm_"):
                        name = name[5:]  # Remove "timm_"
                    # Add "timm/" prefix for model creation
                    if not name.startswith("timm/"):
                        name = f"timm/{name}"
                    return name
                
                nail_backbone_orig = denormalize_backbone_name(nail_backbone)
                conj_backbone_orig = denormalize_backbone_name(conj_backbone)
                
                result = generate_fusion_predictions(
                    config=config,
                    fold_idx=fold_idx,
                    nail_backbone=nail_backbone_orig,
                    conj_backbone=conj_backbone_orig,
                    phase=phase,
                    use_demographics=use_demographics,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    save_targets=save_targets,
                    device_id=device_id,
                )
                logger.info(f"Generated predictions for fold {fold_idx}, {pair_name}{demo_suffix}")
            except Exception as e:
                logger.error(f"Error processing {pair_name}{demo_suffix}: {e}")
                continue


def run_generate_predictions_multi_gpu(
    config: Dict[str, Any],
    num_gpus: int,
) -> None:
    """여러 GPU에 자동 분배해서 병렬로 예측값 생성"""
    selected_models_csv = config.get("selected_models_csv")
    phase = config.get("phase")
    
    if not selected_models_csv:
        raise ValueError("selected_models_csv is required")
    
    df = pd.read_csv(selected_models_csv)
    
    # Handle different CSV formats (Phase 1 vs Phase 2)
    # Phase 1: "Fold", "Nail Model", "Conj Model" (w_demo/wo_demo는 별도 CSV 파일)
    # Phase 2: "fold", "nail_backbone", "conj_backbone", "version"
    fold_col = "Fold" if "Fold" in df.columns else "fold"
    nail_col = "Nail Model" if "Nail Model" in df.columns else "nail_backbone"
    conj_col = "Conj Model" if "Conj Model" in df.columns else "conj_backbone"
    
    # For Phase 2 CSV, we have "version" column
    # For Phase 1 CSV, use_demographics is determined by which CSV file is being processed
    has_version_col = "version" in df.columns
    
    # Generate all tasks: (fold_idx, nail_backbone, conj_backbone, use_demographics)
    all_tasks = []
    checkpoint_dir = config.get("checkpoint_dir")
    phase = config.get("phase")
    logger = setup_logger("generate_predictions", config.get("log_file", "logs/generate_predictions.log"))
    
    for _, row in df.iterrows():
        fold_idx = int(row[fold_col])
        nail_backbone = row[nail_col]
        conj_backbone = row[conj_col]
        
        # Normalize backbone names for checkpoint filename
        # CSV may have "timm/convnext_small.in12k" or "convnext_small.in12k"
        # Checkpoint files use "timm_convnext_small.in12k" format
        def normalize_backbone_name(name: str) -> str:
            # Remove "timm/" prefix if present
            if name.startswith("timm/"):
                name = name[5:]  # Remove "timm/"
            # Add "timm_" prefix if not present
            if not name.startswith("timm_"):
                name = f"timm_{name}"
            return name
        
        nail_backbone_norm = normalize_backbone_name(nail_backbone)
        conj_backbone_norm = normalize_backbone_name(conj_backbone)
        
        if has_version_col:
            use_demographics = row["version"] == "w_demo"
        else:
            # Phase 1: use_demographics is determined by which CSV file is being processed
            # This will be set in the config when calling the script
            use_demographics = config.get("use_demographics", True)
        
        # Check if checkpoint exists before adding to tasks
        demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
        pair_name = f"nail-{nail_backbone_norm}_conj-{conj_backbone_norm}"
        ckpt_filename = f"p{phase}_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt"
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Skipping model (checkpoint not found): "
                f"fold={fold_idx}, nail={nail_backbone}, conj={conj_backbone}, demo={use_demographics}\n"
                f"  Expected: {checkpoint_path}"
            )
            continue
        
        all_tasks.append((fold_idx, nail_backbone_norm, conj_backbone_norm, use_demographics))
    
    # Split tasks across GPUs
    n = len(all_tasks)
    num_gpus = max(1, min(num_gpus, n))
    base = n // num_gpus
    rem = n % num_gpus
    task_splits: List[List[Tuple]] = []
    start = 0
    for i in range(num_gpus):
        extra = 1 if i < rem else 0
        end = start + base + extra
        task_splits.append(all_tasks[start:end])
        start = end
    
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    
    for gpu_id, tasks in enumerate(task_splits):
        if not tasks:
            continue
        cfg = copy.deepcopy(config)
        cfg["assigned_tasks"] = tasks
        cfg["device_id"] = gpu_id
        p = ctx.Process(target=_generate_predictions_worker, args=(cfg,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="앙상블을 위한 예측값 생성")
    parser.add_argument("--config", type=str, required=True, help="Config 파일 경로")
    parser.add_argument(
        "--selected_models_csv",
        type=str,
        help="선택된 모델 정보 CSV 파일 (Phase 1 또는 Phase 2)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Phase 번호 (1 또는 2)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ensemble_predictions",
        help="예측값 저장 디렉토리",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="특정 fold만 처리 (지정하지 않으면 모든 fold 처리)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="사용할 GPU 개수 (병렬 처리)",
    )
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Add script-specific config
    config["selected_models_csv"] = args.selected_models_csv
    config["phase"] = args.phase
    config["checkpoint_dir"] = args.checkpoint_dir
    config["output_dir"] = args.output_dir
    
    # Phase 1 CSV 파일의 경우, 파일명에서 w_demo/wo_demo 판단
    if args.phase == 1 and args.selected_models_csv:
        if "w_demo" in args.selected_models_csv.lower():
            config["use_demographics"] = True
        elif "wo_demo" in args.selected_models_csv.lower():
            config["use_demographics"] = False
    
    logger = setup_logger("generate_predictions", config.get("log_file", "logs/generate_predictions.log"))
    
    # 멀티 GPU 병렬 처리
    if args.num_gpus > 1:
        logger.info(f"Using {args.num_gpus} GPUs for parallel prediction generation")
        run_generate_predictions_multi_gpu(config, args.num_gpus)
    else:
        # 단일 GPU 순차 처리
        logger.info("Using single GPU for sequential prediction generation")
        _generate_predictions_worker(config)


if __name__ == "__main__":
    main()

