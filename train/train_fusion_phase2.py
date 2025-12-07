import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.fusion_dataset import FusionDataset, FusionSample, load_fusion_metadata
from datasets.transforms import build_transforms
from models.backbone_factory import set_backbone_trainable
from models.fusion_models import Phase2MultiLevelFusionModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.cv_split import create_fusion_splits
from utils.logger import setup_logger
from utils.lr_schedulers import lr_warmup_cosine, set_param_group_lrs
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def find_phase1_fusion_checkpoint(
    phase1_ckpt_dir: str,
    fold_idx: int,
    nail_name: str,
    conj_name: str,
    use_demographics: bool,
) -> str | None:
    """
    Find Phase 1 Fusion checkpoint for a specific fold, pair, and demo version.
    Returns the path to the checkpoint, or None if not found.
    """
    pair_name = f"nail-{nail_name}_conj-{conj_name}".replace("/", "_")
    demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
    ckpt_filename = f"p1_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt"
    ckpt_path = os.path.join(phase1_ckpt_dir, ckpt_filename)
    
    if os.path.exists(ckpt_path):
        return ckpt_path
    return None


def find_phase2_checkpoint(
    ckpt_dir: str,
    fold_idx: int,
    nail_name: str,
    conj_name: str,
    use_demographics: bool,
) -> str | None:
    """
    Find Phase 2 checkpoint for a specific fold, pair, and demo version.
    Returns the path to the checkpoint, or None if not found.
    """
    pair_name = f"nail-{nail_name}_conj-{conj_name}".replace("/", "_")
    demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
    ckpt_filename = f"p2_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    
    if os.path.exists(ckpt_path):
        return ckpt_path
    return None


def load_phase2_metrics_history(
    ckpt_dir: str,
    fold_idx: int,
    nail_name: str,
    conj_name: str,
    use_demographics: bool,
) -> List[Dict[str, Any]]:
    """
    Load Phase 2 metrics history from JSON file.
    Returns empty list if not found.
    """
    pair_name = f"nail-{nail_name}_conj-{conj_name}".replace("/", "_")
    demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
    metrics_path = os.path.join(
        ckpt_dir, f"p2_fold{fold_idx}_{pair_name}{demo_suffix}_metrics.json"
    )
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except:
            pass
    return []


def load_backbone_from_phase1_fusion_checkpoint(
    backbone: nn.Module, checkpoint_path: str, backbone_type: str
) -> None:
    """
    Load backbone weights from a Phase 1 Fusion checkpoint.
    Phase 1 Fusion model has structure: Phase1FusionModel(nail_backbone, conj_backbone, fusion_head)
    We need to extract either nail_backbone or conj_backbone.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model_state"]
    
    # Phase 1 Fusion model keys look like "nail_backbone.layer1.weight", "conj_backbone.layer1.weight", etc.
    # Extract only the backbone we need
    backbone_state = {}
    prefix = f"{backbone_type}_backbone."
    for key, value in model_state.items():
        if key.startswith(prefix):
            # Remove prefix to get backbone parameter name
            backbone_key = key[len(prefix):]
            backbone_state[backbone_key] = value
    
    # Load the backbone state
    if backbone_state:
        missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
    else:
        # Fallback: try loading directly (might be different structure)
        try:
            backbone.load_state_dict(model_state, strict=False)
        except Exception:
            pass


def build_fusion_dataloaders_for_fold(
    samples: List[FusionSample],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_s = [samples[i] for i in train_idx]
    val_s = [samples[i] for i in val_idx]

    nail_train_tf = build_transforms("train", image_size=image_size, modality="nail")
    conj_train_tf = build_transforms("train", image_size=image_size, modality="conj")
    nail_eval_tf = build_transforms("val", image_size=image_size, modality="nail")
    conj_eval_tf = build_transforms("val", image_size=image_size, modality="conj")

    train_ds = FusionDataset(
        train_s, nail_transform=nail_train_tf, conj_transform=conj_train_tf
    )
    val_ds = FusionDataset(
        val_s, nail_transform=nail_eval_tf, conj_transform=conj_eval_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # 메모리 절약을 위해 False로 변경
        prefetch_factor=1,  # 메모리 절약 (기본값 2에서 1로 감소)
        persistent_workers=False,  # 메모리 절약
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # 메모리 절약을 위해 False로 변경
        prefetch_factor=1,  # 메모리 절약
        persistent_workers=False,  # 메모리 절약
    )
    return train_loader, val_loader


def train_fusion_phase2(config: Dict[str, Any], device_id: int | None = None) -> None:
    # Select specific GPU if device_id is provided
    if device_id is not None:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))
    
    # Set log and checkpoint directory to logs/fusion_phase2
    log_dir = "logs/fusion_phase2"
    os.makedirs(log_dir, exist_ok=True)
    log_file = config.get("log_file", os.path.join(log_dir, "fusion_phase2.log"))
    logger = setup_logger("train_fusion_p2", log_file)

    # TODO: create fusion_meta.csv under /home/monetai/Desktop/URP/multimodal_dataset
    samples: List[FusionSample] = load_fusion_metadata(config["fusion_metadata_csv"])

    import pandas as pd

    df = pd.DataFrame(
        {
            "patient_id": [s.patient_id for s in samples],
        }
    )
    splits = create_fusion_splits(df, patient_col="patient_id")

    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    nail_backbone_names = config["nail_backbones"]
    conj_backbone_names = config["conj_backbones"]

    total_epochs = config.get("epochs", 80)
    warmup_epochs = config.get("warmup_epochs", 10)
    criterion = nn.SmoothL1Loss(beta=1.0)

    # Use logs/fusion_phase2 for checkpoints as well
    ckpt_dir = config.get("checkpoint_dir", log_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("Start Phase 2 multi-level fusion training")
    
    # Filter tasks if assigned_tasks is provided (multi-GPU mode)
    assigned_tasks = config.get("assigned_tasks", None)
    if assigned_tasks is not None:
        logger.info(f"GPU {device_id}: Processing {len(assigned_tasks)} assigned tasks")
    else:
        logger.info(f"Processing all tasks on device {device_id if device_id is not None else 'default'}")

    for fold_idx, (train_idx, val_idx) in enumerate(splits.folds):
        logger.info(f"Fold {fold_idx + 1}/{len(splits.folds)}")

        for nail_name in nail_backbone_names:
            for conj_name in conj_backbone_names:
                # Skip if this task is not assigned to this GPU (multi-GPU mode)
                if assigned_tasks is not None:
                    if (fold_idx, nail_name, conj_name) not in assigned_tasks:
                        continue
                
                # 파일 경로에 사용할 수 있도록 슬래시를 언더스코어로 변환
                pair_name = f"nail-{nail_name}_conj-{conj_name}".replace("/", "_")
                logger.info(f"Training pair: {pair_name}")
                
                # 각 실험을 try-except로 감싸서 오류가 나도 다음 실험 계속 진행
                try:
                    # OOM 방지: Phase 2는 multi-level features를 사용하므로 메모리 사용량이 큼
                    # 특정 모델 조합에 대해 batch size를 줄임
                    pair_batch_size = batch_size
                    # 큰 모델들이 포함된 조합은 batch size를 더 줄임
                    if "regnety_120" in nail_name or "regnety_120" in conj_name:
                        pair_batch_size = max(4, batch_size // 2)  # 최소 4
                        logger.info(f"Using reduced batch size {pair_batch_size} for regnety_120 combination")
                    elif "rexnetr_300" in nail_name or "rexnetr_300" in conj_name:
                        pair_batch_size = max(4, batch_size // 2)  # 최소 4
                        logger.info(f"Using reduced batch size {pair_batch_size} for rexnetr_300 combination")
                    elif "convnext_base" in nail_name or "convnext_base" in conj_name:
                        pair_batch_size = max(8, batch_size // 2)  # 최소 8
                        logger.info(f"Using reduced batch size {pair_batch_size} for convnext_base combination")
                    
                    train_loader, val_loader = build_fusion_dataloaders_for_fold(
                        samples,
                        train_idx,
                        val_idx,
                        image_size=image_size,
                        batch_size=pair_batch_size,
                        num_workers=num_workers,
                    )

                    # Check if demographic features should be used
                    use_demographics = config.get("use_demographics", True)
                    demo_dim = 2 if use_demographics else 0
                    
                    model = Phase2MultiLevelFusionModel(
                        nail_backbone_name=nail_name,
                        conj_backbone_name=conj_name,
                        demo_dim=demo_dim,
                    ).to(device)
                    
                    if not use_demographics:
                        logger.info("Training WITHOUT demographic features (age, gender)")
                    else:
                        logger.info("Training WITH demographic features (age, gender)")

                    # Load pretrained backbones from Phase 1 Fusion checkpoints
                    # Phase 2 uses Phase 1 Fusion checkpoints, matching the demo version
                    phase1_ckpt_dir = config.get("phase1_checkpoint_dir", "checkpoints/fusion_phase1")
                    load_pretrained_backbones = config.get("load_pretrained_backbones", True)
                    skip_if_no_phase1 = config.get("skip_if_no_phase1_checkpoint", False)
                    
                    # Track which pretrained source is used
                    pretrained_source = None
                    
                    if load_pretrained_backbones:
                        # Find Phase 1 Fusion checkpoint for this fold, pair, and demo version
                        phase1_ckpt = find_phase1_fusion_checkpoint(
                            phase1_ckpt_dir, fold_idx, nail_name, conj_name, use_demographics
                        )
                        if phase1_ckpt:
                            logger.info(
                                f"[PRETRAINED: Phase 1 Fusion] Loading backbones from Phase 1 Fusion checkpoint: {phase1_ckpt}"
                            )
                            pretrained_source = "Phase1_Fusion"
                            load_backbone_from_phase1_fusion_checkpoint(
                                model.nail_backbone, phase1_ckpt, "nail"
                            )
                            load_backbone_from_phase1_fusion_checkpoint(
                                model.conj_backbone, phase1_ckpt, "conj"
                            )
                        else:
                            if skip_if_no_phase1:
                                logger.warning(
                                    f"[SKIP] No Phase 1 Fusion checkpoint found for fold {fold_idx}, "
                                    f"pair ({nail_name}, {conj_name}), demo={use_demographics}. "
                                    f"Skipping this combination (skip_if_no_phase1_checkpoint=True)."
                                )
                                continue
                            else:
                                logger.warning(
                                    f"[PRETRAINED: ImageNet] No Phase 1 Fusion checkpoint found for fold {fold_idx}, "
                                    f"pair ({nail_name}, {conj_name}), demo={use_demographics}. "
                                    f"Using ImageNet pretrained backbones instead."
                                )
                                pretrained_source = "ImageNet"
                    else:
                        logger.info(
                            f"[PRETRAINED: ImageNet] load_pretrained_backbones=False, using ImageNet pretrained backbones."
                        )
                        pretrained_source = "ImageNet"

                    for p in model.nail_backbone.parameters():
                        p.requires_grad = False
                    for p in model.conj_backbone.parameters():
                        p.requires_grad = False

                    head_params = list(model.fusion_head.parameters())
                    nail_params = list(model.nail_backbone.parameters())
                    conj_params = list(model.conj_backbone.parameters())

                    optimizer = AdamW(
                        [
                            {"params": head_params, "lr": 0.0},
                            {"params": nail_params, "lr": 0.0},
                            {"params": conj_params, "lr": 0.0},
                        ],
                        weight_decay=float(config.get("weight_decay", 1e-4)),
                        betas=(0.9, 0.999),
                        eps=1e-8,
                    )

                    # Mixed precision training scaler (메모리 절약)
                    scaler = torch.cuda.amp.GradScaler()

                    # Try to resume from checkpoint
                    start_epoch = 0
                    best_metrics_for_pair: List[Dict[str, Any]] = []
                    backbone_unfrozen = False
                    
                    phase2_ckpt = find_phase2_checkpoint(
                        ckpt_dir, fold_idx, nail_name, conj_name, use_demographics
                    )
                    
                    if phase2_ckpt:
                        logger.info(f"Resuming from checkpoint: {phase2_ckpt}")
                        checkpoint_state = load_checkpoint(phase2_ckpt, model=model, optimizer=optimizer)
                        start_epoch = checkpoint_state.get("epoch", 0) + 1
                        logger.info(f"Resuming from epoch {start_epoch}/{total_epochs}")
                        
                        # Load metrics history
                        best_metrics_for_pair = load_phase2_metrics_history(
                            ckpt_dir, fold_idx, nail_name, conj_name, use_demographics
                        )
                        
                        # Check if backbone was already unfrozen
                        if start_epoch >= 20:
                            backbone_unfrozen = True
                            set_backbone_trainable(
                                model.nail_backbone, train_last_n_stages=2
                            )
                            set_backbone_trainable(
                                model.conj_backbone, train_last_n_stages=2
                            )
                    else:
                        logger.info("No checkpoint found, starting from scratch")

                    for epoch in range(start_epoch, total_epochs):
                        # LR Schedule:
                        # Epoch 0-10: LR_head 0 → 5e-4 (warmup), LR_bb = 0
                        # Epoch 10-20: LR_head ~5e-4 (keep flat), LR_bb = 0
                        # Epoch 20-80: LR_head cosine decay 3e-4 → 3e-6, LR_bb cosine decay 5e-5 → 5e-7
                        
                        if epoch < 10:
                            # Linear warmup: 0 → 5e-4
                            lr_head = 5e-4 * (epoch + 1) / 10.0
                            lr_bb = 0.0
                        elif epoch < 20:
                            # Keep LR_head flat at 5e-4
                            lr_head = 5e-4
                            lr_bb = 0.0
                        else:
                            # Unfreeze backbone last 2 stages (only once at epoch 20)
                            if not backbone_unfrozen:
                                set_backbone_trainable(
                                    model.nail_backbone, train_last_n_stages=2
                                )
                                set_backbone_trainable(
                                    model.conj_backbone, train_last_n_stages=2
                                )
                                backbone_unfrozen = True
                            # Cosine decay from epoch 20 to 80
                            # For LR_head: 3e-4 → 3e-6
                            # For LR_bb: 5e-5 → 5e-7
                            progress = float(epoch - 20) / float(80 - 20)  # 0 to 1
                            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                            lr_head = 3e-6 + (3e-4 - 3e-6) * cosine
                            lr_bb = 5e-7 + (5e-5 - 5e-7) * cosine

                        set_param_group_lrs(
                            optimizer.param_groups,
                            [lr_head, lr_bb, lr_bb],
                        )

                        model.train()
                        train_losses: List[float] = []
                        for batch in tqdm(
                            train_loader,
                            desc=f"Fold {fold_idx} {pair_name} Epoch {epoch+1}/{total_epochs} [train]",
                        ):
                            nail_img, conj_img, hb, _, demo = batch
                            nail_img = nail_img.to(device, non_blocking=False)  # pin_memory=False이므로 non_blocking=False
                            conj_img = conj_img.to(device, non_blocking=False)
                            hb = hb.to(device, dtype=torch.float32)
                            demo = demo.to(device, dtype=torch.float32) if use_demographics else None

                            optimizer.zero_grad()
                            # Mixed precision training (FP16)으로 메모리 절약
                            with torch.cuda.amp.autocast():
                                preds = model(nail_img, conj_img, demo)
                                loss = criterion(preds, hb)
                            # Gradient scaling for mixed precision
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            train_losses.append(loss.item())

                        model.eval()
                        val_preds_list: List[np.ndarray] = []
                        val_targets_list: List[np.ndarray] = []
                        with torch.no_grad():
                            for batch in tqdm(
                                val_loader,
                                desc=f"Fold {fold_idx} {pair_name} Epoch {epoch+1}/{total_epochs} [val]",
                            ):
                                nail_img, conj_img, hb, _, demo = batch
                                nail_img = nail_img.to(device, non_blocking=False)  # pin_memory=False이므로 non_blocking=False
                                conj_img = conj_img.to(device, non_blocking=False)
                                hb = hb.to(device, dtype=torch.float32)
                                demo = demo.to(device, dtype=torch.float32) if use_demographics else None

                                # Mixed precision inference (메모리 절약)
                                with torch.cuda.amp.autocast():
                                    preds = model(nail_img, conj_img, demo)
                                val_preds_list.append(tensor_to_numpy(preds))
                                val_targets_list.append(tensor_to_numpy(hb))

                        y_true = np.concatenate(val_targets_list)
                        y_pred = np.concatenate(val_preds_list)
                        metrics = compute_regression_metrics(y_true, y_pred)
                        metrics["train_loss"] = float(np.mean(train_losses))
                        logger.info(
                            f"Fold {fold_idx} Pair {pair_name} Epoch {epoch}: {metrics}"
                        )

                        if metrics["r2"] > 0:
                            best_metrics_for_pair.append(
                                {"epoch": epoch, "metrics": metrics}
                            )
                            best_metrics_for_pair = sorted(
                                best_metrics_for_pair,
                                key=lambda m: (
                                    m["metrics"]["mae"],
                                    -m["metrics"]["acc@1.0"],
                                    -m["metrics"]["acc@0.5"],
                                    -m["metrics"]["r2"],
                                ),
                            )
                            best_metrics_for_pair = best_metrics_for_pair[:1]
                            if best_metrics_for_pair[0]["epoch"] == epoch:
                                demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
                                # Phase 1처럼 best.pt로 저장하여 이전 체크포인트 자동 덮어쓰기
                                ckpt_path = os.path.join(
                                    ckpt_dir,
                                    f"p2_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt",
                                )
                                save_checkpoint(
                                    ckpt_path,
                                    epoch=epoch,
                                    model=model,
                                    optimizer=optimizer,
                                    metrics=metrics,
                                    extra={
                                        "fold": fold_idx,
                                        "pair": pair_name,
                                        "use_demographics": use_demographics,
                                        "pretrained_source": pretrained_source or "ImageNet",
                                    },
                                )

                    demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
                    hist_path = os.path.join(
                        ckpt_dir, f"p2_fold{fold_idx}_{pair_name}{demo_suffix}_metrics.json"
                    )
                    # Add pretrained_source info to metrics
                    metrics_data = {
                        "pretrained_source": pretrained_source or "ImageNet",
                        "best_metrics": best_metrics_for_pair
                    }
                    with open(hist_path, "w") as f:
                        json.dump(metrics_data, f, indent=2)
                    
                    # Log pretrained source info for easy tracking
                    if best_metrics_for_pair:
                        best_epoch = best_metrics_for_pair[0]["epoch"]
                        logger.info(
                            f"[COMPLETED] Fold {fold_idx} Pair {pair_name} ({demo_suffix}): "
                            f"Best epoch {best_epoch}, Pretrained source: {pretrained_source or 'ImageNet'}"
                        )
                
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(
                        f"OOM Error for Fold {fold_idx} Pair {pair_name}: {str(e)}. "
                        f"Skipping this combination and continuing with next."
                    )
                    # 메모리 정리
                    if 'model' in locals():
                        del model
                    if 'optimizer' in locals():
                        del optimizer
                    if 'train_loader' in locals():
                        del train_loader
                    if 'val_loader' in locals():
                        del val_loader
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(
                        f"ERROR in Fold {fold_idx} Pair {pair_name}: {str(e)}",
                        exc_info=True
                    )
                    # 메모리 정리
                    if 'model' in locals():
                        del model
                    if 'optimizer' in locals():
                        del optimizer
                    if 'train_loader' in locals():
                        del train_loader
                    if 'val_loader' in locals():
                        del val_loader
                    torch.cuda.empty_cache()
                    # 오류가 발생해도 다음 실험 계속 진행
                    continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train_fusion_phase2(config)


if __name__ == "__main__":
    main()


