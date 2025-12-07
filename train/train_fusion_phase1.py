import argparse
import json
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
from models.fusion_models import Phase1FusionModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.cv_split import create_fusion_splits
from utils.logger import setup_logger
from utils.lr_schedulers import lr_warmup_cosine, set_param_group_lrs
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def find_best_checkpoint(
    checkpoint_dir: str, backbone_name: str, modality: str, top_k: int = 5
) -> str | None:
    """
    Find the best checkpoint for a given backbone by checking all checkpoint files.
    Returns the path to the best checkpoint, or None if not found.
    """
    backbone_dir_name = backbone_name.replace("/", "_")
    backbone_dir = os.path.join(checkpoint_dir, backbone_dir_name)
    
    if not os.path.exists(backbone_dir):
        return None
    
    checkpoint_files = [
        f
        for f in os.listdir(backbone_dir)
        if f.endswith(".pt") and f.startswith(f"{modality}_{backbone_dir_name}_epoch")
    ]
    
    if not checkpoint_files:
        return None
    
    # Load all checkpoints and find the best one based on metrics
    best_ckpt_path = None
    best_score = (float("inf"), 0.0, 0.0, 0.0)  # (mae, -acc@1.0, -acc@0.5, -r2)
    
    for ckpt_file in checkpoint_files:
        try:
            ckpt_path = os.path.join(backbone_dir, ckpt_file)
            state = torch.load(ckpt_path, map_location="cpu")
            if "metrics" in state:
                metrics = state["metrics"]
                # Skip if R² <= 0
                if metrics.get("r2", 0.0) <= 0:
                    continue
                # Score: lower MAE is better, then higher acc@1.0, acc@0.5, r2
                score = (
                    metrics.get("mae", float("inf")),
                    -metrics.get("acc@1.0", 0.0),
                    -metrics.get("acc@0.5", 0.0),
                    -metrics.get("r2", 0.0),
                )
                if score < best_score:
                    best_score = score
                    best_ckpt_path = ckpt_path
        except (ValueError, IndexError, Exception) as e:
            continue
    
    return best_ckpt_path


def load_backbone_from_checkpoint(
    backbone: nn.Module, checkpoint_path: str, modality: str
) -> None:
    """
    Load backbone weights from a single-modality checkpoint.
    The checkpoint contains a Sequential model with [backbone, head],
    so we need to extract just the backbone part.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model_state"]
    
    # The checkpoint was saved as Sequential([backbone, head])
    # We need to extract only the backbone weights
    # In Sequential, backbone is the first module (index 0)
    backbone_state = {}
    for key, value in model_state.items():
        # Sequential model keys look like "0.weight", "0.bias", "1.weight", etc.
        # "0" is backbone, "1" is head
        if key.startswith("0."):
            # Remove "0." prefix to get backbone parameter name
            backbone_key = key[2:]
            backbone_state[backbone_key] = value
    
    # Load the backbone state
    missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
    if missing_keys and len(backbone_state) == 0:
        # If no keys matched with "0." prefix, try loading directly (might be different structure)
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_fusion_phase1(config: Dict[str, Any], device_id: int | None = None) -> None:
    # Select specific GPU if device_id is provided
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    set_seed(config.get("seed", 42))
    logger = setup_logger("train_fusion_p1", config.get("log_file"))

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

    total_epochs = config.get("epochs", 60)
    warmup_epochs_head = config.get("warmup_epochs_head", 5)
    criterion = nn.SmoothL1Loss(beta=1.0)

    ckpt_dir = config.get("checkpoint_dir", "checkpoints/fusion_phase1")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get assigned tasks if running in multi-GPU mode
    assigned_tasks = config.get("assigned_tasks", None)  # List of (fold_idx, nail_name, conj_name)
    
    logger.info("Start Phase 1 fusion training")
    if device_id is not None:
        logger.info(f"Running on GPU {device_id} with {len(assigned_tasks) if assigned_tasks else 'all'} tasks")

    # Generate all tasks: (fold_idx, nail_name, conj_name)
    all_tasks = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits.folds):
        for nail_name in nail_backbone_names:
            for conj_name in conj_backbone_names:
                all_tasks.append((fold_idx, nail_name, conj_name, train_idx, val_idx))
    
    # Filter tasks if assigned_tasks is provided (multi-GPU mode)
    if assigned_tasks is not None:
        # assigned_tasks is list of (fold_idx, nail_name, conj_name) tuples
        task_dict = {(f, n, c): (tr, val) for f, n, c, tr, val in all_tasks}
        tasks_to_run = []
        for fold_idx, nail_name, conj_name in assigned_tasks:
            if (fold_idx, nail_name, conj_name) in task_dict:
                train_idx, val_idx = task_dict[(fold_idx, nail_name, conj_name)]
                tasks_to_run.append((fold_idx, nail_name, conj_name, train_idx, val_idx))
        all_tasks = tasks_to_run
        logger.info(f"GPU {device_id}: Processing {len(all_tasks)} tasks")

    for fold_idx, nail_name, conj_name, train_idx, val_idx in all_tasks:
        logger.info(f"Fold {fold_idx + 1}/{len(splits.folds)} - Pair: nail-{nail_name}_conj-{conj_name}")
        # Replace '/' with '_' in backbone names to avoid path issues
        safe_nail_name = nail_name.replace("/", "_")
        safe_conj_name = conj_name.replace("/", "_")
        pair_name = f"nail-{safe_nail_name}_conj-{safe_conj_name}"
        logger.info(f"Training pair: {pair_name}")
        
        try:
            # OOM 방지: rexnetr_300이 포함된 조합은 배치 사이즈를 낮춤
            # OOM 테스트 결과: rexnetr_300이 포함된 조합은 backward pass 시 메모리 부족
            pair_batch_size = batch_size
            if "rexnetr_300" in nail_name or "rexnetr_300" in conj_name:
                pair_batch_size = 64  # rexnetr_300 포함 조합은 배치 64로 낮춤
                logger.info(f"Using reduced batch size {pair_batch_size} for rexnetr_300 combination")

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
        
            model = Phase1FusionModel(
            nail_backbone_name=nail_name,
            conj_backbone_name=conj_name,
            demo_dim=demo_dim,
            ).to(device)
        
            if not use_demographics:
                logger.info("Training WITHOUT demographic features (age, gender)")
            else:
                logger.info("Training WITH demographic features (age, gender)")

            # Optionally load pretrained backbones from single-modality checkpoints
            nail_ckpt_dir = config.get("nail_checkpoint_dir", "checkpoints/nail")
            conj_ckpt_dir = config.get("conj_checkpoint_dir", "checkpoints/conj")
            load_pretrained_backbones = config.get("load_pretrained_backbones", True)
        
            if load_pretrained_backbones:
                nail_ckpt = find_best_checkpoint(
                nail_ckpt_dir, nail_name, "nail", top_k=5
            )
            if nail_ckpt:
                logger.info(f"Loading nail backbone from {nail_ckpt}")
                load_backbone_from_checkpoint(
                    model.nail_backbone, nail_ckpt, "nail"
                )
            else:
                logger.warning(
                    f"No checkpoint found for nail backbone {nail_name}, using ImageNet pretrained"
                )
            
            conj_ckpt = find_best_checkpoint(
                conj_ckpt_dir, conj_name, "conj", top_k=5
            )
            if conj_ckpt:
                logger.info(f"Loading conj backbone from {conj_ckpt}")
                load_backbone_from_checkpoint(
                    model.conj_backbone, conj_ckpt, "conj"
                )
            else:
                logger.warning(
                    f"No checkpoint found for conj backbone {conj_name}, using ImageNet pretrained"
                )

            for p in model.nail_backbone.parameters():
                p.requires_grad = False
            for p in model.conj_backbone.parameters():
                p.requires_grad = False

            optimizer = AdamW(
            [
                {"params": model.fusion_head.parameters(), "lr": 0.0},
                {"params": model.nail_backbone.parameters(), "lr": 0.0},
                {"params": model.conj_backbone.parameters(), "lr": 0.0},
            ],
            weight_decay=float(config.get("weight_decay", 1e-4)),
            betas=(0.9, 0.999),
            eps=1e-8,
            )

            best_metrics_for_pair: List[Dict[str, Any]] = []
            all_metrics_for_pair: List[Dict[str, Any]] = []  # 모든 epoch의 metrics 저장

            for epoch in range(total_epochs):
                # LR schedule according to README:
                # Epoch 0-5: LR_head 0→5e-4 (linear warmup), LR_bb=0
                # Epoch 5-60: LR_head cosine decay 5e-4→~5e-6
                # LR_bb: Epoch 10부터 1e-4로 시작해서 cosine decay
                if epoch < 5:
                    # Warmup phase: LR_head linear warmup, LR_bb=0
                    lr_head = lr_warmup_cosine(
                        epoch,
                        total_epochs=5,  # Warmup for 5 epochs
                        warmup_epochs=5,
                        base_lr=5e-4,
                        min_lr=5e-4,  # During warmup, stay at base_lr
                    )
                    lr_bb = 0.0
                elif epoch < 10:
                    # Epoch 5-10: LR_head cosine decay starts, LR_bb still 0
                    lr_head = lr_warmup_cosine(
                        epoch,
                        total_epochs=total_epochs,
                        warmup_epochs=5,  # Warmup ended at epoch 5
                        base_lr=5e-4,
                        min_lr=5e-6,
                    )
                    lr_bb = 0.0
                else:
                    # Epoch 10-60: LR_head continues cosine decay, LR_bb cosine decay starts
                    # LR_head: cosine decay from 5e-4 to 5e-6 over epochs 5-60
                    lr_head = lr_warmup_cosine(
                        epoch,
                        total_epochs=total_epochs,
                        warmup_epochs=5,  # Warmup ended at epoch 5
                        base_lr=5e-4,
                        min_lr=5e-6,
                    )
                    # LR_bb: cosine decay from 1e-4 to 1e-6 over epochs 10-60
                    lr_bb = lr_warmup_cosine(
                        epoch,
                        total_epochs=total_epochs,
                        warmup_epochs=10,  # Backbone starts at epoch 10
                        base_lr=1e-4,
                        min_lr=1e-6,
                    )
                    # Unfreeze last 1-2 stages of backbones
                    set_backbone_trainable(model.nail_backbone, train_last_n_stages=2)
                    set_backbone_trainable(model.conj_backbone, train_last_n_stages=2)

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
                    nail_img = nail_img.to(device)
                    conj_img = conj_img.to(device)
                    hb = hb.to(device, dtype=torch.float32)
                    demo = demo.to(device, dtype=torch.float32) if use_demographics else None

                    optimizer.zero_grad()
                    preds = model(nail_img, conj_img, demo)
                    loss = criterion(preds, hb)
                    loss.backward()
                    optimizer.step()
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
                        nail_img = nail_img.to(device)
                        conj_img = conj_img.to(device)
                        hb = hb.to(device, dtype=torch.float32)
                        demo = demo.to(device, dtype=torch.float32) if use_demographics else None

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

                # R² 조건 없이 모든 epoch의 metrics 기록
                all_metrics_for_pair.append(
                    {"epoch": epoch, "metrics": metrics}
                )
            
                # Best metrics 추적 (checkpoint 저장용)
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
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"p1_fold{fold_idx}_{pair_name}{demo_suffix}_best.pt",
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
                        },
                    )

            demo_suffix = "_w_demo" if use_demographics else "_wo_demo"
            # 모든 epoch의 metrics 저장
            hist_path = os.path.join(
                ckpt_dir, f"p1_fold{fold_idx}_{pair_name}{demo_suffix}_metrics.json"
            )
            os.makedirs(os.path.dirname(hist_path), exist_ok=True)
            with open(hist_path, "w") as f:
                json.dump(all_metrics_for_pair, f, indent=2)

            # Evaluate on external test set using best validation model
            if best_metrics_for_pair:
                best_epoch = best_metrics_for_pair[0]["epoch"]
                best_ckpt_path = os.path.join(
                    ckpt_dir,
                    f"p1_fold{fold_idx}_{pair_name}{demo_suffix}_epoch{best_epoch}.pt",
                )
                
                if os.path.exists(best_ckpt_path):
                    logger.info(
                        f"Loading best model (epoch {best_epoch}) for external test evaluation"
                    )
                    load_checkpoint(best_ckpt_path, model=model)
                    model.eval()
                    
                    # Build external test loader
                    test_samples = [samples[i] for i in splits.external_test_idx]
                    nail_test_tf = build_transforms("test", image_size=image_size, modality="nail")
                    conj_test_tf = build_transforms("test", image_size=image_size, modality="conj")
                    test_ds = FusionDataset(
                        test_samples, nail_transform=nail_test_tf, conj_transform=conj_test_tf
                    )
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                    
                    # Evaluate on test set
                    test_preds_list: List[np.ndarray] = []
                    test_targets_list: List[np.ndarray] = []
                    with torch.no_grad():
                        for batch in tqdm(
                            test_loader,
                            desc=f"Fold {fold_idx} {pair_name} [test]",
                        ):
                            nail_img, conj_img, hb, _, demo = batch
                            nail_img = nail_img.to(device)
                            conj_img = conj_img.to(device)
                            hb = hb.to(device, dtype=torch.float32)
                            demo = demo.to(device, dtype=torch.float32) if use_demographics else None
                            
                            preds = model(nail_img, conj_img, demo)
                            test_preds_list.append(tensor_to_numpy(preds))
                            test_targets_list.append(tensor_to_numpy(hb))
                    
                    y_true_test = np.concatenate(test_targets_list)
                    y_pred_test = np.concatenate(test_preds_list)
                    test_metrics = compute_regression_metrics(y_true_test, y_pred_test)
                    
                    logger.info(
                        f"Fold {fold_idx} Pair {pair_name} External Test Metrics: {test_metrics}"
                    )
                    
                    # Save test metrics
                    test_metrics_path = os.path.join(
                        ckpt_dir, f"p1_fold{fold_idx}_{pair_name}{demo_suffix}_test_metrics.json"
                    )
                    os.makedirs(os.path.dirname(test_metrics_path), exist_ok=True)
                    with open(test_metrics_path, "w") as f:
                        json.dump(test_metrics, f, indent=2)
        
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
                f"Error for Fold {fold_idx} Pair {pair_name}: {str(e)}. "
                f"Skipping this combination and continuing with next."
            )
            import traceback
            logger.error(traceback.format_exc())
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train_fusion_phase1(config)


if __name__ == "__main__":
    main()


