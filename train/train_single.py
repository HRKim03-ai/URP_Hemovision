import argparse
import copy
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

from datasets.conj_dataset import (
    ConjDataset,
    ConjSample,
    load_conj_metadata,
    split_conj_by_patient,
)
from datasets.nail_dataset import (
    NailDataset,
    NailSample,
    load_nail_metadata,
    split_nail_by_patient,
)
from datasets.transforms import build_transforms
from models.backbone_factory import create_backbone, get_backbone_output_dim
from models.heads import RegressionHead
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger
from utils.lr_schedulers import lr_warmup_cosine, set_param_group_lrs
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def build_single_dataloaders(
    modality: str,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    if modality == "nail":
        # TODO: create a CSV like /home/monetai/Desktop/URP/singlemodal_dataset/nail_meta.csv
        samples: List[NailSample] = load_nail_metadata(config["metadata_csv"])
        train_s, val_s, test_s = split_nail_by_patient(samples)
        train_tf = build_transforms("train", image_size=image_size, modality="nail")
        val_tf = build_transforms("val", image_size=image_size, modality="nail")
        test_tf = build_transforms("test", image_size=image_size, modality="nail")
        train_ds = NailDataset(train_s, transform=train_tf)
        val_ds = NailDataset(val_s, transform=val_tf)
        test_ds = NailDataset(test_s, transform=test_tf)
    elif modality == "conj":
        # TODO: create CSVs like conj_folder1.csv, conj_folder2.csv under singlemodal_dataset
        samples_c: List[ConjSample] = load_conj_metadata(
            config["metadata_csv_folder1"], config["metadata_csv_folder2"]
        )
        train_s, val_s, test_s = split_conj_by_patient(samples_c)
        train_tf = build_transforms("train", image_size=image_size, modality="conj")
        val_tf = build_transforms("val", image_size=image_size, modality="conj")
        test_tf = build_transforms("test", image_size=image_size, modality="conj")
        train_ds = ConjDataset(train_s, transform=train_tf)
        val_ds = ConjDataset(val_s, transform=val_tf)
        test_ds = ConjDataset(test_s, transform=test_tf)
    else:
        raise ValueError(f"Unknown modality {modality}")

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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_single(
    config: Dict[str, Any],
    modality: str,
    backbone_start: int | None = None,
    backbone_end: int | None = None,
) -> None:
    # 선택적으로 특정 GPU 로 고정
    device_id = config.get("device_id")
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    set_seed(config.get("seed", 42))
    logger = setup_logger("train_single", config.get("log_file"))

    # 하나 또는 여러 개의 backbone 을 순차적으로 학습할 수 있도록 지원
    backbone_names = config.get("backbone_names")
    if backbone_names is None:
        backbone_names = [config["backbone_name"]]

    # 병렬 실행 시 백본 리스트를 분할해서 사용할 수 있도록 범위 슬라이싱 지원
    if backbone_start is not None or backbone_end is not None:
        s = backbone_start or 0
        e = backbone_end or len(backbone_names)
        backbone_names = backbone_names[s:e]

    total_epochs = config.get("epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 5)
    freeze_epochs = config.get("freeze_epochs", 10)
    ckpt_dir_root = config.get("checkpoint_dir", f"checkpoints/{modality}")
    os.makedirs(ckpt_dir_root, exist_ok=True)

    logger.info(
        f"Start training single-modality models for {modality}. "
        f"Backbones: {backbone_names}"
    )

    for backbone_name in backbone_names:
        logger.info(f"=== Training backbone: {backbone_name} ===")

        # 고정 batch size 사용 (예: config 에서 16으로 설정)
        train_loader, val_loader, test_loader = build_single_dataloaders(
            modality, config
        )
        
        # Log dataset sizes for debugging
        logger.info(
            f"[{backbone_name}] Dataset sizes - "
            f"Train: {len(train_loader.dataset)}, "
            f"Val: {len(val_loader.dataset)}, "
            f"Test: {len(test_loader.dataset)}"
        )
        
        if len(val_loader.dataset) == 0:
            logger.error(
                f"[{backbone_name}] Validation set is empty! "
                "This will cause NaN metrics. Check your data split."
            )

        backbone = create_backbone(
            backbone_name, pretrained=True, features_only=False
        )
        feat_dim = get_backbone_output_dim(backbone)
        head = RegressionHead(feat_dim)
        model = nn.Sequential(backbone, head).to(device)

        backbone_params = list(backbone.parameters())
        head_params = list(head.parameters())
        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": 0.0},
                {"params": head_params, "lr": 0.0},
            ],
            weight_decay=float(config.get("weight_decay", 1e-4)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        criterion = nn.SmoothL1Loss(beta=1.0)

        best_models: List[Dict[str, Any]] = []
        best_metric_history: List[Dict[str, Any]] = []
        best_state_dict = None

        ckpt_dir = os.path.join(ckpt_dir_root, backbone_name.replace("/", "_"))
        os.makedirs(ckpt_dir, exist_ok=True)

        batch_size = config.get("batch_size", 32)
        logger.info(
            f"[{backbone_name}] Start training with fixed batch_size={batch_size}"
        )

        for epoch in range(total_epochs):
            lr_bb = lr_warmup_cosine(
                epoch, total_epochs, warmup_epochs, base_lr=3e-5, min_lr=3e-6
            )
            lr_head = lr_warmup_cosine(
                epoch, total_epochs, warmup_epochs, base_lr=1e-4, min_lr=1e-5
            )

            # Optionally freeze backbone for the first few epochs to stabilize head training
            if epoch < freeze_epochs:
                lr_bb = 0.0

            set_param_group_lrs(optimizer.param_groups, [lr_bb, lr_head])

            # Train
            model.train()
            train_losses: List[float] = []
            for batch in tqdm(
                train_loader,
                desc=f"{backbone_name} Epoch {epoch+1}/{total_epochs} [train, bs={batch_size}]",
            ):
                if modality == "nail":
                    imgs, hb, _ = batch
                else:
                    imgs, hb, _, _ = batch
                imgs = imgs.to(device)
                hb = hb.to(device, dtype=torch.float32)

                optimizer.zero_grad()
                preds = model(imgs)
                
                # Check for NaN in training
                if torch.isnan(preds).any() and epoch == 0:
                    logger.error(
                        f"[{backbone_name}] Epoch {epoch}: NaN in training predictions! "
                        f"This indicates a model initialization problem."
                    )
                
                loss = criterion(preds, hb)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.error(
                        f"[{backbone_name}] Epoch {epoch}: NaN loss detected! "
                        f"Pred stats: min={preds.min().item()}, max={preds.max().item()}, "
                        f"Target stats: min={hb.min().item()}, max={hb.max().item()}"
                    )
                    # Skip this batch
                    continue
                
                loss.backward()
                # Gradient clipping for stability on small datasets
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        if not has_nan_grad:
                            logger.error(
                                f"[{backbone_name}] Epoch {epoch}: NaN gradient detected!"
                            )
                        logger.error(f"  NaN gradient in: {name}")
                        has_nan_grad = True
                        # Zero out NaN gradients
                        param.grad[torch.isnan(param.grad)] = 0.0
                
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_preds_list: List[np.ndarray] = []
            val_targets_list: List[np.ndarray] = []
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"{backbone_name} Epoch {epoch+1}/{total_epochs} [val, bs={batch_size}]",
                ):
                    if modality == "nail":
                        imgs, hb, _ = batch
                    else:
                        imgs, hb, _, _ = batch
                    imgs = imgs.to(device)
                    hb = hb.to(device, dtype=torch.float32)
                    preds = model(imgs)
                    
                    # Check for NaN in predictions or targets
                    if torch.isnan(preds).any():
                        # More detailed debugging
                        nan_count = torch.isnan(preds).sum().item()
                        total_count = preds.numel()
                        logger.warning(
                            f"[{backbone_name}] Epoch {epoch}: NaN detected in predictions! "
                            f"NaN count: {nan_count}/{total_count}, "
                            f"Pred stats: min={preds.min().item() if not torch.isnan(preds).all() else 'all_nan'}, "
                            f"max={preds.max().item() if not torch.isnan(preds).all() else 'all_nan'}, "
                            f"mean={preds.mean().item() if not torch.isnan(preds).all() else 'all_nan'}"
                        )
                        
                        # Check model parameters for NaN
                        has_nan_params = False
                        for name, param in model.named_parameters():
                            if torch.isnan(param).any():
                                logger.error(
                                    f"[{backbone_name}] Epoch {epoch}: NaN in model parameter: {name}"
                                )
                                has_nan_params = True
                        
                        # Check backbone output
                        with torch.no_grad():
                            backbone_out = backbone(imgs)
                            if torch.isnan(backbone_out).any():
                                logger.error(
                                    f"[{backbone_name}] Epoch {epoch}: NaN in backbone output! "
                                    f"Backbone output shape: {backbone_out.shape}"
                                )
                    
                    if torch.isnan(hb).any():
                        logger.warning(
                            f"[{backbone_name}] Epoch {epoch}: NaN detected in targets!"
                        )
                    
                    val_preds_list.append(tensor_to_numpy(preds))
                    val_targets_list.append(tensor_to_numpy(hb))

            if len(val_preds_list) == 0 or len(val_targets_list) == 0:
                logger.error(
                    f"[{backbone_name}] Epoch {epoch}: Validation set is empty! "
                    f"val_loader length: {len(val_loader)}"
                )
                metrics = {
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "acc@0.5": 0.0,
                    "acc@1.0": 0.0,
                    "acc@2.0": 0.0,
                    "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                }
            else:
                y_true = np.concatenate(val_targets_list)
                y_pred = np.concatenate(val_preds_list)
                
                # Ensure shapes match
                if y_true.shape != y_pred.shape:
                    logger.error(
                        f"[{backbone_name}] Epoch {epoch}: Shape mismatch! "
                        f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}"
                    )
                    # Try to fix by flattening
                    y_true = y_true.flatten()
                    y_pred = y_pred.flatten()
                
                # Additional debugging info
                if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
                    logger.warning(
                        f"[{backbone_name}] Epoch {epoch}: NaN in concatenated arrays! "
                        f"y_true NaN count: {np.isnan(y_true).sum()}, "
                        f"y_pred NaN count: {np.isnan(y_pred).sum()}, "
                        f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}"
                    )
                
                # Log statistics for debugging R² issues (especially in early epochs)
                if epoch % 5 == 0 or epoch < 10:  # Log every 5 epochs or first 10 epochs
                    logger.info(
                        f"[{backbone_name}] Epoch {epoch} Stats - "
                        f"y_true: mean={np.mean(y_true):.3f}, std={np.std(y_true):.3f}, "
                        f"min={np.min(y_true):.3f}, max={np.max(y_true):.3f} | "
                        f"y_pred: mean={np.mean(y_pred):.3f}, std={np.std(y_pred):.3f}, "
                        f"min={np.min(y_pred):.3f}, max={np.max(y_pred):.3f}"
                    )
                
                metrics = compute_regression_metrics(y_true, y_pred)
                metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else float("nan")
            
            logger.info(f"[{backbone_name}] Epoch {epoch}: {metrics}")

            # Best model tracking (R2 > 0 조건 포함)
            if metrics["r2"] > 0:
                entry = {
                    "epoch": epoch,
                    "metrics": metrics,
                }
                best_models.append(entry)
                best_metric_history.append(metrics)

                best_models = sorted(
                    best_models,
                    key=lambda m: (
                        m["metrics"]["mae"],
                        -m["metrics"]["acc@1.0"],
                        -m["metrics"]["acc@0.5"],
                        -m["metrics"]["r2"],
                    ),
                )
                top_k = config.get("top_k_models", 5)
                best_models = best_models[:top_k]

                # 이 epoch 이 현재 best 모델이면 state_dict 를 따로 저장
                if best_models[0]["epoch"] == epoch:
                    best_state_dict = copy.deepcopy(model.state_dict())

                # top_k 안에 들어가는 epoch 은 checkpoint 로 저장 (fusion 용)
                if any(m["epoch"] == epoch for m in best_models):
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"{modality}_{backbone_name.replace('/', '_')}_epoch{epoch}.pt",
                    )
                    save_checkpoint(
                        ckpt_path,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        extra={
                            "modality": modality,
                            "backbone": backbone_name,
                        },
                    )
                
                # Clean up old checkpoints that are no longer in top_k
                top_k_epochs = {m["epoch"] for m in best_models}
                for existing_file in os.listdir(ckpt_dir):
                    if existing_file.endswith(".pt") and existing_file.startswith(f"{modality}_{backbone_name.replace('/', '_')}_epoch"):
                        # Extract epoch number from filename
                        try:
                            epoch_num = int(existing_file.split("epoch")[1].split(".")[0])
                            if epoch_num not in top_k_epochs:
                                old_ckpt_path = os.path.join(ckpt_dir, existing_file)
                                os.remove(old_ckpt_path)
                                logger.debug(f"Removed old checkpoint: {existing_file}")
                        except (ValueError, IndexError):
                            # Skip files that don't match expected pattern
                            continue

        # history 저장
        hist_path = os.path.join(
            ckpt_dir,
            f"{modality}_{backbone_name.replace('/', '_')}_history.json",
        )
        with open(hist_path, "w") as f:
            json.dump(best_metric_history, f, indent=2)

        # 학습이 끝난 후, best_state_dict 기준으로 test set 평가
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(
                f"[{backbone_name}] Loaded best validation model for test evaluation."
            )

        model.eval()
        test_preds_list: List[np.ndarray] = []
        test_targets_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(
                test_loader,
                desc=f"{backbone_name} [test, bs={batch_size}]",
            ):
                if modality == "nail":
                    imgs, hb, _ = batch
                else:
                    imgs, hb, _, _ = batch
                imgs = imgs.to(device)
                hb = hb.to(device, dtype=torch.float32)
                preds = model(imgs)
                test_preds_list.append(tensor_to_numpy(preds))
                test_targets_list.append(tensor_to_numpy(hb))

        y_true_test = np.concatenate(test_targets_list)
        y_pred_test = np.concatenate(test_preds_list)
        test_metrics = compute_regression_metrics(y_true_test, y_pred_test)

        logger.info(f"[{backbone_name}] Test metrics: {test_metrics}")

        test_metrics_path = os.path.join(
            ckpt_dir,
            f"{modality}_{backbone_name.replace('/', '_')}_test_metrics.json",
        )
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)


    logger.info(f"All backbones finished for modality={modality}. Checkpoints in {ckpt_dir_root}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--modality", type=str, choices=["nail", "conj"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train_single(config, args.modality)


if __name__ == "__main__":
    main()


