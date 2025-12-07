import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

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
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def build_eval_loader(
    modality: str,
    config: Dict[str, Any],
    split: str = "test",
) -> DataLoader:
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    if modality == "nail":
        samples: List[NailSample] = load_nail_metadata(config["metadata_csv"])
        train_s, val_s, test_s = split_nail_by_patient(samples)
        if split == "val":
            ds = NailDataset(val_s, transform=build_transforms("val", image_size, "nail"))
        else:
            ds = NailDataset(
                test_s, transform=build_transforms("test", image_size, "nail")
            )
    else:
        samples_c: List[ConjSample] = load_conj_metadata(
            config["metadata_csv_folder1"], config["metadata_csv_folder2"]
        )
        train_s, val_s, test_s = split_conj_by_patient(samples_c)
        if split == "val":
            ds = ConjDataset(
                val_s, transform=build_transforms("val", image_size, "conj")
            )
        else:
            ds = ConjDataset(
                test_s, transform=build_transforms("test", image_size, "conj")
            )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def evaluate_single(config: Dict[str, Any], modality: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))
    logger = setup_logger("eval_single", config.get("log_file"))

    loader = build_eval_loader(modality, config, split=config.get("split", "test"))

    backbone_name = config["backbone_name"]
    backbone = create_backbone(backbone_name, pretrained=False, features_only=False)
    feat_dim = get_backbone_output_dim(backbone)
    head = RegressionHead(feat_dim)
    model = nn.Sequential(backbone, head).to(device)

    load_checkpoint(config["checkpoint_path"], model=model)
    model.eval()

    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if modality == "nail":
                imgs, hb, _ = batch
            else:
                imgs, hb, _, _ = batch
            imgs = imgs.to(device)
            hb = hb.to(device, dtype=torch.float32)
            preds = model(imgs)
            preds_list.append(tensor_to_numpy(preds))
            targets_list.append(tensor_to_numpy(hb))

    y_true = np.concatenate(targets_list)
    y_pred = np.concatenate(preds_list)
    metrics = compute_regression_metrics(y_true, y_pred)
    logger.info(f"Evaluation metrics ({modality}): {metrics}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--modality", type=str, choices=["nail", "conj"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    evaluate_single(config, args.modality)


if __name__ == "__main__":
    main()


