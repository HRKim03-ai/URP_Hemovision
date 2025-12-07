import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.fusion_dataset import FusionDataset, FusionSample, load_fusion_metadata
from datasets.transforms import build_transforms
from models.fusion_models import Phase1FusionModel, Phase2MultiLevelFusionModel
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.metrics import compute_regression_metrics, tensor_to_numpy
from utils.seed import set_seed


def build_fusion_eval_loader(
    samples: List[FusionSample],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    nail_tf = build_transforms("test", image_size=image_size, modality="nail")
    conj_tf = build_transforms("test", image_size=image_size, modality="conj")
    ds = FusionDataset(samples, nail_transform=nail_tf, conj_transform=conj_tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def evaluate_fusion(config: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))
    logger = setup_logger("eval_fusion", config.get("log_file"))

    samples: List[FusionSample] = load_fusion_metadata(config["fusion_metadata_csv"])

    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    loader = build_fusion_eval_loader(samples, image_size, batch_size, num_workers)

    fusion_type = config.get("fusion_type", "phase1")
    nail_name = config["nail_backbone"]
    conj_name = config["conj_backbone"]
    use_demographics = config.get("use_demographics", True)
    demo_dim = 2 if use_demographics else 0

    if fusion_type == "phase1":
        model = Phase1FusionModel(
            nail_backbone_name=nail_name,
            conj_backbone_name=conj_name,
            demo_dim=demo_dim,
        ).to(device)
    else:
        model = Phase2MultiLevelFusionModel(
            nail_backbone_name=nail_name,
            conj_backbone_name=conj_name,
            demo_dim=demo_dim,
        ).to(device)

    load_checkpoint(config["checkpoint_path"], model=model)
    model.eval()

    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            nail_img, conj_img, hb, _, demo = batch
            nail_img = nail_img.to(device)
            conj_img = conj_img.to(device)
            hb = hb.to(device, dtype=torch.float32)
            demo = demo.to(device, dtype=torch.float32) if use_demographics else None
            preds = model(nail_img, conj_img, demo)
            preds_list.append(tensor_to_numpy(preds))
            targets_list.append(tensor_to_numpy(hb))

    y_true = np.concatenate(targets_list)
    y_pred = np.concatenate(preds_list)
    metrics = compute_regression_metrics(y_true, y_pred)
    logger.info(f"Fusion evaluation metrics ({fusion_type}): {metrics}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    evaluate_fusion(config)


if __name__ == "__main__":
    main()


