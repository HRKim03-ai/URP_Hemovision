from typing import Any

import timm
import torch
from torch import nn



def _normalize_model_name(model_name: str) -> str:
    """
    timm.create_model 에 직접 넘길 이름을 정리.

    현재는 특별한 정규화는 하지 않고, 원본을 그대로 반환한다.
    (허깅페이스 timm 콜렉션 이름 'timm/xxx' 는 별도로 hf-hub: 프리픽스로 처리)
    """
    return model_name


def create_backbone(
    model_name: str,
    pretrained: bool = True,
    features_only: bool = False,
) -> nn.Module:
    """
    Wrapper for timm backbones.

    For features_only=False: returns a model with global pooling and no classifier (num_classes=0).
    For features_only=True: returns a timm FeatureListNet that outputs multi-level features.
    """
    model_name_norm = _normalize_model_name(model_name)

    try:
        if features_only:
            # 먼저 모델을 생성해서 feature_info 확인
            temp_model = timm.create_model(
                model_name_norm,
                pretrained=pretrained,
                features_only=True,
            )
            num_stages = len(temp_model.feature_info)
            # 마지막 4개 stage 사용 (모델이 4개 이하면 모두 사용)
            if num_stages >= 4:
                out_indices = tuple(range(num_stages - 4, num_stages))
            else:
                out_indices = tuple(range(num_stages))
            # 올바른 out_indices로 다시 생성
            model = timm.create_model(
                model_name_norm,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )
        else:
            model = timm.create_model(
                model_name_norm,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
    except RuntimeError as e:
        # 허깅페이스 timm 콜렉션 이름인 경우, hf-hub: 프리픽스를 붙여 다시 시도
        if model_name.startswith("timm/"):
            hf_name = f"hf-hub:{model_name}"
            if features_only:
                # 먼저 모델을 생성해서 feature_info 확인
                temp_model = timm.create_model(
                    hf_name,
                    pretrained=pretrained,
                    features_only=True,
                )
                num_stages = len(temp_model.feature_info)
                # 마지막 4개 stage 사용 (모델이 4개 이하면 모두 사용)
                if num_stages >= 4:
                    out_indices = tuple(range(num_stages - 4, num_stages))
                else:
                    out_indices = tuple(range(num_stages))
                # 올바른 out_indices로 다시 생성
                model = timm.create_model(
                    hf_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=out_indices,
                )
            else:
                model = timm.create_model(
                    hf_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
        else:
            raise e
    return model


def get_backbone_output_dim(backbone: nn.Module) -> int:
    """
    Try to infer backbone output feature dimension.
    """
    if hasattr(backbone, "num_features"):
        return int(backbone.num_features)
    if hasattr(backbone, "feature_info"):
        return int(backbone.feature_info[-1]["num_chs"])
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        y: Any = backbone(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        if y.ndim == 4:
            y = torch.nn.functional.adaptive_avg_pool2d(y, 1).view(1, -1)
        return int(y.shape[1])


def set_backbone_trainable(backbone: nn.Module, train_last_n_stages: int = 1) -> None:
    """
    Freeze all backbone params, then unfreeze the last N stages/blocks.

    NOTE: This is heuristic and may need adaptation per-architecture.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    children = list(backbone.children())
    for child in children[-train_last_n_stages:]:
        for p in child.parameters():
            p.requires_grad = True


