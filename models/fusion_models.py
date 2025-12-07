from typing import List

import torch
from torch import nn

from .backbone_factory import create_backbone, get_backbone_output_dim
from .heads import FusionHead, MultiLevelProjector


class Phase1FusionModel(nn.Module):
    """
    Phase 1: Late fusion using last-layer pooled features.
    """

    def __init__(
        self,
        nail_backbone_name: str,
        conj_backbone_name: str,
        demo_dim: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.nail_backbone = create_backbone(
            nail_backbone_name, pretrained=pretrained, features_only=False
        )
        self.conj_backbone = create_backbone(
            conj_backbone_name, pretrained=pretrained, features_only=False
        )

        nail_dim = get_backbone_output_dim(self.nail_backbone)
        conj_dim = get_backbone_output_dim(self.conj_backbone)
        fusion_in_dim = nail_dim + conj_dim + demo_dim
        self.fusion_head = FusionHead(fusion_in_dim)

    def forward(self, nail_img: torch.Tensor, conj_img: torch.Tensor, demo: torch.Tensor | None = None):
        f_n = self.nail_backbone(nail_img)
        f_c = self.conj_backbone(conj_img)
        if f_n.ndim > 2:
            f_n = torch.flatten(f_n, 1)
        if f_c.ndim > 2:
            f_c = torch.flatten(f_c, 1)
        if demo is not None and demo.numel() > 0:
            x = torch.cat([f_n, f_c, demo], dim=1)
        else:
            x = torch.cat([f_n, f_c], dim=1)
        out = self.fusion_head(x)
        return out


class Phase2MultiLevelFusionModel(nn.Module):
    """
    Phase 2: Multi-level feature fusion.
    """

    def __init__(
        self,
        nail_backbone_name: str,
        conj_backbone_name: str,
        demo_dim: int = 2,
        proj_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.nail_backbone = create_backbone(
            nail_backbone_name, pretrained=pretrained, features_only=True
        )
        self.conj_backbone = create_backbone(
            conj_backbone_name, pretrained=pretrained, features_only=True
        )

        nail_chs = [fi["num_chs"] for fi in self.nail_backbone.feature_info]
        conj_chs = [fi["num_chs"] for fi in self.conj_backbone.feature_info]

        self.nail_projector = MultiLevelProjector(
            in_channels_list=nail_chs[-4:], proj_dim=proj_dim
        )
        self.conj_projector = MultiLevelProjector(
            in_channels_list=conj_chs[-4:], proj_dim=proj_dim
        )

        fusion_in_dim = 8 * proj_dim + demo_dim
        self.fusion_head = FusionHead(fusion_in_dim)

    def forward(self, nail_img: torch.Tensor, conj_img: torch.Tensor, demo: torch.Tensor | None = None):
        nail_feats: List[torch.Tensor] = self.nail_backbone(nail_img)
        conj_feats: List[torch.Tensor] = self.conj_backbone(conj_img)

        nail_multi = self.nail_projector(nail_feats[-4:])
        conj_multi = self.conj_projector(conj_feats[-4:])

        if demo is not None and demo.numel() > 0:
            x = torch.cat([nail_multi, conj_multi, demo], dim=1)
        else:
            x = torch.cat([nail_multi, conj_multi], dim=1)
        out = self.fusion_head(x)
        return out


