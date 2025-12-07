from typing import List, Sequence

import torch
from torch import nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiLevelProjector(nn.Module):
    """
    Project multi-stage CNN feature maps to a common dimension with GAP + Linear.
    """

    def __init__(self, in_channels_list: Sequence[int], proj_dim: int = 256):
        super().__init__()
        self.proj_layers = nn.ModuleList(
            [nn.Linear(c, proj_dim) for c in in_channels_list]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        proj_feats = []
        for feat, proj in zip(features, self.proj_layers):
            x = self.pool(feat).flatten(1)
            x = proj(x)
            proj_feats.append(x)
        return torch.cat(proj_feats, dim=1)


