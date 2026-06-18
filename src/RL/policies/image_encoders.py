from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _ensure_hw(image_shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(image_shape) != 3:
        raise ValueError(
            f"Expected image shape with 3 dims (C, H, W), got {tuple(image_shape)}"
        )
    channels, height, width = image_shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {tuple(image_shape)}")
    return int(channels), int(height), int(width)


class SimpleCNNEncoder(nn.Module):
    """
    Lightweight CNN image encoder for online PPO training.
    """

    def __init__(self, image_shape: Sequence[int], feature_dim: int = 256):
        super().__init__()
        channels, height, width = _ensure_hw(image_shape)
        self.image_shape = (channels, height, width)
        self.feature_dim = int(feature_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            flattened_dim = self.backbone(sample).shape[1]

        self.projection = nn.Sequential(
            nn.Linear(flattened_dim, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.projection(x)


class SimpleViTEncoder(nn.Module):
    """
    Small ViT-style encoder implemented only with PyTorch modules.

    This is intentionally compact so it can be used without external
    dependencies such as timm.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        feature_dim: int = 256,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels, height, width = _ensure_hw(image_shape)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "SimpleViTEncoder requires height and width to be divisible by "
                f"patch_size={patch_size}, got {(height, width)}"
            )

        self.image_shape = (channels, height, width)
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_patches = grid_h * grid_w

        self.patch_embed = nn.Conv2d(
            channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.projection = nn.Linear(self.embed_dim, self.feature_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.projection(x)
