from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

try:
    from gymnasium import spaces
except Exception:
    spaces = None

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:
    BaseFeaturesExtractor = nn.Module

from RL.policies.image_encoders import SimpleCNNEncoder, SimpleViTEncoder


class MultiModalExtractor(BaseFeaturesExtractor):
    """
    SB3 feature extractor for dict observations with keys:
    - ``state``: low-dimensional proprioceptive/task state
    - ``wrist_image``: wrist camera RGB tensor in (C, H, W)
    - ``base_image``: base camera RGB tensor in (C, H, W)

    For backward compatibility, a single ``image`` key is also supported.
    """

    def __init__(
        self,
        observation_space,
        state_key: str = "state",
        wrist_image_key: str = "wrist_image",
        base_image_key: str = "base_image",
        image_key: str = "image",
        image_encoder_type: str = "cnn",
        state_feature_dim: int = 128,
        image_feature_dim: int = 256,
        fused_feature_dim: Optional[int] = None,
        state_hidden_dims: Sequence[int] = (128, 128),
        normalize_images: bool = True,
        vit_patch_size: int = 8,
        vit_embed_dim: int = 128,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
    ):
        self.state_key = state_key
        self.normalize_images = bool(normalize_images)
        self.image_encoder_type = image_encoder_type.strip().lower()

        self.use_dual_images = False
        self.legacy_image_key = None
        if wrist_image_key in observation_space.spaces and base_image_key in observation_space.spaces:
            self.use_dual_images = True
            self.wrist_image_key = wrist_image_key
            self.base_image_key = base_image_key
        elif image_key in observation_space.spaces:
            self.legacy_image_key = image_key
        else:
            raise KeyError(
                "Missing visual observation keys. Expected either "
                f"('{wrist_image_key}', '{base_image_key}') or '{image_key}'."
            )

        total_image_feature_dim = (
            int(image_feature_dim) * 2 if self.use_dual_images else int(image_feature_dim)
        )

        features_dim = 1
        if fused_feature_dim is not None:
            features_dim = int(fused_feature_dim)
        else:
            features_dim = int(state_feature_dim) + total_image_feature_dim

        super().__init__(observation_space, features_dim=features_dim)

        if spaces is not None and not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "MultiModalExtractor expects a gymnasium.spaces.Dict observation space."
            )

        if state_key not in observation_space.spaces:
            raise KeyError(f"Missing '{state_key}' in observation space.")

        state_space = observation_space.spaces[state_key]
        primary_image_space = observation_space.spaces[
            wrist_image_key if self.use_dual_images else self.legacy_image_key
        ]

        state_dim = int(torch.tensor(state_space.shape).prod().item())
        image_shape = tuple(int(dim) for dim in primary_image_space.shape)

        self.state_encoder = self._build_state_encoder(
            state_dim=state_dim,
            hidden_dims=tuple(int(dim) for dim in state_hidden_dims),
            output_dim=int(state_feature_dim),
        )

        if self.use_dual_images:
            wrist_image_space = observation_space.spaces[wrist_image_key]
            base_image_space = observation_space.spaces[base_image_key]
            wrist_image_shape = tuple(int(dim) for dim in wrist_image_space.shape)
            base_image_shape = tuple(int(dim) for dim in base_image_space.shape)
            self.wrist_image_encoder = self._build_image_encoder(
                image_shape=wrist_image_shape,
                image_feature_dim=int(image_feature_dim),
                vit_patch_size=int(vit_patch_size),
                vit_embed_dim=int(vit_embed_dim),
                vit_depth=int(vit_depth),
                vit_num_heads=int(vit_num_heads),
                vit_mlp_ratio=float(vit_mlp_ratio),
            )
            self.base_image_encoder = self._build_image_encoder(
                image_shape=base_image_shape,
                image_feature_dim=int(image_feature_dim),
                vit_patch_size=int(vit_patch_size),
                vit_embed_dim=int(vit_embed_dim),
                vit_depth=int(vit_depth),
                vit_num_heads=int(vit_num_heads),
                vit_mlp_ratio=float(vit_mlp_ratio),
            )
        else:
            self.image_encoder = self._build_image_encoder(
                image_shape=image_shape,
                image_feature_dim=int(image_feature_dim),
                vit_patch_size=int(vit_patch_size),
                vit_embed_dim=int(vit_embed_dim),
                vit_depth=int(vit_depth),
                vit_num_heads=int(vit_num_heads),
                vit_mlp_ratio=float(vit_mlp_ratio),
            )

        concat_dim = int(state_feature_dim) + total_image_feature_dim
        self.fusion = None
        if fused_feature_dim is not None:
            self.fusion = nn.Sequential(
                nn.Linear(concat_dim, int(fused_feature_dim)),
                nn.ReLU(),
            )

    def _build_image_encoder(
        self,
        image_shape,
        image_feature_dim: int,
        vit_patch_size: int,
        vit_embed_dim: int,
        vit_depth: int,
        vit_num_heads: int,
        vit_mlp_ratio: float,
    ):
        if self.image_encoder_type == "cnn":
            return SimpleCNNEncoder(
                image_shape=image_shape,
                feature_dim=int(image_feature_dim),
            )
        if self.image_encoder_type == "vit":
            return SimpleViTEncoder(
                image_shape=image_shape,
                feature_dim=int(image_feature_dim),
                patch_size=int(vit_patch_size),
                embed_dim=int(vit_embed_dim),
                depth=int(vit_depth),
                num_heads=int(vit_num_heads),
                mlp_ratio=float(vit_mlp_ratio),
            )
        raise ValueError(
            f"Unsupported image_encoder_type='{self.image_encoder_type}'. Use 'cnn' or 'vit'."
        )

    @staticmethod
    def _build_state_encoder(
        state_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ) -> nn.Sequential:
        layers = []
        input_dim = int(state_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, int(output_dim)))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _prepare_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.dim() > 2:
            state_tensor = state_tensor.view(state_tensor.shape[0], -1)
        return state_tensor.float()

    def _prepare_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.float()
        if self.normalize_images:
            if image_tensor.max().item() > 1.0:
                image_tensor = image_tensor / 255.0
        return image_tensor

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_tensor = self._prepare_state(observations[self.state_key])

        state_features = self.state_encoder(state_tensor)
        if self.use_dual_images:
            wrist_image_tensor = self._prepare_image(observations[self.wrist_image_key])
            base_image_tensor = self._prepare_image(observations[self.base_image_key])
            wrist_image_features = self.wrist_image_encoder(wrist_image_tensor)
            base_image_features = self.base_image_encoder(base_image_tensor)
            fused = torch.cat(
                [state_features, wrist_image_features, base_image_features], dim=1
            )
        else:
            image_tensor = self._prepare_image(observations[self.legacy_image_key])
            image_features = self.image_encoder(image_tensor)
            fused = torch.cat([state_features, image_features], dim=1)

        if self.fusion is not None:
            fused = self.fusion(fused)
        return fused
