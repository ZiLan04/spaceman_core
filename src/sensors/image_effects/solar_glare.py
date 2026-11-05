"""Batched, torch-native synthetic solar glare for camera observations."""

from copy import deepcopy
import math

import torch
import torch.nn.functional as F


DEFAULT_SOLAR_GLARE_CONFIG = {
    "enabled": False,
    "direction_source": "scene_light",
    "light_index": 0,
    "direction": (0.0, 0.0, -1.0),
    "strength": 1.0,
    "probability": 1.0,
    "broad_sigma_ratio": 0.22,
    "small_sigma_ratio": 0.065,
    "very_broad_sigma_ratio": 0.55,
    "veil_sigma_ratio": 0.035,
    "ghost_sigma_ratio": 0.10,
    "broad_weight": 0.48,
    "small_weight": 0.32,
    "very_broad_weight": 0.12,
    "veil_weight": 0.09,
    "ghost_weight": 0.08,
    "color": (1.0, 0.97, 0.89),
}


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)


class SolarGlare:
    """Apply a projected solar bloom, veil, and lens ghost to batched RGB."""

    def __init__(self, config=None, device=None):
        self.config = deepcopy(DEFAULT_SOLAR_GLARE_CONFIG)
        if config:
            self.config.update(config)
        self.enabled = bool(self.config["enabled"])
        self.device = device
        source = self.config["direction_source"]
        if source not in {"scene_light", "camera_forward", "fixed_world"}:
            raise ValueError(f"Unknown solar glare direction_source: {source!r}")

    @staticmethod
    def _as_batched_vec3(value, batch_size, device, dtype, name):
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.shape == (3,):
            value = value.unsqueeze(0).expand(batch_size, -1)
        if value.shape != (batch_size, 3):
            raise ValueError(
                f"{name} must have shape (3,) or ({batch_size}, 3), got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _gaussian(height, width, center_x, center_y, sigma, device, dtype):
        yy = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
        distance_sq = (xx - center_x[:, None, None]).square()
        distance_sq = distance_sq + (yy - center_y[:, None, None]).square()
        return torch.exp(-distance_sq / (2.0 * sigma * sigma))

    @staticmethod
    def _gaussian_blur(image, sigma):
        if sigma <= 0.0:
            return image
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-coords.square() / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        channels = image.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        blurred = F.conv2d(image, horizontal, padding=(0, radius), groups=channels)
        return F.conv2d(blurred, vertical, padding=(radius, 0), groups=channels)

    def _sample_strength(self, batch_size, device, dtype):
        strength = self.config["strength"]
        if isinstance(strength, (tuple, list)):
            if len(strength) != 2:
                raise ValueError("solar glare strength range must contain two values")
            low, high = map(float, strength)
            return low + (high - low) * torch.rand(batch_size, device=device, dtype=dtype)
        return torch.full((batch_size,), float(strength), device=device, dtype=dtype)

    def _sun_direction(self, camera_forward, scene_light_direction):
        source = self.config["direction_source"]
        if source == "camera_forward":
            return camera_forward
        if source == "fixed_world":
            travel = self.config["direction"]
        else:
            if scene_light_direction is None:
                raise ValueError(
                    "scene_light_direction is required when direction_source='scene_light'"
                )
            travel = scene_light_direction
        travel = self._as_batched_vec3(
            travel,
            camera_forward.shape[0],
            camera_forward.device,
            camera_forward.dtype,
            "light travel direction",
        )
        return -_normalize(travel)

    def apply(
        self,
        rgb,
        camera_pos,
        camera_lookat,
        camera_up,
        fov,
        scene_light_direction=None,
    ):
        if not self.enabled:
            return rgb

        original = torch.as_tensor(rgb)
        if original.ndim != 4:
            raise ValueError(f"solar glare expects batched RGB, got {tuple(original.shape)}")
        channels_first = original.shape[1] == 3
        if not channels_first and original.shape[-1] != 3:
            raise ValueError(f"solar glare cannot determine RGB layout: {tuple(original.shape)}")

        image = original if channels_first else original.permute(0, 3, 1, 2)
        input_dtype = image.dtype
        image = image.to(device=self.device or image.device, dtype=torch.float32)
        if input_dtype == torch.uint8:
            image = image / 255.0

        batch_size, _, height, width = image.shape
        camera_pos = self._as_batched_vec3(
            camera_pos, batch_size, image.device, image.dtype, "camera_pos"
        )
        camera_lookat = self._as_batched_vec3(
            camera_lookat, batch_size, image.device, image.dtype, "camera_lookat"
        )
        camera_up = self._as_batched_vec3(
            camera_up, batch_size, image.device, image.dtype, "camera_up"
        )

        forward = _normalize(camera_lookat - camera_pos)
        right = _normalize(torch.linalg.cross(forward, _normalize(camera_up), dim=-1))
        up = _normalize(torch.linalg.cross(right, forward, dim=-1))
        sun = self._sun_direction(forward, scene_light_direction)

        camera_x = (sun * right).sum(dim=-1)
        camera_y = (sun * up).sum(dim=-1)
        camera_z = (sun * forward).sum(dim=-1)
        safe_z = camera_z.clamp_min(1e-8)
        focal_y = (height / 2.0) / math.tan(math.radians(float(fov)) / 2.0)
        focal_x = focal_y * (width / height)
        sun_x = (width - 1) / 2.0 + focal_x * camera_x / safe_z
        sun_y = (height - 1) / 2.0 - focal_y * camera_y / safe_z

        diagonal = math.sqrt(height * height + width * width)
        broad = self._gaussian(
            height, width, sun_x, sun_y,
            float(self.config["broad_sigma_ratio"]) * diagonal,
            image.device, image.dtype,
        )
        small = self._gaussian(
            height, width, sun_x, sun_y,
            float(self.config["small_sigma_ratio"]) * diagonal,
            image.device, image.dtype,
        )
        very_broad = self._gaussian(
            height, width, sun_x, sun_y,
            float(self.config["very_broad_sigma_ratio"]) * diagonal,
            image.device, image.dtype,
        )
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        ghost = self._gaussian(
            height, width, 2.0 * center_x - sun_x, 2.0 * center_y - sun_y,
            float(self.config["ghost_sigma_ratio"]) * diagonal,
            image.device, image.dtype,
        )

        glare = (
            float(self.config["broad_weight"]) * broad
            + float(self.config["small_weight"]) * small
            + float(self.config["very_broad_weight"]) * very_broad
            + float(self.config["ghost_weight"]) * ghost
        )
        strength = self._sample_strength(batch_size, image.device, image.dtype)
        probability = min(1.0, max(0.0, float(self.config["probability"])))
        selected = torch.rand(batch_size, device=image.device) < probability
        active = selected & (camera_z > 0.0)
        scale = strength * active.to(image.dtype)

        veil_weight = float(self.config["veil_weight"])
        blurred = self._gaussian_blur(
            image, float(self.config["veil_sigma_ratio"]) * width
        )
        veil = (veil_weight * scale).clamp(0.0, 1.0)[:, None, None, None]
        output = (1.0 - veil) * image + veil * blurred
        color = torch.as_tensor(
            self.config["color"], device=image.device, dtype=image.dtype
        ).view(1, 3, 1, 1)
        output = output + scale[:, None, None, None] * glare[:, None] * color
        output = output.clamp(0.0, 1.0)

        if input_dtype == torch.uint8:
            output = (output * 255.0).to(torch.uint8)
        else:
            output = output.to(input_dtype)
        if not channels_first:
            output = output.permute(0, 2, 3, 1).contiguous()
        return output
