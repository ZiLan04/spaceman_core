import sys
from pathlib import Path

import torch


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from sensors.image_effects.solar_glare import SolarGlare


def _camera_pose(batch_size=1):
    position = torch.zeros(batch_size, 3)
    lookat = torch.tensor([[0.0, 0.0, 1.0]]).expand(batch_size, -1)
    up = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size, -1)
    return position, lookat, up


def test_disabled_glare_returns_input_unchanged():
    image = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    position, lookat, up = _camera_pose(batch_size=2)
    effect = SolarGlare({"enabled": False})

    output = effect.apply(image, position, lookat, up, fov=60)

    assert output is image


def test_camera_forward_glare_brightens_image_center():
    image = torch.zeros(1, 3, 33, 33, dtype=torch.uint8)
    position, lookat, up = _camera_pose()
    effect = SolarGlare(
        {
            "enabled": True,
            "direction_source": "camera_forward",
            "strength": 1.0,
            "probability": 1.0,
        }
    )

    output = effect.apply(image, position, lookat, up, fov=60)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert output[0, :, 16, 16].float().mean() > output[0, :, 0, 0].float().mean()


def test_scene_light_behind_camera_does_not_change_image():
    image = torch.randint(0, 256, (2, 24, 24, 3), dtype=torch.uint8)
    position, lookat, up = _camera_pose(batch_size=2)
    effect = SolarGlare(
        {
            "enabled": True,
            "direction_source": "scene_light",
            "probability": 1.0,
        }
    )

    # Light travels toward +Z, so the sun is at -Z, behind this camera.
    output = effect.apply(
        image,
        position,
        lookat,
        up,
        fov=60,
        scene_light_direction=(0.0, 0.0, 1.0),
    )

    assert torch.equal(output, image)


def test_fixed_world_supports_normalized_float_images():
    image = torch.zeros(2, 3, 20, 20, dtype=torch.float32)
    position, lookat, up = _camera_pose(batch_size=2)
    effect = SolarGlare(
        {
            "enabled": True,
            "direction_source": "fixed_world",
            "direction": (0.0, 0.0, -1.0),
            "strength": (0.5, 0.5),
            "probability": 1.0,
        }
    )

    output = effect.apply(image, position, lookat, up, fov=60)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert torch.all((0.0 <= output) & (output <= 1.0))
    assert torch.all(output.amax(dim=(1, 2, 3)) > 0.0)
