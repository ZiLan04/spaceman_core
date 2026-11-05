"""YAML loader for complete capture-scene specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from configs.schema import SceneCameraChannelSpec, SceneCameraSpec, SceneSpec


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} must not contain empty IDs")
    return result


def load_scene_spec(path: str | Path) -> SceneSpec:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), str(config_path))

    known = {"id", "selection", "target_randomization", "cameras"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown scene fields in {config_path}: {unknown}")

    selection = _mapping(raw.get("selection", {}), "selection")
    randomization = _mapping(
        raw.get("target_randomization", {}), "target_randomization"
    )
    cameras = _mapping(raw.get("cameras", {}), "cameras")
    noise_range = randomization.get("position_range", (0.03, 0.03, 0.02))
    if not isinstance(noise_range, (list, tuple)):
        raise ValueError("target_randomization.position_range must be a list")

    return SceneSpec(
        scene_id=str(raw.get("id", "")).strip(),
        selection_mode=str(selection.get("mode", "fixed")).strip(),
        satellite_manipulators=_ids(
            selection.get("satellite_manipulators", ()),
            "selection.satellite_manipulators",
        ),
        target_satellites=_ids(
            selection.get("target_satellites", ()),
            "selection.target_satellites",
        ),
        seed=selection.get("seed"),
        target_position_noise=bool(randomization.get("position_noise", False)),
        target_position_noise_range=tuple(float(value) for value in noise_range),
        cameras=SceneCameraSpec(
            wrist=_load_camera_runtime(cameras.get("wrist", {}), "cameras.wrist"),
            base=_load_camera_runtime(cameras.get("base", {}), "cameras.base"),
        ),
    )


def _load_camera_runtime(value: Any, name: str) -> SceneCameraChannelSpec:
    raw = _mapping(value, name)
    known = {
        "enabled", "recording_enabled", "episodes_per_video",
        "image_effects", "rl_observation",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown fields in {name}: {unknown}")
    return SceneCameraChannelSpec(
        enabled=bool(raw.get("enabled", False)),
        recording_enabled=bool(raw.get("recording_enabled", False)),
        episodes_per_video=int(raw.get("episodes_per_video", 1)),
        image_effects=dict(_mapping(raw.get("image_effects", {}), f"{name}.image_effects")),
        rl_observation=dict(
            _mapping(raw.get("rl_observation", {}), f"{name}.rl_observation")
        ),
    )
