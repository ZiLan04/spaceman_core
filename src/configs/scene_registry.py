"""Registry and resolver for YAML-defined capture scenes."""

from __future__ import annotations

from pathlib import Path

from configs.registry import AssetRegistry, registry as asset_registry
from configs.scene_loaders import load_scene_spec
from configs.schema import SatelliteManipulatorSpec, SatelliteSpec, SceneSpec

_SCENE_DIR = Path(__file__).resolve().parent / "scenes"


class SceneRegistry:
    def __init__(self, assets: AssetRegistry = asset_registry) -> None:
        self._assets = assets
        self._specs: dict[str, SceneSpec] = {}

    def register(self, spec: SceneSpec) -> None:
        if spec.scene_id in self._specs:
            raise ValueError(f"Scene '{spec.scene_id}' is already registered")
        for asset_id in spec.satellite_manipulators:
            if not isinstance(self._assets.get(asset_id), SatelliteManipulatorSpec):
                raise TypeError(f"Scene asset {asset_id!r} is not a satellite_manipulator")
        for asset_id in spec.target_satellites:
            if type(self._assets.get(asset_id)) is not SatelliteSpec:
                raise TypeError(f"Scene asset {asset_id!r} is not a satellite")
        self._specs[spec.scene_id] = spec

    def load(self, path: str | Path) -> SceneSpec:
        spec = load_scene_spec(path)
        self.register(spec)
        return spec

    def load_directory(self, directory: str | Path) -> None:
        for path in sorted(Path(directory).glob("*.yaml")):
            self.load(path)

    def get(self, scene_id: str) -> SceneSpec:
        try:
            return self._specs[scene_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown scene {scene_id!r}. Available: {self.list_ids()}"
            ) from exc

    def list_ids(self) -> list[str]:
        return sorted(self._specs)


registry = SceneRegistry()
registry.load_directory(_SCENE_DIR)


def get_scene_spec(scene_id: str) -> SceneSpec:
    return registry.get(scene_id)


def resolve_scene_assets(
    scene_id: str,
    *,
    seed: int | None = None,
    satellite_manipulator_id: str | None = None,
    target_satellite_id: str | None = None,
) -> tuple[SceneSpec, str, str]:
    spec = get_scene_spec(scene_id)
    selected_manipulator, selected_target = spec.select_assets(seed)
    return (
        spec,
        satellite_manipulator_id or selected_manipulator,
        target_satellite_id or selected_target,
    )
