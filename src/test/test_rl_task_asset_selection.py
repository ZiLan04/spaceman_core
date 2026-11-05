import pytest

from configs.schema import SatelliteManipulatorSpec, SatelliteSpec
from tasks.RL_task import resolve_task_asset_specs


def test_resolve_task_asset_specs_accepts_valid_pair():
    manipulator, target = resolve_task_asset_specs("franka_merge", "starlink")

    assert isinstance(manipulator, SatelliteManipulatorSpec)
    assert type(target) is SatelliteSpec


def test_resolve_task_asset_specs_rejects_reversed_pair():
    with pytest.raises(TypeError, match="not a satellite_manipulator"):
        resolve_task_asset_specs("starlink", "franka_merge")
