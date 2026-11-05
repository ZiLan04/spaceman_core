"""Scene configuration tests that do not initialize Genesis."""

from configs.scene_loaders import load_scene_spec
from configs.scene_registry import get_scene_spec, resolve_scene_assets
from configs.registry import get_spec


def test_builtin_fixed_scene_resolves_current_assets():
    spec, manipulator_id, target_id = resolve_scene_assets(
        "starlink_franka_capture"
    )
    assert spec.selection_mode == "fixed"
    assert manipulator_id == "franka_merge"
    assert target_id == "starlink"
    assert spec.target_position_noise is False
    assert spec.target_position_noise_range == (0.03, 0.03, 0.02)
    assert get_scene_spec(spec.scene_id) is spec
    assert spec.cameras.wrist.rl_observation["episodes_per_video"] == 50
    assert spec.cameras.base.rl_observation["episodes_per_video"] == 25
    assert spec.cameras.wrist.image_effects["solar_glare"]["enabled"] is True

    merged = get_spec(manipulator_id).camera.to_legacy_dict(spec.cameras)
    assert merged["end_effector_link"] == get_spec(manipulator_id).end_effector_link
    assert merged["pos_offset"] == [0.05, -0.08, -0.07]
    assert merged["wrist_camera"] is False
    assert merged["rl_obs"]["enabled"] is True
    assert merged["rl_obs"]["episodes_per_video"] == 50
    assert merged["base_camera"]["rl_obs"]["episodes_per_video"] == 25


def test_random_scene_selection_is_reproducible(tmp_path):
    path = tmp_path / "random_scene.yaml"
    path.write_text(
        """
id: random_capture
selection:
  mode: random
  satellite_manipulators: [arm_a, arm_b, arm_c]
  target_satellites: [sat_a, sat_b, sat_c]
  seed: 7
target_randomization:
  position_noise: true
  position_range: [0.1, 0.2, 0.3]
""".strip(),
        encoding="utf-8",
    )
    spec = load_scene_spec(path)
    assert spec.select_assets() == spec.select_assets()
    assert spec.select_assets(seed=19) == spec.select_assets(seed=19)
    assert spec.target_position_noise is True
    assert spec.target_position_noise_range == (0.1, 0.2, 0.3)


def test_explicit_asset_ids_override_scene_selection():
    _, manipulator_id, target_id = resolve_scene_assets(
        "starlink_franka_capture",
        satellite_manipulator_id="franka_merge",
        target_satellite_id="starlink",
    )
    assert manipulator_id == "franka_merge"
    assert target_id == "starlink"
