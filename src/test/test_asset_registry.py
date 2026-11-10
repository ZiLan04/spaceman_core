"""Fast configuration tests that do not initialize Genesis."""

from pathlib import Path
import xml.etree.ElementTree as ET

from configs.registry import (
    AssetRegistry,
    get_legacy_camera,
    get_legacy_params,
    get_legacy_pid,
    get_spec,
    registry,
)
from configs.schema import ManipulatorSpec, SatelliteManipulatorSpec, SatelliteSpec


FLEET_SATELLITES = {'artificial2', 'catch01', 'design', 'satellite_11', 'satellite_model1', 'legacy10', 'solidworks_satellite', 'satellite_8', 'gps', 'artificial1', 'satellite_17', 'satellite_16', 'satellite_9', 'navstar', 'satellite_12', 'satellite_15', 'sputnik', 'satellite_5', 'satellite_14', 'satellite_2', 'satellite_sb1', 'uydu', 'satellite_6', 'satellite_4', 'scifi', 'satellite_13', 'station'}
FLEET_MANIPULATORS = {"airbus_vispa", "starlink_sldasm9"}
FLEET_COMBINATIONS = {
    f"{satellite}__{manipulator}"
    for satellite in FLEET_SATELLITES
    for manipulator in FLEET_MANIPULATORS
}


def test_builtin_assets_are_registered():
    assert set(registry.list_ids()) == (
        FLEET_SATELLITES
        | FLEET_MANIPULATORS
        | FLEET_COMBINATIONS
        | {"franka_merge", "starlink"}
    )
    assert set(registry.list_ids(kind="satellite")) == FLEET_SATELLITES | {"starlink"}
    assert set(registry.list_ids(kind="manipulator")) == FLEET_MANIPULATORS
    assert set(registry.list_ids(kind="satellite_manipulator")) == (
        FLEET_COMBINATIONS | {"franka_merge"}
    )
    assert type(get_spec("starlink")) is SatelliteSpec
    assert isinstance(get_spec("franka_merge"), SatelliteManipulatorSpec)
    assert isinstance(get_spec("airbus_vispa"), ManipulatorSpec)


def test_fleet_configs_reference_real_urdf_names_and_meshes():
    for asset_id in FLEET_SATELLITES | FLEET_MANIPULATORS | FLEET_COMBINATIONS:
        spec = get_spec(asset_id)
        assert spec.urdf.is_file()
        root = ET.parse(spec.urdf).getroot()
        links = {element.attrib["name"] for element in root.findall("link")}
        joints = {element.attrib["name"] for element in root.findall("joint")}

        assert spec.base_link in links
        assert all(site.link in links for site in spec.grasp_sites)
        if isinstance(spec, SatelliteSpec):
            assert spec.mount_site is not None
            assert spec.mount_site.link in links
        if isinstance(spec, (ManipulatorSpec, SatelliteManipulatorSpec)):
            assert spec.manipulator_base_link in links
            assert spec.end_effector_link in links
            assert set(spec.arm_joint_names) <= joints
            assert set(spec.gripper.joint_names) <= joints
            assert set(spec.gripper.contact_links) <= links

        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename", "")
            if not filename.startswith("package://"):
                assert (spec.urdf.parent / filename).resolve().is_file()


def test_starlink_spec_preserves_existing_configuration():
    spec = get_spec("starlink")
    params = get_legacy_params("starlink")

    assert spec.urdf.is_absolute()
    assert spec.urdf == Path(params["path"])
    assert params["base"] == "base_star_body_link"
    assert params["grasp_link"] == "base_star_body_link"
    assert params["grasp_offset_local"] == [-0.33519137, 0.06895784, 0.22731882]
    assert spec.morph.links_to_keep == ("base_star_cylinder_link",)
    assert spec.morph.coacd["resolution"] == 10000
    assert spec.mount_site.link == "base_star_body_link"
    assert spec.mount_site.position == (-0.35, 0.75, 0.2)
    assert get_legacy_pid("starlink")["enable_pid"] is False


def test_franka_merge_spec_preserves_existing_configuration():
    spec = get_spec("franka_merge")
    params = get_legacy_params("franka_merge")
    camera = get_legacy_camera("franka_merge")

    assert isinstance(spec, SatelliteManipulatorSpec)
    assert spec.base_link == "starlink_base_star_link"
    assert spec.manipulator_base_link.endswith("space_manipulator_base_link")
    assert len(spec.arm_joint_names) == 6
    assert len(spec.gripper.joint_names) == 2
    assert spec.control.kp[0] == 4500
    assert spec.ik.smooth_factor == 0.3
    assert spec.tcp.link == spec.end_effector_link
    assert spec.tcp.position == (0.002085, -0.2, 0.0004827)
    assert spec.tcp.quaternion == (1.0, 0.0, 0.0, 0.0)
    assert spec.reference_pose.link == spec.end_effector_link
    assert spec.reference_pose.joint == spec.arm_joint_names[-1]
    assert spec.reference_pose.position == (-1.65419686, -0.06428438, 0.92678845)
    assert len(spec.gripper.contact_links) == 2
    assert spec.camera.wrist.mount_link == spec.end_effector_link
    assert spec.camera.base.mount_link == "starlink_base_star_link"
    assert spec.camera.wrist.camera["fov"] == 60
    assert spec.camera.base.camera["fov"] == 82
    assert spec.urdf.exists()
    assert params["motor"] == 6
    assert params["finger"] == 8
    assert len(params["joints"]) == 8
    assert params["config"]["initial_dofs"] == [
        0.14, -0.55, -1.36, 0.81, 1.31, -0.52, 0.0, 0.0
    ]
    assert get_legacy_pid("franka_merge")["dt"] == 0.003
    assert camera["base_camera"]["mount_link"] == "starlink_base_star_link"
    assert camera["end_effector_link"] == spec.end_effector_link
    assert camera["wrist_camera"] is False
    assert camera["image_effects"] == {}
    assert camera["rl_obs"] == {}


def test_legacy_dicts_are_independent_copies():
    first = get_legacy_pid("franka_merge")
    first["P"][0] = -1
    assert get_legacy_pid("franka_merge")["P"][0] == 800


def test_registry_discovers_new_yaml_without_python_registration(tmp_path):
    urdf = tmp_path / "new_satellite.urdf"
    urdf.write_text("<robot name='new_satellite'/>", encoding="utf-8")
    config = tmp_path / "new_satellite.yaml"
    config.write_text(
        """
id: new_satellite
kind: satellite
urdf: new_satellite.urdf
links:
  base: base_link
morph:
  spawn:
    position: [0, 0, 0]
material: {}
grasp_sites:
  - id: default
    link: base_link
    position: [0, 0, 0]
mount_site:
  link: base_link
  position: [0, 0, 0]
pid: {}
cameras: {}
""".strip(),
        encoding="utf-8",
    )

    isolated_registry = AssetRegistry()
    isolated_registry.load_directory(tmp_path)

    assert isolated_registry.list_ids() == ["new_satellite"]
    assert isolated_registry.get("new_satellite").urdf == urdf.resolve()
