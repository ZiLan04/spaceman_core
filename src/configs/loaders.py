"""YAML loaders for simulation asset specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from configs.schema import (
    CameraSpec,
    CameraChannelSpec,
    GripperSpec,
    GraspSiteSpec,
    IKSpec,
    JointControlSpec,
    MaterialSpec,
    ManipulatorSpec,
    ManipulatorReferencePoseSpec,
    MountSiteSpec,
    MorphSpec,
    PIDSpec,
    PoseSpec,
    SatelliteManipulatorSpec,
    SatelliteSpec,
    TCPSpec,
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML mappings without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_with_extends(
    config_path: Path, loading: tuple[Path, ...] = ()
) -> Mapping[str, Any]:
    config_path = config_path.resolve()
    if config_path in loading:
        chain = " -> ".join(str(path) for path in (*loading, config_path))
        raise ValueError(f"Circular asset config inheritance: {chain}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), str(config_path))
    parent_value = raw.get("extends")
    if parent_value is None:
        return raw
    parent_path = Path(str(parent_value))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _load_yaml_with_extends(parent_path, (*loading, config_path))
    override = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_merge(parent, override)


def _mapping(data: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return data


def _tuple(data: Any, field_name: str) -> tuple[Any, ...]:
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    return tuple(data)


def load_asset_spec(
    path: str | Path,
) -> SatelliteSpec | ManipulatorSpec | SatelliteManipulatorSpec:
    config_path = Path(path).resolve()
    raw = _load_yaml_with_extends(config_path)

    asset_id = str(raw.get("id", "")).strip()
    urdf_value = Path(str(raw.get("urdf", "")))
    urdf = urdf_value if urdf_value.is_absolute() else (config_path.parent / urdf_value)
    urdf = urdf.resolve()

    links = _mapping(raw.get("links", {}), "links")
    morph_raw = _mapping(raw.get("morph", {}), "morph")
    spawn = _mapping(morph_raw.get("spawn", {}), "morph.spawn")
    material_raw = _mapping(raw.get("material", {}), "material")

    grasp_sites = []
    for index, item in enumerate(raw.get("grasp_sites", [])):
        grasp = _mapping(item, f"grasp_sites[{index}]")
        grasp_sites.append(
            GraspSiteSpec(
                site_id=str(grasp.get("id", "")),
                link=str(grasp.get("link", "")),
                position=_tuple(grasp.get("position", ()), "grasp position"),
                quaternion=_tuple(
                    grasp.get("quaternion", (1.0, 0.0, 0.0, 0.0)),
                    "grasp quaternion",
                ),
                marker_offset=_tuple(
                    grasp.get("marker_offset", (0.0, 0.0, 0.0)),
                    "grasp marker offset",
                ),
            )
        )

    known_keys = {
        "id", "kind", "urdf", "links", "morph", "material", "pid",
        "cameras", "grasp_sites", "mount_site", "parameters", "manipulator",
    }
    unknown = sorted(set(raw) - known_keys)
    if unknown:
        raise ValueError(f"Unknown top-level fields in {config_path}: {unknown}")

    common = dict(
        asset_id=asset_id,
        kind=str(raw.get("kind", "")),
        urdf=urdf,
        base_link=str(links.get("base", "")),
        morph=MorphSpec(
            pose=PoseSpec(
                position=_tuple(spawn.get("position", (0.0, 0.0, 0.0)), "spawn.position"),
                euler=_tuple(spawn.get("euler", (0.0, 0.0, 0.0)), "spawn.euler"),
            ),
            scale=float(morph_raw.get("scale", 1.0)),
            fixed=bool(morph_raw.get("fixed", False)),
            collision=bool(morph_raw.get("collision", True)),
            visualization=bool(morph_raw.get("visualization", True)),
            merge_fixed_links=morph_raw.get("merge_fixed_links"),
            links_to_keep=tuple(morph_raw.get("links_to_keep", ())),
            convexify=bool(morph_raw.get("convexify", False)),
            decompose_object_error_threshold=morph_raw.get(
                "decompose_object_error_threshold"
            ),
            coacd=dict(morph_raw.get("coacd", {})),
        ),
        material=MaterialSpec(
            gravity_compensation=float(material_raw.get("gravity_compensation", 1.0))
        ),
        pid=_load_pid(raw.get("pid", {})),
        camera=_load_cameras(asset_id, raw.get("cameras", {})),
        grasp_sites=tuple(grasp_sites),
        parameters=dict(_mapping(raw.get("parameters", {}), "parameters")),
    )
    kind = str(raw.get("kind", ""))
    mount_site = None
    if kind in {"satellite", "satellite_manipulator"}:
        mount_raw = _mapping(raw.get("mount_site", {}), "mount_site")
        mount_site = MountSiteSpec(
            link=str(mount_raw.get("link", "")),
            position=_tuple(mount_raw.get("position", ()), "mount_site.position"),
            euler=_tuple(
                mount_raw.get("euler", (0.0, 0.0, 0.0)), "mount_site.euler"
            ),
        )
    if kind == "satellite":
        return SatelliteSpec(**common, mount_site=mount_site)
    if kind not in {"manipulator", "satellite_manipulator"}:
        raise ValueError(
            f"Unsupported asset kind {kind!r}; expected 'satellite', "
            "'manipulator', or 'satellite_manipulator'"
        )

    manipulator = _mapping(raw.get("manipulator", {}), "manipulator")
    arm_joints = _tuple(manipulator.get("arm_joints", ()), "manipulator.arm_joints")
    gripper_raw = _mapping(manipulator.get("gripper", {}), "manipulator.gripper")
    gripper_joints = _tuple(
        gripper_raw.get("joint_names", ()), "manipulator.gripper.joint_names"
    )
    control_raw = _mapping(manipulator.get("control", {}), "manipulator.control")
    ik_raw = _mapping(manipulator.get("ik", {}), "manipulator.ik")
    tcp_raw = _mapping(manipulator.get("tcp", {}), "manipulator.tcp")
    reference_raw = _mapping(
        manipulator.get("reference_pose", {}), "manipulator.reference_pose"
    )
    manipulator_common = dict(
        **common,
        manipulator_base_link=str(manipulator.get("base_link", "")),
        end_effector_link=str(manipulator.get("end_effector_link", "")),
        pid_apply_link=manipulator.get("pid_apply_link"),
        arm_joint_names=tuple(str(name) for name in arm_joints),
        control=JointControlSpec(
            initial_dofs=_tuple(control_raw.get("initial_dofs", ()), "initial_dofs"),
            kp=_tuple(control_raw.get("kp", ()), "control.kp"),
            kv=_tuple(control_raw.get("kv", ()), "control.kv"),
            force_min=_tuple(control_raw.get("force_min", ()), "control.force_min"),
            force_max=_tuple(control_raw.get("force_max", ()), "control.force_max"),
        ),
        ik=IKSpec(
            smooth_factor=float(ik_raw.get("smooth_factor", 0.0)),
            max_joint_change=float(ik_raw.get("max_joint_change", 0.0)),
        ),
        tcp=TCPSpec(
            link=str(tcp_raw.get("link", "")),
            position=_tuple(tcp_raw.get("position", ()), "manipulator.tcp.position"),
            quaternion=_tuple(
                tcp_raw.get("quaternion", (1.0, 0.0, 0.0, 0.0)),
                "manipulator.tcp.quaternion",
            ),
        ),
        reference_pose=ManipulatorReferencePoseSpec(
            link=str(reference_raw.get("link", "")),
            joint=str(reference_raw.get("joint", "")),
            position=_tuple(
                reference_raw.get("position", ()),
                "manipulator.reference_pose.position",
            ),
            quaternion=_tuple(
                reference_raw.get("quaternion", ()),
                "manipulator.reference_pose.quaternion",
            ),
        ),
        gripper=GripperSpec(
            joint_names=tuple(str(name) for name in gripper_joints),
            revolute=_tuple(gripper_raw.get("revolute", ()), "gripper.revolute"),
            open_position=float(gripper_raw.get("open_position", 0.0)),
            closed_position=float(gripper_raw.get("closed_position", 0.0)),
            force_scale=float(gripper_raw.get("force_scale", 1.0)),
            contact_links=tuple(gripper_raw.get("contact_links", ())),
        ),
    )
    if kind == "manipulator":
        return ManipulatorSpec(**manipulator_common)
    return SatelliteManipulatorSpec(**manipulator_common, mount_site=mount_site)


def _load_pid(data: Any) -> PIDSpec:
    raw = _mapping(data, "pid")
    return PIDSpec(
        enabled=bool(raw.get("enable_pid", False)),
        p=_tuple(raw.get("P", ()), "pid.P"),
        i=_tuple(raw.get("I", ()), "pid.I"),
        d=_tuple(raw.get("D", ()), "pid.D"),
        setpoint=_tuple(raw.get("setpoint", ()), "pid.setpoint"),
        dt=float(raw.get("dt", 0.01)),
        limits=_tuple(raw.get("limits", ()), "pid.limits"),
    )


def _load_cameras(asset_id: str, data: Any) -> CameraSpec:
    raw = _mapping(data, "cameras")
    return CameraSpec(
        name=str(raw.get("name", asset_id)),
        wrist=_load_camera_channel(raw.get("wrist", {}), "cameras.wrist"),
        base=_load_camera_channel(raw.get("base", {}), "cameras.base"),
    )


def _load_camera_channel(data: Any, field_name: str) -> CameraChannelSpec:
    raw = _mapping(data, field_name)
    runtime_fields = {
        "enabled", "recording_enabled", "episodes_per_video",
        "image_effects", "rl_observation",
    }
    misplaced = sorted(runtime_fields.intersection(raw))
    if misplaced:
        raise ValueError(
            f"{field_name} contains scene-owned fields {misplaced}; "
            "move them to the scene YAML cameras section"
        )
    return CameraChannelSpec(
        camera=dict(_mapping(raw.get("camera", {}), f"{field_name}.camera")),
        mount_link=raw.get("mount_link"),
        position_offset=_tuple(
            raw.get("position_offset", (0.0, 0.0, 0.0)),
            f"{field_name}.position_offset",
        ),
        lookat_offset=_tuple(
            raw.get("lookat_offset", (1.0, 0.0, 0.0)),
            f"{field_name}.lookat_offset",
        ),
        up_offset=_tuple(
            raw.get("up_offset", (0.0, 0.0, 1.0)),
            f"{field_name}.up_offset",
        ),
    )
