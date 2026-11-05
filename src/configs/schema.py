"""Typed schemas for data-driven simulation assets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _require_vector(name: str, value: tuple[float, ...], size: int) -> None:
    if len(value) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(value)}")


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    selection_mode: str
    satellite_manipulators: tuple[str, ...]
    target_satellites: tuple[str, ...]
    seed: int | None = None
    target_position_noise: bool = False
    target_position_noise_range: tuple[float, float, float] = (0.03, 0.03, 0.02)
    cameras: SceneCameraSpec = field(default_factory=lambda: SceneCameraSpec())

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene id must not be empty")
        if self.selection_mode not in {"fixed", "random"}:
            raise ValueError("scene selection mode must be 'fixed' or 'random'")
        if not self.satellite_manipulators or not self.target_satellites:
            raise ValueError("scene asset candidate lists must not be empty")
        if self.selection_mode == "fixed" and (
            len(self.satellite_manipulators) != 1 or len(self.target_satellites) != 1
        ):
            raise ValueError("fixed scenes must define exactly one asset of each kind")
        _require_vector(
            "target position noise range", self.target_position_noise_range, 3
        )

    def select_assets(self, seed: int | None = None) -> tuple[str, str]:
        rng = random.Random(self.seed if seed is None else seed)
        return (
            rng.choice(self.satellite_manipulators),
            rng.choice(self.target_satellites),
        )


@dataclass(frozen=True)
class PoseSpec:
    position: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _require_vector("position", self.position, 3)
        _require_vector("euler", self.euler, 3)


@dataclass(frozen=True)
class GraspSiteSpec:
    site_id: str
    link: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    marker_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.site_id or not self.link:
            raise ValueError("grasp site id and link must not be empty")
        _require_vector("grasp position", self.position, 3)
        _require_vector("grasp quaternion", self.quaternion, 4)
        _require_vector("grasp marker offset", self.marker_offset, 3)


@dataclass(frozen=True)
class MountSiteSpec:
    link: str
    position: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.link:
            raise ValueError("mount site link must not be empty")
        _require_vector("mount position", self.position, 3)
        _require_vector("mount euler", self.euler, 3)


@dataclass(frozen=True)
class MorphSpec:
    pose: PoseSpec
    scale: float = 1.0
    fixed: bool = False
    collision: bool = True
    visualization: bool = True
    merge_fixed_links: bool | None = None
    links_to_keep: tuple[str, ...] = ()
    convexify: bool = False
    decompose_object_error_threshold: float | None = None
    coacd: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialSpec:
    gravity_compensation: float = 1.0


@dataclass(frozen=True)
class PIDSpec:
    enabled: bool = False
    p: tuple[float, ...] = ()
    i: tuple[float, ...] = ()
    d: tuple[float, ...] = ()
    setpoint: tuple[float, ...] = ()
    dt: float = 0.01
    limits: tuple[float | None, ...] = ()

    def to_legacy_dict(self, asset_id: str) -> dict[str, Any]:
        return {
            "name": asset_id,
            "enable_pid": self.enabled,
            "P": list(self.p),
            "I": list(self.i),
            "D": list(self.d),
            "setpoint": list(self.setpoint),
            "dt": self.dt,
            "limits": list(self.limits),
        }


@dataclass(frozen=True)
class SceneCameraChannelSpec:
    enabled: bool = False
    recording_enabled: bool = False
    episodes_per_video: int = 1
    image_effects: Mapping[str, Any] = field(default_factory=dict)
    rl_observation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episodes_per_video < 1:
            raise ValueError("camera episodes_per_video must be at least 1")


@dataclass(frozen=True)
class SceneCameraSpec:
    wrist: SceneCameraChannelSpec = field(default_factory=SceneCameraChannelSpec)
    base: SceneCameraChannelSpec = field(default_factory=SceneCameraChannelSpec)


@dataclass(frozen=True)
class CameraChannelSpec:
    camera: Mapping[str, Any] = field(default_factory=dict)
    mount_link: str | None = None
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lookat_offset: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up_offset: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        _require_vector("camera position offset", self.position_offset, 3)
        _require_vector("camera lookat offset", self.lookat_offset, 3)
        _require_vector("camera up offset", self.up_offset, 3)

    def to_legacy_dict(
        self,
        *,
        base_camera: bool,
        runtime: SceneCameraChannelSpec | None = None,
    ) -> dict[str, Any]:
        runtime = runtime or SceneCameraChannelSpec()
        config = {
            "camera": dict(self.camera),
            "enable_recording": runtime.recording_enabled,
            "episodes_per_video": runtime.episodes_per_video,
            "pos_offset": list(self.position_offset),
            "lookat_offset": list(self.lookat_offset),
            "up_offset": list(self.up_offset),
            "image_effects": dict(runtime.image_effects),
            "rl_obs": dict(runtime.rl_observation),
        }
        if base_camera:
            config["enabled"] = runtime.enabled
            if self.mount_link:
                config["mount_link"] = self.mount_link
        else:
            config["wrist_camera"] = runtime.enabled
            if self.mount_link:
                config["end_effector_link"] = self.mount_link
        return config


@dataclass(frozen=True)
class CameraSpec:
    name: str
    wrist: CameraChannelSpec = field(default_factory=CameraChannelSpec)
    base: CameraChannelSpec = field(default_factory=CameraChannelSpec)

    def to_legacy_dict(self, runtime: SceneCameraSpec | None = None) -> dict[str, Any]:
        runtime = runtime or SceneCameraSpec()
        config = {"name": self.name}
        config.update(
            self.wrist.to_legacy_dict(base_camera=False, runtime=runtime.wrist)
        )
        config["base_camera"] = self.base.to_legacy_dict(
            base_camera=True, runtime=runtime.base
        )
        return config


@dataclass(frozen=True)
class JointControlSpec:
    initial_dofs: tuple[float, ...]
    kp: tuple[float, ...]
    kv: tuple[float, ...]
    force_min: tuple[float, ...]
    force_max: tuple[float, ...]

    def __post_init__(self) -> None:
        sizes = {
            len(self.initial_dofs), len(self.kp), len(self.kv),
            len(self.force_min), len(self.force_max),
        }
        if len(sizes) != 1:
            raise ValueError("all joint control vectors must have the same length")

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "initial_dofs": list(self.initial_dofs),
            "control": {
                "kp": list(self.kp),
                "kv": list(self.kv),
                "force_range_min": list(self.force_min),
                "force_range_max": list(self.force_max),
            },
        }


@dataclass(frozen=True)
class IKSpec:
    smooth_factor: float
    max_joint_change: float


@dataclass(frozen=True)
class TCPSpec:
    link: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.link:
            raise ValueError("TCP link must not be empty")
        _require_vector("TCP position", self.position, 3)
        _require_vector("TCP quaternion", self.quaternion, 4)


@dataclass(frozen=True)
class ManipulatorReferencePoseSpec:
    link: str
    joint: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.link or not self.joint:
            raise ValueError("manipulator reference link and joint must not be empty")
        _require_vector("manipulator reference position", self.position, 3)
        _require_vector("manipulator reference quaternion", self.quaternion, 4)


@dataclass(frozen=True)
class GripperSpec:
    joint_names: tuple[str, ...]
    revolute: tuple[float, ...]
    open_position: float
    closed_position: float
    force_scale: float = 1.0
    contact_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.revolute):
            raise ValueError("gripper joint_names and revolute must have equal lengths")


@dataclass(frozen=True)
class BaseAssetSpec:
    asset_id: str
    kind: str
    urdf: Path
    base_link: str
    morph: MorphSpec
    material: MaterialSpec
    pid: PIDSpec
    camera: CameraSpec
    grasp_sites: tuple[GraspSiteSpec, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id or not self.kind or not self.base_link:
            raise ValueError("asset id, kind, and base link must not be empty")
        if not self.urdf.is_absolute():
            raise ValueError(f"asset URDF path must be absolute: {self.urdf}")

    @property
    def default_grasp_site(self) -> GraspSiteSpec | None:
        return self.grasp_sites[0] if self.grasp_sites else None

    def to_legacy_params(self) -> dict[str, Any]:
        """Return the dictionary shape consumed by the current Robot classes."""
        params = dict(self.parameters)
        params.update({"name": self.asset_id, "base": self.base_link, "path": self.urdf})
        grasp = self.default_grasp_site
        if grasp is not None:
            params.update(
                {
                    "grasp_link": grasp.link,
                    "grasp_offset_local": list(grasp.position),
                    "grasp_marker_offset_local": list(grasp.marker_offset),
                }
            )
        return params


@dataclass(frozen=True)
class SatelliteSpec(BaseAssetSpec):
    mount_site: MountSiteSpec | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.grasp_sites:
            raise ValueError("satellite must define at least one grasp site")
        if self.mount_site is None:
            raise ValueError("satellite must define a manipulator mount site")


@dataclass(frozen=True)
class ManipulatorSpec(BaseAssetSpec):
    """A standalone arm asset; unlike a combined vehicle it has no mount site."""

    manipulator_base_link: str = ""
    end_effector_link: str = ""
    pid_apply_link: str | None = None
    arm_joint_names: tuple[str, ...] = ()
    control: JointControlSpec | None = None
    ik: IKSpec | None = None
    tcp: TCPSpec | None = None
    reference_pose: ManipulatorReferencePoseSpec | None = None
    gripper: GripperSpec | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_manipulator(self)

    @property
    def joint_names(self) -> tuple[str, ...]:
        assert self.gripper is not None
        return self.arm_joint_names + self.gripper.joint_names

    def to_legacy_params(self) -> dict[str, Any]:
        return _manipulator_legacy_params(self, super().to_legacy_params())


@dataclass(frozen=True)
class SatelliteManipulatorSpec(SatelliteSpec):
    manipulator_base_link: str = ""
    end_effector_link: str = ""
    pid_apply_link: str | None = None
    arm_joint_names: tuple[str, ...] = ()
    control: JointControlSpec | None = None
    ik: IKSpec | None = None
    tcp: TCPSpec | None = None
    reference_pose: ManipulatorReferencePoseSpec | None = None
    gripper: GripperSpec | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_manipulator(self)

    @property
    def joint_names(self) -> tuple[str, ...]:
        assert self.gripper is not None
        return self.arm_joint_names + self.gripper.joint_names

    def to_legacy_params(self) -> dict[str, Any]:
        return _manipulator_legacy_params(self, super().to_legacy_params())


def _validate_manipulator(spec: ManipulatorSpec | SatelliteManipulatorSpec) -> None:
    if not spec.manipulator_base_link or not spec.end_effector_link:
        raise ValueError("manipulator base and end-effector links are required")
    if spec.control is None or spec.ik is None or spec.tcp is None or spec.gripper is None:
        raise ValueError("manipulator control, ik, TCP, and gripper specs are required")
    if spec.tcp.link != spec.end_effector_link:
        raise ValueError("TCP link must match the manipulator end-effector link")
    if spec.reference_pose is None:
        raise ValueError("manipulator reference pose is required")
    if spec.reference_pose.link != spec.end_effector_link:
        raise ValueError("reference pose link must match the end-effector link")
    if spec.reference_pose.joint not in spec.arm_joint_names:
        raise ValueError("reference pose joint must be an arm joint")
    if len(spec.gripper.contact_links) not in {0, 2}:
        raise ValueError("gripper must define zero or exactly two contact links")
    if len(spec.control.initial_dofs) != len(spec.joint_names):
        raise ValueError("initial DOFs must match arm and gripper joint count")


def _manipulator_legacy_params(
    spec: ManipulatorSpec | SatelliteManipulatorSpec,
    params: dict[str, Any],
) -> dict[str, Any]:
    assert spec.control is not None and spec.ik is not None and spec.gripper is not None
    params.update(
        {
            # Existing controllers use "base" for the arm base, while
            # BaseAssetSpec.base_link remains the satellite root link.
            "base": spec.manipulator_base_link,
            "end_effector": spec.end_effector_link,
            "pid_apply_link": spec.pid_apply_link or spec.base_link,
            "joints": spec.joint_names,
            "motor": len(spec.arm_joint_names),
            "finger": len(spec.joint_names),
            "gripper_revolute": list(spec.gripper.revolute),
            "gripper_force_scale": spec.gripper.force_scale,
            "finger_open": [spec.gripper.open_position]
            * len(spec.gripper.joint_names),
            "finger_close": [spec.gripper.closed_position]
            * len(spec.gripper.joint_names),
            "ik_params": {
                "smooth_factor": spec.ik.smooth_factor,
                "max_joint_change": spec.ik.max_joint_change,
            },
            "config": spec.control.to_legacy_dict(),
        }
    )
    return params
