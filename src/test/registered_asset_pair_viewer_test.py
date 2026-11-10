"""Load a target satellite and a satellite-manipulator in the Genesis viewer.

The script intentionally creates no task, controller, sensor, or camera. It
exercises only the YAML registry -> Genesis morph -> scene build path. The two
entities use the original Starlink capture-scene layout and remain fixed at
their initial poses until the viewer is closed.

Examples:

    python src/test/registered_asset_pair_viewer_test.py
    python src/test/registered_asset_pair_viewer_test.py --target-satellite gps
    python src/test/registered_asset_pair_viewer_test.py \
        --satellite-manipulator gps__airbus_vispa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import genesis as gs


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs.configs import (  # noqa: E402
    APP_SETTINGS,
    LIGHT_SETTINGS,
    SCENE_SETTINGS,
    TPV_CAM_SETTINGS,
    SPACE_BACKGROUND_SETTINGS,
    EARTH_BACKGROUND_SETTINGS,
)
from configs.registry import build_genesis_asset, registry  # noqa: E402
from configs.schema import SatelliteManipulatorSpec, SatelliteSpec  # noqa: E402
from envs.genesis_env import GenesisSim  # noqa: E402


DEFAULT_TARGET_SATELLITE = "artificial2"
DEFAULT_SATELLITE_MANIPULATOR = "artificial2__starlink_sldasm9"
DEFAULT_TARGET_POSITION = (-5.0, 0.0, 0.5)
DEFAULT_MANIPULATOR_POSITION = (5.0, 0.0, 0.0)
DEFAULT_VIEWER_POSITION = (-1.5, -3.5, 2.0)
DEFAULT_VIEWER_LOOKAT = (-0.75, 0.0, 0.45)


def parse_vec3(value: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a target satellite and satellite-manipulator in a viewer."
    )
    parser.add_argument("--target-satellite", default=DEFAULT_TARGET_SATELLITE)
    parser.add_argument("--scene-plan", type=Path, help="JSON from asset_pair_api.py plan; applies safe poses and named joints")
    parser.add_argument(
        "--satellite-manipulator", default=DEFAULT_SATELLITE_MANIPULATOR
    )
    parser.add_argument(
        "--target-position",
        type=parse_vec3,
        default=DEFAULT_TARGET_POSITION,
    )
    parser.add_argument(
        "--manipulator-position",
        type=parse_vec3,
        default=DEFAULT_MANIPULATOR_POSITION,
    )
    parser.add_argument(
        "--viewer-position", type=parse_vec3, default=DEFAULT_VIEWER_POSITION
    )
    parser.add_argument(
        "--viewer-lookat", type=parse_vec3, default=DEFAULT_VIEWER_LOOKAT
    )
    parser.add_argument("--viewer-fov", type=float, default=45.0)
    parser.add_argument(
        "--light-preset",
        choices=tuple(LIGHT_SETTINGS.get("presets", {})),
        default=LIGHT_SETTINGS.get("active_preset", "left"),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Build headlessly for automated smoke checks.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many simulation steps; 0 runs until viewer close.",
    )
    parser.add_argument("--sleep", type=float, default=0.01)
    return parser


def resolve_asset_specs(
    target_id: str, manipulator_id: str
) -> tuple[SatelliteSpec, SatelliteManipulatorSpec]:
    target = registry.get(target_id)
    manipulator = registry.get(manipulator_id)
    if type(target) is not SatelliteSpec:
        raise TypeError(
            f"{target_id!r} must be a standalone satellite; "
            f"got kind={target.kind!r}"
        )
    if not isinstance(manipulator, SatelliteManipulatorSpec):
        raise TypeError(
            f"{manipulator_id!r} must be a satellite_manipulator; "
            f"got kind={manipulator.kind!r}"
        )
    return target, manipulator


def configure_scene(args: argparse.Namespace) -> None:
    APP_SETTINGS["logging_level"] = "warning"
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not args.no_viewer
    if args.no_viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
        APP_SETTINGS["backend"] = gs.cpu
        SPACE_BACKGROUND_SETTINGS["enable"] = False
        EARTH_BACKGROUND_SETTINGS["enable"] = False
        # Upstream with_camera() tests key presence, not enable_recording.
        # Remove the camera for a true display-free loading test.
        TPV_CAM_SETTINGS.pop("camera", None)
    SCENE_SETTINGS["viewer_options"] = gs.options.ViewerOptions(
        camera_pos=args.viewer_position,
        camera_lookat=args.viewer_lookat,
        camera_fov=float(args.viewer_fov),
    )
    TPV_CAM_SETTINGS["enable_recording"] = False


def main() -> None:
    args = build_arg_parser().parse_args()
    plan = None
    if args.scene_plan:
        from asset_pair_api import resolve_specs
        plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
        resolve_specs(plan)
        args.target_satellite = plan["target_satellite"]
        args.satellite_manipulator = plan["pair_id"]
        args.target_position = plan["target_pose"]["position_m"]
        args.manipulator_position = plan["carrier_pose"]["position_m"]
        args.viewer_position = plan["camera"]["position_m"]
        args.viewer_lookat = plan["camera"]["lookat_m"]
    if args.no_viewer and args.steps <= 0:
        args.steps = 2

    target_spec, manipulator_spec = resolve_asset_specs(
        args.target_satellite, args.satellite_manipulator
    )
    configure_scene(args)

    sim = GenesisSim(num_envs=1, light_preset=args.light_preset)
    target = sim.scene.add_entity(
        **build_genesis_asset(target_spec, pos=args.target_position, **({"euler": plan["target_pose"]["euler_deg"]} if plan else {})),
        name=f"registry_test_target_{args.target_satellite}",
    )
    manipulator = sim.scene.add_entity(
        **build_genesis_asset(
            manipulator_spec, pos=args.manipulator_position, **({"euler": plan["carrier_pose"]["euler_deg"]} if plan else {})
        ),
        name=f"registry_test_manipulator_{args.satellite_manipulator}",
    )

    sim.start()
    if plan:
        from asset_pair_api import apply_initial_pose
        apply_initial_pose({"carrier": manipulator}, plan)
    target_qpos = target.get_qpos().detach().clone()
    manipulator_qpos = manipulator.get_qpos().detach().clone()
    print(
        "[REGISTERED_ASSET_PAIR] Scene built successfully: "
        f"target={args.target_satellite} at {tuple(args.target_position)}, "
        f"satellite_manipulator={args.satellite_manipulator} "
        f"at {tuple(args.manipulator_position)}.",
        flush=True,
    )
    if not args.no_viewer:
        print("[REGISTERED_ASSET_PAIR] Close the viewer or press Ctrl+C to exit.")

    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            if not args.no_viewer and (
                sim.viewer is None or not sim.viewer.is_alive()
            ):
                break
            target.set_qpos(target_qpos, zero_velocity=True)
            manipulator.set_qpos(manipulator_qpos, zero_velocity=True)
            sim.step()
            step += 1
            if args.sleep > 0.0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[REGISTERED_ASSET_PAIR] Interrupted by user.", flush=True)


if __name__ == "__main__":
    main()
