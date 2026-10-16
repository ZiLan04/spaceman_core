"""Inspect the satellite scene through the server BatchRenderer cameras.

This script is intentionally persistent: it runs without a viewer and records
the RL wrist/base RGB streams until the process receives Ctrl+C.
"""

import argparse
import sys
import time
from pathlib import Path

import genesis as gs


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from configs.configs import LIGHT_SETTINGS, SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from controllers.backend import EmptyBackend  # noqa: E402
from envs.genesis_env import GenesisSim  # noqa: E402
from robots.robot import Robot  # noqa: E402
from robots.satellite_manipulator import SatelliteManipulator  # noqa: E402
from sensors import rl_image_recorder  # noqa: E402
from satellite_manipulator_viewer_test import (  # noqa: E402
    DEFAULT_MANIPULATOR_ASSET,
    DEFAULT_TARGET_ASSET,
    capture_rl_camera_frames,
    override_asset_vis_mode,
)


DEFAULT_RECORDINGS_DIR = PROJECT_SRC / "recordings" / "batch_renderer_scene_test"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a persistent, viewer-free satellite scene using Genesis "
            "BatchRenderer and record the RL wrist/base cameras."
        )
    )
    parser.add_argument("--target-asset", default=DEFAULT_TARGET_ASSET)
    parser.add_argument("--manipulator-asset", default=DEFAULT_MANIPULATOR_ASSET)
    parser.add_argument(
        "--vis-mode",
        choices=("visual", "collision"),
        default="visual",
        help="Render normal visual meshes or color-coded collision meshes.",
    )
    parser.add_argument(
        "--light-preset",
        default=LIGHT_SETTINGS.get("active_preset", "front_right"),
        choices=tuple(LIGHT_SETTINGS.get("presets", {}).keys()),
        help="Lighting preset shared by BatchRenderer and the regular rasterizer.",
    )
    parser.add_argument(
        "--camera-segment-seconds",
        type=float,
        default=60.0,
        help="Recorded MP4 duration per file; 0 saves one file when exiting.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help="Output directory for wrist/base MP4 files.",
    )
    parser.add_argument(
        "--enable-physics-collision",
        action="store_true",
        help="Enable physical collision detection in the inspection scene.",
    )
    parser.add_argument(
        "--collision-algorithm",
        choices=("gjk", "mpr"),
        default="gjk",
        help="Physics narrow-phase algorithm; GJK is slower but more stable.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Optional delay after each frame to reduce idle CPU usage.",
    )
    return parser


def configure_batch_scene(args: argparse.Namespace) -> None:
    SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
    SCENE_SETTINGS["show_viewer"] = False
    SCENE_SETTINGS["rigid_options"].enable_collision = bool(
        args.enable_physics_collision
    )
    SCENE_SETTINGS["rigid_options"].use_gjk_collision = (
        args.collision_algorithm == "gjk"
    )

    # The TPV camera is a debug camera and would use a separate EGL rasterizer.
    # This test records only the two non-debug cameras consumed by RL.
    TPV_CAM_SETTINGS["enable_recording"] = False

    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enabled"] = True
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = True
    FRANKA_S_Q_CAMERA["rl_obs"]["use_batch_renderer"] = True

    base_camera = FRANKA_S_Q_CAMERA["base_camera"]
    base_camera["enabled"] = False
    base_camera["enable_recording"] = False
    base_camera["rl_obs"]["enabled"] = True
    base_camera["rl_obs"]["enable_recording"] = True
    base_camera["rl_obs"]["use_batch_renderer"] = True


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.camera_segment_seconds < 0.0:
        raise ValueError("--camera-segment-seconds must be non-negative.")

    configure_batch_scene(args)
    override_asset_vis_mode(args.target_asset, args.vis_mode)
    override_asset_vis_mode(args.manipulator_asset, args.vis_mode)
    rl_image_recorder._RECORDINGS_DIR = args.recordings_dir.resolve()

    sim = GenesisSim(num_envs=1, light_preset=args.light_preset)
    manipulator = SatelliteManipulator(
        name=args.manipulator_asset,
        sensors=[],
        backends=[EmptyBackend()],
    )
    target = Robot(
        name=args.target_asset,
        sensors=[],
        backends=[EmptyBackend()],
    )

    try:
        sim.start()
        manipulator.initialize()
        target.initialize()

        manipulator_qpos = manipulator.robot.get_qpos().detach().clone()
        target_qpos = target.robot.get_qpos().detach().clone()
        wrist_config = FRANKA_S_Q_CAMERA["rl_obs"]
        base_config = FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]

        print(
            "[BATCH TEST] Scene ready: "
            f"target={args.target_asset}, manipulator={args.manipulator_asset}, "
            f"vis_mode={args.vis_mode}, light={args.light_preset}, "
            "renderer=BatchRenderer(use_rasterizer=True), viewer=False.",
            flush=True,
        )
        print(
            f"[BATCH TEST] Recording to {args.recordings_dir}: "
            f"wrist_res={FRANKA_S_Q_CAMERA['camera']['res']} "
            f"wrist_fov={FRANKA_S_Q_CAMERA['camera']['fov']} "
            f"wrist_fps={wrist_config['record_fps']}; "
            f"base_res={FRANKA_S_Q_CAMERA['base_camera']['camera']['res']} "
            f"base_fov={FRANKA_S_Q_CAMERA['base_camera']['camera']['fov']} "
            f"base_fps={base_config['record_fps']}; "
            f"segment_seconds={args.camera_segment_seconds}.",
            flush=True,
        )
        print("[BATCH TEST] Press Ctrl+C to stop and flush recordings.", flush=True)

        while True:
            manipulator.robot.set_qpos(manipulator_qpos, zero_velocity=True)
            target.robot.set_qpos(target_qpos, zero_velocity=True)
            sim.step()
            capture_rl_camera_frames(
                manipulator,
                segment_seconds=args.camera_segment_seconds,
            )
            if args.sleep > 0.0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[BATCH TEST] Interrupted by user.", flush=True)
    finally:
        target.stop()
        manipulator.stop()
        sim.stop()


if __name__ == "__main__":
    main()
