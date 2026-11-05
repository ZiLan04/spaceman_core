"""Display one target satellite and one satellite-mounted manipulator.

This is an inspection scene rather than an automated assertion test. It keeps
the initial model poses unchanged and runs until the Genesis viewer is closed
or the process receives Ctrl+C.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import genesis as gs


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs import asset_configs  # noqa: E402
from configs.registry import get_legacy_camera  # noqa: E402
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
from configs.configs import LIGHT_SETTINGS, SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from controllers.backend import EmptyBackend  # noqa: E402
from envs.genesis_env import GenesisSim  # noqa: E402
from robots.robot import Robot  # noqa: E402
from robots.satellite_manipulator import SatelliteManipulator  # noqa: E402
from sensors import rl_image_recorder  # noqa: E402


DEFAULT_TARGET_ASSET = "starlink"
DEFAULT_MANIPULATOR_ASSET = "franka_merge"
DEFAULT_VIEWER_POS = (-1.5, -3.5, 2.0)
DEFAULT_VIEWER_LOOKAT = (-0.75, 0.0, 0.45)


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a persistent viewer with a satellite and satellite manipulator."
    )
    parser.add_argument("--target-asset", default=DEFAULT_TARGET_ASSET)
    parser.add_argument("--manipulator-asset", default=DEFAULT_MANIPULATOR_ASSET)
    parser.add_argument(
        "--vis-mode",
        choices=("visual", "collision"),
        default="visual",
        help="Show the normal render meshes or the color-coded collision meshes.",
    )
    parser.add_argument(
        "--light-preset",
        default=LIGHT_SETTINGS.get("active_preset", "front_right"),
        choices=tuple(LIGHT_SETTINGS.get("presets", {}).keys()),
    )
    parser.add_argument("--viewer-pos", type=parse_vec3, default=DEFAULT_VIEWER_POS)
    parser.add_argument(
        "--viewer-lookat", type=parse_vec3, default=DEFAULT_VIEWER_LOOKAT
    )
    parser.add_argument("--viewer-fov", type=float, default=45.0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the interactive viewer; camera recording remains enabled.",
    )
    parser.add_argument(
        "--enable-physics-collision",
        action="store_true",
        help="Enable physical collision detection; not needed for model inspection.",
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
        help="Optional delay after each simulation step to reduce idle CPU usage.",
    )
    parser.add_argument(
        "--no-camera-recording",
        action="store_true",
        help="Disable the RL wrist/base cameras and their MP4 recordings.",
    )
    parser.add_argument(
        "--camera-segment-seconds",
        type=float,
        default=60.0,
        help="MP4 duration before saving a new chunk; 0 saves only on exit.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=PROJECT_SRC / "recordings",
        help="Directory for wrist/base MP4 files.",
    )
    return parser


def configure_viewer(args: argparse.Namespace) -> None:
    """Configure the viewer and the two RL observation camera streams."""
    record_cameras = not args.no_camera_recording
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not args.no_viewer
    if args.no_viewer:
        # This machine's EGL device selection crashes in the NVIDIA driver.
        # Pyglet provides a hidden native OpenGL context without a viewer window.
        os.environ["PYOPENGL_PLATFORM"] = "pyglet"
    SCENE_SETTINGS["viewer_options"] = gs.options.ViewerOptions(
        camera_pos=args.viewer_pos,
        camera_lookat=args.viewer_lookat,
        camera_fov=float(args.viewer_fov),
    )
    SCENE_SETTINGS["rigid_options"].enable_collision = bool(
        args.enable_physics_collision
    )
    SCENE_SETTINGS["rigid_options"].use_gjk_collision = (
        args.collision_algorithm == "gjk"
    )
    TPV_CAM_SETTINGS["enable_recording"] = False

    # Keep the RL optical and mounting parameters while using one ordinary
    # rasterizer camera per stream. In headless mode they share Pyglet's hidden
    # OpenGL context; in viewer mode they share the visible viewer context.
    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enabled"] = record_cameras
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = record_cameras
    FRANKA_S_Q_CAMERA["rl_obs"]["use_batch_renderer"] = False

    base_camera = FRANKA_S_Q_CAMERA["base_camera"]
    base_camera["enabled"] = False
    base_camera["enable_recording"] = False
    base_camera["rl_obs"]["enabled"] = record_cameras
    base_camera["rl_obs"]["enable_recording"] = record_cameras
    base_camera["rl_obs"]["use_batch_renderer"] = False


def capture_rl_camera_frames(
    manipulator: SatelliteManipulator,
    *,
    segment_seconds: float,
) -> None:
    """Capture and record the same wrist/base RGB streams used by RL."""
    wrist_camera = getattr(manipulator, "rl_wrist_camera", None)
    base_camera = getattr(manipulator, "rl_base_camera", None)
    cameras = tuple(
        camera for camera in (wrist_camera, base_camera) if camera is not None
    )
    can_share_batch_render = len(cameras) == 2 and all(
        hasattr(camera, "prepare_batch_render")
        and hasattr(camera, "render_prepared_batch")
        and getattr(camera, "_batch_cam", None) is not None
        for camera in cameras
    )
    if can_share_batch_render:
        prepared = [
            (camera, camera.prepare_batch_render()) for camera in cameras
        ]
        trigger_used = False
        for camera, should_render in prepared:
            if should_render:
                camera.render_prepared_batch(force_render=not trigger_used)
                trigger_used = True
            else:
                camera.capture()
    else:
        for camera in cameras:
            camera.capture()

    if segment_seconds <= 0.0:
        return
    for camera in cameras:
        recorder = camera.recorder
        segment_frames = max(1, round(recorder.fps * segment_seconds))
        if recorder.buffered_frame_count >= segment_frames:
            recorder.save_segment()


def override_asset_vis_mode(asset_name: str, vis_mode: str) -> None:
    """Apply a process-local visualization mode to one registered asset."""
    try:
        original_factory = asset_configs._ASSET_FACTORIES[asset_name]
    except KeyError as exc:
        available = ", ".join(sorted(asset_configs._ASSET_FACTORIES))
        raise ValueError(
            f"Unknown asset {asset_name!r}. Available assets: {available}"
        ) from exc

    def make_asset_with_vis_mode():
        asset = original_factory()
        asset["vis_mode"] = vis_mode
        return asset

    asset_configs._ASSET_FACTORIES[asset_name] = make_asset_with_vis_mode


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_viewer(args)
    override_asset_vis_mode(args.target_asset, args.vis_mode)
    override_asset_vis_mode(args.manipulator_asset, args.vis_mode)
    rl_image_recorder._RECORDINGS_DIR = args.recordings_dir.resolve()

    sim = GenesisSim(num_envs=1, light_preset=args.light_preset)
    empty_backend = EmptyBackend()
    manipulator = SatelliteManipulator(
        name=args.manipulator_asset,
        sensors=[],
        backends=[empty_backend],
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

        print(
            "[VIEWER] Scene ready: "
            f"target={args.target_asset}, manipulator={args.manipulator_asset}, "
            f"vis_mode={args.vis_mode}, light={args.light_preset}, "
            f"physics_collision={args.enable_physics_collision}, "
            f"collision_algorithm={args.collision_algorithm}.",
            flush=True,
        )
        if args.no_camera_recording:
            print("[CAMERA] RL wrist/base recording disabled.", flush=True)
        else:
            wrist_config = FRANKA_S_Q_CAMERA["rl_obs"]
            base_config = FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]
            print(
                f"[CAMERA] Recording RL wrist/base RGB streams to {args.recordings_dir}: "
                f"wrist_res={FRANKA_S_Q_CAMERA['camera']['res']} "
                f"wrist_fov={FRANKA_S_Q_CAMERA['camera']['fov']} "
                f"wrist_fps={wrist_config['record_fps']}; "
                f"base_res={FRANKA_S_Q_CAMERA['base_camera']['camera']['res']} "
                f"base_fov={FRANKA_S_Q_CAMERA['base_camera']['camera']['fov']} "
                f"base_fps={base_config['record_fps']}; "
                f"segment_seconds={args.camera_segment_seconds}.",
                flush=True,
            )
        if args.no_viewer:
            print("[VIEWER] Disabled; press Ctrl+C to exit.", flush=True)
        else:
            print(
                "[VIEWER] Close the viewer window or press Ctrl+C to exit.",
                flush=True,
            )

        while args.no_viewer or (sim.viewer is not None and sim.viewer.is_alive()):
            # Reset all root and joint states so this remains a static model and
            # lighting inspection scene even after many simulation steps.
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
        print("\n[VIEWER] Interrupted by user.", flush=True)
    finally:
        target.stop()
        manipulator.stop()
        sim.stop()


if __name__ == "__main__":
    main()
