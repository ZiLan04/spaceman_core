"""
Headless test for the RL base camera + Genesis BatchRenderer.

IMPORTANT:
- This test uses the SAME RL base camera defined in
  FRANKA_S_Q_CAMERA["base_camera"].
- The scene renderer is explicitly changed to Genesis BatchRenderer.
- NEVER call BatchedMountedCamera.capture() in this test.
  capture() falls back to the normal OpenGL/pyrender renderer and
  causes the "Attempt to retrieve context when no valid context" error.
- The BatchRenderer directly provides RGB + depth.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import genesis as gs


# ============================================================
# Project path
# ============================================================

PROJECT_SRC = Path(__file__).resolve().parents[1]

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


from configs import asset_configs
from configs.registry import get_legacy_camera
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
from configs.configs import (
    LIGHT_SETTINGS,
    SCENE_SETTINGS,
    TPV_CAM_SETTINGS,
)
from controllers.backend import EmptyBackend
from envs.genesis_env import GenesisSim
from robots.robot import Robot
from robots.satellite_manipulator import SatelliteManipulator


# ============================================================
# Arguments
# ============================================================

def parse_vec3(value):
    values = tuple(
        float(x.strip())
        for x in value.split(",")
    )

    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "Expected x,y,z"
        )

    return values


def build_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-asset",
        default="starlink",
    )

    parser.add_argument(
        "--manipulator-asset",
        default="franka_merge",
    )

    parser.add_argument(
        "--vis-mode",
        choices=("visual", "collision"),
        default="visual",
    )

    parser.add_argument(
        "--light-preset",
        default=LIGHT_SETTINGS.get(
            "active_preset",
            "front_right",
        ),
        choices=tuple(
            LIGHT_SETTINGS.get(
                "presets",
                {},
            ).keys()
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--sun-mode",
        choices=(
            "camera_forward",
            "light",
        ),
        default="camera_forward",
    )

    parser.add_argument(
        "--light-dir",
        type=parse_vec3,
        default=None,
        help="Genesis light travel direction: x,y,z",
    )

    parser.add_argument(
        "--glare-strength",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=PROJECT_SRC / "recordings",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
    )

    return parser


# ============================================================
# Configure scene
# ============================================================

def configure_scene(args):

    # --------------------------------------------------------
    # NO viewer.
    # --------------------------------------------------------

    SCENE_SETTINGS["show_viewer"] = False

    # --------------------------------------------------------
    # THIS IS THE CRITICAL FIX.
    #
    # Previously:
    #
    #     SCENE_SETTINGS["renderer"] = None
    #
    # That creates the normal Rasterizer. Therefore the
    # BatchedMountedCamera's _batch_cam is actually a normal
    # Genesis camera with _batch_renderer == None.
    #
    # Then prepare_batch_render() passes [1,3] to set_pose(),
    # while the normal camera expects [3], producing:
    #
    #     Pos shape torch.Size([1, 3]) does not match (N, 3)
    #
    # We explicitly enable the real Genesis BatchRenderer.
    # --------------------------------------------------------

    SCENE_SETTINGS["renderer"] = (
        gs.renderers.BatchRenderer(
            use_rasterizer=True,
        )
    )

    # --------------------------------------------------------
    # We only need the RL base camera.
    # --------------------------------------------------------

    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False

    FRANKA_S_Q_CAMERA["rl_obs"]["enabled"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = False

    base_camera = FRANKA_S_Q_CAMERA[
        "base_camera"
    ]

    base_camera["enabled"] = True
    base_camera["enable_recording"] = False

    base_camera["rl_obs"]["enabled"] = True
    base_camera["rl_obs"]["enable_recording"] = False

    # This must remain True.
    base_camera["rl_obs"][
        "use_batch_renderer"
    ] = True

    base_camera["rl_obs"][
        "update_interval"
    ] = 1

    # --------------------------------------------------------
    # No ordinary camera recording.
    # --------------------------------------------------------

    TPV_CAM_SETTINGS[
        "enable_recording"
    ] = False


# ============================================================
# Asset visualization
# ============================================================

def override_asset_vis_mode(
    asset_name,
    vis_mode,
):

    try:
        original_factory = (
            asset_configs._ASSET_FACTORIES[
                asset_name
            ]
        )
    except KeyError as exc:

        available = ", ".join(
            sorted(
                asset_configs._ASSET_FACTORIES
            )
        )

        raise ValueError(
            f"Unknown asset {asset_name!r}. "
            f"Available assets: {available}"
        ) from exc

    def factory():

        asset = original_factory()

        asset["vis_mode"] = vis_mode

        return asset

    asset_configs._ASSET_FACTORIES[
        asset_name
    ] = factory


# ============================================================
# Tensor utilities
# ============================================================

def to_numpy(x):

    if x is None:
        return None

    if torch.is_tensor(x):

        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def first_env(x):

    x = to_numpy(x)

    if x is None:
        return None

    # BatchRenderer with n_envs=1 normally returns:
    #
    #   [1,H,W,3]
    #
    # or:
    #
    #   [1,H,W]
    #
    if x.ndim >= 1 and x.shape[0] == 1:
        return x[0]

    return x


def vec3(x):

    x = to_numpy(x)

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    x = x.reshape(-1)

    if x.size < 3:
        raise RuntimeError(
            f"Cannot convert {x.shape} to vec3."
        )

    return x[:3]


def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float32,
    )

    norm = np.linalg.norm(v)

    if norm < 1e-8:
        return v

    return v / norm


# ============================================================
# RGB
# ============================================================

def rgb_to_uint8(rgb):

    rgb = first_env(rgb)

    if rgb is None:
        raise RuntimeError(
            "RGB result is None."
        )

    if rgb.ndim != 3:
        raise RuntimeError(
            f"Unexpected RGB shape: {rgb.shape}"
        )

    # H,W,3
    if rgb.shape[-1] == 3:

        pass

    # 3,H,W
    elif rgb.shape[0] == 3:

        rgb = np.transpose(
            rgb,
            (1, 2, 0),
        )

    else:

        raise RuntimeError(
            f"Cannot determine RGB layout: "
            f"{rgb.shape}"
        )

    if np.issubdtype(
        rgb.dtype,
        np.floating,
    ):

        max_value = np.nanmax(
            rgb
        )

        if max_value <= 1.0:

            rgb = rgb * 255.0

    rgb = np.clip(
        rgb,
        0,
        255,
    )

    return np.ascontiguousarray(
        rgb.astype(
            np.uint8
        )
    )


# ============================================================
# Depth
# ============================================================

def depth_to_uint8(depth):

    depth = first_env(depth)

    if depth is None:
        return None

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    if depth.ndim == 3:

        depth = depth.squeeze()

    valid = (
        np.isfinite(depth)
        & (depth > 0)
    )

    if not np.any(valid):

        return np.zeros(
            depth.shape,
            dtype=np.uint8,
        )

    near = np.percentile(
        depth[valid],
        2,
    )

    far = np.percentile(
        depth[valid],
        98,
    )

    if far <= near:

        far = near + 1e-6

    normalized = (
        depth - near
    ) / (
        far - near
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    # Close = white, far = black.
    normalized = 1.0 - normalized

    normalized[
        ~valid
    ] = 0.0

    return (
        normalized * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# Camera pose
# ============================================================

def get_camera_pose(
    batch_camera,
):

    pos = vec3(
        batch_camera.get_pos()
    )

    lookat = vec3(
        batch_camera.get_lookat()
    )

    up = vec3(
        batch_camera.get_up()
    )

    return (
        pos,
        lookat,
        up,
    )


# ============================================================
# Light direction
# ============================================================

def get_light_direction(
    preset_name,
):

    preset = LIGHT_SETTINGS[
        "presets"
    ][preset_name]

    lights = preset.get(
        "lights",
        (),
    )

    if not lights:

        raise RuntimeError(
            f"Light preset {preset_name!r} "
            "does not contain any lights."
        )

    light = lights[0]

    if isinstance(
        light,
        dict,
    ):

        for key in (
            "dir",
            "direction",
        ):

            if key in light:

                return normalize(
                    vec3(
                        light[key]
                    )
                )

    for key in (
        "dir",
        "direction",
        "_dir",
        "_direction",
    ):

        if hasattr(
            light,
            key,
        ):

            value = getattr(
                light,
                key,
            )

            if value is not None:

                return normalize(
                    vec3(value)
                )

    raise RuntimeError(
        "Cannot extract light direction "
        "from current light configuration."
    )


# ============================================================
# Project sun into image
# ============================================================

def project_sun(
    cam_pos,
    cam_lookat,
    cam_up,
    light_travel_dir,
    width,
    height,
    fov,
):

    forward = normalize(
        cam_lookat - cam_pos
    )

    right = normalize(
        np.cross(
            forward,
            normalize(cam_up),
        )
    )

    up = normalize(
        np.cross(
            right,
            forward,
        )
    )

    # Genesis directional light vector means:
    #
    #     light -> scene
    #
    # Therefore:
    #
    #     camera -> sun = -light_dir
    #
    sun = -normalize(
        light_travel_dir
    )

    x = float(
        np.dot(
            sun,
            right,
        )
    )

    y = float(
        np.dot(
            sun,
            up,
        )
    )

    z = float(
        np.dot(
            sun,
            forward,
        )
    )

    if z <= 0.0:

        return (
            None,
            None,
            z,
        )

    fov_y = np.deg2rad(
        float(fov)
    )

    focal_y = (
        height / 2.0
    ) / np.tan(
        fov_y / 2.0
    )

    aspect = (
        width / height
    )

    focal_x = (
        focal_y
        * aspect
    )

    u = (
        (width - 1) / 2.0
        + focal_x * x / z
    )

    v = (
        (height - 1) / 2.0
        - focal_y * y / z
    )

    return (
        float(u),
        float(v),
        float(z),
    )


# ============================================================
# Gaussian
# ============================================================

def gaussian(
    h,
    w,
    cx,
    cy,
    sigma,
):

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    return np.exp(
        -(
            (
                xx - cx
            ) ** 2
            + (
                yy - cy
            ) ** 2
        )
        / (
            2.0
            * sigma
            * sigma
        )
    ).astype(
        np.float32
    )


# ============================================================
# Add synthetic solar glare
# ============================================================

def add_glare(
    rgb,
    sun_x,
    sun_y,
    strength=1.0,
):

    image = (
        rgb.astype(
            np.float32
        )
        / 255.0
    )

    h, w = image.shape[:2]

    diag = np.sqrt(
        h * h
        + w * w
    )

    # --------------------------------------------------------
    # Broad halo
    # --------------------------------------------------------

    halo_broad = gaussian(
        h,
        w,
        sun_x,
        sun_y,
        0.22 * diag,
    )

    # --------------------------------------------------------
    # Compact bright bloom
    # --------------------------------------------------------

    halo_small = gaussian(
        h,
        w,
        sun_x,
        sun_y,
        0.065 * diag,
    )

    # --------------------------------------------------------
    # Very broad veiling glare
    # --------------------------------------------------------

    halo_very_broad = gaussian(
        h,
        w,
        sun_x,
        sun_y,
        0.55 * diag,
    )

    glare = (
        0.48 * halo_broad
        + 0.32 * halo_small
        + 0.12 * halo_very_broad
    )

    # --------------------------------------------------------
    # Lens ghost opposite the sun.
    # --------------------------------------------------------

    image_cx = (
        w - 1
    ) / 2.0

    image_cy = (
        h - 1
    ) / 2.0

    ghost_x = (
        2.0 * image_cx
        - sun_x
    )

    ghost_y = (
        2.0 * image_cy
        - sun_y
    )

    ghost = (
        0.08
        * gaussian(
            h,
            w,
            ghost_x,
            ghost_y,
            0.10 * diag,
        )
    )

    # --------------------------------------------------------
    # Blur-based veiling glare.
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=max(
            1.0,
            w * 0.035,
        ),
        sigmaY=max(
            1.0,
            h * 0.035,
        ),
    )

    veil_strength = (
        0.09
        * float(strength)
    )

    output = (
        (
            1.0
            - veil_strength
        )
        * image
        + veil_strength
        * blurred
    )

    # Slightly warm solar flare.
    solar_rgb = np.array(
        [
            1.0,
            0.97,
            0.89,
        ],
        dtype=np.float32,
    )

    output += (
        float(strength)
        * (
            glare
            + ghost
        )[..., None]
        * solar_rgb
    )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    return (
        output * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# Video writer
# ============================================================

def create_writer(
    path,
    width,
    height,
    fps,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(fps),
        (
            int(width),
            int(height),
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Cannot create video: {path}"
        )

    return writer


# ============================================================
# Save RGB
# ============================================================

def save_rgb(
    path,
    rgb,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )


# ============================================================
# Save depth
# ============================================================

def save_depth(
    path,
    depth,
):

    if depth is None:
        return

    depth_vis = depth_to_uint8(
        depth
    )

    if depth_vis is None:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        depth_vis,
    )


# ============================================================
# Render ONE frame using BatchRenderer
# ============================================================

def render_base_camera(
    camera,
):

    # --------------------------------------------------------
    # This updates the camera mounting pose.
    #
    # Because the SCENE renderer is now the actual
    # BatchRenderer, _batch_cam._is_batched is True and
    # BatchedMountedCamera's [1,3] pose is now valid.
    # --------------------------------------------------------

    camera.prepare_batch_render()

    batch_camera = camera._batch_cam

    # --------------------------------------------------------
    # Sanity check.
    # --------------------------------------------------------

    if getattr(
        batch_camera,
        "_batch_renderer",
        None,
    ) is None:

        raise RuntimeError(
            "\n"
            "BatchRenderer is NOT attached to the RL base camera.\n"
            "\n"
            "This means the scene was not constructed with:\n"
            "    gs.renderers.BatchRenderer(...)\n"
        )

    # --------------------------------------------------------
    # DIRECT BatchRenderer render.
    #
    # NEVER use:
    #
    #     camera.capture()
    #
    # here.
    # --------------------------------------------------------

    rgb, depth, seg, normal = (
        batch_camera.render(
            rgb=True,
            depth=True,
            segmentation=False,
            normal=False,
            force_render=True,
        )
    )

    rgb = rgb_to_uint8(
        rgb
    )

    depth = first_env(
        depth
    )

    return (
        rgb,
        depth,
    )


# ============================================================
# Main test
# ============================================================

def run_test(
    args,
    sim,
    manipulator,
    target,
):

    camera = getattr(
        manipulator,
        "rl_base_camera",
        None,
    )

    if camera is None:

        raise RuntimeError(
            "SatelliteManipulator.rl_base_camera does not exist."
        )

    batch_camera = getattr(
        camera,
        "_batch_cam",
        None,
    )

    if batch_camera is None:

        raise RuntimeError(
            "rl_base_camera._batch_cam is None."
        )

    # --------------------------------------------------------
    # Check actual renderer.
    # --------------------------------------------------------

    actual_batch_renderer = getattr(
        batch_camera,
        "_batch_renderer",
        None,
    )

    print(
        "\n[CHECK] =======================================",
        flush=True,
    )

    print(
        "[CHECK] RL base camera:",
        type(camera),
        flush=True,
    )

    print(
        "[CHECK] Genesis camera:",
        type(batch_camera),
        flush=True,
    )

    print(
        "[CHECK] Genesis BatchRenderer:",
        actual_batch_renderer,
        flush=True,
    )

    print(
        "[CHECK] Camera is batched:",
        getattr(
            batch_camera,
            "_is_batched",
            None,
        ),
        flush=True,
    )

    print(
        "[CHECK] =======================================\n",
        flush=True,
    )

    if actual_batch_renderer is None:

        raise RuntimeError(
            "STOP: _batch_cam._batch_renderer is None. "
            "The scene is still using the normal rasterizer."
        )

    # --------------------------------------------------------
    # Static scene state.
    # --------------------------------------------------------

    manipulator_qpos = (
        manipulator.robot
        .get_qpos()
        .detach()
        .clone()
    )

    target_qpos = (
        target.robot
        .get_qpos()
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    output_dir = (
        args.recordings_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_writer = None
    glare_writer = None
    compare_writer = None

    # --------------------------------------------------------
    # Camera configuration.
    # --------------------------------------------------------

    base_cfg = FRANKA_S_Q_CAMERA[
        "base_camera"
    ]

    camera_cfg = base_cfg[
        "camera"
    ]

    fov = float(
        camera_cfg[
            "fov"
        ]
    )

    print(
        "[CAMERA] configured resolution:",
        camera_cfg.get(
            "res",
            None,
        ),
        flush=True,
    )

    print(
        "[CAMERA] configured FOV:",
        fov,
        flush=True,
    )

    # --------------------------------------------------------
    # Light direction.
    # --------------------------------------------------------

    if args.light_dir is not None:

        light_travel_dir = normalize(
            np.asarray(
                args.light_dir,
                dtype=np.float32,
            )
        )

    elif args.sun_mode == "light":

        light_travel_dir = (
            get_light_direction(
                args.light_preset
            )
        )

    else:

        light_travel_dir = None

    # --------------------------------------------------------
    # Render.
    # --------------------------------------------------------

    try:

        for frame_idx in range(
            args.num_frames
        ):

            # ------------------------------------------------
            # Keep scene static.
            # ------------------------------------------------

            manipulator.robot.set_qpos(
                manipulator_qpos,
                zero_velocity=True,
            )

            target.robot.set_qpos(
                target_qpos,
                zero_velocity=True,
            )

            # ------------------------------------------------
            # Simulation step.
            # ------------------------------------------------

            sim.step()

            # ------------------------------------------------
            # BatchRenderer RGB + depth.
            # ------------------------------------------------

            raw, depth = (
                render_base_camera(
                    camera
                )
            )

            h, w = raw.shape[:2]

            # ------------------------------------------------
            # Initialize writers.
            # ------------------------------------------------

            if raw_writer is None:

                print(
                    f"[CAMERA] actual image size: "
                    f"{w} x {h}",
                    flush=True,
                )

                raw_writer = create_writer(
                    output_dir
                    / "base_raw.mp4",
                    w,
                    h,
                    args.fps,
                )

                glare_writer = create_writer(
                    output_dir
                    / "base_glare.mp4",
                    w,
                    h,
                    args.fps,
                )

                compare_writer = create_writer(
                    output_dir
                    / "base_compare.mp4",
                    2 * w,
                    h,
                    args.fps,
                )

            # ------------------------------------------------
            # Get actual camera pose.
            # ------------------------------------------------

            (
                cam_pos,
                cam_lookat,
                cam_up,
            ) = get_camera_pose(
                batch_camera
            )

            # ------------------------------------------------
            # Determine sun position.
            # ------------------------------------------------

            if (
                args.sun_mode
                == "camera_forward"
            ):

                # Test mode:
                # put the sun directly in the camera optical axis.
                #
                # This isolates the post-processing effect from
                # the actual Genesis light configuration.

                sun_direction = normalize(
                    cam_lookat
                    - cam_pos
                )

            else:

                sun_direction = (
                    -normalize(
                        light_travel_dir
                    )
                )

            (
                sun_x,
                sun_y,
                sun_z,
            ) = project_sun(
                cam_pos,
                cam_lookat,
                cam_up,
                -sun_direction,
                w,
                h,
                fov,
            )

            # ------------------------------------------------
            # Apply glare.
            # ------------------------------------------------

            if (
                sun_x is None
                or sun_z <= 0.0
            ):

                glare = raw.copy()

            else:

                glare = add_glare(
                    raw,
                    sun_x,
                    sun_y,
                    strength=args.glare_strength,
                )

            # ------------------------------------------------
            # Side-by-side.
            #
            # LEFT  = original BatchRenderer RGB
            # RIGHT = RGB + synthetic solar glare
            # ------------------------------------------------

            compare = np.concatenate(
                [
                    raw,
                    glare,
                ],
                axis=1,
            )

            # ------------------------------------------------
            # Write videos.
            # ------------------------------------------------

            raw_writer.write(
                cv2.cvtColor(
                    raw,
                    cv2.COLOR_RGB2BGR,
                )
            )

            glare_writer.write(
                cv2.cvtColor(
                    glare,
                    cv2.COLOR_RGB2BGR,
                )
            )

            compare_writer.write(
                cv2.cvtColor(
                    compare,
                    cv2.COLOR_RGB2BGR,
                )
            )

            # ------------------------------------------------
            # Save still frames.
            # ------------------------------------------------

            if (
                args.save_every > 0
                and frame_idx
                % args.save_every == 0
            ):

                save_rgb(
                    output_dir
                    / f"base_raw_{frame_idx:05d}.png",
                    raw,
                )

                save_rgb(
                    output_dir
                    / f"base_glare_{frame_idx:05d}.png",
                    glare,
                )

                save_rgb(
                    output_dir
                    / f"base_compare_{frame_idx:05d}.png",
                    compare,
                )

                save_depth(
                    output_dir
                    / f"base_depth_{frame_idx:05d}.png",
                    depth,
                )

            # ------------------------------------------------
            # Print diagnostics.
            # ------------------------------------------------

            if (
                frame_idx % 30 == 0
                or frame_idx
                == args.num_frames - 1
            ):

                print(
                    "[FRAME] "
                    f"{frame_idx:04d} | "
                    f"camera="
                    f"{np.round(cam_pos, 3)} | "
                    f"lookat="
                    f"{np.round(cam_lookat, 3)} | "
                    f"sun_px="
                    f"{None if sun_x is None else round(sun_x, 1)} | "
                    f"sun_py="
                    f"{None if sun_y is None else round(sun_y, 1)}",
                    flush=True,
                )

            if args.sleep > 0:

                time.sleep(
                    args.sleep
                )

    finally:

        if raw_writer is not None:
            raw_writer.release()

        if glare_writer is not None:
            glare_writer.release()

        if compare_writer is not None:
            compare_writer.release()

    print(
        "\n[DONE]",
        flush=True,
    )

    print(
        "[DONE] raw:",
        output_dir
        / "base_raw.mp4",
        flush=True,
    )

    print(
        "[DONE] glare:",
        output_dir
        / "base_glare.mp4",
        flush=True,
    )

    print(
        "[DONE] compare:",
        output_dir
        / "base_compare.mp4",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = (
        build_arg_parser()
        .parse_args()
    )

    # --------------------------------------------------------
    # Configure BEFORE GenesisSim is created.
    # --------------------------------------------------------

    configure_scene(
        args
    )

    override_asset_vis_mode(
        args.target_asset,
        args.vis_mode,
    )

    override_asset_vis_mode(
        args.manipulator_asset,
        args.vis_mode,
    )

    # --------------------------------------------------------
    # Create scene.
    # --------------------------------------------------------

    sim = GenesisSim(
        num_envs=1,
        light_preset=args.light_preset,
    )

    manipulator = SatelliteManipulator(
        name=args.manipulator_asset,
        sensors=[],
        backends=[
            EmptyBackend()
        ],
    )

    target = Robot(
        name=args.target_asset,
        sensors=[],
        backends=[
            EmptyBackend()
        ],
    )

    try:

        print(
            "[INIT] Starting Genesis...",
            flush=True,
        )

        sim.start()

        print(
            "[INIT] Initializing manipulator...",
            flush=True,
        )

        manipulator.initialize()

        print(
            "[INIT] Initializing target...",
            flush=True,
        )

        target.initialize()

        # ----------------------------------------------------
        # Run test.
        # ----------------------------------------------------

        run_test(
            args,
            sim,
            manipulator,
            target,
        )

    except KeyboardInterrupt:

        print(
            "\n[STOP] Interrupted.",
            flush=True,
        )

    finally:

        try:
            target.stop()
        except Exception:
            pass

        try:
            manipulator.stop()
        except Exception:
            pass

        try:
            sim.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()