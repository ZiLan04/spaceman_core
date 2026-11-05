"""Record high-resolution scene-camera images/videos for paper figures.

This script intentionally overrides rendering, camera, viewer, and lighting
settings locally so normal training configs do not need to be edited.
"""

import argparse
import sys
from pathlib import Path

import genesis as gs
import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from configs.registry import get_legacy_camera  # noqa: E402
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
from configs.configs import (  # noqa: E402
    EARTH_BACKGROUND_SETTINGS,
    LIGHT_SETTINGS,
    SCENE_SETTINGS,
    TPV_CAM_SETTINGS,
    make_light_vis_options,
)
from tasks.RL_task_parallel_stage1 import ParallelStage1BaseTask  # noqa: E402
from tasks.RL_task_parallel import _as_batched_tensor  # noqa: E402
from utils.utils import as_rotation_matrix  # noqa: E402


OUTPUT_DIR = current_file_path.parent / "recordings" / "paper_camera"
DEFAULT_IMAGE_RES = (1920, 1080)
DEFAULT_CAMERA_POS = (-3.0, -3.5, 2.5)
DEFAULT_CAMERA_LOOKAT = (0.0, 0.5, 0.5)
DEFAULT_CAMERA_FOV = 40.0
DEFAULT_VIEWER_POS = (-1.5, -2.0, 1.5)
DEFAULT_VIEWER_LOOKAT = (0.0, 0.0, 0.0)
DEFAULT_VIEWER_FOV = 40.0
DEFAULT_EARTH_EULER = (0.0, 0.0, -55.0)
FRAME_COLORS = {
    "origin": (1.0, 1.0, 1.0, 1.0),
    "x": (1.0, 0.05, 0.05, 1.0),
    "y": (0.05, 0.9, 0.05, 1.0),
    "z": (0.1, 0.35, 1.0, 1.0),
}


def parse_vec3(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return tuple(values)


def parse_vec2(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values.")
    return tuple(values)


def parse_resolution(text: str):
    normalized = text.lower().replace("x", ",")
    values = [int(item.strip()) for item in normalized.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 3840x2160.")
    return tuple(values)


def str_to_bool(text: str):
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def select_env_tensor(value, *, task, env_idx: int, expected_last_dim: int):
    tensor = _as_batched_tensor(
        value,
        device=task.device,
        dtype=task.datatype,
        expected_last_dim=expected_last_dim,
    )
    return tensor[min(max(int(env_idx), 0), tensor.shape[0] - 1)]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preview, screenshot, and record the TPV scene camera."
    )
    parser.add_argument("--target-asset", default="starlink")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-spacing", type=parse_vec2, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=("none", "batch"),
        default="none",
        help="Use 'none' on GPUs that cannot create Genesis BatchRenderer.",
    )
    parser.add_argument("--res", type=parse_resolution, default=DEFAULT_IMAGE_RES)
    parser.add_argument("--camera-pos", type=parse_vec3, default=DEFAULT_CAMERA_POS)
    parser.add_argument("--camera-lookat", type=parse_vec3, default=DEFAULT_CAMERA_LOOKAT)
    parser.add_argument("--camera-fov", type=float, default=DEFAULT_CAMERA_FOV)
    parser.add_argument("--camera-far", type=float, default=200.0)
    parser.add_argument("--camera-gui", type=str_to_bool, default=False)
    parser.add_argument("--viewer-pos", type=parse_vec3, default=DEFAULT_VIEWER_POS)
    parser.add_argument("--viewer-lookat", type=parse_vec3, default=DEFAULT_VIEWER_LOOKAT)
    parser.add_argument("--viewer-fov", type=float, default=DEFAULT_VIEWER_FOV)
    parser.add_argument(
        "--light-preset",
        default=None,
        choices=tuple(LIGHT_SETTINGS.get("presets", {}).keys()),
        help="Lighting preset from configs.LIGHT_SETTINGS.",
    )
    parser.add_argument("--ambient", type=parse_vec3, default=None)
    parser.add_argument("--background-color", type=parse_vec3, default=None)
    parser.add_argument("--shadow", type=str_to_bool, default=None)
    parser.add_argument("--show-world-frame", type=str_to_bool, default=False)
    parser.add_argument(
        "--earth-brightness",
        type=float,
        default=2.2,
        help="Multiply only the Earth background texture brightness. Use 1.0 to keep the original.",
    )
    parser.add_argument(
        "--earth-euler",
        type=parse_vec3,
        default=DEFAULT_EARTH_EULER,
        help="Earth background mesh Euler rotation in degrees, e.g. 0,0,-55.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--record-steps", type=int, default=120)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--save-image", dest="save_image", action="store_true", default=True)
    parser.add_argument("--no-save-image", dest="save_image", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--image-name", default="paper_scene_camera.png")
    parser.add_argument("--frame-markers", type=str_to_bool, default=False)
    parser.add_argument("--frame-env-idx", type=int, default=0)
    parser.add_argument("--frame-axis-length", type=float, default=0.28)
    parser.add_argument("--frame-axis-thickness", type=float, default=0.018)
    parser.add_argument("--frame-origin-size", type=float, default=0.04)
    parser.add_argument("--satellite-base-frame-offset", type=parse_vec3, default=(0.0, 0.0, 0.42))
    parser.add_argument("--arm-base-frame-offset", type=parse_vec3, default=(0.0, 0.0, 0.24))
    parser.add_argument("--gripper-frame-offset", type=parse_vec3, default=(0.0, 0.0, 0.12))
    parser.add_argument("--target-frame-offset", type=parse_vec3, default=(0.0, 0.0, 0.42))
    return parser


def make_bright_texture(src_path: str, output_dir: Path, *, brightness: float) -> str:
    if brightness <= 1.0001:
        return src_path

    src = Path(src_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{src.stem}_bright_{brightness:.2f}{src.suffix}"
    if dst.exists():
        return str(dst)

    try:
        from PIL import Image

        image = Image.open(src).convert("RGBA")
        arr = np.asarray(image).astype(np.float32)
        rgb = np.clip(arr[..., :3] * float(brightness), 0, 255)
        arr[..., :3] = rgb
        Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(dst)
        return str(dst)
    except Exception as exc:
        print(
            f"[PAPER_CAMERA] Could not brighten Earth texture, using original: {exc}",
            flush=True,
        )
        return src_path


def configure_for_capture(args):
    """Apply local rendering/camera overrides before GenesisSim is created."""

    if args.renderer == "none":
        SCENE_SETTINGS["renderer"] = None

    SCENE_SETTINGS["show_viewer"] = not args.no_viewer
    SCENE_SETTINGS["viewer_options"] = gs.options.ViewerOptions(
        camera_pos=args.viewer_pos,
        camera_lookat=args.viewer_lookat,
        camera_fov=args.viewer_fov,
    )

    if args.light_preset is not None:
        LIGHT_SETTINGS["active_preset"] = args.light_preset
    if args.ambient is not None:
        LIGHT_SETTINGS["ambient_light"] = args.ambient
    if args.background_color is not None:
        LIGHT_SETTINGS["background_color"] = args.background_color
    if args.shadow is not None:
        LIGHT_SETTINGS["shadow"] = bool(args.shadow)
    LIGHT_SETTINGS["show_world_frame"] = bool(args.show_world_frame)
    SCENE_SETTINGS["vis_options"] = make_light_vis_options(
        LIGHT_SETTINGS.get("active_preset")
    )

    EARTH_BACKGROUND_SETTINGS["texture_path"] = make_bright_texture(
        EARTH_BACKGROUND_SETTINGS["texture_path"],
        args.output_dir,
        brightness=args.earth_brightness,
    )
    EARTH_BACKGROUND_SETTINGS["euler"] = tuple(args.earth_euler)

    TPV_CAM_SETTINGS["enable_recording"] = True
    TPV_CAM_SETTINGS["episodes_per_video"] = 1
    TPV_CAM_SETTINGS["camera"].update(
        {
            "res": args.res,
            "pos": args.camera_pos,
            "lookat": args.camera_lookat,
            "fov": args.camera_fov,
            "far": args.camera_far,
            "GUI": bool(args.camera_gui),
            "debug": True,
        }
    )

    # Keep paper capture independent from RL visual observations and their
    # BatchRenderer path.
    rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
    rl_obs_config["enabled"] = False
    rl_obs_config["enable_recording"] = False
    base_camera_config = FRANKA_S_Q_CAMERA.get("base_camera", {})
    base_camera_config["enabled"] = False
    base_camera_config["enable_recording"] = False
    base_rl_obs_config = base_camera_config.get("rl_obs", {})
    base_rl_obs_config["enabled"] = False
    base_rl_obs_config["enable_recording"] = False


def save_rgb_image(path: Path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
        return
    except Exception:
        pass

    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except Exception as exc:
        raise RuntimeError(
            "Saving PNG requires either imageio or Pillow to be installed."
        ) from exc


class FrameMarkers:
    """Small XYZ frames made from visible geometry for paper screenshots."""

    AXIS_SIZES = {
        "x": lambda length, thickness: (length, thickness, thickness),
        "y": lambda length, thickness: (thickness, length, thickness),
        "z": lambda length, thickness: (thickness, thickness, length),
    }

    AXIS_DIRS = {
        "x": torch.tensor([1.0, 0.0, 0.0]),
        "y": torch.tensor([0.0, 1.0, 0.0]),
        "z": torch.tensor([0.0, 0.0, 1.0]),
    }

    def __init__(
        self,
        scene,
        *,
        names,
        axis_length: float,
        axis_thickness: float,
        origin_size: float,
    ):
        self.scene = scene
        self.names = tuple(names)
        self.axis_length = float(axis_length)
        self.axis_thickness = float(axis_thickness)
        self.origin_size = float(origin_size)
        self.entities = {}
        self.disabled = False
        self._create()

    def _create_box(self, name, part, *, size, color):
        hidden_pos = (1.0e6, 1.0e6, 1.0e6)
        return self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=hidden_pos,
                size=size,
                fixed=True,
                collision=False,
                visualization=True,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
            surface=gs.surfaces.Plastic(color=color),
            name=f"paper_frame_{name}_{part}",
        )

    def _create(self):
        try:
            for name in self.names:
                self.entities[(name, "origin")] = self._create_box(
                    name,
                    "origin",
                    size=(self.origin_size, self.origin_size, self.origin_size),
                    color=FRAME_COLORS["origin"],
                )
                for axis, size_fn in self.AXIS_SIZES.items():
                    self.entities[(name, axis)] = self._create_box(
                        name,
                        axis,
                        size=size_fn(self.axis_length, self.axis_thickness),
                        color=FRAME_COLORS[axis],
                    )
        except Exception as exc:
            print(f"[PAPER_CAMERA] Could not create frame markers: {exc}", flush=True)
            self.entities = {}
            self.disabled = True

    def set_frame(self, name, pos, quat, *, local_offset):
        if self.disabled:
            return
        pos = torch.as_tensor(pos).detach().reshape(3)
        quat = torch.as_tensor(quat).detach().reshape(4)
        local_offset = torch.as_tensor(
            local_offset,
            dtype=pos.dtype,
            device=pos.device,
        ).reshape(3)
        rot = as_rotation_matrix(quat.reshape(1, 4), order="wxyz")[0].detach()
        origin = pos + torch.matmul(rot, local_offset)

        origin_entity = self.entities.get((name, "origin"))
        if origin_entity is not None:
            origin_entity.set_pos(origin, zero_velocity=True)
            origin_entity.set_quat(quat, zero_velocity=True)

        for axis, local_dir in self.AXIS_DIRS.items():
            entity = self.entities.get((name, axis))
            if entity is None:
                continue
            local_dir = local_dir.to(dtype=origin.dtype, device=origin.device)
            center = origin + torch.matmul(rot, local_dir) * (0.5 * self.axis_length)
            entity.set_pos(center, zero_velocity=True)
            entity.set_quat(quat, zero_velocity=True)

    def update_from_task(self, task, args):
        env_idx = min(max(int(args.frame_env_idx), 0), task.num_envs - 1)

        satellite_base_pos, satellite_base_quat = task._get_base_pose()
        self.set_frame(
            "satellite_base",
            satellite_base_pos[env_idx],
            satellite_base_quat[env_idx],
            local_offset=args.satellite_base_frame_offset,
        )

        arm_base_link = task.starlink_manipulator.robot.get_link(
            task.starlink_manipulator.params["base"]
        )
        self.set_frame(
            "arm_base",
            select_env_tensor(arm_base_link.get_pos(), task=task, env_idx=env_idx, expected_last_dim=3),
            select_env_tensor(arm_base_link.get_quat(), task=task, env_idx=env_idx, expected_last_dim=4),
            local_offset=args.arm_base_frame_offset,
        )

        gripper_link = task.starlink_manipulator.robot.get_link(
            task.starlink_manipulator.params["end_effector"]
        )
        self.set_frame(
            "gripper",
            select_env_tensor(gripper_link.get_pos(), task=task, env_idx=env_idx, expected_last_dim=3),
            select_env_tensor(gripper_link.get_quat(), task=task, env_idx=env_idx, expected_last_dim=4),
            local_offset=args.gripper_frame_offset,
        )

        target_base_link = task.target_object.robot.get_link(task.target_object.params["base"])
        self.set_frame(
            "target_satellite",
            select_env_tensor(target_base_link.get_pos(), task=task, env_idx=env_idx, expected_last_dim=3),
            select_env_tensor(target_base_link.get_quat(), task=task, env_idx=env_idx, expected_last_dim=4),
            local_offset=args.target_frame_offset,
        )


def render_one_frame(task):
    if task.gsim.cam is None:
        raise RuntimeError("TPV camera was not created. Check TPV_CAM_SETTINGS.")
    rgb, *rest = task.gsim.cam.render(rgb=True, force_render=True)
    return rgb


def step_scene(task, steps: int, markers=None, args=None):
    for _ in range(max(0, int(steps))):
        if markers is not None:
            markers.update_from_task(task, args)
        task.starlink_manipulator.step()
        task.target_object.step()
        task.gsim.step()
    if markers is not None:
        markers.update_from_task(task, args)


def clear_camera_recording_buffer(task):
    cam = task.gsim.cam
    if cam is None:
        return
    recorded_imgs = getattr(cam, "_recorded_imgs", None)
    if recorded_imgs is not None:
        recorded_imgs.clear()
    if not getattr(cam, "_in_recording", False):
        cam.start_recording()


def main():
    args = build_arg_parser().parse_args()
    configure_for_capture(args)

    task = ParallelStage1BaseTask(
        target_asset_name=args.target_asset,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing[:2] if args.env_spacing is not None else None,
        enable_visual_obs=False,
    )
    markers = None
    if args.frame_markers:
        markers = FrameMarkers(
            task.gsim.scene,
            names=("satellite_base", "arm_base", "gripper", "target_satellite"),
            axis_length=args.frame_axis_length,
            axis_thickness=args.frame_axis_thickness,
            origin_size=args.frame_origin_size,
        )
    if not task.initialize():
        raise RuntimeError("ParallelStage1BaseTask failed to initialize.")
    if not task.reset():
        raise RuntimeError("ParallelStage1BaseTask failed to reset.")

    step_scene(task, args.warmup_steps, markers=markers, args=args)
    clear_camera_recording_buffer(task)

    if args.save_image:
        if markers is not None:
            markers.update_from_task(task, args)
        rgb = render_one_frame(task)
        image_path = args.output_dir / args.image_name
        save_rgb_image(image_path, rgb)
        print(f"[PAPER_CAMERA] Saved image: {image_path}")

    if args.record_steps > 0:
        print(f"[PAPER_CAMERA] Recording {args.record_steps} frames...")
        clear_camera_recording_buffer(task)
        step_scene(task, args.record_steps, markers=markers, args=args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.output_dir / "paper_scene_camera.mp4"
        task.gsim.cam.stop_recording(
            save_to_filename=str(video_path),
            fps=args.video_fps,
        )
        print(f"[PAPER_CAMERA] Saved video: {video_path}")
        task.gsim.camera_enable_recording = False
    else:
        task.gsim.camera_enable_recording = False

    task.stop()


if __name__ == "__main__":
    main()
