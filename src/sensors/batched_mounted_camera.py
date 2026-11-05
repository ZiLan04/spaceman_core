"""
Batched generic link-mounted camera sensor for parallel RL training.
"""

from copy import deepcopy
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from controllers.backend import Backend
from envs.genesis_env import GenesisSim
from sensors.image_effects import CameraImageEffectPipeline
from sensors.rl_image_recorder import RLImageRecorder
from utils.utils import calculate_camera_pose


DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS = {
    "camera": {
        "res": (84, 84),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 70,
        "GUI": False,
    },
    "mount_link": "base_link",
    "pos_offset": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
    "lookat_offset": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "up_offset": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    "image_effects": {},
    "rl_obs": {
        "enabled": True,
        "channels_first": True,
        "normalize": False,
        "enable_recording": False,
        "record_env_idx": 0,
        "record_fps": 30,
        "episodes_per_video": 1,
        "update_interval": 1,
        "use_batch_renderer": True,
    },
}


def _clone_config(config):
    if config is None:
        return deepcopy(DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS)

    merged = deepcopy(DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS)
    for key, value in config.items():
        if key == "camera" and isinstance(value, dict):
            merged["camera"].update(value)
        elif key == "rl_obs" and isinstance(value, dict):
            merged["rl_obs"].update(value)
        else:
            merged[key] = value
    merged["camera"]["res"] = tuple(int(value) for value in merged["camera"]["res"])
    return merged


class BatchedMountedCamera(Backend):
    def __init__(self, config=None):
        super().__init__()
        self.config = _clone_config(config)
        self._sim = GenesisSim()
        self._scene = self._sim.scene
        self.device = self._sim.device
        self.datatype = torch.float32
        self.num_envs = GenesisSim().num_envs

        self._cams = []
        self.mount_link = None
        self.cam_pos = None
        self.cam_lookat = None
        self.cam_up = None

        rl_obs_config = self.config.get("rl_obs", {})
        self.channels_first = bool(rl_obs_config.get("channels_first", True))
        self.normalize = bool(rl_obs_config.get("normalize", False))
        self.update_interval = max(1, int(rl_obs_config.get("update_interval", 1)))
        self.use_batch_renderer = bool(rl_obs_config.get("use_batch_renderer", True))
        self.image_effects = CameraImageEffectPipeline(
            self.config.get("image_effects"), device=self.device
        )
        self.recorder = RLImageRecorder(
            name="rl_base_camera",
            enabled=bool(rl_obs_config.get("enable_recording", False)),
            env_idx=int(rl_obs_config.get("record_env_idx", 0)),
            fps=int(rl_obs_config.get("record_fps", 30)),
            episodes_per_video=int(rl_obs_config.get("episodes_per_video", 1)),
            channels_first=self.channels_first,
            normalize=self.normalize,
        )

        camera_config = deepcopy(self.config["camera"])
        self._batch_cam = None
        if self.use_batch_renderer:
            env_camera_config = deepcopy(camera_config)
            env_camera_config["GUI"] = False
            env_camera_config.pop("env_idx", None)
            self._batch_cam = self._scene.add_camera(**env_camera_config)
            self._cams.append(self._batch_cam)
        else:
            base_gui = bool(camera_config.get("GUI", False))
            for env_idx in range(self.num_envs):
                env_camera_config = deepcopy(camera_config)
                env_camera_config["GUI"] = base_gui and env_idx == 0
                env_camera_config["env_idx"] = env_idx
                self._cams.append(self._scene.add_camera(**env_camera_config))

        self.latest_rgb: Optional[torch.Tensor] = None
        self._step_counter = 0

    def initialize(self, robot=None):
        self._robot = robot
        if robot is None or not hasattr(robot, "robot"):
            raise AttributeError(
                "BatchedMountedCamera requires a Robot wrapper with a 'robot' Genesis entity."
            )
        mount_link_name = self.config.get(
            "mount_link", DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS["mount_link"]
        )
        self.mount_link = robot.robot.get_link(mount_link_name)

        camera_res = self.config["camera"]["res"]
        channels = 3
        if self.channels_first:
            shape = (self.num_envs, channels, camera_res[1], camera_res[0])
        else:
            shape = (self.num_envs, camera_res[1], camera_res[0], channels)
        dtype = torch.float32 if self.normalize else torch.uint8
        self.latest_rgb = torch.zeros(shape, dtype=dtype, device=self.device)

    def _get_offsets(self):
        pos_offset = torch.as_tensor(
            self.config.get("pos_offset", DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS["pos_offset"]),
            dtype=self.datatype,
            device=self.device,
        )
        lookat_offset = torch.as_tensor(
            self.config.get(
                "lookat_offset", DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS["lookat_offset"]
            ),
            dtype=self.datatype,
            device=self.device,
        )
        up_offset = torch.as_tensor(
            self.config.get("up_offset", DEFAULT_BATCHED_MOUNTED_CAM_SETTINGS["up_offset"]),
            dtype=self.datatype,
            device=self.device,
        )
        return pos_offset, lookat_offset, up_offset

    def _ensure_batched_pose(self):
        position = torch.as_tensor(
            self.mount_link.get_pos(), dtype=self.datatype, device=self.device
        )
        quaternion = torch.as_tensor(
            self.mount_link.get_quat(), dtype=self.datatype, device=self.device
        )
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if quaternion.dim() == 1:
            quaternion = quaternion.unsqueeze(0)
        if position.shape[0] != self.num_envs or quaternion.shape[0] != self.num_envs:
            raise ValueError(
                "BatchedMountedCamera pose batch size mismatch: "
                f"expected {self.num_envs}, got pos={tuple(position.shape)}, "
                f"quat={tuple(quaternion.shape)}"
            )
        return position, quaternion

    def _update_batched_pose(self):
        position, quaternion = self._ensure_batched_pose()
        pos_offset, lookat_offset, up_offset = self._get_offsets()
        self.cam_pos, self.cam_lookat, self.cam_up = calculate_camera_pose(
            position,
            quaternion,
            pos_offset,
            lookat_offset,
            up_offset,
        )
        return self.cam_pos, self.cam_lookat, self.cam_up

    def _format_batched_rgb(self, rgb) -> torch.Tensor:
        if torch.is_tensor(rgb):
            rgb_tensor = rgb.detach().to(device=self.device).contiguous()
        else:
            rgb_array = np.asarray(rgb)
            if any(stride < 0 for stride in rgb_array.strides) or not rgb_array.flags["C_CONTIGUOUS"]:
                rgb_array = np.ascontiguousarray(rgb_array)
            rgb_tensor = torch.from_numpy(rgb_array).to(device=self.device)

        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = torch.clamp(rgb_tensor, 0, 255)
            if rgb_tensor.max().item() <= 1.0:
                rgb_tensor = rgb_tensor * 255.0
            rgb_tensor = rgb_tensor.to(torch.uint8)

        if rgb_tensor.dim() != 4:
            raise ValueError(
                f"Expected batched RGB image with 4 dims, got shape {tuple(rgb_tensor.shape)}"
            )
        if rgb_tensor.shape[-1] == 3:
            if self.channels_first:
                rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).contiguous()
        elif rgb_tensor.shape[1] == 3:
            if not self.channels_first:
                rgb_tensor = rgb_tensor.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(
                f"Expected batched RGB image with 3 channels, got shape {tuple(rgb_tensor.shape)}"
            )

        if self.normalize:
            rgb_tensor = rgb_tensor.to(torch.float32) / 255.0
        return rgb_tensor

    def _apply_image_effects(self, rgb: torch.Tensor) -> torch.Tensor:
        if not self.image_effects.enabled:
            return rgb
        glare_config = self.image_effects.solar_glare.config
        light_direction = None
        if glare_config["direction_source"] == "scene_light":
            light_index = int(glare_config.get("light_index", 0))
            directions = self._sim.light_directions
            if not directions:
                return rgb
            if not 0 <= light_index < len(directions):
                raise ValueError(
                    f"solar glare light_index {light_index} is unavailable for "
                    f"light preset {self._sim.light_preset!r}"
                )
            light_direction = directions[light_index]
        return self.image_effects.apply(
            rgb,
            camera_pos=self.cam_pos,
            camera_lookat=self.cam_lookat,
            camera_up=self.cam_up,
            fov=self.config["camera"]["fov"],
            scene_light_direction=light_direction,
        )

    def prepare_batch_render(self, force_render: bool = False) -> bool:
        should_render = (
            bool(force_render)
            or self._step_counter == 0
            or self._step_counter % self.update_interval == 0
        )
        if should_render and self._batch_cam is not None:
            cam_pos, cam_lookat, cam_up = self._update_batched_pose()
            self._batch_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
        return should_render

    def render_prepared_batch(self, force_render: bool = False) -> torch.Tensor:
        if self._batch_cam is None:
            return self.capture(force_render=force_render)
        rgb, *rest = self._batch_cam.render(rgb=True, force_render=force_render)
        self.latest_rgb = self._apply_image_effects(self._format_batched_rgb(rgb))
        self.recorder.record_batch(self.latest_rgb)
        self._step_counter += 1
        return self.latest_rgb

    def _format_rgb(self, rgb) -> torch.Tensor:
        if torch.is_tensor(rgb):
            rgb_tensor = rgb.detach().to(device=self.device).contiguous()
        else:
            rgb_array = np.asarray(rgb)
            if any(stride < 0 for stride in rgb_array.strides) or not rgb_array.flags["C_CONTIGUOUS"]:
                rgb_array = np.ascontiguousarray(rgb_array)
            rgb_tensor = torch.from_numpy(rgb_array).to(device=self.device)

        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = torch.clamp(rgb_tensor, 0, 255)
            if rgb_tensor.max().item() <= 1.0:
                rgb_tensor = rgb_tensor * 255.0
            rgb_tensor = rgb_tensor.to(torch.uint8)

        if rgb_tensor.dim() != 3:
            raise ValueError(
                f"Expected RGB image with 3 dims, got shape {tuple(rgb_tensor.shape)}"
            )
        if rgb_tensor.shape[-1] != 3 and rgb_tensor.shape[0] == 3:
            rgb_tensor = rgb_tensor.permute(1, 2, 0)
        if rgb_tensor.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got shape {tuple(rgb_tensor.shape)}"
            )
        if self.channels_first:
            rgb_tensor = rgb_tensor.permute(2, 0, 1).contiguous()
        if self.normalize:
            rgb_tensor = rgb_tensor.to(torch.float32) / 255.0
        return rgb_tensor

    def capture(self, force_render: bool = False) -> torch.Tensor:
        if self.latest_rgb is None:
            raise RuntimeError(
                "BatchedMountedCamera is not initialized. Call initialize(robot) first."
            )
        should_render = (
            bool(force_render)
            or self._step_counter == 0
            or self._step_counter % self.update_interval == 0
        )
        if not should_render:
            self._step_counter += 1
            return self.latest_rgb

        cam_pos = []
        cam_lookat = []
        cam_up = []
        if self._batch_cam is not None:
            cam_pos, cam_lookat, cam_up = self._update_batched_pose()
            self._batch_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
            rgb, *rest = self._batch_cam.render(rgb=True, force_render=True)
            self.latest_rgb = self._apply_image_effects(self._format_batched_rgb(rgb))
            self.recorder.record_batch(self.latest_rgb)
            self._step_counter += 1
            return self.latest_rgb

        rendered = []
        position, quaternion = self._ensure_batched_pose()
        pos_offset, lookat_offset, up_offset = self._get_offsets()
        for env_idx, cam in enumerate(self._cams):
            env_cam_pos, env_cam_lookat, env_cam_up = calculate_camera_pose(
                position[env_idx],
                quaternion[env_idx],
                pos_offset,
                lookat_offset,
                up_offset,
            )
            cam.set_pose(pos=env_cam_pos, lookat=env_cam_lookat, up=env_cam_up)
            rgb, *rest = cam.render(rgb=True)
            rendered.append(self._format_rgb(rgb))
            cam_pos.append(env_cam_pos)
            cam_lookat.append(env_cam_lookat)
            cam_up.append(env_cam_up)

        self.cam_pos = torch.stack(cam_pos, dim=0)
        self.cam_lookat = torch.stack(cam_lookat, dim=0)
        self.cam_up = torch.stack(cam_up, dim=0)
        self.latest_rgb = self._apply_image_effects(torch.stack(rendered, dim=0))
        self.recorder.record_batch(self.latest_rgb)
        self._step_counter += 1
        return self.latest_rgb

    def step(self):
        return self.capture()

    def reset(self):
        self._step_counter = 0
        self.recorder.reset()
        if self.latest_rgb is not None:
            self.latest_rgb.zero_()

    def on_episode_end(self, episode_count: int = 1, env_ids=None):
        self.recorder.on_episode_end(episode_count, env_ids=env_ids)

    def stop(self):
        self.recorder.stop()
