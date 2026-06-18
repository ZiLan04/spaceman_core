"""
Generic link-mounted camera sensor.

This reuses the existing pose-offset camera convention so cameras can be
attached to arbitrary robot links, such as a satellite base link.
"""

import sys
from pathlib import Path

import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from controllers.backend import Backend
from envs.genesis_env import GenesisSim
from utils.utils import calculate_camera_pose


DEFAULT_MOUNTED_CAM_SETTINGS = {
    "camera": {
        "res": (640, 480),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 70,
        "GUI": False,
    },
    "mount_link": "base_link",
    "pos_offset": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
    "lookat_offset": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "up_offset": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
}


class MountedCamera(Backend):
    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_MOUNTED_CAM_SETTINGS
        self._scene = GenesisSim().scene
        self.device = GenesisSim().device
        self.datatype = torch.float32

        self._cam = self._scene.add_camera(**self.config["camera"])
        self._robot = None
        self.mount_link = None
        self.cam_pos = None
        self.cam_lookat = None
        self.cam_up = None

    @property
    def cam(self):
        return self._cam

    def initialize(self, robot=None):
        self._robot = robot
        if robot is None or not hasattr(robot, "robot"):
            raise AttributeError(
                "MountedCamera requires a Robot wrapper with a 'robot' Genesis entity."
            )
        mount_link_name = self.config.get(
            "mount_link", DEFAULT_MOUNTED_CAM_SETTINGS["mount_link"]
        )
        self.mount_link = robot.robot.get_link(mount_link_name)

    def step(self):
        if self._robot is None:
            raise RuntimeError("Camera not bound to robot. Call initialize(robot) first.")

        position = torch.as_tensor(
            self.mount_link.get_pos(), dtype=self.datatype, device=self.device
        )
        quaternion = torch.as_tensor(
            self.mount_link.get_quat(), dtype=self.datatype, device=self.device
        )

        if position.dim() > 1:
            position = position[0]
        if quaternion.dim() > 1:
            quaternion = quaternion[0]

        pos_offset = self.config.get(
            "pos_offset", DEFAULT_MOUNTED_CAM_SETTINGS["pos_offset"]
        )
        lookat_offset = self.config.get(
            "lookat_offset", DEFAULT_MOUNTED_CAM_SETTINGS["lookat_offset"]
        )
        up_offset = self.config.get(
            "up_offset", DEFAULT_MOUNTED_CAM_SETTINGS["up_offset"]
        )

        self.cam_pos, self.cam_lookat, self.cam_up = calculate_camera_pose(
            position, quaternion, pos_offset, lookat_offset, up_offset
        )
        self.cam.set_pose(pos=self.cam_pos, lookat=self.cam_lookat, up=self.cam_up)
        rgb, *rest = self.cam.render(rgb=True)
        return rgb

    def reset(self):
        pass

    def stop(self):
        pass
