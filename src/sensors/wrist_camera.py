"""
可给机械臂添加手腕部相机,需要在GenesisSim()初始化之后创建
"""

from copy import deepcopy
import torch

# Extension APIs
import torch
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from utils.utils import calculate_camera_pose
from controllers.backend import Backend

# 默认配置（向后兼容）
DEFAULT_WRIST_CAM_SETTINGS = {
    "camera": {
        "res": (640, 480),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 70,
        "GUI": True
    },
    "end_effector_link": "panda_grasptarget",
    "pos_offset": torch.tensor([-0.08, 0.0, 0.12], dtype=torch.float32),
    "lookat_offset": torch.tensor([0.05, -1.0, 0.0], dtype=torch.float32),
    "up_offset": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
}

class WristCamera(Backend):
    """
    可配置的腕部相机传感器
    
    Args:
        config (dict, optional): 相机配置字典，包含以下键：
            - camera: 相机基本参数 (res, pos, lookat, fov, GUI)
            - end_effector_link: 末端执行器链接名称
            - pos_offset: 相机位置偏移
            - lookat_offset: 相机观察点偏移  
            - up_offset: 相机上方向偏移
        如果不提供 config，则使用默认配置
    """
    def __init__(self, config=None):
        # 使用传入的配置或默认配置
        self.config = deepcopy(
            config if config is not None else DEFAULT_WRIST_CAM_SETTINGS
        )
        self.config["camera"]["res"] = tuple(
            int(value) for value in self.config["camera"]["res"]
        )
        
        # Add wrist camera to the scene
        self._scene = GenesisSim().scene
        self.device = GenesisSim().device
        self.datatype = torch.float32

        # Use private attribute to avoid property conflict
        self._cam = self._scene.add_camera(**self.config["camera"])
        self._robot = None
        self.end_effector = None
        self.cam_pos = None
        self.cam_lookat = None
        self.cam_up = None

    @property
    def cam(self):
        return self._cam
    
    @property
    def pos(self):
        return self.cam_pos
    
    @property
    def lookat(self):
        return self.cam_lookat

    def initialize(self, robot=None):
        self._robot = robot
        
        # Access Genesis robot entity through Robot wrapper (robot.robot has get_link())
        if hasattr(robot, 'robot'):
            genesis_robot = robot.robot
            end_effector_link = self.config.get("end_effector_link", DEFAULT_WRIST_CAM_SETTINGS["end_effector_link"])
            self.end_effector = genesis_robot.get_link(end_effector_link)
        else:
            raise AttributeError(f"Robot object {robot} does not have a 'robot' attribute containing the Genesis entity")

    def step(self):
        """更新相机位置并渲染图像"""
        if self._robot is None:
            raise RuntimeError("Camera not bound to robot. Call bind() first.")
        
        # 获取机器人末端执行器位置和姿态
        position = self.end_effector.get_pos()
        quaternion = self.end_effector.get_quat()
        
        # 转换为张量
        position = torch.tensor(position, dtype=self.datatype, device=self.device)
        quaternion = torch.tensor(quaternion, dtype=self.datatype, device=self.device)

        # A single Genesis camera can only render one env at a time. When the
        # robot state is batched, default to env 0 for wrist-camera rendering.
        if position.dim() > 1:
            position = position[0]
        if quaternion.dim() > 1:
            quaternion = quaternion[0]
        
        # 使用配置中的偏移参数计算相机位姿
        pos_offset = self.config.get("pos_offset", DEFAULT_WRIST_CAM_SETTINGS["pos_offset"])
        lookat_offset = self.config.get("lookat_offset", DEFAULT_WRIST_CAM_SETTINGS["lookat_offset"])
        up_offset = self.config.get("up_offset", DEFAULT_WRIST_CAM_SETTINGS["up_offset"])
        
        # 计算相机位姿
        self.cam_pos, self.cam_lookat, self.cam_up = calculate_camera_pose(
            position, quaternion, pos_offset, lookat_offset, up_offset
        )
        
        # 更新相机位姿
        self.cam.set_pose(pos=self.cam_pos, lookat=self.cam_lookat, up=self.cam_up)
        
        # 渲染图像
        rgb, *rest = self.cam.render(
            rgb=True,
            # depth        = True,
            # segmentation = True,
        )
        return rgb

    def reset(self):
        """无需reset"""
        pass

    def stop(self):
        """无需stop"""
        pass
