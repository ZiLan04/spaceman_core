"""Configuration for the satellite-mounted Franka Panda URDF model."""

from pathlib import Path

import genesis as gs
import torch


_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
_ASSET_PATH = _ROOT_PATH / "src" / "assets"
_SATELLITE_PANDA_URDF = (
    _ASSET_PATH
    / "urdf"
    / "satellite_combine_panda"
    / "satellite_combine_panda.urdf"
)


SATELLITE_PANDA_CONFIG = {
    "initial_dofs": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    "control": {
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        "kv": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        "force_range_min": [-87, -87, -87, -87, -12, -12, -12, -20, -20],
        "force_range_max": [87, 87, 87, 87, 12, 12, 12, 20, 20],
    },
}

SATELLITE_PANDA_PARAMS = {
    "name": "satellite_panda",
    "end_effector": "panda_hand",
    "base": "satellite_base_link",
    "pid_apply_link": "satellite_base_link",
    "config": SATELLITE_PANDA_CONFIG,
    "joints": (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ),
    "motor": 7,
    "finger": 9,
    "gripper_revolute": [1, 1],
    "gripper_force_scale": 20.0,
    "finger_open": [0.04, 0.04],
    "finger_close": [0.0, 0.0],
    "ik_params": {
        "smooth_factor": 0.3,
        "max_joint_change": 0.05,
    },
    "path": _SATELLITE_PANDA_URDF,
}

SATELLITE_PANDA_PID = {
    "name": "satellite_panda",
    "enable_pid": False,
    "P": [800, 800, 800, 20, 20, 20],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [200, 200, 200, 10, 10, 10],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [120, 120, 120, 10, 10, 10],
}

SATELLITE_PANDA_CAMERA = {
    "name": "satellite_panda",
    "wrist_camera": False,
    "camera": {
        "res": (640, 480),
        "pos": (-1.0, -1.0, -1.0),
        "lookat": (0.0, 0.0, 0.0),
        "fov": 70,
        "GUI": False,
    },
    "enable_recording": False,
    "end_effector_link": "panda_grasptarget",
    "pos_offset": torch.tensor([0.21, 0.0, -0.36], dtype=torch.float32),
    "lookat_offset": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
    "up_offset": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
}


def _make_satellite_panda():
    return {
        "morph": gs.morphs.URDF(
            file=_SATELLITE_PANDA_URDF.as_posix(),
            pos=(0.5, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            merge_fixed_links=False,
            scale=3e-1,
            fixed=False,
        ),
        "material": gs.materials.Rigid(gravity_compensation=1.0),
    }


__all__ = [
    "SATELLITE_PANDA_CONFIG",
    "SATELLITE_PANDA_PARAMS",
    "SATELLITE_PANDA_PID",
    "SATELLITE_PANDA_CAMERA",
    "_make_satellite_panda",
]
