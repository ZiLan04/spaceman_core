
"""
Configuration for starlink_space_manipulator_merged URDF model.
"""
from pathlib import Path
import genesis as gs
import torch
to_posix = lambda p: p.as_posix()

_current_file_path = Path(__file__).resolve().parent
_ROOT_PATH = _current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_STARLINK_MANIPULATOR_URDF = (
    _ASSET_PATH / 'urdf' / 'starlink_combine_starlink_space_manipulator_merged.SLDASM9' 
    / 'starlink_combine_starlink_space_manipulator_merged.SLDASM9.urdf'
)

FRANKA_S_Q_CONFIG = {
    "initial_dofs": [0.14, -0.55, -1.36, 0.81, 1.31, -0.52, 0.0, 0.0],
    "control": {
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 100, 100],
        "kv": [450, 450, 350, 350, 200, 200, 10, 10],
        "force_range_min": [-87, -87, -87, -87, -12, -12, -100, -100],
        "force_range_max": [87, 87, 87, 87, 12, 12, 100, 100],
    }
}

IK_PARAMS = {
    "smooth_factor": 0.3,
    "max_joint_change": 0.05,
}

FRANKA_S_Q_PARAMS = {
    "name": "franka_merge",
    "end_effector": "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link",
    "base": "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_base_link",
    "pid_apply_link": "starlink_base_star_link",
    "config": FRANKA_S_Q_CONFIG,
    "joints": (
        # "root_joint",
        # "qf_space_manipulator_Shoulder_link_1_yaw",
        # "qf_space_manipulator_Upper_arm_Link_1_roll",
        # "qf_space_manipulator_Mid_arm_Link_1_roll",
        # "qf_space_manipulator_Upper_wrist_Link_1_yaw",
        # "qf_space_manipulator_Upper_wrist_Link_2_roll",
        # "qf_space_manipulator_2F-Body_Link_pitch", 
        # # hand
        # "qf_space_manipulator_Finger1_2_Link_roll",
        # "qf_space_manipulator_Finger3_2_Link_roll", # all gripper joints mimic this
        # "qf_space_manipulator_Finger3_1_Link_roll",
        # "qf_space_manipulator_Finger1_1_Link_roll", # the only positive mutiplier
        # "qf_space_manipulator_Finger4_2_Link_roll",
        # "qf_space_manipulator_Finger4_1_Link_roll",
        #"root_joint",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Shoulder_link_1_yaw",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_arm_Link_1_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Mid_arm_Link_1_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_wrist_Link_1_yaw",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_wrist_Link_2_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link_yaw", 
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link_roll",
       
    ),
    "motor": 6,
    "finger": 8,
    "gripper_revolute": [1, 1],
    "gripper_force_scale": 1.0,
    
    "finger_open": [0, 0],
    "finger_close": [0.4, 0.4],
    "ik_params": IK_PARAMS,
    "path": _STARLINK_MANIPULATOR_URDF,
}

FRANKA_S_Q_PID = {
    "name": "franka_merge",
    "enable_pid": True,
    "P": [800, 800, 0, 20, 20, 20],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [200, 200, 0, 10, 10, 10],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [
        120,       # x
        120,       # y
        0,         # z
        10,         # roll
        10,         # pitch
        10          # yaw
    ]
}

FRANKA_S_Q_CAMERA = {
    "name": "franka",
    # Regular debug/recording camera. RL observations are controlled by
    # rl_obs.enabled below and do not require this to be True.
    "wrist_camera": False,
    "camera": {
        # Keep a small training resolution here because the current batched RL
        # camera implementation reuses this value directly.
        "res": (84, 84),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 60,
        "GUI": False,
    },
    "enable_recording": False,  # 训练时先关闭录制，避免额外渲染负担
    "episodes_per_video": 1,
    "end_effector_link": "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link",
    "pos_offset": torch.tensor([0.05, -0.08, -0.07], dtype=torch.float32),
    "lookat_offset": torch.tensor([0.05, -1.0, -0.1], dtype=torch.float32),
    "up_offset": torch.tensor([0.15, -1.0, -1.0], dtype=torch.float32),
    "base_camera": {
        # Regular debug/recording base camera. RL base observations are
        # controlled by base_camera.rl_obs.enabled below.
        "enabled": False,
        "camera": {
            # Match the RL visual pipeline resolution first; we can increase it later
            # after the mounting pose is stable.
            "res": (84, 84),
            "pos": (-1, -1, -1),
            "lookat": (0, 0, 0),
            "fov": 82,
            "GUI": False,
        },
        "enable_recording": False,
        "episodes_per_video": 1,
        "mount_link": "starlink_base_star_link",
        # Camera mounted on the left-rear side of the satellite base, looking
        # toward the arm workspace in the right-front / forward-up direction.
        "pos_offset": torch.tensor([0.25, 0.5, 0.4], dtype=torch.float32),
        "lookat_offset": torch.tensor([-0.5, 0.55, 0.75], dtype=torch.float32),
        "up_offset": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "rl_obs": {
            "enabled": True,
            "channels_first": True,
            "normalize": False,
            "enable_recording": True,
            "record_env_idx": 0,
            "record_fps": 30,
            "episodes_per_video": 10,
            "use_batch_renderer": True,
            "update_interval": 1,
        },
    },
    "rl_obs": {
        "enabled": True,
        # SB3 expects image tensors in (C, H, W) for MultiInputPolicy.
        "channels_first": True,
        # Keep uint8 in the env and normalize in the extractor by default.
        "normalize": False,
        "enable_recording": True,
        "record_env_idx": 0,
        "record_fps": 30,
        "episodes_per_video": 50,
        "use_batch_renderer": True,
        # Render every control step first; if training is slow, try 2 or 4.
        "update_interval": 1,
    },
}

def _make_starlink_manipulator():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_STARLINK_MANIPULATOR_URDF),
            pos=(0.5, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            merge_fixed_links=False,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }

__all__ = [
    'FRANKA_S_Q_PARAMS',
    'FRANKA_S_Q_PID',
    'FRANKA_S_Q_CAMERA',
    '_make_starlink_manipulator',
]
