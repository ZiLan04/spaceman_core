"""
Configuration for satellite URDF model.
"""
import sys
from pathlib import Path
import genesis as gs

current_file_path = Path(__file__).resolve().parent
sys.path.insert(0, str(current_file_path))

to_posix = lambda p: p.as_posix()

_ROOT_PATH = current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_STARLINK_PATHS = {
    'urdf': _ASSET_PATH / 'urdf' / 'starlink' / 'urdf' / 'starlink.urdf',
}

STARLINK_PARAMS = {
    "name": "starlink",
    "base": "base_star_link",
    "grasp_link": "base_star_link",
    "grasp_offset_local": [-0.36, 0.07, 0.23],
    "grasp_marker_offset_local": [0.0, 0.0, 0.0],
    "path": _STARLINK_PATHS['urdf'],
}

STARLINK_PID = {
    "name": "starlink",
    "enable_pid": False,  # 控制是否启用PID控制器
    "P": [0, 0, 0, 0, 0, 0],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [0, 0, 0, 0, 0, 0],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [
        None,        # x
        None,        # y
        None,        # z
        None,        # roll
        None,        # pitch
        None         # yaw
    ]
}

STARLINK_CAMERA = {
    "name": "starlink",
    "wrist_camera": False,
    "camera": {
        "res": (640, 480),
        "pos": (0.0, 0.0, 0.0),  # 将在机器人初始化时设置
        "lookat": (0.0, 0.0, 0.0),  # 将在机器人初始化时设置
        "fov": 40,
        "GUI": False,
    },
    "enable_recording": False  # 控制是否启用录制的flag
}

def _make_starlink():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_STARLINK_PATHS['urdf']),
            pos=(-2.0, 0.0, 0.5),
            euler=(0.0, 0.0, 180.0),
            scale=7e-1,
            # merge_fixed_links=False,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }

__all__ = [
    'STARLINK_PARAMS',
    'STARLINK_PID',
    'STARLINK_CAMERA',
    '_make_starlink',
]
