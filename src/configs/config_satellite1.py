"""
Configuration for the satellite1 URDF model.

This config keeps the original exported URDF package directory untouched and
registers an ASCII alias used by the Python side of the simulator.
"""
import sys
from pathlib import Path
import genesis as gs

current_file_path = Path(__file__).resolve().parent
sys.path.insert(0, str(current_file_path))

to_posix = lambda p: p.as_posix()

_ROOT_PATH = current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_SATELLITE1_PATHS = {
    'urdf': _ASSET_PATH / 'urdf' / '卫星1号urdf.SLDASM' / 'urdf' / '卫星1号urdf.SLDASM.urdf',
}

SATELLITE1_PARAMS = {
    "name": "satellite1",
    "base": "base_link",
    "grasp_link": "base_link",
    "grasp_offset_local": [-0.56, 0.115, 0.3],
    "path": _SATELLITE1_PATHS['urdf'],
}

SATELLITE1_PID = {
    "name": "satellite1",
    "enable_pid": False,
    "P": [0, 0, 0, 0, 0, 0],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [0, 0, 0, 0, 0, 0],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [
        None,
        None,
        None,
        None,
        None,
        None,
    ]
}

SATELLITE1_CAMERA = {
    "name": "satellite1",
    "wrist_camera": False,
    "camera": {
        "res": (640, 480),
        "pos": (0.0, 0.0, 0.0),
        "lookat": (0.0, 0.0, 0.0),
        "fov": 40,
        "GUI": False,
    },
    "enable_recording": False,
}

def _make_satellite1():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_SATELLITE1_PATHS['urdf']),
            pos=(-2.0, 0.0, 0.5),
            euler=(0.0, 0.0, 90.0),
            scale=3e-1,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }

__all__ = [
    'SATELLITE1_PARAMS',
    'SATELLITE1_PID',
    'SATELLITE1_CAMERA',
    '_make_satellite1',
]
