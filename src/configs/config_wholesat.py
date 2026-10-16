"""
Configuration for the wholesat URDF model.
"""
import sys
from pathlib import Path

import genesis as gs

current_file_path = Path(__file__).resolve().parent
sys.path.insert(0, str(current_file_path))

to_posix = lambda p: p.as_posix()

_ROOT_PATH = current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_WHOLESAT_PATHS = {
    'urdf': _ASSET_PATH / 'urdf' / 'wholesat.SLDASM' / 'urdf' / 'wholesat.SLDASM.urdf',
}

WHOLESAT_PARAMS = {
    "name": "wholesat",
    "base": "base_link",
    "grasp_link": "base_link",
    "grasp_offset_local": [-0.25, 0.0, 0.2],
    "path": _WHOLESAT_PATHS['urdf'],
}

WHOLESAT_PID = {
    "name": "wholesat",
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

WHOLESAT_CAMERA = {
    "name": "wholesat",
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


def _make_wholesat():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_WHOLESAT_PATHS['urdf']),
            pos=(-2.0, 0.0, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=5e-1,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }


__all__ = [
    'WHOLESAT_PARAMS',
    'WHOLESAT_PID',
    'WHOLESAT_CAMERA',
    '_make_wholesat',
]
