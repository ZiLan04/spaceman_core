from configs.config_satellite import *
from configs.config_satellite1 import *
from configs.config_wholesat import *
from configs.config_franka import *
from configs.config_starlink import *
from configs.config_starlink_manipulator import *
from configs.config_satellite_panda import *

_ASSET_FACTORIES = {
    "franka": _make_franka_urdf,
    "satellite": _make_satellite,
    "satellite1": _make_satellite1,
    "wholesat": _make_wholesat,
    "starlink": _make_starlink,
    "satellite_part": _make_satellite_part,
    "franka_mjcf": _make_franka_mjcf,
    "franka_merge":_make_starlink_manipulator,
    "satellite_panda": _make_satellite_panda,
}

_ASSET_CONFIG_FACTORIES = {
    "satellite": SATELLITE_PARAMS,
    "satellite_part": SATELLITE_PART_PARAMS,
    "satellite1": SATELLITE1_PARAMS,
    "wholesat": WHOLESAT_PARAMS,
    "starlink": STARLINK_PARAMS,
    "franka": FRANKA_PARAMS,
    "franka_merge": FRANKA_S_Q_PARAMS,
    "satellite_panda": SATELLITE_PANDA_PARAMS,
}

_ASSET_PID_FACTORIES = {
    "satellite": SATELLITE_PID,
    "satellite_part": SATELLITE_PART_PID,
    "satellite1": SATELLITE1_PID,
    "wholesat": WHOLESAT_PID,
    "starlink": STARLINK_PID,
    "franka": FRANKA_PID,
    "franka_merge": FRANKA_S_Q_PID,
    "satellite_panda": SATELLITE_PANDA_PID,
}

_ASSET_WristCamera_FACTORIES = {
    "satellite": SATELLITE_CAMERA,
    "satellite_part": SATELLITE_PART_CAMERA,
    "satellite1": SATELLITE1_CAMERA,
    "wholesat": WHOLESAT_CAMERA,
    "starlink": STARLINK_CAMERA,
    "franka": FRANKA_CAMERA,
    "franka_merge": FRANKA_S_Q_CAMERA,
    "satellite_panda": SATELLITE_PANDA_CAMERA,
}

def get_asset(name: str) -> dict:
    """Build asset dict lazily after gs.init()."""
    if name not in _ASSET_FACTORIES:
        raise KeyError(f"Unknown asset '{name}'. Available: {list(_ASSET_FACTORIES)}")
    return _ASSET_FACTORIES[name]()

def get_configs(name: str) ->dict:
    if name not in _ASSET_CONFIG_FACTORIES:
        raise KeyError(f"Unknown asset '{name}'. Available: {list(_ASSET_CONFIG_FACTORIES)}")
    return _ASSET_CONFIG_FACTORIES[name]

def get_pid(name: str) ->dict:
    if name not in _ASSET_PID_FACTORIES:
        raise KeyError(f"Unknown asset '{name}'. Available: {list(_ASSET_PID_FACTORIES)}")
    return _ASSET_PID_FACTORIES[name]

def get_wrist_camera(name: str) ->dict:
    if name not in _ASSET_WristCamera_FACTORIES:
        raise KeyError(f"Unknown asset '{name}'. Available: {list(_ASSET_WristCamera_FACTORIES)}")
    return _ASSET_WristCamera_FACTORIES[name]

__all__ = [
    'get_asset',
    'get_configs',
    'get_pid',
    'get_wrist_camera',
]
