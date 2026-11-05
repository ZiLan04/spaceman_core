from configs.config_satellite import *
from configs.config_satellite1 import *
from configs.config_wholesat import *
from configs.config_franka import *
from configs.config_satellite_panda import *
from configs.registry import (
    get_legacy_camera as get_registered_camera,
    get_legacy_params as get_registered_params,
    get_legacy_pid as get_registered_pid,
    get_registered_asset,
    registered_ids,
)

_ASSET_FACTORIES = {
    "franka": _make_franka_urdf,
    "satellite": _make_satellite,
    "satellite1": _make_satellite1,
    "wholesat": _make_wholesat,
    "starlink": lambda: get_registered_asset("starlink"),
    "satellite_part": _make_satellite_part,
    "franka_mjcf": _make_franka_mjcf,
    "franka_merge": lambda: get_registered_asset("franka_merge"),
    "satellite_panda": _make_satellite_panda,
}

_ASSET_CONFIG_FACTORIES = {
    "satellite": SATELLITE_PARAMS,
    "satellite_part": SATELLITE_PART_PARAMS,
    "satellite1": SATELLITE1_PARAMS,
    "wholesat": WHOLESAT_PARAMS,
    "franka": FRANKA_PARAMS,
    "satellite_panda": SATELLITE_PANDA_PARAMS,
}

_ASSET_PID_FACTORIES = {
    "satellite": SATELLITE_PID,
    "satellite_part": SATELLITE_PART_PID,
    "satellite1": SATELLITE1_PID,
    "wholesat": WHOLESAT_PID,
    "franka": FRANKA_PID,
    "satellite_panda": SATELLITE_PANDA_PID,
}

_ASSET_WristCamera_FACTORIES = {
    "satellite": SATELLITE_CAMERA,
    "satellite_part": SATELLITE_PART_CAMERA,
    "satellite1": SATELLITE1_CAMERA,
    "wholesat": WHOLESAT_CAMERA,
    "franka": FRANKA_CAMERA,
    "satellite_panda": SATELLITE_PANDA_CAMERA,
}

def get_asset(name: str) -> dict:
    """Build asset dict lazily after gs.init()."""
    if name in _ASSET_FACTORIES:
        return _ASSET_FACTORIES[name]()
    try:
        return get_registered_asset(name)
    except KeyError:
        available = sorted(set(_ASSET_FACTORIES) | set(registered_ids()))
        raise KeyError(f"Unknown asset '{name}'. Available: {available}") from None

def get_configs(name: str) ->dict:
    if name in _ASSET_CONFIG_FACTORIES:
        config = _ASSET_CONFIG_FACTORIES[name]
        return config() if callable(config) else config
    try:
        return get_registered_params(name)
    except KeyError:
        available = sorted(set(_ASSET_CONFIG_FACTORIES) | set(registered_ids()))
        raise KeyError(f"Unknown asset '{name}'. Available: {available}") from None

def get_pid(name: str) ->dict:
    if name in _ASSET_PID_FACTORIES:
        config = _ASSET_PID_FACTORIES[name]
        return config() if callable(config) else config
    try:
        return get_registered_pid(name)
    except KeyError:
        available = sorted(set(_ASSET_PID_FACTORIES) | set(registered_ids()))
        raise KeyError(f"Unknown asset '{name}'. Available: {available}") from None

def get_wrist_camera(name: str) ->dict:
    if name in _ASSET_WristCamera_FACTORIES:
        config = _ASSET_WristCamera_FACTORIES[name]
        return config() if callable(config) else config
    try:
        return get_registered_camera(name)
    except KeyError:
        available = sorted(set(_ASSET_WristCamera_FACTORIES) | set(registered_ids()))
        raise KeyError(f"Unknown asset '{name}'. Available: {available}") from None

__all__ = [
    'get_asset',
    'get_configs',
    'get_pid',
    'get_wrist_camera',
]
