# Modified.
"""
Default simulation application settings.

This module defines default configurations for the simulation environment,
including scene setup, camera settings, and general application parameters.
"""

import genesis as gs

from pathlib import Path
_current_file_path = Path(__file__).resolve().parent
_ROOT_PATH = _current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'
ASSET_PATH_STR = str(_ASSET_PATH)

SPACE_BACKGROUND_SETTINGS = {
    "enable": True,
    "mesh_path": str(_ASSET_PATH / 'meshes' / 'env_sphere' / 'env_sphere.obj'),
    "texture_path": str(_ASSET_PATH / 'textures' / 'space_skybox_8k.png'),
    "scale": 150.0,
    "pos": (0.0, 0.0, 0.0),
}

EARTH_BACKGROUND_SETTINGS = {
    "enable": True,
    "mesh_path": str(_ASSET_PATH / 'meshes' / 'env_sphere' / 'env_sphere.obj'),
    "texture_path": str(_ASSET_PATH / 'textures' / 'earth_2k.png'),
    "scale": 60.0,
    "pos": (0.0, 0.0, -70.0),
}

APP_SETTINGS = {
    "seed": 0,
    "backend":gs.gpu,
    "precision": "32",
    "logging_level": "debug",
}

SCENE_SETTINGS = {
    "sim_options": gs.options.SimOptions(
        dt=3e-3,
        substeps=10,
    ),
    "rigid_options": gs.options.RigidOptions(
        gravity=(0, 0, 0),
        enable_collision=True,
        enable_joint_limit=True,
        enable_self_collision=False,
        enable_adjacent_collision=True,
        box_box_detection=True,
        constraint_timeconst=0.01,
    ),
    "viewer_options": gs.options.ViewerOptions(
        camera_pos=(-1.5, -2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    "vis_options": gs.options.VisOptions(
        show_world_frame=True,
        visualize_mpm_boundary=False,
    ),
    "show_viewer": True,
    "show_FPS": False,
}

TPV_CAM_SETTINGS = {
    "camera": {
        "res": (640, 480),
        "pos": (-3.0, -3.5, 2.5),
        "lookat": (0, 0.5, 0.5),
        "fov": 40,
        "far": 100.0,
        "GUI": True,
    },
    "enable_recording": True,  # 控制是否启用录制的flag
    "episodes_per_video": 10,
}

PARALLEL_ENV_SETTINGS = {
    "num_envs": 1,
    "env_spacing": (10.0, 10.0),
    "n_envs_per_row": None,
    "center_envs_at_origin": True,
}

__all__ = [
    'APP_SETTINGS',
    'EARTH_BACKGROUND_SETTINGS',
    'PARALLEL_ENV_SETTINGS',
    'SCENE_SETTINGS', 
    'SPACE_BACKGROUND_SETTINGS',
    'TPV_CAM_SETTINGS',
]
