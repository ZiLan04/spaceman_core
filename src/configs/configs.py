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
    "euler": (0.0, 0.0, 0.0),
}

EARTH_BACKGROUND_SETTINGS = {
    "enable": True,
    "mesh_path": str(_ASSET_PATH / 'meshes' / 'env_sphere' / 'env_sphere.obj'),
    "texture_path": str(_ASSET_PATH / 'textures' / 'earth_2k.png'),
    "scale": 60.0,
    "pos": (0.0, 0.0, -70.0),
    "euler": (0.0, 0.0, 0.0),
}

APP_SETTINGS = {
    "seed": 0,
    "backend":gs.gpu,
    "precision": "32",
    "logging_level": "debug",
}

LIGHT_SETTINGS = {
    "enabled": True,
    "show_world_frame": True,
    "active_preset": "front_right",
    "ambient_light": (0.08, 0.08, 0.08),
    "background_color": (0.04, 0.08, 0.12),
    "shadow": True,
    "presets": {
        "front": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, -1.0),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.0,
                ),
            ),
        },
        "back": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, 1.0),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.0,
                ),
            ),
        },
        "left": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(1.0, 0.0, -0.2),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.2,
                ),
            ),
        },
        "right": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(-1.0, 0.0, -0.2),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.2,
                ),
            ),
        },
        "top": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, -1.0),
                    color=(1.0, 1.0, 1.0),
                    intensity=6.0,
                ),
            ),
        },
        "front_left": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(0.7, 0.5, -0.5),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.5,
                ),
            ),
        },
        "front_right": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(-0.7, 0.5, -0.5),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.5,
                ),
            ),
        },
        "dual_side": {
            "lights": (
                gs.options.vis.DirectionalLight(
                    dir=(0.8, 0.3, -0.4),
                    color=(1.0, 1.0, 1.0),
                    intensity=4.4,
                ),
                gs.options.vis.DirectionalLight(
                    dir=(-0.8, -0.2, -0.3),
                    color=(0.85, 0.9, 1.0),
                    intensity=2.2,
                ),
            ),
        },
    },
}


def make_light_vis_options(preset_name: str | None = None):
    active_preset = preset_name or LIGHT_SETTINGS.get("active_preset", "front")
    presets = LIGHT_SETTINGS.get("presets", {})
    if active_preset not in presets:
        raise KeyError(
            f"Unknown rasterizer light preset '{active_preset}'. "
            f"Available presets: {sorted(presets.keys())}"
        )

    preset = presets[active_preset]
    lights = preset.get("lights", ())
    if not LIGHT_SETTINGS.get("enabled", True):
        lights = ()

    return gs.options.VisOptions(
        show_world_frame=bool(LIGHT_SETTINGS.get("show_world_frame", True)),
        visualize_mpm_boundary=False,
        shadow=bool(LIGHT_SETTINGS.get("shadow", True)),
        ambient_light=tuple(LIGHT_SETTINGS.get("ambient_light", (0.1, 0.1, 0.1))),
        background_color=tuple(
            LIGHT_SETTINGS.get("background_color", (0.04, 0.08, 0.12))
        ),
        lights=lights,
    )

SCENE_SETTINGS = {
    "sim_options": gs.options.SimOptions(
        dt=3e-3,
        substeps=10,
    ),
    # BatchRenderer lets RL cameras render all parallel envs in one batched call.
    # TPV/debug cameras should set camera.debug=True so they do not enter this
    # renderer, because BatchRenderer requires all non-debug cameras to share
    # the same resolution.
    #"renderer": gs.renderers.BatchRenderer(use_rasterizer=True),
    "renderer":None,
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
    "vis_options": make_light_vis_options(),
    "show_viewer": True,
    "show_FPS": False,
}

TPV_CAM_SETTINGS = {
    "camera": {
        "res": (640, 480),
        "pos": (-3.0, -3.5, 2.5),
        "lookat": (0, 0.5, 0.5),
        "fov": 40,
        "far": 200.0,
        "GUI": False,
        "debug": True,
    },
    "enable_recording": True,  # TPV debug recording renders every sim step; RL recording is configured per rl_obs.
    "episodes_per_video": 1,
}

PARALLEL_ENV_SETTINGS = {
    "num_envs": 1,
    "env_spacing": (10.0, 10.0),
    "n_envs_per_row": None,
    "center_envs_at_origin": True,
    "background_anchor_env": "rendered_env",
}

__all__ = [
    'APP_SETTINGS',
    'EARTH_BACKGROUND_SETTINGS',
    'PARALLEL_ENV_SETTINGS',
    'LIGHT_SETTINGS',
    'SCENE_SETTINGS', 
    'SPACE_BACKGROUND_SETTINGS',
    'TPV_CAM_SETTINGS',
    'make_light_vis_options',
]
