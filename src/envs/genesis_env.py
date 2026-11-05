import sys
import math
import torch
from threading import Lock
import genesis as gs
import numpy as np

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from configs.configs import (
    APP_SETTINGS,
    EARTH_BACKGROUND_SETTINGS,
    LIGHT_SETTINGS,
    PARALLEL_ENV_SETTINGS,
    SCENE_SETTINGS,
    SPACE_BACKGROUND_SETTINGS,
    TPV_CAM_SETTINGS,
    add_batch_renderer_lights,
    get_light_directions,
    make_light_vis_options,
)
from sensors.timer import SimpleTimer
from utils.utils import with_camera, generate_filename

_RECORDINGS_DIR = current_file_path.parent / "recordings"
_BACKGROUND_NULL_POS = (1.0e6, 1.0e6, 1.0e6)

class GenesisSim:
    """
    GenesisSim is a singleton class (there is only one object instance at any given time) that is responsible 
    for orchestrating the entire simulation: it sets up the scene, constructs robots and map objects 
    (the actors), and provides a set of APIs to manage the simulation world.
    """
    # The object instance
    _instance = None
    _is_initialized = False
    _init_config = None

    # Lock for safe multi-threading
    _lock: Lock = Lock()

    def __init__(
        self,
        num_envs: int = None,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
        light_preset: str = None,
    ):
        """
        Initialize the GenesisSim environment.
        """
        has_explicit_parallel_config = any(
            value is not None
            for value in (
                num_envs,
                env_spacing,
                n_envs_per_row,
                center_envs_at_origin,
            )
        )

        # If we already have an instance of the PegasusInterface, do not overwrite it!
        if GenesisSim._is_initialized:
            if has_explicit_parallel_config:
                requested_config = self._normalize_init_config(
                    num_envs=num_envs,
                    env_spacing=env_spacing,
                    n_envs_per_row=n_envs_per_row,
                    center_envs_at_origin=center_envs_at_origin,
                )
                if GenesisSim._init_config != requested_config:
                    raise ValueError(
                        "GenesisSim is a singleton and has already been initialized with "
                        f"{GenesisSim._init_config}. Requested {requested_config}. "
                        "Please reuse the existing instance or reset the singleton first."
                    )
            return
        
        GenesisSim._is_initialized = True
        self._build_config = self._normalize_init_config(
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
        )
        GenesisSim._init_config = dict(self._build_config)

        self.light_preset = light_preset or LIGHT_SETTINGS.get("active_preset", "front")
        self.light_directions = get_light_directions(self.light_preset)
        SCENE_SETTINGS["vis_options"] = make_light_vis_options(self.light_preset)

        # Initialize the simulation
        gs.init(**APP_SETTINGS)

        try:
            print(
                "[DEBUG] APP_SETTINGS: "
                f"seed={APP_SETTINGS.get('seed', None)} "
                f"precision={APP_SETTINGS.get('precision', None)} "
                f"backend={APP_SETTINGS.get('backend', None)} "
                f"num_envs={self.num_envs}"
            )
        except Exception:
            pass

        self.scene = gs.Scene(**SCENE_SETTINGS)
        if isinstance(self.scene.renderer_options, gs.renderers.BatchRenderer):
            batch_light_count = add_batch_renderer_lights(
                self.scene,
                self.light_preset,
            )
            print(
                f"[DEBUG] Added {batch_light_count} BatchRenderer light(s) "
                f"from preset '{self.light_preset}'."
            )
        self.space_background = None
        self.earth_background = None
        self._add_space_background()
        self._add_earth_background()
        self.rigid = self.scene.sim.rigid_solver
        self.viewer = self.scene.viewer
        if torch.cuda.is_available():
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")
        self.datatype = torch.float32
        
        # Add third-person view (TPV) camera only when explicitly recording.
        self.camera_enable_recording = with_camera(TPV_CAM_SETTINGS)
        self.cam = None
        if self.camera_enable_recording:
            self.cam = self.scene.add_camera(**TPV_CAM_SETTINGS["camera"])
        self.camera_episodes_per_video = max(
            1, int(TPV_CAM_SETTINGS.get("episodes_per_video", 5))
        )
        self._camera_recorded_episode_count = 0
        self._camera_video_segment_index = 0
        
        # plane = self.scene.add_entity(
        #     gs.morphs.Plane(
        #         pos=(0, 0, -100.0),
        #     ),
        # )

    @staticmethod
    def _normalize_init_config(
        num_envs,
        env_spacing,
        n_envs_per_row,
        center_envs_at_origin,
    ):
        default_num_envs = PARALLEL_ENV_SETTINGS.get("num_envs", 1)
        default_env_spacing = PARALLEL_ENV_SETTINGS.get("env_spacing", (1.0, 1.0))
        default_n_envs_per_row = PARALLEL_ENV_SETTINGS.get("n_envs_per_row", None)
        default_center_envs_at_origin = PARALLEL_ENV_SETTINGS.get(
            "center_envs_at_origin", True
        )
        default_background_anchor_env = PARALLEL_ENV_SETTINGS.get(
            "background_anchor_env", "rendered_env"
        )

        resolved_num_envs = default_num_envs if num_envs is None else num_envs
        resolved_env_spacing = (
            default_env_spacing if env_spacing is None else env_spacing
        )
        resolved_n_envs_per_row = (
            default_n_envs_per_row if n_envs_per_row is None else n_envs_per_row
        )
        resolved_center_envs_at_origin = (
            default_center_envs_at_origin
            if center_envs_at_origin is None
            else center_envs_at_origin
        )

        normalized_num_envs = max(1, int(resolved_num_envs))
        if (
            not isinstance(resolved_env_spacing, (tuple, list))
            or len(resolved_env_spacing) != 2
        ):
            raise ValueError("env_spacing must be a tuple/list of length 2.")
        return {
            "num_envs": normalized_num_envs,
            "env_spacing": (
                float(resolved_env_spacing[0]),
                float(resolved_env_spacing[1]),
            ),
            "n_envs_per_row": resolved_n_envs_per_row,
            "center_envs_at_origin": bool(resolved_center_envs_at_origin),
            "background_anchor_env": str(default_background_anchor_env),
        }

    @staticmethod
    def compute_env_offsets(
        num_envs: int,
        env_spacing,
        n_envs_per_row=None,
        center_envs_at_origin: bool = True,
    ) -> np.ndarray:
        normalized_num_envs = max(1, int(num_envs))
        if not isinstance(env_spacing, (tuple, list)) or len(env_spacing) != 2:
            raise ValueError("env_spacing must be a tuple/list of length 2.")

        resolved_n_envs_per_row = n_envs_per_row
        if resolved_n_envs_per_row is None:
            resolved_n_envs_per_row = int(math.ceil(math.sqrt(normalized_num_envs)))
        else:
            resolved_n_envs_per_row = max(1, int(resolved_n_envs_per_row))

        spacing_x = float(env_spacing[0])
        spacing_y = float(env_spacing[1])
        offset_x = (np.arange(normalized_num_envs) // resolved_n_envs_per_row) * spacing_x
        offset_y = (np.arange(normalized_num_envs) % resolved_n_envs_per_row) * spacing_y
        offset_z = np.zeros((normalized_num_envs,), dtype=np.float32)
        env_offsets = np.stack((offset_x, offset_y, offset_z), axis=-1).astype(np.float32)

        if center_envs_at_origin:
            center = (np.max(env_offsets, axis=0) + np.min(env_offsets, axis=0)) / 2.0
            env_offsets -= center

        return env_offsets

    @staticmethod
    def find_central_env_idx(
        num_envs: int,
        env_spacing,
        n_envs_per_row=None,
        center_envs_at_origin: bool = True,
    ) -> int:
        env_offsets = GenesisSim.compute_env_offsets(
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
        )
        distances = np.linalg.norm(env_offsets, axis=1)
        return int(np.argmin(distances))

    @property
    def num_envs(self):
        return self._build_config["num_envs"]

    @property
    def env_spacing(self):
        return self._build_config["env_spacing"]

    @property
    def n_envs_per_row(self):
        return self._build_config["n_envs_per_row"]

    @property
    def center_envs_at_origin(self):
        return self._build_config["center_envs_at_origin"]

    @property
    def background_anchor_env(self):
        return self._build_config["background_anchor_env"]

    def _resolve_background_anchor_env_idx(self) -> int:
        anchor_mode = str(self.background_anchor_env).strip().lower()
        if self.num_envs <= 1:
            return 0
        if anchor_mode == "rendered_env":
            return 0
        if anchor_mode == "center":
            return self.find_central_env_idx(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                n_envs_per_row=self.n_envs_per_row,
                center_envs_at_origin=self.center_envs_at_origin,
            )
        raise ValueError(
            "Unsupported parallel background anchor mode: "
            f"{self.background_anchor_env!r}. Expected 'rendered_env' or 'center'."
        )

    def _add_space_background(self):
        if not SPACE_BACKGROUND_SETTINGS["enable"]:
            return

        self.space_background = self._add_textured_sphere(
            SPACE_BACKGROUND_SETTINGS,
            name="space_background",
            double_sided=True,
        )

    def _add_earth_background(self):
        if not EARTH_BACKGROUND_SETTINGS["enable"]:
            return

        self.earth_background = self._add_textured_sphere(
            EARTH_BACKGROUND_SETTINGS,
            name="earth_background",
            double_sided=False,
        )

    def _build_background_morph_variants(self, settings):
        return gs.morphs.Mesh(
            file=settings["mesh_path"],
            pos=settings["pos"],
            euler=settings.get("euler", (0.0, 0.0, 0.0)),
            scale=settings["scale"],
            fixed=True,
            collision=False,
            visualization=True,
            decimate=False,
            convexify=False,
        )

    def _apply_parallel_background_layout(self):
        if self.num_envs <= 1:
            return

        anchor_env_idx = self._resolve_background_anchor_env_idx()
        hidden_env_ids = [env_idx for env_idx in range(self.num_envs) if env_idx != anchor_env_idx]
        if not hidden_env_ids:
            return

        hidden_env_ids_tensor = torch.as_tensor(
            hidden_env_ids, dtype=torch.long, device=self.device
        )

        for settings, entity in (
            (SPACE_BACKGROUND_SETTINGS, self.space_background),
            (EARTH_BACKGROUND_SETTINGS, self.earth_background),
        ):
            if entity is None:
                continue

            visible_pos = torch.tensor(
                settings["pos"], dtype=self.datatype, device=self.device
            ).reshape(1, 3)
            hidden_pos = torch.tensor(
                _BACKGROUND_NULL_POS, dtype=self.datatype, device=self.device
            ).reshape(1, 3).repeat(len(hidden_env_ids), 1)

            entity.set_pos(visible_pos, envs_idx=[anchor_env_idx], zero_velocity=True)
            entity.set_pos(hidden_pos, envs_idx=hidden_env_ids_tensor, zero_velocity=True)

    def _add_textured_sphere(self, settings, name: str, double_sided: bool):
        texture = gs.textures.ImageTexture(
            image_path=settings["texture_path"],
            encoding="srgb",
        )
        return self.scene.add_entity(
            morph=self._build_background_morph_variants(settings),
            material=gs.materials.Rigid(gravity_compensation=1.0),
            surface=gs.surfaces.Plastic(
                diffuse_texture=texture,
                emissive_texture=texture,
                double_sided=double_sided,
            ),
            name=name,
        )
    
    """
    Properties
    """
    @property
    def time(self):
        return self.timer.time

    """
    Operations
    """
    def start(self):
        print("[DEBUG] Building scene...")
        self.scene.build(
            n_envs=self.num_envs,
            env_spacing=self.env_spacing,
            n_envs_per_row=self.n_envs_per_row,
            center_envs_at_origin=self.center_envs_at_origin,
        )
        self._apply_parallel_background_layout()
        print("[DEBUG] Scene built.")
        
        self.timer = SimpleTimer()
        self._camera_recorded_episode_count = 0
        self._camera_video_segment_index = 0
        if self.camera_enable_recording:
            self.cam.start_recording()

    def step(self):
        self.scene.step()
        if self.camera_enable_recording:
            self.cam.render()

    def _save_camera_segment(self):
        if not self.camera_enable_recording:
            return
        if not getattr(self.cam, "_in_recording", False):
            return
        if not getattr(self.cam, "_recorded_imgs", None):
            return

        self._camera_video_segment_index += 1
        save_file = generate_filename(
            prefix=f"video_ep_chunk_{self._camera_video_segment_index:04d}",
            extension="mp4",
            folder_path=str(_RECORDINGS_DIR),
        )
        self.cam.stop_recording(save_to_filename=save_file, fps=60)
        print("Recorded video is saved to ", save_file)

    def on_episode_end(self, episode_count: int = 1):
        if not self.camera_enable_recording:
            return

        self._camera_recorded_episode_count += max(0, int(episode_count))
        if self._camera_recorded_episode_count < self.camera_episodes_per_video:
            return

        self._save_camera_segment()
        self._camera_recorded_episode_count = 0
        self.cam.start_recording()

    def stop(self):
        if self.camera_enable_recording:
            self._save_camera_segment()


    def reset(self):
        self.scene.reset()
        self.timer.reset()

    def attach_robot_to_object(self, robot, obj):
        # create fixed contraints
        # robot_idx = robot.get_link("hand").idx
        # object_idx = obj.get_link("attachment").idx
        # self.rigid.add_weld_constraint(robot_idx, object_idx)
        self.scene.link_entities(obj, robot, "base", "panda_link0")


    def __new__(cls, *args, **kwargs):
        """Allocates the memory and creates the actual simulation object is not instance exists yet. Otherwise,
        returns the existing instance of the simulation class.

        Returns:
            cls: the single instance of this class
        """

        # Use a lock in here to make sure we do not have a race condition
        # when using multi-threading and creating the first instance of this class
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GenesisSim, cls).__new__(cls)

        return cls._instance
    
    def __del__(self):
        """Destructor for the object. Destroys the only existing instance of this class."""
        GenesisSim._instance = None
        GenesisSim._is_initialized = False
        GenesisSim._init_config = None



if __name__ == "__main__":
    # 
    GS = GenesisSim()
    
    GS.start()
    # for i in range(1000):
    #     # print("current simulation time: ", GS.time)
    #     GS.update()
    
    while True:
        GS.step()
        if not GS.viewer.is_alive():
            print("Viewer window has been closed.")
            break
