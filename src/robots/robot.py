
import sys
import torch
import logging
import genesis as gs
import math

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from configs.asset_configs import *
from configs.configs import TPV_CAM_SETTINGS
from sensors.batched_mounted_camera import BatchedMountedCamera
from sensors.batched_wrist_camera import BatchedWristCamera
from sensors.mounted_camera import MountedCamera
from sensors.wrist_camera import WristCamera
from controllers.backend import EmptyBackend, Backend
from controllers.pid import PIDController
from utils.singlelink_state import SingleLinkState
from utils.setup_logger import setup_logger
from utils.utils import generate_filename

_RECORDINGS_DIR = current_file_path.parent / "recordings"

class Robot:
    def __init__(
        self,
        name="franka",
        sensors=None,
        backends=None,
    ):
        self.logger = setup_logger(f"Robot.{name}")
        self.logger.info(f"Initializing robot: {name}")

        # Get the current scene 
        self._robot_name = name
        self._scene = GenesisSim().scene
        self.device = GenesisSim().device
        self.datatype = GenesisSim().datatype
        self.static_camera_config = TPV_CAM_SETTINGS["camera"]
        self.wrist_camera = None
        self.rl_wrist_camera = None
        self.base_camera = None
        self.rl_base_camera = None
        self.wrist_camera_enable_recording = False
        self.base_camera_enable_recording = False
        # Initialize the base's position and orientation
        self._base_state = SingleLinkState(device=self.device)
           
        # entity, config = parse_asset_config(ASSETS, name)
        try:
            asset = get_asset(name)
            self.params = get_configs(name)
            self.pid_params = get_pid(name)
            self.wrist_camera_params = get_wrist_camera(name)
            self.robot = self._scene.add_entity(**asset)
            self.logger.info(f"Robot asset loaded and added to scene: {name}")
        except Exception as e:
            self.logger.error(f"Failed to load robot asset: {e}")
            raise

        # 处理传感器 - 确保是列表
        if sensors is None:
            sensors = []
        elif not isinstance(sensors, list):
            sensors = [sensors]

        if self.wrist_camera_params["wrist_camera"]:
            try:
                rl_obs_config = self.wrist_camera_params.get("rl_obs", {})
                rl_obs_enabled = bool(rl_obs_config.get("enabled", False))

                wrist_cam = WristCamera(config=self.wrist_camera_params)
                sensors.append(wrist_cam)
                self.logger.info(f"Added wrist camera for robot: {self._robot_name}")
                
                # 添加腕部相机到场景
                camera_params = self.wrist_camera_params.get("camera", {})
                if camera_params:
                    self.wrist_camera = wrist_cam._cam
                    self.wrist_camera_enable_recording = self.wrist_camera_params.get("enable_recording", False)
                    self.wrist_camera_episodes_per_video = max(
                        1, int(self.wrist_camera_params.get("episodes_per_video", 5))
                    )
                    self._wrist_camera_recorded_episode_count = 0
                    self._wrist_camera_video_segment_index = 0
                    self.logger.info(f"Wrist camera recording enabled: {self.wrist_camera_enable_recording}")
                else:
                    self.logger.warning("No camera parameters found for wrist camera")

                if rl_obs_enabled:
                    batched_wrist_cam = BatchedWristCamera(config=self.wrist_camera_params)
                    sensors.append(batched_wrist_cam)
                    self.rl_wrist_camera = batched_wrist_cam
                    self.logger.info(
                        f"Added batched RL wrist camera for robot: {self._robot_name}"
                    )
                    
            except ImportError as e:
                self.logger.warning(f"Cannot import WristCamera: {e}")
            except Exception as e:
                self.logger.error(f"Failed to add wrist camera: {e}")

        base_camera_config = self.wrist_camera_params.get("base_camera", {})
        if base_camera_config.get("enabled", False):
            try:
                rl_obs_config = base_camera_config.get("rl_obs", {})
                rl_obs_enabled = bool(rl_obs_config.get("enabled", False))

                base_cam = MountedCamera(config=base_camera_config)
                sensors.append(base_cam)
                camera_params = base_camera_config.get("camera", {})
                if camera_params:
                    self.base_camera = base_cam.cam
                    self.base_camera_enable_recording = bool(
                        base_camera_config.get("enable_recording", False)
                    )
                    self.base_camera_episodes_per_video = max(
                        1, int(base_camera_config.get("episodes_per_video", 5))
                    )
                    self._base_camera_recorded_episode_count = 0
                    self._base_camera_video_segment_index = 0
                    self.logger.info(
                        f"Base camera recording enabled: {self.base_camera_enable_recording}"
                    )
                else:
                    self.logger.warning("No camera parameters found for base camera")
                self.logger.info(f"Added base camera for robot: {self._robot_name}")

                if rl_obs_enabled:
                    batched_base_cam = BatchedMountedCamera(config=base_camera_config)
                    sensors.append(batched_base_cam)
                    self.rl_base_camera = batched_base_cam
                    self.logger.info(
                        f"Added batched RL base camera for robot: {self._robot_name}"
                    )
            except ImportError as e:
                self.logger.warning(f"Cannot import MountedCamera: {e}")
            except Exception as e:
                self.logger.error(f"Failed to add base camera: {e}")
        self._sensors = sensors

        # 处理后端 - 确保是列表
        if backends is None:
            backends = []
        elif not isinstance(backends, list):
            backends = [backends]
        
        # 如果没有提供后端，创建默认的PIDController
        if not backends and self.pid_params.get("enable_pid", False):
            self.pid_params["setpoint"] = torch.cat([self._base_state.position_global, self._base_state.orient_global])
            pid_controller = PIDController(
                P=self.pid_params["P"],
                I=self.pid_params["I"],
                D=self.pid_params["D"],
                setpoint=self.pid_params["setpoint"],
                dt=self.pid_params["dt"],
                output_limits=self.pid_params["limits"],
                name=f"{name}_pid"
            )
            backends = [pid_controller]
            
        self._backends = backends
        for backend in self._backends:
            if isinstance(backend, PIDController):
                self.pid = backend
                break

        self._links = []
        self._get_links()


        self.logger.debug("Base state initialized")

        # --------------------------------------------------------------------
        # -------------------- Add sensors to the robot --------------------
        # --------------------------------------------------------------------
        self.logger.info(f"Adding {len(self._sensors)} sensors to robot")

        for sensor in self._sensors:
            sensor.initialize(self)
        self.logger.info("All sensors initialized")

        # --------------------------------------------------------------------
        # -------------------- Add control backends to the robot -----------
        # --------------------------------------------------------------------
        self.logger.info(f"Adding {len(self._backends)} backends to robot")

        # Initialize the backends
        for backend in self._backends:
            backend.initialize(self)
        self.logger.info("All backends initialized")

    """
    Properties
    """
    
    @property
    def base_state(self):
        """The state of the robot.

        Returns:
            State: The current state of the robot, i.e., position, orientation, linear and angular velocities...
        """
        return self._base_state
    
    @property
    def name(self) -> str:
        """Robot name.

        Returns:
            Robot name (str): 
        """
        return self._robot_name
    
    @property
    def joints_info(self):
        """Retrieves the current state of all robot joints.
        Returns:
            JointsData: Information about each joint, typically including 
                    name, dof_idx_local, qs_idx_local, and etc.
        """
        return self.robot.joints

    @property
    def links_info(self):
        """Retrieves the current state of all robot links (rigid bodies).
        Returns:
            LinksData: Information about each link.
        """
        return self.robot.links

    """
    Operations
    """
    def initialize(self):
        self.logger.debug(f"Robot {self._robot_name} initialization completed")
        print(self.joints_info)
        print(self.links_info)
        if hasattr(self, "_wrist_camera_recorded_episode_count"):
            self._wrist_camera_recorded_episode_count = 0
        if hasattr(self, "_wrist_camera_video_segment_index"):
            self._wrist_camera_video_segment_index = 0
        if hasattr(self, "_base_camera_recorded_episode_count"):
            self._base_camera_recorded_episode_count = 0
        if hasattr(self, "_base_camera_video_segment_index"):
            self._base_camera_video_segment_index = 0
        # 启动腕部相机录制（如果启用）
        if hasattr(self, 'wrist_camera') and hasattr(self, 'wrist_camera_enable_recording'):
            if self.wrist_camera_enable_recording:
                self.wrist_camera.start_recording()
                self.logger.info("Wrist camera recording started")
        if hasattr(self, "base_camera") and hasattr(self, "base_camera_enable_recording"):
            if self.base_camera_enable_recording:
                self.base_camera.start_recording()
                self.logger.info("Base camera recording started")
        
        return
    
    def step(self, dt: float = None):
        self.logger.debug(f"Robot {self._robot_name} step method called")
                
        self._update_state()

        for sensor in self._sensors:
            sensor.step()
        
        for backend in self._backends:
            if hasattr(backend, 'update_state') and self._base_state:
                backend.update_state(self._base_state.position_global, self._base_state.quat_global)

            backend.step()

            if hasattr(backend, 'input_reference'):
                control_output = backend.input_reference()
                self._apply_control_output(control_output=control_output, link_name=self.params["base"])
        
        return self._base_state
    
    def _update_state(self):
        self._base_state.update_from_global_frame(
            position=self.robot.get_link(self.params["base"]).get_pos(),
            quat = self.robot.get_link(self.params["base"]).get_quat(),
        )
    
    def _apply_control_output(self, control_output, link_name='base_link'):
        if control_output is None:
            return
            
        if isinstance(control_output, dict):
            force = control_output.get('position')
            torque = control_output.get('orientation')
            
            # 检查力和力矩是否非零
            force_nonzero = False
            torque_nonzero = False
            if force is not None:
                force_tensor = torch.as_tensor(force, dtype=self.datatype, device=self.device)
                force_nonzero = not torch.allclose(force_tensor, torch.zeros_like(force_tensor))
            if torque is not None:
                torque_tensor = torch.as_tensor(torque, dtype=self.datatype, device=self.device)
                torque_nonzero = not torch.allclose(torque_tensor, torch.zeros_like(torque_tensor))
            
            if force_nonzero or torque_nonzero:
                self.apply_force(force=force, torque=torque, link_name=link_name)
                # print(force, torque)

    def stop(self):
        self.logger.info("Stopping robot sensors and backends")
        
        # 停止腕部相机录制（如果启用）
        if hasattr(self, 'wrist_camera') and hasattr(self, 'wrist_camera_enable_recording'):
            if self.wrist_camera_enable_recording:
                try:
                    self._save_wrist_camera_segment()
                except Exception as e:
                    self.logger.error(f"Failed to stop wrist camera recording: {e}")
        if hasattr(self, "base_camera") and hasattr(self, "base_camera_enable_recording"):
            if self.base_camera_enable_recording:
                try:
                    self._save_base_camera_segment()
                except Exception as e:
                    self.logger.error(f"Failed to stop base camera recording: {e}")
        
        for sensor in self._sensors:
            sensor.stop()
        
        for backend in self._backends:
            backend.stop()
            
        self.logger.info("All robot components stopped")
        return

    def _save_wrist_camera_segment(self):
        if not hasattr(self, "wrist_camera") or not getattr(self, "wrist_camera_enable_recording", False):
            return
        if not getattr(self.wrist_camera, "_in_recording", False):
            return
        if not getattr(self.wrist_camera, "_recorded_imgs", None):
            return

        self._wrist_camera_video_segment_index += 1
        save_file = generate_filename(
            prefix=f"wrist_camera_{self._robot_name}_ep_chunk_{self._wrist_camera_video_segment_index:04d}",
            extension="mp4",
            folder_path=str(_RECORDINGS_DIR),
        )
        self.wrist_camera.stop_recording(save_to_filename=save_file, fps=60)
        self.logger.info(f"Wrist camera recording saved to: {save_file}")

    def _save_base_camera_segment(self):
        if not hasattr(self, "base_camera") or not getattr(self, "base_camera_enable_recording", False):
            return
        if not getattr(self.base_camera, "_in_recording", False):
            return
        if not getattr(self.base_camera, "_recorded_imgs", None):
            return

        self._base_camera_video_segment_index += 1
        save_file = generate_filename(
            prefix=f"base_camera_{self._robot_name}_ep_chunk_{self._base_camera_video_segment_index:04d}",
            extension="mp4",
            folder_path=str(_RECORDINGS_DIR),
        )
        self.base_camera.stop_recording(save_to_filename=save_file, fps=60)
        self.logger.info(f"Base camera recording saved to: {save_file}")

    def on_episode_end(self, episode_count: int = 1):
        if hasattr(self, "wrist_camera") and getattr(self, "wrist_camera_enable_recording", False):
            self._wrist_camera_recorded_episode_count += max(0, int(episode_count))
            if self._wrist_camera_recorded_episode_count >= self.wrist_camera_episodes_per_video:
                self._save_wrist_camera_segment()
                self._wrist_camera_recorded_episode_count = 0
                self.wrist_camera.start_recording()

        if hasattr(self, "base_camera") and getattr(self, "base_camera_enable_recording", False):
            self._base_camera_recorded_episode_count += max(0, int(episode_count))
            if self._base_camera_recorded_episode_count >= self.base_camera_episodes_per_video:
                self._save_base_camera_segment()
                self._base_camera_recorded_episode_count = 0
                self.base_camera.start_recording()

    def reset(self):
        self.logger.info("Robot reset method called")
        return

    def _get_links(self):
        self._links = {
            link.name: link.idx
            for link in getattr(self.robot, "links", [])
        }
        self.logger.debug(f"Found {len(self._links)} links in robot")

    def show_info(self):
        self.logger.info(f"Robot Name: {self._robot_name}")
        self.logger.info(f"Available Links: {list(self._links.keys())}")
        self.logger.info(f"Link Indices: {list(self._links.values())}")
        self.logger.info(f"Total Links: {len([link for link in self.robot.links])}")
        return

    def apply_force(self, force=None, torque=None, link_name='base_link'):
        """
        Method that will apply a force on the rigidbody, on the part specified in the 'body_part' at its relative position
        given by 'pos' (following a FLU) convention. 

        Args:
            force (list): A 3-dimensional vector of floats with the force [Fx, Fy, Fz] on the body axis of the vehicle according to a FLU convention.
            pos (list): _description_. Defaults to [0.0, 0.0, 0.0].
            body_part (str): . Defaults to "/body".
        """
        if not self._links:
            self._get_links()
        link_idx = self._links.get(link_name)
        if link_idx is None:
            error_msg = f"Unknown link '{link_name}'. Available: {list(self._links.keys())}"
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        force_tensor = self._format_wrench_input(force)
        torque_tensor = self._format_wrench_input(torque)

        links_idx = link_idx
        if force_tensor is not None and force_tensor.dim() == 2 and force_tensor.shape[0] > 1:
            links_idx = [link_idx] * force_tensor.shape[0]
        elif torque_tensor is not None and torque_tensor.dim() == 2 and torque_tensor.shape[0] > 1:
            links_idx = [link_idx] * torque_tensor.shape[0]

        # Get the handle of the rigidbody that we will apply the force to
        if force_tensor is not None:
            self._scene.sim.rigid_solver.apply_links_external_force(
                force=force_tensor,
                links_idx=links_idx,
            )
            self.logger.debug(
                f"Applied force with shape {tuple(force_tensor.shape)} to link '{link_name}' (ID: {link_idx})"
            )
        if torque_tensor is not None:
            self._scene.sim.rigid_solver.apply_links_external_torque(
                torque=torque_tensor,
                links_idx=links_idx,
            )
            self.logger.debug(
                f"Applied torque with shape {tuple(torque_tensor.shape)} to link '{link_name}' (ID: {link_idx})"
            )
        # print("Yes")
        
    def _format_wrench_input(self, vec):
        if vec is None:
            return None

        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=self.datatype, device=self.device)
        else:
            vec = vec.to(dtype=self.datatype, device=self.device)

        if vec.shape[-1] != 3:
            error_msg = f"Wrench inputs must have trailing dim 3, got shape {tuple(vec.shape)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if vec.dim() == 1:
            vec = vec.reshape(1, 3)
        elif vec.dim() > 2:
            error_msg = f"Wrench inputs must be shape (3,) or (N, 3), got {tuple(vec.shape)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return vec
