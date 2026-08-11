
import sys
import torch
import logging
import numpy as np
import genesis as gs

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from robots.robot import Robot
from envs.genesis_env import GenesisSim
from configs.asset_configs import get_pid
from utils.twolink_state import TwoLinkState
from utils.utils import convert_dict_to_tensors, map_to_range
from controllers.smooth_IK_solver import SmoothIKSolver
from controllers.pid import PIDController
from utils.setup_logger import setup_logger


def _round_nested(value, digits: int = 5):
    if isinstance(value, list):
        return [_round_nested(v, digits) for v in value]
    return round(float(value), digits)


class Manipulator(Robot):
    def __init__(
        self,
        name="franka",
        sensors=[],
        backends=[]
    ):
        self.device = GenesisSim().device
        # baselink_state = self._base_state.global_state
        self.ee_state = TwoLinkState(device=self.device) # body state is relative to the baselink of robot arm

        # 在调用父类初始化之前，先处理backends
        # 处理后端 - 确保是列表
        if backends is None:
            backends = []
        elif not isinstance(backends, list):
            backends = [backends]
        
        # 如果没有提供后端，创建默认的PIDController
        if not backends:
            manipulator_pid_params = get_pid(name)
            
            # 检查是否启用PID控制器
            if manipulator_pid_params.get("enable_pid", False):
                manipulator_pid_params["setpoint"] = torch.cat(
                    [
                        self.ee_state.link_parent_global_state.position,
                        self.ee_state.link_parent_global_state.orient,
                    ],
                    dim=-1,
                )
                
                pid_controller = PIDController(
                    P=manipulator_pid_params["P"],
                    I=manipulator_pid_params["I"],
                    D=manipulator_pid_params["D"],
                    setpoint=manipulator_pid_params["setpoint"],
                    dt=manipulator_pid_params["dt"],
                    output_limits=manipulator_pid_params["limits"],
                    name=f"{name}_pid"
                )
                backends = [pid_controller]
        
        # 现在backends已经有内容了，父类不会创建默认PID
        # Initialize the Robot object, and add robot to backends
        # Let FrankaMerge be able to have own posix
        super().__init__(
            name=name, 
            sensors=sensors,
            backends=backends
            )
        # 初始化logger
        self.logger = setup_logger(f"Franka.{name}")
        
        # Get the current world at which we want to spawn the Robot
        self.manipulator_name = name
        self._scene = GenesisSim().scene
        
        ### state
        self.end_effector = self.robot.get_link(self.params["end_effector"])
        self.base = self.robot.get_link(self.params["base"])
        self.config = convert_dict_to_tensors(self.params["config"], self.datatype, self.device)
        
        ### controller
        self.joints_name = self.params["joints"]

        motors_qs_idx = [self.robot.get_joint(name).qs_idx_local[0] for name in self.joints_name]
        self.motors_qs = motors_qs_idx[:self.params["motor"]]
        self.fingers_qs = motors_qs_idx[self.params["motor"] : self.params["finger"]]

        motors_dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joints_name]
        self.motors_dof = motors_dof_idx[:self.params["motor"]]
        self.fingers_dof = motors_dof_idx[self.params["motor"] : self.params["finger"]]

        # gripper controller
        self.config_gripper_joints = torch.tensor(
            self.params["gripper_revolute"],
            dtype=self.datatype,
            device=self.device,
        )
        self.gripper_force_scale = torch.tensor(
            self.params.get("gripper_force_scale", 1.0),
            dtype=self.datatype,
            device=self.device,
        )

        self.finger_open = torch.tensor(self.params["finger_open"][0], dtype=self.datatype, device=self.device) * self.config_gripper_joints
        self.finger_close = torch.tensor(self.params["finger_close"][0], dtype=self.datatype, device=self.device) * self.config_gripper_joints

        # gripper states
        self.gripper_value = torch.tensor(self.params["finger_open"][0], dtype=self.datatype, device=self.device)  # absolute value for gripper
        self.gripper_state = 1.0  # gripper state within range of [0, 1], and 1 for hand open
        self.base_pose_batch = None
        self.ee_pose_batch = None
        self.pid_enabled = hasattr(self, "pid")

    """
    Properties
    """
    
    @property
    def end_effector_state(self):
        """The state of the robot end effector.
        Returns:
            State: The current state of the robot, i.e., position, orientation, linear and angular velocities...
        """
        return self.ee_state
    
    @property
    def ee_global_position(self):
        """The global position of the end effector.
        Returns:
            Position state (torch.tensor): 
        """
        return self.ee_state.link_child_global_state.position
    
    @property
    def ee_global_quaternion(self):
        """The global orientation of the end effector.
        Returns:
            Quaternion state (torch.tensor):
        """
        return self.ee_state.link_child_global_state.quat

    @property
    def name(self) -> str:
        """Robot name.
        Returns:
            Robot name (str): 
        """
        return self._robot_name
    
 
    """
    Operations
    """

    def initialize(self):
        """After the scene is built."""
        # Initialize Franka robot's configuration
        self.set_config(self.config)

        # Initialize base and end effector's position and attitude
        self.update_state()

        # If a base-hold PID backend is enabled, lock its setpoint to the
        # manipulator's current base pose after the scene is built. Otherwise
        # the default zero setpoint can pull the whole satellite away during
        # grasping.
        if hasattr(self, "pid"):
            self.pid.reset()
            base_pose_setpoint = torch.cat(
                [
                    self.ee_state.link_parent_global_state.position.detach().clone(),
                    self.ee_state.link_parent_global_state.orient.detach().clone(),
                ],
                dim=-1,
            )
            self.pid.update_setpoint(base_pose_setpoint)
            self.logger.info(
                "Base PID setpoint initialized to current base pose: %s",
                _round_nested(base_pose_setpoint.detach().cpu().tolist()),
            )

        self.logger.info(f"End effector state initialized: {self.ee_state}")

        self.IK = SmoothIKSolver(
            robot=self.robot, 
            end_effector=self.end_effector,
            smooth_factor=self.params["ik_params"]["smooth_factor"], 
            max_joint_change=self.params["ik_params"]["max_joint_change"],
        )
        self.logger.info("IK solver initialized")
        
        # 启动腕部相机录制（如果启用）
        if hasattr(self, 'wrist_camera') and hasattr(self, 'wrist_camera_enable_recording'):
            if self.wrist_camera_enable_recording:
                self.wrist_camera.start_recording()
                self.logger.info("Wrist camera recording started")
        
    def set_config(self, config):
        """
        Apply PD, limits, and initial DOF state
        """
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, (np.ndarray, list, tuple)):
                return list(x)
            return [float(x)]

        kp    = to_list(config["control"]["kp"])
        kv    = to_list(config["control"]["kv"])
        frmin = to_list(config["control"]["force_range_min"])
        frmax = to_list(config["control"]["force_range_max"])
        init  = to_list(config["initial_dofs"])

        dofs = self.motors_dof+self.fingers_dof
        
        self.robot.set_dofs_kp(kp, dofs_idx_local=dofs)
        self.robot.set_dofs_kv(kv, dofs_idx_local=dofs)
        self.robot.set_dofs_force_range(frmin, frmax, dofs)
        self.robot.set_dofs_position(init, dofs)
        self.robot.set_dofs_velocity([0.0] * len(init), dofs)
        self.logger.debug("Robot configuration applied")

    def update_state(self):
        base_pos = torch.as_tensor(self.base.get_pos(), dtype=self.datatype, device=self.device)
        base_quat = torch.as_tensor(self.base.get_quat(), dtype=self.datatype, device=self.device)
        ee_pos = torch.as_tensor(self.end_effector.get_pos(), dtype=self.datatype, device=self.device)
        ee_quat = torch.as_tensor(self.end_effector.get_quat(), dtype=self.datatype, device=self.device)

        if base_pos.dim() == 1:
            base_pos_batch = base_pos.unsqueeze(0)
            base_quat_batch = base_quat.unsqueeze(0)
            ee_pos_batch = ee_pos.unsqueeze(0)
            ee_quat_batch = ee_quat.unsqueeze(0)
        else:
            base_pos_batch = base_pos
            base_quat_batch = base_quat
            ee_pos_batch = ee_pos
            ee_quat_batch = ee_quat

        self.base_pose_batch = {
            "position": base_pos_batch,
            "quat": base_quat_batch,
        }
        self.ee_pose_batch = {
            "position": ee_pos_batch,
            "quat": ee_quat_batch,
        }

        # Update end effector's base and child state with either single-env or
        # batched pose tensors. The state containers now support both shapes.
        self.ee_state.update_parent_from_global_frame(
            position=base_pos_batch if base_pos.dim() > 1 else base_pos_batch[0],
            quat=base_quat_batch if base_quat.dim() > 1 else base_quat_batch[0],  # quaternion: [qw, qx, qy, qz]
        )

        self.ee_state.update_child_from_global_frame(
            position=ee_pos_batch if ee_pos.dim() > 1 else ee_pos_batch[0],
            quat=ee_quat_batch if ee_quat.dim() > 1 else ee_quat_batch[0],  # quaternion: [qw, qx, qy, qz]
        )

        self.ee_state.update_child_in_parent()
    
    def step(self):
        # Update base state and then ee state of franka
        self.update_state()

        for sensor in self._sensors:
            sensor.step()

        for backend in self._backends:
            if backend is getattr(self, "pid", None) and not self.pid_enabled:
                continue
            if hasattr(backend, 'update_state') and self._base_state:
                backend.update_state(
                    position=self.ee_state.link_parent_global_state.position, 
                    orien=self.ee_state.link_parent_global_state.orient)

            backend.step()

            if hasattr(backend, 'input_reference'):
                control_output = backend.input_reference()
                self._apply_control_output(
                    control_output=control_output,
                    link_name=self.params.get("pid_apply_link", self.params["base"]),
                )

    def set_pid_enabled(self, enabled: bool):
        self.pid_enabled = bool(enabled) and hasattr(self, "pid")
        if hasattr(self, "pid") and hasattr(self.pid, "set_enabled"):
            self.pid.set_enabled(self.pid_enabled)
        return self.pid_enabled

    def capture_current_base_pose_as_pid_setpoint(self):
        if not hasattr(self, "pid"):
            raise RuntimeError("PID backend is not configured for this manipulator.")
        self.update_state()
        base_pose_setpoint = torch.cat(
            [
                self.ee_state.link_parent_global_state.position.detach().clone(),
                self.ee_state.link_parent_global_state.orient.detach().clone(),
            ],
            dim=-1 if self.ee_state.link_parent_global_state.position.dim() > 1 else 0,
        )
        self.pid.update_setpoint(base_pose_setpoint)
        return base_pose_setpoint
    
    def control_joints(self, position, quaternion):
        """
        Input: 
        position: [x, y, z] (torch.tensor, list, tuple or numpy array)
        quaternion: [qw, qx, qy, qz] (torch.tensor, list, tuple or numpy array)
        """
        # Convert inputs to torch tensors with correct dtype and device if needed
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.datatype, device=self.device)
        else:
            position = position.to(dtype=self.datatype, device=self.device)
        
        if not isinstance(quaternion, torch.Tensor):
            quaternion = torch.tensor(quaternion, dtype=self.datatype, device=self.device)
        else:
            quaternion = quaternion.to(dtype=self.datatype, device=self.device)
        
        # Ensure correct shape
        if position.dim() == 0:
            position = position.unsqueeze(0)
        if quaternion.dim() == 0:
            quaternion = quaternion.unsqueeze(0)
        
        # Ensure position has shape [3] and quat has shape [4]
        if position.shape[-1] != 3:
            error_msg = f"Position should have shape [3], got {position.shape}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if quaternion.shape[-1] != 4:
            error_msg = f"Quaternion should have shape [4], got {quaternion.shape}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Compute joints' angles under global frame
        self.logger.debug(f"Controlling joints to position: {position}, quaternion: {quaternion}")
        
        # Compute joints' angles under global frame
        qpos = self.IK.solve(position, quaternion)
        self.logger.info(f"Target command: {qpos}")
        if qpos.shape[-1] <= max(self.motors_qs):
            error_msg = (
                f"IK solution shape {tuple(qpos.shape)} is too short for motor qs indexes "
                f"{self.motors_qs}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Genesis IK returns the full qpos vector for the articulated system:
        # - root_joint (FREE): 7 values -> [x, y, z, qw, qx, qy, qz]
        # - arm + gripper joints: follow each joint's qs index
        # For this robot:
        #   qpos[0:7]   -> floating root pose
        #   qpos[7:13]  -> 6 arm motor joints
        #   qpos[13:19] -> 6 finger joints
        motor_qpos = qpos[self.motors_qs]
        self.logger.info(
            "Motor joint target extracted from IK using qs indexes %s: %s",
            self.motors_qs,
            motor_qpos,
        )
        
        # Check for NaN values
        if torch.isnan(qpos).any():
            self.logger.warning("IK solution contains NaN values, skipping joint control")
            return False

        # Control joints' dofs
        try:
            self.robot.control_dofs_position(position=motor_qpos, dofs_idx_local=self.motors_dof)
            
            # Optional: Get feedback for verification
            # qpos_fb = self.robot.get_qpos()
            # print("Current command: ", qpos_fb)
            self.logger.info("Joint control command executed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set joint positions: {e}")
            return False
        
    def control_gripper(self, gripper_open):
        """
        Control the gripper with force commands while keeping the input as an
        open/close command.
        Args:
            gripper_open: bool or value that can be converted to bool
                        True to open gripper, False to close
        """
        try:
            # Ensure input is boolean
            if not isinstance(gripper_open, bool):
                gripper_open = bool(gripper_open)
                self.logger.debug(f"Converted gripper_open to boolean: {gripper_open}")
            
            # Map open/close to a signed force scalar, then project it onto the
            # configured finger joint directions.
            force_scalar = self.gripper_force_scale if gripper_open else -self.gripper_force_scale
            finger_force = force_scalar * self.config_gripper_joints
            action = "opening" if gripper_open else "closing"

            gripper_value = self.get_gripper_value()
            self.logger.info(f"Gripper {action} with finger state: {gripper_value}")
            self.robot.control_dofs_force(force=finger_force, dofs_idx_local=self.fingers_dof)
            
            # Update current state
            # self.gripper_state = gripper_open
            self.logger.info(f"Gripper {action} completed successfully")
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to control gripper: {e}"
            self.logger.error(error_msg)
            return False

    def get_gripper_value(self):
        """Read current gripper joint position from the simulator.

        This queries the robot's qpos for the finger joints (`self.fingers_qs`), then:
        - updates `self.gripper_value` with the first finger qpos (torch scalar)
        - updates `self.gripper_state` with a normalized value in [0, 1] (python float)

        Returns:
            torch.Tensor: Scalar tensor of the first finger qpos (absolute joint position).
        """
        if not hasattr(self, 'fingers_qs') or self.fingers_qs is None or len(self.fingers_qs) == 0:
            raise ValueError("fingers_qs is empty; cannot read gripper qpos")

        # qpos_fingers: torch.Tensor with shape (n_qs,) or (n_envs, n_qs)
        qpos_fingers = self.robot.get_qpos(qs_idx_local=self.fingers_qs)
        qpos_fingers = torch.as_tensor(qpos_fingers, device=self.device, dtype=self.datatype)
        if qpos_fingers.numel() == 0:
            raise ValueError("gripper qpos array is empty")

        if qpos_fingers.dim() == 1:
            self.gripper_value = qpos_fingers.reshape(-1)[0]
        else:
            self.gripper_value = qpos_fingers[:, 0]

        # Normalize to [0, 1] using configured open/close endpoints
        finger_close_0 = self.finger_close.reshape(-1)[0]
        finger_open_0 = self.finger_open.reshape(-1)[0]
        mapped = map_to_range(self.gripper_value, finger_close_0, finger_open_0, 0.0, 1.0)
        if isinstance(mapped, torch.Tensor):
            if mapped.dim() == 0:
                self.gripper_state = float(mapped.detach().cpu().item())
            else:
                self.gripper_state = mapped.detach().clone()
        else:
            self.gripper_state = float(mapped)
        return self.gripper_value

    def show_info(self):
        self.logger.info(f"End effector body state: {self.ee_state.link_child_in_parent}")
        self.logger.info(f"End effector global state: {self.ee_state.link_child_global_state}")
        self.logger.info(f"Base global state: {self.ee_state.link_parent_global_state}")
        return super().show_info()
