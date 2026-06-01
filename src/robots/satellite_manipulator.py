import sys
import torch
import logging

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from robots.manipulator import Manipulator
from utils.utils import convert_dict_to_tensors,map_to_range
from controllers.smooth_IK_solver import SmoothIKSolver
from utils.setup_logger import setup_logger


def _round_nested(value, digits: int = 5):
    if isinstance(value, list):
        return [_round_nested(v, digits) for v in value]
    return round(float(value), digits)


class SatelliteManipulator(Manipulator):
    def __init__(
        self,
        name="franka_merge",
        sensors=[],
        backends=[]
    ):
        super().__init__(
            name=name,
            sensors=sensors,
            backends=backends
        )
        self.logger = setup_logger(f"SatelliteManipulator.{name}")
        
        # Logging the robot information
        self.logger.info(f"Joint dof indexes: {self.motors_dof+self.fingers_dof}")
        self.logger.info(f"Joint qs indexes: {self.motors_qs+self.fingers_qs}")
        self.logger.info(f"Joints all: {self.joints_info}")
        self.logger.info(f"Links all: {self.links_info}")
        

    def initialize(self):
        """After the scene is built."""
        # Initialize Franka robot's configuration
        self.set_config(self.config)

        # Initialize base and end effector's position and attitude
        self.update_state()
        self.logger.debug(f"End effector state: {self.ee_state}")

        # If a base PID backend is enabled, lock its target to the actual
        # startup pose. Otherwise the default zero setpoint keeps dragging the
        # free-floating satellite toward one fixed direction.
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

        self.IK = SmoothIKSolver(
            robot=self.robot, 
            end_effector=self.end_effector,
            smooth_factor=self.params["ik_params"]["smooth_factor"], 
            max_joint_change=self.params["ik_params"]["max_joint_change"],
        )
        if hasattr(self, "wrist_camera") and hasattr(self, "wrist_camera_enable_recording"):
            if self.wrist_camera_enable_recording:
                self.wrist_camera.start_recording()
                self.logger.info("Wrist camera recording started")
        self.logger.info(f"SatelliteManipulator robot {self._robot_name} initialization completed")


    def control_joint_pos(self, joint_position):
        """
        Input: 
        position: [x, y, z] (torch.tensor, list, tuple or numpy array)
        quaternion: [qw, qx, qy, qz] (torch.tensor, list, tuple or numpy array)
        """
        # Convert inputs to torch tensors with correct dtype and device if needed
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(joint_position, dtype=self.datatype, device=self.device)
        else:
            joint_position = joint_position.to(dtype=self.datatype, device=self.device)
        
        # Ensure correct shape
        if joint_position.dim() == 0:
            joint_position = joint_position.unsqueeze(0)

        if joint_position.dim() == 1:
            joint_position = joint_position.reshape(1, -1)
        elif joint_position.dim() > 2:
            error_msg = (
                f"Joint_position should have shape [6] or [N, 6], got {joint_position.shape}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure joint position has shape [6] or [N, 6]
        if joint_position.shape[-1] != 6:
            error_msg = (
                f"Joint_position should have trailing dim 6, got {joint_position.shape}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Control joints' dofs
        try:
            self.robot.control_dofs_position(
                position=joint_position,
                dofs_idx_local=self.motors_dof,
            )

            # Optional: Get feedback for verification
            qpos_fb = self.robot.get_qpos()
            self.logger.debug(
                f"Current joint feedback position shape: {getattr(qpos_fb, 'shape', None)}"
            )
            self.logger.debug(
                f"Target joint position shape: {tuple(joint_position.shape)}"
            )
            self.logger.debug(f"joint index: {self.motors_qs}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to set joint positions: {e}")
            return False

    def control_gripper(self, gripper_value):
        """
        Control the gripper with force commands.
        Args:
            gripper_value: in range [0, 1]
                        1 to open gripper, 0 to close
        """
        try:         
            gripper_value = torch.as_tensor(gripper_value, dtype=self.datatype, device=self.device)
            if gripper_value.dim() == 0:
                gripper_value = gripper_value.reshape(1)
            elif gripper_value.dim() > 1 and gripper_value.shape[-1] == 1:
                gripper_value = gripper_value.reshape(-1)
            elif gripper_value.dim() > 1:
                error_msg = (
                    f"gripper_value should be scalar or shape [N], got {tuple(gripper_value.shape)}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            force_scalar = map_to_range(
                gripper_value,
                0.0,
                1.0,
                self.gripper_force_scale,
                -self.gripper_force_scale,
            )
            finger_force = force_scalar.unsqueeze(-1) * self.config_gripper_joints.reshape(1, -1)
            self.logger.debug(
                f"gripper_value shape={tuple(gripper_value.shape)}, finger_force shape={tuple(finger_force.shape)}"
            )

            self.robot.control_dofs_force(force=finger_force, dofs_idx_local=self.fingers_dof)
            
            _ = self.get_gripper_value()
            self.logger.debug(f"\rGripper current value={self.gripper_value}")
            
            if isinstance(self.gripper_value, torch.Tensor) and self.gripper_value.dim() > 0:
                self.gripper_state = (
                    self.gripper_value < self.params["finger_close"][0]
                ).to(dtype=self.datatype)
            else:
                if self.gripper_value > self.params["finger_open"][0] and self.gripper_state == 1.0:
                    self.gripper_state = 0.0
                elif self.gripper_value < self.params["finger_close"][0] and self.gripper_state == 0.0:
                    self.gripper_state = 1.0
            self.logger.debug(f"\rGripper control completed")
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to control gripper: {e}"
            self.logger.error(error_msg)
            return False
