# 输入：TCP位姿、目标上抓取点位姿、基座卫星位姿（18维）
# 输出：机械臂关节角、卫星二维平面位置（8维）
# 奖励：底座先对位，再让TCP逼近抓取点，最后闭合夹爪完成抓取

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import genesis as gs

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from robots.robot import Robot
from robots.satellite_manipulator import SatelliteManipulator
from tasks.task import Task, TaskStatus
from utils.utils import as_rotation_matrix

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def resolve_training_device(device: str = "auto") -> str:
    device = device.strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False."
        )
    return device


class RLTask(Task):
    def __init__(
        self,
        target_asset_name: str = "starlink",
        num_envs: int = None,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
        target_position_noise: bool = True,
        target_position_noise_range=(0.03, 0.03, 0.02),
    ):
        super().__init__(
            task_name="RL_task",
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
        )

        self.starlink_manipulator: Optional[SatelliteManipulator] = None
        self.target_object: Optional[Robot] = None
        self.target_asset_name = target_asset_name
        self.target_position_noise = bool(target_position_noise)
        self.target_position_noise_range = torch.tensor(
            target_position_noise_range,
            dtype=self.datatype,
            device=self.device,
        )

        self.target_link_name = ""
        self.target_grasp_offset_local = torch.tensor(
            [0.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )
        self.grasp_marker_offset_local = torch.tensor(
            [0.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )

        # TCP uses the gripper grasp center instead of the arm-hand connection link.
        self.tcp_offset_local = torch.tensor(
            [0.002085, -0.2, 0.0004827],
            dtype=self.datatype,
            device=self.device,
        )

        self.max_steps = 300
        self.base_alignment_threshold = 0.18
        self.grasp_success_threshold = 0.06
        self.base_force_scale_xy = 100.0
        self.joint_action_scale = 0.5
        self.base_action_scale = 0.03
        self.gripper_close_value = 0.0
        self.gripper_open_value = 1.0
        self.time_penalty = 0.01
        self.base_planar_reference = None
        self.joint_reference = None
        self.initial_base_pos = None
        self.initial_base_quat = None
        self.initial_target_pos = None
        self.initial_target_quat = None
        self.initial_joint_positions = None

        # 输入的三个量
        self.tcp_pose = None
        self.target_grasp_pose = None
        self.base_satellite_pose = None

        # 输出的两个量
        self.arm_joint_positions = None
        self.base_planar_position = None

        self.observation = None
        self.action = None

        self.base_alignment_error = None
        self.tcp_to_grasp_error = None
        self.phase = "not_started"
        self.last_gripper_command = self.gripper_open_value
        self.contact_hold_steps = 12
        self.edge_contact_tolerance = 0.08
        self.contact_force_threshold = 1e-3
        self.edge_contact_hold_count = 0
        self.edge_contact_active = False
        self.approach_alignment_threshold = 0.85
        self.approach_alignment = None
        self.grasp_finger_link_names = (
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
        )
        self.env_step_count = None
        self.env_episode_return = None
        self.last_done_mask = None
        self.last_truncated_mask = None
        self.grasp_point_marker = None
        self.grasp_point_marker_radius = 0.025
        self.grasp_point_marker_height = 0.006
        self.debug_grasp_marker = True
        self.debug_grasp_marker_interval = 30

    def _create_grasp_point_marker(self):
        marker = self._scene.add_entity(
            morph=gs.morphs.Cylinder(
                pos=(-10.0, -10.0, -10.0),
                radius=self.grasp_point_marker_radius,
                height=self.grasp_point_marker_height,
                fixed=True,
                collision=False,
                visualization=True,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            surface=gs.surfaces.Plastic(
                color=(1.0, 0.1, 0.1, 1.0),
            ),
            name=f"{self.task_name}_grasp_point_marker",
        )
        return marker

    def _sync_grasp_point_marker(
        self,
        grasp_point_pos: torch.Tensor,
        grasp_point_quat: Optional[torch.Tensor] = None,
    ):
        if self.grasp_point_marker is None:
            return

        marker_pos = self._get_grasp_marker_pos(grasp_point_pos, grasp_point_quat)
        if marker_pos.dim() == 1:
            self.grasp_point_marker.set_pos(marker_pos, zero_velocity=True)
        else:
            self.grasp_point_marker.set_pos(marker_pos, zero_velocity=True)

    def _log_grasp_marker_debug(self, grasp_point_pos: torch.Tensor):
        if not self.debug_grasp_marker:
            return

        if self.step_count != 0 and self.step_count % self.debug_grasp_marker_interval != 0:
            return

        target_link = self.target_object.robot.get_link(self.target_link_name)
        target_link_pos = self._as_task_tensor(
            target_link.get_pos(), expected_last_dim=3
        ).detach().cpu()
        grasp_point_pos_cpu = grasp_point_pos.detach().cpu()

        if self.grasp_point_marker is None:
            marker_pos_cpu = None
        else:
            marker_pos_cpu = self._as_task_tensor(
                self.grasp_point_marker.get_pos(), expected_last_dim=3
            ).detach().cpu().tolist()

        print(
            "[GRASP_MARKER_DEBUG]",
            f"step={self.step_count}",
            f"target_link_pos={target_link_pos.tolist()}",
            f"grasp_point_pos={grasp_point_pos_cpu.tolist()}",
            f"marker_pos={marker_pos_cpu}",
            flush=True,
        )

    def _as_task_tensor(
        self,
        value,
        expected_last_dim: Optional[int] = None,
        keep_batch_dim: bool = False,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=self.datatype, device=self.device)
        if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Expected trailing dim {expected_last_dim}, got shape {tuple(tensor.shape)}"
            )
        if self.num_envs > 1:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] != self.num_envs:
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(self.num_envs, *([1] * (tensor.dim() - 1)))
                else:
                    raise ValueError(
                        f"Expected batch size {self.num_envs}, got shape {tuple(tensor.shape)}"
                    )
        if keep_batch_dim and tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _get_tcp_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ee_link = self.starlink_manipulator.robot.get_link(
            self.starlink_manipulator.params["end_effector"]
        )
        ee_pos = self._as_task_tensor(ee_link.get_pos(), expected_last_dim=3)
        ee_quat = self._as_task_tensor(ee_link.get_quat(), expected_last_dim=4)
        ee_rot = as_rotation_matrix(ee_quat, order="wxyz")
        tcp_pos = ee_pos + torch.matmul(ee_rot, self.tcp_offset_local)
        return tcp_pos, ee_quat

    def _get_target_grasp_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        target_link = self.target_object.robot.get_link(self.target_link_name)
        target_pos = self._as_task_tensor(target_link.get_pos(), expected_last_dim=3)
        target_quat = self._as_task_tensor(target_link.get_quat(), expected_last_dim=4)
        target_rot = as_rotation_matrix(target_quat, order="wxyz")
        grasp_point_pos = target_pos + torch.matmul(
            target_rot, self.target_grasp_offset_local
        )
        return grasp_point_pos, target_quat

    def _get_grasp_marker_pos(
        self,
        grasp_point_pos: torch.Tensor,
        grasp_point_quat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        marker_pos = grasp_point_pos.detach().clone()
        if grasp_point_quat is None:
            return marker_pos + self.grasp_marker_offset_local
        target_rot = as_rotation_matrix(grasp_point_quat, order="wxyz")
        return marker_pos + torch.matmul(target_rot, self.grasp_marker_offset_local)

    def _get_base_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        root_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=list(range(7))
        )
        root_qpos = self._as_task_tensor(root_qpos, expected_last_dim=7)
        return root_qpos[..., :3], root_qpos[..., 3:7]

    def _compute_approach_alignment(self, tcp_quat: torch.Tensor) -> torch.Tensor:
        tcp_rot = as_rotation_matrix(tcp_quat, order="wxyz")
        local_approach_axis = self.tcp_offset_local / torch.clamp(
            torch.norm(self.tcp_offset_local),
            min=torch.tensor(1e-6, dtype=self.datatype, device=self.device),
        )
        world_approach_axis = torch.matmul(tcp_rot, local_approach_axis)
        world_down_axis = torch.tensor(
            [0.0, 0.0, -1.0], dtype=self.datatype, device=self.device
        )
        if world_approach_axis.dim() == 1:
            return torch.dot(world_approach_axis, world_down_axis)
        return torch.sum(world_approach_axis * world_down_axis, dim=-1)

    def _get_contact_world_positions(self, finger_link_name: str):
        finger_link = self.starlink_manipulator.robot.get_link(finger_link_name)
        contacts = self.starlink_manipulator.robot.get_contacts(with_entity=self.target_object.robot)
        positions = []

        if contacts is None:
            return positions

        link_a = contacts.get("link_a")
        link_b = contacts.get("link_b")
        contact_pos = contacts.get("position")
        force_a = contacts.get("force_a")
        force_b = contacts.get("force_b")

        if link_a is None or link_b is None or contact_pos is None:
            return positions

        for idx in range(link_a.shape[0]):
            match_a = int(link_a[idx].item()) == finger_link.idx
            match_b = int(link_b[idx].item()) == finger_link.idx
            if not (match_a or match_b):
                continue

            force = force_a[idx] if match_a else force_b[idx]
            if torch.norm(force).item() < self.contact_force_threshold:
                continue

            positions.append(contact_pos[idx].detach().clone())

        return positions

    def _check_edge_grasp_contact(self) -> bool:
        if self.num_envs > 1:
            self.edge_contact_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if not isinstance(self.edge_contact_hold_count, torch.Tensor):
                self.edge_contact_hold_count = torch.zeros(
                    self.num_envs, dtype=torch.int32, device=self.device
                )
            self.edge_contact_hold_count = torch.where(
                self.edge_contact_active,
                self.edge_contact_hold_count + 1,
                torch.zeros_like(self.edge_contact_hold_count),
            )
            return False

        grasp_point_pos, _ = self._get_target_grasp_pose()
        finger_a_contacts = self._get_contact_world_positions(self.grasp_finger_link_names[0])
        finger_b_contacts = self._get_contact_world_positions(self.grasp_finger_link_names[1])

        has_finger_a_edge_contact = any(
            torch.norm(contact_pos - grasp_point_pos).item() <= self.edge_contact_tolerance
            for contact_pos in finger_a_contacts
        )
        has_finger_b_edge_contact = any(
            torch.norm(contact_pos - grasp_point_pos).item() <= self.edge_contact_tolerance
            for contact_pos in finger_b_contacts
        )

        self.edge_contact_active = has_finger_a_edge_contact and has_finger_b_edge_contact
        if self.edge_contact_active:
            self.edge_contact_hold_count += 1
        else:
            self.edge_contact_hold_count = 0

        return self.edge_contact_active

    def _update_io_tensors(self):
        tcp_pos, tcp_quat = self._get_tcp_pose()
        grasp_point_pos, grasp_point_quat = self._get_target_grasp_pose()
        self._sync_grasp_point_marker(grasp_point_pos, grasp_point_quat)
        self._log_grasp_marker_debug(grasp_point_pos)
        base_pos, base_quat = self._get_base_pose()
        joint_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )

        if self.num_envs > 1:
            joint_qpos = self._as_task_tensor(joint_qpos, expected_last_dim=6)
            self.tcp_pose = torch.cat([tcp_pos, tcp_quat], dim=-1).to(
                dtype=self.datatype, device=self.device
            )
            self.target_grasp_pose = torch.cat(
                [grasp_point_pos, grasp_point_quat], dim=-1
            ).to(dtype=self.datatype, device=self.device)
            self.base_satellite_pose = torch.cat([base_pos, base_quat], dim=-1).to(
                dtype=self.datatype, device=self.device
            )
            self.arm_joint_positions = joint_qpos.reshape(self.num_envs, -1)
            self.base_planar_position = base_pos[:, :2].to(
                dtype=self.datatype, device=self.device
            )

            self.observation = torch.cat(
                [self.tcp_pose, self.target_grasp_pose, self.base_satellite_pose], dim=-1
            )
            self.action = torch.cat(
                [self.arm_joint_positions, self.base_planar_position], dim=-1
            )

            self.base_alignment_error = torch.norm(
                grasp_point_pos[:, :2] - self.base_planar_position, dim=-1
            )
            self.tcp_to_grasp_error = torch.norm(grasp_point_pos - tcp_pos, dim=-1)
            self.approach_alignment = self._compute_approach_alignment(tcp_quat)

            phase = np.where(
                self.base_alignment_error.detach().cpu().numpy() <= self.base_alignment_threshold,
                "arm_approach",
                "base_alignment",
            ).astype(object)
            grasp_mask = (
                (self.tcp_to_grasp_error <= self.grasp_success_threshold)
                & (self.approach_alignment >= self.approach_alignment_threshold)
            )
            self.phase = np.where(grasp_mask.detach().cpu().numpy(), "grasp", phase).astype(object)
        else:
            self.tcp_pose = torch.cat([tcp_pos, tcp_quat]).to(
                dtype=self.datatype, device=self.device
            )
            self.target_grasp_pose = torch.cat([grasp_point_pos, grasp_point_quat]).to(
                dtype=self.datatype, device=self.device
            )
            self.base_satellite_pose = torch.cat([base_pos, base_quat]).to(
                dtype=self.datatype, device=self.device
            )
            self.arm_joint_positions = torch.as_tensor(
                joint_qpos, dtype=self.datatype, device=self.device
            ).reshape(-1)
            self.base_planar_position = base_pos[:2].to(
                dtype=self.datatype, device=self.device
            )

            self.observation = torch.cat(
                [self.tcp_pose, self.target_grasp_pose, self.base_satellite_pose]
            )
            self.action = torch.cat(
                [self.arm_joint_positions, self.base_planar_position]
            )

            self.base_alignment_error = torch.norm(
                grasp_point_pos[:2] - self.base_planar_position
            )
            self.tcp_to_grasp_error = torch.norm(grasp_point_pos - tcp_pos)
            self.approach_alignment = self._compute_approach_alignment(tcp_quat)

            if self.base_alignment_error <= self.base_alignment_threshold:
                self.phase = "arm_approach"
            else:
                self.phase = "base_alignment"

            if (
                self.tcp_to_grasp_error <= self.grasp_success_threshold
                and self.approach_alignment >= self.approach_alignment_threshold
            ):
                self.phase = "grasp"

    def _reset_rollout_tracking(self):
        if self.num_envs > 1:
            self.env_step_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.env_episode_return = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.last_done_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_truncated_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            self.env_step_count = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.env_episode_return = torch.zeros(
                1, dtype=self.datatype, device=self.device
            )
            self.last_done_mask = torch.zeros(1, dtype=torch.bool, device=self.device)
            self.last_truncated_mask = torch.zeros(
                1, dtype=torch.bool, device=self.device
            )

    def get_done_mask(self) -> torch.Tensor:
        if self.last_done_mask is None:
            self._reset_rollout_tracking()
        return self.last_done_mask.detach().clone()

    def get_truncated_mask(self) -> torch.Tensor:
        if self.last_truncated_mask is None:
            self._reset_rollout_tracking()
        return self.last_truncated_mask.detach().clone()

    def reset_done_envs(self, env_ids=None) -> bool:
        if env_ids is None:
            return self.reset()
        env_ids = list(env_ids)
        if not env_ids:
            return True
        if self.num_envs == 1 and env_ids == [0]:
            return self.reset()
        raise NotImplementedError(
            "Partial env reset is only implemented by batched task variants."
        )

    def _sample_target_reset_noise(self, env_count: int):
        if self.target_position_noise:
            noise_range = self.target_position_noise_range.detach().cpu().numpy()
            target_pos_noise = torch.tensor(
                np.stack(
                    [
                        np.random.uniform(-noise_range[0], noise_range[0], size=env_count),
                        np.random.uniform(-noise_range[1], noise_range[1], size=env_count),
                        np.random.uniform(-noise_range[2], noise_range[2], size=env_count),
                    ],
                    axis=-1,
                ),
                dtype=self.datatype,
                device=self.device,
            )
        else:
            target_pos_noise = torch.zeros(
                (env_count, 3), dtype=self.datatype, device=self.device
            )
        yaw_quat = torch.stack(
            [
                torch.ones(env_count, dtype=self.datatype, device=self.device),
                torch.zeros(env_count, dtype=self.datatype, device=self.device),
                torch.zeros(env_count, dtype=self.datatype, device=self.device),
                torch.zeros(env_count, dtype=self.datatype, device=self.device),
            ],
            dim=-1,
        )
        return target_pos_noise, yaw_quat

    def _reset_task_state_buffers(self):
        self.step_count = 0
        if self.num_envs > 1:
            self.total_reward = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.success = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.edge_contact_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.edge_contact_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_gripper_command = torch.full(
                (self.num_envs,),
                self.gripper_open_value,
                dtype=self.datatype,
                device=self.device,
            )
        else:
            self.total_reward = 0.0
            self.success = False
            self.edge_contact_hold_count = 0
            self.edge_contact_active = False
            self.last_gripper_command = self.gripper_open_value
        self._reset_rollout_tracking()

    def _apply_reset_state(
        self,
        base_pos,
        base_quat,
        target_pos,
        target_quat,
        relative_yaw_quat,
        initial_full_dofs,
        zero_velocity,
        env_ids=None,
    ):
        root_qpos = torch.cat([base_pos, base_quat], dim=-1)
        self.starlink_manipulator.robot.set_qpos(
            root_qpos,
            qs_idx_local=list(range(7)),
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self.target_object.robot.set_pos(target_pos, envs_idx=env_ids, zero_velocity=True)
        self.target_object.robot.set_quat(target_quat, envs_idx=env_ids, zero_velocity=True)
        self.target_object.robot.set_quat(
            relative_yaw_quat,
            envs_idx=env_ids,
            zero_velocity=True,
            relative=True,
        )
        self.starlink_manipulator.robot.set_dofs_position(
            initial_full_dofs,
            dofs_idx_local=self.starlink_manipulator.motors_dof
            + self.starlink_manipulator.fingers_dof,
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self.starlink_manipulator.robot.set_dofs_velocity(
            zero_velocity,
            dofs_idx_local=self.starlink_manipulator.motors_dof
            + self.starlink_manipulator.fingers_dof,
            envs_idx=env_ids,
        )

    def initialize(self) -> bool:
        try:
            if self.grasp_point_marker is None:
                self.grasp_point_marker = self._create_grasp_point_marker()
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(
                    name="franka_merge",
                    sensors=[],
                    backends=[],
                )
            if self.target_object is None:
                self.target_object = Robot(name=self.target_asset_name)

            self.gsim.start()
            self.starlink_manipulator.initialize()
            self.target_object.initialize()

            self.target_link_name = self.target_object.params.get(
                "grasp_link", self.target_object.params["base"]
            )
            self.target_grasp_offset_local = torch.tensor(
                self.target_object.params.get("grasp_offset_local", [0.0, 0.0, 0.0]),
                dtype=self.datatype,
                device=self.device,
            )
            self.grasp_marker_offset_local = torch.tensor(
                self.target_object.params.get(
                    "grasp_marker_offset_local", [0.0, 0.0, 0.0]
                ),
                dtype=self.datatype,
                device=self.device,
            )
            self.starlink_manipulator.update_state()
            self._update_io_tensors()

            self.initial_target_pos = self.target_object.robot.get_link(
                self.target_link_name
            ).get_pos()
            self.initial_target_quat = self.target_object.robot.get_link(
                self.target_link_name
            ).get_quat()
            self.initial_target_pos = self._as_task_tensor(
                self.initial_target_pos, expected_last_dim=3
            ).detach().clone()
            self.initial_target_quat = self._as_task_tensor(
                self.initial_target_quat, expected_last_dim=4
            ).detach().clone()
            base_pose = self.base_satellite_pose
            if getattr(base_pose, "dim", lambda: 0)() > 1:
                base_pos = base_pose[..., :3]
                base_quat = base_pose[..., 3:]
            else:
                base_pos = base_pose[:3]
                base_quat = base_pose[3:]
            self.initial_base_pos = self._as_task_tensor(
                base_pos,
                expected_last_dim=3,
            ).detach().clone()
            self.initial_base_quat = self._as_task_tensor(
                base_quat,
                expected_last_dim=4,
            ).detach().clone()
            self.initial_joint_positions = self._as_task_tensor(
                self.arm_joint_positions, expected_last_dim=6
            ).detach().clone()

            self.base_planar_reference = self._as_task_tensor(
                self.base_planar_position, expected_last_dim=2
            ).detach().clone()
            self.joint_reference = self._as_task_tensor(
                self.arm_joint_positions, expected_last_dim=6
            ).detach().clone()

            self._reset_task_state_buffers()
            self.status = TaskStatus.RUNNING
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def _apply_base_action(self, planar_action: torch.Tensor):
        planar_action = self._as_task_tensor(
            planar_action,
            expected_last_dim=2,
            keep_batch_dim=self.num_envs > 1,
        )
        force_xy = planar_action * self.base_force_scale_xy
        if self.num_envs > 1:
            force_cmd = torch.cat(
                [
                    force_xy,
                    torch.zeros((force_xy.shape[0], 1), dtype=self.datatype, device=self.device),
                ],
                dim=-1,
            )
        else:
            force_cmd = torch.tensor(
                [force_xy[0], force_xy[1], 0.0],
                dtype=self.datatype,
                device=self.device,
            )
        self.starlink_manipulator.apply_force(
            force=force_cmd,
            link_name=self.starlink_manipulator.params["base"],
        )

    def _apply_arm_action(self, joint_action: torch.Tensor):
        joint_action = self._as_task_tensor(
            joint_action,
            expected_last_dim=6,
            keep_batch_dim=self.num_envs > 1,
        )
        target_joint = self.arm_joint_positions + joint_action * self.joint_action_scale
        self.starlink_manipulator.control_joint_pos(target_joint)

    def apply_rl_action(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=self.datatype, device=self.device)
        if self.num_envs > 1:
            action = self._as_task_tensor(action, expected_last_dim=8)
            joint_action = torch.clamp(action[:, :6], -1.0, 1.0)
            planar_action = torch.clamp(action[:, 6:], -1.0, 1.0)
        else:
            action = action.reshape(-1)
            if action.shape[0] != 8:
                raise ValueError(f"RL action should have 8 dims, got {tuple(action.shape)}")
            joint_action = torch.clamp(action[:6], -1.0, 1.0)
            planar_action = torch.clamp(action[6:], -1.0, 1.0)

        self._apply_base_action(planar_action)
        self._apply_arm_action(joint_action)

        if self.num_envs > 1:
            grasp_mask = torch.as_tensor(
                self.phase == "grasp", dtype=self.datatype, device=self.device
            )
            self.last_gripper_command = torch.where(
                grasp_mask > 0,
                torch.full((self.num_envs,), self.gripper_close_value, dtype=self.datatype, device=self.device),
                torch.full((self.num_envs,), self.gripper_open_value, dtype=self.datatype, device=self.device),
            )
        else:
            if self.phase == "grasp":
                self.last_gripper_command = self.gripper_close_value
            else:
                self.last_gripper_command = self.gripper_open_value
        self.starlink_manipulator.control_gripper(self.last_gripper_command)

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            if action is not None:
                self.apply_rl_action(action)
            else:
                self.starlink_manipulator.control_gripper(self.last_gripper_command)

            self.starlink_manipulator.step()
            self.target_object.step()
            self.gsim.step()
            self._update_io_tensors()
            self._check_edge_grasp_contact()

            reward = self.reward()
            if self.num_envs > 1:
                self.total_reward = self.total_reward + reward
            else:
                self.total_reward += reward.item()
            self.step_count += 1
            self.env_step_count += 1
            self.env_episode_return += reward.reshape(-1)

            success_mask = self.is_success()
            if self.num_envs > 1:
                self.success = success_mask
                self.last_done_mask = success_mask.detach().clone()
                self.last_truncated_mask = self.env_step_count >= self.max_steps
                if bool(torch.all(success_mask).item()):
                    self.status = TaskStatus.SUCCEED
                    return False
            else:
                self.last_done_mask = torch.tensor(
                    [bool(success_mask)], dtype=torch.bool, device=self.device
                )
                self.last_truncated_mask = torch.tensor(
                    [self.step_count >= self.max_steps],
                    dtype=torch.bool,
                    device=self.device,
                )
                if success_mask:
                    self.success = True
                    self.status = TaskStatus.SUCCEED
                    return False

            if self.step_count >= self.max_steps:
                self.status = TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reward(self) -> torch.Tensor:
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            if self.num_envs > 1:
                return torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)
            return torch.tensor(0.0, dtype=self.datatype, device=self.device)

        reward = -0 * self.base_alignment_error - 1.5 * self.tcp_to_grasp_error
        #reward = reward + 2.0 * self.approach_alignment

        #if self.base_alignment_error <= self.base_alignment_threshold:
        #   reward = reward + 2.0
        if self.num_envs > 1:
            reward = reward + torch.where(
                self.tcp_to_grasp_error <= self.grasp_success_threshold,
                torch.full_like(reward, 10.0),
                torch.zeros_like(reward),
            )
        elif self.tcp_to_grasp_error <= self.grasp_success_threshold:
            reward = reward + 10.0
        #if self.approach_alignment >= self.approach_alignment_threshold:
        #    reward = reward + 4.0
        #if self.edge_contact_active:
        #    reward = reward + 5.0
        #if self.edge_contact_hold_count >= self.contact_hold_steps:
        #    reward = reward + 15.0
        if self.num_envs > 1:
            reward = reward - torch.where(
                self.is_success(),
                torch.zeros_like(reward),
                torch.full_like(reward, self.time_penalty),
            )
        elif not self.is_success():
            reward = reward - self.time_penalty

        return reward.to(dtype=self.datatype, device=self.device)

    def is_success(self) -> bool:
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            if self.num_envs > 1:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return False
        return (
            self.tcp_to_grasp_error <= self.grasp_success_threshold
            #and self.base_alignment_error <= self.base_alignment_threshold
            #and self.approach_alignment >= self.approach_alignment_threshold
            #and self.last_gripper_command == self.gripper_close_value
            #and self.edge_contact_hold_count >= self.contact_hold_steps
        )

    def reset(self) -> bool:
        try:
            if self.starlink_manipulator is None or self.target_object is None:
                self.logger.warning("Reset skipped because task is not initialized.")
                return False

            self.gsim.on_episode_end(1)
            self.starlink_manipulator.on_episode_end(1)
            self.target_object.on_episode_end(1)

            self.gsim.reset()

            if self.initial_joint_positions is None:
                self.initial_joint_positions = self._as_task_tensor(
                    self.joint_reference, expected_last_dim=6
                ).detach().clone()
            if self.initial_target_pos is None or self.initial_target_quat is None:
                target_link = self.target_object.robot.get_link(self.target_link_name)
                self.initial_target_pos = self._as_task_tensor(
                    target_link.get_pos(), expected_last_dim=3
                ).detach().clone()
                self.initial_target_quat = self._as_task_tensor(
                    target_link.get_quat(), expected_last_dim=4
                ).detach().clone()

            if self.num_envs > 1:
                base_pos = self.initial_base_pos
                base_quat = self.initial_base_quat
                target_pos_noise, yaw_quat = self._sample_target_reset_noise(self.num_envs)
                target_pos = self.initial_target_pos + target_pos_noise
                target_quat = self.initial_target_quat
                relative_yaw_quat = yaw_quat
            else:
                base_pos = self.initial_base_pos[0]
                base_quat = self.initial_base_quat[0]
                target_pos_noise, yaw_quat = self._sample_target_reset_noise(1)
                target_pos = (self.initial_target_pos + target_pos_noise)[0]
                target_quat = self.initial_target_quat[0]
                relative_yaw_quat = yaw_quat[0]

            all_dofs = self.starlink_manipulator.motors_dof + self.starlink_manipulator.fingers_dof
            initial_full_dofs = self.starlink_manipulator.config["initial_dofs"].detach().clone()
            if self.num_envs > 1:
                initial_full_dofs = initial_full_dofs.unsqueeze(0).repeat(self.num_envs, 1)
                initial_full_dofs[:, :6] = self.initial_joint_positions
                zero_velocity = torch.zeros_like(initial_full_dofs)
            else:
                initial_full_dofs[:6] = self.initial_joint_positions.reshape(-1)
                zero_velocity = [0.0] * len(all_dofs)
            self._apply_reset_state(
                base_pos,
                base_quat,
                target_pos,
                target_quat,
                relative_yaw_quat,
                initial_full_dofs,
                zero_velocity,
            )

            self.starlink_manipulator.update_state()
            if hasattr(self.starlink_manipulator, "pid"):
                self.starlink_manipulator.pid.reset()
                self.starlink_manipulator.capture_current_base_pose_as_pid_setpoint()

            self.starlink_manipulator.control_gripper(self.gripper_open_value)
            self.gsim.step()
            self.starlink_manipulator.update_state()
            self._update_io_tensors()

            self._reset_task_state_buffers()
            self.status = TaskStatus.RUNNING
            return True
        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def stop(self) -> bool:
        try:
            if self.target_object is not None:
                self.target_object.stop()
            if self.starlink_manipulator is not None:
                self.starlink_manipulator.stop()
            self.gsim.stop()

            if self.status not in {TaskStatus.SUCCEED, TaskStatus.FAILED}:
                self.status = TaskStatus.CANCELLED
            return True
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
            self.status = TaskStatus.FAILED
            return False


if gym is not None and spaces is not None:
    class SatelliteSideGraspEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            target_asset_name: str = "starlink",
            num_envs: int = None,
            env_spacing=None,
            n_envs_per_row=None,
            center_envs_at_origin: bool = None,
        ):
            super().__init__()
            self.task = RLTask(
                target_asset_name=target_asset_name,
                num_envs=num_envs,
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
            )
            if self.task.num_envs != 1:
                raise ValueError(
                    "SatelliteSideGraspEnv is the serial Gym wrapper and only supports "
                    f"num_envs=1, got {self.task.num_envs}."
                )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32
            )

        def _get_obs(self):
            return self.task.observation.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError("RLTask initialize failed during env reset")
            else:
                if not self.task.reset():
                    raise RuntimeError("RLTask reset failed during env reset")
            return self._get_obs(), {}

        def step(self, action):
            keep_running = self.task.step(action=action)
            obs = self._get_obs()
            reward = float(self.task.reward().item())
            terminated = self.task.status in {TaskStatus.SUCCEED, TaskStatus.FAILED}
            truncated = (not keep_running) and self.task.status == TaskStatus.COMPLETED
            info = {
                "phase": self.task.phase,
                "base_alignment_error": float(self.task.base_alignment_error.item()),
                "tcp_to_grasp_error": float(self.task.tcp_to_grasp_error.item()),
                "success": self.task.success,
            }
            return obs, reward, terminated, truncated, info

        def close(self):
            self.task.stop()


def train_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_satellite_side_grasp",
    device: str = "auto",
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    if PPO is None:
        raise ImportError("stable_baselines3 is not installed. Please install stable-baselines3 first.")

    env = SatelliteSideGraspEnv()
    training_device = resolve_training_device(device)
    model = PPO("MlpPolicy", env, verbose=1, device=training_device)
    print(f"Using PPO training device: {training_device}")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    return model


def describe_observation_layout():
    return {
        "tcp_pose": 7,
        "target_grasp_pose": 7,
        "base_satellite_pose": 7,
        "total": 21,
    }


def describe_action_layout():
    return {
        "arm_joint_angles": 6,
        "base_planar_motion": 2,
        "total": 8,
    }
