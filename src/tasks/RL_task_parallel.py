"""Batched parallel RL task wrapper for the satellite side grasp task.

This module moves away from ``DummyVecEnv/SubprocVecEnv`` and instead treats
Genesis batched scenes as the primary source of parallelism.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from tasks.RL_task import (  # noqa: E402
    PPO,
    RLTask,
    TaskStatus,
    describe_action_layout,
    describe_observation_layout,
    gym,
    resolve_training_device,
    spaces,
)
from utils.utils import as_rotation_matrix  # noqa: E402

try:
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
except Exception:
    VecMonitor = None
    VecEnv = None
    VecEnvStepReturn = None


def _as_batched_tensor(value, *, device, dtype, expected_last_dim: Optional[int] = None):
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Expected tensor with trailing dim {expected_last_dim}, got {tuple(tensor.shape)}"
        )
    return tensor


class ParallelRLTask(RLTask):
    """Batched variant of ``RLTask`` built on Genesis parallel scenes."""

    def __init__(
        self,
        target_asset_name: str = "starlink",
        num_envs: int = 1,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
        enable_visual_obs: bool = False,
        image_shape: Sequence[int] = (3, 84, 84),
        normalize_images: bool = False,
    ):
        super().__init__(
            target_asset_name=target_asset_name,
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
        )
        self.task_name = "RL_task_parallel"
        self.env_done = None
        self.env_truncated = None
        self.enable_visual_obs = bool(enable_visual_obs)
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.normalize_images = bool(normalize_images)
        self.state_observation = None
        self.image_observation = None
        self.reach_distance_threshold = 0.06
        self.hold_velocity_threshold = 0.02
        self.success_hold_steps = 15
        self.hold_zone_bonus = 0.5
        self.stable_hold_bonus = 1.0
        self.success_bonus = 100.0
        self.progress_reward_scale = 5.0
        self.distance_penalty_scale = 0.5
        self.action_penalty_scale = 1e-3
        self.target_hold_count = None
        self.prev_tcp_to_grasp_error = None
        self.prev_tcp_pos = None
        self.tcp_speed = None
        self.last_action_tensor = None

    def _zero_image_observation(self) -> torch.Tensor:
        dtype = torch.float32 if self.normalize_images else torch.uint8
        return torch.zeros(
            (self.num_envs, *self.image_shape),
            dtype=dtype,
            device=self.device,
        )

    def _capture_visual_observation(self) -> Optional[torch.Tensor]:
        if not self.enable_visual_obs:
            return None

        camera_sensor = getattr(self.starlink_manipulator, "rl_wrist_camera", None)
        if camera_sensor is None:
            return self._zero_image_observation()

        rgb = camera_sensor.capture()
        rgb = torch.as_tensor(rgb, device=self.device)
        expected_shape = (self.num_envs, *self.image_shape)
        if tuple(rgb.shape) != expected_shape:
            raise ValueError(
                "Unexpected batched wrist camera output shape: "
                f"expected {expected_shape}, got {tuple(rgb.shape)}"
            )

        if self.normalize_images:
            rgb = rgb.float()
            if rgb.max().item() > 1.0:
                rgb = rgb / 255.0
        else:
            if rgb.dtype != torch.uint8:
                rgb = torch.clamp(rgb, 0, 255)
                if rgb.max().item() <= 1.0:
                    rgb = rgb * 255.0
                rgb = rgb.to(torch.uint8)
        return rgb

    def _reset_parallel_buffers(self):
        self.env_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.env_truncated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _ensure_parallel_state_buffers(self):
        self.total_reward = torch.as_tensor(
            self.total_reward, dtype=self.datatype, device=self.device
        ).reshape(self.num_envs)
        self.success = torch.as_tensor(
            self.success, dtype=torch.bool, device=self.device
        ).reshape(self.num_envs)
        self.edge_contact_hold_count = torch.as_tensor(
            self.edge_contact_hold_count, dtype=torch.int32, device=self.device
        ).reshape(self.num_envs)
        self.edge_contact_active = torch.as_tensor(
            self.edge_contact_active, dtype=torch.bool, device=self.device
        ).reshape(self.num_envs)
        self.last_gripper_command = torch.as_tensor(
            self.last_gripper_command, dtype=self.datatype, device=self.device
        ).reshape(self.num_envs)

    def _update_hold_metrics(self, tcp_pos: torch.Tensor):
        if self.prev_tcp_pos is None:
            self.prev_tcp_pos = tcp_pos.detach().clone()
            self.tcp_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
        else:
            self.tcp_speed = torch.norm(tcp_pos - self.prev_tcp_pos, dim=-1)
            self.prev_tcp_pos = tcp_pos.detach().clone()

        if self.target_hold_count is None:
            self.target_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )

        hold_mask = (
            (self.tcp_to_grasp_error <= self.reach_distance_threshold)
            & (self.tcp_speed <= self.hold_velocity_threshold)
            & (self.approach_alignment >= self.approach_alignment_threshold)
        )
        self.target_hold_count = torch.where(
            hold_mask,
            self.target_hold_count + 1,
            torch.zeros_like(self.target_hold_count),
        )

    def _get_tcp_pose(self):
        ee_link = self.starlink_manipulator.robot.get_link(
            self.starlink_manipulator.params["end_effector"]
        )
        ee_pos = _as_batched_tensor(
            ee_link.get_pos(), device=self.device, dtype=self.datatype, expected_last_dim=3
        )
        ee_quat = _as_batched_tensor(
            ee_link.get_quat(), device=self.device, dtype=self.datatype, expected_last_dim=4
        )
        tcp_positions: List[torch.Tensor] = []
        for env_idx in range(ee_pos.shape[0]):
            ee_rot = as_rotation_matrix(ee_quat[env_idx], order="wxyz")
            tcp_positions.append(ee_pos[env_idx] + torch.matmul(ee_rot, self.tcp_offset_local))
        tcp_pos = torch.stack(tcp_positions, dim=0)
        return tcp_pos, ee_quat

    def _get_target_grasp_pose(self):
        target_link = self.target_object.robot.get_link(self.target_link_name)
        target_pos = _as_batched_tensor(
            target_link.get_pos(), device=self.device, dtype=self.datatype, expected_last_dim=3
        )
        target_quat = _as_batched_tensor(
            target_link.get_quat(), device=self.device, dtype=self.datatype, expected_last_dim=4
        )
        grasp_positions: List[torch.Tensor] = []
        for env_idx in range(target_pos.shape[0]):
            target_rot = as_rotation_matrix(target_quat[env_idx], order="wxyz")
            grasp_positions.append(
                target_pos[env_idx] + torch.matmul(target_rot, self.target_grasp_offset_local)
            )
        grasp_point_pos = torch.stack(grasp_positions, dim=0)
        return grasp_point_pos, target_quat

    def _get_base_pose(self):
        root_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=list(range(7))
        )
        root_qpos = _as_batched_tensor(
            root_qpos, device=self.device, dtype=self.datatype, expected_last_dim=7
        )
        return root_qpos[:, :3], root_qpos[:, 3:7]

    def _compute_approach_alignment(self, tcp_quat: torch.Tensor) -> torch.Tensor:
        if tcp_quat.dim() == 1:
            return super()._compute_approach_alignment(tcp_quat)

        local_approach_axis = self.tcp_offset_local / torch.clamp(
            torch.norm(self.tcp_offset_local),
            min=torch.tensor(1e-6, dtype=self.datatype, device=self.device),
        )
        world_down_axis = torch.tensor(
            [0.0, 0.0, -1.0], dtype=self.datatype, device=self.device
        )
        alignments = []
        for env_idx in range(tcp_quat.shape[0]):
            tcp_rot = as_rotation_matrix(tcp_quat[env_idx], order="wxyz")
            world_approach_axis = torch.matmul(tcp_rot, local_approach_axis)
            alignments.append(torch.dot(world_approach_axis, world_down_axis))
        return torch.stack(alignments, dim=0)

    def _check_edge_grasp_contact(self):
        self.edge_contact_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.edge_contact_hold_count = torch.where(
            self.edge_contact_active,
            self.edge_contact_hold_count + 1,
            torch.zeros_like(self.edge_contact_hold_count),
        )
        return self.edge_contact_active

    def _update_io_tensors(self):
        tcp_pos, tcp_quat = self._get_tcp_pose()
        grasp_point_pos, grasp_point_quat = self._get_target_grasp_pose()
        self._sync_grasp_point_marker(grasp_point_pos)
        if self.debug_grasp_marker:
            if self.step_count == 0 or self.step_count % self.debug_grasp_marker_interval == 0:
                target_link = self.target_object.robot.get_link(self.target_link_name)
                target_link_pos = _as_batched_tensor(
                    target_link.get_pos(),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=3,
                ).detach().cpu()
                grasp_point_pos_cpu = grasp_point_pos.detach().cpu()
                if self.grasp_point_marker is None:
                    marker_pos_cpu = None
                else:
                    marker_pos_cpu = _as_batched_tensor(
                        self.grasp_point_marker.get_pos(),
                        device=self.device,
                        dtype=self.datatype,
                        expected_last_dim=3,
                    ).detach().cpu().tolist()
                print(
                    "[GRASP_MARKER_DEBUG_PARALLEL]",
                    f"step={self.step_count}",
                    f"env_step_count={int(self.env_step_count[0].item()) if self.env_step_count is not None else 'NA'}",
                    f"env=0",
                    f"target_link_pos={target_link_pos[0].tolist()}",
                    f"grasp_point_pos={grasp_point_pos_cpu[0].tolist()}",
                    f"marker_pos={None if marker_pos_cpu is None else marker_pos_cpu[0]}",
                    f"tcp_to_grasp_error={float(torch.norm(grasp_point_pos_cpu[0] - tcp_pos.detach().cpu()[0]).item())}",
                    flush=True,
                )
        base_pos, base_quat = self._get_base_pose()
        joint_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )
        joint_qpos = _as_batched_tensor(
            joint_qpos, device=self.device, dtype=self.datatype, expected_last_dim=6
        )

        self.tcp_pose = torch.cat([tcp_pos, tcp_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.target_grasp_pose = torch.cat([grasp_point_pos, grasp_point_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.base_satellite_pose = torch.cat([base_pos, base_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.arm_joint_positions = joint_qpos.reshape(self.num_envs, -1)
        self.base_planar_position = base_pos[:, :2].to(dtype=self.datatype, device=self.device)

        self.state_observation = torch.cat(
            [self.tcp_pose, self.base_satellite_pose, self.arm_joint_positions], dim=-1
        )
        if self.enable_visual_obs:
            self.image_observation = self._capture_visual_observation()
            self.observation = {
                "state": self.state_observation,
                "image": self.image_observation,
            }
        else:
            self.image_observation = None
            self.observation = self.state_observation
        self.action = torch.cat(
            [self.arm_joint_positions, self.base_planar_position], dim=-1
        )

        self.base_alignment_error = torch.norm(
            grasp_point_pos[:, :2] - self.base_planar_position, dim=-1
        )
        self.tcp_to_grasp_error = torch.norm(grasp_point_pos - tcp_pos, dim=-1)
        self.approach_alignment = self._compute_approach_alignment(tcp_quat)
        self._update_hold_metrics(tcp_pos)

        self.phase = np.where(
            self.base_alignment_error.detach().cpu().numpy() <= self.base_alignment_threshold,
            "arm_approach",
            "base_alignment",
        ).astype(object)

        grasp_mask = self.target_hold_count >= self.success_hold_steps
        grasp_mask_np = grasp_mask.detach().cpu().numpy()
        self.phase = np.where(grasp_mask_np, "grasp", self.phase).astype(object)

    def initialize(self) -> bool:
        result = super().initialize()
        if result:
            self._ensure_parallel_state_buffers()
            self._reset_parallel_buffers()
            self.target_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.prev_tcp_to_grasp_error = self.tcp_to_grasp_error.detach().clone()
            self.prev_tcp_pos = self._get_tcp_pose()[0].detach().clone()
            self.tcp_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.last_action_tensor = torch.zeros(
                (self.num_envs, 8), dtype=self.datatype, device=self.device
            )
        return result

    def _apply_base_action(self, planar_action: torch.Tensor):
        planar_action = _as_batched_tensor(
            planar_action, device=self.device, dtype=self.datatype, expected_last_dim=2
        )
        force_xy = planar_action * self.base_force_scale_xy
        zeros = torch.zeros((force_xy.shape[0], 1), dtype=self.datatype, device=self.device)
        force_cmd = torch.cat([force_xy, zeros], dim=-1)
        self.starlink_manipulator.apply_force(
            force=force_cmd,
            link_name=self.starlink_manipulator.params["base"],
        )

    def _apply_arm_action(self, joint_action: torch.Tensor):
        joint_action = _as_batched_tensor(
            joint_action, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        target_joint = self.arm_joint_positions + joint_action * self.joint_action_scale
        self.starlink_manipulator.control_joint_pos(target_joint)

    def apply_rl_action(self, action: torch.Tensor):
        action = _as_batched_tensor(
            action, device=self.device, dtype=self.datatype, expected_last_dim=8
        )
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Batched RL action should have shape ({self.num_envs}, 8), got {tuple(action.shape)}"
            )

        joint_action = torch.clamp(action[:, :6], -1.0, 1.0)
        planar_action = torch.clamp(action[:, 6:], -1.0, 1.0)
        self.last_action_tensor = action.detach().clone()

        self._apply_base_action(planar_action)
        self._apply_arm_action(joint_action)
        self.last_gripper_command = torch.full(
            (self.num_envs,),
            self.gripper_open_value,
            dtype=self.datatype,
            device=self.device,
        )
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
            self.total_reward = self.total_reward + reward
            self.step_count += 1
            self.env_step_count += 1
            self.env_episode_return += reward

            success_mask = self.is_success()
            self.success = success_mask
            truncated_mask = self.env_step_count >= self.max_steps
            self.env_done = success_mask.detach().clone()
            self.env_truncated = truncated_mask.detach().clone()
            self.last_done_mask = self.env_done.detach().clone()
            self.last_truncated_mask = self.env_truncated.detach().clone()

            if bool(torch.all(success_mask).item()):
                self.status = TaskStatus.SUCCEED
                return False

            if bool(torch.all(truncated_mask).item()):
                self.status = TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Parallel step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reward(self) -> torch.Tensor:
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            return torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)

        if self.prev_tcp_to_grasp_error is None:
            progress_reward = torch.zeros_like(self.tcp_to_grasp_error)
        else:
            progress_reward = self.prev_tcp_to_grasp_error - self.tcp_to_grasp_error
        self.prev_tcp_to_grasp_error = self.tcp_to_grasp_error.detach().clone()

        reward = self.progress_reward_scale * progress_reward
        reward = reward - self.distance_penalty_scale * self.tcp_to_grasp_error
        reward = reward + torch.where(
            self.tcp_to_grasp_error <= self.reach_distance_threshold,
            torch.full_like(reward, self.hold_zone_bonus),
            torch.zeros_like(reward),
        )
        stable_hold_mask = (
            (self.tcp_to_grasp_error <= self.reach_distance_threshold)
            & (self.tcp_speed <= self.hold_velocity_threshold)
            & (self.approach_alignment >= self.approach_alignment_threshold)
        )
        reward = reward + torch.where(
            stable_hold_mask,
            torch.full_like(reward, self.stable_hold_bonus),
            torch.zeros_like(reward),
        )
        reward = reward + torch.where(
            self.is_success(),
            torch.full_like(reward, self.success_bonus),
            torch.zeros_like(reward),
        )
        if self.last_action_tensor is not None:
            reward = reward - self.action_penalty_scale * torch.sum(
                self.last_action_tensor ** 2, dim=-1
            )
        reward = reward - torch.where(
            self.is_success(),
            torch.zeros_like(reward),
            torch.full_like(reward, self.time_penalty),
        )
        return reward.to(dtype=self.datatype, device=self.device)

    def is_success(self):
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.target_hold_count >= self.success_hold_steps

    def reset(self) -> bool:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return self.reset_done_envs(env_ids=env_ids)

    def reset_done_envs(self, env_ids=None) -> bool:
        if env_ids is None:
            return self.reset()

        env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return True

        try:
            env_ids_list = env_ids.detach().cpu().tolist()
            self._ensure_parallel_state_buffers()
            print(
                "[PARTIAL_RESET_DEBUG]",
                f"global_step={self.step_count}",
                f"reset_count={int(env_ids.numel())}",
                f"env_ids={env_ids_list}",
                f"includes_env0={0 in env_ids_list}",
                flush=True,
            )

            probe_env_id = None
            for candidate_env_id in range(self.num_envs):
                if candidate_env_id not in env_ids_list:
                    probe_env_id = candidate_env_id
                    break

            finished_episodes = int(env_ids.numel())
            self.gsim.on_episode_end(finished_episodes)
            self.starlink_manipulator.on_episode_end(finished_episodes)
            self.target_object.on_episode_end(finished_episodes)

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

            target_link = self.target_object.robot.get_link(self.target_link_name)
            base_pos = self.initial_base_pos[env_ids].detach().clone()
            base_quat = self.initial_base_quat[env_ids].detach().clone()
            target_pos_noise, yaw_quat = self._sample_target_reset_noise(int(env_ids.numel()))
            target_pos = self.initial_target_pos[env_ids].detach().clone() + target_pos_noise
            target_quat = self.initial_target_quat[env_ids].detach().clone()
            relative_yaw_quat = yaw_quat.detach().clone()

            all_dofs = self.starlink_manipulator.motors_dof + self.starlink_manipulator.fingers_dof
            current_dofs = self.starlink_manipulator.config["initial_dofs"].detach().clone()
            current_dofs = current_dofs.unsqueeze(0).repeat(int(env_ids.numel()), 1)
            current_dofs[:, :6] = self.initial_joint_positions[env_ids]

            probe_before = None
            if probe_env_id is not None:
                probe_root_qpos_before = _as_batched_tensor(
                    self.starlink_manipulator.robot.get_qpos(qs_idx_local=list(range(7))),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=7,
                ).detach().clone()
                probe_target_pos_before = _as_batched_tensor(
                    target_link.get_pos(),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=3,
                ).detach().clone()
                probe_target_quat_before = _as_batched_tensor(
                    target_link.get_quat(),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=4,
                ).detach().clone()
                probe_dofs_before = _as_batched_tensor(
                    self.starlink_manipulator.robot.get_qpos(qs_idx_local=all_dofs),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=len(all_dofs),
                ).detach().clone()
                probe_before = {
                    "base_pos": probe_root_qpos_before[probe_env_id, :3].detach().clone(),
                    "base_quat": probe_root_qpos_before[probe_env_id, 3:7].detach().clone(),
                    "target_pos": probe_target_pos_before[probe_env_id].detach().clone(),
                    "target_quat": probe_target_quat_before[probe_env_id].detach().clone(),
                    "dofs": probe_dofs_before[probe_env_id].detach().clone(),
                }

            zero_velocity = torch.zeros_like(current_dofs)

            self._apply_reset_state(
                base_pos,
                base_quat,
                target_pos,
                target_quat,
                relative_yaw_quat,
                current_dofs,
                zero_velocity,
                env_ids=env_ids,
            )
            self.starlink_manipulator.update_state()

            if isinstance(self.last_gripper_command, torch.Tensor):
                updated_gripper_command = self.last_gripper_command.detach().clone()
                updated_gripper_command[env_ids] = self.gripper_open_value
            else:
                updated_gripper_command = torch.full(
                    (self.num_envs,),
                    self.gripper_open_value,
                    dtype=self.datatype,
                    device=self.device,
                )
            self.starlink_manipulator.control_gripper(updated_gripper_command)

            if hasattr(self.starlink_manipulator, "pid"):
                current_setpoint = self.starlink_manipulator.pid.setpoint.detach().clone()
                refreshed_setpoint = torch.cat(
                    [
                        self.starlink_manipulator.ee_state.link_parent_global_state.position.detach().clone(),
                        self.starlink_manipulator.ee_state.link_parent_global_state.orient.detach().clone(),
                    ],
                    dim=-1,
                )
                print(
                    "[PID_SETPOINT_DEBUG]",
                    f"step={self.step_count}",
                    f"current_setpoint_shape={tuple(current_setpoint.shape)}",
                    f"refreshed_setpoint_shape={tuple(refreshed_setpoint.shape)}",
                    f"dim_match={(current_setpoint.dim() == 2 and refreshed_setpoint.dim() == 2)}",
                    f"env_ids={env_ids_list}",
                    f"setpoint_norm={float(torch.norm(current_setpoint).item()):.4f}",
                    f"refreshed_norm={float(torch.norm(refreshed_setpoint).item()):.4f}",
                    f"orient_sample={refreshed_setpoint[0,3:].detach().cpu().tolist() if refreshed_setpoint.dim()>1 else refreshed_setpoint[3:].detach().cpu().tolist()}",
                    flush=True,
                )
                if current_setpoint.dim() == 2 and refreshed_setpoint.dim() == 2:
                    current_setpoint[env_ids] = refreshed_setpoint[env_ids]
                else:
                    current_setpoint = refreshed_setpoint
                if hasattr(self.starlink_manipulator.pid, "reset_envs"):
                    self.starlink_manipulator.pid.reset_envs(env_ids)
                self.starlink_manipulator.pid.update_setpoint(current_setpoint)
                print(
                    "[PID_SETPOINT_AFTER]",
                    f"updated_setpoint_shape={tuple(self.starlink_manipulator.pid.setpoint.shape)}",
                    f"setpoint_0={self.starlink_manipulator.pid.setpoint[0].detach().cpu().tolist() if self.starlink_manipulator.pid.setpoint.dim()>1 else self.starlink_manipulator.pid.setpoint.detach().cpu().tolist()}",
                    flush=True,
                )
            self._update_io_tensors()

            if probe_before is not None:
                probe_target_link = self.target_object.robot.get_link(self.target_link_name)
                probe_root_qpos_after = _as_batched_tensor(
                    self.starlink_manipulator.robot.get_qpos(qs_idx_local=list(range(7))),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=7,
                ).detach().clone()
                probe_target_pos_after = _as_batched_tensor(
                    probe_target_link.get_pos(),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=3,
                ).detach().clone()
                probe_target_quat_after = _as_batched_tensor(
                    probe_target_link.get_quat(),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=4,
                ).detach().clone()
                probe_dofs_after = _as_batched_tensor(
                    self.starlink_manipulator.robot.get_qpos(qs_idx_local=all_dofs),
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=len(all_dofs),
                ).detach().clone()

                probe_base_pos_delta = torch.norm(
                    probe_root_qpos_after[probe_env_id, :3] - probe_before["base_pos"]
                ).item()
                probe_base_quat_delta = torch.norm(
                    probe_root_qpos_after[probe_env_id, 3:7] - probe_before["base_quat"]
                ).item()
                probe_target_pos_delta = torch.norm(
                    probe_target_pos_after[probe_env_id] - probe_before["target_pos"]
                ).item()
                probe_target_quat_delta = torch.norm(
                    probe_target_quat_after[probe_env_id] - probe_before["target_quat"]
                ).item()
                probe_dofs_delta = torch.norm(
                    probe_dofs_after[probe_env_id] - probe_before["dofs"]
                ).item()

                if max(
                    probe_base_pos_delta,
                    probe_base_quat_delta,
                    probe_target_pos_delta,
                    probe_target_quat_delta,
                    probe_dofs_delta,
                ) > 1e-5:
                    print(
                        "[RESET_SIDE_EFFECT_DEBUG]",
                        f"global_step={self.step_count}",
                        f"reset_env_ids={env_ids_list}",
                        f"probe_env={probe_env_id}",
                        f"base_pos_delta={probe_base_pos_delta:.6f}",
                        f"base_quat_delta={probe_base_quat_delta:.6f}",
                        f"target_pos_delta={probe_target_pos_delta:.6f}",
                        f"target_quat_delta={probe_target_quat_delta:.6f}",
                        f"dofs_delta={probe_dofs_delta:.6f}",
                        flush=True,
                    )

            self.env_step_count[env_ids] = 0
            self.env_episode_return[env_ids] = 0.0
            self.total_reward[env_ids] = 0.0
            self.success[env_ids] = False
            self.edge_contact_hold_count[env_ids] = 0
            self.edge_contact_active[env_ids] = False
            self.target_hold_count[env_ids] = 0
            self.prev_tcp_to_grasp_error[env_ids] = self.tcp_to_grasp_error[env_ids]
            self.prev_tcp_pos[env_ids] = self._get_tcp_pose()[0][env_ids].detach().clone()
            self.tcp_speed[env_ids] = 0.0
            self.last_gripper_command = updated_gripper_command
            self.env_done[env_ids] = False
            self.env_truncated[env_ids] = False
            self.last_done_mask[env_ids] = False
            self.last_truncated_mask[env_ids] = False

            self.status = TaskStatus.RUNNING
            return True
        except Exception as e:
            self.logger.error(f"Partial reset failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def get_info(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            infos.append(
                {
                    "phase": str(self.phase[env_idx]),
                    "base_alignment_error": float(self.base_alignment_error[env_idx].item()),
                    "tcp_to_grasp_error": float(self.tcp_to_grasp_error[env_idx].item()),
                    "success": bool(self.success[env_idx].item()),
                }
            )
        return infos


if gym is not None and spaces is not None:
    class ParallelSatelliteSideGraspEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            target_asset_name: str = "starlink",
            num_envs: int = 4,
            env_spacing=None,
            n_envs_per_row=None,
            center_envs_at_origin: bool = None,
            enable_visual_obs: bool = False,
            image_shape: Sequence[int] = (3, 84, 84),
            normalize_images: bool = False,
        ):
            super().__init__()
            self.task = ParallelRLTask(
                target_asset_name=target_asset_name,
                num_envs=num_envs,
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
                enable_visual_obs=enable_visual_obs,
                image_shape=image_shape,
                normalize_images=normalize_images,
            )
            self.num_envs = self.task.num_envs
            if self.task.enable_visual_obs:
                image_low = 0.0 if self.task.normalize_images else 0
                image_high = 1.0 if self.task.normalize_images else 255
                image_dtype = np.float32 if self.task.normalize_images else np.uint8
                self.single_observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(20,),
                            dtype=np.float32,
                        ),
                        "image": spaces.Box(
                            low=image_low,
                            high=image_high,
                            shape=tuple(self.task.image_shape),
                            dtype=image_dtype,
                        ),
                    }
                )
            else:
                self.single_observation_space = spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(20,), dtype="float32"
                )
            self.single_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype="float32"
            )
            if self.task.enable_visual_obs:
                image_low = 0.0 if self.task.normalize_images else 0
                image_high = 1.0 if self.task.normalize_images else 255
                image_dtype = np.float32 if self.task.normalize_images else np.uint8
                self.observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self.num_envs, 20),
                            dtype=np.float32,
                        ),
                        "image": spaces.Box(
                            low=image_low,
                            high=image_high,
                            shape=(self.num_envs, *tuple(self.task.image_shape)),
                            dtype=image_dtype,
                        ),
                    }
                )
            else:
                self.observation_space = spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.num_envs, 20),
                    dtype="float32",
                )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_envs, 8),
                dtype="float32",
            )

        def _get_obs(self):
            if self.task.enable_visual_obs:
                state_obs = self.task.state_observation.detach().cpu().numpy().astype("float32")
                image_obs = self.task.image_observation.detach().cpu().numpy()
                if self.task.normalize_images:
                    image_obs = image_obs.astype("float32")
                return {
                    "state": state_obs,
                    "image": image_obs,
                }
            return self.task.observation.detach().cpu().numpy().astype("float32")

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError(
                        "ParallelRLTask initialize failed during env reset"
                    )
            else:
                if not self.task.reset():
                    raise RuntimeError("ParallelRLTask reset failed during env reset")
            return self._get_obs(), {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            keep_running = self.task.step(action=action)
            obs = self._get_obs()
            reward = self.task.reward().detach().cpu().numpy().astype(np.float32)
            terminated = self.task.get_done_mask().detach().cpu().numpy().astype(bool)
            truncated = self.task.get_truncated_mask().detach().cpu().numpy().astype(bool)
            info = self.task.get_info()
            return obs, reward, terminated, truncated, info

        def close(self):
            self.task.stop()


if VecEnv is not None and VecEnvStepReturn is not None:
    class BatchedGenesisVecEnv(VecEnv):
        """SB3-compatible VecEnv wrapper over one batched Genesis scene."""

        def __init__(
            self,
            target_asset_name: str = "starlink",
            num_envs: int = 4,
            env_spacing=None,
            n_envs_per_row=None,
            center_envs_at_origin: bool = None,
            enable_visual_obs: bool = False,
            image_shape: Sequence[int] = (3, 84, 84),
            normalize_images: bool = False,
        ):
            self.env = ParallelSatelliteSideGraspEnv(
                target_asset_name=target_asset_name,
                num_envs=num_envs,
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
                enable_visual_obs=enable_visual_obs,
                image_shape=image_shape,
                normalize_images=normalize_images,
            )
            super().__init__(
                num_envs=self.env.num_envs,
                observation_space=self.env.single_observation_space,
                action_space=self.env.single_action_space,
            )
            self._pending_actions = None
            self.reset_infos = [{} for _ in range(self.num_envs)]

        def reset(self):
            obs, reset_info = self.env.reset()
            self.reset_infos = [dict(reset_info) for _ in range(self.num_envs)]
            return obs

        def step_async(self, actions: np.ndarray) -> None:
            self._pending_actions = actions

        def step_wait(self) -> VecEnvStepReturn:
            if self._pending_actions is None:
                raise RuntimeError("step_async must be called before step_wait")
            obs, rewards, terminated, truncated, infos = self.env.step(self._pending_actions)
            dones = np.logical_or(terminated, truncated)
            done_indices = np.flatnonzero(dones)
            for idx, done in enumerate(dones):
                if done:
                    infos[idx] = dict(infos[idx])
                    if isinstance(obs, dict):
                        infos[idx]["terminal_observation"] = {
                            key: value[idx].copy() for key, value in obs.items()
                        }
                    else:
                        infos[idx]["terminal_observation"] = obs[idx].copy()
                    infos[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            if done_indices.size > 0:
                if not self.env.task.reset_done_envs(done_indices.tolist()):
                    raise RuntimeError("ParallelRLTask partial reset failed during VecEnv step")
                obs = self.env._get_obs()
            self._pending_actions = None
            return obs, rewards, dones, infos

        def close(self) -> None:
            self.env.close()

        def get_attr(self, attr_name, indices=None):
            indices = self._get_indices(indices)
            value = getattr(self.env.task, attr_name)
            return [value for _ in indices]

        def set_attr(self, attr_name, value, indices=None) -> None:
            indices = self._get_indices(indices)
            if len(indices) != self.num_envs:
                raise NotImplementedError(
                    "BatchedGenesisVecEnv only supports setting attributes for all envs at once."
                )
            setattr(self.env.task, attr_name, value)

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            indices = self._get_indices(indices)
            method = getattr(self.env, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        def env_is_wrapped(self, wrapper_class, indices=None):
            indices = self._get_indices(indices)
            return [False for _ in indices]


def build_parallel_env(
    target_asset: str = "starlink",
    num_envs: int = 4,
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
    enable_visual_obs: bool = False,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    return ParallelSatelliteSideGraspEnv(
        target_asset_name=target_asset,
        num_envs=num_envs,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=normalize_images,
    )


def make_parallel_vec_env(
    num_envs: int = 4,
    target_asset: str = "starlink",
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
    enable_visual_obs: bool = False,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
):
    if VecEnv is None or VecMonitor is None:
        raise ImportError(
            "stable_baselines3 vectorized env utilities are not installed."
        )
    vec_env = BatchedGenesisVecEnv(
        target_asset_name=target_asset,
        num_envs=num_envs,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=normalize_images,
    )
    return VecMonitor(vec_env)


def train_parallel_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_satellite_side_grasp_parallel",
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    device: str = "auto",
    target_asset: str = "starlink",
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
):
    if PPO is None:
        raise ImportError(
            "stable_baselines3 is not installed. Please install stable-baselines3 first."
        )

    training_device = resolve_training_device(device)
    env = make_parallel_vec_env(
        num_envs=num_envs,
        target_asset=target_asset,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        seed=seed,
        device=training_device,
    )
    print(f"Using batched PPO training device: {training_device}")
    print(f"Using Genesis batched scene count: {num_envs}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.close()
    return model


__all__ = [
    "ParallelRLTask",
    "ParallelSatelliteSideGraspEnv",
    "BatchedGenesisVecEnv",
    "build_parallel_env",
    "make_parallel_vec_env",
    "train_parallel_ppo",
    "describe_observation_layout",
    "describe_action_layout",
]
