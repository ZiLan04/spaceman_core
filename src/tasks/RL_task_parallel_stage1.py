# 输入：基座相机图像、基座位姿
# 输出：以当前基座原点为中心的局部平面目标位置偏移
# 用PID控制基座到达由局部偏移转换出的世界目标位置

"""Stage-1 batched RL task: move the satellite manipulator base to target.

The stage-1 policy commands a planar target offset in the current base frame.
The low-level base PID moves the base toward the resulting world target
position, while arm joints are held at the initial pose throughout the episode.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from tasks.RL_task_parallel import (  # noqa: E402
    PPO,
    ParallelRLTask,
    TaskStatus,
    VecEnv,
    VecEnvStepReturn,
    VecMonitor,
    _as_batched_tensor,
    gym,
    resolve_training_device,
    spaces,
)
from utils.utils import as_rotation_matrix  # noqa: E402


class ParallelStage1BaseTask(ParallelRLTask):
    """Batched stage-1 task for base-only target alignment."""

    def __init__(
        self,
        target_asset_name: str = "starlink",
        num_envs: int = 1,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
        enable_visual_obs: bool = True,
        image_shape: Sequence[int] = (3, 84, 84),
        normalize_images: bool = False,
        enable_target_relative_obs: Optional[bool] = None,
        target_position_noise: bool = False,
        target_position_noise_range=(0.03, 0.03, 0.02),
    ):
        super().__init__(
            target_asset_name=target_asset_name,
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
            enable_visual_obs=enable_visual_obs,
            image_shape=image_shape,
            normalize_images=normalize_images,
            target_position_noise=target_position_noise,
            target_position_noise_range=target_position_noise_range,
        )
        self.task_name = "RL_task_parallel_stage1"
        self.enable_target_relative_obs = (not self.enable_visual_obs) and (
            True if enable_target_relative_obs is None else bool(enable_target_relative_obs)
        )

        self.base_control_decimation = 10
        self.control_dt = self.control_dt * self.base_control_decimation
        self.max_steps = 220
        self.base_goal_threshold = 1.3
        self.reward_base_goal_threshold = None
        self.base_goal_band = 0.08
        self.base_stop_speed_threshold = 0.025
        self.success_hold_steps = 12
        self.base_goal_hold_count = None
        self.target_dock_dir_local = torch.tensor(
            [-1.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )
        self.desired_base_planar_position = None
        self.reward_desired_base_planar_position = None
        self.dock_pos_error = None
        self.episode_start_base_quat = None
        self.base_roll_pitch_error = None
        self.base_yaw_error = None
        self.env_failed = None

        self.base_goal_error_progress_reward_scale = 10.0
        self.base_goal_error_penalty_scale = 2.0
        self.base_speed_penalty_scale = 0.2
        self.near_goal_speed_penalty_scale = 3.0
        self.base_action_penalty_scale = 2e-3
        self.base_slow_zone_radius = 0.30
        self.slow_zone_error_penalty_scale = 4.0
        self.base_target_to_goal_penalty_scale = 0.5
        self.slow_zone_target_to_goal_penalty_scale = 2.0
        self.away_speed_penalty_scale = 2.0
        self.slow_zone_away_speed_penalty_scale = 5.0
        self.slow_zone_speed_penalty_scale = 3.0
        self.goal_band_entry_bonus = 180.0
        self.goal_band_leave_penalty = 300.0
        self.goal_band_hold_bonus = 8.0
        self.stable_hold_bonus = 15.0
        self.success_bonus = 400.0
        self.too_close_penalty_scale = 20.0
        self.base_runaway_distance_threshold = 2.5
        self.base_runaway_speed_threshold = 1.2
        self.base_roll_pitch_failure_threshold = 0.35
        self.base_yaw_failure_threshold = 0.8
        self.base_attitude_penalty_scale = 4.0
        self.base_yaw_penalty_scale = 1.0
        self.failure_penalty = 200.0
        self.prev_base_goal_error = None
        self.prev_in_goal_band = None

        self.base_relative_action_low_xy = torch.tensor(
            [-0.3, -0.3], dtype=self.datatype, device=self.device
        )
        self.base_relative_action_high_xy = torch.tensor(
            [0.3, 0.3], dtype=self.datatype, device=self.device
        )
        self.base_target_offset_command = None
        self.base_target_position_command = None
        self.prev_base_target_position_command = None
        self.debug_pid_trace = True
        self.debug_pid_trace_env_idx = 0
        self.debug_pid_trace_interval = 50
        fixed_action = os.environ.get("STAGE1_DEBUG_FIXED_ACTION", "").strip()
        self.debug_fixed_base_action = None
        if fixed_action:
            try:
                values = [float(item) for item in fixed_action.split(",")]
            except ValueError as exc:
                raise ValueError(
                    "STAGE1_DEBUG_FIXED_ACTION should be formatted as 'x,y', "
                    f"got {fixed_action!r}"
                ) from exc
            if len(values) != 2:
                raise ValueError(
                    "STAGE1_DEBUG_FIXED_ACTION should contain exactly two values, "
                    f"got {fixed_action!r}"
                )
            self.debug_fixed_base_action = torch.tensor(
                values, dtype=self.datatype, device=self.device
            )
            print(
                "[STAGE1_DEBUG] overriding policy base action with "
                f"{self.debug_fixed_base_action.detach().cpu().tolist()}",
                flush=True,
            )
        self.episode_start_base_pos = None
        self.last_step_reward = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )

    def _zero_stage1_image_observation(self) -> torch.Tensor:
        return self._zero_image_observation()

    def _capture_base_image_observation(self) -> Optional[torch.Tensor]:
        if not self.enable_visual_obs:
            return None
        return self._capture_named_visual_observation("rl_base_camera")

    def _update_base_motion_metrics(self, base_pos: torch.Tensor):
        if self.prev_base_pos is None:
            self.prev_base_pos = base_pos.detach().clone()
            self.base_planar_velocity = torch.zeros(
                (self.num_envs, 2), dtype=self.datatype, device=self.device
            )
            self.base_planar_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            return

        base_delta_xy = base_pos[:, :2] - self.prev_base_pos[:, :2]
        self.base_planar_velocity = base_delta_xy / self.control_dt
        self.base_planar_speed = torch.norm(self.base_planar_velocity, dim=-1)
        self.prev_base_pos = base_pos.detach().clone()

    def _sample_target_reset_noise(self, env_count: int):
        return super()._sample_target_reset_noise(env_count)

    def _compute_desired_base_xy(
        self,
        grasp_point_pos: torch.Tensor,
        grasp_point_quat: torch.Tensor,
        goal_distance: Optional[float] = None,
    ) -> torch.Tensor:
        if goal_distance is None:
            goal_distance = self.base_goal_threshold
        target_rot = as_rotation_matrix(grasp_point_quat, order="wxyz")
        dock_dir_world = torch.matmul(
            target_rot, self.target_dock_dir_local.view(3, 1)
        ).squeeze(-1)
        dock_dir_xy = dock_dir_world[:, :2]
        dock_dir_xy = dock_dir_xy / torch.clamp(
            torch.norm(dock_dir_xy, dim=-1, keepdim=True),
            min=1e-6,
        )
        return grasp_point_pos[:, :2] + dock_dir_xy * float(goal_distance)

    def _world_xy_to_base_local(
        self, world_xy: torch.Tensor, base_quat: torch.Tensor
    ) -> torch.Tensor:
        base_euler = self.starlink_manipulator.pid.quaternion_to_euler(base_quat)
        yaw = base_euler[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        local_x = cos_yaw * world_xy[:, 0] + sin_yaw * world_xy[:, 1]
        local_y = -sin_yaw * world_xy[:, 0] + cos_yaw * world_xy[:, 1]
        return torch.stack([local_x, local_y], dim=-1)

    def _build_non_visual_state_observation(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        last_target_offset_command: torch.Tensor,
    ) -> torch.Tensor:
        dock_target_offset_world_xy = (
            self.desired_base_planar_position - base_pos[:, :2]
        )
        dock_target_offset_local_xy = self._world_xy_to_base_local(
            dock_target_offset_world_xy, base_quat
        )
        dock_target_distance = torch.norm(
            dock_target_offset_local_xy, dim=-1, keepdim=True
        )
        base_planar_velocity_local_xy = self._world_xy_to_base_local(
            self.base_planar_velocity, base_quat
        )

        return torch.cat(
            [
                dock_target_offset_local_xy,
                dock_target_distance,
                base_planar_velocity_local_xy,
                last_target_offset_command,
            ],
            dim=-1,
        )

    def _update_goal_hold_count(self):
        if self.base_goal_hold_count is None:
            self.base_goal_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )

        if self.dock_pos_error is None:
            return

        stable_goal_mask = (
            (self.dock_pos_error <= self.base_goal_band)
            & (self.base_planar_speed <= self.base_stop_speed_threshold)
        )
        self.base_goal_hold_count = torch.where(
            stable_goal_mask,
            self.base_goal_hold_count + 1,
            torch.zeros_like(self.base_goal_hold_count),
        )

    def _update_io_tensors(self):
        grasp_point_pos, grasp_point_quat = self._get_target_grasp_pose()
        self._sync_grasp_point_marker(grasp_point_pos, grasp_point_quat)
        base_pos, base_quat = self._get_base_pose()
        joint_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )
        joint_qpos = _as_batched_tensor(
            joint_qpos, device=self.device, dtype=self.datatype, expected_last_dim=6
        )

        self.base_satellite_pose = torch.cat([base_pos, base_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.target_grasp_pose = torch.cat([grasp_point_pos, grasp_point_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.arm_joint_positions = joint_qpos.reshape(self.num_envs, -1)
        self.base_planar_position = base_pos[:, :2].to(dtype=self.datatype, device=self.device)

        self._update_base_motion_metrics(base_pos)
        if self.base_planar_velocity is None:
            self.base_planar_velocity = torch.zeros(
                (self.num_envs, 2), dtype=self.datatype, device=self.device
            )
        if self.base_planar_speed is None:
            self.base_planar_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )

        last_target_offset_command = (
            self.base_target_offset_command
            if self.base_target_offset_command is not None
            else torch.zeros((self.num_envs, 2), dtype=self.datatype, device=self.device)
        )
        self.desired_base_planar_position = self._compute_desired_base_xy(
            grasp_point_pos, grasp_point_quat
        )
        reward_goal_distance = (
            self.base_goal_threshold
            if self.reward_base_goal_threshold is None
            else self.reward_base_goal_threshold
        )
        self.reward_desired_base_planar_position = self._compute_desired_base_xy(
            grasp_point_pos,
            grasp_point_quat,
            goal_distance=reward_goal_distance,
        )

        if self.enable_visual_obs:
            self.state_observation = torch.cat(
                [
                    self.base_satellite_pose,
                    self.base_planar_velocity,
                    last_target_offset_command,
                ],
                dim=-1,
            )
            self.base_image_observation = self._capture_base_image_observation()
            self.image_observation = self.base_image_observation
            self.wrist_image_observation = None
            self.observation = {
                "state": self.state_observation,
                "image": self.base_image_observation,
            }
        else:
            self.state_observation = self._build_non_visual_state_observation(
                base_pos,
                base_quat,
                last_target_offset_command,
            )
            self.base_image_observation = None
            self.image_observation = None
            self.wrist_image_observation = None
            self.observation = self.state_observation

        self.action = (
            self.base_target_offset_command
            if self.base_target_offset_command is not None
            else torch.zeros((self.num_envs, 2), dtype=self.datatype, device=self.device)
        )
        self.base_alignment_error = torch.norm(
            grasp_point_pos[:, :2] - self.base_planar_position, dim=-1
        )
        self.dock_pos_error = torch.norm(
            self.base_planar_position - self.reward_desired_base_planar_position,
            dim=-1,
        )

        # Keep these populated for inherited reset/debug helpers.
        self.tcp_pose = torch.zeros((self.num_envs, 7), dtype=self.datatype, device=self.device)
        self.tcp_to_grasp_error = self.dock_pos_error.detach().clone()
        self.approach_alignment = torch.ones(
            self.num_envs, dtype=self.datatype, device=self.device
        )

        self._update_goal_hold_count()
        in_goal_band_np = (self.dock_pos_error <= self.base_goal_band).detach().cpu().numpy()
        self.phase = np.where(
            in_goal_band_np,
            "base_hold",
            "base_approach",
        ).astype(object)

    def initialize(self) -> bool:
        result = super().initialize()
        if result:
            self.base_goal_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.prev_base_alignment_error = self.base_alignment_error.detach().clone()
            self.prev_base_goal_error = self.dock_pos_error.detach().clone()
            self.prev_in_goal_band = (self.dock_pos_error <= self.base_goal_band).detach().clone()
            self.last_action_tensor = torch.zeros(
                (self.num_envs, 2), dtype=self.datatype, device=self.device
            )
            self.base_target_offset_command = torch.zeros(
                (self.num_envs, 2), dtype=self.datatype, device=self.device
            )
            self.base_target_position_command = self.base_planar_position.detach().clone()
            self.prev_base_target_position_command = (
                self.base_target_position_command.detach().clone()
            )
            self.episode_start_base_pos = self._get_base_pose()[0].detach().clone()
            self.episode_start_base_quat = self._get_base_pose()[1].detach().clone()
            self.base_roll_pitch_error = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.base_yaw_error = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.env_failed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_step_reward = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
        return result

    def _wrap_angle_to_pi(self, angles: torch.Tensor) -> torch.Tensor:
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def _compute_base_attitude_errors(self):
        if self.episode_start_base_quat is None or not hasattr(self.starlink_manipulator, "pid"):
            zeros = torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)
            return zeros, zeros

        _, base_quat = self._get_base_pose()
        current_euler = self.starlink_manipulator.pid.quaternion_to_euler(base_quat)
        start_euler = self.starlink_manipulator.pid.quaternion_to_euler(
            self.episode_start_base_quat
        )
        euler_error = self._wrap_angle_to_pi(current_euler - start_euler)
        roll_pitch_error = torch.norm(euler_error[:, :2], dim=-1)
        yaw_error = torch.abs(euler_error[:, 2])
        self.base_roll_pitch_error = roll_pitch_error.detach().clone()
        self.base_yaw_error = yaw_error.detach().clone()
        return roll_pitch_error, yaw_error

    def _action_to_base_target_offset(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            action,
            min=self.base_relative_action_low_xy,
            max=self.base_relative_action_high_xy,
        )

    def _current_pid_pose_setpoint(self) -> torch.Tensor:
        base_pos, base_quat = self._get_base_pose()
        base_euler = self.starlink_manipulator.pid.quaternion_to_euler(base_quat)
        return torch.cat([base_pos, base_euler], dim=-1)

    def _base_offset_to_world_target(self, local_offset_xy: torch.Tensor) -> torch.Tensor:
        base_pos, base_quat = self._get_base_pose()
        base_euler = self.starlink_manipulator.pid.quaternion_to_euler(base_quat)
        yaw = base_euler[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        world_offset_x = cos_yaw * local_offset_xy[:, 0] - sin_yaw * local_offset_xy[:, 1]
        world_offset_y = sin_yaw * local_offset_xy[:, 0] + cos_yaw * local_offset_xy[:, 1]
        world_offset_xy = torch.stack([world_offset_x, world_offset_y], dim=-1)
        return base_pos[:, :2] + world_offset_xy

    def _apply_base_action(self, local_offset_action: torch.Tensor):
        if not hasattr(self.starlink_manipulator, "pid"):
            raise RuntimeError("Stage-1 base position control requires a base PID backend.")

        local_offset_action = _as_batched_tensor(
            local_offset_action, device=self.device, dtype=self.datatype, expected_last_dim=2
        )
        target_offset_xy = self._action_to_base_target_offset(local_offset_action)
        target_xy = self._base_offset_to_world_target(target_offset_xy)
        root_pos, root_quat = self._get_base_pose()
        target_root_pos = root_pos.detach().clone()
        target_root_pos[:, :2] = target_xy
        target_root_quat = (
            self.episode_start_base_quat
            if self.episode_start_base_quat is not None
            else root_quat
        )
        setpoint = self._root_target_to_pid_link_setpoint(target_root_pos, target_root_quat)

        if hasattr(self.starlink_manipulator, "set_pid_enabled"):
            self.starlink_manipulator.set_pid_enabled(True)
        self.starlink_manipulator.pid.update_setpoint(setpoint)

        self.prev_base_target_position_command = (
            self.base_target_position_command.detach().clone()
            if self.base_target_position_command is not None
            else target_xy.detach().clone()
        )
        self.base_target_offset_command = target_offset_xy.detach().clone()
        self.base_target_position_command = target_xy.detach().clone()

    def _print_pid_trace(self):
        if not self.debug_pid_trace:
            return
        if self.debug_pid_trace_interval > 1 and self.step_count % self.debug_pid_trace_interval != 0:
            return
        if not hasattr(self.starlink_manipulator, "pid"):
            return

        env_idx = min(int(self.debug_pid_trace_env_idx), self.num_envs - 1)
        pid = self.starlink_manipulator.pid
        root_base_pos, root_base_quat = self._get_base_pose()
        setpoint = _as_batched_tensor(
            pid.setpoint, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        current_pos = _as_batched_tensor(
            getattr(pid, "_current_position", torch.zeros(3, device=self.device, dtype=self.datatype)),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=3,
        )
        current_orien = _as_batched_tensor(
            getattr(pid, "_current_orientation", torch.zeros(3, device=self.device, dtype=self.datatype)),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=3,
        )

        control_output = pid.input_reference()
        force = None
        torque = None
        if isinstance(control_output, dict):
            if control_output.get("position") is not None:
                force = _as_batched_tensor(
                    control_output["position"],
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=3,
                )
            if control_output.get("orientation") is not None:
                torque = _as_batched_tensor(
                    control_output["orientation"],
                    device=self.device,
                    dtype=self.datatype,
                    expected_last_dim=3,
                )

        pid_error = setpoint[:, :3] - current_pos
        episode_delta_xy = None
        if self.episode_start_base_pos is not None:
            episode_delta_xy = root_base_pos[:, :2] - self.episode_start_base_pos[:, :2]

        def sample(tensor):
            if tensor is None:
                return None
            return [round(float(v), 5) for v in tensor[env_idx].detach().cpu().tolist()]

        print(
            "[STAGE1_PID_TRACE]",
            f"step={self.step_count}",
            f"env_step={int(self.env_step_count[env_idx].item()) if self.env_step_count is not None else 'NA'}",
            f"env={env_idx}",
            f"action_local_xy={sample(self.base_target_offset_command)}",
            f"target_root_world_xy={sample(self.base_target_position_command)}",
            f"root_base_xyz={sample(root_base_pos)}",
            f"episode_root_delta_xy={sample(episode_delta_xy)}",
            f"base_planar_vel_xy={sample(self.base_planar_velocity)}",
            f"base_planar_speed={round(float(self.base_planar_speed[env_idx].item()), 6) if self.base_planar_speed is not None else 'NA'}",
            f"pid_link_setpoint_xyzrpy={sample(setpoint)}",
            f"pid_current_xyz={sample(current_pos)}",
            f"pid_error_xyz={sample(pid_error)}",
            f"pid_current_rpy={sample(current_orien)}",
            f"pid_force={sample(force)}",
            f"pid_torque={sample(torque)}",
            f"apply_link={self.starlink_manipulator.params['base']}",
            flush=True,
        )

    def apply_rl_action(self, action: torch.Tensor):
        action = _as_batched_tensor(
            action, device=self.device, dtype=self.datatype, expected_last_dim=2
        )
        if self.debug_fixed_base_action is not None:
            action = self.debug_fixed_base_action.unsqueeze(0).repeat(self.num_envs, 1)
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Stage-1 action should have shape ({self.num_envs}, 2), got {tuple(action.shape)}"
            )

        target_offset_action = self._action_to_base_target_offset(action)
        self.last_action_tensor = target_offset_action.detach().clone()
        self._apply_base_action(target_offset_action)

        if self.joint_reference is not None:
            self.starlink_manipulator.control_joint_pos(self.joint_reference)
        self.last_gripper_command = torch.full(
            (self.num_envs,),
            self.gripper_open_value,
            dtype=self.datatype,
            device=self.device,
        )
        self.starlink_manipulator.control_gripper(self.last_gripper_command)

    def reward(self) -> torch.Tensor:
        if self.dock_pos_error is None or self.base_alignment_error is None:
            return torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)

        goal_error = self.dock_pos_error
        in_goal_band = goal_error <= self.base_goal_band
        in_slow_zone = goal_error <= self.base_slow_zone_radius

        if self.prev_base_goal_error is None:
            goal_error_progress = torch.zeros_like(goal_error)
        else:
            goal_error_progress = self.prev_base_goal_error - goal_error
        self.prev_base_goal_error = goal_error.detach().clone()
        self.prev_base_alignment_error = self.base_alignment_error.detach().clone()

        reward = self.base_goal_error_progress_reward_scale * goal_error_progress
        reward = reward - self.base_goal_error_penalty_scale * goal_error
        reward = reward - torch.where(
            in_slow_zone,
            self.slow_zone_error_penalty_scale * goal_error,
            torch.zeros_like(reward),
        )

        if self.base_planar_speed is not None:
            reward = reward - self.base_speed_penalty_scale * self.base_planar_speed
            reward = reward - torch.where(
                in_slow_zone,
                self.slow_zone_speed_penalty_scale * self.base_planar_speed,
                torch.zeros_like(reward),
            )

        if self.base_planar_velocity is not None:
            reward_goal_position = (
                self.reward_desired_base_planar_position
                if self.reward_desired_base_planar_position is not None
                else self.desired_base_planar_position
            )
            goal_vec = reward_goal_position - self.base_planar_position
            goal_dir = goal_vec / torch.clamp(goal_error.unsqueeze(-1), min=1e-6)
            speed_toward_goal = torch.sum(self.base_planar_velocity * goal_dir, dim=-1)
            away_speed = torch.clamp(-speed_toward_goal, min=0.0)
            reward = reward - self.away_speed_penalty_scale * away_speed
            reward = reward - torch.where(
                in_slow_zone,
                self.slow_zone_away_speed_penalty_scale * away_speed,
                torch.zeros_like(reward),
            )

        roll_pitch_error, yaw_error = self._compute_base_attitude_errors()
        reward = reward - self.base_attitude_penalty_scale * roll_pitch_error
        reward = reward - self.base_yaw_penalty_scale * yaw_error

        reward_goal_threshold = (
            self.base_goal_threshold
            if self.reward_base_goal_threshold is None
            else self.reward_base_goal_threshold
        )
        too_close_error = torch.clamp(
            (float(reward_goal_threshold) - self.base_goal_band)
            - self.base_alignment_error,
            min=0.0,
        )
        reward = reward - self.too_close_penalty_scale * too_close_error

        if self.prev_in_goal_band is None:
            entered_goal_band = in_goal_band
            left_goal_band = torch.zeros_like(in_goal_band)
        else:
            entered_goal_band = in_goal_band & (~self.prev_in_goal_band)
            left_goal_band = (~in_goal_band) & self.prev_in_goal_band

        reward = reward + torch.where(
            entered_goal_band,
            torch.full_like(reward, self.goal_band_entry_bonus),
            torch.zeros_like(reward),
        )
        reward = reward - torch.where(
            left_goal_band,
            torch.full_like(reward, self.goal_band_leave_penalty),
            torch.zeros_like(reward),
        )

        stopped_in_goal = in_goal_band & (
            self.base_planar_speed <= self.base_stop_speed_threshold
        )
        reward = reward + torch.where(
            in_goal_band,
            torch.full_like(reward, self.goal_band_hold_bonus),
            torch.zeros_like(reward),
        )
        reward = reward + torch.where(
            stopped_in_goal,
            torch.full_like(reward, self.stable_hold_bonus),
            torch.zeros_like(reward),
        )
        if self.base_planar_speed is not None:
            reward = reward - torch.where(
                in_goal_band,
                self.near_goal_speed_penalty_scale * self.base_planar_speed,
                torch.zeros_like(reward),
            )

        self.prev_in_goal_band = in_goal_band.detach().clone()

        if self.last_action_tensor is not None:
            target_pos = (
                self.base_target_position_command
                if self.base_target_position_command is not None
                else self.base_planar_position
            )
            target_jump = torch.zeros_like(self.base_alignment_error)
            if self.prev_base_target_position_command is not None:
                target_jump = torch.norm(
                    target_pos - self.prev_base_target_position_command,
                    dim=-1,
                )
            target_to_goal = torch.norm(
                target_pos
                - (
                    self.reward_desired_base_planar_position
                    if self.reward_desired_base_planar_position is not None
                    else self.desired_base_planar_position
                ),
                dim=-1,
            )
            reward = reward - self.base_action_penalty_scale * target_jump
            reward = reward - self.base_target_to_goal_penalty_scale * target_to_goal
            reward = reward - torch.where(
                in_slow_zone,
                self.slow_zone_target_to_goal_penalty_scale * target_to_goal,
                torch.zeros_like(reward),
            )

        reward = reward + torch.where(
            self.is_success(),
            torch.full_like(reward, self.success_bonus),
            torch.zeros_like(reward),
        )
        reward = reward - torch.where(
            self.is_success(),
            torch.zeros_like(reward),
            torch.full_like(reward, self.time_penalty),
        )
        reward = reward - torch.where(
            self.is_failure(),
            torch.full_like(reward, self.failure_penalty),
            torch.zeros_like(reward),
        )
        return reward.to(dtype=self.datatype, device=self.device)

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            if action is not None:
                self.apply_rl_action(action)
            else:
                if self.joint_reference is not None:
                    self.starlink_manipulator.control_joint_pos(self.joint_reference)
                self.starlink_manipulator.control_gripper(self.last_gripper_command)

            for decimation_idx in range(self.base_control_decimation):
                self.starlink_manipulator.step()
                if decimation_idx == 0:
                    self._print_pid_trace()
                self.target_object.step()
                self.gsim.step()
            self._update_io_tensors()

            reward = self.reward()
            self.last_step_reward = reward.detach().clone()
            self.total_reward = self.total_reward + reward
            self.step_count += 1
            self.env_step_count += 1
            self.env_episode_return += reward

            success_mask = self.is_success()
            failure_mask = self.is_failure()
            self.success = success_mask
            self.env_failed = failure_mask.detach().clone()
            truncated_mask = self.env_step_count >= self.max_steps
            terminated_mask = success_mask | failure_mask
            self.env_done = terminated_mask.detach().clone()
            self.env_truncated = truncated_mask.detach().clone()
            self.last_done_mask = self.env_done.detach().clone()
            self.last_truncated_mask = self.env_truncated.detach().clone()

            if bool(torch.all(success_mask).item()):
                self.status = TaskStatus.SUCCEED
                return False

            if bool(torch.all(truncated_mask).item()):
                self.status = TaskStatus.COMPLETED
                return False

            if bool(torch.all(terminated_mask).item()):
                self.status = TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Stage-1 step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def is_success(self):
        if self.dock_pos_error is None or self.base_goal_hold_count is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        in_goal_band = self.dock_pos_error <= self.base_goal_band
        return (
            in_goal_band
            & (self.base_planar_speed <= self.base_stop_speed_threshold)
            & (self.base_goal_hold_count >= self.success_hold_steps)
        )

    def is_failure(self):
        if self.dock_pos_error is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        runaway = self.dock_pos_error > self.base_runaway_distance_threshold
        if self.base_planar_speed is not None:
            runaway = runaway | (
                self.base_planar_speed > self.base_runaway_speed_threshold
            )

        roll_pitch_error, yaw_error = self._compute_base_attitude_errors()
        attitude_failure = (
            (roll_pitch_error > self.base_roll_pitch_failure_threshold)
            | (yaw_error > self.base_yaw_failure_threshold)
        )
        return runaway | attitude_failure

    def reset_done_envs(self, env_ids=None) -> bool:
        result = super().reset_done_envs(env_ids=env_ids)
        if result:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            else:
                env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
            if self.base_goal_hold_count is not None and env_ids.numel() > 0:
                self.base_goal_hold_count[env_ids] = 0
            if self.prev_base_alignment_error is not None and env_ids.numel() > 0:
                self.prev_base_alignment_error[env_ids] = self.base_alignment_error[env_ids]
            if self.prev_base_goal_error is not None and env_ids.numel() > 0:
                self.prev_base_goal_error[env_ids] = self.dock_pos_error[env_ids]
            if self.prev_in_goal_band is not None and env_ids.numel() > 0:
                self.prev_in_goal_band[env_ids] = self.dock_pos_error[env_ids] <= self.base_goal_band
            if self.last_action_tensor is not None and env_ids.numel() > 0:
                self.last_action_tensor[env_ids] = 0.0
            if self.base_target_offset_command is not None and env_ids.numel() > 0:
                self.base_target_offset_command[env_ids] = 0.0
            if self.base_target_position_command is not None and env_ids.numel() > 0:
                self.base_target_position_command[env_ids] = self.base_planar_position[env_ids]
            if self.prev_base_target_position_command is not None and env_ids.numel() > 0:
                self.prev_base_target_position_command[env_ids] = self.base_planar_position[env_ids]
            if self.episode_start_base_pos is not None and env_ids.numel() > 0:
                self.episode_start_base_pos[env_ids] = self._get_base_pose()[0][env_ids].detach().clone()
            if self.episode_start_base_quat is not None and env_ids.numel() > 0:
                self.episode_start_base_quat[env_ids] = self._get_base_pose()[1][env_ids].detach().clone()
            if self.last_step_reward is not None and env_ids.numel() > 0:
                self.last_step_reward[env_ids] = 0.0
            if self.base_roll_pitch_error is not None and env_ids.numel() > 0:
                self.base_roll_pitch_error[env_ids] = 0.0
            if self.base_yaw_error is not None and env_ids.numel() > 0:
                self.base_yaw_error[env_ids] = 0.0
            if self.env_failed is not None and env_ids.numel() > 0:
                self.env_failed[env_ids] = False
        return result

    def get_info(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            reward_desired = (
                self.reward_desired_base_planar_position
                if self.reward_desired_base_planar_position is not None
                else self.desired_base_planar_position
            )
            infos.append(
                {
                    "phase": str(self.phase[env_idx]),
                    "base_alignment_error": float(self.base_alignment_error[env_idx].item()),
                    "dock_pos_error": float(self.dock_pos_error[env_idx].item()),
                    "desired_base_x": float(reward_desired[env_idx, 0].item()),
                    "desired_base_y": float(reward_desired[env_idx, 1].item()),
                    "policy_desired_base_x": float(
                        self.desired_base_planar_position[env_idx, 0].item()
                    ),
                    "policy_desired_base_y": float(
                        self.desired_base_planar_position[env_idx, 1].item()
                    ),
                    "stage1_base_goal_distance": float(self.base_goal_threshold),
                    "stage1_reward_goal_distance": float(
                        self.base_goal_threshold
                        if self.reward_base_goal_threshold is None
                        else self.reward_base_goal_threshold
                    ),
                    "base_planar_speed": float(self.base_planar_speed[env_idx].item()),
                    "base_roll_pitch_error": (
                        float(self.base_roll_pitch_error[env_idx].item())
                        if self.base_roll_pitch_error is not None
                        else 0.0
                    ),
                    "base_yaw_error": (
                        float(self.base_yaw_error[env_idx].item())
                        if self.base_yaw_error is not None
                        else 0.0
                    ),
                    "failure": (
                        bool(self.env_failed[env_idx].item())
                        if self.env_failed is not None
                        else False
                    ),
                    "success": bool(self.success[env_idx].item()),
                }
            )
        return infos


if gym is not None and spaces is not None:

    class ParallelStage1BaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            target_asset_name: str = "starlink",
            num_envs: int = 4,
            env_spacing=None,
            n_envs_per_row=None,
            center_envs_at_origin: bool = None,
            enable_visual_obs: bool = True,
            image_shape: Sequence[int] = (3, 84, 84),
            normalize_images: bool = False,
            enable_target_relative_obs: Optional[bool] = None,
            target_position_noise: bool = False,
            target_position_noise_range=(0.03, 0.03, 0.02),
        ):
            super().__init__()
            self.task = ParallelStage1BaseTask(
                target_asset_name=target_asset_name,
                num_envs=num_envs,
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
                enable_visual_obs=enable_visual_obs,
                image_shape=image_shape,
                normalize_images=normalize_images,
                enable_target_relative_obs=enable_target_relative_obs,
                target_position_noise=target_position_noise,
                target_position_noise_range=target_position_noise_range,
            )
            self.num_envs = self.task.num_envs
            self.state_dim = 11 if self.task.enable_visual_obs else 7

            if self.task.enable_visual_obs:
                image_low = 0.0 if self.task.normalize_images else 0
                image_high = 1.0 if self.task.normalize_images else 255
                image_dtype = np.float32 if self.task.normalize_images else np.uint8
                self.single_observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self.state_dim,),
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
                self.observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self.num_envs, self.state_dim),
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
                self.single_observation_space = spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.state_dim,),
                    dtype=np.float32,
                )
                self.observation_space = spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.num_envs, self.state_dim),
                    dtype=np.float32,
                )

            action_low = self.task.base_relative_action_low_xy.detach().cpu().numpy().astype(np.float32)
            action_high = self.task.base_relative_action_high_xy.detach().cpu().numpy().astype(np.float32)
            self.single_action_space = spaces.Box(
                low=action_low, high=action_high, shape=(2,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.repeat(action_low[None, :], self.num_envs, axis=0),
                high=np.repeat(action_high[None, :], self.num_envs, axis=0),
                shape=(self.num_envs, 2),
                dtype=np.float32,
            )

        def _get_obs(self):
            if self.task.enable_visual_obs:
                image_obs = self.task.base_image_observation
                if image_obs is None:
                    image_obs = self.task._zero_stage1_image_observation()
                image_obs = image_obs.detach().cpu().numpy()
                if self.task.normalize_images:
                    image_obs = image_obs.astype(np.float32)
                return {
                    "state": self.task.state_observation.detach().cpu().numpy().astype(np.float32),
                    "image": image_obs,
                }
            return self.task.observation.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError("ParallelStage1BaseTask initialize failed during env reset")
            else:
                if not self.task.reset():
                    raise RuntimeError("ParallelStage1BaseTask reset failed during env reset")
            return self._get_obs(), {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            self.task.step(action=action)
            obs = self._get_obs()
            reward = self.task.last_step_reward.detach().cpu().numpy().astype(np.float32)
            terminated = self.task.get_done_mask().detach().cpu().numpy().astype(bool)
            truncated = self.task.get_truncated_mask().detach().cpu().numpy().astype(bool)
            info = self.task.get_info()
            return obs, reward, terminated, truncated, info

        def close(self):
            self.task.stop()


if VecEnv is not None and VecEnvStepReturn is not None:

    class BatchedGenesisStage1VecEnv(VecEnv):
        """SB3-compatible VecEnv wrapper over one batched Genesis stage-1 scene."""

        def __init__(
            self,
            target_asset_name: str = "starlink",
            num_envs: int = 4,
            env_spacing=None,
            n_envs_per_row=None,
            center_envs_at_origin: bool = None,
            enable_visual_obs: bool = True,
            image_shape: Sequence[int] = (3, 84, 84),
            normalize_images: bool = False,
            enable_target_relative_obs: Optional[bool] = None,
            target_position_noise: bool = False,
            target_position_noise_range=(0.03, 0.03, 0.02),
        ):
            self.env = ParallelStage1BaseEnv(
                target_asset_name=target_asset_name,
                num_envs=num_envs,
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
                enable_visual_obs=enable_visual_obs,
                image_shape=image_shape,
                normalize_images=normalize_images,
                enable_target_relative_obs=enable_target_relative_obs,
                target_position_noise=target_position_noise,
                target_position_noise_range=target_position_noise_range,
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
                    infos[idx]["TimeLimit.truncated"] = bool(
                        truncated[idx] and not terminated[idx]
                    )
            if done_indices.size > 0:
                if not self.env.task.reset_done_envs(done_indices.tolist()):
                    raise RuntimeError("ParallelStage1BaseTask partial reset failed")
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
                    "BatchedGenesisStage1VecEnv only supports setting attributes for all envs."
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


def build_stage1_env(
    target_asset: str = "starlink",
    num_envs: int = 4,
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
    enable_visual_obs: bool = True,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
    enable_target_relative_obs: Optional[bool] = None,
    target_position_noise: bool = False,
    target_position_noise_range=(0.03, 0.03, 0.02),
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    return ParallelStage1BaseEnv(
        target_asset_name=target_asset,
        num_envs=num_envs,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=normalize_images,
        enable_target_relative_obs=enable_target_relative_obs,
        target_position_noise=target_position_noise,
        target_position_noise_range=target_position_noise_range,
    )


def make_stage1_vec_env(
    num_envs: int = 4,
    target_asset: str = "starlink",
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
    enable_visual_obs: bool = True,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
    enable_target_relative_obs: Optional[bool] = None,
    target_position_noise: bool = False,
    target_position_noise_range=(0.03, 0.03, 0.02),
):
    if VecEnv is None or VecMonitor is None:
        raise ImportError("stable_baselines3 vectorized env utilities are not installed.")
    vec_env = BatchedGenesisStage1VecEnv(
        target_asset_name=target_asset,
        num_envs=num_envs,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=normalize_images,
        enable_target_relative_obs=enable_target_relative_obs,
        target_position_noise=target_position_noise,
        target_position_noise_range=target_position_noise_range,
    )
    return VecMonitor(vec_env)


def train_stage1_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_satellite_stage1_base",
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
    enable_visual_obs: bool = True,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
    enable_target_relative_obs: Optional[bool] = None,
    policy: str = "MultiInputPolicy",
    policy_kwargs=None,
):
    if PPO is None:
        raise ImportError(
            "stable_baselines3 is not installed. Please install stable-baselines3 first."
        )

    training_device = resolve_training_device(device)
    env = make_stage1_vec_env(
        num_envs=num_envs,
        target_asset=target_asset,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=normalize_images,
        enable_target_relative_obs=enable_target_relative_obs,
    )
    if not enable_visual_obs:
        policy = "MlpPolicy"

    model = PPO(
        policy=policy,
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
        policy_kwargs=policy_kwargs,
    )
    print(f"Using stage-1 PPO training device: {training_device}")
    print(f"Using Genesis batched scene count: {num_envs}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.close()
    return model


__all__ = [
    "ParallelStage1BaseTask",
    "ParallelStage1BaseEnv",
    "BatchedGenesisStage1VecEnv",
    "build_stage1_env",
    "make_stage1_vec_env",
    "train_stage1_ppo",
]
