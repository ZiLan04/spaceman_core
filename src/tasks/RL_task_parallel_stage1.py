# 输入：基座相机图像、基座6自由度位姿、基座线/角速度
# 输出：以当前基座原点为中心的局部6自由度目标位姿增量（3维平移 + 3维姿态）
# 用PID控制基座到达由局部增量转换出的世界目标位姿（位置 + 姿态）

"""Stage-1 batched RL task: move the satellite manipulator base to target.

The stage-1 policy commands a 6-DoF base motion in the current base frame: a
3-D position offset plus a 3-D attitude increment (rotation vector). The
low-level base PID moves the base toward the resulting world target pose, while
arm joints are held at the initial pose throughout the episode.

The desired base pose extends the legacy planar Stage-1 target into 6 DoF:

* x/y: ``grasp_point_xy + dock_dir_xy * base_goal_threshold``
* z: the base height from the original scene configuration
* attitude: the base attitude from the original scene configuration

so the policy must converge in both translation and rotation.
"""

import os
import sys
import math
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
        target_asset_name: str | None = None,
        satellite_manipulator_id: str | None = None,
        target_satellite_id: str | None = None,
        scene_id: str = "starlink_franka_capture",
        scene_selection_seed: int | None = None,
        num_envs: int = 1,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
        enable_visual_obs: bool = True,
        image_shape: Sequence[int] = (3, 84, 84),
        normalize_images: bool = False,
        enable_target_relative_obs: Optional[bool] = None,
        target_position_noise: bool | None = None,
        target_position_noise_range=None,
    ):
        super().__init__(
            target_asset_name=target_asset_name,
            satellite_manipulator_id=satellite_manipulator_id,
            target_satellite_id=target_satellite_id,
            scene_id=scene_id,
            scene_selection_seed=scene_selection_seed,
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
        # The validated 5 m PID trajectory settles after about 423 policy steps.
        # Keep the full 21 s / 700-step horizon used by the feasibility test so
        # exploration still has time to brake and hold the final pose.
        self.max_steps = 700
        self.base_goal_threshold = 1.1
        self.reward_base_goal_threshold = None
        self.base_goal_band = 0.05
        self.base_attitude_band = 0.03
        self.base_stop_speed_threshold = 0.025
        self.base_stop_angular_speed_threshold = 0.05
        self.success_hold_steps = 12
        self.base_goal_hold_count = None
        self.target_dock_dir_local = torch.tensor(
            [-1.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )
        self.desired_base_position = None
        self.desired_base_quat = None
        self.reward_desired_base_position = None
        self.reward_desired_base_quat = None
        self.dock_pos_error = None
        self.dock_attitude_error = None
        self.base_attitude_error_vector = None
        self.base_linear_velocity = None
        self.base_angular_velocity = None
        self.base_linear_speed = None
        self.base_angular_speed = None
        self.base_to_target_dir_nominal = None
        self.episode_start_base_quat = None
        self.env_failed = None

        # Fixed curriculum start copied from stage1_pid_long_range_test.py.
        self.stage1_start_standoff = 5.0
        self.stage1_start_lateral_offset = 0.75
        self.stage1_start_vertical_offset = 0.50
        self.stage1_start_attitude_offset_deg = (8.0, -10.0, 12.0)
        self.nominal_base_pos = None
        self.nominal_base_quat = None

        self.base_goal_error_progress_reward_scale = 60.0
        self.base_goal_error_penalty_scale = 0.15
        self.base_attitude_error_progress_reward_scale = 30.0
        self.base_attitude_error_penalty_scale = 0.5
        self.base_speed_penalty_scale = 0.03
        self.near_goal_speed_penalty_scale = 8.0
        self.base_angular_speed_penalty_scale = 0.05
        self.near_goal_angular_speed_penalty_scale = 6.0
        self.base_action_penalty_scale = 2e-3
        self.base_slow_zone_radius = 0.30
        self.slow_zone_error_penalty_scale = 4.0
        self.base_target_to_goal_penalty_scale = 0.5
        self.base_target_attitude_to_goal_penalty_scale = 1.5
        self.slow_zone_target_to_goal_penalty_scale = 2.0
        self.away_speed_penalty_scale = 2.0
        self.slow_zone_away_speed_penalty_scale = 5.0
        self.slow_zone_speed_penalty_scale = 3.0
        self.goal_band_entry_bonus = 40.0
        self.goal_band_leave_penalty = 60.0
        self.goal_band_hold_bonus = 2.0
        self.stable_hold_bonus = 8.0
        self.success_bonus = 250.0
        self.too_close_penalty_scale = 20.0
        self.base_runaway_distance_threshold = 6.0
        self.base_runaway_speed_threshold = 1.2
        self.base_runaway_angular_speed_threshold = 1.5
        self.base_attitude_failure_threshold = 0.9
        self.failure_penalty = 200.0
        self.camera_out_of_view_penalty = 2.0
        self.camera_visible = None
        self.camera_center_error = None
        self.camera_horizontal_tangent = None
        self.camera_vertical_tangent = None
        self.prev_base_goal_error = None
        self.prev_base_attitude_error = None
        self.prev_in_goal_band = None

        # 6-DoF base action: [dx, dy, dz, droll, dpitch, dyaw] in the current
        # base frame (meters for translation, radians for rotation).
        self.base_action_dim = 6
        self.base_translation_action_limit = 0.3
        self.base_rotation_action_limit = 0.2
        self.base_relative_action_low = torch.tensor(
            [
                -self.base_translation_action_limit,
                -self.base_translation_action_limit,
                -self.base_translation_action_limit,
                -self.base_rotation_action_limit,
                -self.base_rotation_action_limit,
                -self.base_rotation_action_limit,
            ],
            dtype=self.datatype,
            device=self.device,
        )
        self.base_relative_action_high = torch.tensor(
            [
                self.base_translation_action_limit,
                self.base_translation_action_limit,
                self.base_translation_action_limit,
                self.base_rotation_action_limit,
                self.base_rotation_action_limit,
                self.base_rotation_action_limit,
            ],
            dtype=self.datatype,
            device=self.device,
        )
        self.base_target_offset_command = None
        self.base_target_position_command = None
        self.base_target_quat_command = None
        self.prev_base_target_position_command = None
        self.prev_base_target_quat_command = None
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
                    "STAGE1_DEBUG_FIXED_ACTION should be formatted as "
                    "'dx,dy,dz,droll,dpitch,dyaw', "
                    f"got {fixed_action!r}"
                ) from exc
            if len(values) != 6:
                raise ValueError(
                    "STAGE1_DEBUG_FIXED_ACTION should contain exactly six values, "
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
        self.prev_base_quat = None
        self.last_step_reward = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )

        # Per-step reward components and per-episode reward sums. These mirror
        # the stage-2 bookkeeping so TensorBoard can show the contribution of
        # every reward term (see ``Stage1MetricsCallback`` in the trainer).
        zero_reward = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )
        for reward_attr in self._stage1_reward_attr_names():
            setattr(self, reward_attr, zero_reward.clone())
        self.episode_reward_sums = {
            reward_attr: zero_reward.clone()
            for reward_attr in self._stage1_reward_attr_names()
        }
        # Per-episode summary (min/max) metrics, reset whenever an env resets.
        self.ever_in_goal_band = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_min_dock_pos_error = torch.full(
            (self.num_envs,), float("inf"), dtype=self.datatype, device=self.device
        )
        self.episode_min_base_alignment_error = torch.full(
            (self.num_envs,), float("inf"), dtype=self.datatype, device=self.device
        )
        self.episode_min_attitude_error_deg = torch.full(
            (self.num_envs,), float("inf"), dtype=self.datatype, device=self.device
        )
        self.episode_max_hold_count = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )
        self.episode_camera_visible_steps = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )
        self.episode_camera_sample_count = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )
        self.episode_max_camera_center_error = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )

    def _zero_stage1_image_observation(self) -> torch.Tensor:
        return self._zero_image_observation()

    @staticmethod
    def _stage1_reward_attr_names() -> tuple[str, ...]:
        return (
            "reward_goal_progress",
            "reward_attitude_progress",
            "reward_goal_error",
            "reward_attitude_error",
            "reward_camera_out_of_view",
            "reward_slow_zone_error",
            "reward_near_goal_speed",
            "reward_speed",
            "reward_angular_speed",
            "reward_away_speed",
            "reward_too_close",
            "reward_goal_band_entry",
            "reward_goal_band_leave",
            "reward_goal_band_hold",
            "reward_stable_hold",
            "reward_action",
            "reward_target_position",
            "reward_target_attitude",
            "reward_slow_zone_target",
            "reward_success",
            "reward_failure",
            "reward_time",
        )

    def _fixed_stage1_start_pose(
        self, grasp_pos: torch.Tensor, base_pos: torch.Tensor, base_quat: torch.Tensor
    ):
        """Return the fixed off-axis 5 m start used by the PID feasibility run."""
        radial_dir = base_pos - grasp_pos
        radial_dir = radial_dir / torch.clamp(
            torch.norm(radial_dir, dim=-1, keepdim=True), min=1e-8
        )
        world_up = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self.datatype, device=self.device
        ).expand_as(radial_dir)
        lateral_dir = torch.cross(world_up, radial_dir, dim=-1)
        lateral_norm = torch.norm(lateral_dir, dim=-1, keepdim=True)
        fallback = torch.tensor(
            [0.0, 1.0, 0.0], dtype=self.datatype, device=self.device
        ).expand_as(radial_dir)
        lateral_dir = torch.where(lateral_norm > 1e-6, lateral_dir, fallback)
        lateral_dir = lateral_dir / torch.clamp(
            torch.norm(lateral_dir, dim=-1, keepdim=True), min=1e-8
        )
        vertical_dir = torch.cross(radial_dir, lateral_dir, dim=-1)
        transverse_sq = (
            self.stage1_start_lateral_offset**2
            + self.stage1_start_vertical_offset**2
        )
        radial_distance = math.sqrt(self.stage1_start_standoff**2 - transverse_sq)
        start_pos = (
            grasp_pos
            + radial_dir * radial_distance
            + lateral_dir * self.stage1_start_lateral_offset
            + vertical_dir * self.stage1_start_vertical_offset
        )
        attitude_offset = torch.tensor(
            self.stage1_start_attitude_offset_deg,
            dtype=self.datatype,
            device=self.device,
        ).view(1, 3).expand(base_pos.shape[0], -1) * (math.pi / 180.0)
        start_quat = self._quaternion_multiply_wxyz(
            base_quat, self._rotation_vector_to_quaternion_wxyz(attitude_offset)
        )
        return start_pos, start_quat

    def state_observation_dim(self) -> int:
        """Dimension of the flat state vector produced by ``_update_io_tensors``."""
        if self.enable_visual_obs:
            # base pose (3 + 4) + linear velocity (3) + angular velocity (3)
            # + last 6-DoF action (6)
            return 7 + 3 + 3 + self.base_action_dim
        # local position error (3) + distance (1) + attitude error (3)
        # + local linear velocity (3) + local angular velocity (3)
        # + last 6-DoF action (6)
        return 3 + 1 + 3 + 3 + 3 + self.base_action_dim

    def _capture_base_image_observation(self) -> Optional[torch.Tensor]:
        if not self.enable_visual_obs:
            return None
        return self._capture_named_visual_observation("rl_base_camera")

    # ------------------------------------------------------------------
    # Quaternion helpers (all quaternions use the [qw, qx, qy, qz] order)
    # ------------------------------------------------------------------
    def _quaternion_multiply_wxyz(
        self, q1: torch.Tensor, q2: torch.Tensor
    ) -> torch.Tensor:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        quat = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )
        return self._normalize_quaternion_wxyz(quat)

    def _quaternion_conjugate_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        conjugate = quat.clone()
        conjugate[..., 1:] = -conjugate[..., 1:]
        return conjugate

    def _normalize_quaternion_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)
        return quat / norm

    def _rotation_vector_to_quaternion_wxyz(
        self, rotation_vector: torch.Tensor
    ) -> torch.Tensor:
        angle = torch.norm(rotation_vector, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        sin_half = torch.sin(half_angle)
        scale = torch.where(
            angle > 1e-8,
            sin_half / torch.clamp(angle, min=1e-8),
            torch.full_like(angle, 0.5),
        )
        quat = torch.cat(
            [torch.cos(half_angle), rotation_vector * scale],
            dim=-1,
        )
        return self._normalize_quaternion_wxyz(quat)

    def _quaternion_to_rotation_vector_wxyz(
        self, quat: torch.Tensor
    ) -> torch.Tensor:
        quat = self._normalize_quaternion_wxyz(quat)
        # Keep the shortest arc so the error stays on [0, pi].
        quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
        vector = quat[..., 1:]
        vector_norm = torch.norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(
            vector_norm.squeeze(-1),
            quat[..., 0],
        ).unsqueeze(-1)
        scale = torch.where(
            vector_norm > 1e-8,
            angle / torch.clamp(vector_norm, min=1e-8),
            torch.zeros_like(vector_norm),
        )
        return vector * scale

    def _minimal_rotation_between_vectors_wxyz(
        self, start_dir: torch.Tensor, end_dir: torch.Tensor
    ) -> torch.Tensor:
        """Quaternion rotating ``start_dir`` onto ``end_dir`` by the shortest arc."""
        start_dir = start_dir / torch.clamp(
            torch.norm(start_dir, dim=-1, keepdim=True), min=1e-8
        )
        end_dir = end_dir / torch.clamp(
            torch.norm(end_dir, dim=-1, keepdim=True), min=1e-8
        )
        cos_angle = torch.sum(start_dir * end_dir, dim=-1, keepdim=True)
        cross = torch.cross(start_dir, end_dir, dim=-1)
        quat = torch.cat([1.0 + cos_angle, cross], dim=-1)

        # Antiparallel directions: pick any axis perpendicular to start_dir.
        helper = torch.where(
            torch.abs(start_dir[..., 0:1]) > 0.9,
            torch.tensor([0.0, 1.0, 0.0], dtype=self.datatype, device=self.device).expand_as(start_dir),
            torch.tensor([1.0, 0.0, 0.0], dtype=self.datatype, device=self.device).expand_as(start_dir),
        )
        fallback_axis = torch.cross(start_dir, helper, dim=-1)
        fallback_axis = fallback_axis / torch.clamp(
            torch.norm(fallback_axis, dim=-1, keepdim=True), min=1e-8
        )
        fallback = torch.cat([torch.zeros_like(cos_angle), fallback_axis], dim=-1)
        quat = torch.where(cos_angle <= -1.0 + 1e-6, fallback, quat)
        return self._normalize_quaternion_wxyz(quat)

    def _world_vector_to_base_local(
        self, world_vector: torch.Tensor, base_quat: torch.Tensor
    ) -> torch.Tensor:
        base_rot = as_rotation_matrix(base_quat, order="wxyz")
        return torch.matmul(
            base_rot.transpose(-1, -2), world_vector.unsqueeze(-1)
        ).squeeze(-1)

    def _update_base_motion_metrics(
        self, base_pos: torch.Tensor, base_quat: torch.Tensor
    ):
        if self.prev_base_pos is None or self.prev_base_quat is None:
            self.prev_base_pos = base_pos.detach().clone()
            self.prev_base_quat = base_quat.detach().clone()
            self.base_linear_velocity = torch.zeros(
                (self.num_envs, 3), dtype=self.datatype, device=self.device
            )
            self.base_angular_velocity = torch.zeros(
                (self.num_envs, 3), dtype=self.datatype, device=self.device
            )
            self.base_linear_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.base_angular_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            return

        self.base_linear_velocity = (base_pos - self.prev_base_pos) / self.control_dt
        # omega_world = 2 * vec(q_current * q_previous^-1) / dt
        delta_quat = self._quaternion_multiply_wxyz(
            base_quat,
            self._quaternion_conjugate_wxyz(self.prev_base_quat),
        )
        self.base_angular_velocity = (
            2.0 * self._quaternion_to_rotation_vector_wxyz(delta_quat) / self.control_dt
        )
        self.base_linear_speed = torch.norm(self.base_linear_velocity, dim=-1)
        self.base_angular_speed = torch.norm(self.base_angular_velocity, dim=-1)
        self.prev_base_pos = base_pos.detach().clone()
        self.prev_base_quat = base_quat.detach().clone()

    def _sample_target_reset_noise(self, env_count: int):
        return super()._sample_target_reset_noise(env_count)

    def _compute_desired_base_pose(
        self,
        grasp_point_pos: torch.Tensor,
        grasp_point_quat: torch.Tensor,
        goal_distance: Optional[float] = None,
    ):
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
        nominal_pos = self.nominal_base_pos
        if nominal_pos is None:
            nominal_pos = self._get_base_pose()[0]
        desired_position = nominal_pos.detach().clone()
        desired_position[:, :2] = (
            grasp_point_pos[:, :2] + dock_dir_xy * float(goal_distance)
        )
        desired_quat = (
            self.nominal_base_quat.detach().clone()
            if self.nominal_base_quat is not None
            else self._get_base_pose()[1].detach().clone()
        )
        return desired_position, desired_quat

    def _update_camera_visibility_metrics(self, target_pos: torch.Tensor):
        """Project the target point into the square base-camera view frustum."""
        camera_cfg = self.satellite_manipulator_spec.camera.base
        if camera_cfg.mount_link is None:
            raise RuntimeError("Stage-1 base camera requires a mount link")
        mount_link = self.satellite_manipulator.robot.get_link(camera_cfg.mount_link)
        mount_pos = _as_batched_tensor(
            mount_link.get_pos(), device=self.device, dtype=self.datatype, expected_last_dim=3
        )
        mount_quat = _as_batched_tensor(
            mount_link.get_quat(), device=self.device, dtype=self.datatype, expected_last_dim=4
        )
        mount_rot = as_rotation_matrix(mount_quat, order="wxyz")

        def local_to_world(offset):
            local = torch.as_tensor(
                offset, dtype=self.datatype, device=self.device
            ).view(1, 3).expand(self.num_envs, -1)
            return mount_pos + torch.matmul(mount_rot, local.unsqueeze(-1)).squeeze(-1)

        camera_pos = local_to_world(camera_cfg.position_offset)
        camera_lookat = local_to_world(camera_cfg.lookat_offset)
        forward = camera_lookat - camera_pos
        forward = forward / torch.clamp(torch.norm(forward, dim=-1, keepdim=True), min=1e-8)
        up_offset = torch.as_tensor(
            camera_cfg.up_offset, dtype=self.datatype, device=self.device
        ).view(1, 3).expand(self.num_envs, -1)
        # calculate_camera_pose() treats up_offset as a direction vector, not
        # as a point relative to the mount-link origin. Match the renderer.
        up_hint = torch.matmul(mount_rot, up_offset.unsqueeze(-1)).squeeze(-1)
        right = torch.cross(forward, up_hint, dim=-1)
        right = right / torch.clamp(torch.norm(right, dim=-1, keepdim=True), min=1e-8)
        up = torch.cross(right, forward, dim=-1)

        target_vec = target_pos - camera_pos
        depth = torch.sum(target_vec * forward, dim=-1)
        safe_depth = torch.clamp(depth, min=1e-6)
        horizontal_tangent = torch.sum(target_vec * right, dim=-1) / safe_depth
        vertical_tangent = torch.sum(target_vec * up, dim=-1) / safe_depth
        tan_half_fov = math.tan(
            math.radians(float(camera_cfg.camera["fov"]) * 0.5)
        )
        visible = (
            (depth > 0.0)
            & (torch.abs(horizontal_tangent) <= tan_half_fov)
            & (torch.abs(vertical_tangent) <= tan_half_fov)
        )
        normalized_radius = torch.sqrt(
            horizontal_tangent.square() + vertical_tangent.square()
        ) / tan_half_fov
        self.camera_visible = visible
        self.camera_center_error = torch.clamp(normalized_radius, min=0.0, max=2.0)
        self.camera_horizontal_tangent = horizontal_tangent
        self.camera_vertical_tangent = vertical_tangent

    def _build_non_visual_state_observation(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        last_target_offset_command: torch.Tensor,
    ) -> torch.Tensor:
        dock_target_offset_world = self.desired_base_position - base_pos
        dock_target_offset_local = self._world_vector_to_base_local(
            dock_target_offset_world, base_quat
        )
        dock_target_distance = torch.norm(
            dock_target_offset_local, dim=-1, keepdim=True
        )
        base_linear_velocity_local = self._world_vector_to_base_local(
            self.base_linear_velocity, base_quat
        )
        base_angular_velocity_local = self._world_vector_to_base_local(
            self.base_angular_velocity, base_quat
        )
        # Attitude error as a rotation vector (expressed in the desired frame).
        attitude_error_quat = self._quaternion_multiply_wxyz(
            self._quaternion_conjugate_wxyz(self.desired_base_quat),
            base_quat,
        )
        attitude_error_vector = self._quaternion_to_rotation_vector_wxyz(
            attitude_error_quat
        )

        return torch.cat(
            [
                dock_target_offset_local,
                dock_target_distance,
                attitude_error_vector,
                base_linear_velocity_local,
                base_angular_velocity_local,
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
            & (self.dock_attitude_error <= self.base_attitude_band)
            & (self.base_linear_speed <= self.base_stop_speed_threshold)
            & (
                self.base_angular_speed
                <= self.base_stop_angular_speed_threshold
            )
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
        joint_qpos = self.satellite_manipulator.robot.get_qpos(
            qs_idx_local=self.satellite_manipulator.motors_qs
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
        self.base_position = base_pos.to(dtype=self.datatype, device=self.device)
        # Bookkeeping required by the shared RLTask.initialize path (it stores a
        # planar reference for stage-2/full-task buffers). Stage-1 logic below
        # only ever uses the full 3-D pose.
        self.base_planar_position = self.base_position[:, :2]

        # The nominal base->target bearing defines the attitude the policy has
        # to hold; capture it from the reset pose the first time we run.
        if self.episode_start_base_quat is None:
            self.episode_start_base_quat = base_quat.detach().clone()
        if self.base_to_target_dir_nominal is None:
            nominal_bearing = grasp_point_pos - base_pos
            self.base_to_target_dir_nominal = nominal_bearing / torch.clamp(
                torch.norm(nominal_bearing, dim=-1, keepdim=True), min=1e-8
            )

        self._update_base_motion_metrics(base_pos, base_quat)
        if self.base_linear_velocity is None:
            self.base_linear_velocity = torch.zeros(
                (self.num_envs, 3), dtype=self.datatype, device=self.device
            )
            self.base_angular_velocity = torch.zeros(
                (self.num_envs, 3), dtype=self.datatype, device=self.device
            )
            self.base_linear_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.base_angular_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )

        last_target_offset_command = (
            self.base_target_offset_command
            if self.base_target_offset_command is not None
            else torch.zeros((self.num_envs, 6), dtype=self.datatype, device=self.device)
        )
        self.desired_base_position, self.desired_base_quat = self._compute_desired_base_pose(
            grasp_point_pos, grasp_point_quat
        )
        reward_goal_distance = (
            self.base_goal_threshold
            if self.reward_base_goal_threshold is None
            else self.reward_base_goal_threshold
        )
        (
            self.reward_desired_base_position,
            self.reward_desired_base_quat,
        ) = self._compute_desired_base_pose(
            grasp_point_pos,
            grasp_point_quat,
            goal_distance=reward_goal_distance,
        )

        if self.enable_visual_obs:
            self.state_observation = torch.cat(
                [
                    self.base_satellite_pose,
                    self.base_linear_velocity,
                    self.base_angular_velocity,
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
            else torch.zeros((self.num_envs, 6), dtype=self.datatype, device=self.device)
        )
        self.base_alignment_error = torch.norm(
            grasp_point_pos - self.base_position, dim=-1
        )
        self.dock_pos_error = torch.norm(
            self.base_position - self.reward_desired_base_position,
            dim=-1,
        )
        self._compute_base_attitude_error()
        self._update_camera_visibility_metrics(grasp_point_pos)

        # Keep these populated for inherited reset/debug helpers.
        self.tcp_pose = torch.zeros((self.num_envs, 7), dtype=self.datatype, device=self.device)
        self.tcp_to_grasp_error = self.dock_pos_error.detach().clone()
        self.approach_alignment = torch.ones(
            self.num_envs, dtype=self.datatype, device=self.device
        )

        self._update_goal_hold_count()
        in_goal_band_np = (
            (self.dock_pos_error <= self.base_goal_band)
            & (self.dock_attitude_error <= self.base_attitude_band)
        ).detach().cpu().numpy()
        self.phase = np.where(
            in_goal_band_np,
            "base_hold",
            "base_approach",
        ).astype(object)

    def initialize(self) -> bool:
        result = super().initialize()
        if result:
            # Preserve the original scene pose as the fixed docking goal, then
            # replace the reset pose with the validated off-axis 5 m start.
            self.nominal_base_pos = self.initial_base_pos.detach().clone()
            self.nominal_base_quat = self.initial_base_quat.detach().clone()
            grasp_pos, _ = self._get_target_grasp_pose()
            fixed_pos, fixed_quat = self._fixed_stage1_start_pose(
                grasp_pos, self.nominal_base_pos, self.nominal_base_quat
            )
            self.initial_base_pos = fixed_pos.detach().clone()
            self.initial_base_quat = fixed_quat.detach().clone()
            self.satellite_manipulator.robot.set_qpos(
                torch.cat([fixed_pos, fixed_quat], dim=-1),
                qs_idx_local=list(range(7)),
                zero_velocity=True,
            )
            self.satellite_manipulator.update_state()
            self.prev_base_pos = fixed_pos.detach().clone()
            self.prev_base_quat = fixed_quat.detach().clone()

            # ParallelRLTask.initialize() creates the legacy planar command
            # buffer with shape (num_envs, 2). Replace it before rebuilding the
            # first Stage-1 observation; otherwise reset() returns a stale
            # 15-D state once even though the declared space is 19-D.
            self.base_target_offset_command = torch.zeros(
                (self.num_envs, self.base_action_dim),
                dtype=self.datatype,
                device=self.device,
            )
            self.last_action_tensor = self.base_target_offset_command.detach().clone()
            self._update_io_tensors()
            if self.state_observation.shape[-1] != self.state_observation_dim():
                raise RuntimeError(
                    "Stage-1 state observation dimension mismatch after initialization: "
                    f"got {self.state_observation.shape[-1]}, "
                    f"expected {self.state_observation_dim()}"
                )

            # The PID feasibility run showed that the stock arm servo permits
            # visible drift under base acceleration. Use the same physical
            # high-authority fixture here; no kinematic qpos writes occur while
            # stepping, so the free-floating base dynamics remain intact.
            fixture_dofs = list(self.satellite_manipulator.motors_dof) + list(
                self.satellite_manipulator.fingers_dof
            )
            self.satellite_manipulator.robot.set_dofs_kp(
                [20000.0] * len(fixture_dofs), dofs_idx_local=fixture_dofs
            )
            self.satellite_manipulator.robot.set_dofs_kv(
                [2000.0] * len(fixture_dofs), dofs_idx_local=fixture_dofs
            )
            self.satellite_manipulator.robot.set_dofs_force_range(
                [-1000.0] * len(fixture_dofs),
                [1000.0] * len(fixture_dofs),
                dofs_idx_local=fixture_dofs,
            )
            self.base_goal_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.prev_base_alignment_error = self.base_alignment_error.detach().clone()
            self.prev_base_goal_error = self.dock_pos_error.detach().clone()
            self.prev_base_attitude_error = self.dock_attitude_error.detach().clone()
            self.prev_in_goal_band = (
                (self.dock_pos_error <= self.base_goal_band)
                & (self.dock_attitude_error <= self.base_attitude_band)
            ).detach().clone()
            self.last_action_tensor.zero_()
            self.base_target_offset_command.zero_()
            self.base_target_position_command = self.base_position.detach().clone()
            self.base_target_quat_command = self._get_base_pose()[1].detach().clone()
            self.prev_base_target_position_command = (
                self.base_target_position_command.detach().clone()
            )
            self.episode_start_base_pos = self._get_base_pose()[0].detach().clone()
            self.episode_start_base_quat = self._get_base_pose()[1].detach().clone()
            nominal_bearing = (
                self._get_target_grasp_pose()[0] - self.episode_start_base_pos
            )
            self.base_to_target_dir_nominal = nominal_bearing / torch.clamp(
                torch.norm(nominal_bearing, dim=-1, keepdim=True), min=1e-8
            )
            self.env_failed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_step_reward = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self._reset_stage1_episode_metrics()
        return result

    def _reset_stage1_episode_metrics(self, env_ids=None):
        """Clear per-episode reward sums and summary metrics."""
        for reward_attr in self._stage1_reward_attr_names():
            sums = self.episode_reward_sums[reward_attr]
            if env_ids is None:
                sums.zero_()
            else:
                sums[env_ids] = 0.0

        if env_ids is None:
            self.ever_in_goal_band.fill_(False)
            self.episode_min_dock_pos_error.fill_(float("inf"))
            self.episode_min_base_alignment_error.fill_(float("inf"))
            self.episode_min_attitude_error_deg.fill_(float("inf"))
            self.episode_max_hold_count.zero_()
            self.episode_camera_visible_steps.zero_()
            self.episode_camera_sample_count.zero_()
            self.episode_max_camera_center_error.zero_()
            return

        self.ever_in_goal_band[env_ids] = False
        self.episode_min_dock_pos_error[env_ids] = float("inf")
        self.episode_min_base_alignment_error[env_ids] = float("inf")
        self.episode_min_attitude_error_deg[env_ids] = float("inf")
        self.episode_max_hold_count[env_ids] = 0.0
        self.episode_camera_visible_steps[env_ids] = 0.0
        self.episode_camera_sample_count[env_ids] = 0.0
        self.episode_max_camera_center_error[env_ids] = 0.0

    def _compute_base_attitude_error(self):
        """Attitude error between the current base pose and the desired one.

        Returns the shortest-arc rotation vector (in the desired frame) that
        takes the desired attitude to the current attitude, together with its
        magnitude in radians.
        """
        desired_quat = (
            self.reward_desired_base_quat
            if self.reward_desired_base_quat is not None
            else self.desired_base_quat
        )
        if desired_quat is None:
            zeros = torch.zeros((self.num_envs, 3), dtype=self.datatype, device=self.device)
            self.base_attitude_error_vector = zeros
            self.dock_attitude_error = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            return self.base_attitude_error_vector, self.dock_attitude_error

        _, base_quat = self._get_base_pose()
        error_quat = self._quaternion_multiply_wxyz(
            self._quaternion_conjugate_wxyz(desired_quat),
            base_quat,
        )
        error_vector = self._quaternion_to_rotation_vector_wxyz(error_quat)
        self.base_attitude_error_vector = error_vector.detach().clone()
        self.dock_attitude_error = torch.norm(error_vector, dim=-1)
        return self.base_attitude_error_vector, self.dock_attitude_error

    def _action_to_base_target_offset(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            action,
            min=self.base_relative_action_low,
            max=self.base_relative_action_high,
        )

    def _current_pid_pose_setpoint(self) -> torch.Tensor:
        base_pos, base_quat = self._get_base_pose()
        base_euler = self.satellite_manipulator.pid.quaternion_to_euler(base_quat)
        return torch.cat([base_pos, base_euler], dim=-1)

    def _base_local_offset_to_world(self, local_offset: torch.Tensor) -> torch.Tensor:
        """Rotate a base-frame offset into the world frame."""
        base_pos, base_quat = self._get_base_pose()
        base_rot = as_rotation_matrix(base_quat, order="wxyz")
        world_offset = torch.matmul(
            base_rot, local_offset.unsqueeze(-1)
        ).squeeze(-1)
        return base_pos + world_offset

    def _apply_base_action(self, local_offset_action: torch.Tensor):
        if not hasattr(self.satellite_manipulator, "pid"):
            raise RuntimeError("Stage-1 base pose control requires a base PID backend.")

        local_offset_action = _as_batched_tensor(
            local_offset_action, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        target_offset_command = self._action_to_base_target_offset(local_offset_action)
        local_translation = target_offset_command[:, :3]
        local_rotation = target_offset_command[:, 3:]

        root_pos, root_quat = self._get_base_pose()
        target_root_pos = self._base_local_offset_to_world(local_translation)
        # Body-frame attitude increment: target = current * delta(action).
        target_root_quat = self._quaternion_multiply_wxyz(
            root_quat,
            self._rotation_vector_to_quaternion_wxyz(local_rotation),
        )
        setpoint = self._root_target_to_pid_link_setpoint(target_root_pos, target_root_quat)

        if hasattr(self.satellite_manipulator, "set_pid_enabled"):
            self.satellite_manipulator.set_pid_enabled(True)
        self.satellite_manipulator.pid.update_setpoint(setpoint)

        self.prev_base_target_position_command = (
            self.base_target_position_command.detach().clone()
            if self.base_target_position_command is not None
            else target_root_pos.detach().clone()
        )
        self.prev_base_target_quat_command = (
            self.base_target_quat_command.detach().clone()
            if self.base_target_quat_command is not None
            else target_root_quat.detach().clone()
        )
        self.base_target_offset_command = target_offset_command.detach().clone()
        self.base_target_position_command = target_root_pos.detach().clone()
        self.base_target_quat_command = target_root_quat.detach().clone()

    def _print_pid_trace(self):
        if not self.debug_pid_trace:
            return
        if self.debug_pid_trace_interval > 1 and self.step_count % self.debug_pid_trace_interval != 0:
            return
        if not hasattr(self.satellite_manipulator, "pid"):
            return

        env_idx = min(int(self.debug_pid_trace_env_idx), self.num_envs - 1)
        pid = self.satellite_manipulator.pid
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
        pid_attitude_error = setpoint[:, 3:] - current_orien
        episode_delta_xyz = None
        if self.episode_start_base_pos is not None:
            episode_delta_xyz = root_base_pos - self.episode_start_base_pos

        def sample(tensor):
            if tensor is None:
                return None
            return [round(float(v), 5) for v in tensor[env_idx].detach().cpu().tolist()]

        print(
            "[STAGE1_PID_TRACE]",
            f"step={self.step_count}",
            f"env_step={int(self.env_step_count[env_idx].item()) if self.env_step_count is not None else 'NA'}",
            f"env={env_idx}",
            f"action_local_dxyz_drpy={sample(self.base_target_offset_command)}",
            f"target_root_world_xyz={sample(self.base_target_position_command)}",
            f"target_root_world_quat={sample(self.base_target_quat_command)}",
            f"root_base_xyz={sample(root_base_pos)}",
            f"episode_root_delta_xyz={sample(episode_delta_xyz)}",
            f"base_linear_vel={sample(self.base_linear_velocity)}",
            f"base_angular_vel={sample(self.base_angular_velocity)}",
            f"base_linear_speed={round(float(self.base_linear_speed[env_idx].item()), 6) if self.base_linear_speed is not None else 'NA'}",
            f"base_angular_speed={round(float(self.base_angular_speed[env_idx].item()), 6) if self.base_angular_speed is not None else 'NA'}",
            f"dock_pos_error={round(float(self.dock_pos_error[env_idx].item()), 6) if self.dock_pos_error is not None else 'NA'}",
            f"dock_attitude_error={round(float(self.dock_attitude_error[env_idx].item()), 6) if self.dock_attitude_error is not None else 'NA'}",
            f"pid_link_setpoint_xyzrpy={sample(setpoint)}",
            f"pid_current_xyz={sample(current_pos)}",
            f"pid_error_xyz={sample(pid_error)}",
            f"pid_current_rpy={sample(current_orien)}",
            f"pid_error_rpy={sample(pid_attitude_error)}",
            f"pid_force={sample(force)}",
            f"pid_torque={sample(torque)}",
            f"apply_link={self.satellite_manipulator.params['base']}",
            flush=True,
        )

    def apply_rl_action(self, action: torch.Tensor):
        action = _as_batched_tensor(
            action, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        if self.debug_fixed_base_action is not None:
            action = self.debug_fixed_base_action.unsqueeze(0).repeat(self.num_envs, 1)
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Stage-1 action should have shape ({self.num_envs}, 6), got {tuple(action.shape)}"
            )

        target_offset_action = self._action_to_base_target_offset(action)
        self.last_action_tensor = target_offset_action.detach().clone()
        self._apply_base_action(target_offset_action)

        if self.joint_reference is not None:
            self.satellite_manipulator.control_joint_pos(self.joint_reference)
        self.last_gripper_command = torch.full(
            (self.num_envs,),
            self.gripper_open_value,
            dtype=self.datatype,
            device=self.device,
        )
        self.satellite_manipulator.control_gripper(self.last_gripper_command)

    def reward(self) -> torch.Tensor:
        if self.dock_pos_error is None or self.base_alignment_error is None:
            return torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)

        goal_error = self.dock_pos_error
        attitude_error = self.dock_attitude_error
        in_goal_band = (goal_error <= self.base_goal_band) & (
            attitude_error <= self.base_attitude_band
        )
        in_slow_zone = goal_error <= self.base_slow_zone_radius

        if self.prev_base_goal_error is None:
            goal_error_progress = torch.zeros_like(goal_error)
        else:
            goal_error_progress = self.prev_base_goal_error - goal_error
        if self.prev_base_attitude_error is None:
            attitude_error_progress = torch.zeros_like(attitude_error)
        else:
            attitude_error_progress = self.prev_base_attitude_error - attitude_error
        self.prev_base_goal_error = goal_error.detach().clone()
        self.prev_base_attitude_error = attitude_error.detach().clone()
        self.prev_base_alignment_error = self.base_alignment_error.detach().clone()

        zero_reward = torch.zeros_like(goal_error)

        self.reward_goal_progress = (
            self.base_goal_error_progress_reward_scale * goal_error_progress
        )
        self.reward_attitude_progress = (
            self.base_attitude_error_progress_reward_scale * attitude_error_progress
        )
        self.reward_goal_error = -self.base_goal_error_penalty_scale * goal_error
        self.reward_attitude_error = (
            -self.base_attitude_error_penalty_scale * attitude_error
        )
        if self.camera_visible is None:
            self.reward_camera_out_of_view = zero_reward.clone()
        else:
            self.reward_camera_out_of_view = -torch.where(
                self.camera_visible,
                zero_reward,
                torch.full_like(zero_reward, self.camera_out_of_view_penalty),
            )
        self.reward_slow_zone_error = -torch.where(
            in_slow_zone,
            self.slow_zone_error_penalty_scale * goal_error,
            zero_reward,
        )

        self.reward_near_goal_speed = zero_reward.clone()
        self.reward_speed = zero_reward.clone()
        if self.base_linear_speed is not None:
            self.reward_speed = -self.base_speed_penalty_scale * self.base_linear_speed
            self.reward_speed = self.reward_speed - torch.where(
                in_slow_zone,
                self.slow_zone_speed_penalty_scale * self.base_linear_speed,
                zero_reward,
            )
            self.reward_near_goal_speed = -torch.where(
                in_goal_band,
                self.near_goal_speed_penalty_scale * self.base_linear_speed,
                zero_reward,
            )

        self.reward_angular_speed = zero_reward.clone()
        if self.base_angular_speed is not None:
            self.reward_angular_speed = -(
                self.base_angular_speed_penalty_scale * self.base_angular_speed
            )
            self.reward_angular_speed = self.reward_angular_speed - torch.where(
                in_goal_band,
                self.near_goal_angular_speed_penalty_scale
                * self.base_angular_speed,
                zero_reward,
            )

        self.reward_away_speed = zero_reward.clone()
        if self.base_linear_velocity is not None:
            reward_goal_position = (
                self.reward_desired_base_position
                if self.reward_desired_base_position is not None
                else self.desired_base_position
            )
            goal_vec = reward_goal_position - self.base_position
            goal_dir = goal_vec / torch.clamp(goal_error.unsqueeze(-1), min=1e-6)
            speed_toward_goal = torch.sum(self.base_linear_velocity * goal_dir, dim=-1)
            away_speed = torch.clamp(-speed_toward_goal, min=0.0)
            self.reward_away_speed = -self.away_speed_penalty_scale * away_speed
            self.reward_away_speed = self.reward_away_speed - torch.where(
                in_slow_zone,
                self.slow_zone_away_speed_penalty_scale * away_speed,
                zero_reward,
            )

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
        self.reward_too_close = -self.too_close_penalty_scale * too_close_error

        if self.prev_in_goal_band is None:
            entered_goal_band = in_goal_band
            left_goal_band = torch.zeros_like(in_goal_band)
        else:
            entered_goal_band = in_goal_band & (~self.prev_in_goal_band)
            left_goal_band = (~in_goal_band) & self.prev_in_goal_band

        self.reward_goal_band_entry = torch.where(
            entered_goal_band,
            torch.full_like(zero_reward, self.goal_band_entry_bonus),
            zero_reward,
        )
        self.reward_goal_band_leave = -torch.where(
            left_goal_band,
            torch.full_like(zero_reward, self.goal_band_leave_penalty),
            zero_reward,
        )

        stopped_in_goal = (
            in_goal_band
            & (self.base_linear_speed <= self.base_stop_speed_threshold)
            & (self.base_angular_speed <= self.base_stop_angular_speed_threshold)
        )
        self.reward_goal_band_hold = torch.where(
            in_goal_band,
            torch.full_like(zero_reward, self.goal_band_hold_bonus),
            zero_reward,
        )
        self.reward_stable_hold = torch.where(
            stopped_in_goal,
            torch.full_like(zero_reward, self.stable_hold_bonus),
            zero_reward,
        )

        self.prev_in_goal_band = in_goal_band.detach().clone()
        if self.ever_in_goal_band is None:
            self.ever_in_goal_band = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self.ever_in_goal_band = (
            self.ever_in_goal_band | in_goal_band.detach().clone()
        )

        if self.last_action_tensor is not None:
            target_pos = (
                self.base_target_position_command
                if self.base_target_position_command is not None
                else self.base_position
            )
            target_pos_jump = torch.zeros_like(self.base_alignment_error)
            if self.prev_base_target_position_command is not None:
                target_pos_jump = torch.norm(
                    target_pos - self.prev_base_target_position_command,
                    dim=-1,
                )
            target_attitude_jump = torch.zeros_like(target_pos_jump)
            if (
                self.base_target_quat_command is not None
                and self.prev_base_target_quat_command is not None
            ):
                jump_quat = self._quaternion_multiply_wxyz(
                    self._quaternion_conjugate_wxyz(
                        self.prev_base_target_quat_command
                    ),
                    self.base_target_quat_command,
                )
                target_attitude_jump = torch.norm(
                    self._quaternion_to_rotation_vector_wxyz(jump_quat), dim=-1
                )
            target_jump = target_pos_jump + target_attitude_jump
            target_to_goal = torch.norm(
                target_pos
                - (
                    self.reward_desired_base_position
                    if self.reward_desired_base_position is not None
                    else self.desired_base_position
                ),
                dim=-1,
            )
            target_attitude_to_goal = torch.zeros_like(target_to_goal)
            if self.base_target_quat_command is not None:
                reward_desired_quat = (
                    self.reward_desired_base_quat
                    if self.reward_desired_base_quat is not None
                    else self.desired_base_quat
                )
                target_error_quat = self._quaternion_multiply_wxyz(
                    self._quaternion_conjugate_wxyz(reward_desired_quat),
                    self.base_target_quat_command,
                )
                target_attitude_to_goal = torch.norm(
                    self._quaternion_to_rotation_vector_wxyz(target_error_quat),
                    dim=-1,
                )
            self.reward_action = -self.base_action_penalty_scale * target_jump
            self.reward_target_position = (
                -self.base_target_to_goal_penalty_scale * target_to_goal
            )
            self.reward_target_attitude = (
                -self.base_target_attitude_to_goal_penalty_scale
                * target_attitude_to_goal
            )
            self.reward_slow_zone_target = -torch.where(
                in_slow_zone,
                self.slow_zone_target_to_goal_penalty_scale * target_to_goal,
                zero_reward,
            )
        else:
            self.reward_action = zero_reward.clone()
            self.reward_target_position = zero_reward.clone()
            self.reward_target_attitude = zero_reward.clone()
            self.reward_slow_zone_target = zero_reward.clone()

        self.reward_success = torch.where(
            self.is_success(),
            torch.full_like(zero_reward, self.success_bonus),
            zero_reward,
        )
        self.reward_failure = -torch.where(
            self.is_failure(),
            torch.full_like(zero_reward, self.failure_penalty),
            zero_reward,
        )
        self.reward_time = -torch.where(
            self.is_success(),
            zero_reward,
            torch.full_like(zero_reward, self.time_penalty),
        )

        reward = (
            self.reward_goal_progress
            + self.reward_attitude_progress
            + self.reward_goal_error
            + self.reward_attitude_error
            + self.reward_camera_out_of_view
            + self.reward_slow_zone_error
            + self.reward_speed
            + self.reward_angular_speed
            + self.reward_away_speed
            + self.reward_too_close
            + self.reward_goal_band_entry
            + self.reward_goal_band_leave
            + self.reward_goal_band_hold
            + self.reward_stable_hold
            + self.reward_near_goal_speed
            + self.reward_action
            + self.reward_target_position
            + self.reward_target_attitude
            + self.reward_slow_zone_target
            + self.reward_success
            + self.reward_failure
            + self.reward_time
        )

        for reward_attr in self._stage1_reward_attr_names():
            self.episode_reward_sums[reward_attr] += getattr(self, reward_attr)
        self._update_episode_summaries()

        return reward.to(dtype=self.datatype, device=self.device)

    def _update_episode_summaries(self):
        """Track best-so-far metrics for the episode currently in progress.

        Mirrors the stage-2 summary bookkeeping: the trainer reads these keys
        from ``get_info`` when an episode finishes, so TensorBoard shows the
        per-episode extrema instead of a noisy per-step value.
        """
        if self.dock_pos_error is not None:
            self.episode_min_dock_pos_error = torch.minimum(
                self.episode_min_dock_pos_error, self.dock_pos_error
            )
        if self.base_alignment_error is not None:
            self.episode_min_base_alignment_error = torch.minimum(
                self.episode_min_base_alignment_error, self.base_alignment_error
            )
        if self.dock_attitude_error is not None:
            self.episode_min_attitude_error_deg = torch.minimum(
                self.episode_min_attitude_error_deg,
                torch.rad2deg(self.dock_attitude_error),
            )
        if self.base_goal_hold_count is not None:
            self.episode_max_hold_count = torch.maximum(
                self.episode_max_hold_count,
                self.base_goal_hold_count.to(dtype=self.datatype),
            )
        if self.camera_visible is not None:
            self.episode_camera_visible_steps += self.camera_visible.to(self.datatype)
            self.episode_camera_sample_count += 1.0
        if self.camera_center_error is not None:
            self.episode_max_camera_center_error = torch.maximum(
                self.episode_max_camera_center_error, self.camera_center_error
            )

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            if action is not None:
                self.apply_rl_action(action)
            else:
                if self.joint_reference is not None:
                    self.satellite_manipulator.control_joint_pos(self.joint_reference)
                self.satellite_manipulator.control_gripper(self.last_gripper_command)

            for decimation_idx in range(self.base_control_decimation):
                self.satellite_manipulator.step()
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
        in_goal_band = (self.dock_pos_error <= self.base_goal_band) & (
            self.dock_attitude_error <= self.base_attitude_band
        )
        return (
            in_goal_band
            & (self.base_linear_speed <= self.base_stop_speed_threshold)
            & (
                self.base_angular_speed
                <= self.base_stop_angular_speed_threshold
            )
            & (self.base_goal_hold_count >= self.success_hold_steps)
        )

    def is_failure(self):
        if self.dock_pos_error is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        runaway = self.dock_pos_error > self.base_runaway_distance_threshold
        if self.base_linear_speed is not None:
            runaway = runaway | (
                self.base_linear_speed > self.base_runaway_speed_threshold
            )
        if self.base_angular_speed is not None:
            runaway = runaway | (
                self.base_angular_speed > self.base_runaway_angular_speed_threshold
            )

        attitude_failure = (
            self.dock_attitude_error is not None
            and self.base_attitude_failure_threshold is not None
        )
        if attitude_failure:
            attitude_failure = (
                self.dock_attitude_error > self.base_attitude_failure_threshold
            )
        else:
            attitude_failure = torch.zeros_like(runaway)
        return runaway | attitude_failure

    def reset_done_envs(self, env_ids=None) -> bool:
        # The inherited reset writes the planar projection into the base target
        # command buffers. Stage-1 stores full 6-DoF commands there, so hide
        # them from the parent and restore them afterwards.
        stashed_commands = (
            self.base_target_offset_command,
            self.base_target_position_command,
            self.prev_base_target_position_command,
        )
        self.base_target_offset_command = None
        self.base_target_position_command = None
        self.prev_base_target_position_command = None
        try:
            result = super().reset_done_envs(env_ids=env_ids)
        finally:
            (
                self.base_target_offset_command,
                self.base_target_position_command,
                self.prev_base_target_position_command,
            ) = stashed_commands
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
            if self.prev_base_attitude_error is not None and env_ids.numel() > 0:
                self.prev_base_attitude_error[env_ids] = self.dock_attitude_error[env_ids]
            if self.prev_in_goal_band is not None and env_ids.numel() > 0:
                self.prev_in_goal_band[env_ids] = (
                    (self.dock_pos_error[env_ids] <= self.base_goal_band)
                    & (self.dock_attitude_error[env_ids] <= self.base_attitude_band)
                )
            if self.last_action_tensor is not None and env_ids.numel() > 0:
                self.last_action_tensor[env_ids] = 0.0
            if self.base_target_offset_command is not None and env_ids.numel() > 0:
                self.base_target_offset_command[env_ids] = 0.0
            if self.base_target_position_command is not None and env_ids.numel() > 0:
                self.base_target_position_command[env_ids] = self.base_position[env_ids]
            if self.prev_base_target_position_command is not None and env_ids.numel() > 0:
                self.prev_base_target_position_command[env_ids] = self.base_position[env_ids]
            if self.base_target_quat_command is not None and env_ids.numel() > 0:
                self.base_target_quat_command[env_ids] = self._get_base_pose()[1][env_ids]
            if self.prev_base_target_quat_command is not None and env_ids.numel() > 0:
                self.prev_base_target_quat_command[env_ids] = self._get_base_pose()[1][env_ids]
            if self.episode_start_base_pos is not None and env_ids.numel() > 0:
                self.episode_start_base_pos[env_ids] = self._get_base_pose()[0][env_ids].detach().clone()
            if self.episode_start_base_quat is not None and env_ids.numel() > 0:
                self.episode_start_base_quat[env_ids] = self._get_base_pose()[1][env_ids].detach().clone()
            if self.base_to_target_dir_nominal is not None and env_ids.numel() > 0:
                nominal_bearing = (
                    self._get_target_grasp_pose()[0][env_ids]
                    - self._get_base_pose()[0][env_ids]
                )
                self.base_to_target_dir_nominal[env_ids] = nominal_bearing / torch.clamp(
                    torch.norm(nominal_bearing, dim=-1, keepdim=True), min=1e-8
                )
            if self.last_step_reward is not None and env_ids.numel() > 0:
                self.last_step_reward[env_ids] = 0.0
            if self.base_attitude_error_vector is not None and env_ids.numel() > 0:
                self.base_attitude_error_vector[env_ids] = 0.0
            if self.base_linear_velocity is not None and env_ids.numel() > 0:
                self.base_linear_velocity[env_ids] = 0.0
            if self.base_angular_velocity is not None and env_ids.numel() > 0:
                self.base_angular_velocity[env_ids] = 0.0
            if self.prev_base_pos is not None and env_ids.numel() > 0:
                self.prev_base_pos[env_ids] = self._get_base_pose()[0][env_ids]
            if self.prev_base_quat is not None and env_ids.numel() > 0:
                self.prev_base_quat[env_ids] = self._get_base_pose()[1][env_ids]
            if self.env_failed is not None and env_ids.numel() > 0:
                self.env_failed[env_ids] = False
            if env_ids.numel() > 0:
                self._reset_stage1_episode_metrics(env_ids)
        return result

    def get_info(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            reward_desired = (
                self.reward_desired_base_position
                if self.reward_desired_base_position is not None
                else self.desired_base_position
            )
            infos.append(
                {
                    "phase": str(self.phase[env_idx]),
                    "base_alignment_error": float(self.base_alignment_error[env_idx].item()),
                    "dock_pos_error": float(self.dock_pos_error[env_idx].item()),
                    "desired_base_x": float(reward_desired[env_idx, 0].item()),
                    "desired_base_y": float(reward_desired[env_idx, 1].item()),
                    "desired_base_z": float(reward_desired[env_idx, 2].item()),
                    "policy_desired_base_x": float(
                        self.desired_base_position[env_idx, 0].item()
                    ),
                    "policy_desired_base_y": float(
                        self.desired_base_position[env_idx, 1].item()
                    ),
                    "policy_desired_base_z": float(
                        self.desired_base_position[env_idx, 2].item()
                    ),
                    "stage1_base_goal_distance": float(self.base_goal_threshold),
                    "stage1_reward_goal_distance": float(
                        self.base_goal_threshold
                        if self.reward_base_goal_threshold is None
                        else self.reward_base_goal_threshold
                    ),
                    "base_linear_speed": float(self.base_linear_speed[env_idx].item()),
                    "base_angular_speed": float(self.base_angular_speed[env_idx].item()),
                    "dock_attitude_error": (
                        float(self.dock_attitude_error[env_idx].item())
                        if self.dock_attitude_error is not None
                        else 0.0
                    ),
                    "dock_attitude_error_deg": (
                        float(
                            torch.rad2deg(self.dock_attitude_error[env_idx]).item()
                        )
                        if self.dock_attitude_error is not None
                        else 0.0
                    ),
                    "base_goal_hold_count": (
                        int(self.base_goal_hold_count[env_idx].item())
                        if self.base_goal_hold_count is not None
                        else 0
                    ),
                    "ever_in_goal_band": bool(self.ever_in_goal_band[env_idx].item()),
                    "camera_visible": bool(self.camera_visible[env_idx].item()),
                    "camera_center_error": float(
                        self.camera_center_error[env_idx].item()
                    ),
                    "base_attitude_error_x": (
                        float(self.base_attitude_error_vector[env_idx, 0].item())
                        if self.base_attitude_error_vector is not None
                        else 0.0
                    ),
                    "base_attitude_error_y": (
                        float(self.base_attitude_error_vector[env_idx, 1].item())
                        if self.base_attitude_error_vector is not None
                        else 0.0
                    ),
                    "base_attitude_error_z": (
                        float(self.base_attitude_error_vector[env_idx, 2].item())
                        if self.base_attitude_error_vector is not None
                        else 0.0
                    ),
                    "base_linear_velocity_x": (
                        float(self.base_linear_velocity[env_idx, 0].item())
                        if self.base_linear_velocity is not None
                        else 0.0
                    ),
                    "base_linear_velocity_y": (
                        float(self.base_linear_velocity[env_idx, 1].item())
                        if self.base_linear_velocity is not None
                        else 0.0
                    ),
                    "base_linear_velocity_z": (
                        float(self.base_linear_velocity[env_idx, 2].item())
                        if self.base_linear_velocity is not None
                        else 0.0
                    ),
                    "failure": (
                        bool(self.env_failed[env_idx].item())
                        if self.env_failed is not None
                        else False
                    ),
                    "success": bool(self.success[env_idx].item()),
                    "episode_min_dock_pos_error": float(
                        self.episode_min_dock_pos_error[env_idx].item()
                    ),
                    "episode_min_base_alignment_error": float(
                        self.episode_min_base_alignment_error[env_idx].item()
                    ),
                    "episode_min_attitude_error_deg": float(
                        self.episode_min_attitude_error_deg[env_idx].item()
                    ),
                    "episode_max_hold_count": float(
                        self.episode_max_hold_count[env_idx].item()
                    ),
                    "episode_camera_visible_fraction": float(
                        (
                            self.episode_camera_visible_steps[env_idx]
                            / torch.clamp(
                                self.episode_camera_sample_count[env_idx], min=1.0
                            )
                        ).item()
                    ),
                    "episode_max_camera_center_error": float(
                        self.episode_max_camera_center_error[env_idx].item()
                    ),
                }
            )
            infos[-1].update(
                {
                    reward_attr: float(getattr(self, reward_attr)[env_idx].item())
                    for reward_attr in self._stage1_reward_attr_names()
                }
            )
            infos[-1].update(
                {
                    f"episode_{reward_attr}": float(
                        self.episode_reward_sums[reward_attr][env_idx].item()
                    )
                    for reward_attr in self._stage1_reward_attr_names()
                }
            )
        return infos


if gym is not None and spaces is not None:

    class ParallelStage1BaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            target_asset_name: str = "starlink",
            satellite_manipulator_id: str = "franka_merge",
            target_satellite_id: str | None = None,
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
                satellite_manipulator_id=satellite_manipulator_id,
                target_satellite_id=target_satellite_id,
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
            self.state_dim = self.task.state_observation_dim()
            self.action_dim = self.task.base_action_dim

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

            action_low = (
                self.task.base_relative_action_low.detach().cpu().numpy().astype(np.float32)
            )
            action_high = (
                self.task.base_relative_action_high.detach().cpu().numpy().astype(np.float32)
            )
            self.single_action_space = spaces.Box(
                low=action_low,
                high=action_high,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.repeat(action_low[None, :], self.num_envs, axis=0),
                high=np.repeat(action_high[None, :], self.num_envs, axis=0),
                shape=(self.num_envs, self.action_dim),
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
            satellite_manipulator_id: str = "franka_merge",
            target_satellite_id: str | None = None,
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
                satellite_manipulator_id=satellite_manipulator_id,
                target_satellite_id=target_satellite_id,
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
    satellite_manipulator_id: str = "franka_merge",
    target_satellite_id: str | None = None,
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
        satellite_manipulator_id=satellite_manipulator_id,
        target_satellite_id=target_satellite_id,
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
    target_asset: str | None = None,
    satellite_manipulator_id: str | None = None,
    target_satellite_id: str | None = None,
    scene_id: str = "starlink_franka_capture",
    scene_selection_seed: int | None = None,
    env_spacing=None,
    n_envs_per_row=None,
    center_envs_at_origin: bool = None,
    enable_visual_obs: bool = True,
    image_shape: Sequence[int] = (3, 84, 84),
    normalize_images: bool = False,
    enable_target_relative_obs: Optional[bool] = None,
    target_position_noise: bool | None = None,
    target_position_noise_range=None,
):
    if VecEnv is None or VecMonitor is None:
        raise ImportError("stable_baselines3 vectorized env utilities are not installed.")
    from configs.scene_registry import resolve_scene_assets
    scene_spec, satellite_manipulator_id, target_satellite_id = resolve_scene_assets(
        scene_id,
        seed=scene_selection_seed,
        satellite_manipulator_id=satellite_manipulator_id,
        target_satellite_id=target_satellite_id or target_asset,
    )
    if target_position_noise is None:
        target_position_noise = scene_spec.target_position_noise
    if target_position_noise_range is None:
        target_position_noise_range = scene_spec.target_position_noise_range
    vec_env = BatchedGenesisStage1VecEnv(
        target_asset_name=target_satellite_id,
        satellite_manipulator_id=satellite_manipulator_id,
        target_satellite_id=target_satellite_id,
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
    satellite_manipulator_id: str = "franka_merge",
    target_satellite_id: str | None = None,
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
        satellite_manipulator_id=satellite_manipulator_id,
        target_satellite_id=target_satellite_id,
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
