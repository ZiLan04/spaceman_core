# 输入：机械臂关节角，腕部相机图像
# 输出：机械臂关节角

"""Stage-2 batched RL task: hold the base and move the arm to grasp pose.

The stage-2 policy sees arm joint positions, joint velocities, the previous
action, and the wrist camera image. It commands six normalized joint
increments. Target grasp pose information is
used only inside reward/success computation.
"""

import math
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
from utils.utils import as_rotation_matrix, calculate_camera_pose  # noqa: E402


class ParallelStage2ArmTask(ParallelRLTask):
    """Batched stage-2 task for arm-only visual servo grasp-pose training."""

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
        stage2_base_distance: float = 1.1,
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
        self.task_name = "RL_task_parallel_stage2"
        self.enable_target_relative_obs = (not self.enable_visual_obs) and (
            True if enable_target_relative_obs is None else bool(enable_target_relative_obs)
        )

        self.arm_control_decimation = 4
        self.control_dt = self.control_dt * self.arm_control_decimation
        self.max_steps = 200
        self.joint_action_low = torch.full(
            (6,), -1.0, dtype=self.datatype, device=self.device
        )
        self.joint_action_high = torch.full(
            (6,), 1.0, dtype=self.datatype, device=self.device
        )
        self.joint_target_low = torch.full(
            (6,), -math.pi, dtype=self.datatype, device=self.device
        )
        self.joint_target_high = torch.full(
            (6,), math.pi, dtype=self.datatype, device=self.device
        )
        self.joint_delta_scale = torch.tensor(
            [0.040, 0.040, 0.040, 0.050, 0.060, 0.050],
            dtype=self.datatype,
            device=self.device,
        )

        self.stage2_base_distance = float(stage2_base_distance)
        self.target_dock_dir_local = torch.tensor(
            [-1.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )
        self.stage2_base_hold_pos = None
        self.stage2_base_hold_quat = None

        self.reference_yaw_joint_name = (
            "starlink_space_manipulator_merged.SLDASM9_"
            "starlink__space_manipulator_2F-Body_Link_yaw"
        )
        self.reference_yaw_link_name = (
            "starlink_space_manipulator_merged.SLDASM9_"
            "starlink__space_manipulator_2F-Body_Link"
        )
        self.reference_yaw_link_xyz = torch.tensor(
            [-1.65419686, -0.06428438, 0.92678845],
            dtype=self.datatype,
            device=self.device,
        )
        self.reference_yaw_link_quat = torch.tensor(
            [-0.17702265, -0.20225112, -0.6957292, -0.66612172],
            dtype=self.datatype,
            device=self.device,
        )
        self.reference_target_offset_local = None
        self.reference_target_quat_local = None
        self.desired_reference_link_pose = None

        self.tcp_position_threshold = 0.025
        self.pose_orientation_angle_threshold = math.radians(15.0)
        self.pose_quat_similarity_threshold = math.cos(
            0.5 * self.pose_orientation_angle_threshold
        )
        self.hold_velocity_threshold = 0.04
        self.success_hold_steps = 10

        self.tcp_progress_reward_scale = 40.0
        self.tcp_position_penalty_scale = 0.5
        self.orientation_progress_reward_scale = 20.0
        self.pose_position_penalty_scale = 0.0
        self.pose_orientation_penalty_scale = 0.35
        self.orientation_gate_start_distance = 0.65
        self.orientation_gate_full_distance = 0.25
        self.orientation_gate_min = 0.50
        self.pose_dense_reward_scale = 4.0
        self.pose_dense_distance_width = 0.10
        self.pose_dense_orientation_width = 0.45
        self.speed_gate_start_distance = 0.15
        self.speed_gate_full_distance = 0.05
        self.speed_penalty_linear_scale = 4.0
        self.speed_penalty_scale = 8.0
        self.speed_limit_distance_scale = 2.0
        self.speed_limit_max = 0.30
        self.joint_action_penalty_scale = 3e-3
        self.near_target_action_penalty_scale = 2.7e-2
        self.base_drift_penalty_scale = 8.0
        self.target_out_of_view_penalty = 2.0
        self.target_view_edge_margin = 0.03
        self.hold_zone_bonus = 8.0
        self.stable_hold_bonus = 20.0
        self.hold_zone_leave_penalty = 20.0
        self.success_bonus = 250.0
        self.approach_milestone_distance = 0.15
        self.prepose_milestone_distance = 0.10
        self.prepose_milestone_angle = math.radians(45.0)
        self.midpose_milestone_distance = 0.075
        self.midpose_milestone_angle = math.radians(35.0)
        self.nearpose_milestone_distance = 0.05
        self.nearpose_milestone_angle = math.radians(25.0)
        self.precision_milestone_distance = 0.035
        self.precision_milestone_angle = math.radians(20.0)
        self.approach_milestone_bonus = 5.0
        self.prepose_milestone_bonus = 15.0
        self.midpose_milestone_bonus = 20.0
        self.nearpose_milestone_bonus = 30.0
        self.precision_milestone_bonus = 40.0

        self.target_hold_count = None
        self.prev_in_hold_zone = None
        self.ever_in_hold_zone = None
        self.reached_approach_milestone = None
        self.reached_prepose_milestone = None
        self.reached_midpose_milestone = None
        self.reached_nearpose_milestone = None
        self.reached_precision_milestone = None
        self.episode_min_tcp_to_grasp_error = None
        self.episode_min_pose_orientation_angle_deg = None
        self.episode_max_hold_count = None
        self.prev_tcp_to_grasp_error = None
        self.prev_pose_orientation_angle = None
        self.prev_tcp_pos = None
        self.prev_reference_link_pos = None
        self.tcp_speed = None
        self.reference_link_speed = None
        self.base_drift_error = None
        self.pose_quat_similarity = None
        self.pose_orientation_error = None
        self.pose_orientation_angle = None
        self.pose_orientation_angle_deg = None
        self.orientation_reward_gate = None
        self.speed_reward_gate = None
        self.tool_speed = None
        self.tool_speed_limit = None
        self.pose_position_error = None
        self.reference_yaw_link_pose = None
        self.tcp_to_grasp_error = None
        self.target_in_wrist_view = None
        self.target_wrist_camera_uv = None
        self.target_wrist_camera_depth = None
        self.reference_target_pose_link = None
        zero_reward = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )
        self.reward_tcp = zero_reward
        self.reward_orientation = zero_reward
        self.reward_pose_dense = zero_reward
        self.reward_speed = zero_reward
        self.reward_action = zero_reward
        self.reward_link_position = zero_reward
        self.reward_base = zero_reward
        self.reward_view = zero_reward
        self.reward_approach_milestone = zero_reward
        self.reward_prepose_milestone = zero_reward
        self.reward_midpose_milestone = zero_reward
        self.reward_nearpose_milestone = zero_reward
        self.reward_precision_milestone = zero_reward
        self.reward_hold_zone = zero_reward
        self.reward_stable_hold = zero_reward
        self.reward_hold_zone_leave = zero_reward
        self.reward_success = zero_reward
        self.reward_time = zero_reward
        self.episode_reward_sums = {
            reward_attr: torch.zeros_like(zero_reward)
            for reward_attr in self._stage2_reward_attr_names()
        }
        self.last_action_tensor = torch.zeros(
            (self.num_envs, 6), dtype=self.datatype, device=self.device
        )
        self.last_step_reward = torch.zeros(
            self.num_envs, dtype=self.datatype, device=self.device
        )

    def _zero_stage2_image_observation(self) -> torch.Tensor:
        return self._zero_image_observation()

    @staticmethod
    def _stage2_reward_attr_names() -> tuple[str, ...]:
        return (
            "reward_tcp",
            "reward_orientation",
            "reward_pose_dense",
            "reward_speed",
            "reward_action",
            "reward_link_position",
            "reward_base",
            "reward_view",
            "reward_approach_milestone",
            "reward_prepose_milestone",
            "reward_midpose_milestone",
            "reward_nearpose_milestone",
            "reward_precision_milestone",
            "reward_hold_zone",
            "reward_stable_hold",
            "reward_hold_zone_leave",
            "reward_success",
            "reward_time",
        )

    def _capture_wrist_image_observation(self) -> Optional[torch.Tensor]:
        if not self.enable_visual_obs:
            return None
        return self._capture_named_visual_observation("rl_wrist_camera")

    def _compute_stage2_base_hold_pose(
        self, grasp_point_pos: torch.Tensor, grasp_point_quat: torch.Tensor
    ):
        target_rot = as_rotation_matrix(grasp_point_quat, order="wxyz")
        dock_dir_world = torch.matmul(
            target_rot, self.target_dock_dir_local.view(3, 1)
        ).squeeze(-1)
        dock_dir_xy = dock_dir_world[:, :2]
        dock_dir_xy = dock_dir_xy / torch.clamp(
            torch.norm(dock_dir_xy, dim=-1, keepdim=True), min=1e-6
        )
        base_pos = self.initial_base_pos.detach().clone()
        base_pos[:, :2] = grasp_point_pos[:, :2] + dock_dir_xy * self.stage2_base_distance
        base_quat = self.initial_base_quat.detach().clone()
        return base_pos, base_quat

    def _set_base_hold_pid_target(self):
        if self.stage2_base_hold_pos is None or self.stage2_base_hold_quat is None:
            return
        if not hasattr(self.starlink_manipulator, "pid"):
            raise RuntimeError("Stage-2 base hold requires a base PID backend.")
        if hasattr(self.starlink_manipulator, "set_pid_enabled"):
            self.starlink_manipulator.set_pid_enabled(True)
        setpoint = self._root_target_to_pid_link_setpoint(
            self.stage2_base_hold_pos, self.stage2_base_hold_quat
        )
        self.starlink_manipulator.pid.update_setpoint(setpoint)

    def _compute_orientation_metrics(self, tcp_quat: torch.Tensor):
        reference_quat = self.reference_yaw_link_quat.view(1, 4).expand_as(tcp_quat)
        quat_similarity = torch.abs(torch.sum(tcp_quat * reference_quat, dim=-1))
        return quat_similarity, 1.0 - torch.clamp(quat_similarity, max=1.0)

    def _get_reference_yaw_link_pose(self):
        yaw_link = self.starlink_manipulator.robot.get_link(self.reference_yaw_link_name)
        link_pos = _as_batched_tensor(
            yaw_link.get_pos(),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=3,
        )
        link_quat = _as_batched_tensor(
            yaw_link.get_quat(),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=4,
        )
        return link_pos, link_quat

    def _initialize_reference_target_transform(
        self, grasp_pos: torch.Tensor, grasp_quat: torch.Tensor
    ) -> None:
        grasp_rot = as_rotation_matrix(grasp_quat, order="wxyz")
        nominal_target_pos = self.reference_yaw_link_xyz.view(1, 3).expand_as(grasp_pos)
        nominal_target_quat = self.reference_yaw_link_quat.view(1, 4).expand_as(
            grasp_quat
        )
        self.reference_target_offset_local = torch.matmul(
            grasp_rot.transpose(-1, -2),
            (nominal_target_pos - grasp_pos).unsqueeze(-1),
        ).squeeze(-1)
        grasp_quat_inv = grasp_quat.detach().clone()
        grasp_quat_inv[:, 1:] = -grasp_quat_inv[:, 1:]
        self.reference_target_quat_local = self._quaternion_multiply_wxyz(
            grasp_quat_inv, nominal_target_quat
        ).detach()

    def _compute_desired_reference_pose(
        self, grasp_pos: torch.Tensor, grasp_quat: torch.Tensor
    ):
        if (
            self.reference_target_offset_local is None
            or self.reference_target_quat_local is None
        ):
            target_pos = self.reference_yaw_link_xyz.view(1, 3).expand_as(grasp_pos)
            target_quat = self.reference_yaw_link_quat.view(1, 4).expand_as(grasp_quat)
            return target_pos, target_quat

        grasp_rot = as_rotation_matrix(grasp_quat, order="wxyz")
        target_pos = grasp_pos + torch.matmul(
            grasp_rot, self.reference_target_offset_local.unsqueeze(-1)
        ).squeeze(-1)
        target_quat = self._quaternion_multiply_wxyz(
            grasp_quat, self.reference_target_quat_local
        )
        return target_pos, target_quat

    def _compute_reference_pose_metrics(
        self,
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ):
        position_error = torch.norm(link_pos - target_pos, dim=-1)
        link_quat = link_quat / torch.clamp(
            torch.norm(link_quat, dim=-1, keepdim=True), min=1e-8
        )
        target_quat = target_quat / torch.clamp(
            torch.norm(target_quat, dim=-1, keepdim=True), min=1e-8
        )
        quat_similarity = torch.abs(torch.sum(link_quat * target_quat, dim=-1))
        quat_similarity = torch.clamp(quat_similarity, min=0.0, max=1.0)
        orientation_angle = 2.0 * torch.acos(quat_similarity)
        return position_error, quat_similarity, orientation_angle

    def _compute_orientation_reward_gate(
        self, position_error: torch.Tensor
    ) -> torch.Tensor:
        gate_range = max(
            self.orientation_gate_start_distance
            - self.orientation_gate_full_distance,
            1e-6,
        )
        gate_progress = torch.clamp(
            (self.orientation_gate_start_distance - position_error) / gate_range,
            min=0.0,
            max=1.0,
        )
        return self.orientation_gate_min + (
            1.0 - self.orientation_gate_min
        ) * gate_progress

    def _compute_speed_reward_gate(self, position_error: torch.Tensor) -> torch.Tensor:
        gate_range = max(
            self.speed_gate_start_distance - self.speed_gate_full_distance,
            1e-6,
        )
        return torch.clamp(
            (self.speed_gate_start_distance - position_error) / gate_range,
            min=0.0,
            max=1.0,
        )

    def _compute_tool_speed_limit(self, position_error: torch.Tensor) -> torch.Tensor:
        speed_limit = (
            self.hold_velocity_threshold
            + self.speed_limit_distance_scale
            * torch.clamp(
                position_error - self.tcp_position_threshold,
                min=0.0,
            )
        )
        return torch.clamp(
            speed_limit,
            min=self.hold_velocity_threshold,
            max=self.speed_limit_max,
        )

    def _update_tcp_motion_metrics(self, tcp_pos: torch.Tensor):
        if self.prev_tcp_pos is None:
            self.prev_tcp_pos = tcp_pos.detach().clone()
            self.tcp_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            return
        self.tcp_speed = torch.norm(tcp_pos - self.prev_tcp_pos, dim=-1) / max(
            self.control_dt, 1e-6
        )
        self.prev_tcp_pos = tcp_pos.detach().clone()

    def _update_reference_link_motion_metrics(self, link_pos: torch.Tensor):
        if self.prev_reference_link_pos is None:
            self.prev_reference_link_pos = link_pos.detach().clone()
            self.reference_link_speed = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            return
        self.reference_link_speed = torch.norm(
            link_pos - self.prev_reference_link_pos, dim=-1
        ) / max(self.control_dt, 1e-6)
        self.prev_reference_link_pos = link_pos.detach().clone()

    def _stage2_hold_zone_mask(self) -> torch.Tensor:
        if (
            self.tcp_to_grasp_error is None
            or self.pose_orientation_angle is None
            or self.target_in_wrist_view is None
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            (self.tcp_to_grasp_error <= self.tcp_position_threshold)
            & (self.pose_orientation_angle <= self.pose_orientation_angle_threshold)
            & self.target_in_wrist_view
        )

    def _stage2_stable_hold_mask(self) -> torch.Tensor:
        if self.tool_speed is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            self._stage2_hold_zone_mask()
            & (self.tool_speed <= self.hold_velocity_threshold)
        )

    def _update_stage2_hold_count(self):
        if self.target_hold_count is None:
            self.target_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
        stable_mask = self._stage2_stable_hold_mask()
        self.target_hold_count = torch.where(
            stable_mask,
            self.target_hold_count + 1,
            torch.zeros_like(self.target_hold_count),
        )

    def _compute_target_wrist_camera_visibility(self, target_pos: torch.Tensor):
        if not self.enable_visual_obs:
            visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            uv = torch.full((self.num_envs, 2), 0.5, dtype=self.datatype, device=self.device)
            depth = torch.ones(self.num_envs, dtype=self.datatype, device=self.device)
            return visible, uv, depth

        camera_config = getattr(self.starlink_manipulator, "wrist_camera_params", {})
        image_config = camera_config.get("camera", {})
        resolution = image_config.get("res", (self.image_shape[-1], self.image_shape[-2]))
        width, height = float(resolution[0]), float(resolution[1])
        aspect = max(width / max(height, 1.0), 1e-6)
        vertical_fov = math.radians(float(image_config.get("fov", 70.0)))
        tan_v = math.tan(0.5 * vertical_fov)
        tan_h = tan_v * aspect

        mount_link_name = camera_config.get(
            "end_effector_link", self.starlink_manipulator.params["end_effector"]
        )
        mount_link = self.starlink_manipulator.robot.get_link(mount_link_name)
        mount_pos = _as_batched_tensor(
            mount_link.get_pos(),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=3,
        )
        mount_quat = _as_batched_tensor(
            mount_link.get_quat(),
            device=self.device,
            dtype=self.datatype,
            expected_last_dim=4,
        )

        cam_pos, cam_lookat, cam_up = calculate_camera_pose(
            mount_pos,
            mount_quat,
            camera_config.get("pos_offset"),
            camera_config.get("lookat_offset"),
            camera_config.get("up_offset"),
        )
        forward = cam_lookat - cam_pos
        forward = forward / torch.clamp(torch.norm(forward, dim=-1, keepdim=True), min=1e-6)
        up_hint = cam_up / torch.clamp(torch.norm(cam_up, dim=-1, keepdim=True), min=1e-6)
        right = torch.cross(forward, up_hint, dim=-1)
        right = right / torch.clamp(torch.norm(right, dim=-1, keepdim=True), min=1e-6)
        up = torch.cross(right, forward, dim=-1)
        up = up / torch.clamp(torch.norm(up, dim=-1, keepdim=True), min=1e-6)

        rel = target_pos - cam_pos
        depth = torch.sum(rel * forward, dim=-1)
        x_over_z = torch.sum(rel * right, dim=-1) / torch.clamp(depth, min=1e-6)
        y_over_z = torch.sum(rel * up, dim=-1) / torch.clamp(depth, min=1e-6)
        uv = torch.stack(
            [
                0.5 + 0.5 * x_over_z / max(tan_h, 1e-6),
                0.5 - 0.5 * y_over_z / max(tan_v, 1e-6),
            ],
            dim=-1,
        )

        margin = float(self.target_view_edge_margin)
        visible = (
            (depth > 1e-3)
            & (uv[:, 0] >= margin)
            & (uv[:, 0] <= 1.0 - margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 1] <= 1.0 - margin)
        )
        return visible, uv, depth

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
        return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-6)

    def _compute_target_pose_in_frame(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        frame_pos: torch.Tensor,
        frame_quat: torch.Tensor,
    ) -> torch.Tensor:
        frame_rot = as_rotation_matrix(frame_quat, order="wxyz")
        relative_pos = torch.matmul(
            frame_rot.transpose(-1, -2),
            (target_pos - frame_pos).unsqueeze(-1),
        ).squeeze(-1)
        frame_quat_inv = frame_quat.clone()
        frame_quat_inv[:, 1:] = -frame_quat_inv[:, 1:]
        relative_quat = self._quaternion_multiply_wxyz(frame_quat_inv, target_quat)
        return torch.cat([relative_pos, relative_quat], dim=-1)

    def _update_io_tensors(self):
        tcp_pos, tcp_quat = self._get_tcp_pose()
        reference_link_pos, reference_link_quat = self._get_reference_yaw_link_pose()
        grasp_point_pos, grasp_point_quat = self._get_target_grasp_pose()
        grasp_marker_pos = self._get_grasp_marker_pos(grasp_point_pos, grasp_point_quat)
        self._sync_grasp_point_marker(grasp_point_pos, grasp_point_quat)
        base_pos, base_quat = self._get_base_pose()
        joint_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )
        joint_qpos = _as_batched_tensor(
            joint_qpos, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        joint_qvel = self.starlink_manipulator.robot.get_dofs_velocity(
            dofs_idx_local=self.starlink_manipulator.motors_dof
        )
        joint_qvel = _as_batched_tensor(
            joint_qvel, device=self.device, dtype=self.datatype, expected_last_dim=6
        )

        reference_target_pos, reference_target_quat = (
            self._compute_desired_reference_pose(grasp_point_pos, grasp_point_quat)
        )
        self.desired_reference_link_pose = torch.cat(
            [reference_target_pos, reference_target_quat], dim=-1
        )
        self.reference_target_pose_link = self._compute_target_pose_in_frame(
            reference_target_pos,
            reference_target_quat,
            reference_link_pos,
            reference_link_quat,
        )

        self.tcp_pose = torch.cat([tcp_pos, tcp_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.reference_yaw_link_pose = torch.cat(
            [reference_link_pos, reference_link_quat], dim=-1
        ).to(dtype=self.datatype, device=self.device)
        self.target_grasp_pose = torch.cat([grasp_point_pos, grasp_point_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.base_satellite_pose = torch.cat([base_pos, base_quat], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self.arm_joint_positions = joint_qpos.reshape(self.num_envs, -1)
        self.arm_joint_velocities = joint_qvel.reshape(self.num_envs, -1)
        self.base_planar_position = base_pos[:, :2].to(dtype=self.datatype, device=self.device)
        previous_action = self.last_action_tensor
        if previous_action is None:
            previous_action = torch.zeros_like(self.arm_joint_positions)
        state_tensors = [
            self.arm_joint_positions,
            self.arm_joint_velocities,
            previous_action,
        ]
        if self.enable_target_relative_obs:
            state_tensors.append(self.reference_target_pose_link)
        self.state_observation = torch.cat(state_tensors, dim=-1)

        if self.enable_visual_obs:
            self.wrist_image_observation = self._capture_wrist_image_observation()
            self.image_observation = self.wrist_image_observation
            self.base_image_observation = None
            self.observation = {
                "state": self.state_observation,
                "wrist_image": self.wrist_image_observation,
            }
        else:
            self.image_observation = None
            self.wrist_image_observation = None
            self.base_image_observation = None
            self.observation = self.state_observation

        self.action = self.arm_joint_positions
        self.base_alignment_error = torch.norm(
            grasp_point_pos[:, :2] - self.base_planar_position, dim=-1
        )
        self.tcp_to_grasp_error = torch.norm(grasp_point_pos - tcp_pos, dim=-1)
        (
            self.pose_position_error,
            self.pose_quat_similarity,
            self.pose_orientation_angle,
        ) = self._compute_reference_pose_metrics(
            reference_link_pos,
            reference_link_quat,
            reference_target_pos,
            reference_target_quat,
        )
        self.orientation_reward_gate = self._compute_orientation_reward_gate(
            self.tcp_to_grasp_error
        )
        self.speed_reward_gate = self._compute_speed_reward_gate(
            self.tcp_to_grasp_error
        )
        self.pose_orientation_error = self.pose_orientation_angle
        self.pose_orientation_angle_deg = torch.rad2deg(
            self.pose_orientation_angle
        )
        self.quat_similarity = self.pose_quat_similarity
        self.down_alignment = self.pose_quat_similarity
        self.approach_alignment = self.pose_quat_similarity
        (
            self.target_in_wrist_view,
            self.target_wrist_camera_uv,
            self.target_wrist_camera_depth,
        ) = self._compute_target_wrist_camera_visibility(grasp_marker_pos)
        self._update_tcp_motion_metrics(tcp_pos)
        self._update_reference_link_motion_metrics(reference_link_pos)
        self.tool_speed = torch.maximum(self.tcp_speed, self.reference_link_speed)
        self.tool_speed_limit = self._compute_tool_speed_limit(
            self.tcp_to_grasp_error
        )

        if self.stage2_base_hold_pos is not None:
            self.base_drift_error = torch.norm(
                base_pos[:, :2] - self.stage2_base_hold_pos[:, :2], dim=-1
            )
        else:
            self.base_drift_error = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )

        self._update_stage2_hold_count()
        success_mask_np = self.is_success().detach().cpu().numpy()
        near_mask_np = self._stage2_hold_zone_mask().detach().cpu().numpy()
        self.phase = np.where(near_mask_np, "pose_hold", "arm_approach").astype(object)
        self.phase = np.where(success_mask_np, "stage2_success", self.phase).astype(object)

    def initialize(self) -> bool:
        result = super().initialize()
        if result:
            grasp_pos, grasp_quat = self._get_target_grasp_pose()
            self._initialize_reference_target_transform(grasp_pos, grasp_quat)
            self.stage2_base_hold_pos, self.stage2_base_hold_quat = (
                self._compute_stage2_base_hold_pose(grasp_pos, grasp_quat)
            )
            self.initial_base_pos = self.stage2_base_hold_pos.detach().clone()
            self.initial_base_quat = self.stage2_base_hold_quat.detach().clone()
            self.joint_reference = self._as_task_tensor(
                self.starlink_manipulator.config["initial_dofs"][:6],
                expected_last_dim=6,
            ).detach().clone()
            if self.joint_reference.shape[0] == 1 and self.num_envs > 1:
                self.joint_reference = self.joint_reference.repeat(self.num_envs, 1)
            self.initial_joint_positions = self.joint_reference.detach().clone()
            self.target_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.prev_in_hold_zone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.ever_in_hold_zone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reached_approach_milestone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reached_prepose_milestone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reached_midpose_milestone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reached_nearpose_milestone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reached_precision_milestone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.episode_min_tcp_to_grasp_error = torch.full(
                (self.num_envs,),
                float("inf"),
                dtype=self.datatype,
                device=self.device,
            )
            self.episode_min_pose_orientation_angle_deg = torch.full(
                (self.num_envs,),
                float("inf"),
                dtype=self.datatype,
                device=self.device,
            )
            self.episode_max_hold_count = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self.prev_tcp_to_grasp_error = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.prev_pose_orientation_angle = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            self.prev_tcp_pos = None
            self.prev_reference_link_pos = None
            self.last_action_tensor = torch.zeros(
                (self.num_envs, 6), dtype=self.datatype, device=self.device
            )
            self.last_step_reward = torch.zeros(
                self.num_envs, dtype=self.datatype, device=self.device
            )
            if not self.reset_done_envs(torch.arange(self.num_envs, device=self.device)):
                return False
        return result

    def _clamp_joint_targets(self, joint_targets: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            joint_targets,
            min=self.joint_target_low.view(1, -1),
            max=self.joint_target_high.view(1, -1),
        )

    def apply_rl_action(self, action: torch.Tensor):
        action = _as_batched_tensor(
            action, device=self.device, dtype=self.datatype, expected_last_dim=6
        )
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Stage-2 action should have shape ({self.num_envs}, 6), got {tuple(action.shape)}"
            )
        joint_delta_action = torch.clamp(action, -1.0, 1.0)
        joint_targets = self._clamp_joint_targets(
            self.arm_joint_positions + joint_delta_action * self.joint_delta_scale
        )
        self.last_action_tensor = joint_delta_action.detach().clone()
        self._set_base_hold_pid_target()
        self.starlink_manipulator.control_joint_pos(joint_targets)
        self.last_gripper_command = torch.full(
            (self.num_envs,),
            self.gripper_open_value,
            dtype=self.datatype,
            device=self.device,
        )
        self.starlink_manipulator.control_gripper(self.last_gripper_command)

    def reward(self) -> torch.Tensor:
        if self.pose_position_error is None:
            return torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)

        self.episode_min_tcp_to_grasp_error = torch.minimum(
            self.episode_min_tcp_to_grasp_error,
            self.tcp_to_grasp_error,
        )
        self.episode_min_pose_orientation_angle_deg = torch.minimum(
            self.episode_min_pose_orientation_angle_deg,
            self.pose_orientation_angle_deg,
        )
        self.episode_max_hold_count = torch.maximum(
            self.episode_max_hold_count,
            self.target_hold_count,
        )

        if self.prev_tcp_to_grasp_error is None:
            tcp_progress = torch.zeros_like(self.tcp_to_grasp_error)
        else:
            tcp_progress = (
                self.prev_tcp_to_grasp_error - self.tcp_to_grasp_error
            )
        self.prev_tcp_to_grasp_error = self.tcp_to_grasp_error.detach().clone()

        if self.prev_pose_orientation_angle is None:
            orientation_progress = torch.zeros_like(self.pose_orientation_angle)
        else:
            orientation_progress = (
                self.prev_pose_orientation_angle - self.pose_orientation_angle
            )
        self.prev_pose_orientation_angle = self.pose_orientation_angle.detach().clone()
        self.reward_tcp = (
            self.tcp_progress_reward_scale * tcp_progress
            - self.tcp_position_penalty_scale * self.tcp_to_grasp_error
        )
        self.reward_orientation = self.orientation_reward_gate * (
            self.orientation_progress_reward_scale * orientation_progress
            - self.pose_orientation_penalty_scale * self.pose_orientation_angle
        )
        self.reward_pose_dense = self.pose_dense_reward_scale * torch.exp(
            -torch.square(
                self.tcp_to_grasp_error / self.pose_dense_distance_width
            )
            -torch.square(
                self.pose_orientation_angle / self.pose_dense_orientation_width
            )
        )
        speed_excess = torch.clamp(
            self.tool_speed - self.tool_speed_limit,
            min=0.0,
        )
        self.reward_speed = -self.speed_reward_gate * (
            self.speed_penalty_linear_scale * speed_excess
            + self.speed_penalty_scale * speed_excess.square()
        )
        self.reward_link_position = (
            -self.pose_position_penalty_scale * self.pose_position_error
        )
        self.reward_base = -self.base_drift_penalty_scale * self.base_drift_error
        if self.target_in_wrist_view is not None:
            self.reward_view = -torch.where(
                self.target_in_wrist_view,
                torch.zeros_like(self.tcp_to_grasp_error),
                torch.full_like(
                    self.tcp_to_grasp_error, self.target_out_of_view_penalty
                ),
            )
        else:
            self.reward_view = torch.zeros_like(self.tcp_to_grasp_error)

        if self.last_action_tensor is not None:
            action_magnitude = torch.sum(
                self.last_action_tensor * self.last_action_tensor, dim=-1
            )
            action_penalty_scale = (
                self.joint_action_penalty_scale
                + self.near_target_action_penalty_scale * self.speed_reward_gate
            )
            self.reward_action = -action_penalty_scale * action_magnitude
        else:
            self.reward_action = torch.zeros_like(self.tcp_to_grasp_error)

        reward = (
            self.reward_tcp
            + self.reward_orientation
            + self.reward_pose_dense
            + self.reward_speed
            + self.reward_action
            + self.reward_link_position
            + self.reward_base
            + self.reward_view
        )

        in_hold_zone = self._stage2_hold_zone_mask()
        stable_hold = self._stage2_stable_hold_mask()
        approach_milestone = (
            self.tcp_to_grasp_error <= self.approach_milestone_distance
        )
        prepose_milestone = (
            (self.tcp_to_grasp_error <= self.prepose_milestone_distance)
            & (self.pose_orientation_angle <= self.prepose_milestone_angle)
            & self.target_in_wrist_view
        )
        midpose_milestone = (
            (self.tcp_to_grasp_error <= self.midpose_milestone_distance)
            & (self.pose_orientation_angle <= self.midpose_milestone_angle)
            & self.target_in_wrist_view
        )
        nearpose_milestone = (
            (self.tcp_to_grasp_error <= self.nearpose_milestone_distance)
            & (self.pose_orientation_angle <= self.nearpose_milestone_angle)
            & self.target_in_wrist_view
        )
        precision_milestone = (
            (self.tcp_to_grasp_error <= self.precision_milestone_distance)
            & (self.pose_orientation_angle <= self.precision_milestone_angle)
            & self.target_in_wrist_view
        )
        self.reward_approach_milestone = torch.where(
            approach_milestone & (~self.reached_approach_milestone),
            torch.full_like(reward, self.approach_milestone_bonus),
            torch.zeros_like(reward),
        )
        self.reward_prepose_milestone = torch.where(
            prepose_milestone & (~self.reached_prepose_milestone),
            torch.full_like(reward, self.prepose_milestone_bonus),
            torch.zeros_like(reward),
        )
        self.reward_midpose_milestone = torch.where(
            midpose_milestone & (~self.reached_midpose_milestone),
            torch.full_like(reward, self.midpose_milestone_bonus),
            torch.zeros_like(reward),
        )
        self.reward_nearpose_milestone = torch.where(
            nearpose_milestone & (~self.reached_nearpose_milestone),
            torch.full_like(reward, self.nearpose_milestone_bonus),
            torch.zeros_like(reward),
        )
        self.reward_precision_milestone = torch.where(
            precision_milestone & (~self.reached_precision_milestone),
            torch.full_like(reward, self.precision_milestone_bonus),
            torch.zeros_like(reward),
        )
        self.reached_approach_milestone = (
            self.reached_approach_milestone | approach_milestone.detach()
        )
        self.reached_prepose_milestone = (
            self.reached_prepose_milestone | prepose_milestone.detach()
        )
        self.reached_midpose_milestone = (
            self.reached_midpose_milestone | midpose_milestone.detach()
        )
        self.reached_nearpose_milestone = (
            self.reached_nearpose_milestone | nearpose_milestone.detach()
        )
        self.reached_precision_milestone = (
            self.reached_precision_milestone | precision_milestone.detach()
        )
        if self.ever_in_hold_zone is None:
            self.ever_in_hold_zone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if self.prev_in_hold_zone is None:
            self.prev_in_hold_zone = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        left_hold_zone = self.prev_in_hold_zone & (~in_hold_zone)
        self.ever_in_hold_zone = self.ever_in_hold_zone | in_hold_zone.detach()
        self.reward_hold_zone = torch.where(
            in_hold_zone,
            torch.full_like(reward, self.hold_zone_bonus),
            torch.zeros_like(reward),
        )
        self.reward_stable_hold = torch.where(
            stable_hold,
            torch.full_like(reward, self.stable_hold_bonus),
            torch.zeros_like(reward),
        )
        self.reward_hold_zone_leave = -torch.where(
            left_hold_zone,
            torch.full_like(reward, self.hold_zone_leave_penalty),
            torch.zeros_like(reward),
        )
        self.reward_success = torch.where(
            self.is_success(),
            torch.full_like(reward, self.success_bonus),
            torch.zeros_like(reward),
        )
        self.reward_time = torch.full_like(reward, -self.time_penalty)
        reward = reward + (
            self.reward_approach_milestone
            + self.reward_prepose_milestone
            + self.reward_midpose_milestone
            + self.reward_nearpose_milestone
            + self.reward_precision_milestone
            + self.reward_hold_zone
            + self.reward_stable_hold
            + self.reward_hold_zone_leave
            + self.reward_success
            + self.reward_time
        )
        for reward_attr in self._stage2_reward_attr_names():
            self.episode_reward_sums[reward_attr] += getattr(self, reward_attr)
        self.prev_in_hold_zone = in_hold_zone.detach().clone()
        return reward.to(dtype=self.datatype, device=self.device)

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            if action is not None:
                self.apply_rl_action(action)
            else:
                self._set_base_hold_pid_target()
                self.starlink_manipulator.control_gripper(self.last_gripper_command)

            for _ in range(self.arm_control_decimation):
                self._set_base_hold_pid_target()
                self.starlink_manipulator.step()
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
            self.logger.error(f"Stage-2 step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def is_success(self):
        if self.target_hold_count is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        stable_hold_mask = self._stage2_stable_hold_mask()
        return stable_hold_mask & (self.target_hold_count >= self.success_hold_steps)

    def reset_done_envs(self, env_ids=None) -> bool:
        result = super().reset_done_envs(env_ids=env_ids)
        if result:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            else:
                env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
            if env_ids.numel() > 0:
                grasp_pos, grasp_quat = self._get_target_grasp_pose()
                hold_pos, hold_quat = self._compute_stage2_base_hold_pose(grasp_pos, grasp_quat)
                self.stage2_base_hold_pos[env_ids] = hold_pos[env_ids]
                self.stage2_base_hold_quat[env_ids] = hold_quat[env_ids]
                root_qpos = torch.cat(
                    [self.stage2_base_hold_pos[env_ids], self.stage2_base_hold_quat[env_ids]],
                    dim=-1,
                )
                self.starlink_manipulator.robot.set_qpos(
                    root_qpos,
                    qs_idx_local=list(range(7)),
                    envs_idx=env_ids,
                    zero_velocity=True,
                )
                self.starlink_manipulator.update_state()
                self._set_base_hold_pid_target()
                self.starlink_manipulator.control_gripper(
                    torch.full(
                        (self.num_envs,),
                        self.gripper_open_value,
                        dtype=self.datatype,
                        device=self.device,
                    )
                )
                if self.last_action_tensor is not None:
                    self.last_action_tensor[env_ids] = 0.0
                self._update_io_tensors()
                if self.target_hold_count is not None:
                    self.target_hold_count[env_ids] = 0
                if self.prev_in_hold_zone is not None:
                    self.prev_in_hold_zone[env_ids] = False
                if self.ever_in_hold_zone is not None:
                    self.ever_in_hold_zone[env_ids] = False
                if self.reached_approach_milestone is not None:
                    self.reached_approach_milestone[env_ids] = False
                if self.reached_prepose_milestone is not None:
                    self.reached_prepose_milestone[env_ids] = False
                if self.reached_midpose_milestone is not None:
                    self.reached_midpose_milestone[env_ids] = False
                if self.reached_nearpose_milestone is not None:
                    self.reached_nearpose_milestone[env_ids] = False
                if self.reached_precision_milestone is not None:
                    self.reached_precision_milestone[env_ids] = False
                if self.episode_min_tcp_to_grasp_error is not None:
                    self.episode_min_tcp_to_grasp_error[env_ids] = float("inf")
                if self.episode_min_pose_orientation_angle_deg is not None:
                    self.episode_min_pose_orientation_angle_deg[env_ids] = float(
                        "inf"
                    )
                if self.episode_max_hold_count is not None:
                    self.episode_max_hold_count[env_ids] = 0
                if self.prev_tcp_to_grasp_error is not None:
                    self.prev_tcp_to_grasp_error[env_ids] = self.tcp_to_grasp_error[
                        env_ids
                    ]
                if self.prev_pose_orientation_angle is not None:
                    self.prev_pose_orientation_angle[env_ids] = (
                        self.pose_orientation_angle[env_ids]
                    )
                if self.prev_tcp_pos is not None:
                    self.prev_tcp_pos[env_ids] = self._get_tcp_pose()[0][env_ids].detach().clone()
                if self.tcp_speed is not None:
                    self.tcp_speed[env_ids] = 0.0
                if self.prev_reference_link_pos is not None:
                    self.prev_reference_link_pos[env_ids] = (
                        self._get_reference_yaw_link_pose()[0][env_ids].detach().clone()
                    )
                if self.reference_link_speed is not None:
                    self.reference_link_speed[env_ids] = 0.0
                if self.tool_speed is not None:
                    self.tool_speed[env_ids] = 0.0
                if self.last_step_reward is not None:
                    self.last_step_reward[env_ids] = 0.0
                for reward_attr in self._stage2_reward_attr_names():
                    getattr(self, reward_attr)[env_ids] = 0.0
                    self.episode_reward_sums[reward_attr][env_ids] = 0.0
        return result

    def get_info(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            infos.append(
                {
                    "phase": str(self.phase[env_idx]),
                    "pose_position_error": float(
                        self.pose_position_error[env_idx].item()
                    ),
                    "pose_quat_similarity": float(
                        self.pose_quat_similarity[env_idx].item()
                    ),
                    "pose_orientation_error": float(
                        self.pose_orientation_error[env_idx].item()
                    ),
                    "pose_orientation_angle": float(
                        self.pose_orientation_angle[env_idx].item()
                    ),
                    "pose_orientation_angle_deg": float(
                        self.pose_orientation_angle_deg[env_idx].item()
                    ),
                    "orientation_reward_gate": float(
                        self.orientation_reward_gate[env_idx].item()
                    ),
                    "desired_reference_link_xyz": (
                        self.desired_reference_link_pose[env_idx, :3]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "desired_reference_link_quat_wxyz": (
                        self.desired_reference_link_pose[env_idx, 3:]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "reference_link_speed": float(
                        self.reference_link_speed[env_idx].item()
                    ),
                    "speed_reward_gate": float(
                        self.speed_reward_gate[env_idx].item()
                    ),
                    "tool_speed": float(self.tool_speed[env_idx].item()),
                    "tool_speed_limit": float(
                        self.tool_speed_limit[env_idx].item()
                    ),
                    "tcp_to_grasp_error": float(
                        self.tcp_to_grasp_error[env_idx].item()
                    ),
                    "base_drift_error": float(self.base_drift_error[env_idx].item()),
                    "tcp_speed": float(self.tcp_speed[env_idx].item()),
                    "target_hold_count": int(self.target_hold_count[env_idx].item()),
                    "ever_in_hold_zone": bool(self.ever_in_hold_zone[env_idx].item()),
                    "reached_approach_milestone": bool(
                        self.reached_approach_milestone[env_idx].item()
                    ),
                    "reached_prepose_milestone": bool(
                        self.reached_prepose_milestone[env_idx].item()
                    ),
                    "reached_midpose_milestone": bool(
                        self.reached_midpose_milestone[env_idx].item()
                    ),
                    "reached_nearpose_milestone": bool(
                        self.reached_nearpose_milestone[env_idx].item()
                    ),
                    "reached_precision_milestone": bool(
                        self.reached_precision_milestone[env_idx].item()
                    ),
                    "episode_min_tcp_to_grasp_error": float(
                        self.episode_min_tcp_to_grasp_error[env_idx].item()
                    ),
                    "episode_min_pose_orientation_angle_deg": float(
                        self.episode_min_pose_orientation_angle_deg[env_idx].item()
                    ),
                    "episode_max_hold_count": int(
                        self.episode_max_hold_count[env_idx].item()
                    ),
                    "target_in_wrist_view": bool(
                        self.target_in_wrist_view[env_idx].item()
                    ),
                    "target_wrist_u": float(
                        self.target_wrist_camera_uv[env_idx, 0].item()
                    ),
                    "target_wrist_v": float(
                        self.target_wrist_camera_uv[env_idx, 1].item()
                    ),
                    "target_wrist_depth": float(
                        self.target_wrist_camera_depth[env_idx].item()
                    ),
                    "reward_tcp": float(self.reward_tcp[env_idx].item()),
                    "reward_orientation": float(
                        self.reward_orientation[env_idx].item()
                    ),
                    "reward_pose_dense": float(
                        self.reward_pose_dense[env_idx].item()
                    ),
                    "reward_speed": float(self.reward_speed[env_idx].item()),
                    "reward_action": float(self.reward_action[env_idx].item()),
                    "reward_link_position": float(
                        self.reward_link_position[env_idx].item()
                    ),
                    "reward_base": float(self.reward_base[env_idx].item()),
                    "reward_view": float(self.reward_view[env_idx].item()),
                    "reward_approach_milestone": float(
                        self.reward_approach_milestone[env_idx].item()
                    ),
                    "reward_prepose_milestone": float(
                        self.reward_prepose_milestone[env_idx].item()
                    ),
                    "reward_midpose_milestone": float(
                        self.reward_midpose_milestone[env_idx].item()
                    ),
                    "reward_nearpose_milestone": float(
                        self.reward_nearpose_milestone[env_idx].item()
                    ),
                    "reward_precision_milestone": float(
                        self.reward_precision_milestone[env_idx].item()
                    ),
                    "reward_hold_zone": float(
                        self.reward_hold_zone[env_idx].item()
                    ),
                    "reward_stable_hold": float(
                        self.reward_stable_hold[env_idx].item()
                    ),
                    "reward_hold_zone_leave": float(
                        self.reward_hold_zone_leave[env_idx].item()
                    ),
                    "reward_success": float(self.reward_success[env_idx].item()),
                    "reward_time": float(self.reward_time[env_idx].item()),
                    "success": bool(self.success[env_idx].item()),
                }
            )
            infos[-1].update(
                {
                    f"episode_{reward_attr}": float(
                        self.episode_reward_sums[reward_attr][env_idx].item()
                    )
                    for reward_attr in self._stage2_reward_attr_names()
                }
            )
        return infos


if gym is not None and spaces is not None:

    class ParallelStage2ArmEnv(gym.Env):
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
            stage2_base_distance: float = 1.1,
        ):
            super().__init__()
            self.task = ParallelStage2ArmTask(
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
                stage2_base_distance=stage2_base_distance,
            )
            self.num_envs = self.task.num_envs
            self.state_dim = 18 + (7 if self.task.enable_target_relative_obs else 0)

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
                        "wrist_image": spaces.Box(
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
                        "wrist_image": spaces.Box(
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

            action_low = self.task.joint_action_low.detach().cpu().numpy().astype(np.float32)
            action_high = self.task.joint_action_high.detach().cpu().numpy().astype(np.float32)
            self.single_action_space = spaces.Box(
                low=action_low, high=action_high, shape=(6,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.repeat(action_low[None, :], self.num_envs, axis=0),
                high=np.repeat(action_high[None, :], self.num_envs, axis=0),
                shape=(self.num_envs, 6),
                dtype=np.float32,
            )

        def _get_obs(self):
            if self.task.enable_visual_obs:
                image_obs = self.task.wrist_image_observation
                if image_obs is None:
                    image_obs = self.task._zero_stage2_image_observation()
                image_obs = image_obs.detach().cpu().numpy()
                if self.task.normalize_images:
                    image_obs = image_obs.astype(np.float32)
                return {
                    "state": self.task.state_observation.detach().cpu().numpy().astype(np.float32),
                    "wrist_image": image_obs,
                }
            return self.task.observation.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError("ParallelStage2ArmTask initialize failed during env reset")
            else:
                if not self.task.reset():
                    raise RuntimeError("ParallelStage2ArmTask reset failed during env reset")
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

    class BatchedGenesisStage2VecEnv(VecEnv):
        """SB3-compatible VecEnv wrapper over one batched Genesis stage-2 scene."""

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
            stage2_base_distance: float = 1.1,
        ):
            self.env = ParallelStage2ArmEnv(
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
                stage2_base_distance=stage2_base_distance,
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
                    raise RuntimeError("ParallelStage2ArmTask partial reset failed")
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
                    "BatchedGenesisStage2VecEnv only supports setting attributes for all envs."
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


def build_stage2_env(
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
    stage2_base_distance: float = 1.1,
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    return ParallelStage2ArmEnv(
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
        stage2_base_distance=stage2_base_distance,
    )


def make_stage2_vec_env(
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
    stage2_base_distance: float = 1.1,
):
    if VecEnv is None or VecMonitor is None:
        raise ImportError("stable_baselines3 vectorized env utilities are not installed.")
    vec_env = BatchedGenesisStage2VecEnv(
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
        stage2_base_distance=stage2_base_distance,
    )
    return VecMonitor(vec_env)


def train_stage2_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_satellite_stage2_arm",
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
    env = make_stage2_vec_env(
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
    print(f"Using stage-2 PPO training device: {training_device}")
    print(f"Using Genesis batched scene count: {num_envs}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.close()
    return model


__all__ = [
    "ParallelStage2ArmTask",
    "ParallelStage2ArmEnv",
    "BatchedGenesisStage2VecEnv",
    "build_stage2_env",
    "make_stage2_vec_env",
    "train_stage2_ppo",
]
