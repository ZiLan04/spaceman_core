"""Show the ideal Stage-2 success pose in the Genesis viewer.

The test uses the real Stage-2 task metrics. It places the target satellite at
its nominal pose, places the base at the Stage-2 PID hold pose, solves the arm
IK for the task's fixed reference-link pose, and holds all of them in place.
No wrist, base, or recording cameras are created.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import genesis as gs
import torch


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs import asset_configs  # noqa: E402
from configs.config_starlink import make_starlink_morph  # noqa: E402
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from tasks.RL_task_parallel import TaskStatus  # noqa: E402
from tasks.RL_task_parallel_stage2 import ParallelStage2ArmTask  # noqa: E402
from utils.utils import as_rotation_matrix  # noqa: E402


class Stage2IdealViewerTask(ParallelStage2ArmTask):
    """Stage-2 task with a non-colliding marker that follows the TCP."""

    tcp_marker_length = 0.18
    tcp_marker_thickness = 0.015

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tcp_markers = []

    def _create_grasp_point_marker(self):
        grasp_marker = super()._create_grasp_point_marker()
        length = self.tcp_marker_length
        thickness = self.tcp_marker_thickness
        axis_sizes = {
            "x": (length, thickness, thickness),
            "y": (thickness, length, thickness),
            "z": (thickness, thickness, length),
        }
        self.tcp_markers = [
            self._scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(-10.0, -10.0, -10.0),
                    size=size,
                    fixed=True,
                    collision=False,
                    visualization=True,
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
                surface=gs.surfaces.Plastic(color=(0.05, 1.0, 0.05, 1.0)),
                name=f"stage2_ideal_tcp_marker_{axis_name}",
            )
            for axis_name, size in axis_sizes.items()
        ]
        return grasp_marker

    def _update_io_tensors(self):
        super()._update_io_tensors()
        if self.tcp_markers and self.tcp_pose is not None:
            for marker in self.tcp_markers:
                marker.set_pos(self.tcp_pose[:, :3], zero_velocity=True)


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display and verify the ideal Stage-2 RL success pose."
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--viewer-steps",
        type=int,
        default=0,
        help="Decision steps to show after success; 0 means until the viewer closes.",
    )
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--stage2-base-distance", type=float, default=1.1)
    parser.add_argument(
        "--viewer-pos",
        type=parse_vec3,
        default=(-1.6, -3.2, 2.0),
    )
    parser.add_argument(
        "--viewer-lookat",
        type=parse_vec3,
        default=(-1.6, 0.0, 0.65),
    )
    parser.add_argument("--viewer-fov", type=float, default=45.0)
    return parser


def configure_no_camera_viewer(args: argparse.Namespace) -> None:
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not args.no_viewer
    SCENE_SETTINGS["viewer_options"] = gs.options.ViewerOptions(
        camera_pos=args.viewer_pos,
        camera_lookat=args.viewer_lookat,
        camera_fov=float(args.viewer_fov),
    )
    TPV_CAM_SETTINGS["enable_recording"] = False

    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enabled"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["enabled"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]["enabled"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]["enable_recording"] = False


def configure_collision_free_target() -> None:
    def make_starlink_for_stage2_viewer():
        return {
            "morph": make_starlink_morph(
                collision=False,
                visualization=True,
            ),
            "material": gs.materials.Rigid(gravity_compensation=1.0),
        }

    asset_configs._ASSET_FACTORIES["starlink"] = make_starlink_for_stage2_viewer


def as_batch(value, task, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=task.datatype, device=task.device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (task.num_envs, width):
        raise ValueError(f"Expected shape ({task.num_envs}, {width}), got {tuple(tensor.shape)}")
    return tensor


def solve_reference_joint_target(task: ParallelStage2ArmTask) -> torch.Tensor:
    grasp_pos, grasp_quat = task._get_target_grasp_pose()
    target_pos, target_quat = task._compute_desired_reference_pose(
        grasp_pos, grasp_quat
    )
    initial_qpos = torch.as_tensor(
        task.starlink_manipulator.robot.get_qpos(),
        dtype=task.datatype,
        device=task.device,
    )
    solution = task.starlink_manipulator.robot.inverse_kinematics(
        link=task.starlink_manipulator.robot.get_link(task.reference_yaw_link_name),
        pos=target_pos,
        quat=target_quat,
        init_qpos=initial_qpos,
        dofs_idx_local=task.starlink_manipulator.motors_dof,
        max_samples=8,
        max_solver_iters=50,
        damping=0.03,
        max_step_size=0.2,
        rot_mask=[True, True, True],
    )
    solution = torch.as_tensor(solution, dtype=task.datatype, device=task.device)
    if solution.dim() == 1:
        solution = solution.unsqueeze(0)
    return solution[:, task.starlink_manipulator.motors_qs].detach().clone()


def verify_target_relative_pose(task: ParallelStage2ArmTask) -> None:
    grasp_pos, grasp_quat = task._get_target_grasp_pose()
    target_pos, target_quat = task._compute_desired_reference_pose(
        grasp_pos, grasp_quat
    )

    half_yaw = torch.tensor(0.31, dtype=task.datatype, device=task.device)
    delta_quat = torch.stack(
        [
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ]
    ).view(1, 4).expand(task.num_envs, 4)
    delta_rot = as_rotation_matrix(delta_quat, order="wxyz")
    translation = torch.tensor(
        [0.17, -0.11, 0.09], dtype=task.datatype, device=task.device
    ).view(1, 3)

    moved_grasp_pos = translation + torch.matmul(
        delta_rot, grasp_pos.unsqueeze(-1)
    ).squeeze(-1)
    moved_grasp_quat = task._quaternion_multiply_wxyz(delta_quat, grasp_quat)
    moved_target_pos, moved_target_quat = task._compute_desired_reference_pose(
        moved_grasp_pos, moved_grasp_quat
    )

    expected_target_pos = translation + torch.matmul(
        delta_rot, target_pos.unsqueeze(-1)
    ).squeeze(-1)
    expected_target_quat = task._quaternion_multiply_wxyz(delta_quat, target_quat)
    position_delta = torch.max(torch.abs(moved_target_pos - expected_target_pos))
    quat_similarity = torch.abs(
        torch.sum(moved_target_quat * expected_target_quat, dim=-1)
    )
    if float(position_delta.item()) > 1e-5 or bool(
        torch.any(quat_similarity < 1.0 - 1e-5).item()
    ):
        raise AssertionError(
            "Desired reference-link pose does not follow the target grasp frame."
        )
    print(
        "[STAGE2_IDEAL] Target-relative pose invariance verified "
        f"(max position delta={float(position_delta.item()):.2e}, "
        f"min quaternion similarity={float(quat_similarity.min().item()):.8f}).",
        flush=True,
    )


def set_arm_pose(task: ParallelStage2ArmTask, joint_target: torch.Tensor) -> None:
    task.starlink_manipulator.robot.set_dofs_position(
        joint_target,
        dofs_idx_local=task.starlink_manipulator.motors_dof,
        zero_velocity=True,
    )
    task.starlink_manipulator.robot.set_dofs_velocity(
        torch.zeros_like(joint_target),
        dofs_idx_local=task.starlink_manipulator.motors_dof,
    )
    task.starlink_manipulator.control_joint_pos(joint_target)


def lock_target_pose(task: ParallelStage2ArmTask, target_pose) -> None:
    target_pos, target_quat = target_pose
    task.target_object.robot.set_pos(target_pos, zero_velocity=True)
    task.target_object.robot.set_quat(target_quat, zero_velocity=True)


def prepare_success_metrics(task: ParallelStage2ArmTask) -> None:
    task.last_action_tensor.zero_()
    task._update_io_tensors()
    task.target_hold_count.zero_()
    task.prev_in_hold_zone.zero_()
    task.ever_in_hold_zone.zero_()
    task.reached_approach_milestone.zero_()
    task.reached_prepose_milestone.zero_()
    task.reached_midpose_milestone.zero_()
    task.reached_nearpose_milestone.zero_()
    task.reached_precision_milestone.zero_()
    for episode_reward_sum in task.episode_reward_sums.values():
        episode_reward_sum.zero_()
    task.episode_min_tcp_to_grasp_error.fill_(float("inf"))
    task.episode_min_pose_orientation_angle_deg.fill_(float("inf"))
    task.episode_max_hold_count.zero_()
    task.prev_tcp_to_grasp_error = task.tcp_to_grasp_error.detach().clone()
    task.prev_pose_orientation_angle = task.pose_orientation_angle.detach().clone()
    task.prev_tcp_pos = task._get_tcp_pose()[0].detach().clone()
    task.prev_reference_link_pos = task._get_reference_yaw_link_pose()[0].detach().clone()
    task.tcp_speed.zero_()
    task.reference_link_speed.zero_()
    task.tool_speed.zero_()
    task.status = TaskStatus.RUNNING


def advance_decision_step(
    task: ParallelStage2ArmTask,
    joint_target: torch.Tensor,
    target_pose,
) -> tuple[torch.Tensor, torch.Tensor]:
    open_gripper = torch.full(
        (task.num_envs,),
        task.gripper_open_value,
        dtype=task.datatype,
        device=task.device,
    )
    for _ in range(task.arm_control_decimation):
        lock_target_pose(task, target_pose)
        task._set_base_hold_pid_target()
        task.starlink_manipulator.control_joint_pos(joint_target)
        task.starlink_manipulator.control_gripper(open_gripper)
        task.starlink_manipulator.step()
        task.target_object.step()
        task.gsim.step()

    task._update_io_tensors()
    reward = task.reward()
    success = task.is_success()
    task.last_step_reward = reward.detach().clone()
    task.success = success.detach().clone()
    return reward, success


def print_metrics(prefix: str, task: ParallelStage2ArmTask, reward=None) -> None:
    link_pos, link_quat = task._get_reference_yaw_link_pose()
    tcp_pos, _ = task._get_tcp_pose()
    grasp_pos = task.target_grasp_pose[:, :3]
    reference_goal_pos = task.desired_reference_link_pose[:, :3]
    grasp_delta = grasp_pos - tcp_pos
    tcp_to_grasp_error = torch.norm(grasp_delta, dim=-1)
    quat_similarity = task.pose_quat_similarity
    angle_deg = task.pose_orientation_angle_deg
    reward_text = "" if reward is None else f" reward={float(reward[0].item()):.4f}"
    print(
        f"{prefix}{reward_text} success={bool(task.is_success()[0].item())} "
        f"hold={int(task.target_hold_count[0].item())}/{task.success_hold_steps} "
        f"pos_err={float(task.pose_position_error[0].item()):.6f}m "
        f"quat_sim={float(quat_similarity[0].item()):.6f} "
        f"angle_err={float(angle_deg[0].item()):.3f}deg "
        f"view={bool(task.target_in_wrist_view[0].item())} "
        f"tcp_speed={float(task.tcp_speed[0].item()):.6f}m/s "
        f"link_speed={float(task.reference_link_speed[0].item()):.6f}m/s "
        f"tool_speed={float(task.tool_speed[0].item()):.6f}m/s "
        f"speed_limit={float(task.tool_speed_limit[0].item()):.6f}m/s "
        f"base_drift={float(task.base_drift_error[0].item()):.6f}m",
        flush=True,
    )
    print(
        "  reference_link: "
        f"xyz={link_pos[0].detach().cpu().tolist()} "
        f"goal_xyz={reference_goal_pos[0].detach().cpu().tolist()} "
        f"quat_wxyz={link_quat[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        "  tcp_marker: "
        f"xyz={tcp_pos[0].detach().cpu().tolist()} "
        f"cross_length={Stage2IdealViewerTask.tcp_marker_length:.3f}m "
        f"thickness={Stage2IdealViewerTask.tcp_marker_thickness:.3f}m color=green",
        flush=True,
    )
    print(
        "  target_grasp: "
        f"xyz={grasp_pos[0].detach().cpu().tolist()} "
        f"target_minus_tcp_xyz={grasp_delta[0].detach().cpu().tolist()} "
        f"tcp_error={float(tcp_to_grasp_error[0].item()):.6f}m",
        flush=True,
    )


def viewer_is_alive(task: ParallelStage2ArmTask) -> bool:
    viewer = getattr(task.gsim, "viewer", None)
    return viewer is not None and viewer.is_alive()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_viewer and args.viewer_steps <= 0:
        args.viewer_steps = 20

    configure_no_camera_viewer(args)
    configure_collision_free_target()

    task = Stage2IdealViewerTask(
        target_asset_name="starlink",
        num_envs=1,
        center_envs_at_origin=True,
        enable_visual_obs=False,
        enable_target_relative_obs=False,
        target_position_noise=False,
        stage2_base_distance=args.stage2_base_distance,
    )
    try:
        if not task.initialize():
            raise RuntimeError("ParallelStage2ArmTask failed to initialize.")

        verify_target_relative_pose(task)
        target_pose = (
            as_batch(task.target_object.robot.get_pos(), task, 3).detach().clone(),
            as_batch(task.target_object.robot.get_quat(), task, 4).detach().clone(),
        )
        joint_target = solve_reference_joint_target(task)
        set_arm_pose(task, joint_target)
        lock_target_pose(task, target_pose)
        task.starlink_manipulator.update_state()
        prepare_success_metrics(task)
        if tuple(task.state_observation.shape) != (1, 18):
            raise RuntimeError(
                "Stage-2 state must contain 6 joint positions, 6 joint "
                f"velocities, and 6 previous actions; got {task.state_observation.shape}."
            )
        if not torch.allclose(
            task.state_observation[:, 12:18],
            torch.zeros_like(task.state_observation[:, 12:18]),
        ):
            raise RuntimeError("Stage-2 initial previous-action observation is not zero.")

        print("\n[STAGE2_IDEAL] Ideal initialization", flush=True)
        print(f"  arm_joint_target={joint_target[0].detach().cpu().tolist()}", flush=True)
        print(
            f"  target_xyz={target_pose[0][0].detach().cpu().tolist()} "
            f"target_quat_wxyz={target_pose[1][0].detach().cpu().tolist()}",
            flush=True,
        )
        print(
            f"  reference_xyz={task.reference_yaw_link_xyz.detach().cpu().tolist()} "
            f"reference_quat_wxyz={task.reference_yaw_link_quat.detach().cpu().tolist()}",
            flush=True,
        )
        print(
            "  success_thresholds: "
            f"tcp_error<={task.tcp_position_threshold:.3f}m "
            f"orientation_angle<={math.degrees(task.pose_orientation_angle_threshold):.1f}deg "
            "target_in_wrist_view=True "
            f"tool_speed<={task.hold_velocity_threshold:.3f}m/s "
            f"hold_steps={task.success_hold_steps}",
            flush=True,
        )
        print_metrics("[STAGE2_IDEAL][initial]", task)

        success = torch.zeros(1, dtype=torch.bool, device=task.device)
        episode_return = 0.0
        max_verify_steps = task.success_hold_steps + 20
        for step_idx in range(1, max_verify_steps + 1):
            reward, success = advance_decision_step(task, joint_target, target_pose)
            episode_return += float(reward[0].item())
            milestone_rewards = (
                float(task.reward_approach_milestone[0].item()),
                float(task.reward_prepose_milestone[0].item()),
                float(task.reward_midpose_milestone[0].item()),
                float(task.reward_nearpose_milestone[0].item()),
                float(task.reward_precision_milestone[0].item()),
            )
            expected_milestone_rewards = (
                (
                    task.approach_milestone_bonus,
                    task.prepose_milestone_bonus,
                    task.midpose_milestone_bonus,
                    task.nearpose_milestone_bonus,
                    task.precision_milestone_bonus,
                )
                if step_idx == 1
                else (0.0, 0.0, 0.0, 0.0, 0.0)
            )
            if any(
                not math.isclose(actual, expected, abs_tol=1e-5)
                for actual, expected in zip(
                    milestone_rewards, expected_milestone_rewards
                )
            ):
                raise RuntimeError(
                    "Stage-2 milestone rewards were not granted exactly once: "
                    f"step={step_idx}, actual={milestone_rewards}, "
                    f"expected={expected_milestone_rewards}."
                )
            print_metrics(f"[STAGE2_IDEAL][verify step={step_idx}]", task, reward)
            if bool(success[0].item()):
                task.status = TaskStatus.SUCCEED
                break
        if not bool(success[0].item()):
            raise RuntimeError(
                "Ideal pose did not satisfy the real Stage-2 success condition within "
                f"{max_verify_steps} decision steps."
            )
        terminal_info = task.get_info()[0]
        component_return = sum(
            terminal_info[f"episode_{reward_attr}"]
            for reward_attr in task._stage2_reward_attr_names()
        )
        if not math.isclose(component_return, episode_return, abs_tol=1e-3):
            raise RuntimeError(
                "Episode reward-component cache does not match the episode return: "
                f"components={component_return}, return={episode_return}."
            )
        if not math.isclose(
            terminal_info["episode_reward_precision_milestone"],
            task.precision_milestone_bonus,
            abs_tol=1e-5,
        ):
            raise RuntimeError("Precision milestone was not preserved in terminal info.")

        print(
            "\n[STAGE2_IDEAL] SUCCESS verified. Holding the target, base, and arm pose; "
            "close the viewer or press Ctrl+C to exit.",
            flush=True,
        )

        display_step = 0
        while args.viewer_steps <= 0 or display_step < args.viewer_steps:
            if not args.no_viewer and not viewer_is_alive(task):
                break
            display_step += 1
            advance_decision_step(task, joint_target, target_pose)
            if display_step == 1 or display_step % max(1, args.log_every) == 0:
                print_metrics(f"[STAGE2_IDEAL][hold step={display_step}]", task)
            if args.sleep > 0.0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[STAGE2_IDEAL] Interrupted.", flush=True)
    finally:
        task.gsim.stop()


if __name__ == "__main__":
    main()
