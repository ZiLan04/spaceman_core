"""Audit the gripper-closing physics from the ideal Stage-2 success pose.

The satellite manipulator base and the target satellite are placed at the same
relative pose that the trained Stage-2 policy treats as success: the base sits
at the Stage-2 PID hold pose and the arm is solved (IK) onto the task's fixed
reference-link pose. No approach motion happens, so the only thing that can
move the target is the gripper closing.

The test then closes the gripper (and nothing else) and reports whether the
closing physics looks reasonable:

  * do both fingers touch the target and with what force,
  * does the gripper actually stop short of its free-closed position (blocked)
    or does it close all the way through empty space,
  * does the target stay put, or get squeezed/knocked away,
  * optionally, does the target follow a small pull (grasp retention).

No wrist/base/recording cameras are created, so it runs headless with
``--no-viewer``.
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import genesis as gs  # noqa: E402

from configs import asset_configs  # noqa: E402
from configs.registry import make_starlink_morph  # noqa: E402
from configs.registry import get_legacy_camera  # noqa: E402
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
from configs.configs import APP_SETTINGS, SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from tasks.RL_task_parallel_stage2 import ParallelStage2ArmTask  # noqa: E402
from utils.utils import as_rotation_matrix  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_SRC / "recordings" / "stage2_gripper_close_test"

# Stage-2 ideal-pose IK solution at the default 1.1 m base distance.  This is
# useful for moving only the free-floating base while keeping the arm geometry
# exactly unchanged.
DEFAULT_FIXED_ARM_JOINT_TARGET = (
    0.9435901045799255,
    0.12462592869997025,
    -1.0988271236419678,
    3.0923664569854736,
    -2.251356363296509,
    -2.751229763031006,
)

FINGER_LINK_NAMES = (
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
)


def parse_vec3(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return tuple(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Place the manipulator and target at the ideal Stage-2 success pose, "
            "close only the gripper, and audit the closing physics."
        )
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run headless.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage2-base-distance", type=float, default=1.1)
    parser.add_argument(
        "--solve-arm-ik",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Solve arm IK after placing the base (default). Use --no-solve-arm-ik "
            "to keep the fixed arm joint angles calibrated at the default 1.1 m pose."
        ),
    )
    parser.add_argument(
        "--arm-joint-5",
        type=float,
        default=None,
        metavar="RAD",
        help=(
            "Override arm[5] after IK/fixed-reference initialization, in radians. "
            "Other arm joints are unchanged."
        ),
    )
    parser.add_argument(
        "--viewer-pos", type=parse_vec3, default=(-1.6, -3.2, 2.0)
    )
    parser.add_argument(
        "--viewer-lookat", type=parse_vec3, default=(-1.6, 0.0, 0.65)
    )
    parser.add_argument("--viewer-fov", type=float, default=45.0)
    parser.add_argument(
        "--show-collision-models",
        action="store_true",
        help=(
            "Render the exact Genesis collision geometry for both the manipulator "
            "and target instead of their visual meshes. Viewer only."
        ),
    )
    parser.add_argument(
        "--show-contact-forces",
        action="store_true",
        help=(
            "Draw diagnostic contact-force arrows in the viewer. The arrows are "
            "visual aids only and do not participate in collision."
        ),
    )
    parser.add_argument(
        "--hide-target-model",
        action="store_true",
        help=(
            "Hide the target satellite in the viewer while keeping its collision "
            "and dynamics active. Useful for inspecting the manipulator collision model."
        ),
    )
    parser.add_argument(
        "--hide-manipulator-model",
        action="store_true",
        help=(
            "Hide the satellite manipulator in the viewer while keeping its collision "
            "and dynamics active. Useful for inspecting the target collision model."
        ),
    )
    parser.add_argument(
        "--target-collision",
        dest="target_collision",
        action="store_true",
        default=None,
        help="Enable physical collision on the target (needed for a real grasp test).",
    )
    parser.add_argument(
        "--no-target-collision",
        dest="target_collision",
        action="store_false",
        help="Disable physical collision on the target (fingers close through it).",
    )
    parser.add_argument(
        "--lock-target",
        dest="lock_target",
        action="store_true",
        default=None,
        help="Pin the target at its initial pose (isolates finger/geometry checks).",
    )
    parser.add_argument(
        "--no-lock-target",
        dest="lock_target",
        action="store_false",
        help="Leave the target free so contact can push it (default).",
    )
    parser.add_argument(
        "--verify-success-steps",
        type=int,
        default=12,
        help=(
            "Decision steps to hold the ideal pose (gripper open) and confirm the "
            "Stage-2 success condition before closing. Also measures the baseline "
            "target drift caused by resting at the pose with collision on."
        ),
    )
    parser.add_argument(
        "--close-value", type=float, default=0.0, help="Gripper close command (0=close)."
    )
    parser.add_argument(
        "--close-steps",
        type=int,
        default=20,
        help="Decision steps used to ramp the gripper from open to the close command.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=120,
        help="Decision steps to keep the gripper closed and watch the target.",
    )
    parser.add_argument(
        "--pull-distance",
        type=float,
        default=0.0,
        help="Extra: pull the base away by this many meters to test retention (0=off).",
    )
    parser.add_argument("--pull-steps", type=int, default=60)
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument(
        "--contact-force-threshold",
        type=float,
        default=1e-3,
        help="Finger contact force above this counts as a real contact.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.02,
        help="Target grasp-point drift (m) above this is reported as knocked/squeezed away.",
    )
    parser.add_argument(
        "--blocked-closure-threshold",
        type=float,
        default=0.9,
        help=(
            "If the closed finger only reaches this fraction of its free-closed "
            "position, it is considered blocked by the target."
        ),
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-record-csv", action="store_true", help="Do not write the per-step CSV."
    )
    parser.set_defaults(target_collision=None, lock_target=None)
    return parser


def configure_runtime(args) -> None:
    APP_SETTINGS["seed"] = int(args.seed)
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not bool(args.no_viewer)
    try:
        SCENE_SETTINGS["viewer_options"] = gs.options.ViewerOptions(
            camera_pos=tuple(args.viewer_pos),
            camera_lookat=tuple(args.viewer_lookat),
            camera_fov=float(args.viewer_fov),
        )
    except Exception:
        pass
    TPV_CAM_SETTINGS["enable_recording"] = False

    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA.get("rl_obs", {})["enabled"] = False
    FRANKA_S_Q_CAMERA.get("rl_obs", {})["enable_recording"] = False
    FRANKA_S_Q_CAMERA.get("base_camera", {})["enabled"] = False
    FRANKA_S_Q_CAMERA.get("base_camera", {})["enable_recording"] = False


def configure_asset_visualization(
    enable_collision: bool,
    show_collision_models: bool,
    show_contact_forces: bool,
    hide_target_model: bool,
    hide_manipulator_model: bool,
) -> None:
    def target_factory():
        return {
            "morph": make_starlink_morph(
                collision=bool(enable_collision),
                visualization=not hide_target_model,
            ),
            "material": gs.materials.Rigid(gravity_compensation=1.0),
            # Rendering an empty visual set keeps the physical collision entity
            # active while hiding it from the viewer.
            "vis_mode": (
                "visual"
                if hide_target_model
                else ("collision" if show_collision_models else "visual")
            ),
            "visualize_contact": show_contact_forces,
        }

    manipulator_factory = asset_configs._ASSET_FACTORIES["franka_merge"]

    def debug_manipulator_factory():
        asset = dict(manipulator_factory())
        if hide_manipulator_model:
            asset["morph"].visualization = False
        asset["vis_mode"] = (
            "visual"
            if hide_manipulator_model
            else ("collision" if show_collision_models else "visual")
        )
        asset["visualize_contact"] = show_contact_forces
        return asset

    asset_configs._ASSET_FACTORIES["starlink"] = target_factory
    asset_configs._ASSET_FACTORIES["franka_merge"] = debug_manipulator_factory
    print(
        f"[GRIPPER_CLOSE_TEST] target collision enabled: {bool(enable_collision)}",
        flush=True,
    )
    print(
        "[GRIPPER_CLOSE_TEST] viewer geometry: "
        + ("COLLISION" if show_collision_models else "VISUAL"),
        flush=True,
    )
    print(
        f"[GRIPPER_CLOSE_TEST] contact-force arrows: {show_contact_forces}",
        flush=True,
    )
    print(
        f"[GRIPPER_CLOSE_TEST] target model hidden: {hide_target_model}",
        flush=True,
    )
    print(
        f"[GRIPPER_CLOSE_TEST] manipulator model hidden: {hide_manipulator_model}",
        flush=True,
    )


def as_batch(value, task, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=task.datatype, device=task.device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (task.num_envs, width):
        raise ValueError(
            f"Expected shape ({task.num_envs}, {width}), got {tuple(tensor.shape)}"
        )
    return tensor


def get_target_root_pose(task):
    return (
        as_batch(task.target_object.robot.get_pos(), task, 3).detach().clone(),
        as_batch(task.target_object.robot.get_quat(), task, 4).detach().clone(),
    )


def lock_target_pose(task, target_pose) -> None:
    if target_pose is None:
        return
    task.target_object.robot.set_pos(target_pose[0], zero_velocity=True)
    task.target_object.robot.set_quat(target_pose[1], zero_velocity=True)


def solve_reference_joint_target(task) -> torch.Tensor:
    """IK for the fixed reference-link pose that defines Stage-2 success."""
    grasp_pos, grasp_quat = task._get_target_grasp_pose()
    target_pos, target_quat = task._compute_desired_reference_pose(grasp_pos, grasp_quat)
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


def fixed_reference_joint_target(task) -> torch.Tensor:
    """Return the default 1.1 m ideal-pose arm angles without running IK."""
    return torch.tensor(
        DEFAULT_FIXED_ARM_JOINT_TARGET,
        dtype=task.datatype,
        device=task.device,
    ).view(1, -1).repeat(task.num_envs, 1)


def set_arm_pose(task, joint_target: torch.Tensor) -> None:
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


def print_joint_values(task, joint_target: torch.Tensor) -> None:
    """Print configured arm and finger joints with unambiguous names."""
    manip = task.starlink_manipulator
    joint_names = list(manip.joints_name)
    q_indices = list(manip.motors_qs) + list(manip.fingers_qs)
    actual = torch.as_tensor(
        manip.robot.get_qpos(qs_idx_local=q_indices),
        dtype=task.datatype,
        device=task.device,
    )
    if actual.dim() == 1:
        actual = actual.unsqueeze(0)

    motor_count = len(manip.motors_qs)
    print("  joint_values_rad:", flush=True)
    for index, name in enumerate(joint_names):
        actual_value = float(actual[0, index].item())
        if index < motor_count:
            target_value = float(joint_target[0, index].item())
            print(
                f"    arm[{index}] target={target_value:+.8f} "
                f"actual={actual_value:+.8f} name={name}",
                flush=True,
            )
        else:
            finger_index = index - motor_count
            print(
                f"    finger[{finger_index}] actual={actual_value:+.8f} name={name}",
                flush=True,
            )


def prepare_metrics(task) -> None:
    task.last_action_tensor.zero_()
    task._update_io_tensors()
    task.target_hold_count.zero_()
    task.prev_in_hold_zone.zero_()
    task.ever_in_hold_zone.zero_()
    for attr in (
        "reached_approach_milestone",
        "reached_prepose_milestone",
        "reached_midpose_milestone",
        "reached_nearpose_milestone",
        "reached_precision_milestone",
    ):
        getattr(task, attr).zero_()
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


def _to_numpy(value):
    if value is None:
        return None
    try:
        return np.asarray(torch.as_tensor(value).detach().cpu(), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError):
        return None


def finger_metrics(task):
    """Return (per-finger qpos, mean, closure in [0,1], free-closed reference)."""
    manip = task.starlink_manipulator
    positions = np.full(2, np.nan)
    try:
        qpos = _to_numpy(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
        if qpos is not None:
            qpos = qpos.reshape(-1)
            for index in range(min(2, qpos.size)):
                positions[index] = float(qpos[index])
    except Exception:
        pass

    valid = ~np.isnan(positions)
    mean = float(np.mean(positions[valid])) if np.any(valid) else float("nan")

    closed_reference = float("nan")
    try:
        limits = manip.robot.get_dofs_limit(dofs_idx_local=manip.fingers_dof)
        upper = _to_numpy(limits[1])
        if upper is not None and upper.size > 0:
            closed_reference = float(np.max(np.abs(upper)))
    except Exception:
        pass
    if np.isnan(closed_reference) or closed_reference <= 1e-9:
        try:
            closed_reference = float(_to_numpy(manip.finger_close).reshape(-1)[0])
        except Exception:
            closed_reference = float("nan")

    if np.isnan(mean) or np.isnan(closed_reference) or closed_reference <= 1e-9:
        closure = float("nan")
    else:
        closure = float(np.clip(mean / closed_reference, 0.0, 1.0))
    return positions, mean, closure, closed_reference


def contact_metrics(task) -> dict:
    """Contacts between the manipulator and target, isolating the finger links."""
    result = {
        "finger_contacts": 0,
        "both_fingers": False,
        "total_contacts": 0,
        "finger_force": 0.0,
        "contact_links": "",
        "contact_pairs": "",
    }
    manip = task.starlink_manipulator
    try:
        contacts = manip.robot.get_contacts(with_entity=task.target_object.robot)
    except Exception:
        return result
    if not contacts:
        return result

    link_a = _to_numpy(contacts.get("link_a"))
    if link_a is None or link_a.size == 0:
        return result
    link_a = link_a.reshape(-1)
    link_b = _to_numpy(contacts.get("link_b"))
    link_b = link_b.reshape(-1) if link_b is not None else None

    force = contacts.get("force_a")
    if force is None:
        force = contacts.get("force_b")
    force = _to_numpy(force)
    if force is not None:
        force = force.reshape(-1, 3)

    try:
        manip_link_names = {
            int(link.idx): getattr(link, "name", f"link_{int(link.idx)}")
            for link in manip.robot.links
        }
        target_link_names = {
            int(link.idx): getattr(link, "name", f"link_{int(link.idx)}")
            for link in task.target_object.robot.links
        }
        link_names = {**manip_link_names, **target_link_names}
    except Exception:
        link_names = {}

    finger_indices = set()
    for name in FINGER_LINK_NAMES:
        try:
            finger_indices.add(int(manip.robot.get_link(name).idx))
        except Exception:
            pass

    contacted = set()
    partners = []
    contact_pairs = set()
    total_force = 0.0
    for index in range(link_a.size):
        a_index = int(link_a[index])
        b_index = int(link_b[index]) if link_b is not None and link_b.size > index else -1
        contact_pairs.add(
            f"{link_names.get(a_index, str(a_index))}<->{link_names.get(b_index, str(b_index))}"
        )
        a_finger = a_index in finger_indices
        b_finger = b_index in finger_indices
        if not (a_finger or b_finger):
            continue
        result["finger_contacts"] += 1
        if a_finger:
            contacted.add(a_index)
            partners.append(link_names.get(b_index, str(b_index)))
        if b_finger:
            contacted.add(b_index)
            partners.append(link_names.get(a_index, str(a_index)))
        if force is not None and force.shape[0] > index:
            total_force += float(np.linalg.norm(force[index]))

    result["both_fingers"] = len(contacted) >= 2
    result["finger_force"] = total_force
    result["total_contacts"] = int(link_a.size)
    result["contact_links"] = ",".join(sorted(set(partners)))
    result["contact_pairs"] = ";".join(sorted(contact_pairs))
    return result


def step_decision(
    task,
    joint_target: torch.Tensor,
    gripper_cmd: torch.Tensor,
    target_pose=None,
) -> None:
    """One decision step holding the arm, driving the gripper, locking base/target."""
    manip = task.starlink_manipulator
    for _ in range(max(1, int(task.arm_control_decimation))):
        lock_target_pose(task, target_pose)
        task._set_base_hold_pid_target()
        manip.control_joint_pos(joint_target)
        manip.control_gripper(gripper_cmd)
        manip.step()
        task.target_object.step()
        task.gsim.step()
    lock_target_pose(task, target_pose)
    task._update_io_tensors()


def make_gripper_cmd(task, value: float) -> torch.Tensor:
    return torch.full(
        (task.num_envs,), float(value), dtype=task.datatype, device=task.device
    )


def print_metrics(prefix: str, task, reward=None) -> None:
    tcp_pos, _ = task._get_tcp_pose()
    grasp_pos = task.target_grasp_pose[:, :3]
    tcp_error = float(torch.norm(tcp_pos - grasp_pos, dim=-1)[0].item())
    positions, mean, closure, closed_reference = finger_metrics(task)
    contact = contact_metrics(task)
    reward_text = "" if reward is None else f" reward={float(reward[0].item()):.3f}"
    print(
        f"{prefix}{reward_text} "
        f"tcp_to_grasp={tcp_error:.5f}m "
        f"pose_err={float(task.pose_position_error[0].item()):.5f}m "
        f"angle_err={float(task.pose_orientation_angle_deg[0].item()):.3f}deg "
        f"finger_mean={mean:.5f} closure={closure:.3f} "
        f"(free_closed={closed_reference:.5f}) "
        f"finger_contacts={contact['finger_contacts']}/{contact['total_contacts']} "
        f"both={contact['both_fingers']} force={contact['finger_force']:.5f}",
        flush=True,
    )
    if contact["contact_pairs"]:
        print(f"{prefix}contact_pairs={contact['contact_pairs']}", flush=True)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def viewer_is_alive(task) -> bool:
    viewer = getattr(task.gsim, "viewer", None)
    return viewer is not None and viewer.is_alive()


def main() -> None:
    args = build_arg_parser().parse_args()
    target_collision = True if args.target_collision is None else bool(args.target_collision)
    lock_target = False if args.lock_target is None else bool(args.lock_target)
    log_every = max(1, int(args.log_every))

    configure_runtime(args)
    configure_asset_visualization(
        target_collision,
        bool(args.show_collision_models),
        bool(args.show_contact_forces),
        bool(args.hide_target_model),
        bool(args.hide_manipulator_model),
    )

    task = ParallelStage2ArmTask(
        target_asset_name="starlink",
        num_envs=1,
        center_envs_at_origin=True,
        enable_visual_obs=False,
        enable_target_relative_obs=False,
        target_position_noise=False,
        stage2_base_distance=float(args.stage2_base_distance),
    )
    try:
        if not task.initialize():
            raise RuntimeError("ParallelStage2ArmTask failed to initialize.")

        if args.solve_arm_ik:
            joint_target = solve_reference_joint_target(task)
            joint_target_source = "IK solved for current base distance"
        else:
            joint_target = fixed_reference_joint_target(task)
            joint_target_source = "fixed default-distance reference (IK disabled)"
        if args.arm_joint_5 is not None:
            joint_target = joint_target.detach().clone()
            joint_target[:, 5] = float(args.arm_joint_5)
            joint_target_source += f"; arm[5] overridden to {float(args.arm_joint_5):+.8f} rad"
        set_arm_pose(task, joint_target)
        target_pose = get_target_root_pose(task) if lock_target else None
        lock_target_pose(task, target_pose)
        task.starlink_manipulator.update_state()
        prepare_metrics(task)

        open_cmd = make_gripper_cmd(task, task.gripper_open_value)
        close_value = float(np.clip(args.close_value, 0.0, 1.0))
        close_cmd = make_gripper_cmd(task, close_value)

        _, _, initial_closure, free_closed = finger_metrics(task)
        print("\n[GRIPPER_CLOSE_TEST] Ideal Stage-2 pose initialized", flush=True)
        print(f"  arm_joint_target_source={joint_target_source}", flush=True)
        print(f"  arm_joint_target={joint_target[0].detach().cpu().tolist()}", flush=True)
        print_joint_values(task, joint_target)
        print(
            f"  free-space closed finger reference (joint limit) = "
            f"{free_closed:.5f} rad (open closure={initial_closure:.3f})",
            flush=True,
        )
        print(
            "  success thresholds: "
            f"tcp<={task.tcp_position_threshold:.3f}m "
            f"angle<={math.degrees(task.pose_orientation_angle_threshold):.1f}deg "
            f"hold_steps={task.success_hold_steps}",
            flush=True,
        )
        print_metrics("[GRIPPER_CLOSE_TEST][initial] ", task)

        rows = []
        step_index = 0

        def record(stage: str, gripper_command: float, reward=None):
            nonlocal step_index
            step_index += 1
            tcp_pos, _ = task._get_tcp_pose()
            grasp_pos = task.target_grasp_pose[:, :3]
            positions, mean, closure, _ = finger_metrics(task)
            contact = contact_metrics(task)
            rows.append(
                {
                    "step": int(step_index),
                    "stage": stage,
                    "gripper_cmd": float(gripper_command),
                    "finger_qpos_1": float(positions[0]),
                    "finger_qpos_2": float(positions[1]),
                    "finger_mean": float(mean),
                    "closure": float(closure),
                    "finger_contacts": int(contact["finger_contacts"]),
                    "total_contacts": int(contact["total_contacts"]),
                    "both_fingers": bool(contact["both_fingers"]),
                    "finger_force": float(contact["finger_force"]),
                    "contact_links": contact["contact_links"],
                    "contact_pairs": contact["contact_pairs"],
                    "grasp_x": float(grasp_pos[0, 0].item()),
                    "grasp_y": float(grasp_pos[0, 1].item()),
                    "grasp_z": float(grasp_pos[0, 2].item()),
                    "tcp_to_grasp_error": float(
                        torch.norm(tcp_pos - grasp_pos, dim=-1)[0].item()
                    ),
                }
            )

        def snapshot():
            return task.target_grasp_pose[:, :3].detach().clone()

        # Phase 0: hold the ideal pose with the gripper open and collision on.
        # This measures the baseline drift (e.g. geometry overlap pushing the target).
        settle_steps = max(0, int(args.verify_success_steps))
        baseline_start = snapshot()
        success = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        for local_step in range(settle_steps):
            step_decision(task, joint_target, open_cmd, target_pose)
            reward = task.reward()
            success = task.is_success()
            record("settle", float(open_cmd[0].item()), reward)
            if local_step == 0 or (local_step + 1) % log_every == 0 or local_step == settle_steps - 1:
                print_metrics(f"[GRIPPER_CLOSE_TEST][settle {local_step + 1:03d}] ", task, reward)
        baseline_drift = float(
            torch.norm(snapshot() - baseline_start, dim=-1)[0].item()
        )
        tcp_pos, _ = task._get_tcp_pose()
        geometric_success = bool(
            (
                task.tcp_to_grasp_error <= task.tcp_position_threshold
            )[0].item()
            and (
                task.pose_orientation_angle <= task.pose_orientation_angle_threshold
            )[0].item()
            and bool(task.target_in_wrist_view[0].item())
        )
        print(
            f"[GRIPPER_CLOSE_TEST] At the ideal pose before closing: "
            f"geometric_success={geometric_success}, "
            f"full_success={bool(success[0].item())}, "
            f"tcp_to_grasp={float(torch.norm(tcp_pos - snapshot(), dim=-1)[0].item()):.5f}m, "
            f"angle_err={float(task.pose_orientation_angle_deg[0].item()):.3f}deg | "
            f"baseline target drift over {settle_steps} open-gripper steps = "
            f"{baseline_drift:.5f} m",
            flush=True,
        )

        # Phase 1: close the gripper (and nothing else).
        close_steps = max(1, int(args.close_steps))
        close_start = snapshot()
        for local_step in range(close_steps):
            fraction = (local_step + 1) / float(close_steps)
            command = make_gripper_cmd(
                task, float(task.gripper_open_value) + (close_value - float(task.gripper_open_value)) * fraction
            )
            step_decision(task, joint_target, command, target_pose)
            reward = task.reward()
            close_cmd = command
            record("close", float(command[0].item()), reward)
            if local_step == 0 or (local_step + 1) % log_every == 0 or local_step == close_steps - 1:
                print_metrics(f"[GRIPPER_CLOSE_TEST][close {local_step + 1:03d}] ", task, reward)
        close_cmd = make_gripper_cmd(task, close_value)
        close_drift = float(torch.norm(snapshot() - close_start, dim=-1)[0].item())

        # Phase 2: keep the gripper closed.
        hold_steps = max(0, int(args.hold_steps))
        hold_start = snapshot()
        for local_step in range(hold_steps):
            step_decision(task, joint_target, close_cmd, target_pose)
            reward = task.reward()
            record("hold", float(close_cmd[0].item()), reward)
            if local_step == 0 or (local_step + 1) % log_every == 0 or local_step == hold_steps - 1:
                print_metrics(f"[GRIPPER_CLOSE_TEST][hold {local_step + 1:03d}] ", task, reward)
        hold_drift = float(torch.norm(snapshot() - hold_start, dim=-1)[0].item())
        _, final_mean, final_closure, _ = finger_metrics(task)
        final_contact = contact_metrics(task)

        # Phase 3 (optional): pull the base away to test retention.
        pull_distance = max(0.0, float(args.pull_distance))
        pull_steps = max(1, int(args.pull_steps))
        pull_follow = 0.0
        if pull_distance > 0.0:
            base_hold_ref = task.stage2_base_hold_pos.detach().clone()
            direction = (base_hold_ref[:, :2] - close_start[:, :2]).detach().clone()
            direction = direction / torch.clamp(
                torch.norm(direction, dim=-1, keepdim=True), min=1e-6
            )
            pull_start = snapshot()
            for local_step in range(pull_steps):
                fraction = min(1.0, (local_step + 1) / float(pull_steps))
                task.stage2_base_hold_pos[:, :2] = (
                    base_hold_ref[:, :2] + direction * pull_distance * fraction
                )
                step_decision(task, joint_target, close_cmd, target_pose)
                reward = task.reward()
                record("pull", float(close_cmd[0].item()), reward)
                if local_step == 0 or (local_step + 1) % log_every == 0 or local_step == pull_steps - 1:
                    print_metrics(f"[GRIPPER_CLOSE_TEST][pull {local_step + 1:03d}] ", task, reward)
            pull_follow = float(
                torch.sum(
                    (snapshot() - pull_start)[:, :2] * direction, dim=-1
                )[0].item()
            )
            for local_step in range(max(0, int(args.settle_steps))):
                step_decision(task, joint_target, close_cmd, target_pose)
                task.reward()
                record("settle_after_pull", float(close_cmd[0].item()))

        # ---- Verdict ------------------------------------------------------
        max_contact_force = max((row["finger_force"] for row in rows), default=0.0)
        max_total_contacts = max((row["total_contacts"] for row in rows), default=0)
        ever_contact = any(row["finger_contacts"] > 0 for row in rows)
        ever_both = any(row["both_fingers"] for row in rows)
        closing_drift = close_drift + hold_drift
        blocked = (
            not math.isnan(final_closure)
            and final_closure < float(args.blocked_closure_threshold)
        )
        knocked_away = closing_drift >= float(args.drift_threshold)

        if not ever_contact:
            verdict = "NO_CONTACT"
        elif knocked_away:
            verdict = "TARGET_KNOCKED_AWAY"
        elif not blocked:
            verdict = "CLOSED_THROUGH"
        elif float(args.pull_distance) > 0.0 and pull_follow < 0.5 * float(args.pull_distance):
            verdict = "GRASP_SLIPPED"
        else:
            verdict = "CLOSE_OK"

        print("\n[GRIPPER_CLOSE_TEST] RESULT", flush=True)
        print(f"  verdict                : {verdict}", flush=True)
        print(f"  finger_contact         : {ever_contact} (both={ever_both})", flush=True)
        print(f"  max_contact_force      : {max_contact_force:.5f}", flush=True)
        print(f"  max_total_contacts     : {max_total_contacts}", flush=True)
        print(
            f"  finger closed qpos     : {final_mean:.5f} "
            f"(free-closed ref {free_closed:.5f}, closure {final_closure:.3f}, "
            f"blocked={blocked})",
            flush=True,
        )
        print(
            f"  target drift           : baseline(open)={baseline_drift:.5f} m, "
            f"close={close_drift:.5f} m, hold={hold_drift:.5f} m, "
            f"close+hold={closing_drift:.5f} m (threshold {float(args.drift_threshold):.5f})",
            flush=True,
        )
        if float(args.pull_distance) > 0.0:
            print(
                f"  pull retention         : follow={pull_follow:.5f} / "
                f"{float(args.pull_distance):.5f} m",
                flush=True,
            )
        print(
            "  interpretation         : "
            + {
                "NO_CONTACT": "fingers never touched the target (check collision/pose).",
                "TARGET_KNOCKED_AWAY": "closing the gripper pushed the target away.",
                "CLOSED_THROUGH": "fingers touched but fully closed; target was not held.",
                "GRASP_SLIPPED": "contact made but the target did not follow the pull.",
                "CLOSE_OK": "contact + blocked close + no knock-away: closing looks physical.",
            }[verdict],
            flush=True,
        )

        if not args.no_record_csv and rows:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = args.output_dir / f"gripper_close_{timestamp}.csv"
            write_csv(csv_path, rows)
            print(f"  csv                    : {csv_path}", flush=True)

        if not args.no_viewer and viewer_is_alive(task):
            print(
                "\n[GRIPPER_CLOSE_TEST] Viewer is open; close it to exit.",
                flush=True,
            )
            while viewer_is_alive(task):
                time.sleep(0.1)
    finally:
        task.stop()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
