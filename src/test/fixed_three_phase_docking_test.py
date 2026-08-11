"""Fixed three-phase docking/grasp smoke test.

Phases:
1. Hold the arm still and move the satellite-manipulator base with PID until
   the base root is 1.3 m from the target grasp point.
2. Hold the base with PID, keep the gripper open, and move the TCP to the
   target grasp point with a downward approach orientation.
3. Close the gripper, keep the base with PID, and pull the target a short
   distance toward the manipulator base.
"""

import argparse
import sys
from pathlib import Path

import genesis as gs
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

import configs.asset_configs as asset_configs  # noqa: E402
from configs.config_starlink import _STARLINK_PATHS, to_posix  # noqa: E402
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from tasks.RL_task_parallel import ParallelRLTask, TaskStatus  # noqa: E402
from utils.utils import as_rotation_matrix  # noqa: E402

PHASE2_REFERENCE_YAW_JOINT_NAME = (
    "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link_yaw"
)
PHASE2_REFERENCE_LINK_NAME = (
    "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link"
)


def as_list(tensor: torch.Tensor, digits: int = 5):
    tensor = torch.as_tensor(tensor).detach().cpu().reshape(-1)
    return [round(float(value), digits) for value in tensor.tolist()]


def parse_vec3(text: str):
    if text is None:
        return None
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return values


def parse_vec2(text: str):
    if text is None:
        return None
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values.")
    return values


def axis_rot_mask(local_axis):
    axis_tensor = torch.as_tensor(local_axis, dtype=torch.float32).abs()
    axis_index = int(torch.argmax(axis_tensor).item())
    return [axis_index == 0, axis_index == 1, axis_index == 2]


def configure_cameras(
    record_scene_video: bool,
    show_viewer: bool = True,
    show_wrist_camera: bool = False,
    record_wrist_camera: bool = False,
    wrist_camera_res=None,
    wrist_camera_fov: float = 70.0,
):
    if not record_scene_video and not show_wrist_camera and not record_wrist_camera:
        SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = bool(show_viewer)
    TPV_CAM_SETTINGS["enable_recording"] = bool(record_scene_video)

    FRANKA_S_Q_CAMERA["wrist_camera"] = bool(show_wrist_camera or record_wrist_camera)
    FRANKA_S_Q_CAMERA["enable_recording"] = bool(record_wrist_camera)
    wrist_camera_config = FRANKA_S_Q_CAMERA.get("camera", {})
    wrist_camera_config["GUI"] = bool(show_wrist_camera)
    if wrist_camera_res is not None:
        wrist_camera_config["res"] = tuple(int(value) for value in wrist_camera_res)
    wrist_camera_config["fov"] = float(wrist_camera_fov)

    rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
    rl_obs_config["enabled"] = False
    rl_obs_config["enable_recording"] = False

    base_camera_config = FRANKA_S_Q_CAMERA.get("base_camera", {})
    base_camera_config["enabled"] = False
    base_camera_config["enable_recording"] = False
    base_rl_obs_config = base_camera_config.get("rl_obs", {})
    base_rl_obs_config["enabled"] = False
    base_rl_obs_config["enable_recording"] = False


def configure_target_collision(enable_collision: bool):
    def make_starlink_for_fixed_test():
        return {
            "morph": gs.morphs.URDF(
                file=to_posix(_STARLINK_PATHS["urdf"]),
                pos=(-2.0, 0.0, 0.5),
                euler=(0.0, 0.0, 180.0),
                scale=7e-1,
                fixed=False,
                collision=bool(enable_collision),
                visualization=True,
            ),
            "material": gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
        }

    asset_configs._ASSET_FACTORIES["starlink"] = make_starlink_for_fixed_test


def normalize(vector: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vector / torch.clamp(torch.norm(vector, dim=-1, keepdim=True), min=eps)


def normalize_xy(vector: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vector[..., :2] / torch.clamp(
        torch.norm(vector[..., :2], dim=-1, keepdim=True),
        min=eps,
    )


def rotation_matrix_from_vectors(local_axis: torch.Tensor, world_axis: torch.Tensor) -> torch.Tensor:
    """Return R such that R @ local_axis ~= world_axis."""

    local_axis = normalize(local_axis)
    world_axis = normalize(world_axis)
    v = torch.cross(local_axis, world_axis, dim=-1)
    c = torch.sum(local_axis * world_axis, dim=-1)
    s = torch.norm(v, dim=-1)

    batch_shape = world_axis.shape[:-1]
    eye = torch.eye(3, dtype=world_axis.dtype, device=world_axis.device).expand(
        *batch_shape, 3, 3
    )
    vx = torch.zeros((*batch_shape, 3, 3), dtype=world_axis.dtype, device=world_axis.device)
    vx[..., 0, 1] = -v[..., 2]
    vx[..., 0, 2] = v[..., 1]
    vx[..., 1, 0] = v[..., 2]
    vx[..., 1, 2] = -v[..., 0]
    vx[..., 2, 0] = -v[..., 1]
    vx[..., 2, 1] = v[..., 0]

    factor = ((1.0 - c) / torch.clamp(s * s, min=1e-6)).reshape(*batch_shape, 1, 1)
    rot = eye + vx + torch.matmul(vx, vx) * factor

    parallel = s <= 1e-6
    same = parallel & (c > 0.0)
    opposite = parallel & (c <= 0.0)
    if torch.any(same):
        rot = torch.where(same.reshape(*batch_shape, 1, 1), eye, rot)
    if torch.any(opposite):
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=world_axis.dtype, device=world_axis.device)
        fallback = fallback.expand_as(local_axis)
        too_close = torch.abs(torch.sum(fallback * local_axis, dim=-1)) > 0.9
        alt = torch.tensor([0.0, 1.0, 0.0], dtype=world_axis.dtype, device=world_axis.device)
        fallback = torch.where(too_close.unsqueeze(-1), alt.expand_as(fallback), fallback)
        ortho = normalize(torch.cross(local_axis, fallback, dim=-1))
        kx = torch.zeros_like(vx)
        kx[..., 0, 1] = -ortho[..., 2]
        kx[..., 0, 2] = ortho[..., 1]
        kx[..., 1, 0] = ortho[..., 2]
        kx[..., 1, 2] = -ortho[..., 0]
        kx[..., 2, 0] = -ortho[..., 1]
        kx[..., 2, 1] = ortho[..., 0]
        rot_pi = eye + 2.0 * torch.matmul(kx, kx)
        rot = torch.where(opposite.reshape(*batch_shape, 1, 1), rot_pi, rot)
    return rot


def set_base_root_pid_target(
    task: ParallelRLTask,
    root_target_pos: torch.Tensor,
    root_target_quat: torch.Tensor,
):
    if not hasattr(task.starlink_manipulator, "pid"):
        raise RuntimeError("This test requires a base PID backend.")
    if hasattr(task.starlink_manipulator, "set_pid_enabled"):
        task.starlink_manipulator.set_pid_enabled(True)
    setpoint = task._root_target_to_pid_link_setpoint(root_target_pos, root_target_quat)
    task.starlink_manipulator.pid.update_setpoint(setpoint)
    return setpoint


def reset_base_pid_to_current_pose(task: ParallelRLTask):
    root_pos, root_quat = task._get_base_pose()
    if hasattr(task.starlink_manipulator, "pid"):
        task.starlink_manipulator.pid.reset()
    set_base_root_pid_target(task, root_pos.detach().clone(), root_quat.detach().clone())
    return root_pos.detach().clone(), root_quat.detach().clone()


def build_stage2_base_hold_pose(
    task: ParallelRLTask,
    grasp_point_pos: torch.Tensor,
    grasp_point_quat: torch.Tensor,
    base_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_rot = as_rotation_matrix(grasp_point_quat, order="wxyz")
    dock_dir_local = torch.tensor(
        [-1.0, 0.0, 0.0], dtype=task.datatype, device=task.device
    )
    dock_dir_world = torch.matmul(target_rot, dock_dir_local.view(3, 1)).squeeze(-1)
    dock_dir_xy = normalize_xy(dock_dir_world)
    current_root_pos, current_root_quat = task._get_base_pose()
    hold_root_pos = current_root_pos.detach().clone()
    hold_root_pos[:, :2] = grasp_point_pos[:, :2] + dock_dir_xy * float(base_distance)
    return hold_root_pos, current_root_quat.detach().clone()


def set_base_root_pose(
    task: ParallelRLTask,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
):
    root_qpos = torch.cat([root_pos, root_quat], dim=-1)
    task.starlink_manipulator.robot.set_qpos(
        root_qpos,
        qs_idx_local=list(range(7)),
        zero_velocity=True,
    )
    task.starlink_manipulator.update_state()
    task._update_io_tensors()
    return reset_base_pid_to_current_pose(task)


def build_phase1_base_target(
    current_root_pos: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    start_root_xy: torch.Tensor,
    desired_root_pos: torch.Tensor,
    *,
    lookahead: float,
    brake_distance: float,
    brake_setpoint_backoff: float,
    cruise_speed: float,
    brake_accel: float,
    speed_deadband: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    approach_delta = desired_root_pos[:, :2] - start_root_xy
    approach_distance = torch.clamp(
        torch.norm(approach_delta, dim=-1, keepdim=True),
        min=1e-6,
    )
    approach_dir = approach_delta / approach_distance

    current_delta = current_root_pos[:, :2] - start_root_xy
    current_progress = torch.sum(current_delta * approach_dir, dim=-1, keepdim=True)
    total_progress = approach_distance
    remaining = total_progress - current_progress

    approach_speed = torch.sum(current_velocity_xy * approach_dir, dim=-1, keepdim=True)
    brake_speed_limit = torch.sqrt(
        torch.clamp(2.0 * float(brake_accel) * torch.clamp(remaining, min=0.0), min=0.0)
    )
    speed_limit = torch.where(
        remaining <= float(brake_distance),
        brake_speed_limit,
        torch.full_like(brake_speed_limit, float(cruise_speed)),
    )
    speed_limit = torch.clamp(speed_limit, max=float(cruise_speed))

    far_target_progress = current_progress + torch.clamp(
        remaining,
        min=0.0,
        max=float(lookahead),
    )
    brake_target_progress = current_progress - torch.clamp(
        approach_speed,
        min=0.0,
    ) * float(brake_setpoint_backoff)
    should_brake = approach_speed > speed_limit + float(speed_deadband)
    target_progress = torch.where(
        should_brake,
        brake_target_progress,
        far_target_progress,
    )
    target_progress = torch.clamp(target_progress, min=0.0)
    target_progress = torch.minimum(target_progress, total_progress)

    target_root_pos = current_root_pos.detach().clone()
    target_root_pos[:, :2] = start_root_xy + approach_dir * target_progress
    return (
        target_root_pos,
        remaining.squeeze(-1),
        approach_speed.squeeze(-1),
        speed_limit.squeeze(-1),
    )


def get_target_grasp_pose(task: ParallelRLTask):
    grasp_pos, grasp_quat = task._get_target_grasp_pose()
    return grasp_pos.detach().clone(), grasp_quat.detach().clone()


def get_tcp_pose(task: ParallelRLTask):
    tcp_pos, tcp_quat = task._get_tcp_pose()
    return tcp_pos.detach().clone(), tcp_quat.detach().clone()


def get_link_pose_tensor(entity, link_name: str, task: ParallelRLTask):
    link = entity.robot.get_link(link_name)
    pos = torch.as_tensor(link.get_pos(), dtype=task.datatype, device=task.device)
    quat = torch.as_tensor(link.get_quat(), dtype=task.datatype, device=task.device)
    if pos.dim() == 1:
        pos = pos.reshape(1, 3)
    if quat.dim() == 1:
        quat = quat.reshape(1, 4)
    return pos.detach().clone(), quat.detach().clone()


def local_point_to_world(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    local_point: torch.Tensor,
) -> torch.Tensor:
    local_point = local_point.reshape(1, 3, 1).expand(link_pos.shape[0], 3, 1)
    link_rot = as_rotation_matrix(link_quat, order="wxyz")
    return link_pos + torch.matmul(link_rot, local_point).squeeze(-1)


def get_cylinder_center_world(
    task: ParallelRLTask,
    *,
    entity,
    link_name: str,
    offset_xy,
    scale: float,
) -> torch.Tensor:
    link_pos, link_quat = get_link_pose_tensor(entity, link_name, task)
    local_offset = torch.tensor(
        [float(offset_xy[0]) * float(scale), float(offset_xy[1]) * float(scale), 0.0],
        dtype=task.datatype,
        device=task.device,
    )
    return local_point_to_world(link_pos, link_quat, local_offset).detach().clone()


def tcp_target_to_ee_target(
    task: ParallelRLTask,
    tcp_target_pos: torch.Tensor,
    ee_target_quat: torch.Tensor,
) -> torch.Tensor:
    ee_rot = as_rotation_matrix(ee_target_quat, order="wxyz")
    tcp_offset = task.tcp_offset_local.reshape(1, 3, 1).expand(task.num_envs, 3, 1)
    return tcp_target_pos - torch.matmul(ee_rot, tcp_offset).squeeze(-1)


def solve_arm_joint_target(
    task: ParallelRLTask,
    tcp_target_pos: torch.Tensor,
    ee_target_quat: torch.Tensor = None,
    rot_mask=None,
) -> torch.Tensor:
    init_qpos = task.starlink_manipulator.robot.get_qpos()
    init_qpos = torch.as_tensor(init_qpos, dtype=task.datatype, device=task.device)
    qpos = task.starlink_manipulator.robot.inverse_kinematics(
        link=task.starlink_manipulator.end_effector,
        pos=tcp_target_pos,
        quat=ee_target_quat,
        local_point=task.tcp_offset_local,
        init_qpos=init_qpos,
        dofs_idx_local=task.starlink_manipulator.motors_dof,
        max_samples=8,
        max_solver_iters=35,
        damping=0.03,
        max_step_size=0.25,
        rot_mask=rot_mask if rot_mask is not None else [True, True, True],
    )
    qpos = torch.as_tensor(qpos, dtype=task.datatype, device=task.device)
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0)
    return qpos[:, task.starlink_manipulator.motors_qs]


def limit_joint_step(
    current_joint: torch.Tensor,
    target_joint: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    delta = target_joint - current_joint
    delta_norm = torch.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_delta) / torch.clamp(delta_norm, min=1e-6), max=1.0)
    return current_joint + delta * scale


def move_tcp_toward_target(
    task: ParallelRLTask,
    tcp_target_pos: torch.Tensor,
    ee_target_quat: torch.Tensor,
    current_joint: torch.Tensor,
    *,
    tcp_step_size: float,
    max_joint_delta: float,
    orient_distance: float,
    rot_mask=None,
) -> torch.Tensor:
    tcp_pos, _ = get_tcp_pose(task)
    delta = tcp_target_pos - tcp_pos
    distance = torch.norm(delta, dim=-1, keepdim=True)
    step_scale = torch.clamp(float(tcp_step_size) / torch.clamp(distance, min=1e-6), max=1.0)
    next_tcp_target = tcp_pos + delta * step_scale
    use_orientation = bool(torch.all(distance <= float(orient_distance)).item())
    raw_joint_target = solve_arm_joint_target(
        task,
        next_tcp_target,
        ee_target_quat=ee_target_quat if use_orientation else None,
        rot_mask=rot_mask if use_orientation else None,
    )
    return limit_joint_step(current_joint, raw_joint_target, max_joint_delta)


def move_joints_toward_tcp_ik(
    task: ParallelRLTask,
    tcp_target_pos: torch.Tensor,
    ee_target_quat: torch.Tensor,
    current_joint: torch.Tensor,
    *,
    max_joint_delta: float,
    rot_mask=None,
) -> torch.Tensor:
    raw_joint_target = solve_arm_joint_target(
        task,
        tcp_target_pos,
        ee_target_quat=ee_target_quat,
        rot_mask=rot_mask,
    )
    return limit_joint_step(current_joint, raw_joint_target, max_joint_delta)


def clamp_joint_to_reference_window(
    joint_target: torch.Tensor,
    reference_joint: torch.Tensor,
    joint_index: int,
    max_offset: float,
) -> torch.Tensor:
    joint_target = joint_target.detach().clone()
    joint_index = int(joint_index)
    if joint_index < 0:
        joint_index += joint_target.shape[-1]
    if joint_index < 0 or joint_index >= joint_target.shape[-1]:
        raise IndexError(
            f"Joint index {joint_index} is outside target shape {tuple(joint_target.shape)}."
        )
    reference_value = reference_joint[:, joint_index]
    joint_target[:, joint_index] = torch.clamp(
        joint_target[:, joint_index],
        min=reference_value - float(max_offset),
        max=reference_value + float(max_offset),
    )
    return joint_target


def make_downward_ee_quat(
    task: ParallelRLTask,
    local_down_axis,
) -> torch.Tensor:
    local_approach_axis = torch.tensor(
        local_down_axis,
        dtype=task.datatype,
        device=task.device,
    ).reshape(1, 3).expand(task.num_envs, 3)
    world_down_axis = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=task.datatype,
        device=task.device,
    ).reshape(1, 3).expand(task.num_envs, 3)
    ee_rot = rotation_matrix_from_vectors(local_approach_axis, world_down_axis)
    return task._rotation_matrix_to_quaternion_wxyz(ee_rot)


def compute_down_axis_alignment(
    ee_quat: torch.Tensor,
    local_down_axis,
    task: ParallelRLTask,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_axis = torch.tensor(
        local_down_axis,
        dtype=task.datatype,
        device=task.device,
    ).reshape(1, 3, 1).expand(task.num_envs, 3, 1)
    ee_rot = as_rotation_matrix(ee_quat, order="wxyz")
    world_axis = torch.matmul(ee_rot, local_axis).squeeze(-1)
    world_down = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=task.datatype,
        device=task.device,
    ).reshape(1, 3).expand(task.num_envs, 3)
    alignment = torch.sum(normalize(world_axis) * world_down, dim=-1)
    return world_axis.detach().clone(), alignment.detach().clone()


def compute_ee_axis_down_alignments(
    ee_quat: torch.Tensor,
    task: ParallelRLTask,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ee_rot = as_rotation_matrix(ee_quat, order="wxyz")
    local_axes = torch.eye(3, dtype=task.datatype, device=task.device)
    local_axes = local_axes.reshape(1, 3, 3).expand(task.num_envs, 3, 3)
    world_axes = torch.matmul(ee_rot, local_axes)
    world_down = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=task.datatype,
        device=task.device,
    ).reshape(1, 3, 1).expand(task.num_envs, 3, 1)
    down_alignments = torch.sum(normalize(world_axes) * world_down, dim=1)
    return (
        world_axes[:, :, 0].detach().clone(),
        world_axes[:, :, 1].detach().clone(),
        world_axes[:, :, 2].detach().clone(),
        down_alignments[:, 0].detach().clone(),
        down_alignments[:, 1].detach().clone(),
        down_alignments[:, 2].detach().clone(),
    )


def get_gripper_qpos(task: ParallelRLTask) -> torch.Tensor:
    qpos = task.starlink_manipulator.robot.get_qpos(
        qs_idx_local=task.starlink_manipulator.fingers_qs
    )
    qpos = torch.as_tensor(qpos, dtype=task.datatype, device=task.device)
    if qpos.dim() == 1:
        qpos = qpos.reshape(1, -1)
    return qpos.detach().clone()


def get_gripper_open_qpos(task: ParallelRLTask) -> torch.Tensor:
    open_qpos = task.starlink_manipulator.finger_open
    open_qpos = torch.as_tensor(open_qpos, dtype=task.datatype, device=task.device)
    return open_qpos.reshape(1, -1).expand(task.num_envs, -1).detach().clone()


def get_arm_joint_qpos(task: ParallelRLTask) -> torch.Tensor:
    joint_qpos = task.starlink_manipulator.robot.get_qpos(
        qs_idx_local=task.starlink_manipulator.motors_qs
    )
    joint_qpos = torch.as_tensor(joint_qpos, dtype=task.datatype, device=task.device)
    if joint_qpos.dim() == 1:
        joint_qpos = joint_qpos.reshape(1, -1)
    return joint_qpos.detach().clone()


def get_arm_joint_qvel(task: ParallelRLTask) -> torch.Tensor:
    joint_qvel = task.starlink_manipulator.robot.get_dofs_velocity(
        dofs_idx_local=task.starlink_manipulator.motors_dof
    )
    joint_qvel = torch.as_tensor(joint_qvel, dtype=task.datatype, device=task.device)
    if joint_qvel.dim() == 1:
        joint_qvel = joint_qvel.reshape(1, -1)
    return joint_qvel.detach().clone()


def get_motor_joint_names(task: ParallelRLTask):
    joint_names = list(getattr(task.starlink_manipulator, "joints_name", []))
    return joint_names[: len(task.starlink_manipulator.motors_qs)]


def hold_gripper_qpos(task: ParallelRLTask, qpos: torch.Tensor):
    task.starlink_manipulator.robot.control_dofs_position(
        position=qpos,
        dofs_idx_local=task.starlink_manipulator.fingers_dof,
    )


def build_link_name_map(entity) -> dict[int, str]:
    return {
        int(link.idx): getattr(link, "name", f"link_{int(link.idx)}")
        for link in entity.links
    }


def lock_target_pose(task: ParallelRLTask, locked_target_pose):
    if locked_target_pose is None:
        return
    target_pos, target_quat = locked_target_pose
    task.target_object.robot.set_pos(target_pos, zero_velocity=True)
    task.target_object.robot.set_quat(target_quat, zero_velocity=True)


def get_target_root_pose(task: ParallelRLTask):
    target_pos = torch.as_tensor(
        task.target_object.robot.get_pos(),
        dtype=task.datatype,
        device=task.device,
    )
    target_quat = torch.as_tensor(
        task.target_object.robot.get_quat(),
        dtype=task.datatype,
        device=task.device,
    )
    if target_pos.dim() == 1:
        target_pos = target_pos.reshape(1, 3)
    if target_quat.dim() == 1:
        target_quat = target_quat.reshape(1, 4)
    return target_pos.detach().clone(), target_quat.detach().clone()


def log_base_initialization_debug(
    task: ParallelRLTask,
    label: str,
    env_idx: int,
    *,
    planned_root_pos: torch.Tensor = None,
    planned_root_quat: torch.Tensor = None,
    reference_grasp_pos: torch.Tensor = None,
):
    root_pos, root_quat = task._get_base_pose()
    pid_link_pos, pid_link_quat = task._get_pid_link_pose()
    target_root_pos, target_root_quat = get_target_root_pose(task)
    grasp_pos, grasp_quat = get_target_grasp_pose(task)

    root_to_pid = pid_link_pos - root_pos
    root_to_grasp_xy = torch.norm(root_pos[:, :2] - grasp_pos[:, :2], dim=-1)
    root_to_grasp_xyz = torch.norm(root_pos - grasp_pos, dim=-1)
    pid_to_grasp_xy = torch.norm(pid_link_pos[:, :2] - grasp_pos[:, :2], dim=-1)
    planned_delta = None
    if planned_root_pos is not None:
        planned_delta = root_pos - planned_root_pos

    print(f"\n[THREE_PHASE_TEST][init_debug:{label}]")
    print(f"  env_idx: {env_idx}")
    print(f"  root_xyz: {as_list(root_pos[env_idx], digits=8)}")
    print(f"  root_quat: {as_list(root_quat[env_idx], digits=8)}")
    print(f"  pid_link_xyz: {as_list(pid_link_pos[env_idx], digits=8)}")
    print(f"  pid_link_quat: {as_list(pid_link_quat[env_idx], digits=8)}")
    print(f"  root_to_pid_xyz: {as_list(root_to_pid[env_idx], digits=8)}")
    print(f"  target_root_xyz: {as_list(target_root_pos[env_idx], digits=8)}")
    print(f"  target_root_quat: {as_list(target_root_quat[env_idx], digits=8)}")
    print(f"  grasp_xyz: {as_list(grasp_pos[env_idx], digits=8)}")
    print(f"  grasp_quat: {as_list(grasp_quat[env_idx], digits=8)}")
    print(f"  root_to_grasp_xy: {float(root_to_grasp_xy[env_idx].item()):.8f}")
    print(f"  root_to_grasp_xyz: {float(root_to_grasp_xyz[env_idx].item()):.8f}")
    print(f"  pid_link_to_grasp_xy: {float(pid_to_grasp_xy[env_idx].item()):.8f}")
    if reference_grasp_pos is not None:
        grasp_shift = grasp_pos - reference_grasp_pos
        print(f"  grasp_shift_from_setup_xyz: {as_list(grasp_shift[env_idx], digits=8)}")
    if planned_root_pos is not None:
        print(f"  planned_root_xyz: {as_list(planned_root_pos[env_idx], digits=8)}")
        print(f"  root_minus_planned_xyz: {as_list(planned_delta[env_idx], digits=8)}")
    if planned_root_quat is not None:
        print(f"  planned_root_quat: {as_list(planned_root_quat[env_idx], digits=8)}")
    if hasattr(task.starlink_manipulator, "pid"):
        pid_setpoint = task.starlink_manipulator.pid.setpoint.detach().clone()
        if pid_setpoint.dim() == 1:
            print(f"  pid_setpoint: {as_list(pid_setpoint, digits=8)}")
        else:
            print(f"  pid_setpoint: {as_list(pid_setpoint[env_idx], digits=8)}")
    print("", flush=True)


def build_attached_target_pose(
    task: ParallelRLTask,
    attach_target_pos: torch.Tensor,
    attach_target_quat: torch.Tensor,
    attach_tcp_pos: torch.Tensor,
    *,
    target_z_offset: float,
    follow_z: bool,
):
    tcp_pos, _ = get_tcp_pose(task)
    tcp_delta = tcp_pos - attach_tcp_pos
    if not follow_z:
        tcp_delta = tcp_delta.detach().clone()
        tcp_delta[:, 2] = 0.0
    target_pos = attach_target_pos + tcp_delta
    target_pos[:, 2] += float(target_z_offset)
    return target_pos, attach_target_quat


def get_contact_debug_rows(task: ParallelRLTask, max_rows: int = 6):
    contacts = task.starlink_manipulator.robot.get_contacts(
        with_entity=task.target_object.robot
    )
    if contacts is None:
        return []

    link_a = contacts.get("link_a")
    link_b = contacts.get("link_b")
    position = contacts.get("position")
    force_a = contacts.get("force_a")
    force_b = contacts.get("force_b")
    if link_a is None or link_b is None or position is None:
        return []

    link_a = torch.as_tensor(link_a).reshape(-1)
    link_b = torch.as_tensor(link_b).reshape(-1)
    position = torch.as_tensor(position, dtype=task.datatype, device=task.device).reshape(-1, 3)
    if force_a is not None:
        force_a = torch.as_tensor(force_a, dtype=task.datatype, device=task.device).reshape(-1, 3)
    if force_b is not None:
        force_b = torch.as_tensor(force_b, dtype=task.datatype, device=task.device).reshape(-1, 3)

    name_map = {}
    name_map.update(build_link_name_map(task.starlink_manipulator.robot))
    name_map.update(build_link_name_map(task.target_object.robot))

    rows = []
    row_count = min(int(link_a.numel()), int(link_b.numel()), int(position.shape[0]), int(max_rows))
    for row_idx in range(row_count):
        link_a_idx = int(link_a[row_idx].item())
        link_b_idx = int(link_b[row_idx].item())
        force = None
        if force_a is not None and row_idx < force_a.shape[0]:
            force = force_a[row_idx]
        elif force_b is not None and row_idx < force_b.shape[0]:
            force = force_b[row_idx]
        force_norm = float(torch.norm(force).item()) if force is not None else 0.0
        rows.append(
            {
                "link_a": name_map.get(link_a_idx, str(link_a_idx)),
                "link_b": name_map.get(link_b_idx, str(link_b_idx)),
                "position": as_list(position[row_idx]),
                "force_norm": round(force_norm, 6),
            }
        )
    return rows


def step_scene(
    task: ParallelRLTask,
    *,
    joint_target: torch.Tensor,
    gripper_value,
    gripper_qpos_target: torch.Tensor = None,
    locked_target_pose=None,
    steps: int,
):
    for _ in range(max(1, int(steps))):
        lock_target_pose(task, locked_target_pose)
        task.starlink_manipulator.control_joint_pos(joint_target)
        if gripper_qpos_target is not None:
            hold_gripper_qpos(task, gripper_qpos_target)
        if gripper_value is not None:
            task.starlink_manipulator.control_gripper(gripper_value)
        task.starlink_manipulator.step()
        task.target_object.step()
        task.gsim.step()
        lock_target_pose(task, locked_target_pose)
    task.starlink_manipulator.update_state()
    task._update_io_tensors()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a fixed three-phase docking test.")
    parser.add_argument("--target-asset", default="starlink")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument(
        "--target-position-noise",
        action="store_true",
        help="Enable reset-time target position noise. Disabled by default for this fixed test.",
    )
    parser.add_argument(
        "--free-target",
        action="store_true",
        help="Let the target satellite move under contact. By default this fixed test locks it in place.",
    )
    parser.add_argument(
        "--enable-target-collision",
        action="store_true",
        help="Enable physical target collision. Disabled by default because this test uses a virtual grasp.",
    )
    parser.add_argument("--base-distance", type=float, default=1.1)
    parser.add_argument("--base-distance-tol", type=float, default=0.05)
    parser.add_argument("--base-speed-tol", type=float, default=0.025)
    parser.add_argument("--base-hold-steps", type=int, default=20)
    parser.add_argument("--base-target-lookahead", type=float, default=0.22)
    parser.add_argument("--base-brake-distance", type=float, default=0.6)
    parser.add_argument("--base-brake-setpoint-backoff", type=float, default=0.9)
    parser.add_argument("--base-cruise-speed", type=float, default=0.32)
    parser.add_argument("--base-brake-accel", type=float, default=0.1)
    parser.add_argument("--base-speed-deadband", type=float, default=0.015)
    parser.add_argument("--tcp-tol", type=float, default=0.055)
    parser.add_argument(
        "--virtual-grasp-height",
        type=float,
        default=0.08,
        help="Stop phase2 this far above the physical grasp point and treat it as grasped.",
    )
    parser.add_argument("--pregrasp-height", type=float, default=0.25)
    parser.add_argument("--pregrasp-tol", type=float, default=0.08)
    parser.add_argument("--grasp-clearance", type=float, default=0.06)
    parser.add_argument("--ee-down-axis-local", type=parse_vec3, default=None)
    parser.add_argument("--orient-distance", type=float, default=10.0)
    parser.add_argument("--tcp-step-size", type=float, default=0.14)
    parser.add_argument("--max-joint-delta", type=float, default=0.22)
    parser.add_argument(
        "--phase2-free-end-yaw",
        action="store_true",
        help="Allow the phase2 end yaw joint to move without a reference window.",
    )
    parser.add_argument("--phase2-end-yaw-joint-index", type=int, default=5)
    parser.add_argument(
        "--phase2-end-yaw-max-offset",
        type=float,
        default=0.9,
        help="Keep phase2 end yaw within this many radians of its phase2-start value to avoid flipping.",
    )
    parser.add_argument("--pull-distance", type=float, default=0.18)
    parser.add_argument("--cylinder-offset-xy", type=parse_vec2, default=[-0.57, 0.12])
    parser.add_argument("--base-cylinder-scale", type=float, default=1.0)
    parser.add_argument("--target-cylinder-scale", type=float, default=0.7)
    parser.add_argument("--cylinder-align-tol", type=float, default=0.04)
    parser.add_argument("--cylinder-align-hold-steps", type=int, default=15)
    parser.add_argument(
        "--target-dock-z-offset",
        type=float,
        default=0.05,
        help="Raise the virtually attached TCP/target together by this much before phase3 xy docking.",
    )
    parser.add_argument("--phase3-lift-max-steps", type=int, default=250)
    parser.add_argument("--phase3-lift-tol", type=float, default=0.01)
    parser.add_argument(
        "--attached-follow-z",
        action="store_true",
        help="Make the virtually grasped target follow TCP z as well. Default only follows xy.",
    )
    parser.add_argument(
        "--phase3-lock-gripper-orientation",
        action="store_true",
        help="Keep the full downward gripper orientation during phase3. Disabled by default to prioritize xy docking.",
    )
    parser.add_argument("--phase1-max-steps", type=int, default=3500)
    parser.add_argument("--phase2-max-steps", type=int, default=2500)
    parser.add_argument("--phase3-steps", type=int, default=900)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--contact-debug-every", type=int, default=20)
    parser.add_argument("--contact-debug-shift", type=float, default=0.005)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the main Genesis viewer window.",
    )
    parser.add_argument(
        "--start-at-stage2",
        action="store_true",
        help="Skip phase1 and initialize the base at the same ideal hold pose used by Stage-2 RL.",
    )
    parser.add_argument(
        "--show-wrist-camera",
        action="store_true",
        help="Open the regular wrist-camera GUI view during the fixed flow.",
    )
    parser.add_argument(
        "--record-wrist-camera",
        action="store_true",
        help="Record the regular/debug wrist camera to src/recordings without opening its GUI unless --show-wrist-camera is also set.",
    )
    parser.add_argument(
        "--wrist-camera-res",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="Resolution for the debug wrist-camera GUI.",
    )
    parser.add_argument(
        "--wrist-camera-fov",
        type=float,
        default=70.0,
        help="Field of view for the debug wrist-camera GUI.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    record_wrist_camera = bool(args.record_wrist_camera)
    configure_cameras(
        args.record_video,
        show_viewer=not args.no_viewer,
        show_wrist_camera=args.show_wrist_camera,
        record_wrist_camera=record_wrist_camera,
        wrist_camera_res=args.wrist_camera_res,
        wrist_camera_fov=args.wrist_camera_fov,
    )
    configure_target_collision(args.enable_target_collision)

    if args.show_wrist_camera or record_wrist_camera:
        wrist_camera_config = FRANKA_S_Q_CAMERA.get("camera", {})
        rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
        print(
            "[THREE_PHASE_TEST] wrist camera config:",
            f"viewer={not args.no_viewer}",
            f"regular_enabled={FRANKA_S_Q_CAMERA.get('wrist_camera')}",
            f"regular_gui={wrist_camera_config.get('GUI')}",
            f"regular_recording={FRANKA_S_Q_CAMERA.get('enable_recording')}",
            f"rl_enabled={rl_obs_config.get('enabled')}",
            f"rl_recording={rl_obs_config.get('enable_recording')}",
            f"res={wrist_camera_config.get('res')}",
            f"fov={wrist_camera_config.get('fov')}",
            f"end_effector_link={FRANKA_S_Q_CAMERA.get('end_effector_link')}",
            f"pos_offset={as_list(FRANKA_S_Q_CAMERA.get('pos_offset'), digits=6)}",
            f"lookat_offset={as_list(FRANKA_S_Q_CAMERA.get('lookat_offset'), digits=6)}",
            f"up_offset={as_list(FRANKA_S_Q_CAMERA.get('up_offset'), digits=6)}",
            flush=True,
        )

    task = ParallelRLTask(
        target_asset_name=args.target_asset,
        num_envs=args.num_envs,
        enable_visual_obs=False,
        target_position_noise=args.target_position_noise,
    )
    if not task.initialize():
        raise RuntimeError("ParallelRLTask failed to initialize.")
    if not task.reset():
        raise RuntimeError("ParallelRLTask failed to reset.")
    task.status = TaskStatus.RUNNING
    if args.ee_down_axis_local is None:
        args.ee_down_axis_local = as_list(normalize(task.tcp_offset_local), digits=6)
    locked_target_pose = None
    if not args.free_target:
        locked_target_pose = (
            torch.as_tensor(
                task.target_object.robot.get_pos(),
                dtype=task.datatype,
                device=task.device,
            ).detach().clone(),
            torch.as_tensor(
                task.target_object.robot.get_quat(),
                dtype=task.datatype,
                device=task.device,
            ).detach().clone(),
        )
        lock_target_pose(task, locked_target_pose)

    env_idx = min(max(int(args.env_idx), 0), task.num_envs - 1)
    log_every = max(1, int(args.log_every))
    open_cmd = torch.full(
        (task.num_envs,),
        task.gripper_open_value,
        dtype=task.datatype,
        device=task.device,
    )
    close_cmd = torch.full(
        (task.num_envs,),
        task.gripper_close_value,
        dtype=task.datatype,
        device=task.device,
    )
    gripper_open_qpos = get_gripper_open_qpos(task)

    initial_joint = task.arm_joint_positions.detach().clone()
    root_pos0, root_quat0 = task._get_base_pose()
    grasp_pos0, grasp_quat0 = get_target_grasp_pose(task)
    approach_dir_xy = normalize_xy(root_pos0 - grasp_pos0)
    desired_base_xy = grasp_pos0[:, :2] + approach_dir_xy * float(args.base_distance)
    desired_root_pos = root_pos0.detach().clone()
    desired_root_pos[:, :2] = desired_base_xy
    stage2_hold_root_pos, stage2_hold_root_quat = build_stage2_base_hold_pose(
        task,
        grasp_pos0,
        grasp_quat0,
        args.base_distance,
    )

    print("\n[THREE_PHASE_TEST] setup")
    print(f"  env_idx: {env_idx}")
    print(f"  start_at_stage2: {args.start_at_stage2}")
    print(f"  target_link_name: {task.target_link_name}")
    print(f"  target_position_noise: {args.target_position_noise}")
    print(f"  target_locked: {not args.free_target}")
    print(f"  target_collision: {args.enable_target_collision}")
    print(f"  start_base_xyz: {as_list(root_pos0[env_idx])}")
    print(f"  start_grasp_xyz: {as_list(grasp_pos0[env_idx])}")
    print(f"  desired_base_xyz: {as_list(desired_root_pos[env_idx])}")
    print(f"  stage2_rl_hold_base_xyz: {as_list(stage2_hold_root_pos[env_idx])}")
    print(f"  base_distance_goal: {args.base_distance}")
    print(
        "  base_motion:",
        f"cruise_speed={args.base_cruise_speed}",
        f"lookahead={args.base_target_lookahead}",
        f"brake_distance={args.base_brake_distance}",
        f"brake_backoff={args.base_brake_setpoint_backoff}",
        f"brake_accel={args.base_brake_accel}",
        f"speed_deadband={args.base_speed_deadband}",
    )
    print(f"  ee_down_axis_local: {args.ee_down_axis_local}")
    print(f"  orient_distance: {args.orient_distance}")
    print(f"  tcp_step_size: {args.tcp_step_size}")
    print(f"  max_joint_delta: {args.max_joint_delta}")
    print(f"  ee_down_rot_mask: {axis_rot_mask(args.ee_down_axis_local)}")
    print(f"  phase2_constrain_end_yaw: {not args.phase2_free_end_yaw}")
    print(f"  phase2_end_yaw_joint_index: {args.phase2_end_yaw_joint_index}")
    print(f"  phase2_end_yaw_max_offset: {args.phase2_end_yaw_max_offset}")
    print(f"  grasp_clearance: {args.grasp_clearance}")
    print(f"  virtual_grasp_height: {args.virtual_grasp_height}")
    print(f"  target_dock_z_offset: {args.target_dock_z_offset}")
    print(f"  phase3_lift_max_steps: {args.phase3_lift_max_steps}")
    print(f"  phase3_lift_tol: {args.phase3_lift_tol}")
    print(f"  phase3_lock_gripper_orientation: {args.phase3_lock_gripper_orientation}")
    print(
        "  cylinder_alignment:",
        f"offset_xy={args.cylinder_offset_xy}",
        f"base_scale={args.base_cylinder_scale}",
        f"target_scale={args.target_cylinder_scale}",
        f"tol={args.cylinder_align_tol}",
    )
    print(f"  gripper_open_qpos: {as_list(gripper_open_qpos[env_idx])}")
    log_base_initialization_debug(
        task,
        "after_reset_before_phase_selection",
        env_idx,
        planned_root_pos=stage2_hold_root_pos,
        planned_root_quat=stage2_hold_root_quat,
        reference_grasp_pos=grasp_pos0,
    )

    phase1_done = False
    base_hold_count = 0
    sim_dt = float(getattr(task, "control_dt", 3e-3))
    task.starlink_manipulator.control_gripper(open_cmd)

    if args.start_at_stage2:
        print("\n[THREE_PHASE_TEST] phase1: skipped, initializing at Stage-2 RL base hold pose")
        log_base_initialization_debug(
            task,
            "stage2_before_set_qpos",
            env_idx,
            planned_root_pos=stage2_hold_root_pos,
            planned_root_quat=stage2_hold_root_quat,
            reference_grasp_pos=grasp_pos0,
        )
        hold_root_pos, hold_root_quat = set_base_root_pose(
            task,
            stage2_hold_root_pos,
            stage2_hold_root_quat,
        )
        log_base_initialization_debug(
            task,
            "stage2_after_set_qpos",
            env_idx,
            planned_root_pos=stage2_hold_root_pos,
            planned_root_quat=stage2_hold_root_quat,
            reference_grasp_pos=grasp_pos0,
        )
        step_scene(
            task,
            joint_target=initial_joint,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=args.settle_steps,
        )
        hold_root_pos, hold_root_quat = reset_base_pid_to_current_pose(task)
        log_base_initialization_debug(
            task,
            "stage2_after_settle_and_pid_reset",
            env_idx,
            planned_root_pos=stage2_hold_root_pos,
            planned_root_quat=stage2_hold_root_quat,
            reference_grasp_pos=grasp_pos0,
        )
        phase1_done = True
        print(
            "[THREE_PHASE_TEST][stage2_start]",
            f"base_xyz={as_list(hold_root_pos[env_idx])}",
            f"base_quat={as_list(hold_root_quat[env_idx])}",
            f"arm_joint_qpos={as_list(get_arm_joint_qpos(task)[env_idx])}",
            flush=True,
        )
    else:
        print("\n[THREE_PHASE_TEST] phase1: base PID approach, arm fixed, gripper open")
        start_root_xy = root_pos0[:, :2].detach().clone()
        prev_base_xy = root_pos0[:, :2].detach().clone()
        base_velocity_xy = torch.zeros_like(prev_base_xy)
        for step_idx in range(1, max(1, int(args.phase1_max_steps)) + 1):
            current_root_pos, _ = task._get_base_pose()
            phase1_target_pos, remaining_to_goal, approach_speed, speed_limit = build_phase1_base_target(
                current_root_pos,
                base_velocity_xy,
                start_root_xy,
                desired_root_pos,
                lookahead=args.base_target_lookahead,
                brake_distance=args.base_brake_distance,
                brake_setpoint_backoff=args.base_brake_setpoint_backoff,
                cruise_speed=args.base_cruise_speed,
                brake_accel=args.base_brake_accel,
                speed_deadband=args.base_speed_deadband,
            )
            set_base_root_pid_target(task, phase1_target_pos, root_quat0)
            step_scene(
                task,
                joint_target=initial_joint,
                gripper_value=open_cmd,
                gripper_qpos_target=gripper_open_qpos,
                locked_target_pose=locked_target_pose,
                steps=1,
            )
            root_pos, root_quat = task._get_base_pose()
            grasp_pos, _ = get_target_grasp_pose(task)
            distance = torch.norm(root_pos[:, :2] - grasp_pos[:, :2], dim=-1)
            error = torch.abs(distance - float(args.base_distance))
            base_delta_xy = root_pos[:, :2] - prev_base_xy
            base_velocity_xy = base_delta_xy / max(sim_dt, 1e-6)
            base_speed = torch.norm(base_velocity_xy, dim=-1)
            prev_base_xy = root_pos[:, :2].detach().clone()
            stable_mask = (error <= args.base_distance_tol) & (base_speed <= args.base_speed_tol)
            if bool(torch.all(stable_mask).item()):
                base_hold_count += 1
            else:
                base_hold_count = 0

            if step_idx % log_every == 0 or base_hold_count > 0:
                print(
                    "[THREE_PHASE_TEST][phase1]",
                    f"step={step_idx}",
                    f"base_xy={as_list(root_pos[env_idx, :2])}",
                    f"distance={float(distance[env_idx].item()):.5f}",
                    f"error={float(error[env_idx].item()):.5f}",
                    f"base_speed={float(base_speed[env_idx].item()):.5f}",
                    f"approach_speed={float(approach_speed[env_idx].item()):.5f}",
                    f"speed_limit={float(speed_limit[env_idx].item()):.5f}",
                    f"remaining={float(remaining_to_goal[env_idx].item()):.5f}",
                    f"target_xy={as_list(phase1_target_pos[env_idx, :2])}",
                    f"hold={base_hold_count}/{args.base_hold_steps}",
                    flush=True,
                )
            if base_hold_count >= max(1, int(args.base_hold_steps)):
                phase1_done = True
                break

        if not phase1_done:
            task.stop()
            raise RuntimeError(
                f"Phase1 did not reach a stopped {args.base_distance:.3f} m base distance target."
            )

        hold_root_pos, hold_root_quat = reset_base_pid_to_current_pose(task)
        step_scene(
            task,
            joint_target=initial_joint,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=args.settle_steps,
        )
        hold_root_pos, hold_root_quat = reset_base_pid_to_current_pose(task)

    print("\n[THREE_PHASE_TEST] phase2: base hold, pregrasp then TCP down to grasp point, gripper open")
    ee_down_quat = make_downward_ee_quat(task, args.ee_down_axis_local)
    ee_down_rot_mask = axis_rot_mask(args.ee_down_axis_local)
    physical_grasp_pos, _ = get_target_grasp_pose(task)
    phase2_grasp_pos = physical_grasp_pos.detach().clone()
    phase2_grasp_pos[:, 2] += float(args.virtual_grasp_height)
    phase2_pregrasp_pos = phase2_grasp_pos.detach().clone()
    phase2_pregrasp_pos[:, 2] += float(args.pregrasp_height)
    phase2_done = False
    phase2_subphase = "pregrasp"
    phase2_joint_reference = get_arm_joint_qpos(task)
    phase2_end_yaw_joint_index = int(args.phase2_end_yaw_joint_index)
    phase2_end_yaw_joint_name = get_motor_joint_names(task)[phase2_end_yaw_joint_index]
    for step_idx in range(1, max(1, int(args.phase2_max_steps)) + 1):
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        grasp_pos, _ = get_target_grasp_pose(task)
        pregrasp_pos = phase2_pregrasp_pos
        tcp_target_pos = pregrasp_pos if phase2_subphase == "pregrasp" else phase2_grasp_pos
        active_target_name = phase2_subphase
        tcp_pos_before, _ = get_tcp_pose(task)
        target_error_before = torch.norm(tcp_pos_before - tcp_target_pos, dim=-1)
        current_joint = task.arm_joint_positions.detach().clone()
        rot_mask = [True, True, True]
        joint_target = move_joints_toward_tcp_ik(
            task,
            tcp_target_pos,
            ee_down_quat,
            current_joint,
            max_joint_delta=args.max_joint_delta,
            rot_mask=rot_mask,
        )
        raw_end_yaw_target = joint_target[:, phase2_end_yaw_joint_index].detach().clone()
        if not args.phase2_free_end_yaw:
            joint_target = clamp_joint_to_reference_window(
                joint_target,
                phase2_joint_reference,
                phase2_end_yaw_joint_index,
                args.phase2_end_yaw_max_offset,
            )
        step_scene(
            task,
            joint_target=joint_target,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=1,
        )
        tcp_pos, tcp_quat = get_tcp_pose(task)
        gripper_qpos = get_gripper_qpos(task)
        down_axis_world, down_alignment = compute_down_axis_alignment(
            tcp_quat,
            args.ee_down_axis_local,
            task,
        )
        (
            ee_x_world,
            ee_y_world,
            ee_z_world,
            ee_x_down,
            ee_y_down,
            ee_z_down,
        ) = compute_ee_axis_down_alignments(tcp_quat, task)
        joint_delta = torch.abs(joint_target - current_joint)
        pregrasp_error = torch.norm(tcp_pos - pregrasp_pos, dim=-1)
        tcp_to_grasp_point_error = torch.norm(tcp_pos - phase2_grasp_pos, dim=-1)
        current_target_shift = torch.norm(grasp_pos - physical_grasp_pos, dim=-1)
        active_target_error = torch.norm(tcp_pos - tcp_target_pos, dim=-1)
        if phase2_subphase == "pregrasp" and bool(
            torch.all(pregrasp_error <= args.pregrasp_tol).item()
        ):
            phase2_subphase = "grasp"

        if step_idx % log_every == 0 or bool(
            torch.all(tcp_to_grasp_point_error <= args.tcp_tol).item()
        ):
            print(
                "[THREE_PHASE_TEST][phase2]",
                f"step={step_idx}",
                f"subphase={active_target_name}",
                f"rot_mask={rot_mask}",
                f"target_xyz={as_list(tcp_target_pos[env_idx])}",
                f"tcp_xyz={as_list(tcp_pos[env_idx])}",
                f"virtual_grasp_xyz={as_list(phase2_grasp_pos[env_idx])}",
                f"physical_grasp_xyz={as_list(physical_grasp_pos[env_idx])}",
                f"current_grasp_xyz={as_list(grasp_pos[env_idx])}",
                f"current_target_shift={float(current_target_shift[env_idx].item()):.5f}",
                f"target_error={float(active_target_error[env_idx].item()):.5f}",
                f"down_axis_world={as_list(down_axis_world[env_idx])}",
                f"down_alignment={float(down_alignment[env_idx].item()):.5f}",
                f"ee_x_world={as_list(ee_x_world[env_idx])}",
                f"ee_y_world={as_list(ee_y_world[env_idx])}",
                f"ee_z_world={as_list(ee_z_world[env_idx])}",
                f"ee_axis_down_dot=[x:{float(ee_x_down[env_idx].item()):.5f},",
                f"y:{float(ee_y_down[env_idx].item()):.5f},",
                f"z:{float(ee_z_down[env_idx].item()):.5f}]",
                f"gripper_qpos={as_list(gripper_qpos[env_idx])}",
                f"end_yaw_joint={phase2_end_yaw_joint_name}",
                f"end_yaw_current={float(current_joint[env_idx, phase2_end_yaw_joint_index].item()):.5f}",
                f"end_yaw_raw_target={float(raw_end_yaw_target[env_idx].item()):.5f}",
                f"end_yaw_target={float(joint_target[env_idx, phase2_end_yaw_joint_index].item()):.5f}",
                f"end_yaw_ref={float(phase2_joint_reference[env_idx, phase2_end_yaw_joint_index].item()):.5f}",
                f"end_yaw_constrained={not args.phase2_free_end_yaw}",
                f"end_yaw_max_offset={args.phase2_end_yaw_max_offset}",
                f"joint_delta={as_list(joint_delta[env_idx])}",
                f"pregrasp_error={float(pregrasp_error[env_idx].item()):.5f}",
                f"tcp_to_grasp_point_error={float(tcp_to_grasp_point_error[env_idx].item()):.5f}",
                flush=True,
            )
        should_print_contacts = (
            step_idx % max(1, int(args.contact_debug_every)) == 0
            or float(current_target_shift[env_idx].item()) > float(args.contact_debug_shift)
        )
        if should_print_contacts:
            contact_rows = get_contact_debug_rows(task)
            if contact_rows:
                print(
                    "[THREE_PHASE_TEST][contact_debug]",
                    f"step={step_idx}",
                    f"target_shift={float(current_target_shift[env_idx].item()):.5f}",
                    f"contacts={contact_rows}",
                    flush=True,
                )
        if phase2_subphase == "grasp" and bool(
            torch.all(tcp_to_grasp_point_error <= args.tcp_tol).item()
        ):
            phase2_done = True
            break

    if not phase2_done:
        task.stop()
        raise RuntimeError("Phase2 did not reach the target grasp point; not entering phase3.")

    reference_joint_qpos = get_arm_joint_qpos(task)
    reference_joint_qvel = get_arm_joint_qvel(task)
    reference_tcp_pos, reference_tcp_quat = get_tcp_pose(task)
    reference_yaw_link_pos, reference_yaw_link_quat = get_link_pose_tensor(
        task.starlink_manipulator,
        PHASE2_REFERENCE_LINK_NAME,
        task,
    )
    reference_grasp_pos, reference_grasp_quat = get_target_grasp_pose(task)
    reference_marker_pos = task._get_grasp_marker_pos(
        reference_grasp_pos, reference_grasp_quat
    )
    reference_gripper_qpos = get_gripper_qpos(task)
    reference_down_axis_world, reference_down_alignment = compute_down_axis_alignment(
        reference_tcp_quat,
        args.ee_down_axis_local,
        task,
    )
    reference_tcp_to_virtual_grasp = torch.norm(reference_tcp_pos - phase2_grasp_pos, dim=-1)
    reference_tcp_to_physical_grasp = torch.norm(reference_tcp_pos - reference_grasp_pos, dim=-1)
    wrist_camera_config = FRANKA_S_Q_CAMERA
    base_camera_config = FRANKA_S_Q_CAMERA.get("base_camera", {})
    print("\n[THREE_PHASE_TEST][phase2_reference]")
    print(f"  arm_joint_names: {get_motor_joint_names(task)}")
    print(f"  arm_joint_qpos_rad: {as_list(reference_joint_qpos[env_idx], digits=8)}")
    print(f"  arm_joint_qvel_rad_s: {as_list(reference_joint_qvel[env_idx], digits=8)}")
    print(f"  gripper_qpos_open_reference: {as_list(reference_gripper_qpos[env_idx], digits=8)}")
    print(f"  tcp_xyz_reference: {as_list(reference_tcp_pos[env_idx], digits=8)}")
    print(f"  tcp_quat_wxyz_reference: {as_list(reference_tcp_quat[env_idx], digits=8)}")
    print(f"  yaw_joint_name_reference: {PHASE2_REFERENCE_YAW_JOINT_NAME}")
    print(f"  yaw_link_name_reference: {PHASE2_REFERENCE_LINK_NAME}")
    print(f"  yaw_link_xyz_reference: {as_list(reference_yaw_link_pos[env_idx], digits=8)}")
    print(f"  yaw_link_quat_wxyz_reference: {as_list(reference_yaw_link_quat[env_idx], digits=8)}")
    print(f"  target_grasp_xyz_current: {as_list(reference_grasp_pos[env_idx], digits=8)}")
    print(f"  target_grasp_quat_wxyz_current: {as_list(reference_grasp_quat[env_idx], digits=8)}")
    print(f"  grasp_marker_offset_local: {as_list(task.grasp_marker_offset_local, digits=8)}")
    print(f"  grasp_marker_xyz_current: {as_list(reference_marker_pos[env_idx], digits=8)}")
    print(f"  virtual_grasp_xyz_reference: {as_list(phase2_grasp_pos[env_idx], digits=8)}")
    print(
        "  tcp_error_reference:",
        f"to_virtual={float(reference_tcp_to_virtual_grasp[env_idx].item()):.8f}",
        f"to_physical={float(reference_tcp_to_physical_grasp[env_idx].item()):.8f}",
    )
    print(f"  ee_down_axis_local: {args.ee_down_axis_local}")
    print(f"  down_axis_world_reference: {as_list(reference_down_axis_world[env_idx], digits=8)}")
    print(f"  down_alignment_reference: {float(reference_down_alignment[env_idx].item()):.8f}")
    print(
        "  wrist_camera_mount:",
        f"end_effector_link={wrist_camera_config.get('end_effector_link')}",
        f"pos_offset={as_list(wrist_camera_config.get('pos_offset'), digits=8)}",
        f"lookat_offset={as_list(wrist_camera_config.get('lookat_offset'), digits=8)}",
        f"up_offset={as_list(wrist_camera_config.get('up_offset'), digits=8)}",
    )
    print(
        "  base_camera_mount:",
        f"mount_link={base_camera_config.get('mount_link')}",
        f"pos_offset={as_list(base_camera_config.get('pos_offset'), digits=8)}",
        f"lookat_offset={as_list(base_camera_config.get('lookat_offset'), digits=8)}",
        f"up_offset={as_list(base_camera_config.get('up_offset'), digits=8)}",
    )

    print("\n[THREE_PHASE_TEST] phase3: virtual attach target to TCP, lift together, then align cylinder centers in xy")
    attach_target_pos, attach_target_quat = get_target_root_pose(task)
    attach_tcp_pos, _ = get_tcp_pose(task)
    attached_target_pose = build_attached_target_pose(
        task,
        attach_target_pos,
        attach_target_quat,
        attach_tcp_pos,
        target_z_offset=0.0,
        follow_z=True,
    )
    lock_target_pose(task, attached_target_pose)

    phase3_lift_done = abs(float(args.target_dock_z_offset)) <= float(args.phase3_lift_tol)
    lift_start_tcp_pos = attach_tcp_pos.detach().clone()
    lift_tcp_target = lift_start_tcp_pos.detach().clone()
    lift_tcp_target[:, 2] += float(args.target_dock_z_offset)
    for lift_step_idx in range(1, max(1, int(args.phase3_lift_max_steps)) + 1):
        if phase3_lift_done:
            break
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        current_joint = task.arm_joint_positions.detach().clone()
        phase3_ee_quat = ee_down_quat if args.phase3_lock_gripper_orientation else None
        phase3_rot_mask = [True, True, True] if args.phase3_lock_gripper_orientation else None
        lift_joint_target = move_joints_toward_tcp_ik(
            task,
            lift_tcp_target,
            phase3_ee_quat,
            current_joint,
            max_joint_delta=args.max_joint_delta,
            rot_mask=phase3_rot_mask,
        )
        step_scene(
            task,
            joint_target=lift_joint_target,
            gripper_value=close_cmd,
            locked_target_pose=attached_target_pose,
            steps=1,
        )
        attached_target_pose = build_attached_target_pose(
            task,
            attach_target_pos,
            attach_target_quat,
            attach_tcp_pos,
            target_z_offset=0.0,
            follow_z=True,
        )
        lock_target_pose(task, attached_target_pose)
        tcp_now, _ = get_tcp_pose(task)
        lift_error = torch.abs(tcp_now[:, 2] - lift_tcp_target[:, 2])
        lift_delta_z = tcp_now[:, 2] - lift_start_tcp_pos[:, 2]
        if lift_step_idx % log_every == 0 or bool(
            torch.all(lift_error <= float(args.phase3_lift_tol)).item()
        ):
            target_root_now, _ = get_target_root_pose(task)
            print(
                "[THREE_PHASE_TEST][phase3_lift]",
                f"step={lift_step_idx}",
                f"tcp_xyz={as_list(tcp_now[env_idx])}",
                f"target_root_xyz={as_list(target_root_now[env_idx])}",
                f"lift_delta_z={float(lift_delta_z[env_idx].item()):.5f}",
                f"lift_error={float(lift_error[env_idx].item()):.5f}",
                f"target_z_offset={args.target_dock_z_offset}",
                f"phase3_rot_mask={phase3_rot_mask}",
                flush=True,
            )
        if bool(torch.all(lift_error <= float(args.phase3_lift_tol)).item()):
            phase3_lift_done = True
            break

    if not phase3_lift_done:
        task.stop()
        raise RuntimeError("Phase3 lift did not move the TCP/target to the requested z offset.")

    attach_target_pos, attach_target_quat = get_target_root_pose(task)
    attach_tcp_pos, _ = get_tcp_pose(task)
    attached_target_pose = build_attached_target_pose(
        task,
        attach_target_pos,
        attach_target_quat,
        attach_tcp_pos,
        target_z_offset=0.0,
        follow_z=args.attached_follow_z,
    )
    lock_target_pose(task, attached_target_pose)

    phase3_done = False
    cylinder_hold_count = 0
    for step_idx in range(1, max(1, int(args.phase3_steps)) + 1):
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        base_cylinder_center = get_cylinder_center_world(
            task,
            entity=task.starlink_manipulator,
            link_name="starlink_base_star_link",
            offset_xy=args.cylinder_offset_xy,
            scale=args.base_cylinder_scale,
        )
        target_cylinder_center = get_cylinder_center_world(
            task,
            entity=task.target_object,
            link_name=task.target_link_name,
            offset_xy=args.cylinder_offset_xy,
            scale=args.target_cylinder_scale,
        )
        cylinder_error_xy = base_cylinder_center[:, :2] - target_cylinder_center[:, :2]
        cylinder_error = torch.norm(cylinder_error_xy, dim=-1)
        tcp_now, _ = get_tcp_pose(task)
        tcp_delta = torch.zeros_like(tcp_now)
        tcp_delta[:, :2] = cylinder_error_xy
        tcp_step_norm = torch.norm(tcp_delta, dim=-1, keepdim=True)
        tcp_step_scale = torch.clamp(
            float(args.tcp_step_size) / torch.clamp(tcp_step_norm, min=1e-6),
            max=1.0,
        )
        pull_tcp_target = tcp_now + tcp_delta * tcp_step_scale
        current_joint = task.arm_joint_positions.detach().clone()
        phase3_ee_quat = ee_down_quat if args.phase3_lock_gripper_orientation else None
        phase3_rot_mask = [True, True, True] if args.phase3_lock_gripper_orientation else None
        pull_joint_target = move_joints_toward_tcp_ik(
            task,
            pull_tcp_target,
            phase3_ee_quat,
            current_joint,
            max_joint_delta=args.max_joint_delta,
            rot_mask=phase3_rot_mask,
        )
        step_scene(
            task,
            joint_target=pull_joint_target,
            gripper_value=close_cmd,
            locked_target_pose=attached_target_pose,
            steps=1,
        )
        attached_target_pose = build_attached_target_pose(
            task,
            attach_target_pos,
            attach_target_quat,
            attach_tcp_pos,
            target_z_offset=0.0,
            follow_z=args.attached_follow_z,
        )
        lock_target_pose(task, attached_target_pose)
        base_cylinder_center = get_cylinder_center_world(
            task,
            entity=task.starlink_manipulator,
            link_name="starlink_base_star_link",
            offset_xy=args.cylinder_offset_xy,
            scale=args.base_cylinder_scale,
        )
        target_cylinder_center = get_cylinder_center_world(
            task,
            entity=task.target_object,
            link_name=task.target_link_name,
            offset_xy=args.cylinder_offset_xy,
            scale=args.target_cylinder_scale,
        )
        cylinder_error = torch.norm(
            base_cylinder_center[:, :2] - target_cylinder_center[:, :2],
            dim=-1,
        )
        cylinder_z_gap = target_cylinder_center[:, 2] - base_cylinder_center[:, 2]
        if bool(torch.all(cylinder_error <= float(args.cylinder_align_tol)).item()):
            cylinder_hold_count += 1
        else:
            cylinder_hold_count = 0

        if step_idx % log_every == 0 or cylinder_hold_count > 0 or step_idx == args.phase3_steps:
            target_now, _ = get_target_grasp_pose(task)
            tcp_now, _ = get_tcp_pose(task)
            target_root_now, _ = get_target_root_pose(task)
            print(
                "[THREE_PHASE_TEST][phase3]",
                f"step={step_idx}",
                f"tcp_xyz={as_list(tcp_now[env_idx])}",
                f"target_root_xyz={as_list(target_root_now[env_idx])}",
                f"target_grasp_xyz={as_list(target_now[env_idx])}",
                f"base_cylinder_xyz={as_list(base_cylinder_center[env_idx])}",
                f"target_cylinder_xyz={as_list(target_cylinder_center[env_idx])}",
                f"cylinder_xy_error={float(cylinder_error[env_idx].item()):.5f}",
                f"cylinder_z_gap={float(cylinder_z_gap[env_idx].item()):.5f}",
                f"phase3_rot_mask={phase3_rot_mask}",
                f"hold={cylinder_hold_count}/{args.cylinder_align_hold_steps}",
                flush=True,
            )
        if cylinder_hold_count >= max(1, int(args.cylinder_align_hold_steps)):
            phase3_done = True
            break

    final_base, _ = task._get_base_pose()
    final_grasp, _ = get_target_grasp_pose(task)
    final_tcp, _ = get_tcp_pose(task)
    final_base_cylinder_center = get_cylinder_center_world(
        task,
        entity=task.starlink_manipulator,
        link_name="starlink_base_star_link",
        offset_xy=args.cylinder_offset_xy,
        scale=args.base_cylinder_scale,
    )
    final_target_cylinder_center = get_cylinder_center_world(
        task,
        entity=task.target_object,
        link_name=task.target_link_name,
        offset_xy=args.cylinder_offset_xy,
        scale=args.target_cylinder_scale,
    )
    final_base_drift = torch.norm(final_base[:, :2] - hold_root_pos[:, :2], dim=-1)
    final_tcp_error = torch.norm(final_tcp - final_grasp, dim=-1)
    final_cylinder_error = torch.norm(
        final_base_cylinder_center[:, :2] - final_target_cylinder_center[:, :2],
        dim=-1,
    )
    final_cylinder_z_gap = final_target_cylinder_center[:, 2] - final_base_cylinder_center[:, 2]

    print("\n[THREE_PHASE_TEST] summary")
    print(f"  phase1_done: {phase1_done}")
    print(f"  phase2_done: {phase2_done}")
    print(f"  phase3_done: {phase3_done}")
    print(f"  final_base_xyz: {as_list(final_base[env_idx])}")
    print(f"  final_grasp_xyz: {as_list(final_grasp[env_idx])}")
    print(f"  final_tcp_xyz: {as_list(final_tcp[env_idx])}")
    print(f"  final_base_cylinder_xyz: {as_list(final_base_cylinder_center[env_idx])}")
    print(f"  final_target_cylinder_xyz: {as_list(final_target_cylinder_center[env_idx])}")
    print(f"  final_base_hold_drift_xy: {float(final_base_drift[env_idx].item()):.5f}")
    print(f"  final_tcp_to_grasp_error: {float(final_tcp_error[env_idx].item()):.5f}")
    print(f"  final_cylinder_xy_error: {float(final_cylinder_error[env_idx].item()):.5f}")
    print(f"  final_cylinder_z_gap: {float(final_cylinder_z_gap[env_idx].item()):.5f}")
    print(f"  phase2_reference_yaw_joint_name: {PHASE2_REFERENCE_YAW_JOINT_NAME}")
    print(f"  phase2_reference_yaw_link_name: {PHASE2_REFERENCE_LINK_NAME}")
    print(f"  phase2_reference_yaw_link_xyz: {as_list(reference_yaw_link_pos[env_idx], digits=8)}")
    print(f"  phase2_reference_yaw_link_quat_wxyz: {as_list(reference_yaw_link_quat[env_idx], digits=8)}")
    print(f"  passed: {phase1_done and phase2_done and phase3_done}")

    if args.record_video:
        task.gsim.on_episode_end(1)
        print("[THREE_PHASE_TEST] Scene recording finished. Check src/recordings/video_ep_chunk_*.mp4")

    task.stop()
    if not (phase1_done and phase2_done and phase3_done):
        raise RuntimeError("Fixed three-phase test did not reach all required phase targets.")


if __name__ == "__main__":
    main()
