"""Replay a trained stage-2 PPO arm pre-grasp policy in one Genesis env."""

import argparse
import csv
from datetime import datetime
import json
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
project_root_path = project_src_path.parent
sys.path.append(str(project_src_path))

# Keep the custom feature extractor importable when loading visual policies.
import configs.asset_configs as asset_configs  # noqa: E402
from configs.config_starlink import _STARLINK_PATHS, to_posix  # noqa: E402
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from RL.policies.multimodal_extractor import MultiModalExtractor  # noqa: F401,E402
from tasks.RL_task_parallel import PPO  # noqa: E402
from tasks.RL_task_parallel_stage2 import make_stage2_vec_env  # noqa: E402

try:
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:
    VecNormalize = None


DEFAULT_OUTPUT_DIR = project_src_path / "recordings" / "stage2_policy_replay"
DEFAULT_MODEL_PATH = (
    project_root_path
    / "src"
    / "outputs"
    / "rl"
    / "ppo_satellite_stage2_arm"
    / "run_20260724_214752"
    / "checkpoints"
    / "ppo_satellite_stage2_arm_40000000_steps.zip"
)


def parse_vec2(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values, e.g. 10,10.")
    return tuple(values)


def parse_vec3(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. -3,-3,2.")
    return tuple(values)


def parse_resolution(text: str):
    normalized = text.lower().replace("x", ",")
    values = [int(item.strip()) for item in normalized.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 1280x720.")
    return tuple(values)


def find_train_config(model_path: Path) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "train_config.json"
        if candidate.exists():
            return candidate
    return None


def load_train_config(model_path: Path) -> dict:
    config_path = find_train_config(model_path)
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Replay a stage-2 arm PPO policy with num_envs=1."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=140,
        help="Policy steps to run in one replay episode. Stage-2 training used 120.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        default=True,
        help="End the replay as soon as the task success condition is reached.",
    )
    parser.add_argument(
        "--no-stop-on-success",
        dest="stop_on_success",
        action="store_false",
        help="Continue replay until --max-steps even after success.",
    )
    parser.add_argument(
        "--arm-initial-noise",
        type=float,
        default=0.03,
        help="Uniform reset noise range for the first six arm joints, in radians.",
    )
    parser.add_argument(
        "--success-position-threshold",
        type=float,
        default=0.3,
        help="Relaxed replay success position threshold in meters.",
    )
    parser.add_argument(
        "--success-quat-similarity-threshold",
        type=float,
        default=0.5,
        help="Relaxed replay success quaternion similarity threshold.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-rl-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory for per-episode TCP trajectory CSV files. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--no-record-trajectories",
        action="store_true",
        help="Disable writing trajectory CSV files.",
    )
    parser.add_argument("--video-name", default="stage2_policy_replay.mp4")
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--scene-camera-res", type=parse_resolution, default=(1280, 720))
    parser.add_argument("--scene-camera-pos", type=parse_vec3, default=(-3.0, -3.5, 2.5))
    parser.add_argument("--scene-camera-lookat", type=parse_vec3, default=(0.0, 0.5, 0.5))
    parser.add_argument("--scene-camera-fov", type=float, default=40.0)
    parser.add_argument("--target-asset", default=None)
    parser.add_argument("--env-spacing", type=parse_vec2, default=None)
    parser.add_argument("--n-envs-per-row", type=int, default=None)
    parser.add_argument("--center-envs-at-origin", action="store_true", default=None)
    parser.add_argument("--no-center-envs-at-origin", dest="center_envs_at_origin", action="store_false")
    parser.add_argument("--target-position-noise", action="store_true", default=None)
    parser.add_argument("--no-target-position-noise", dest="target_position_noise", action="store_false")
    parser.add_argument(
        "--enable-visual-obs",
        dest="enable_visual_obs",
        action="store_true",
        default=None,
        help="Replay with visual observations, overriding train_config.json.",
    )
    parser.add_argument(
        "--disable-visual-obs",
        dest="enable_visual_obs",
        action="store_false",
        help="Replay with low-dimensional non-visual observations, overriding train_config.json.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=None,
        help=(
            "Path to vecnormalize.pkl for non-visual replay. Defaults to the "
            "file saved next to the training run when available."
        ),
    )
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Do not load VecNormalize stats for non-visual replay.",
    )
    parser.add_argument(
        "--enable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_true",
        default=None,
        help="Replay non-visual policy with the 7D target pose relative observation.",
    )
    parser.add_argument(
        "--disable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_false",
        help="Replay non-visual policy with arm joints only.",
    )
    parser.add_argument("--stage2-base-distance", type=float, default=None)
    parser.add_argument("--stage2-virtual-grasp-height", type=float, default=None)
    parser.add_argument(
        "--stage2-enable-target-collision",
        action="store_true",
        default=None,
        help="Override replay to enable physical target collision.",
    )
    parser.add_argument(
        "--stage2-disable-target-collision",
        dest="stage2_enable_target_collision",
        action="store_false",
        help="Override replay to disable physical target collision.",
    )
    parser.set_defaults(
        center_envs_at_origin=None,
        target_position_noise=None,
        enable_target_relative_obs=None,
        stage2_enable_target_collision=None,
    )
    return parser


def config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def replay_visual_obs(args, config: dict) -> bool:
    if args.enable_visual_obs is not None:
        return bool(args.enable_visual_obs)
    return bool(config_value(config, "enable_visual_obs", True))


def replay_target_relative_obs(args, config: dict, enable_visual_obs: bool) -> bool:
    if enable_visual_obs:
        return False
    if args.enable_target_relative_obs is not None:
        return bool(args.enable_target_relative_obs)
    return bool(config_value(config, "enable_target_relative_obs", True))


def find_vecnormalize_stats(model_path: Path) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    return None


def configure_stage2_target_collision(target_asset: str, enable_collision: bool):
    if target_asset != "starlink":
        print(
            "[STAGE2_REPLAY] Stage-2 collision override is only implemented for "
            f"target_asset='starlink'; got {target_asset!r}.",
            flush=True,
        )
        return

    def make_starlink_for_stage2_replay():
        return {
            "morph": gs.morphs.URDF(
                file=to_posix(_STARLINK_PATHS["urdf"]),
                pos=(-2.0, 0.0, 0.5),
                euler=(0.0, 0.0, 180.0),
                scale=7e-1,
                fixed=False,
                collision=enable_collision,
                visualization=True,
            ),
            "material": gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
        }

    asset_configs._ASSET_FACTORIES["starlink"] = make_starlink_for_stage2_replay
    print(f"[STAGE2_REPLAY] Stage-2 target collision enabled: {enable_collision}", flush=True)


def check_vecnormalize_obs_shape(vec_env, stats_path: Path):
    expected_shape = getattr(vec_env.observation_space, "shape", None)
    obs_rms = getattr(vec_env, "obs_rms", None)
    mean = getattr(obs_rms, "mean", None)
    stats_shape = getattr(mean, "shape", None)
    if expected_shape is None or stats_shape is None:
        return
    if tuple(stats_shape) != tuple(expected_shape):
        raise ValueError(
            "VecNormalize observation shape does not match replay env: "
            f"stats_shape={tuple(stats_shape)}, env_shape={tuple(expected_shape)}, "
            f"stats_path={stats_path}. Check --enable/--disable-target-relative-obs "
            "and use the vecnormalize.pkl from the same training run."
        )


def configure_replay_runtime(args):
    SCENE_SETTINGS["show_viewer"] = not bool(args.no_viewer)

    TPV_CAM_SETTINGS["enable_recording"] = bool(args.record_video)
    TPV_CAM_SETTINGS["episodes_per_video"] = 1_000_000
    TPV_CAM_SETTINGS["camera"].update(
        {
            "res": tuple(args.scene_camera_res),
            "pos": tuple(args.scene_camera_pos),
            "lookat": tuple(args.scene_camera_lookat),
            "fov": float(args.scene_camera_fov),
            "GUI": False,
            "debug": True,
        }
    )

    base_camera_config = FRANKA_S_Q_CAMERA.get("base_camera", {})
    base_rl_obs_config = base_camera_config.get("rl_obs", {})
    base_rl_obs_config["enable_recording"] = False

    wrist_rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
    wrist_rl_obs_config["enable_recording"] = bool(args.record_rl_video)
    wrist_rl_obs_config["episodes_per_video"] = 1_000_000


def make_replay_env(args, config: dict):
    image_shape = tuple(config_value(config, "image_shape", (3, 84, 84)))
    env_spacing = (
        args.env_spacing
        if args.env_spacing is not None
        else tuple(config_value(config, "env_spacing", (10.0, 10.0)))
    )
    target_position_noise = (
        args.target_position_noise
        if args.target_position_noise is not None
        else bool(config_value(config, "target_position_noise", False))
    )
    center_envs_at_origin = (
        args.center_envs_at_origin
        if args.center_envs_at_origin is not None
        else bool(config_value(config, "center_envs_at_origin", True))
    )
    target_asset = args.target_asset or config_value(config, "target_asset", "starlink")
    enable_collision = (
        args.stage2_enable_target_collision
        if args.stage2_enable_target_collision is not None
        else bool(config_value(config, "stage2_enable_target_collision", False))
    )
    stage2_base_distance = (
        args.stage2_base_distance
        if args.stage2_base_distance is not None
        else float(config_value(config, "stage2_base_distance", 1.1))
    )
    virtual_grasp_height = (
        args.stage2_virtual_grasp_height
        if args.stage2_virtual_grasp_height is not None
        else float(config_value(config, "stage2_virtual_grasp_height", 0.08))
    )

    configure_stage2_target_collision(target_asset, bool(enable_collision))
    enable_visual_obs = replay_visual_obs(args, config)
    enable_target_relative_obs = replay_target_relative_obs(args, config, enable_visual_obs)

    env = make_stage2_vec_env(
        num_envs=1,
        target_asset=target_asset,
        env_spacing=env_spacing,
        n_envs_per_row=args.n_envs_per_row
        if args.n_envs_per_row is not None
        else config.get("n_envs_per_row"),
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=bool(config_value(config, "normalize_images", False)),
        enable_target_relative_obs=enable_target_relative_obs,
        target_position_noise=target_position_noise,
        stage2_base_distance=stage2_base_distance,
        virtual_grasp_height=virtual_grasp_height,
    )
    if enable_visual_obs or bool(args.no_vecnormalize):
        return env

    stats_path = args.vecnormalize_path or find_vecnormalize_stats(args.model_path.resolve())
    if stats_path is None:
        print(
            "[STAGE2_REPLAY] Non-visual replay requested, but no vecnormalize.pkl was found; "
            "using raw observations.",
            flush=True,
        )
        return env
    if VecNormalize is None:
        raise ImportError("VecNormalize is required to replay a normalized non-visual policy.")

    env = VecNormalize.load(str(stats_path), env)
    check_vecnormalize_obs_shape(env, stats_path)
    env.training = False
    env.norm_reward = False
    print(f"[STAGE2_REPLAY] Loaded VecNormalize stats: {stats_path}", flush=True)
    return env


def unwrap_stage2_task(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    if not hasattr(current, "env") or not hasattr(current.env, "task"):
        return None
    return current.env.task


def unwrap_stage2_gym_env(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    return getattr(current, "env", None)


def current_replay_obs(vec_env):
    gym_env = unwrap_stage2_gym_env(vec_env)
    if gym_env is None:
        return vec_env.reset()
    obs = gym_env._get_obs()
    if hasattr(vec_env, "normalize_obs"):
        return vec_env.normalize_obs(obs)
    return obs


def scalar_info(info: dict, key: str, default=np.nan):
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return default


def task_scalar(task, attr_name: str, default=np.nan):
    if task is None or not hasattr(task, attr_name):
        return default
    value = getattr(task, attr_name)
    if value is None:
        return default
    try:
        tensor = torch.as_tensor(value)
        return float(tensor.reshape(-1)[0].detach().cpu().item())
    except (TypeError, ValueError, RuntimeError):
        return default


def _stage2_replay_env_ids(task, env_ids=None):
    if env_ids is None:
        return torch.arange(task.num_envs, dtype=torch.long, device=task.device)
    if torch.is_tensor(env_ids):
        return env_ids.to(device=task.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(env_ids), dtype=torch.long, device=task.device).reshape(-1)


def _stage2_replay_initial_dofs(task, count: int):
    initial_dofs = task.starlink_manipulator.config["initial_dofs"].detach().clone()
    initial_dofs = initial_dofs.to(device=task.device, dtype=task.datatype).reshape(1, -1)
    return initial_dofs.repeat(count, 1)


def _apply_stage2_replay_arm_joint_noise(task, initial_dofs: torch.Tensor):
    noise_range = float(getattr(task, "stage2_replay_arm_initial_noise", 0.0))
    if noise_range <= 0.0:
        noise = torch.zeros_like(initial_dofs[:, :6])
    else:
        noise = torch.empty_like(initial_dofs[:, :6]).uniform_(-noise_range, noise_range)
    initial_dofs[:, :6] = initial_dofs[:, :6] + noise

    if hasattr(task, "joint_target_low") and hasattr(task, "joint_target_high"):
        initial_dofs[:, :6] = torch.clamp(
            initial_dofs[:, :6],
            min=task.joint_target_low.view(1, -1),
            max=task.joint_target_high.view(1, -1),
        )
    task.stage2_replay_last_joint_noise = noise.detach().clone()
    return initial_dofs


def _normalize_stage2_replay_joint_reference(task):
    initial_joint_positions = _stage2_replay_initial_dofs(task, task.num_envs)[:, :6]
    task.joint_reference = initial_joint_positions.detach().clone()
    task.initial_joint_positions = initial_joint_positions.detach().clone()


def _apply_stage2_replay_initial_dofs(task, env_ids=None):
    env_ids = _stage2_replay_env_ids(task, env_ids)
    if env_ids.numel() == 0:
        return
    controlled_dofs = (
        task.starlink_manipulator.motors_dof + task.starlink_manipulator.fingers_dof
    )
    initial_dofs = _stage2_replay_initial_dofs(task, int(env_ids.numel()))
    initial_dofs = _apply_stage2_replay_arm_joint_noise(task, initial_dofs)
    zero_velocity = torch.zeros_like(initial_dofs)
    task.starlink_manipulator.robot.set_dofs_position(
        initial_dofs,
        dofs_idx_local=controlled_dofs,
        envs_idx=env_ids,
        zero_velocity=True,
    )
    task.starlink_manipulator.robot.set_dofs_velocity(
        zero_velocity,
        dofs_idx_local=controlled_dofs,
        envs_idx=env_ids,
    )
    if torch.is_tensor(getattr(task, "initial_joint_positions", None)):
        if (
            task.initial_joint_positions.dim() == 2
            and task.initial_joint_positions.shape[0] == task.num_envs
        ):
            task.initial_joint_positions[env_ids] = initial_dofs[:, :6]
        else:
            _normalize_stage2_replay_joint_reference(task)
    task.starlink_manipulator.update_state()
    if hasattr(task, "_update_io_tensors"):
        task._update_io_tensors()


def patch_stage2_replay_reset_pose(task):
    if task is None or getattr(task, "_stage2_replay_pose_patch", False):
        return

    original_reset_done_envs = task.reset_done_envs

    def reset_done_envs_with_replay_pose(env_ids=None):
        _normalize_stage2_replay_joint_reference(task)
        result = original_reset_done_envs(env_ids=env_ids)
        if result:
            _apply_stage2_replay_initial_dofs(task, env_ids=env_ids)
        return result

    task.reset_done_envs = reset_done_envs_with_replay_pose
    task._stage2_replay_pose_patch = True


def unique_video_path(output_dir: Path, video_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = output_dir / video_name
    stem = requested.stem
    suffix = requested.suffix or ".mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{stem}_{timestamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{timestamp}_{counter:02d}{suffix}"
        counter += 1
    return candidate


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available filename near: {path}")


def resolve_trajectory_dir(args):
    if bool(args.no_record_trajectories):
        return None
    base_dir = args.trajectory_dir if args.trajectory_dir is not None else args.output_dir / "trajectory_data"
    trajectory_dir = next_available_path(base_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=False)
    return trajectory_dir


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_env_array(value, default=None):
    if value is None:
        return default
    try:
        array = np.asarray(torch.as_tensor(value).detach().cpu(), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError):
        return default
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim >= 2:
        return array[0]
    return array


def trajectory_row(step_idx: int, reward: float, action, info: dict, task=None):
    action_values = np.asarray(action, dtype=np.float64).reshape(-1)[:6]
    tcp_pose = first_env_array(
        getattr(task, "tcp_pose", None),
        default=np.full(7, np.nan),
    )
    virtual_grasp_pose = first_env_array(
        getattr(task, "virtual_grasp_pose", None),
        default=np.full(7, np.nan),
    )
    target_grasp_pose = first_env_array(
        getattr(task, "target_grasp_pose", None),
        default=np.full(7, np.nan),
    )
    reference_yaw_link_pose = first_env_array(
        getattr(task, "reference_yaw_link_pose", None),
        default=np.full(7, np.nan),
    )
    joint_positions = first_env_array(
        getattr(task, "arm_joint_positions", None),
        default=np.full(6, np.nan),
    )
    joint_noise = first_env_array(
        getattr(task, "stage2_replay_last_joint_noise", None),
        default=np.full(6, np.nan),
    )

    tcp_to_virtual_grasp_error = scalar_info(
        info,
        "tcp_to_virtual_grasp_error",
        task_scalar(task, "tcp_to_virtual_grasp_error"),
    )
    pose_position_error = scalar_info(
        info,
        "pose_position_error",
        task_scalar(task, "pose_position_error"),
    )
    quat_similarity = scalar_info(
        info,
        "pose_quat_similarity",
        task_scalar(task, "pose_quat_similarity"),
    )
    reference_speed = scalar_info(
        info,
        "reference_link_speed",
        task_scalar(task, "reference_link_speed"),
    )

    row = {
        "step": int(step_idx),
        "reward": float(reward),
        "tcp_x": float(tcp_pose[0]),
        "tcp_y": float(tcp_pose[1]),
        "tcp_z": float(tcp_pose[2]),
        "tcp_qw": float(tcp_pose[3]),
        "tcp_qx": float(tcp_pose[4]),
        "tcp_qy": float(tcp_pose[5]),
        "tcp_qz": float(tcp_pose[6]),
        "virtual_grasp_x": float(virtual_grasp_pose[0]),
        "virtual_grasp_y": float(virtual_grasp_pose[1]),
        "virtual_grasp_z": float(virtual_grasp_pose[2]),
        "target_grasp_x": float(target_grasp_pose[0]),
        "target_grasp_y": float(target_grasp_pose[1]),
        "target_grasp_z": float(target_grasp_pose[2]),
        "reference_link_x": float(reference_yaw_link_pose[0]),
        "reference_link_y": float(reference_yaw_link_pose[1]),
        "reference_link_z": float(reference_yaw_link_pose[2]),
        "tcp_to_virtual_grasp_error": float(tcp_to_virtual_grasp_error),
        "pose_position_error": float(pose_position_error),
        "pose_quat_similarity": float(quat_similarity),
        "reference_link_speed": float(reference_speed),
        "base_drift_error": float(
            scalar_info(info, "base_drift_error", task_scalar(task, "base_drift_error"))
        ),
        "target_in_wrist_view": bool(info.get("target_in_wrist_view", False)),
        "success": bool(info.get("success", False)),
    }
    for index in range(6):
        row[f"joint_{index + 1}"] = float(joint_positions[index])
        row[f"initial_noise_joint_{index + 1}"] = float(joint_noise[index])
        row[f"action_{index + 1}"] = float(action_values[index]) if action_values.size > index else np.nan
    return row


def print_step(step_idx: int, reward: float, action, info: dict, task=None):
    action_values = np.asarray(action, dtype=np.float32).reshape(-1)[:6]
    pose_error = scalar_info(info, "pose_position_error")
    quat_similarity = scalar_info(info, "pose_quat_similarity")
    reference_speed = scalar_info(info, "reference_link_speed")
    tcp_error = scalar_info(info, "tcp_to_virtual_grasp_error")
    base_drift = scalar_info(info, "base_drift_error")
    hold_count = scalar_info(info, "target_hold_count", task_scalar(task, "target_hold_count", 0.0))
    in_hold_zone = (
        pose_error <= float(getattr(task, "pose_position_threshold", 0.055))
        and quat_similarity >= float(getattr(task, "pose_quat_similarity_threshold", 0.95))
    )
    stable_hold = in_hold_zone and reference_speed <= float(
        getattr(task, "hold_velocity_threshold", 0.06)
    )
    print(
        "[STAGE2_REPLAY] "
        f"step={step_idx:04d} "
        f"reward={reward:8.3f} "
        f"phase={info.get('phase', 'unknown')} "
        f"pose_err={pose_error:7.4f} "
        f"quat_sim={quat_similarity:6.3f} "
        f"ref_speed={reference_speed:7.4f} "
        f"tcp_err={tcp_error:7.4f} "
        f"base_drift={base_drift:7.4f} "
        f"hold={hold_count:.0f} "
        f"in_hold={in_hold_zone} "
        f"stable={stable_hold} "
        f"view={bool(info.get('target_in_wrist_view', False))} "
        f"success={bool(info.get('success', False))} "
        "action=["
        + ", ".join(f"{value: .3f}" for value in action_values)
        + "]",
        flush=True,
    )


def save_scene_video(task, args):
    if not args.record_video or task is None:
        return
    cam = getattr(getattr(task, "gsim", None), "cam", None)
    if cam is None:
        print("[STAGE2_REPLAY] Scene video was requested, but no TPV camera exists.", flush=True)
        return
    if not getattr(cam, "_recorded_imgs", None):
        print("[STAGE2_REPLAY] Scene video was requested, but no frames were recorded.", flush=True)
        return

    video_path = unique_video_path(args.output_dir, args.video_name)
    cam.stop_recording(save_to_filename=str(video_path), fps=max(1, int(args.video_fps)))
    task.gsim.camera_enable_recording = False
    print(f"[STAGE2_REPLAY] Saved scene video: {video_path}", flush=True)


def main():
    args = build_arg_parser().parse_args()
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {model_path}")
    if PPO is None:
        raise ImportError("stable-baselines3 is not installed in this environment.")

    config = load_train_config(model_path)
    print(f"[STAGE2_REPLAY] Loading policy: {model_path}", flush=True)
    if config:
        print(
            "[STAGE2_REPLAY] Using train_config with num_envs forced to 1: "
            f"target_asset={config_value(config, 'target_asset', 'starlink')}, "
            f"visual_obs={replay_visual_obs(args, config)}, "
            f"target_relative_obs={replay_target_relative_obs(args, config, replay_visual_obs(args, config))}, "
            f"base_distance={config_value(config, 'stage2_base_distance', 1.1)}, "
            f"virtual_height={config_value(config, 'stage2_virtual_grasp_height', 0.08)}, "
            f"target_noise={config_value(config, 'target_position_noise', False)}",
            flush=True,
        )

    configure_replay_runtime(args)
    env = make_replay_env(args, config)
    task = unwrap_stage2_task(env)
    patch_stage2_replay_reset_pose(task)
    if task is not None:
        task.stage2_replay_arm_initial_noise = max(0.0, float(args.arm_initial_noise))
        task.pose_position_threshold = max(0.0, float(args.success_position_threshold))
        task.pose_quat_similarity_threshold = float(args.success_quat_similarity_threshold)
        task.replay_success_hold_steps = int(getattr(task, "success_hold_steps", 6))
        task.max_steps = max(1, int(args.max_steps))
        print(
            "[STAGE2_REPLAY] Relaxed success thresholds: "
            f"position<={task.pose_position_threshold:.3f}m, "
            f"quat_sim>={task.pose_quat_similarity_threshold:.3f}, "
            f"speed<={task.hold_velocity_threshold:.3f}, "
            f"hold_steps={task.replay_success_hold_steps}",
            flush=True,
        )
        if not args.stop_on_success:
            task.success_hold_steps = task.max_steps + 1
    model = PPO.load(str(model_path), env=env, device=args.device)
    trajectory_dir = resolve_trajectory_dir(args)
    summary_rows = []
    if trajectory_dir is not None:
        print(f"[STAGE2_REPLAY] Saving TCP trajectory CSV files to: {trajectory_dir}", flush=True)

    try:
        for episode_idx in range(max(1, int(args.episodes))):
            obs = env.reset()
            obs = current_replay_obs(env)
            if task is not None:
                joint_noise = getattr(task, "stage2_replay_last_joint_noise", None)
                if joint_noise is not None:
                    noise_values = joint_noise.reshape(-1, 6)[0].detach().cpu().numpy()
                    print(
                        "[STAGE2_REPLAY] "
                        f"arm_initial_noise=[{', '.join(f'{value: .4f}' for value in noise_values)}]",
                        flush=True,
                    )
            episode_return = 0.0
            final_info = {}
            first_stable_hold_step = None
            max_stable_hold_count = 0
            max_steps = max(1, int(args.max_steps))
            trajectory_rows = [
                trajectory_row(
                    step_idx=0,
                    reward=0.0,
                    action=np.zeros(6, dtype=np.float32),
                    info={},
                    task=task,
                )
            ]
            print(f"[STAGE2_REPLAY] Episode {episode_idx + 1} / {args.episodes}", flush=True)

            for step_idx in range(max_steps):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = env.step(action)
                reward = float(np.asarray(rewards).reshape(-1)[0])
                done = bool(np.asarray(dones).reshape(-1)[0])
                info = dict(infos[0]) if infos else {}
                episode_return += reward
                final_info = info
                trajectory_rows.append(
                    trajectory_row(
                        step_idx=step_idx + 1,
                        reward=reward,
                        action=action[0],
                        info=info,
                        task=task,
                    )
                )

                hold_count = int(scalar_info(info, "target_hold_count", 0.0))
                max_stable_hold_count = max(max_stable_hold_count, hold_count)
                replay_success_hold_steps = int(
                    getattr(task, "replay_success_hold_steps", 6) if task is not None else 6
                )
                if hold_count >= replay_success_hold_steps and first_stable_hold_step is None:
                    first_stable_hold_step = step_idx + 1

                if (
                    step_idx == 0
                    or done
                    or args.log_interval <= 1
                    or (step_idx + 1) % args.log_interval == 0
                ):
                    print_step(step_idx + 1, reward, action[0], info, task=task)

                if args.sleep > 0.0:
                    time.sleep(args.sleep)
                if done:
                    break

            final_tcp_error = scalar_info(final_info, "tcp_to_virtual_grasp_error")
            print(
                "[STAGE2_REPLAY] Episode done: "
                f"steps={step_idx + 1}, "
                f"return={episode_return:.3f}, "
                f"success={bool(final_info.get('success', False))}, "
                f"first_stable_hold_step={first_stable_hold_step}, "
                f"max_hold_count={max_stable_hold_count}, "
                f"final_tcp_to_target_err={final_tcp_error:.5f}, "
                f"final_pose_err={scalar_info(final_info, 'pose_position_error'):.5f}, "
                f"final_quat_sim={scalar_info(final_info, 'pose_quat_similarity'):.5f}, "
                f"final_ref_speed={scalar_info(final_info, 'reference_link_speed'):.5f}",
                flush=True,
            )
            if trajectory_rows:
                initial_row = trajectory_rows[0]
                final_row = trajectory_rows[-1]
                summary_rows.append(
                    {
                        "episode": int(episode_idx + 1),
                        "steps": int(step_idx + 1),
                        "return": float(episode_return),
                        "success": bool(final_info.get("success", False)),
                        "first_stable_hold_step": (
                            int(first_stable_hold_step)
                            if first_stable_hold_step is not None
                            else ""
                        ),
                        "max_hold_count": int(max_stable_hold_count),
                        "initial_tcp_to_virtual_grasp_error": float(
                            initial_row["tcp_to_virtual_grasp_error"]
                        ),
                        "final_tcp_to_virtual_grasp_error": float(
                            final_row["tcp_to_virtual_grasp_error"]
                        ),
                        "final_pose_position_error": float(
                            final_row["pose_position_error"]
                        ),
                        "final_pose_quat_similarity": float(
                            final_row["pose_quat_similarity"]
                        ),
                    }
                )
            if trajectory_dir is not None:
                trajectory_path = trajectory_dir / f"episode_{episode_idx + 1:03d}.csv"
                write_csv(trajectory_path, trajectory_rows)
                print(f"[STAGE2_REPLAY] Saved TCP trajectory: {trajectory_path}", flush=True)
                if summary_rows:
                    summary_path = trajectory_dir / "summary.csv"
                    write_csv(summary_path, summary_rows)
                    print(f"[STAGE2_REPLAY] Saved trajectory summary: {summary_path}", flush=True)
        save_scene_video(task, args)
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
