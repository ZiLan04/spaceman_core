"""Replay a trained stage-2 PPO arm grasp-pose policy in one Genesis env."""

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
from configs.registry import make_starlink_morph  # noqa: E402
from configs.registry import get_legacy_camera  # noqa: E402
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
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
    / "run_20260828_200223"
    / "checkpoints"
    / "ppo_satellite_stage2_arm_18000000_steps.zip"
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Policy steps to run in one replay episode. Stage-2 training uses 200.",
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
        "--success-tcp-threshold",
        type=float,
        default=None,
        help="Override the training TCP-to-grasp success threshold in meters.",
    )
    orientation_threshold_group = parser.add_mutually_exclusive_group()
    orientation_threshold_group.add_argument(
        "--success-orientation-angle-deg",
        type=float,
        default=None,
        help="Override the full-quaternion angular success threshold in degrees.",
    )
    orientation_threshold_group.add_argument(
        "--success-quat-similarity-threshold",
        type=float,
        default=None,
        help="Legacy equivalent override for the full-quaternion angular threshold.",
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
    parser.add_argument(
        "--grasp-check",
        dest="grasp_check",
        action="store_true",
        default=None,
        help=(
            "After the ideal pose / success is reached, hold the arm and close "
            "the gripper, then verify whether the target is actually grasped "
            "(contact forces, gripper blockage, target drift)."
        ),
    )
    parser.add_argument(
        "--no-grasp-check",
        dest="grasp_check",
        action="store_false",
        help="Disable the post-success grasp verification phase.",
    )
    parser.add_argument(
        "--grasp-close-value",
        type=float,
        default=0.0,
        help="Gripper command used to close (1=open, 0=close).",
    )
    parser.add_argument(
        "--grasp-close-steps",
        type=int,
        default=20,
        help="Decision steps used to ramp the gripper from open to the close command.",
    )
    parser.add_argument(
        "--grasp-verify-steps",
        type=int,
        default=120,
        help="Decision steps to keep the gripper closed and monitor the grasp.",
    )
    parser.add_argument(
        "--grasp-pull-steps",
        type=int,
        default=60,
        help="Decision steps used to pull the base/TCP away from the target.",
    )
    parser.add_argument(
        "--grasp-pull-distance",
        type=float,
        default=0.05,
        help=(
            "How far (m) to pull the base/TCP away from the target during the "
            "hold test. Set 0 to skip the pull test."
        ),
    )
    parser.add_argument(
        "--grasp-settle-steps",
        type=int,
        default=40,
        help="Decision steps to hold after the pull before judging the grasp.",
    )
    parser.add_argument(
        "--grasp-target-drift-threshold",
        type=float,
        default=0.02,
        help=(
            "Target grasp-point drift (m) above this during closing is reported "
            "as the target being knocked / pushed away."
        ),
    )
    parser.add_argument(
        "--grasp-contact-force-threshold",
        type=float,
        default=1e-3,
        help="Finger contact force above this counts as a real contact.",
    )
    parser.add_argument(
        "--grasp-log-every",
        type=int,
        default=5,
        help="Log every N grasp-verification decision steps.",
    )
    parser.set_defaults(
        center_envs_at_origin=None,
        target_position_noise=None,
        enable_target_relative_obs=None,
        stage2_enable_target_collision=None,
        grasp_check=None,
    )
    return parser


def config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def resolve_grasp_check(args) -> bool:
    """Grasp verification is enabled by default; --no-grasp-check disables it."""
    return True if getattr(args, "grasp_check", None) is None else bool(args.grasp_check)


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
            "morph": make_starlink_morph(
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
    grasp_check = resolve_grasp_check(args)
    if grasp_check and not enable_collision:
        if args.stage2_enable_target_collision is None:
            enable_collision = True
            print(
                "[STAGE2_REPLAY] Grasp verification is enabled: forcing target "
                "collision ON so the fingers can physically contact the target.",
                flush=True,
            )
        else:
            print(
                "[STAGE2_REPLAY] WARNING: grasp verification is enabled but target "
                "collision was explicitly disabled; the gripper cannot physically "
                "grasp the target and the check will likely report NO_CONTACT.",
                flush=True,
            )
    stage2_base_distance = (
        args.stage2_base_distance
        if args.stage2_base_distance is not None
        else float(config_value(config, "stage2_base_distance", 1.1))
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
        task.last_action_tensor[env_ids] = 0.0
        task._update_io_tensors()
        task.target_hold_count[env_ids] = 0
        task.prev_in_hold_zone[env_ids] = False
        task.ever_in_hold_zone[env_ids] = False
        task.reached_approach_milestone[env_ids] = False
        task.reached_prepose_milestone[env_ids] = False
        task.reached_midpose_milestone[env_ids] = False
        task.reached_nearpose_milestone[env_ids] = False
        task.reached_precision_milestone[env_ids] = False
        for episode_reward_sum in task.episode_reward_sums.values():
            episode_reward_sum[env_ids] = 0.0
        task.episode_min_tcp_to_grasp_error[env_ids] = float("inf")
        task.episode_min_pose_orientation_angle_deg[env_ids] = float("inf")
        task.episode_max_hold_count[env_ids] = 0
        task.prev_tcp_to_grasp_error[env_ids] = task.tcp_to_grasp_error[env_ids]
        task.prev_pose_orientation_angle[env_ids] = task.pose_orientation_angle[
            env_ids
        ]
        task.prev_tcp_pos[env_ids] = task._get_tcp_pose()[0][env_ids].detach().clone()
        task.prev_reference_link_pos[env_ids] = (
            task._get_reference_yaw_link_pose()[0][env_ids].detach().clone()
        )
        task.tcp_speed[env_ids] = 0.0
        task.reference_link_speed[env_ids] = 0.0
        task.tool_speed[env_ids] = 0.0


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

    tcp_to_grasp_error = scalar_info(
        info,
        "tcp_to_grasp_error",
        task_scalar(task, "tcp_to_grasp_error"),
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
    orientation_angle = scalar_info(
        info,
        "pose_orientation_angle",
        task_scalar(task, "pose_orientation_angle"),
    )
    speed_gate = scalar_info(
        info,
        "speed_reward_gate",
        task_scalar(task, "speed_reward_gate"),
    )
    orientation_gate = scalar_info(
        info,
        "orientation_reward_gate",
        task_scalar(task, "orientation_reward_gate"),
    )
    tool_speed = scalar_info(
        info,
        "tool_speed",
        task_scalar(task, "tool_speed"),
    )
    tool_speed_limit = scalar_info(
        info,
        "tool_speed_limit",
        task_scalar(task, "tool_speed_limit"),
    )
    reference_speed = scalar_info(
        info,
        "reference_link_speed",
        task_scalar(task, "reference_link_speed"),
    )
    tcp_speed = scalar_info(
        info,
        "tcp_speed",
        task_scalar(task, "tcp_speed"),
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
        "target_grasp_x": float(target_grasp_pose[0]),
        "target_grasp_y": float(target_grasp_pose[1]),
        "target_grasp_z": float(target_grasp_pose[2]),
        "reference_link_x": float(reference_yaw_link_pose[0]),
        "reference_link_y": float(reference_yaw_link_pose[1]),
        "reference_link_z": float(reference_yaw_link_pose[2]),
        "tcp_to_grasp_error": float(tcp_to_grasp_error),
        "pose_position_error": float(pose_position_error),
        "pose_quat_similarity": float(quat_similarity),
        "pose_orientation_angle_rad": float(orientation_angle),
        "pose_orientation_angle_deg": float(np.degrees(orientation_angle)),
        "orientation_reward_gate": float(orientation_gate),
        "tcp_speed": float(tcp_speed),
        "reference_link_speed": float(reference_speed),
        "speed_reward_gate": float(speed_gate),
        "tool_speed": float(tool_speed),
        "tool_speed_limit": float(tool_speed_limit),
        "base_drift_error": float(
            scalar_info(info, "base_drift_error", task_scalar(task, "base_drift_error"))
        ),
        "target_in_wrist_view": bool(info.get("target_in_wrist_view", False)),
        "reached_approach_milestone": bool(
            info.get("reached_approach_milestone", False)
        ),
        "reached_prepose_milestone": bool(
            info.get("reached_prepose_milestone", False)
        ),
        "reached_midpose_milestone": bool(
            info.get("reached_midpose_milestone", False)
        ),
        "reached_nearpose_milestone": bool(
            info.get("reached_nearpose_milestone", False)
        ),
        "reached_precision_milestone": bool(
            info.get("reached_precision_milestone", False)
        ),
        "reward_tcp": scalar_info(info, "reward_tcp", task_scalar(task, "reward_tcp")),
        "reward_orientation": scalar_info(
            info, "reward_orientation", task_scalar(task, "reward_orientation")
        ),
        "reward_pose_dense": scalar_info(
            info, "reward_pose_dense", task_scalar(task, "reward_pose_dense")
        ),
        "reward_speed": scalar_info(info, "reward_speed", task_scalar(task, "reward_speed")),
        "reward_action": scalar_info(info, "reward_action", task_scalar(task, "reward_action")),
        "reward_link_position": scalar_info(
            info, "reward_link_position", task_scalar(task, "reward_link_position")
        ),
        "reward_base": scalar_info(info, "reward_base", task_scalar(task, "reward_base")),
        "reward_view": scalar_info(info, "reward_view", task_scalar(task, "reward_view")),
        "reward_approach_milestone": scalar_info(
            info,
            "reward_approach_milestone",
            task_scalar(task, "reward_approach_milestone"),
        ),
        "reward_prepose_milestone": scalar_info(
            info,
            "reward_prepose_milestone",
            task_scalar(task, "reward_prepose_milestone"),
        ),
        "reward_midpose_milestone": scalar_info(
            info,
            "reward_midpose_milestone",
            task_scalar(task, "reward_midpose_milestone"),
        ),
        "reward_nearpose_milestone": scalar_info(
            info,
            "reward_nearpose_milestone",
            task_scalar(task, "reward_nearpose_milestone"),
        ),
        "reward_precision_milestone": scalar_info(
            info,
            "reward_precision_milestone",
            task_scalar(task, "reward_precision_milestone"),
        ),
        "reward_hold_zone": scalar_info(
            info, "reward_hold_zone", task_scalar(task, "reward_hold_zone")
        ),
        "reward_stable_hold": scalar_info(
            info, "reward_stable_hold", task_scalar(task, "reward_stable_hold")
        ),
        "reward_hold_zone_leave": scalar_info(
            info, "reward_hold_zone_leave", task_scalar(task, "reward_hold_zone_leave")
        ),
        "reward_success": scalar_info(
            info, "reward_success", task_scalar(task, "reward_success")
        ),
        "reward_time": scalar_info(info, "reward_time", task_scalar(task, "reward_time")),
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
    orientation_angle = scalar_info(info, "pose_orientation_angle")
    reference_speed = scalar_info(info, "reference_link_speed")
    tcp_error = scalar_info(info, "tcp_to_grasp_error")
    tcp_speed = scalar_info(info, "tcp_speed")
    tool_speed = scalar_info(info, "tool_speed")
    tool_speed_limit = scalar_info(info, "tool_speed_limit")
    speed_gate = scalar_info(info, "speed_reward_gate")
    base_drift = scalar_info(info, "base_drift_error")
    hold_count = scalar_info(info, "target_hold_count", task_scalar(task, "target_hold_count", 0.0))
    in_hold_zone = (
        tcp_error <= float(getattr(task, "tcp_position_threshold", 0.025))
        and orientation_angle
        <= float(getattr(task, "pose_orientation_angle_threshold", np.radians(15.0)))
        and bool(info.get("target_in_wrist_view", False))
    )
    velocity_threshold = float(getattr(task, "hold_velocity_threshold", 0.04))
    stable_hold = (
        in_hold_zone
        and tool_speed <= velocity_threshold
    )
    print(
        "[STAGE2_REPLAY] "
        f"step={step_idx:04d} "
        f"reward={reward:8.3f} "
        f"phase={info.get('phase', 'unknown')} "
        f"pose_err={pose_error:7.4f} "
        f"quat_sim={quat_similarity:6.3f} "
        f"angle_err={np.degrees(orientation_angle):6.2f}deg "
        f"tcp_speed={tcp_speed:7.4f} "
        f"ref_speed={reference_speed:7.4f} "
        f"tool_speed={tool_speed:7.4f}/{tool_speed_limit:7.4f} "
        f"speed_gate={speed_gate:5.2f} "
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


GRASP_FINGER_LINK_NAMES = (
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
)


def _tensor_to_numpy(value):
    if value is None:
        return None
    try:
        return np.asarray(torch.as_tensor(value).detach().cpu(), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError):
        return None


def env0_vector(value, size: int) -> np.ndarray:
    """Flatten a per-env tensor and return the first ``size`` values for env 0."""
    array = _tensor_to_numpy(value)
    if array is None:
        return np.full(size, np.nan)
    array = array.reshape(-1)
    if array.size < size:
        return np.full(size, np.nan)
    return array[:size]


def grasp_gripper_metrics(task):
    """Return (per-finger qpos, mean finger qpos, normalized closure in [0, 1]).

    Closure is normalized with the finger joint's upper limit (== fully closed),
    falling back to the configured ``finger_close`` value. Note that in this URDF
    the finger DOF limit is 0.04 rad while the config's ``finger_close`` is 0.4,
    so using the config value directly would understate closure by ~10x.
    """
    manip = getattr(task, "starlink_manipulator", None)
    finger_positions = np.full(2, np.nan)
    if manip is None:
        return finger_positions, float("nan"), float("nan")

    try:
        qpos = _tensor_to_numpy(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
        if qpos is not None:
            qpos = qpos.reshape(-1)
            for index in range(min(2, qpos.size)):
                finger_positions[index] = float(qpos[index])
    except Exception:
        pass

    valid = ~np.isnan(finger_positions)
    finger_mean = float(np.mean(finger_positions[valid])) if np.any(valid) else float("nan")

    close_reference = float("nan")
    try:
        limits = manip.robot.get_dofs_limit(dofs_idx_local=manip.fingers_dof)
        upper = _tensor_to_numpy(limits[1])
        if upper is not None and upper.size > 0:
            close_reference = float(np.max(np.abs(upper)))
    except Exception:
        pass
    if np.isnan(close_reference) or close_reference <= 1e-9:
        try:
            close_reference = float(_tensor_to_numpy(manip.finger_close).reshape(-1)[0])
        except Exception:
            close_reference = float("nan")

    if np.isnan(finger_mean) or np.isnan(close_reference) or close_reference <= 1e-9:
        closure = float("nan")
    else:
        closure = float(np.clip(finger_mean / close_reference, 0.0, 1.0))
    return finger_positions, finger_mean, closure


def grasp_contact_metrics(task) -> dict:
    """Summarize manipulator<->target contacts, isolating the two finger links."""
    result = {
        "finger_contacts": 0,
        "both_fingers": False,
        "total_contacts": 0,
        "finger_force": 0.0,
        "contact_links": "",
    }
    manip = getattr(task, "starlink_manipulator", None)
    if manip is None or getattr(task, "target_object", None) is None:
        return result

    try:
        contacts = manip.robot.get_contacts(with_entity=task.target_object.robot)
    except Exception:
        return result
    if not contacts:
        return result

    link_a = _tensor_to_numpy(contacts.get("link_a"))
    if link_a is None or link_a.size == 0:
        return result
    link_a = link_a.reshape(-1)
    link_b = _tensor_to_numpy(contacts.get("link_b"))
    link_b = link_b.reshape(-1) if link_b is not None else None

    force = contacts.get("force_a")
    if force is None:
        force = contacts.get("force_b")
    force = _tensor_to_numpy(force)
    if force is not None:
        force = force.reshape(-1, 3)

    try:
        link_names = {
            int(link.idx): getattr(link, "name", f"link_{int(link.idx)}")
            for link in manip.robot.links
        }
    except Exception:
        link_names = {}

    finger_link_indices = set()
    for name in GRASP_FINGER_LINK_NAMES:
        try:
            finger_link_indices.add(int(manip.robot.get_link(name).idx))
        except Exception:
            pass

    contacted_fingers = set()
    contact_partners = []
    total_force = 0.0
    for index in range(link_a.size):
        a_index = int(link_a[index])
        b_index = int(link_b[index]) if link_b is not None and link_b.size > index else -1
        a_is_finger = a_index in finger_link_indices
        b_is_finger = b_index in finger_link_indices
        if not (a_is_finger or b_is_finger):
            continue
        result["finger_contacts"] += 1
        if a_is_finger:
            contacted_fingers.add(a_index)
            contact_partners.append(link_names.get(b_index, str(b_index)))
        if b_is_finger:
            contacted_fingers.add(b_index)
            contact_partners.append(link_names.get(a_index, str(a_index)))
        if force is not None and force.shape[0] > index:
            total_force += float(np.linalg.norm(force[index]))

    result["both_fingers"] = len(contacted_fingers) >= 2
    result["finger_force"] = total_force
    result["total_contacts"] = int(link_a.size)
    result["contact_links"] = ",".join(sorted(set(contact_partners)))
    return result


def run_grasp_verification(task, env, args, *, episode_idx: int, trajectory_dir=None):
    """Close the gripper after the ideal pose and check whether the target is grasped.

    The base is held with the stage-2 PID, the arm is frozen at its success pose,
    and the gripper force is ramped to the close command. The phase reports finger
    contact, gripper closure and target drift, then pulls the base away to test
    whether the grasp actually holds.
    """
    num_envs = task.num_envs
    device = task.device
    dtype = task.datatype

    close_value = float(np.clip(args.grasp_close_value, 0.0, 1.0))
    open_value = float(task.gripper_open_value)
    close_steps = max(1, int(args.grasp_close_steps))
    verify_steps = max(0, int(args.grasp_verify_steps))
    pull_steps = max(0, int(args.grasp_pull_steps))
    settle_steps = max(0, int(args.grasp_settle_steps))
    pull_distance = max(0.0, float(args.grasp_pull_distance))
    drift_threshold = max(0.0, float(args.grasp_target_drift_threshold))
    force_threshold = max(0.0, float(args.grasp_contact_force_threshold))
    log_every = max(1, int(args.grasp_log_every))

    hold_joint_target = task.arm_joint_positions.detach().clone()
    base_hold_ref = task.stage2_base_hold_pos.detach().clone()
    grasp_start = task.target_grasp_pose[:, :3].detach().clone()
    tcp_start, _ = task._get_tcp_pose()
    tcp_start = tcp_start.detach().clone()
    rel_start = (tcp_start - grasp_start).detach().clone()

    pull_dir_xy = (base_hold_ref[:, :2] - grasp_start[:, :2]).detach().clone()
    pull_dir_norm = torch.norm(pull_dir_xy, dim=-1, keepdim=True)
    if float(pull_dir_norm.min().item()) < 1e-6:
        pull_dir_xy = torch.zeros_like(pull_dir_xy)
    else:
        pull_dir_xy = pull_dir_xy / pull_dir_norm

    state = {"joint_target": hold_joint_target, "gripper_cmd": open_value}
    original_apply_rl_action = task.apply_rl_action

    def grasp_apply_rl_action(action):
        # Bypass the policy: hold the arm at its success pose and drive the gripper.
        task.last_action_tensor = torch.zeros_like(task.last_action_tensor)
        task._set_base_hold_pid_target()
        task.starlink_manipulator.control_joint_pos(state["joint_target"])
        command = torch.full(
            (num_envs,), float(state["gripper_cmd"]), dtype=dtype, device=device
        )
        task.last_gripper_command = command
        task.starlink_manipulator.control_gripper(command)

    task.apply_rl_action = grasp_apply_rl_action
    zero_action = np.zeros((num_envs, 6), dtype=np.float32)

    close_end = close_steps
    verify_end = close_end + verify_steps
    pull_end = verify_end + pull_steps
    settle_end = pull_end + settle_steps
    total_steps = max(1, settle_end)

    def stage_name(step_index: int) -> str:
        if step_index < close_end:
            return "close"
        if step_index < verify_end:
            return "verify"
        if step_index < pull_end:
            return "pull"
        return "settle"

    rows = []
    max_contact_force = 0.0
    max_closure = 0.0
    max_total_contacts = 0
    ever_finger_contact = False
    ever_both_fingers = False
    drift_close = 0.0
    drift_verify = 0.0
    grasp_at_pull_start = None
    grasp_at_end = None

    print(
        "[STAGE2_GRASP] Starting grasp verification: "
        f"close_steps={close_steps}, verify_steps={verify_steps}, "
        f"pull_steps={pull_steps} ({pull_distance:.3f}m), settle_steps={settle_steps}, "
        f"close_value={close_value:.2f}",
        flush=True,
    )

    try:
        for step_index in range(total_steps):
            stage = stage_name(step_index)

            if step_index < close_end:
                fraction = (step_index + 1) / float(close_steps)
                state["gripper_cmd"] = open_value + (close_value - open_value) * fraction
            else:
                state["gripper_cmd"] = close_value

            if pull_steps > 0 and step_index >= verify_end:
                pull_fraction = min(
                    1.0, (step_index - verify_end + 1) / float(pull_steps)
                )
                task.stage2_base_hold_pos[:, :2] = (
                    base_hold_ref[:, :2] + pull_dir_xy * pull_distance * pull_fraction
                )

            _, rewards, _, _ = env.step(zero_action)

            grasp_now = task.target_grasp_pose[:, :3]
            tcp_now, _ = task._get_tcp_pose()
            target_root_now = env0_vector(task.target_object.robot.get_pos(), 3)
            base_now = env0_vector(task.stage2_base_hold_pos, 3)
            fingers, finger_mean, closure = grasp_gripper_metrics(task)
            contact = grasp_contact_metrics(task)
            drift = float(torch.norm(grasp_now - grasp_start, dim=-1)[0].item())
            rel_error = float(
                torch.norm((tcp_now - grasp_now) - rel_start, dim=-1)[0].item()
            )
            tcp_to_grasp = float(torch.norm(tcp_now - grasp_now, dim=-1)[0].item())

            if stage == "close":
                drift_close = drift
            elif stage == "verify":
                drift_verify = drift
            if stage == "pull" and grasp_at_pull_start is None:
                grasp_at_pull_start = grasp_now.detach().clone()
            grasp_at_end = grasp_now.detach().clone()

            max_contact_force = max(max_contact_force, contact["finger_force"])
            max_total_contacts = max(max_total_contacts, contact["total_contacts"])
            if not np.isnan(closure):
                max_closure = max(max_closure, closure)
            ever_finger_contact = ever_finger_contact or contact["finger_contacts"] > 0
            ever_both_fingers = ever_both_fingers or contact["both_fingers"]

            rows.append(
                {
                    "episode": int(episode_idx + 1),
                    "step": int(step_index + 1),
                    "stage": stage,
                    "gripper_cmd": float(state["gripper_cmd"]),
                    "finger_mean": finger_mean,
                    "gripper_closure": closure,
                    "finger_qpos_1": float(fingers[0]),
                    "finger_qpos_2": float(fingers[1]),
                    "finger_contacts": int(contact["finger_contacts"]),
                    "total_contacts": int(contact["total_contacts"]),
                    "both_fingers": bool(contact["both_fingers"]),
                    "finger_force": float(contact["finger_force"]),
                    "contact_links": contact["contact_links"],
                    "grasp_x": float(grasp_now[0, 0].item()),
                    "grasp_y": float(grasp_now[0, 1].item()),
                    "grasp_z": float(grasp_now[0, 2].item()),
                    "target_root_x": float(target_root_now[0]),
                    "target_root_y": float(target_root_now[1]),
                    "target_root_z": float(target_root_now[2]),
                    "base_hold_x": float(base_now[0]),
                    "base_hold_y": float(base_now[1]),
                    "target_drift": drift,
                    "tcp_to_grasp_error": tcp_to_grasp,
                    "rel_error": rel_error,
                    "reward": float(np.asarray(rewards).reshape(-1)[0]),
                }
            )

            if (
                step_index == 0
                or (step_index + 1) % log_every == 0
                or step_index == total_steps - 1
            ):
                print(
                    "[STAGE2_GRASP] "
                    f"step={step_index + 1:04d} "
                    f"stage={stage:6s} "
                    f"cmd={state['gripper_cmd']:5.2f} "
                    f"finger={finger_mean:7.4f} "
                    f"closure={closure:5.2f} "
                    f"contacts={contact['finger_contacts']} "
                    f"/{contact['total_contacts']} "
                    f"both={contact['both_fingers']} "
                    f"force={contact['finger_force']:8.5f} "
                    f"target_drift={drift:7.5f} "
                    f"tcp_err={tcp_to_grasp:7.5f} "
                    f"rel_err={rel_error:7.5f} "
                    f"partners=[{contact['contact_links']}]",
                    flush=True,
                )
    finally:
        task.apply_rl_action = original_apply_rl_action
        task.stage2_base_hold_pos[:, :] = base_hold_ref

    pull_follow = 0.0
    if pull_distance > 0.0 and grasp_at_pull_start is not None and grasp_at_end is not None:
        target_delta = (grasp_at_end - grasp_at_pull_start).reshape(1, 3)
        pull_follow = float(
            torch.sum(target_delta[:, :2] * pull_dir_xy, dim=-1)[0].item()
        )

    final_drift = (
        float(torch.norm(grasp_at_end - grasp_start, dim=-1)[0].item())
        if grasp_at_end is not None
        else float("nan")
    )

    if not ever_finger_contact:
        verdict = "NO_CONTACT"
    elif drift_verify >= drift_threshold:
        verdict = "TARGET_KNOCKED_AWAY"
    elif pull_distance > 0.0 and pull_follow < 0.5 * pull_distance:
        verdict = "GRASP_SLIPPED"
    else:
        verdict = "GRASP_OK"

    summary = {
        "episode": int(episode_idx + 1),
        "verdict": verdict,
        "finger_contact": bool(ever_finger_contact),
        "both_fingers": bool(ever_both_fingers),
        "max_contact_force": float(max_contact_force),
        "max_total_contacts": int(max_total_contacts),
        "max_closure": float(max_closure) if not np.isnan(max_closure) else float("nan"),
        "drift_after_close": float(drift_close),
        "drift_after_verify": float(drift_verify),
        "final_drift": float(final_drift),
        "pull_distance": float(pull_distance),
        "pull_follow": float(pull_follow),
        "drift_threshold": float(drift_threshold),
        "contact_force_threshold": float(force_threshold),
    }

    print(
        "[STAGE2_GRASP] RESULT "
        f"episode={episode_idx + 1} "
        f"verdict={verdict} "
        f"finger_contact={ever_finger_contact} "
        f"both_fingers={ever_both_fingers} "
        f"max_contact_force={max_contact_force:.5f} "
        f"max_total_contacts={max_total_contacts} "
        f"max_closure={max_closure:.3f} "
        f"drift_after_close={drift_close:.5f} "
        f"drift_after_verify={drift_verify:.5f} "
        f"pull_follow={pull_follow:.5f}/{pull_distance:.5f} "
        f"final_drift={final_drift:.5f} "
        f"drift_threshold={drift_threshold:.5f}",
        flush=True,
    )
    if verdict == "NO_CONTACT":
        print(
            "[STAGE2_GRASP] No finger contact was detected. Check that target "
            "collision is enabled and that the TCP really reaches the grasp point.",
            flush=True,
        )
    elif verdict == "TARGET_KNOCKED_AWAY":
        print(
            "[STAGE2_GRASP] Closing the gripper pushed the target away "
            f"(drift_after_verify={drift_verify:.4f}m >= {drift_threshold:.4f}m).",
            flush=True,
        )
    elif verdict == "GRASP_SLIPPED":
        print(
            "[STAGE2_GRASP] Fingers made contact but the target did not follow the "
            f"pull (follow={pull_follow:.4f}m vs {pull_distance:.4f}m).",
            flush=True,
        )
    elif verdict == "GRASP_OK":
        print(
            "[STAGE2_GRASP] Grasp looks physically consistent: contact detected, "
            "the target was not knocked away and it followed the pull.",
            flush=True,
        )

    if trajectory_dir is not None and rows:
        grasp_path = trajectory_dir / f"episode_{episode_idx + 1:03d}_grasp.csv"
        write_csv(grasp_path, rows)
        print(f"[STAGE2_REPLAY] Saved grasp trajectory: {grasp_path}", flush=True)

    return summary, rows


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
            f"target_noise={config_value(config, 'target_position_noise', False)}",
            flush=True,
        )

    configure_replay_runtime(args)
    env = make_replay_env(args, config)
    task = unwrap_stage2_task(env)
    grasp_check = resolve_grasp_check(args)
    patch_stage2_replay_reset_pose(task)
    if task is not None:
        task.stage2_replay_arm_initial_noise = max(0.0, float(args.arm_initial_noise))
        if args.success_tcp_threshold is not None:
            task.tcp_position_threshold = max(
                0.0, float(args.success_tcp_threshold)
            )
        if args.success_orientation_angle_deg is not None:
            task.pose_orientation_angle_threshold = np.radians(
                max(0.0, float(args.success_orientation_angle_deg))
            )
        elif args.success_quat_similarity_threshold is not None:
            similarity_threshold = np.clip(
                float(args.success_quat_similarity_threshold), 0.0, 1.0
            )
            task.pose_orientation_angle_threshold = 2.0 * np.arccos(
                similarity_threshold
            )
        task.pose_quat_similarity_threshold = np.cos(
            0.5 * task.pose_orientation_angle_threshold
        )
        task.replay_success_hold_steps = int(getattr(task, "success_hold_steps", 10))
        grasp_phase_steps = 0
        if grasp_check:
            grasp_phase_steps = (
                max(1, int(args.grasp_close_steps))
                + max(0, int(args.grasp_verify_steps))
                + max(0, int(args.grasp_pull_steps))
                + max(0, int(args.grasp_settle_steps))
            )
        task.max_steps = max(1, int(args.max_steps)) + grasp_phase_steps + 10
        print(
            "[STAGE2_REPLAY] Success thresholds: "
            f"tcp<={task.tcp_position_threshold:.3f}m, "
            f"orientation_angle<={np.degrees(task.pose_orientation_angle_threshold):.1f}deg, "
            "target_in_wrist_view=True, "
            f"tool_speed<={task.hold_velocity_threshold:.3f}m/s, "
            f"hold_steps={task.replay_success_hold_steps}",
            flush=True,
        )
        if grasp_check:
            # Keep the success termination from resetting the scene before we can
            # close the gripper: we detect success ourselves and break the loop.
            task.success_hold_steps = task.max_steps + 1
            print(
                "[STAGE2_REPLAY] Grasp verification enabled: policy phase stops at "
                "success, then the gripper closes and the grasp is checked for "
                f"{grasp_phase_steps} extra decision steps.",
                flush=True,
            )
        elif not args.stop_on_success:
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
                    getattr(task, "replay_success_hold_steps", 10) if task is not None else 10
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
                if grasp_check and hold_count >= replay_success_hold_steps:
                    print(
                        "[STAGE2_REPLAY] Success hold reached at step "
                        f"{step_idx + 1}; ending the policy phase to close the "
                        "gripper and verify the grasp.",
                        flush=True,
                    )
                    break
                if done:
                    break

            success_reached = first_stable_hold_step is not None
            episode_success = bool(final_info.get("success", False)) or success_reached
            final_tcp_error = scalar_info(final_info, "tcp_to_grasp_error")
            print(
                "[STAGE2_REPLAY] Episode done: "
                f"steps={step_idx + 1}, "
                f"return={episode_return:.3f}, "
                f"success={episode_success}, "
                f"first_stable_hold_step={first_stable_hold_step}, "
                f"max_hold_count={max_stable_hold_count}, "
                f"final_tcp_to_target_err={final_tcp_error:.5f}, "
                f"final_pose_err={scalar_info(final_info, 'pose_position_error'):.5f}, "
                f"final_quat_sim={scalar_info(final_info, 'pose_quat_similarity'):.5f}, "
                f"final_angle_err={np.degrees(scalar_info(final_info, 'pose_orientation_angle')):.2f}deg, "
                f"final_tool_speed={scalar_info(final_info, 'tool_speed'):.5f}",
                flush=True,
            )

            grasp_summary = None
            if grasp_check and task is not None and success_reached:
                grasp_summary, _ = run_grasp_verification(
                    task,
                    env,
                    args,
                    episode_idx=episode_idx,
                    trajectory_dir=trajectory_dir,
                )
            elif grasp_check:
                print(
                    "[STAGE2_REPLAY] Grasp verification skipped: success was not "
                    "reached in this episode.",
                    flush=True,
                )

            if trajectory_rows:
                initial_row = trajectory_rows[0]
                final_row = trajectory_rows[-1]
                summary_row = {
                    "episode": int(episode_idx + 1),
                    "steps": int(step_idx + 1),
                    "return": float(episode_return),
                    "success": bool(episode_success),
                    "first_stable_hold_step": (
                        int(first_stable_hold_step)
                        if first_stable_hold_step is not None
                        else ""
                    ),
                    "max_hold_count": int(max_stable_hold_count),
                    "initial_tcp_to_grasp_error": float(
                        initial_row["tcp_to_grasp_error"]
                    ),
                    "final_tcp_to_grasp_error": float(
                        final_row["tcp_to_grasp_error"]
                    ),
                    "final_pose_position_error": float(
                        final_row["pose_position_error"]
                    ),
                    "final_pose_quat_similarity": float(
                        final_row["pose_quat_similarity"]
                    ),
                    "final_pose_orientation_angle_deg": float(
                        final_row["pose_orientation_angle_deg"]
                    ),
                    "final_tool_speed": float(final_row["tool_speed"]),
                }
                if grasp_summary is not None:
                    summary_row.update(
                        {
                            f"grasp_{key}": value
                            for key, value in grasp_summary.items()
                            if key != "episode"
                        }
                    )
                summary_rows.append(summary_row)
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
