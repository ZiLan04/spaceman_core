"""Replay a trained stage-1 PPO base-alignment policy in one Genesis env."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
project_root_path = project_src_path.parent
sys.path.append(str(project_src_path))

# Keep the custom feature extractor importable when loading visual policies.
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from RL.policies.multimodal_extractor import MultiModalExtractor  # noqa: F401,E402
from tasks.RL_task_parallel import PPO  # noqa: E402
from tasks.RL_task_parallel_stage1 import make_stage1_vec_env  # noqa: E402

try:
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:
    VecNormalize = None


DEFAULT_OUTPUT_DIR = project_src_path / "recordings" / "stage1_policy_replay"
DEFAULT_MODEL_PATH = (
    project_root_path
    / "src"
    / "outputs"
    / "rl"
    / "ppo_satellite_stage1_base"
    / "run_20260718_201139"
    / "checkpoints"
    / "ppo_satellite_stage1_base_40000000_steps.zip"
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
        description="Replay the 40,000,000-step stage-1 base PPO policy with num_envs=1."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=220,
        help="Policy steps to run in one replay episode. Stage-1 training used 220.",
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
        help="Run until --max-steps even if the task success condition is reached.",
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
        help="Directory for per-episode trajectory CSV files. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--no-record-trajectories",
        action="store_true",
        help="Disable writing trajectory CSV files.",
    )
    parser.add_argument("--video-name", default="stage1_policy_replay.mp4")
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
        "--base-xy-noise",
        type=parse_vec2,
        default=(0.05, 0.05),
        help="Uniform replay reset noise range in meters for manipulator base x,y.",
    )
    parser.add_argument(
        "--no-base-xy-noise",
        action="store_true",
        help="Disable manipulator base x,y reset perturbation.",
    )
    parser.add_argument(
        "--stage1-base-goal-distance",
        type=float,
        default=None,
        help=(
            "Replay-only override for the desired base distance from the target "
            "grasp point in meters. Training used 1.3."
        ),
    )
    parser.add_argument(
        "--stage1-eval-goal-distance",
        type=float,
        default=None,
        help=(
            "Replay-only override for reward, success, desired_base_x/y, and "
            "dock_pos_error. Leave unset to evaluate with the policy goal distance."
        ),
    )
    parser.set_defaults(center_envs_at_origin=None, target_position_noise=None)
    return parser


def config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def replay_visual_obs(args, config: dict) -> bool:
    if args.enable_visual_obs is not None:
        return bool(args.enable_visual_obs)
    return bool(config_value(config, "enable_visual_obs", True))


def find_vecnormalize_stats(model_path: Path) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    return None


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
    base_rl_obs_config["enable_recording"] = bool(args.record_rl_video)
    base_rl_obs_config["episodes_per_video"] = 1_000_000

    wrist_rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
    wrist_rl_obs_config["enable_recording"] = False


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

    enable_visual_obs = replay_visual_obs(args, config)
    env = make_stage1_vec_env(
        num_envs=1,
        target_asset=args.target_asset or config_value(config, "target_asset", "starlink"),
        env_spacing=env_spacing,
        n_envs_per_row=args.n_envs_per_row
        if args.n_envs_per_row is not None
        else config.get("n_envs_per_row"),
        center_envs_at_origin=center_envs_at_origin,
        enable_visual_obs=enable_visual_obs,
        image_shape=image_shape,
        normalize_images=bool(config_value(config, "normalize_images", False)),
        target_position_noise=target_position_noise,
    )
    if enable_visual_obs or bool(args.no_vecnormalize):
        return env

    stats_path = args.vecnormalize_path or find_vecnormalize_stats(args.model_path.resolve())
    if stats_path is None:
        print(
            "[STAGE1_REPLAY] Non-visual replay requested, but no vecnormalize.pkl was found; "
            "using raw observations.",
            flush=True,
        )
        return env
    if VecNormalize is None:
        raise ImportError("VecNormalize is required to replay a normalized non-visual policy.")

    env = VecNormalize.load(str(stats_path), env)
    env.training = False
    env.norm_reward = False
    print(f"[STAGE1_REPLAY] Loaded VecNormalize stats: {stats_path}", flush=True)
    return env


def unwrap_stage1_task(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    if not hasattr(current, "env") or not hasattr(current.env, "task"):
        return None
    return current.env.task


def unwrap_stage1_gym_env(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    return getattr(current, "env", None)


def current_replay_obs(vec_env):
    gym_env = unwrap_stage1_gym_env(vec_env)
    if gym_env is None:
        return vec_env.reset()
    obs = gym_env._get_obs()
    if hasattr(vec_env, "normalize_obs"):
        return vec_env.normalize_obs(obs)
    return obs


def apply_replay_goal_distance_override(task, args):
    if task is None or args.stage1_base_goal_distance is None:
        pass
    else:
        if args.stage1_base_goal_distance <= 0.0:
            raise ValueError(
                "--stage1-base-goal-distance should be positive, "
                f"got {args.stage1_base_goal_distance}"
            )
        task.base_goal_threshold = float(args.stage1_base_goal_distance)
        print(
            "[STAGE1_REPLAY] Overriding stage-1 policy goal distance from grasp point: "
            f"{task.base_goal_threshold:.4f} m",
            flush=True,
        )

    if task is None or args.stage1_eval_goal_distance is None:
        return
    if args.stage1_eval_goal_distance <= 0.0:
        raise ValueError(
            "--stage1-eval-goal-distance should be positive, "
            f"got {args.stage1_eval_goal_distance}"
        )
    task.reward_base_goal_threshold = float(args.stage1_eval_goal_distance)
    print(
        "[STAGE1_REPLAY] Overriding stage-1 eval/reward goal distance from grasp point: "
        f"{task.reward_base_goal_threshold:.4f} m",
        flush=True,
    )


def apply_replay_base_xy_noise(task, args):
    if task is None or bool(args.no_base_xy_noise):
        return np.zeros(2, dtype=np.float32)

    noise_range = np.asarray(args.base_xy_noise, dtype=np.float32)
    if noise_range.shape != (2,):
        raise ValueError(f"--base-xy-noise should contain two values, got {args.base_xy_noise}")
    if np.any(noise_range < 0.0):
        raise ValueError(f"--base-xy-noise values should be non-negative, got {args.base_xy_noise}")
    if not np.any(noise_range > 0.0):
        return np.zeros(2, dtype=np.float32)

    noise_xy_np = np.random.uniform(-noise_range, noise_range).astype(np.float32)
    noise_xy = torch.as_tensor(noise_xy_np, dtype=task.datatype, device=task.device)
    base_pos, base_quat = task._get_base_pose()
    perturbed_base_pos = base_pos.detach().clone()
    perturbed_base_pos[:, :2] = perturbed_base_pos[:, :2] + noise_xy.unsqueeze(0)
    root_qpos = torch.cat([perturbed_base_pos, base_quat.detach().clone()], dim=-1)

    root_qpos_to_set = root_qpos if task.num_envs > 1 else root_qpos[0]
    task.starlink_manipulator.robot.set_qpos(
        root_qpos_to_set,
        qs_idx_local=list(range(7)),
        zero_velocity=True,
    )
    task.prev_base_pos = None
    task.starlink_manipulator.update_state()
    if hasattr(task.starlink_manipulator, "pid"):
        task.starlink_manipulator.pid.reset()
        task.starlink_manipulator.capture_current_base_pose_as_pid_setpoint()
    task.gsim.step()
    task.starlink_manipulator.update_state()
    task._update_io_tensors()

    task.episode_start_base_pos = task._get_base_pose()[0].detach().clone()
    task.episode_start_base_quat = task._get_base_pose()[1].detach().clone()
    if task.base_goal_hold_count is not None:
        task.base_goal_hold_count[:] = 0
    if task.prev_base_goal_error is not None:
        task.prev_base_goal_error = task.dock_pos_error.detach().clone()
    if task.prev_in_goal_band is not None:
        task.prev_in_goal_band = (task.dock_pos_error <= task.base_goal_band).detach().clone()
    if task.last_action_tensor is not None:
        task.last_action_tensor[:] = 0.0
    if task.base_target_offset_command is not None:
        task.base_target_offset_command[:] = 0.0
    if task.base_target_position_command is not None:
        task.base_target_position_command = task.base_planar_position.detach().clone()
    if task.prev_base_target_position_command is not None:
        task.prev_base_target_position_command = task.base_planar_position.detach().clone()
    if task.last_step_reward is not None:
        task.last_step_reward[:] = 0.0

    return noise_xy_np


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
    action_xy = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
    terminal_obs = info.get("terminal_observation") if isinstance(info, dict) else None
    terminal_state = None
    if isinstance(terminal_obs, dict) and "state" in terminal_obs:
        terminal_state = np.asarray(terminal_obs["state"], dtype=np.float64).reshape(-1)

    if terminal_state is not None and terminal_state.size >= 7:
        base_pose = terminal_state[:7]
    else:
        base_pose = first_env_array(getattr(task, "base_satellite_pose", None), default=np.full(7, np.nan))

    desired_base = first_env_array(
        getattr(
            task,
            "reward_desired_base_planar_position",
            getattr(task, "desired_base_planar_position", None),
        ),
        default=np.array([np.nan, np.nan]),
    )
    desired_x = scalar_info(info, "desired_base_x", float(desired_base[0]))
    desired_y = scalar_info(info, "desired_base_y", float(desired_base[1]))
    policy_desired_base = first_env_array(
        getattr(task, "desired_base_planar_position", None),
        default=np.array([np.nan, np.nan]),
    )
    policy_desired_x = scalar_info(
        info,
        "policy_desired_base_x",
        float(policy_desired_base[0]),
    )
    policy_desired_y = scalar_info(
        info,
        "policy_desired_base_y",
        float(policy_desired_base[1]),
    )

    dock_error = scalar_info(
        info,
        "dock_pos_error",
        task_scalar(task, "dock_pos_error"),
    )
    target_distance = scalar_info(
        info,
        "base_alignment_error",
        task_scalar(task, "base_alignment_error"),
    )
    base_speed = scalar_info(
        info,
        "base_planar_speed",
        task_scalar(task, "base_planar_speed"),
    )

    return {
        "step": int(step_idx),
        "reward": float(reward),
        "base_x": float(base_pose[0]),
        "base_y": float(base_pose[1]),
        "base_z": float(base_pose[2]),
        "base_qw": float(base_pose[3]),
        "base_qx": float(base_pose[4]),
        "base_qy": float(base_pose[5]),
        "base_qz": float(base_pose[6]),
        "desired_base_x": float(desired_x),
        "desired_base_y": float(desired_y),
        "policy_desired_base_x": float(policy_desired_x),
        "policy_desired_base_y": float(policy_desired_y),
        "dock_pos_error": float(dock_error),
        "target_distance": float(target_distance),
        "base_planar_speed": float(base_speed),
        "action_x": float(action_xy[0]) if action_xy.size > 0 else np.nan,
        "action_y": float(action_xy[1]) if action_xy.size > 1 else np.nan,
        "success": bool(info.get("success", False)) if isinstance(info, dict) else False,
        "failure": bool(info.get("failure", False)) if isinstance(info, dict) else False,
    }


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


def print_step(step_idx: int, reward: float, action, info: dict, task=None):
    action_xy = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
    dock_error = scalar_info(info, "dock_pos_error")
    target_distance = scalar_info(info, "base_alignment_error")
    base_speed = scalar_info(info, "base_planar_speed")
    hold_count = task_scalar(task, "base_goal_hold_count", 0.0)
    roll_pitch_error = scalar_info(info, "base_roll_pitch_error")
    yaw_error = scalar_info(info, "base_yaw_error")
    stable_stop = dock_error <= 0.08 and base_speed <= 0.025
    switch_hold_steps = int(
        getattr(task, "replay_switch_hold_steps", getattr(task, "success_hold_steps", 12))
        if task is not None
        else 12
    )
    switch_ready = stable_stop and hold_count >= switch_hold_steps
    print(
        "[STAGE1_REPLAY] "
        f"step={step_idx:04d} "
        f"reward={reward:8.3f} "
        f"target_dist={target_distance:7.4f} "
        f"dock_error={dock_error:7.4f} "
        f"base_speed={base_speed:7.4f} "
        f"hold={hold_count:.0f} "
        f"stable_stop={stable_stop} "
        f"switch_ready={switch_ready} "
        f"roll_pitch={roll_pitch_error:6.3f} "
        f"yaw={yaw_error:6.3f} "
        f"success={bool(info.get('success', False))} "
        f"failure={bool(info.get('failure', False))} "
        f"action=[{action_xy[0]: .4f}, {action_xy[1]: .4f}]",
        flush=True,
    )


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
    raise RuntimeError(f"Could not find an available video filename near: {path}")


def save_scene_video(task, args):
    if not args.record_video or task is None:
        return
    cam = getattr(getattr(task, "gsim", None), "cam", None)
    if cam is None:
        print("[STAGE1_REPLAY] Scene video was requested, but no TPV camera exists.", flush=True)
        return
    if not getattr(cam, "_recorded_imgs", None):
        print("[STAGE1_REPLAY] Scene video was requested, but no frames were recorded.", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = next_available_path(args.output_dir / args.video_name)
    cam.stop_recording(save_to_filename=str(video_path), fps=max(1, int(args.video_fps)))
    task.gsim.camera_enable_recording = False
    print(f"[STAGE1_REPLAY] Saved scene video: {video_path}", flush=True)


def main():
    args = build_arg_parser().parse_args()
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {model_path}")
    if PPO is None:
        raise ImportError("stable-baselines3 is not installed in this environment.")

    config = load_train_config(model_path)
    print(f"[STAGE1_REPLAY] Loading policy: {model_path}", flush=True)
    if config:
        print(
            "[STAGE1_REPLAY] Using train_config with num_envs forced to 1: "
            f"target_asset={config_value(config, 'target_asset', 'starlink')}, "
            f"visual_obs={replay_visual_obs(args, config)}, "
            f"target_noise={config_value(config, 'target_position_noise', False)}",
            flush=True,
        )

    configure_replay_runtime(args)
    env = make_replay_env(args, config)
    task = unwrap_stage1_task(env)
    if task is not None:
        apply_replay_goal_distance_override(task, args)
        task.replay_switch_hold_steps = int(getattr(task, "success_hold_steps", 12))
        task.max_steps = max(1, int(args.max_steps))
        if not args.stop_on_success:
            task.success_hold_steps = task.max_steps + 1
    model = PPO.load(str(model_path), env=env, device=args.device)
    trajectory_dir = resolve_trajectory_dir(args)
    summary_rows = []
    if trajectory_dir is not None:
        print(f"[STAGE1_REPLAY] Saving trajectory CSV files to: {trajectory_dir}", flush=True)

    try:
        for episode_idx in range(max(1, int(args.episodes))):
            obs = env.reset()
            base_xy_noise = apply_replay_base_xy_noise(task, args)
            obs = current_replay_obs(env)
            episode_return = 0.0
            final_info = {}
            first_switch_ready_step = None
            max_steps = max(1, int(args.max_steps))
            trajectory_rows = [
                trajectory_row(
                    step_idx=0,
                    reward=0.0,
                    action=np.zeros(2, dtype=np.float32),
                    info={},
                    task=task,
                )
            ]
            print(
                f"[STAGE1_REPLAY] Episode {episode_idx + 1} / {args.episodes} "
                f"base_xy_noise=[{base_xy_noise[0]: .4f}, {base_xy_noise[1]: .4f}]",
                flush=True,
            )

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

                if (
                    step_idx == 0
                    or done
                    or args.log_interval <= 1
                    or (step_idx + 1) % args.log_interval == 0
                ):
                    print_step(step_idx + 1, reward, action[0], info, task=task)

                hold_count = task_scalar(task, "base_goal_hold_count", 0.0)
                switch_hold_steps = int(
                    getattr(task, "replay_switch_hold_steps", 12) if task is not None else 12
                )
                switch_ready = (
                    scalar_info(info, "dock_pos_error") <= 0.08
                    and scalar_info(info, "base_planar_speed") <= 0.025
                    and hold_count >= switch_hold_steps
                )
                if switch_ready and first_switch_ready_step is None:
                    first_switch_ready_step = step_idx + 1

                if args.sleep > 0.0:
                    time.sleep(args.sleep)
                if done:
                    break

            initial_row = trajectory_rows[0]
            final_row = trajectory_rows[-1]
            summary_rows.append(
                {
                    "episode": episode_idx + 1,
                    "steps": step_idx + 1,
                    "return": episode_return,
                    "base_xy_noise_x": float(base_xy_noise[0]),
                    "base_xy_noise_y": float(base_xy_noise[1]),
                    "initial_base_x": initial_row["base_x"],
                    "initial_base_y": initial_row["base_y"],
                    "initial_base_z": initial_row["base_z"],
                    "final_base_x": final_row["base_x"],
                    "final_base_y": final_row["base_y"],
                    "final_base_z": final_row["base_z"],
                    "desired_base_x": final_row["desired_base_x"],
                    "desired_base_y": final_row["desired_base_y"],
                    "policy_desired_base_x": final_row["policy_desired_base_x"],
                    "policy_desired_base_y": final_row["policy_desired_base_y"],
                    "final_dock_pos_error": final_row["dock_pos_error"],
                    "final_target_distance": final_row["target_distance"],
                    "success": bool(final_info.get("success", False)),
                    "failure": bool(final_info.get("failure", False)),
                    "first_switch_ready_step": first_switch_ready_step,
                }
            )
            if trajectory_dir is not None:
                trajectory_path = trajectory_dir / f"episode_{episode_idx + 1:03d}.csv"
                write_csv(trajectory_path, trajectory_rows)
                print(f"[STAGE1_REPLAY] Saved trajectory: {trajectory_path}", flush=True)

            print(
                "[STAGE1_REPLAY] Episode done: "
                f"steps={step_idx + 1}, "
                f"return={episode_return:.3f}, "
                f"success={bool(final_info.get('success', False))}, "
                f"failure={bool(final_info.get('failure', False))}, "
                f"first_switch_ready_step={first_switch_ready_step}, "
                f"final_dock_error={scalar_info(final_info, 'dock_pos_error'):.5f}",
                flush=True,
            )
        if trajectory_dir is not None:
            summary_path = trajectory_dir / "summary.csv"
            write_csv(summary_path, summary_rows)
            print(f"[STAGE1_REPLAY] Saved trajectory summary: {summary_path}", flush=True)
        save_scene_video(task, args)
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
