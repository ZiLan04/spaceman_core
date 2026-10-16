"""Replay stage-1, stage-2, then finish with fixed IK grasp closure."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import time

try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    import imageio as imageio_legacy
except Exception:
    imageio_legacy = None

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
project_root_path = project_src_path.parent
sys.path.append(str(project_src_path))

import replay_stage1_policy as stage1_replay  # noqa: E402
import replay_stage2_policy as stage2_replay  # noqa: E402
from fixed_three_phase_docking_test import (  # noqa: E402
    as_list,
    build_phase1_base_target,
    axis_rot_mask,
    build_stage2_base_hold_pose,
    compute_down_axis_alignment,
    get_arm_joint_qpos,
    get_cylinder_center_world,
    get_gripper_open_qpos,
    get_gripper_qpos,
    get_link_pose_tensor,
    get_target_grasp_pose,
    get_target_root_pose,
    get_tcp_pose,
    lock_target_pose,
    make_downward_ee_quat,
    move_joints_toward_tcp_ik,
    reset_base_pid_to_current_pose,
    set_base_root_pid_target,
    set_base_root_pose,
    step_scene,
)
from tasks.RL_task_parallel import PPO  # noqa: E402
from utils.utils import as_rotation_matrix  # noqa: E402


DEFAULT_OUTPUT_DIR = project_src_path / "recordings" / "three_phase_policy_replay"
DEFAULT_VIDEO_NAME = "three_phase_policy_replay.mp4"
FINGER_ATTACH_LINK_NAMES = (
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
)


def parse_vec2(text: str):
    return stage1_replay.parse_vec2(text)


def parse_vec3(text: str):
    return stage1_replay.parse_vec3(text)


def parse_resolution(text: str):
    return stage1_replay.parse_resolution(text)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Replay stage-1 base PPO, stage-2 arm PPO, then use fixed IK to close on the grasp point."
    )
    parser.add_argument("--stage1-model-path", type=Path, default=stage1_replay.DEFAULT_MODEL_PATH)
    parser.add_argument("--stage2-model-path", type=Path, default=stage2_replay.DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stage1-max-steps", type=int, default=220)
    parser.add_argument("--stage2-max-steps", type=int, default=200)
    parser.add_argument("--interphase-max-steps", type=int, default=1200)
    parser.add_argument("--interphase-base-distance", type=float, default=None)
    parser.add_argument("--interphase-pos-threshold", type=float, default=0.08)
    parser.add_argument("--interphase-speed-threshold", type=float, default=0.025)
    parser.add_argument("--interphase-hold-steps", type=int, default=12)
    parser.add_argument("--interphase-settle-steps", type=int, default=10)
    parser.add_argument("--interphase-target-lookahead", type=float, default=0.06)
    parser.add_argument("--interphase-brake-distance", type=float, default=0.6)
    parser.add_argument("--interphase-brake-setpoint-backoff", type=float, default=2.5)
    parser.add_argument("--interphase-cruise-speed", type=float, default=0.08)
    parser.add_argument("--interphase-brake-accel", type=float, default=0.01)
    parser.add_argument("--interphase-speed-deadband", type=float, default=0.015)
    parser.add_argument("--phase3-max-steps", type=int, default=2500)
    parser.add_argument("--phase3-hold-steps", type=int, default=4)
    parser.add_argument("--phase3-tcp-tol", type=float, default=0.055)
    parser.add_argument("--phase3-orient-distance", type=float, default=10.0)
    parser.add_argument("--phase3-down-alignment-threshold", type=float, default=0.995)
    parser.add_argument("--phase3-close-steps", type=int, default=40)
    parser.add_argument(
        "--phase3-final-gripper-qpos",
        type=float,
        default=0.025,
        help="Final finger qpos target for phase3 closure; keep below full close to avoid over-clamping.",
    )
    parser.add_argument("--phase3-max-joint-delta", type=float, default=0.22)
    parser.add_argument("--phase3-tcp-step-size", type=float, default=0.14)
    parser.add_argument("--phase3-dock-max-steps", type=int, default=900)
    parser.add_argument("--cylinder-offset-xy", type=parse_vec2, default=[-0.57, 0.12])
    parser.add_argument("--base-cylinder-scale", type=float, default=1.0)
    parser.add_argument("--target-cylinder-scale", type=float, default=0.7)
    parser.add_argument("--cylinder-align-tol", type=float, default=0.04)
    parser.add_argument("--cylinder-align-hold-steps", type=int, default=15)
    parser.add_argument(
        "--target-dock-z-offset",
        type=float,
        default=0.05,
        help="Raise the virtually attached TCP/target together by this much before xy cylinder docking.",
    )
    parser.add_argument("--phase3-lift-max-steps", type=int, default=250)
    parser.add_argument("--phase3-lift-tol", type=float, default=0.01)
    parser.add_argument(
        "--attached-follow-z",
        action="store_true",
        help="Make the virtually grasped target follow TCP z as well after the initial lift.",
    )
    parser.add_argument(
        "--phase3-lock-gripper-orientation",
        action="store_true",
        help="Keep the full downward gripper orientation during the post-grasp docking pull.",
    )
    parser.add_argument("--phase3-end-yaw-joint-index", type=int, default=5)
    parser.add_argument(
        "--phase3-free-end-yaw",
        action="store_true",
        help="Allow the final yaw joint to move during fixed phase3.",
    )
    parser.add_argument("--arm-initial-noise", type=float, default=0.03)
    parser.add_argument(
        "--success-orientation-angle-deg",
        type=float,
        default=15.0,
        help="Stage-2 full-quaternion angular success threshold in degrees.",
    )
    parser.add_argument("--ee-down-axis-local", type=parse_vec3, default=None)
    parser.add_argument("--phase3-free-orientation", action="store_true", default=False)
    parser.add_argument(
        "--phase3-lock-orientation",
        dest="phase3_free_orientation",
        action="store_false",
        help="Constrain the end-effector downward orientation during fixed phase3.",
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
    parser.add_argument("--video-name", default=DEFAULT_VIDEO_NAME)
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
        help="Path to vecnormalize.pkl for non-visual replay.",
    )
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Do not load VecNormalize stats for non-visual replay.",
    )
    parser.add_argument("--base-pos-noise", type=parse_vec3, default=(0.05, 0.05, 0.05))
    parser.add_argument("--no-base-pos-noise", action="store_true")
    parser.add_argument("--base-rpy-noise", type=parse_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--enable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_true",
        default=None,
        help="Replay non-visual stage-2 policy with the 7D target pose relative observation.",
    )
    parser.add_argument(
        "--disable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_false",
        help="Replay non-visual stage-2 policy with arm joints only.",
    )
    parser.add_argument("--stage2-base-distance", type=float, default=None)
    parser.add_argument("--stage2-enable-target-collision", action="store_true", default=None)
    parser.add_argument(
        "--stage2-disable-target-collision",
        dest="stage2_enable_target_collision",
        action="store_false",
    )
    parser.add_argument("--stage1-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage1-state-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage1-video-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.set_defaults(
        center_envs_at_origin=None,
        target_position_noise=None,
        enable_visual_obs=None,
        enable_target_relative_obs=None,
        stage2_enable_target_collision=None,
    )
    return parser


def scalar_info(info: dict, key: str, default=np.nan):
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return default


def normalize_tensor(vector: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vector / torch.clamp(torch.norm(vector, dim=-1, keepdim=True), min=eps)


def get_finger_attach_frame(task, ee_down_axis):
    finger1_pos, _ = get_link_pose_tensor(
        task.starlink_manipulator,
        FINGER_ATTACH_LINK_NAMES[0],
        task,
    )
    finger2_pos, _ = get_link_pose_tensor(
        task.starlink_manipulator,
        FINGER_ATTACH_LINK_NAMES[1],
        task,
    )
    frame_pos = 0.5 * (finger1_pos + finger2_pos)
    span_axis = normalize_tensor(finger2_pos - finger1_pos)

    _, tcp_quat = get_tcp_pose(task)
    tcp_rot = as_rotation_matrix(tcp_quat, order="wxyz")
    local_down = torch.as_tensor(ee_down_axis, dtype=task.datatype, device=task.device)
    local_down = normalize_tensor(local_down.reshape(1, 3)).reshape(1, 3, 1)
    down_axis = torch.matmul(tcp_rot, local_down.expand(task.num_envs, 3, 1)).squeeze(-1)
    down_axis = down_axis - span_axis * torch.sum(down_axis * span_axis, dim=-1, keepdim=True)

    fallback_axis = torch.zeros_like(down_axis)
    fallback_axis[:, 2] = 1.0
    fallback_axis = fallback_axis - span_axis * torch.sum(
        fallback_axis * span_axis,
        dim=-1,
        keepdim=True,
    )
    fallback_norm = torch.norm(fallback_axis, dim=-1, keepdim=True)
    alt_axis = torch.zeros_like(down_axis)
    alt_axis[:, 1] = 1.0
    fallback_axis = torch.where(fallback_norm > 1e-4, fallback_axis, alt_axis)
    use_fallback = torch.norm(down_axis, dim=-1, keepdim=True) <= 1e-4
    frame_z = normalize_tensor(torch.where(use_fallback, fallback_axis, down_axis))
    frame_y = normalize_tensor(torch.cross(frame_z, span_axis, dim=-1))
    frame_z = normalize_tensor(torch.cross(span_axis, frame_y, dim=-1))
    frame_rot = torch.stack([span_axis, frame_y, frame_z], dim=-1)
    return frame_pos, frame_rot, finger1_pos, finger2_pos


def capture_finger_attached_target_pose(task, ee_down_axis):
    frame_pos, frame_rot, finger1_pos, finger2_pos = get_finger_attach_frame(task, ee_down_axis)
    target_pos, target_quat = get_target_root_pose(task)
    frame_rot_t = frame_rot.transpose(-1, -2)
    relative_pos = torch.matmul(
        frame_rot_t,
        (target_pos - frame_pos).unsqueeze(-1),
    ).squeeze(-1)
    return {
        "relative_pos": relative_pos.detach().clone(),
        "finger1_pos": finger1_pos.detach().clone(),
        "finger2_pos": finger2_pos.detach().clone(),
        "target_pos": target_pos.detach().clone(),
        "target_quat": target_quat.detach().clone(),
    }


def build_finger_attached_target_pose(task, attachment, ee_down_axis, *, follow_z: bool):
    frame_pos, frame_rot, finger1_pos, finger2_pos = get_finger_attach_frame(task, ee_down_axis)
    target_pos = frame_pos + torch.matmul(
        frame_rot,
        attachment["relative_pos"].unsqueeze(-1),
    ).squeeze(-1)
    if not follow_z:
        target_pos = target_pos.detach().clone()
        target_pos[:, 2] = attachment["target_pos"][:, 2]
    target_quat = attachment["target_quat"]
    return target_pos.detach().clone(), target_quat.detach().clone(), finger1_pos, finger2_pos


def make_stage_args(args, *, video_name: str):
    stage_args = argparse.Namespace(**vars(args))
    stage_args.video_name = video_name
    stage_args.max_steps = args.stage1_max_steps
    stage_args.model_path = args.stage1_model_path
    stage_args.stop_on_success = True
    return stage_args


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


def save_scene_video_to_path(task, video_path: Path, fps: int, label: str) -> Path | None:
    if task is None:
        return None
    cam = getattr(getattr(task, "gsim", None), "cam", None)
    if cam is None:
        print(f"[THREE_PHASE_REPLAY] {label}: no TPV camera exists; no segment saved.", flush=True)
        return None
    if not getattr(cam, "_recorded_imgs", None):
        print(f"[THREE_PHASE_REPLAY] {label}: no recorded frames; no segment saved.", flush=True)
        return None

    video_path.parent.mkdir(parents=True, exist_ok=True)
    cam.stop_recording(save_to_filename=str(video_path), fps=max(1, int(fps)))
    task.gsim.camera_enable_recording = False
    print(f"[THREE_PHASE_REPLAY] saved temporary {label} segment: {video_path}", flush=True)
    return video_path


def concat_video_segments(segment_paths, output_path: Path, fps: int) -> Path | None:
    existing_segments = [Path(path) for path in segment_paths if path and Path(path).exists()]
    if not existing_segments:
        print("[THREE_PHASE_REPLAY] no video segments were available to save.", flush=True)
        return None
    if len(existing_segments) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_segments[0].replace(output_path)
        print(f"[THREE_PHASE_REPLAY] saved replay video: {output_path}", flush=True)
        return output_path
    def preserve_segments():
        preserved = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for index, segment_path in enumerate(existing_segments, start=1):
            preserved_path = output_path.parent / (
                f"{output_path.stem}_segment_{index:02d}{segment_path.suffix or '.mp4'}"
            )
            counter = 1
            while preserved_path.exists():
                preserved_path = output_path.parent / (
                    f"{output_path.stem}_segment_{index:02d}_{counter:02d}"
                    f"{segment_path.suffix or '.mp4'}"
                )
                counter += 1
            segment_path.replace(preserved_path)
            preserved.append(preserved_path)
        print(
            "[THREE_PHASE_REPLAY] saved unmerged video segments: "
            + ", ".join(str(path) for path in preserved),
            flush=True,
        )
        return preserved

    if iio is None and imageio_legacy is None:
        print(
            "[THREE_PHASE_REPLAY] imageio is unavailable; cannot merge segments.",
            flush=True,
        )
        preserve_segments()
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concatenated = False
    try:
        with iio.imopen(output_path, "w", plugin="pyav") as writer:
            writer.init_video_stream("libx264", fps=max(1, int(fps)))
            for segment_path in existing_segments:
                for frame in iio.imiter(segment_path):
                    writer.write_frame(frame)
        concatenated = True
    except Exception as exc:
        if imageio_legacy is None:
            print(
                "[THREE_PHASE_REPLAY] could not concatenate video segments; "
                f"preserving segments instead. error={exc}",
                flush=True,
            )
            preserve_segments()
            return None
        try:
            with imageio_legacy.get_writer(output_path, fps=max(1, int(fps))) as writer:
                for segment_path in existing_segments:
                    reader = imageio_legacy.get_reader(segment_path)
                    try:
                        for frame in reader:
                            writer.append_data(frame)
                    finally:
                        reader.close()
            concatenated = True
        except Exception as legacy_exc:
            print(
                "[THREE_PHASE_REPLAY] could not concatenate video segments; "
                f"preserving segments instead. errors=({exc}; {legacy_exc})",
                flush=True,
            )
            preserve_segments()
            return None
    if not concatenated:
        return None
    for segment_path in existing_segments:
        try:
            segment_path.unlink(missing_ok=True)
        except OSError:
            pass
    print(f"[THREE_PHASE_REPLAY] saved replay video: {output_path}", flush=True)
    return output_path


def capture_task_state(task):
    root_pos, root_quat = task._get_base_pose()
    target_pos, target_quat = get_target_root_pose(task)
    arm_joint = get_arm_joint_qpos(task)
    gripper_qpos = get_gripper_qpos(task)
    return {
        "root_pos": root_pos.detach().clone(),
        "root_quat": root_quat.detach().clone(),
        "target_pos": target_pos.detach().clone(),
        "target_quat": target_quat.detach().clone(),
        "arm_joint": arm_joint.detach().clone(),
        "gripper_qpos": gripper_qpos.detach().clone(),
    }


def save_task_state_npz(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        root_pos=state["root_pos"].detach().cpu().numpy(),
        root_quat=state["root_quat"].detach().cpu().numpy(),
        target_pos=state["target_pos"].detach().cpu().numpy(),
        target_quat=state["target_quat"].detach().cpu().numpy(),
        arm_joint=state["arm_joint"].detach().cpu().numpy(),
        gripper_qpos=state["gripper_qpos"].detach().cpu().numpy(),
    )


def load_task_state_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {key: torch.as_tensor(data[key], dtype=torch.float32) for key in data.files}


def run_stage1(args, model, env, task):
    obs = env.reset()
    base_pos_noise, base_rpy_noise = stage1_replay.apply_replay_base_pose_noise(task, args)
    gym_env = stage1_replay.unwrap_stage1_gym_env(env)
    if gym_env is not None:
        obs = gym_env._get_obs()
    if gym_env is None:
        raise RuntimeError("Could not unwrap the stage-1 gym env.")

    final_info = {}
    first_switch_ready_step = None
    print(
        "[THREE_PHASE_REPLAY] stage1 start "
        f"base_pos_noise=[{base_pos_noise[0]: .4f}, {base_pos_noise[1]: .4f}, "
        f"{base_pos_noise[2]: .4f}] "
        f"base_rpy_noise=[{base_rpy_noise[0]: .4f}, {base_rpy_noise[1]: .4f}, "
        f"{base_rpy_noise[2]: .4f}]",
        flush=True,
    )
    for step_idx in range(max(1, int(args.stage1_max_steps))):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        task.step(action=action)
        obs = gym_env._get_obs()
        reward = float(task.last_step_reward.reshape(-1)[0].detach().cpu().item())
        done = bool(
            (task.get_done_mask() | task.get_truncated_mask()).reshape(-1)[0].detach().cpu().item()
        )
        infos = task.get_info()
        info = dict(infos[0]) if infos else {}
        final_info = info

        if (
            step_idx == 0
            or done
            or args.log_interval <= 1
            or (step_idx + 1) % args.log_interval == 0
        ):
            stage1_replay.print_step(step_idx + 1, reward, action[0], info, task=task)

        hold_count = stage1_replay.task_scalar(task, "base_goal_hold_count", 0.0)
        switch_hold_steps = int(getattr(task, "replay_switch_hold_steps", 12))
        switch_ready = (
            scalar_info(info, "dock_pos_error") <= 0.08
            and scalar_info(info, "dock_attitude_error") <= 0.12
            and scalar_info(info, "base_linear_speed") <= 0.025
            and scalar_info(info, "base_angular_speed") <= 0.05
            and hold_count >= switch_hold_steps
        )
        if switch_ready and first_switch_ready_step is None:
            first_switch_ready_step = step_idx + 1
        if args.sleep > 0.0:
            time.sleep(args.sleep)
        if done:
            break

    success = bool(final_info.get("success", False)) and first_switch_ready_step is not None
    print(
        "[THREE_PHASE_REPLAY] stage1 done "
        f"steps={step_idx + 1} success={success} "
        f"failure={bool(final_info.get('failure', False))} "
        f"first_switch_ready_step={first_switch_ready_step} "
        f"final_dock_error={scalar_info(final_info, 'dock_pos_error'):.5f}",
        flush=True,
    )
    return success, capture_task_state(task)


def run_stage1_worker(args, stage1_model_path: Path):
    if args.stage1_state_output is None:
        raise ValueError("--stage1-state-output is required for the hidden stage1 worker.")

    stage1_config = stage1_replay.load_train_config(stage1_model_path)
    stage1_args = make_stage_args(args, video_name="stage1_temp.mp4")
    stage1_args.model_path = stage1_model_path
    stage1_replay.configure_replay_runtime(stage1_args)
    stage1_env = stage1_replay.make_replay_env(stage1_args, stage1_config)
    stage1_task = stage1_replay.unwrap_stage1_task(stage1_env)
    if stage1_task is not None:
        stage1_task.replay_switch_hold_steps = int(getattr(stage1_task, "success_hold_steps", 12))
        stage1_task.max_steps = max(1, int(args.stage1_max_steps))
    stage1_model = PPO.load(str(stage1_model_path), env=stage1_env, device=args.device)

    stage1_success = False
    try:
        stage1_success, stage1_state = run_stage1(
            stage1_args,
            stage1_model,
            stage1_env,
            stage1_task,
        )
        save_task_state_npz(args.stage1_state_output, stage1_state)
        print(f"[THREE_PHASE_REPLAY] saved stage1 state: {args.stage1_state_output}", flush=True)
        if not stage1_success:
            raise RuntimeError(
                "Stage-1 replay did not satisfy the switch success condition; ending early."
            )
    finally:
        if args.stage1_video_output is not None:
            save_scene_video_to_path(
                stage1_task,
                args.stage1_video_output,
                args.video_fps,
                "stage1",
            )
        stage1_env.close()


def run_stage1_subprocess(args, episode_idx: int, video_path: Path) -> dict:
    with tempfile.NamedTemporaryFile(
        prefix=f"three_phase_stage1_ep{episode_idx + 1}_",
        suffix=".npz",
        dir="/tmp",
        delete=False,
    ) as state_file:
        state_path = Path(state_file.name)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--stage1-worker",
        "--stage1-state-output",
        str(state_path),
    ]
    if video_path is not None:
        cmd.extend(["--stage1-video-output", str(video_path)])
    print("[THREE_PHASE_REPLAY] launching stage1 worker process", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return load_task_state_npz(state_path)
    finally:
        try:
            state_path.unlink(missing_ok=True)
        except OSError:
            pass


def apply_stage1_state_to_stage2(task, state):
    state = {
        key: torch.as_tensor(value, dtype=task.datatype, device=task.device)
        for key, value in state.items()
    }
    locked_target_pose = (state["target_pos"], state["target_quat"])
    lock_target_pose(task, locked_target_pose)
    set_base_root_pose(task, state["root_pos"], state["root_quat"])
    task.stage2_base_hold_pos = state["root_pos"].detach().clone()
    task.stage2_base_hold_quat = state["root_quat"].detach().clone()
    task.initial_base_pos = task.stage2_base_hold_pos.detach().clone()
    task.initial_base_quat = task.stage2_base_hold_quat.detach().clone()
    reset_base_pid_to_current_pose(task)
    task.starlink_manipulator.robot.set_dofs_position(
        state["arm_joint"],
        dofs_idx_local=task.starlink_manipulator.motors_dof,
        zero_velocity=True,
    )
    task.starlink_manipulator.robot.set_dofs_position(
        state["gripper_qpos"],
        dofs_idx_local=task.starlink_manipulator.fingers_dof,
        zero_velocity=True,
    )
    task.starlink_manipulator.update_state()
    task.last_action_tensor.zero_()
    task._update_io_tensors()
    if task.target_hold_count is not None:
        task.target_hold_count[:] = 0
    if task.prev_in_hold_zone is not None:
        task.prev_in_hold_zone[:] = False
    if task.ever_in_hold_zone is not None:
        task.ever_in_hold_zone[:] = False
    task.reached_approach_milestone[:] = False
    task.reached_prepose_milestone[:] = False
    task.reached_midpose_milestone[:] = False
    task.reached_nearpose_milestone[:] = False
    task.reached_precision_milestone[:] = False
    for episode_reward_sum in task.episode_reward_sums.values():
        episode_reward_sum.zero_()
    task.episode_min_tcp_to_grasp_error[:] = float("inf")
    task.episode_min_pose_orientation_angle_deg[:] = float("inf")
    task.episode_max_hold_count[:] = 0
    task.prev_tcp_to_grasp_error = task.tcp_to_grasp_error.detach().clone()
    task.prev_pose_orientation_angle = task.pose_orientation_angle.detach().clone()
    task.prev_tcp_pos = task._get_tcp_pose()[0].detach().clone()
    task.prev_reference_link_pos = (
        task._get_reference_yaw_link_pose()[0].detach().clone()
    )
    task.tcp_speed.zero_()
    task.reference_link_speed.zero_()
    task.tool_speed.zero_()
    return locked_target_pose


def run_interphase_base_pid(args, task, locked_target_pose):
    env_idx = 0
    arm_hold_qpos = get_arm_joint_qpos(task)
    gripper_open_qpos = get_gripper_open_qpos(task)
    open_cmd = torch.full(
        (task.num_envs,),
        task.gripper_open_value,
        dtype=task.datatype,
        device=task.device,
    )

    root_pos, _ = task._get_base_pose()
    grasp_pos, grasp_quat = get_target_grasp_pose(task)
    base_distance = (
        float(args.interphase_base_distance)
        if args.interphase_base_distance is not None
        else float(getattr(task, "stage2_base_distance", 1.1))
    )
    hold_root_pos, hold_root_quat = build_stage2_base_hold_pose(
        task,
        grasp_pos,
        grasp_quat,
        base_distance,
    )
    task.stage2_base_hold_pos = hold_root_pos.detach().clone()
    task.stage2_base_hold_quat = hold_root_quat.detach().clone()
    task.initial_base_pos = task.stage2_base_hold_pos.detach().clone()
    task.initial_base_quat = task.stage2_base_hold_quat.detach().clone()

    print(
        "[THREE_PHASE_REPLAY] interphase PID start "
        f"target_distance={base_distance:.3f} "
        f"start_base_xyz={as_list(root_pos[env_idx])} "
        f"target_base_xyz={as_list(hold_root_pos[env_idx])} "
        f"lookahead={args.interphase_target_lookahead} "
        f"brake_distance={args.interphase_brake_distance} "
        f"brake_backoff={args.interphase_brake_setpoint_backoff} "
        f"cruise_speed={args.interphase_cruise_speed} "
        f"brake_accel={args.interphase_brake_accel}",
        flush=True,
    )

    start_root_xy = root_pos[:, :2].detach().clone()
    prev_root_xy = root_pos[:, :2].detach().clone()
    base_velocity_xy = torch.zeros_like(prev_root_xy)
    hold_count = 0
    sim_dt = float(getattr(task, "control_dt", 3e-3))
    done = False
    for step_idx in range(1, max(1, int(args.interphase_max_steps)) + 1):
        root_before, _ = task._get_base_pose()
        pid_target_pos, remaining_to_goal, approach_speed, speed_limit = build_phase1_base_target(
            root_before,
            base_velocity_xy,
            start_root_xy,
            hold_root_pos,
            lookahead=args.interphase_target_lookahead,
            brake_distance=args.interphase_brake_distance,
            brake_setpoint_backoff=args.interphase_brake_setpoint_backoff,
            cruise_speed=args.interphase_cruise_speed,
            brake_accel=args.interphase_brake_accel,
            speed_deadband=args.interphase_speed_deadband,
        )
        set_base_root_pid_target(task, pid_target_pos, hold_root_quat)
        step_scene(
            task,
            joint_target=arm_hold_qpos,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=1,
        )
        root_pos, _ = task._get_base_pose()
        grasp_pos, _ = get_target_grasp_pose(task)
        base_delta_xy = root_pos[:, :2] - prev_root_xy
        base_velocity_xy = base_delta_xy / max(sim_dt, 1e-6)
        base_speed = torch.norm(base_velocity_xy, dim=-1)
        prev_root_xy = root_pos[:, :2].detach().clone()
        dock_pos_error = torch.norm(root_pos[:, :2] - hold_root_pos[:, :2], dim=-1)
        root_to_grasp_distance = torch.norm(root_pos[:, :2] - grasp_pos[:, :2], dim=-1)
        stable_stop = (
            (dock_pos_error <= float(args.interphase_pos_threshold))
            & (base_speed <= float(args.interphase_speed_threshold))
        )
        if bool(torch.all(stable_stop).item()):
            hold_count += 1
        else:
            hold_count = 0

        if (
            step_idx == 1
            or step_idx % max(1, int(args.log_interval)) == 0
            or hold_count > 0
        ):
            print(
                "[THREE_PHASE_REPLAY][interphase]",
                f"step={step_idx}",
                f"base_xyz={as_list(root_pos[env_idx])}",
                f"grasp_xyz={as_list(grasp_pos[env_idx])}",
                f"distance={float(root_to_grasp_distance[env_idx].item()):.5f}",
                f"dock_error={float(dock_pos_error[env_idx].item()):.5f}",
                f"base_speed={float(base_speed[env_idx].item()):.5f}",
                f"approach_speed={float(approach_speed[env_idx].item()):.5f}",
                f"speed_limit={float(speed_limit[env_idx].item()):.5f}",
                f"remaining={float(remaining_to_goal[env_idx].item()):.5f}",
                f"pid_target_xy={as_list(pid_target_pos[env_idx, :2])}",
                f"hold={hold_count}/{args.interphase_hold_steps}",
                flush=True,
            )
        if hold_count >= max(1, int(args.interphase_hold_steps)):
            done = True
            break
        if args.sleep > 0.0:
            time.sleep(args.sleep)

    if not done:
        return False

    pre_stop_root_pos, _ = task._get_base_pose()
    pre_stop_error = torch.norm(pre_stop_root_pos[:, :2] - hold_root_pos[:, :2], dim=-1)
    set_base_root_pose(task, hold_root_pos, hold_root_quat)
    set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
    if int(args.interphase_settle_steps) > 0:
        step_scene(
            task,
            joint_target=arm_hold_qpos,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=args.interphase_settle_steps,
        )
        set_base_root_pose(task, hold_root_pos, hold_root_quat)
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
    task.stage2_base_hold_pos = hold_root_pos.detach().clone()
    task.stage2_base_hold_quat = hold_root_quat.detach().clone()
    task.initial_base_pos = task.stage2_base_hold_pos.detach().clone()
    task.initial_base_quat = task.stage2_base_hold_quat.detach().clone()
    if task.target_hold_count is not None:
        task.target_hold_count[:] = 0
    task.reached_approach_milestone[:] = False
    task.reached_prepose_milestone[:] = False
    task.reached_midpose_milestone[:] = False
    task.reached_nearpose_milestone[:] = False
    task.reached_precision_milestone[:] = False
    for episode_reward_sum in task.episode_reward_sums.values():
        episode_reward_sum.zero_()
    task.episode_min_tcp_to_grasp_error[:] = float("inf")
    task.episode_min_pose_orientation_angle_deg[:] = float("inf")
    task.episode_max_hold_count[:] = 0
    task.prev_tcp_pos = None
    task.prev_reference_link_pos = None
    task.last_action_tensor.zero_()
    task._update_io_tensors()
    task.prev_tcp_to_grasp_error = task.tcp_to_grasp_error.detach().clone()
    task.prev_pose_orientation_angle = task.pose_orientation_angle.detach().clone()
    task.tcp_speed.zero_()
    task.reference_link_speed.zero_()
    task.tool_speed.zero_()
    root_pos, _ = task._get_base_pose()
    final_error = torch.norm(root_pos[:, :2] - hold_root_pos[:, :2], dim=-1)
    print(
        "[THREE_PHASE_REPLAY] interphase PID done "
        f"pre_stop_base_xyz={as_list(pre_stop_root_pos[env_idx])} "
        f"pre_stop_dock_error={float(pre_stop_error[env_idx].item()):.5f} "
        f"final_base_xyz={as_list(root_pos[env_idx])} "
        f"final_dock_error={float(final_error[env_idx].item()):.5f}",
        flush=True,
    )
    return True


def run_stage2(args, model, env, task, stage1_state):
    obs = env.reset()
    locked_target_pose = apply_stage1_state_to_stage2(task, stage1_state)
    gym_env = getattr(getattr(env, "venv", env), "env", None)
    if gym_env is not None and hasattr(gym_env, "_get_obs"):
        interphase_success = run_interphase_base_pid(args, task, locked_target_pose)
        if not interphase_success:
            raise RuntimeError(
                "Interphase PID did not satisfy the stage-1-style switch condition; ending early."
            )
        obs = gym_env._get_obs()
    else:
        raise RuntimeError("Could not unwrap the stage-2 gym env.")

    final_info = {}
    first_stable_hold_step = None
    max_hold_count = 0
    print("[THREE_PHASE_REPLAY] stage2 start from stage1 final state", flush=True)
    for step_idx in range(max(1, int(args.stage2_max_steps))):
        lock_target_pose(task, locked_target_pose)
        action, _ = model.predict(obs, deterministic=args.deterministic)
        task.step(action=action)
        lock_target_pose(task, locked_target_pose)
        obs = gym_env._get_obs()
        reward = float(task.last_step_reward.reshape(-1)[0].detach().cpu().item())
        done = bool(
            (task.get_done_mask() | task.get_truncated_mask()).reshape(-1)[0].detach().cpu().item()
        )
        infos = task.get_info()
        info = dict(infos[0]) if infos else {}
        final_info = info

        hold_count = int(scalar_info(info, "target_hold_count", 0.0))
        max_hold_count = max(max_hold_count, hold_count)
        replay_success_hold_steps = int(getattr(task, "replay_success_hold_steps", 10))
        if hold_count >= replay_success_hold_steps and first_stable_hold_step is None:
            first_stable_hold_step = step_idx + 1

        if (
            step_idx == 0
            or done
            or args.log_interval <= 1
            or (step_idx + 1) % args.log_interval == 0
        ):
            stage2_replay.print_step(step_idx + 1, reward, action[0], info, task=task)

        if args.sleep > 0.0:
            time.sleep(args.sleep)
        if done:
            break

    success = bool(final_info.get("success", False)) and first_stable_hold_step is not None
    print(
        "[THREE_PHASE_REPLAY] stage2 done "
        f"steps={step_idx + 1} success={success} "
        f"first_stable_hold_step={first_stable_hold_step} "
        f"max_hold_count={max_hold_count} "
        f"final_pose_err={scalar_info(final_info, 'pose_position_error'):.5f} "
        f"final_quat_sim={scalar_info(final_info, 'pose_quat_similarity'):.5f} "
        f"final_angle_err={np.degrees(scalar_info(final_info, 'pose_orientation_angle')):.2f}deg",
        flush=True,
    )
    return success, locked_target_pose


def run_fixed_phase3(args, task, locked_target_pose):
    env_idx = 0
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
    hold_root_pos, hold_root_quat = reset_base_pid_to_current_pose(task)
    ee_down_axis = args.ee_down_axis_local
    if ee_down_axis is None:
        ee_down_axis = as_list(task.tcp_offset_local, digits=6)
    ee_down_quat = make_downward_ee_quat(task, ee_down_axis)
    rot_mask = None if args.phase3_free_orientation else axis_rot_mask(ee_down_axis)
    end_yaw_joint_index = int(args.phase3_end_yaw_joint_index)
    phase3_joint_reference = get_arm_joint_qpos(task)
    if end_yaw_joint_index < 0:
        end_yaw_joint_index += phase3_joint_reference.shape[-1]
    if end_yaw_joint_index < 0 or end_yaw_joint_index >= phase3_joint_reference.shape[-1]:
        raise IndexError(
            f"Phase3 end yaw joint index {args.phase3_end_yaw_joint_index} is outside "
            f"joint target shape {tuple(phase3_joint_reference.shape)}."
        )

    print(
        "[THREE_PHASE_REPLAY] phase3 fixed IK start "
        f"tol={args.phase3_tcp_tol} hold_steps={args.phase3_hold_steps} "
        f"orient_distance={args.phase3_orient_distance} "
        f"down_alignment_threshold={args.phase3_down_alignment_threshold} "
        f"rot_mask={rot_mask} "
        f"max_joint_delta={args.phase3_max_joint_delta} "
        f"end_yaw_locked={not args.phase3_free_end_yaw} "
        f"end_yaw_index={end_yaw_joint_index}",
        flush=True,
    )
    hold_count = 0
    phase3_done = False
    for step_idx in range(1, max(1, int(args.phase3_max_steps)) + 1):
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        physical_grasp_pos, _ = get_target_grasp_pose(task)
        grasp_target_pos = physical_grasp_pos.detach().clone()
        tcp_pos_before, _ = get_tcp_pose(task)
        tcp_error_before = torch.norm(tcp_pos_before - grasp_target_pos, dim=-1)
        lock_phase3_orientation = (
            (not args.phase3_free_orientation)
            and bool(torch.all(tcp_error_before <= float(args.phase3_orient_distance)).item())
        )
        current_joint = get_arm_joint_qpos(task)
        joint_target = move_joints_toward_tcp_ik(
            task,
            grasp_target_pos,
            ee_down_quat if lock_phase3_orientation else None,
            current_joint,
            max_joint_delta=args.phase3_max_joint_delta,
            rot_mask=rot_mask if lock_phase3_orientation else None,
        )
        raw_end_yaw_target = joint_target[:, end_yaw_joint_index].detach().clone()
        if not args.phase3_free_end_yaw:
            joint_target = joint_target.detach().clone()
            joint_target[:, end_yaw_joint_index] = phase3_joint_reference[:, end_yaw_joint_index]
        step_scene(
            task,
            joint_target=joint_target,
            gripper_value=open_cmd,
            gripper_qpos_target=gripper_open_qpos,
            locked_target_pose=locked_target_pose,
            steps=1,
        )
        tcp_pos, tcp_quat = get_tcp_pose(task)
        physical_grasp_pos, _ = get_target_grasp_pose(task)
        grasp_target_pos = physical_grasp_pos.detach().clone()
        tcp_error = torch.norm(tcp_pos - grasp_target_pos, dim=-1)
        physical_tcp_error = torch.norm(tcp_pos - physical_grasp_pos, dim=-1)
        down_axis_world, down_alignment = compute_down_axis_alignment(
            tcp_quat,
            ee_down_axis,
            task,
        )
        if args.phase3_free_orientation:
            orientation_ok = torch.ones_like(tcp_error, dtype=torch.bool)
        else:
            orientation_ok = down_alignment >= float(args.phase3_down_alignment_threshold)
        phase3_stable = (tcp_error <= float(args.phase3_tcp_tol)) & orientation_ok
        if bool(torch.all(phase3_stable).item()):
            hold_count += 1
        else:
            hold_count = 0
        if step_idx == 1 or step_idx % max(1, int(args.log_interval)) == 0 or hold_count > 0:
            print(
                "[THREE_PHASE_REPLAY][phase3]",
                f"step={step_idx}",
                f"tcp_xyz={as_list(tcp_pos[env_idx])}",
                f"phase3_target_xyz={as_list(grasp_target_pos[env_idx])}",
                f"physical_grasp_xyz={as_list(physical_grasp_pos[env_idx])}",
                f"tcp_error={float(tcp_error[env_idx].item()):.5f}",
                f"physical_tcp_error={float(physical_tcp_error[env_idx].item()):.5f}",
                f"down_axis_world={as_list(down_axis_world[env_idx])}",
                f"down_alignment={float(down_alignment[env_idx].item()):.5f}",
                f"lock_orientation={lock_phase3_orientation}",
                f"end_yaw_current={float(current_joint[env_idx, end_yaw_joint_index].item()):.5f}",
                f"end_yaw_raw_target={float(raw_end_yaw_target[env_idx].item()):.5f}",
                f"end_yaw_target={float(joint_target[env_idx, end_yaw_joint_index].item()):.5f}",
                f"end_yaw_ref={float(phase3_joint_reference[env_idx, end_yaw_joint_index].item()):.5f}",
                f"hold={hold_count}/{args.phase3_hold_steps}",
                flush=True,
            )
        if hold_count >= max(1, int(args.phase3_hold_steps)):
            phase3_done = True
            break
        if args.sleep > 0.0:
            time.sleep(args.sleep)

    if not phase3_done:
        return False

    final_joint = get_arm_joint_qpos(task)
    if not args.phase3_free_end_yaw:
        final_joint = final_joint.detach().clone()
        final_joint[:, end_yaw_joint_index] = phase3_joint_reference[:, end_yaw_joint_index]
    final_gripper_qpos = torch.full(
        (task.num_envs, gripper_open_qpos.shape[-1]),
        float(args.phase3_final_gripper_qpos),
        dtype=task.datatype,
        device=task.device,
    )
    for _ in range(max(1, int(args.phase3_close_steps))):
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        step_scene(
            task,
            joint_target=final_joint,
            gripper_value=None,
            gripper_qpos_target=final_gripper_qpos,
            locked_target_pose=locked_target_pose,
            steps=1,
        )
    tcp_pos, _ = get_tcp_pose(task)
    physical_grasp_pos, _ = get_target_grasp_pose(task)
    grasp_target_pos = physical_grasp_pos.detach().clone()
    gripper_qpos = get_gripper_qpos(task)
    print(
        "[THREE_PHASE_REPLAY] phase3 grasp closed "
        f"final_tcp_error={float(torch.norm(tcp_pos - grasp_target_pos, dim=-1)[env_idx].item()):.5f} "
        f"final_physical_tcp_error={float(torch.norm(tcp_pos - physical_grasp_pos, dim=-1)[env_idx].item()):.5f} "
        f"gripper_qpos={as_list(gripper_qpos[env_idx])}",
        flush=True,
    )

    print(
        "[THREE_PHASE_REPLAY] phase3 docking start "
        f"target_dock_z_offset={args.target_dock_z_offset} "
        f"cylinder_offset_xy={args.cylinder_offset_xy} "
        f"base_cylinder_scale={args.base_cylinder_scale} "
        f"target_cylinder_scale={args.target_cylinder_scale} "
        f"cylinder_align_tol={args.cylinder_align_tol} "
        f"tcp_step_size={args.phase3_tcp_step_size} "
        f"attached_follow_z={args.attached_follow_z} "
        f"lock_gripper_orientation={args.phase3_lock_gripper_orientation}",
        flush=True,
    )

    finger_attachment = capture_finger_attached_target_pose(task, ee_down_axis)
    attached_target_pos, attached_target_quat, finger1_pos, finger2_pos = (
        build_finger_attached_target_pose(
            task,
            finger_attachment,
            ee_down_axis,
            follow_z=True,
        )
    )
    attached_target_pose = (attached_target_pos, attached_target_quat)
    lock_target_pose(task, attached_target_pose)

    lift_done = abs(float(args.target_dock_z_offset)) <= float(args.phase3_lift_tol)
    lift_start_tcp_pos = tcp_pos.detach().clone()
    lift_tcp_target = lift_start_tcp_pos.detach().clone()
    lift_tcp_target[:, 2] += float(args.target_dock_z_offset)
    phase3_ee_quat = ee_down_quat if args.phase3_lock_gripper_orientation else None
    phase3_rot_mask = [True, True, True] if args.phase3_lock_gripper_orientation else None
    for lift_step_idx in range(1, max(1, int(args.phase3_lift_max_steps)) + 1):
        if lift_done:
            break
        set_base_root_pid_target(task, hold_root_pos, hold_root_quat)
        current_joint = get_arm_joint_qpos(task)
        lift_joint_target = move_joints_toward_tcp_ik(
            task,
            lift_tcp_target,
            phase3_ee_quat,
            current_joint,
            max_joint_delta=args.phase3_max_joint_delta,
            rot_mask=phase3_rot_mask,
        )
        step_scene(
            task,
            joint_target=lift_joint_target,
            gripper_value=close_cmd,
            locked_target_pose=attached_target_pose,
            steps=1,
        )
        attached_target_pos, attached_target_quat, finger1_pos, finger2_pos = (
            build_finger_attached_target_pose(
                task,
                finger_attachment,
                ee_down_axis,
                follow_z=True,
            )
        )
        attached_target_pose = (attached_target_pos, attached_target_quat)
        lock_target_pose(task, attached_target_pose)
        tcp_now, _ = get_tcp_pose(task)
        lift_error = torch.abs(tcp_now[:, 2] - lift_tcp_target[:, 2])
        lift_delta_z = tcp_now[:, 2] - lift_start_tcp_pos[:, 2]
        if (
            lift_step_idx == 1
            or lift_step_idx % max(1, int(args.log_interval)) == 0
            or bool(torch.all(lift_error <= float(args.phase3_lift_tol)).item())
        ):
            target_root_now, _ = get_target_root_pose(task)
            print(
                "[THREE_PHASE_REPLAY][phase3_lift]",
                f"step={lift_step_idx}",
                f"tcp_xyz={as_list(tcp_now[env_idx])}",
                f"target_root_xyz={as_list(target_root_now[env_idx])}",
                f"finger1_xyz={as_list(finger1_pos[env_idx])}",
                f"finger2_xyz={as_list(finger2_pos[env_idx])}",
                f"lift_delta_z={float(lift_delta_z[env_idx].item()):.5f}",
                f"lift_error={float(lift_error[env_idx].item()):.5f}",
                f"target_z_offset={args.target_dock_z_offset}",
                f"phase3_rot_mask={phase3_rot_mask}",
                flush=True,
            )
        if bool(torch.all(lift_error <= float(args.phase3_lift_tol)).item()):
            lift_done = True
            break
        if args.sleep > 0.0:
            time.sleep(args.sleep)

    if not lift_done:
        return False

    finger_attachment = capture_finger_attached_target_pose(task, ee_down_axis)
    attached_target_pos, attached_target_quat, finger1_pos, finger2_pos = (
        build_finger_attached_target_pose(
            task,
            finger_attachment,
            ee_down_axis,
            follow_z=args.attached_follow_z,
        )
    )
    attached_target_pose = (attached_target_pos, attached_target_quat)
    lock_target_pose(task, attached_target_pose)

    dock_done = False
    cylinder_hold_count = 0
    max_dock_steps = max(1, int(args.phase3_dock_max_steps))
    for dock_step_idx in range(1, max_dock_steps + 1):
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
        tcp_now, _ = get_tcp_pose(task)
        tcp_delta = torch.zeros_like(tcp_now)
        tcp_delta[:, :2] = cylinder_error_xy
        tcp_step_norm = torch.norm(tcp_delta, dim=-1, keepdim=True)
        tcp_step_scale = torch.clamp(
            float(args.phase3_tcp_step_size) / torch.clamp(tcp_step_norm, min=1e-6),
            max=1.0,
        )
        pull_tcp_target = tcp_now + tcp_delta * tcp_step_scale
        current_joint = get_arm_joint_qpos(task)
        pull_joint_target = move_joints_toward_tcp_ik(
            task,
            pull_tcp_target,
            phase3_ee_quat,
            current_joint,
            max_joint_delta=args.phase3_max_joint_delta,
            rot_mask=phase3_rot_mask,
        )
        step_scene(
            task,
            joint_target=pull_joint_target,
            gripper_value=close_cmd,
            locked_target_pose=attached_target_pose,
            steps=1,
        )
        attached_target_pos, attached_target_quat, finger1_pos, finger2_pos = (
            build_finger_attached_target_pose(
                task,
                finger_attachment,
                ee_down_axis,
                follow_z=args.attached_follow_z,
            )
        )
        attached_target_pose = (attached_target_pos, attached_target_quat)
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

        if (
            dock_step_idx == 1
            or dock_step_idx % max(1, int(args.log_interval)) == 0
            or cylinder_hold_count > 0
            or dock_step_idx == max_dock_steps
        ):
            target_now, _ = get_target_grasp_pose(task)
            tcp_now, _ = get_tcp_pose(task)
            target_root_now, _ = get_target_root_pose(task)
            print(
                "[THREE_PHASE_REPLAY][phase3_dock]",
                f"step={dock_step_idx}",
                f"tcp_xyz={as_list(tcp_now[env_idx])}",
                f"target_root_xyz={as_list(target_root_now[env_idx])}",
                f"target_grasp_xyz={as_list(target_now[env_idx])}",
                f"finger1_xyz={as_list(finger1_pos[env_idx])}",
                f"finger2_xyz={as_list(finger2_pos[env_idx])}",
                f"base_cylinder_xyz={as_list(base_cylinder_center[env_idx])}",
                f"target_cylinder_xyz={as_list(target_cylinder_center[env_idx])}",
                f"cylinder_xy_error={float(cylinder_error[env_idx].item()):.5f}",
                f"cylinder_z_gap={float(cylinder_z_gap[env_idx].item()):.5f}",
                f"phase3_rot_mask={phase3_rot_mask}",
                f"hold={cylinder_hold_count}/{args.cylinder_align_hold_steps}",
                flush=True,
            )
        if cylinder_hold_count >= max(1, int(args.cylinder_align_hold_steps)):
            dock_done = True
            break
        if args.sleep > 0.0:
            time.sleep(args.sleep)

    final_base, _ = task._get_base_pose()
    final_grasp, _ = get_target_grasp_pose(task)
    final_tcp, _ = get_tcp_pose(task)
    final_target_root, _ = get_target_root_pose(task)
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
    print(
        "[THREE_PHASE_REPLAY] phase3 done "
        f"dock_done={dock_done} "
        f"final_base_xyz={as_list(final_base[env_idx])} "
        f"final_target_root_xyz={as_list(final_target_root[env_idx])} "
        f"final_tcp_to_grasp_error={float(final_tcp_error[env_idx].item()):.5f} "
        f"final_base_hold_drift_xy={float(final_base_drift[env_idx].item()):.5f} "
        f"final_cylinder_xy_error={float(final_cylinder_error[env_idx].item()):.5f} "
        f"final_cylinder_z_gap={float(final_cylinder_z_gap[env_idx].item()):.5f}",
        flush=True,
    )
    return dock_done


def main():
    args = build_arg_parser().parse_args()
    stage1_model_path = args.stage1_model_path.resolve()
    stage2_model_path = args.stage2_model_path.resolve()
    if not stage1_model_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {stage1_model_path}")
    if not stage2_model_path.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint does not exist: {stage2_model_path}")
    if PPO is None:
        raise ImportError("stable-baselines3 is not installed in this environment.")

    if args.stage1_worker:
        run_stage1_worker(args, stage1_model_path)
        return

    stage2_config = stage2_replay.load_train_config(stage2_model_path)

    stage2_args = argparse.Namespace(**vars(args))
    stage2_args.model_path = stage2_model_path
    stage2_args.max_steps = args.stage2_max_steps
    stage2_args.stop_on_success = True
    stage2_args.video_name = "interphase_stage2_phase3_temp.mp4"

    stage2_env = None
    stage2_task = None
    stage2_model = None
    recorded_segments = []
    final_video_path = (
        unique_video_path(args.output_dir, args.video_name)
        if args.record_video
        else None
    )
    try:
        for episode_idx in range(max(1, int(args.episodes))):
            print(
                f"[THREE_PHASE_REPLAY] episode {episode_idx + 1} / {args.episodes}",
                flush=True,
            )
            print(f"[THREE_PHASE_REPLAY] stage1 policy: {stage1_model_path}", flush=True)
            stage1_video_path = None
            if args.record_video:
                stage1_video_path = Path(tempfile.gettempdir()) / (
                    f"three_phase_stage1_ep{episode_idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
                )
            try:
                stage1_state = run_stage1_subprocess(args, episode_idx, stage1_video_path)
            finally:
                if stage1_video_path is not None and stage1_video_path.exists():
                    recorded_segments.append(stage1_video_path)

            if stage2_env is None:
                print(f"[THREE_PHASE_REPLAY] loading stage2 policy: {stage2_model_path}", flush=True)
                stage2_replay.configure_replay_runtime(stage2_args)
                stage2_env = stage2_replay.make_replay_env(stage2_args, stage2_config)
                stage2_task = stage2_replay.unwrap_stage2_task(stage2_env)
                stage2_replay.patch_stage2_replay_reset_pose(stage2_task)
                if stage2_task is not None:
                    stage2_task.stage2_replay_arm_initial_noise = max(
                        0.0, float(args.arm_initial_noise)
                    )
                    stage2_task.pose_orientation_angle_threshold = np.radians(
                        max(0.0, float(args.success_orientation_angle_deg))
                    )
                    stage2_task.pose_quat_similarity_threshold = np.cos(
                        0.5 * stage2_task.pose_orientation_angle_threshold
                    )
                    stage2_task.replay_success_hold_steps = int(
                        getattr(stage2_task, "success_hold_steps", 10)
                    )
                    stage2_task.max_steps = max(1, int(args.stage2_max_steps))
                stage2_model = PPO.load(str(stage2_model_path), env=stage2_env, device=args.device)

            stage2_success, locked_target_pose = run_stage2(
                stage2_args,
                stage2_model,
                stage2_env,
                stage2_task,
                stage1_state,
            )
            if not stage2_success:
                raise RuntimeError(
                    "Stage-2 replay did not satisfy the switch success condition; ending early."
                )
            phase3_success = run_fixed_phase3(stage2_args, stage2_task, locked_target_pose)

            if not phase3_success:
                raise RuntimeError("Fixed phase3 did not finish grasp closure and cylinder docking.")
            print(
                f"[THREE_PHASE_REPLAY] episode {episode_idx + 1} passed",
                flush=True,
            )
    finally:
        if args.record_video and stage2_task is not None:
            stage2_video_path = Path(tempfile.gettempdir()) / (
                f"three_phase_interphase_stage2_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
            )
            saved_stage2_video = save_scene_video_to_path(
                stage2_task,
                stage2_video_path,
                args.video_fps,
                "interphase_stage2_phase3",
            )
            if saved_stage2_video is not None:
                recorded_segments.append(saved_stage2_video)
        if args.record_video and final_video_path is not None:
            concat_video_segments(recorded_segments, final_video_path, args.video_fps)
        if stage2_env is not None:
            stage2_env.close()

    print("[THREE_PHASE_REPLAY] all requested episodes passed", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
