"""Headless 6-DoF stage-1 smoke test.

Builds one stage-1 env and replays scripted base actions (translate along z,
then rotate about z, then hold). No trained policy is required: this exercises
the ``action -> base target pose -> base PID -> base pose`` pathway and checks
that

* the observation/action spaces are the 6-DoF ones,
* the base PID emits force on the z channel,
* the base PID emits torque on the yaw channel,
* the base translates along z and rotates about z.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from configs.registry import get_legacy_camera  # noqa: E402
FRANKA_S_Q_CAMERA = get_legacy_camera("franka_merge")
from tasks.RL_task_parallel import TaskStatus  # noqa: E402
from tasks.RL_task_parallel_stage1 import build_stage1_env  # noqa: E402


def parse_vec3(text: str):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated values, e.g. 0,0,0.05."
        )
    return tuple(values)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage-1 6-DoF base control smoke test."
    )
    parser.add_argument("--steps-per-phase", type=int, default=15)
    parser.add_argument(
        "--translation-action",
        type=parse_vec3,
        default=(0.0, 0.0, 0.05),
        help="Base-frame [dx, dy, dz] command held during the translation phase.",
    )
    parser.add_argument(
        "--rotation-action",
        type=parse_vec3,
        default=(0.0, 0.0, 0.05),
        help="Base-frame [droll, dpitch, dyaw] command held during the rotation phase.",
    )
    parser.add_argument("--enable-visual-obs", action="store_true", default=False)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print the base pose summary every N steps; 0 prints only phase ends.",
    )
    return parser


def state_span(observation_space):
    if hasattr(observation_space, "spaces"):
        return tuple(observation_space["state"].shape)
    return tuple(observation_space.shape)


def pose_summary(task):
    pos, quat = task._get_base_pose()
    grasp_pos, _ = task._get_target_grasp_pose()
    pid_output = task.starlink_manipulator.pid.input_reference()
    force = pid_output.get("position") if isinstance(pid_output, dict) else None
    torque = pid_output.get("orientation") if isinstance(pid_output, dict) else None

    def first(tensor):
        if tensor is None:
            return None
        return [round(float(v), 5) for v in torch.as_tensor(tensor)[0].detach().cpu().tolist()]

    return {
        "base_pos": first(pos),
        "base_euler": first(task.starlink_manipulator.pid.quaternion_to_euler(quat)),
        "linear_velocity": first(task.base_linear_velocity),
        "angular_velocity": first(task.base_angular_velocity),
        "dock_pos_error": round(float(task.dock_pos_error[0].item()), 5),
        "dock_attitude_error": round(float(task.dock_attitude_error[0].item()), 5),
        "grasp_standoff": round(float(torch.norm(pos - grasp_pos, dim=-1)[0].item()), 5),
        "camera_visible": bool(task.camera_visible[0].item()),
        "camera_center_error": round(float(task.camera_center_error[0].item()), 5),
        "failure": bool(task.is_failure()[0].item()),
        "max_steps": int(task.max_steps),
        "target_pose_command": first(task.base_target_position_command),
        "target_quat_command": first(task.base_target_quat_command),
        "pid_setpoint_rpy": first(
            task.starlink_manipulator.pid.setpoint[:, 3:]
            if task.starlink_manipulator.pid.setpoint.dim() > 1
            else task.starlink_manipulator.pid.setpoint[3:]
        ),
        "pid_current_rpy": first(
            task.starlink_manipulator.pid._current_orientation
        ),
        "pid_force": first(force),
        "pid_torque": first(torque),
    }


def main():
    args = build_arg_parser().parse_args()
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not bool(args.no_viewer)
    # Keep the smoke test from writing TPV debug videos into src/recordings.
    TPV_CAM_SETTINGS["enable_recording"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]["enable_recording"] = False

    env = build_stage1_env(
        num_envs=1,
        enable_visual_obs=bool(args.enable_visual_obs),
        target_position_noise=False,
    )
    task = env.task
    try:
        action_shape = tuple(env.single_action_space.shape)
        obs_shape = state_span(env.single_observation_space)
        print(
            f"[STAGE1_6DOF_SMOKE] action_shape={action_shape} state_shape={obs_shape}",
            flush=True,
        )
        assert action_shape == (6,), f"expected a 6-DoF action space, got {action_shape}"
        assert obs_shape == (19,), f"expected a 19-D state vector, got {obs_shape}"

        reset_obs, _ = env.reset()
        actual_state = reset_obs["state"] if isinstance(reset_obs, dict) else reset_obs
        actual_state_shape = tuple(actual_state.shape)
        print(
            f"[STAGE1_6DOF_SMOKE] reset_state_shape={actual_state_shape}",
            flush=True,
        )
        assert actual_state_shape == (task.num_envs, 19), (
            "expected reset to return a batched 19-D state, "
            f"got {actual_state_shape}"
        )
        start = pose_summary(task)
        print(f"[STAGE1_6DOF_SMOKE] reset: {start}", flush=True)

        phases = (
            ("translate", 0, np.asarray(args.translation_action, dtype=np.float32)),
            ("rotate", 3, np.asarray(args.rotation_action, dtype=np.float32)),
            ("hold", 0, np.zeros(3, dtype=np.float32)),
        )
        results = {}
        for name, offset, vector in phases:
            action = np.zeros(6, dtype=np.float32)
            action[offset : offset + vector.size] = vector
            first_step_summary = None
            for step_idx in range(max(1, int(args.steps_per_phase))):
                env.step(action.reshape(1, -1))
                if step_idx == 0:
                    first_step_summary = pose_summary(task)
                if args.log_every > 0 and (step_idx + 1) % args.log_every == 0:
                    print(
                        f"[STAGE1_6DOF_SMOKE] {name} step={step_idx + 1}: "
                        f"{pose_summary(task)}",
                        flush=True,
                    )
            results[name] = pose_summary(task)
            print(
                f"[STAGE1_6DOF_SMOKE] {name} first_step: {first_step_summary}",
                flush=True,
            )
            print(f"[STAGE1_6DOF_SMOKE] {name} end: {results[name]}", flush=True)
            if task.status != TaskStatus.RUNNING:
                print(
                    f"[STAGE1_6DOF_SMOKE] note: episode status became {task.status} "
                    f"during phase '{name}'",
                    flush=True,
                )

        dz = results["translate"]["base_pos"][2] - start["base_pos"][2]
        dyaw = results["rotate"]["base_euler"][2] - start["base_euler"][2]
        translate_force_z = results["translate"]["pid_force"][2]
        rotate_torque_z = results["rotate"]["pid_torque"][2]
        print(
            "[STAGE1_6DOF_SMOKE] checks: "
            f"dz={dz: .5f} m, dyaw={dyaw: .5f} rad, "
            f"force_z_during_translate={translate_force_z}, "
            f"torque_z_during_rotate={rotate_torque_z}",
            flush=True,
        )

        failures = []
        if abs(translate_force_z or 0.0) < 1e-6:
            failures.append("PID produced no force on the z channel")
        if abs(rotate_torque_z or 0.0) < 1e-6:
            failures.append("PID produced no torque on the yaw channel")
        if abs(dz) < 1e-4:
            failures.append("base did not translate along z")
        if abs(dyaw) < 1e-4:
            failures.append("base did not rotate about z")

        if failures:
            print(
                "[STAGE1_6DOF_SMOKE] FAILED: " + "; ".join(failures),
                flush=True,
            )
            return 1
        print("[STAGE1_6DOF_SMOKE] PASSED", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
