"""Stage-1 base-PID long-range bench test.

The stage-1 RL task never lets the low-level base PID see a long position
error: every control step the policy only supplies a *local* offset of at most
``base_translation_action_limit`` (0.3 m / 0.2 rad) and
``ParallelStage1BaseTask._apply_base_action`` turns it into a new PID setpoint
that sits a few tens of centimetres ahead of the measured base pose. At reset
the base stands ``~2.28 m`` from the target grasp point and the stage-1 goal
pose is ``base_goal_threshold`` along the docking axis, so the whole episode
only asks for a ``~1 m`` translation.

This test removes the policy from the loop and asks the low-level controller
to cover a longer distance with the *current* tunings (P=800 N/m,
D=600 N*s/m, +-250 N / +-400 N*m). It mirrors the env plumbing exactly:

* the goal pose is the stage-1 goal (``grasp_point + dock_dir * goal_distance``)
  mapped into the PID link frame with
  ``ParallelStage1BaseTask._root_target_to_pid_link_setpoint``;
* every control step runs ``base_control_decimation`` physics sub-steps of
  ``manipulator.step() -> gsim.step()`` before ``_update_io_tensors()``, so the
  reported speed/error metrics use the real 30 ms control period.
* all six arm joints and both fingers are held at their initial configuration
  by a high-authority position servo, without overwriting simulator state or
  disturbing the free base dynamics.

Four command modes:

``fixed``
    Write the goal setpoint once and hold it. This is the raw PD loop: the
    controller sees the full multi-metre error and both the actuator limits and
    the (low) closed-loop damping show up.

``attitude``
    Hold the starting position and command only the goal attitude, so the
    attitude channel (60/60 gains, +-40 N*m) can be judged on its own, without
    the translation force coupling into it.

``waypoint``
    Pre-compute the whole manoeuvre as a dense track of world-frame target
    poses (one per 30 ms control step) and hand the PID the next waypoint every
    step, exactly like the RL policy does with its local offset commands. The
    track is a straight line / shortest-arc attitude slew with a
    accelerate-then-brake profile derived from the actuator budget
    (``force_limit / mass`` and ``torque_limit / inertia``), scaled down by
    ``--waypoint-brake-frac`` so the controller always keeps authority to stop.
    Because the track is feasible, the PID error stays at centimetre level
    instead of the metres it sees with ``fixed``.

``stepwise``
    Refresh the setpoint every control step from a simple proportional /
    derivative offset command, i.e. the same code path the policy drives
    during training (``_apply_base_action``). The gains are exposed because the
    real policy learns them; here they stand in for "a policy that pushes
    towards the goal and brakes". Note that the command itself is clamped to
    the stage-1 action limits, so this also shows what the current action range
    allows to be covered per step.

The default is ``stepwise`` because that is the mode this test is intended to
validate.  Its fixed final pose deliberately preserves the old planar Stage-1
definition: 1.1 m from the grasp point in the target docking direction, the
base z from reset, and the base attitude from reset.  (The historical planar
task used 1.3 m; this bench test intentionally uses the new 1.1 m value.)

Examples
--------
::

    # default: off-axis 5 m standoff, 6-DoF offset, 700 control steps
    python src/test/stage1_pid_long_range_test.py

    # headless
    python src/test/stage1_pid_long_range_test.py --no-viewer

    # sweep a few standoffs, dump trajectories
    python src/test/stage1_pid_long_range_test.py \
        --start-distances current,3.0,4.0,5.0 --csv /tmp/stage1_pid_sweep.csv

    # judge the gains with a dt that matches the 3 ms physics step
    python src/test/stage1_pid_long_range_test.py --modes fixed --pid-dt 0.003

    # re-tuned loops (gains derived from the measured inertia)
    python src/test/stage1_pid_long_range_test.py --modes waypoint \
        --start-distances current,5.0 --max-control-steps 400 \
        --pid-dt 0.003 --pid-pos-kd 1000 --pid-rot-kp 1700 --pid-rot-kd 2500

Viewer: the window is on by default with the stock camera from
``SCENE_SETTINGS`` (same as the other tests), so ``--no-viewer`` runs headless.
Closing the window stops the run. The simulation is slower than real time
(about one control step per second on the current machine), so the default
700-step case takes several minutes.

Measured rigid-body properties of the stage-1 base (see ``measure_base_inertia``
and the ``run`` output): mass 516.7 kg, I_yy ~ 1177 kg*m^2, force limit +-250 N
(0.484 m/s^2), torque limit +-400 N*m (0.34 rad/s^2). The long-range default
budget is 700 control steps = 21 s; the training task's stock budget remains
220 steps = 6.6 s.

* ``waypoint`` keeps the PID's own error at centimetre level (0.0003 m at a 5 m
  standoff versus 3.95 m for ``fixed``) and does drive the base into the goal
  position band, so pre-computing small targets is the right structure.
* The attitude channel is the limiting factor: both the bandwidth (Kp vs I) and
  the damping (Kd) have to be derived from the inertia, and Kd additionally has
  to stay inside the torque limit - a D term that saturates turns the loop into
  a permanent +/-4 deg rocking limit cycle. See the derivation comment above
  ``FRANKA_S_Q_PID`` in configs/config_starlink_manipulator.py.
* The final attitude equals the reset attitude. The attitude PID still has to
  reject translation/arm coupling and bring angular velocity back to zero.

Treat the exit code as "did every mode settle".
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS  # noqa: E402
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA  # noqa: E402
from tasks.RL_task_parallel_stage1 import build_stage1_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 base PID long-range convergence test."
    )
    parser.add_argument(
        "--start-distances",
        default="5.0",
        help="Comma separated base->grasp standoffs in metres for the episode "
        "start. 'current' keeps the stock reset standoff (~2.28 m).",
    )
    parser.add_argument(
        "--modes",
        default="stepwise",
        help="Comma separated subset of {waypoint, fixed, attitude, stepwise}. "
        "The default is 'stepwise', which exercises the same local-offset "
        "path as Stage-1 RL; 'fixed'/'attitude' are "
        "deliberately un-tuned reference modes and are expected to saturate and "
        "drift, 'stepwise' stands in for the RL policy.",
    )
    parser.add_argument(
        "--goal-distance",
        type=float,
        default=None,
        help="Planar base->grasp distance of the legacy goal pose. Default: "
        "the task's base_goal_threshold (1.1 m).",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=700,
        help="Control steps per run. One control step is "
        "base_control_decimation physics steps (30 ms by default).",
    )
    parser.add_argument(
        "--start-lateral-offset",
        type=float,
        default=0.75,
        help="Initial lateral offset [m] from the nominal approach line while "
        "preserving the requested base->grasp standoff.",
    )
    parser.add_argument(
        "--start-vertical-offset",
        type=float,
        default=0.50,
        help="Initial vertical offset [m] from the nominal approach line while "
        "preserving the requested base->grasp standoff.",
    )
    parser.add_argument(
        "--start-attitude-offset-deg",
        default="8,-10,12",
        help="Body-frame roll,pitch,yaw rotation-vector offset in degrees applied "
        "to the initial base attitude. The goal attitude remains unshifted.",
    )
    parser.add_argument(
        "--no-arm-fixture",
        action="store_true",
        help="Use the normal finite-stiffness arm position controller instead "
        "of fixing all six arm joints and both fingers at their initial values.",
    )
    parser.add_argument(
        "--settle-hold-steps",
        type=int,
        default=12,
        help="Consecutive stable control steps required before a run counts "
        "as settled.",
    )
    parser.add_argument(
        "--pid-dt",
        type=float,
        default=None,
        help="Override the PID's internal dt. The stock config uses 0.01 s "
        "while the controller is actually invoked once per 0.003 s physics "
        "step, so the derivative term counts for only a third of its nominal "
        "value. Pass 0.003 to judge the gains with a consistent dt.",
    )
    parser.add_argument(
        "--pid-rot-kp",
        type=float,
        default=None,
        help="Override the attitude proportional gain (PID channels 3:6).",
    )
    parser.add_argument(
        "--pid-pos-kp",
        type=float,
        default=None,
        help="Override the position proportional gain (PID channels 0:3).",
    )
    parser.add_argument(
        "--pid-pos-kd",
        type=float,
        default=None,
        help="Override the position derivative gain (PID channels 0:3).",
    )
    parser.add_argument(
        "--pid-force-limit",
        type=float,
        default=None,
        help="Override the per-axis linear force limit [N].",
    )
    parser.add_argument(
        "--pid-torque-limit",
        type=float,
        default=None,
        help="Override the per-axis torque limit [N*m].",
    )
    parser.add_argument(
        "--pid-rot-kd",
        type=float,
        default=None,
        help="Override the attitude derivative gain (PID channels 3:6).",
    )
    parser.add_argument(
        "--waypoint-brake-frac",
        type=float,
        default=0.6,
        help="Fraction of the achievable acceleration/angular acceleration the "
        "waypoint track is allowed to use, so the PID keeps margin to brake.",
    )
    parser.add_argument(
        "--waypoint-lead",
        type=float,
        default=0.25,
        help="Look-ahead distance [m] the waypoint track is allowed to run "
        "ahead of the base. The track only advances while the base is within "
        "this distance of the current waypoint, so the PID never chases a "
        "runaway setpoint (0 disables the clamp and replays the track on a "
        "fixed schedule).",
    )
    parser.add_argument(
        "--assumed-inertia",
        type=float,
        default=None,
        help="Base inertia about y in kg*m^2 used to size the waypoint track's "
        "angular profile. Default: measured at startup with a torque impulse.",
    )
    parser.add_argument(
        "--stepwise-kp",
        type=float,
        default=1.0,
        help="Proportional gain of the stand-in policy (offset per metre).",
    )
    parser.add_argument(
        "--stepwise-kd",
        type=float,
        default=1.2,
        help="Derivative gain of the stand-in policy (offset per m/s).",
    )
    parser.add_argument(
        "--stepwise-rot-kp",
        type=float,
        default=1.0,
        help="Attitude proportional gain of the stand-in policy (offset per rad).",
    )
    parser.add_argument(
        "--stepwise-rot-kd",
        type=float,
        default=1.0,
        help="Attitude derivative gain of the stand-in policy "
        "(offset per rad/s).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print a trajectory line every N control steps; 0 disables it.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path; writes one row per control step of every run.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless. The Genesis viewer window is shown by default.",
    )
    return parser.parse_args()


def _configure_cameras():
    """This test never needs the RL/TPV cameras, only the viewer."""
    TPV_CAM_SETTINGS["enable_recording"] = False
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    FRANKA_S_Q_CAMERA["wrist_camera"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enabled"] = False
    FRANKA_S_Q_CAMERA["rl_obs"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["enabled"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["enable_recording"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]["enabled"] = False
    FRANKA_S_Q_CAMERA["base_camera"]["rl_obs"]["enable_recording"] = False


def _configure_viewer(args):
    """Show the stock viewer configured by SCENE_SETTINGS (same camera as the
    other tests); the viewer window is on unless --no-viewer is passed."""
    SCENE_SETTINGS["renderer"] = None
    SCENE_SETTINGS["show_viewer"] = not args.no_viewer


def _parse_start_distances(text):
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        specs.append(item)
    if not specs:
        raise argparse.ArgumentTypeError("--start-distances must not be empty")
    return specs


def _parse_modes(text):
    modes = [item.strip() for item in text.split(",") if item.strip()]
    known = ("waypoint", "fixed", "attitude", "stepwise")
    unknown = [mode for mode in modes if mode not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown mode(s) {unknown}; expected a subset of {'/'.join(known)}"
        )
    return modes


def _parse_vec3(text, option_name):
    try:
        values = [float(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must contain three comma-separated numbers"
        ) from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"{option_name} must contain three comma-separated numbers"
        )
    return values


def _vec3(tensor):
    return [round(float(value), 4) for value in tensor.reshape(-1)[:3].tolist()]


def _vec3_deg(tensor):
    return [round(math.degrees(float(value)), 2) for value in tensor.reshape(-1)[:3].tolist()]


def _as_batch6(tensor):
    tensor = torch.as_tensor(tensor)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_legacy_stage1_goal(task, grasp_pos, grasp_quat, reset_pos, reset_quat, distance):
    """Build the old planar Stage-1 goal as a full 6-DoF fixed pose.

    Git history (e042566) defined only goal x/y from the target docking axis;
    z and attitude were held at their episode-reset values.  Preserve that
    definition while changing the historical 1.3 m distance to 1.1 m.
    """
    from utils.utils import as_rotation_matrix  # noqa: PLC0415

    target_rot = as_rotation_matrix(grasp_quat, order="wxyz")
    dock_dir_world = torch.matmul(
        target_rot, task.target_dock_dir_local.view(3, 1)
    ).squeeze(-1)
    dock_dir_xy = dock_dir_world[:, :2]
    dock_dir_xy = dock_dir_xy / torch.clamp(
        torch.norm(dock_dir_xy, dim=-1, keepdim=True), min=1e-6
    )
    goal_pos = reset_pos.detach().clone()
    goal_pos[:, :2] = grasp_pos[:, :2] + dock_dir_xy * float(distance)
    return goal_pos, reset_quat.detach().clone()


def goal_pose_errors(task, goal_pos, goal_quat):
    """Return position and shortest-arc attitude error to this test's goal."""
    base_pos, base_quat = task._get_base_pose()
    position_error = torch.norm(base_pos - goal_pos, dim=-1)
    error_quat = task._quaternion_multiply_wxyz(
        task._quaternion_conjugate_wxyz(goal_quat), base_quat
    )
    attitude_error = torch.norm(
        task._quaternion_to_rotation_vector_wxyz(error_quat), dim=-1
    )
    return position_error, attitude_error


def configure_arm_fixture(manip):
    """Hold all arm/finger DOFs with a stiff position servo.

    Directly writing qpos every physics step also disturbs the free base state
    in Genesis, so use a high-authority physical joint controller instead.
    """
    dof_indices = list(manip.motors_dof) + list(manip.fingers_dof)
    positions = torch.as_tensor(
        manip.robot.get_dofs_position(dofs_idx_local=dof_indices),
        dtype=manip.datatype,
        device=manip.device,
    ).detach().clone()
    joint_count = len(dof_indices)
    manip.robot.set_dofs_kp([20000.0] * joint_count, dofs_idx_local=dof_indices)
    manip.robot.set_dofs_kv([2000.0] * joint_count, dofs_idx_local=dof_indices)
    manip.robot.set_dofs_force_range(
        [-1000.0] * joint_count,
        [1000.0] * joint_count,
        dofs_idx_local=dof_indices,
    )
    manip.robot.control_dofs_position(positions, dofs_idx_local=dof_indices)
    return dof_indices, positions


def hold_arm_fixture(manip, fixture):
    """Refresh the fixed arm/finger target and return maximum joint error."""
    dof_indices, positions = fixture
    manip.robot.control_dofs_position(positions, dofs_idx_local=dof_indices)
    current = torch.as_tensor(
        manip.robot.get_dofs_position(dofs_idx_local=dof_indices),
        dtype=positions.dtype,
        device=positions.device,
    )
    return float(torch.max(torch.abs(current - positions)).item())


def place_base_at_standoff(task, standoff, lateral_offset, vertical_offset, attitude_offset_deg):
    """Place the base off-axis at an exact grasp-point standoff."""
    grasp_pos, _ = task._get_target_grasp_pose()
    base_pos, base_quat = task._get_base_pose()
    # Stage 1 now resets directly into the fixed offset start. Always construct
    # the scripted test pose from the original scene pose, otherwise each run
    # applies the translation and attitude offsets a second time.
    reference_pos = (
        task.nominal_base_pos if task.nominal_base_pos is not None else base_pos
    )
    reference_quat = (
        task.nominal_base_quat if task.nominal_base_quat is not None else base_quat
    )
    radial_dir = reference_pos - grasp_pos
    radial_dir = radial_dir / torch.clamp(
        torch.norm(radial_dir, dim=-1, keepdim=True), min=1e-8
    )

    world_up = torch.tensor(
        [0.0, 0.0, 1.0], dtype=base_pos.dtype, device=base_pos.device
    ).expand_as(radial_dir)
    lateral_dir = torch.cross(world_up, radial_dir, dim=-1)
    lateral_norm = torch.norm(lateral_dir, dim=-1, keepdim=True)
    fallback = torch.tensor(
        [0.0, 1.0, 0.0], dtype=base_pos.dtype, device=base_pos.device
    ).expand_as(radial_dir)
    lateral_dir = torch.where(lateral_norm > 1e-6, lateral_dir, fallback)
    lateral_dir = lateral_dir / torch.clamp(
        torch.norm(lateral_dir, dim=-1, keepdim=True), min=1e-8
    )
    vertical_dir = torch.cross(radial_dir, lateral_dir, dim=-1)
    transverse_sq = float(lateral_offset) ** 2 + float(vertical_offset) ** 2
    if transverse_sq >= float(standoff) ** 2:
        raise ValueError(
            "start lateral/vertical offsets must be smaller than the standoff"
        )
    radial_distance = math.sqrt(float(standoff) ** 2 - transverse_sq)
    new_pos = (
        grasp_pos
        + radial_dir * radial_distance
        + lateral_dir * float(lateral_offset)
        + vertical_dir * float(vertical_offset)
    )

    attitude_offset_rad = torch.tensor(
        attitude_offset_deg, dtype=base_pos.dtype, device=base_pos.device
    ).view(1, 3) * (math.pi / 180.0)
    new_quat = task._quaternion_multiply_wxyz(
        reference_quat,
        task._rotation_vector_to_quaternion_wxyz(attitude_offset_rad),
    )

    root_qpos = torch.cat([new_pos, new_quat], dim=-1)
    task.starlink_manipulator.robot.set_qpos(
        root_qpos,
        qs_idx_local=list(range(7)),
        zero_velocity=True,
    )
    task.starlink_manipulator.update_state()
    # The placement is a reset operation, not physical motion. Prevent the
    # high-level PD stand-in from interpreting the teleport as velocity.
    task.prev_base_pos = new_pos.detach().clone()
    task.prev_base_quat = new_quat.detach().clone()
    task._update_io_tensors()
    return float(task.base_alignment_error[0].item())


def _stepwise_action(task, goal_pos, goal_quat, args):
    """A stand-in for the RL policy: clamped local offset towards the goal."""
    base_pos, base_quat = task._get_base_pose()
    error_world = goal_pos - base_pos
    error_local = task._world_vector_to_base_local(error_world, base_quat)
    velocity_local = task._world_vector_to_base_local(
        task.base_linear_velocity, base_quat
    )
    offset = args.stepwise_kp * error_local - args.stepwise_kd * velocity_local

    error_quat = task._quaternion_multiply_wxyz(
        task._quaternion_conjugate_wxyz(base_quat), goal_quat
    )
    rotation_error = task._quaternion_to_rotation_vector_wxyz(error_quat)
    angular_velocity_local = task._world_vector_to_base_local(
        task.base_angular_velocity, base_quat
    )
    rotation_offset = (
        args.stepwise_rot_kp * rotation_error
        - args.stepwise_rot_kd * angular_velocity_local
    )
    return torch.cat([offset, rotation_offset], dim=-1)


def measure_base_inertia(task, torque=20.0, steps=200):
    """Torque impulse test: I = tau * t / omega.

    The base PID is disabled for the measurement and the pose is restored
    afterwards, so callers just keep going.
    """
    manip = task.starlink_manipulator
    robot = manip.robot
    dt = float(SCENE_SETTINGS["sim_options"].dt)
    base_pos, base_quat = task._get_base_pose()
    was_enabled = manip.pid_enabled
    manip.set_pid_enabled(False)
    try:
        robot.set_qpos(
            torch.cat([base_pos, base_quat], dim=-1),
            qs_idx_local=list(range(7)),
            zero_velocity=True,
        )
        robot.set_dofs_velocity(torch.zeros(6), dofs_idx_local=list(range(6)))
        task.gsim.step()
        torque_vec = [0.0, 0.0, 0.0]
        torque_vec[1] = float(torque)
        for _ in range(int(steps)):
            manip.apply_force(
                force=None,
                torque=torque_vec,
                link_name=manip.params.get("pid_apply_link", manip.params["base"]),
            )
            task.gsim.step()
        dof_velocity = torch.as_tensor(
            robot.get_dofs_velocity(dofs_idx_local=list(range(6))),
            dtype=torch.float32,
        ).reshape(-1)
        omega = abs(float(dof_velocity[4]))
        inertia = (
            torque * steps * dt / omega if omega > 1e-6 else float("nan")
        )
        robot.set_qpos(
            torch.cat([base_pos, base_quat], dim=-1),
            qs_idx_local=list(range(7)),
            zero_velocity=True,
        )
        task.gsim.step()
    finally:
        manip.set_pid_enabled(was_enabled)
    return inertia


def _profile_progress(t, total, span):
    """Triangular accelerate/brake profile: returns how much of ``span`` is
    covered at time ``t`` of a manoeuvre that lasts ``total`` seconds."""
    if total <= 0.0:
        return span
    if t >= total:
        return span
    accel = 4.0 * span / (total * total)
    if t <= 0.5 * total:
        return 0.5 * accel * t * t
    remaining = total - t
    return span - 0.5 * accel * remaining * remaining


def build_waypoint_track(task, start_pos, goal_pos, start_quat, goal_quat, args):
    """Pre-compute the dense world-frame target track for the ``waypoint`` mode.

    Straight-line translation and a shortest-arc attitude slew, both with an
    accelerate-then-brake profile whose acceleration is a fixed fraction of the
    actuator budget, so the PID can follow it without ever seeing a large error
    and still has authority to stop at the end.
    """
    from utils.utils import as_rotation_matrix  # noqa: PLC0415

    distance = float(torch.norm(goal_pos - start_pos, dim=-1).item())
    direction = (goal_pos - start_pos) / max(distance, 1e-8)

    error_quat = task._quaternion_multiply_wxyz(
        task._quaternion_conjugate_wxyz(start_quat), goal_quat
    )
    rotation_body = task._quaternion_to_rotation_vector_wxyz(error_quat)
    start_rot = as_rotation_matrix(start_quat, order="wxyz")
    rotation_world = torch.matmul(start_rot, rotation_body.unsqueeze(-1)).squeeze(-1)
    angle = float(torch.norm(rotation_world, dim=-1).item())

    mass = float(task.starlink_manipulator.robot.get_mass())
    force_limit = float(max(pair[1] for pair in task.starlink_manipulator.pid.output_limits[:3]))
    torque_limit = float(max(pair[1] for pair in task.starlink_manipulator.pid.output_limits[3:]))
    rot_gain_kp = float(task.starlink_manipulator.pid.Kp[3])
    rot_gain_kd = float(task.starlink_manipulator.pid.Kd[3])
    inertia_y = float(args.assumed_inertia)

    accel_limit = args.waypoint_brake_frac * force_limit / mass
    ang_accel_limit = args.waypoint_brake_frac * torque_limit / inertia_y

    translation_time = 2.0 * math.sqrt(distance / accel_limit) if accel_limit > 0 else 0.0
    rotation_time = 2.0 * math.sqrt(angle / ang_accel_limit) if ang_accel_limit > 0 else 0.0
    duration = max(translation_time, rotation_time, 1e-3)

    n_steps = int(math.ceil(duration / task.control_dt))
    track = []
    for step in range(n_steps + 1):
        t = step * task.control_dt
        travelled = _profile_progress(t, duration, distance)
        turned = _profile_progress(t, duration, angle)
        position = start_pos + direction * travelled
        if angle > 1e-8:
            axis = rotation_world / angle
            quat = task._quaternion_multiply_wxyz(
                task._rotation_vector_to_quaternion_wxyz(axis * turned),
                start_quat,
            )
        else:
            quat = start_quat.clone()
        track.append((position, quat))

    info = {
        "distance": distance,
        "angle": angle,
        "duration": duration,
        "n_steps": len(track),
        "accel": (4.0 * distance / (duration * duration)) if duration > 0 else 0.0,
        "ang_accel": (4.0 * angle / (duration * duration)) if duration > 0 else 0.0,
        "accel_limit": accel_limit,
        "ang_accel_limit": ang_accel_limit,
        "torque_limit": torque_limit,
        "inertia": inertia_y,
        "rot_gain_kp": rot_gain_kp,
        "rot_gain_kd": rot_gain_kd,
    }
    return track, info


def run_case(
    task,
    mode,
    standoff,
    goal_pos,
    goal_quat,
    args,
    log_rows,
    viewer=None,
):
    manip = task.starlink_manipulator
    pid = manip.pid

    measured_standoff = place_base_at_standoff(
        task,
        standoff,
        args.start_lateral_offset,
        args.start_vertical_offset,
        args.start_attitude_offset_deg,
    )
    start_pos, start_quat = task._get_base_pose()
    pid.reset()
    if args.pid_dt is not None:
        pid.dt = float(args.pid_dt)
    if args.pid_rot_kp is not None:
        pid.Kp[3:6] = float(args.pid_rot_kp)
    if args.pid_rot_kd is not None:
        pid.Kd[3:6] = float(args.pid_rot_kd)
    if args.pid_pos_kp is not None:
        pid.Kp[0:3] = float(args.pid_pos_kp)
    if args.pid_pos_kd is not None:
        pid.Kd[0:3] = float(args.pid_pos_kd)
    if args.pid_force_limit is not None:
        pid.output_limits = [
            (-float(args.pid_force_limit), float(args.pid_force_limit))
        ] * 3 + pid.output_limits[3:6]
    if args.pid_torque_limit is not None:
        pid.output_limits = pid.output_limits[0:3] + [
            (-float(args.pid_torque_limit), float(args.pid_torque_limit))
        ] * 3
    manip.update_state()
    arm_fixture = configure_arm_fixture(manip) if not args.no_arm_fixture else None

    track = None
    track_info = None
    if mode == "waypoint":
        track, track_info = build_waypoint_track(
            task, start_pos, goal_pos, start_quat, goal_quat, args
        )
    if mode == "fixed":
        pid.update_setpoint(
            task._root_target_to_pid_link_setpoint(goal_pos, goal_quat)
        )
    elif mode == "attitude":
        # Keep the current position, command only the goal attitude.
        base_pos, _ = task._get_base_pose()
        pid.update_setpoint(
            task._root_target_to_pid_link_setpoint(base_pos, goal_quat)
        )

    grasp_pos, _ = task._get_target_grasp_pose()
    # The legacy target distance is planar; z remains at the reset height.
    goal_distance = float(
        torch.norm(goal_pos[:, :2] - grasp_pos[:, :2], dim=-1)[0].item()
    )
    move_distance = float(torch.norm(start_pos - goal_pos, dim=-1)[0].item())
    goal_standoff_3d = float(torch.norm(goal_pos - grasp_pos, dim=-1)[0].item())
    position_error, attitude_error = goal_pose_errors(task, goal_pos, goal_quat)
    start_dock_error = float(position_error[0].item())
    start_attitude_error = float(attitude_error[0].item())

    print(
        f"[STAGE1_PID_RANGE] run start: mode={mode} standoff={measured_standoff:.4f} m "
        f"(move {move_distance:.4f} m), "
        f"goal_xy_dist={goal_distance:.4f} m, "
        f"dock_err={start_dock_error:.4f} m, att_err={start_attitude_error:.4f} rad, "
        f"start_rpy={_vec3_deg(pid.quaternion_to_euler(start_quat)[0])}deg, "
        f"goal_rpy={_vec3_deg(pid.quaternion_to_euler(goal_quat)[0])}deg, "
        f"steps={args.max_control_steps}",
        flush=True,
    )
    if mode in ("fixed", "attitude"):
        print(
            f"[STAGE1_PID_RANGE]   note: mode '{mode}' is the un-tuned reference "
            "(a single goal setpoint held from the start, no policy / no "
            "trajectory). It is expected to saturate and oscillate; the base "
            "will visibly swing around. Use mode 'waypoint' to watch the "
            "tuned behaviour.",
            flush=True,
        )
    if track_info is not None:
        print(
            f"[STAGE1_PID_RANGE]   waypoints: {track_info['n_steps']} points over "
            f"{track_info['duration']:.2f} s (accel {track_info['accel']:.3f} "
            f"m/s^2 of {track_info['accel_limit']:.3f} allowed, ang accel "
            f"{track_info['ang_accel']:.4f} of "
            f"{track_info['ang_accel_limit']:.4f} rad/s^2 allowed, "
            f"I={track_info['inertia']:.0f} kg*m^2, "
            f"rot P/D={track_info['rot_gain_kp']:.0f}/{track_info['rot_gain_kd']:.0f})",
            flush=True,
        )

    decimation = int(getattr(task, "base_control_decimation", 10))

    min_dock_error = start_dock_error
    min_standoff = measured_standoff
    min_attitude_error = start_attitude_error
    settle_hold = 0
    settle_step = None
    in_band_hold = 0
    in_band_hold_max = 0
    track_index = 0
    max_track_lag = 0.0
    saturated_steps = 0
    max_pid_position_error = 0.0
    max_pid_attitude_error = 0.0
    max_arm_joint_error = 0.0
    wall_start = time.perf_counter()
    viewer_closed = False

    for step in range(1, args.max_control_steps + 1):
        if viewer is not None:
            try:
                viewer_alive = viewer.is_alive()
            except Exception:  # noqa: BLE001 - viewer can die with a GL error
                viewer_alive = False
            if not viewer_alive:
                print(
                    "[STAGE1_PID_RANGE] viewer window closed; stopping the run",
                    flush=True,
                )
                viewer_closed = True
                break
        if mode == "waypoint":
            if args.waypoint_lead > 0.0:
                # Let the track follow the base instead of the clock: advance
                # the look-ahead point only while the base is close to it. This
                # keeps the commanded setpoint reachable and stops the PID from
                # saturating while chasing a setpoint that ran away.
                while True:
                    wp_pos, wp_quat = track[track_index]
                    base_now, _ = task._get_base_pose()
                    lag = float(torch.norm(wp_pos - base_now[0]).item())
                    max_track_lag = max(max_track_lag, lag)
                    if (
                        lag <= args.waypoint_lead
                        and track_index + 1 < len(track)
                    ):
                        track_index += 1
                        continue
                    break
            else:
                wp_pos, wp_quat = track[min(step - 1, len(track) - 1)]
            pid.update_setpoint(
                task._root_target_to_pid_link_setpoint(wp_pos, wp_quat)
            )
            if arm_fixture is None and task.joint_reference is not None:
                manip.control_joint_pos(task.joint_reference)
            if arm_fixture is None:
                manip.control_gripper(task.last_gripper_command)
        elif mode == "stepwise":
            action = _stepwise_action(task, goal_pos, goal_quat, args)
            task._apply_base_action(action)
        else:
            if arm_fixture is None and task.joint_reference is not None:
                manip.control_joint_pos(task.joint_reference)
            if arm_fixture is None:
                manip.control_gripper(task.last_gripper_command)

        for _ in range(decimation):
            if arm_fixture is not None:
                max_arm_joint_error = max(
                    max_arm_joint_error, hold_arm_fixture(manip, arm_fixture)
                )
            manip.step()
            task.target_object.step()
            task.gsim.step()
        task._update_io_tensors()

        control_output = pid.input_reference()
        force = _as_batch6(control_output["position"])[0]
        torque = _as_batch6(control_output["orientation"])[0]
        limits = [pair[1] for pair in pid.output_limits[:3]]
        if any(abs(float(force[i])) >= 0.999 * limits[i] for i in range(3)):
            saturated_steps += 1

        # What the low-level controller itself sees: |setpoint - measured|.
        pid_position_error = float(
            torch.norm(
                pid.setpoint[..., :3] - pid._current_position, dim=-1
            )[0].item()
        )
        pid_attitude_delta = (
            pid.setpoint[..., 3:] - pid._current_orientation + math.pi
        ) % (2.0 * math.pi) - math.pi
        pid_attitude_error = float(
            torch.norm(pid_attitude_delta, dim=-1)[0].item()
        )
        max_pid_position_error = max(max_pid_position_error, pid_position_error)
        max_pid_attitude_error = max(max_pid_attitude_error, pid_attitude_error)

        position_error, pose_attitude_error = goal_pose_errors(
            task, goal_pos, goal_quat
        )
        dock_error = float(position_error[0].item())
        attitude_error = float(pose_attitude_error[0].item())
        standoff_now = float(task.base_alignment_error[0].item())
        linear_speed = float(task.base_linear_speed[0].item())
        angular_speed = float(task.base_angular_speed[0].item())

        min_dock_error = min(min_dock_error, dock_error)
        min_standoff = min(min_standoff, standoff_now)
        min_attitude_error = min(min_attitude_error, attitude_error)

        stable = (
            dock_error <= float(task.base_goal_band)
            and attitude_error <= float(task.base_attitude_band)
            and linear_speed <= float(task.base_stop_speed_threshold)
            and angular_speed <= float(task.base_stop_angular_speed_threshold)
        )
        settle_hold = settle_hold + 1 if stable else 0
        if settle_step is None and settle_hold >= args.settle_hold_steps:
            settle_step = step - args.settle_hold_steps + 1

        # Same band test without the (strict) stop-speed requirement, to
        # separate "reached and held the pose" from "perfectly at rest".
        in_band = (
            dock_error <= float(task.base_goal_band)
            and attitude_error <= float(task.base_attitude_band)
        )
        in_band_hold = in_band_hold + 1 if in_band else 0
        in_band_hold_max = max(in_band_hold_max, in_band_hold)

        base_pos_now, base_quat_now = task._get_base_pose()
        base_euler_now = pid.quaternion_to_euler(base_quat_now)[0]
        angular_velocity_now = task.base_angular_velocity[0]

        log_rows.append(
            {
                "mode": mode,
                "start_standoff": round(measured_standoff, 5),
                "step": step,
                "dock_error": round(dock_error, 6),
                "attitude_error": round(attitude_error, 6),
                "attitude_error_deg": round(math.degrees(attitude_error), 3),
                "base_roll_deg": round(math.degrees(float(base_euler_now[0])), 3),
                "base_pitch_deg": round(math.degrees(float(base_euler_now[1])), 3),
                "base_yaw_deg": round(math.degrees(float(base_euler_now[2])), 3),
                "angular_vel_x_deg": round(math.degrees(float(angular_velocity_now[0])), 3),
                "angular_vel_y_deg": round(math.degrees(float(angular_velocity_now[1])), 3),
                "angular_vel_z_deg": round(math.degrees(float(angular_velocity_now[2])), 3),
                "base_grasp_distance": round(standoff_now, 6),
                "linear_speed": round(linear_speed, 6),
                "angular_speed": round(angular_speed, 6),
                "pid_position_error": round(pid_position_error, 6),
                "pid_attitude_error": round(pid_attitude_error, 6),
                "force_x": round(float(force[0]), 4),
                "force_y": round(float(force[1]), 4),
                "force_z": round(float(force[2]), 4),
                "torque_x": round(float(torque[0]), 4),
                "torque_y": round(float(torque[1]), 4),
                "torque_z": round(float(torque[2]), 4),
            }
        )

        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"[STAGE1_PID_RANGE]   step={step:4d} "
                f"dock_err={dock_error:8.4f} att_err={attitude_error:7.4f} "
                f"({math.degrees(attitude_error):5.2f}deg) "
                f"base_rpy={_vec3_deg(base_euler_now)}deg "
                f"omega={_vec3_deg(angular_velocity_now)}deg/s "
                f"standoff={standoff_now:7.4f} speed={linear_speed:6.4f} "
                f"force={_vec3(force)} torque={_vec3(torque)}",
                flush=True,
            )

    if viewer_closed:
        return None

    wall_time = time.perf_counter() - wall_start
    overshoot = max(0.0, goal_standoff_3d - min_standoff)
    result = {
        "mode": mode,
        "start_standoff": measured_standoff,
        "move_distance": move_distance,
        "goal_distance": goal_distance,
        "start_dock_error": start_dock_error,
        "min_dock_error": min_dock_error,
        "min_attitude_error": min_attitude_error,
        "final_dock_error": float(goal_pose_errors(task, goal_pos, goal_quat)[0][0].item()),
        "final_attitude_error": float(goal_pose_errors(task, goal_pos, goal_quat)[1][0].item()),
        "final_standoff": float(task.base_alignment_error[0].item()),
        "min_standoff": min_standoff,
        "overshoot": overshoot,
        "settle_step": settle_step,
        "settle_time": None if settle_step is None else settle_step * task.control_dt,
        "in_band_hold_max": in_band_hold_max,
        "max_pid_position_error": max_pid_position_error,
        "max_pid_attitude_error": max_pid_attitude_error,
        "max_arm_joint_error": max_arm_joint_error,
        "max_track_lag": max_track_lag,
        "waypoint_duration": None if track_info is None else track_info["duration"],
        "waypoint_steps": None if track_info is None else track_info["n_steps"],
        "saturation_fraction": saturated_steps / float(args.max_control_steps),
        "wall_time": wall_time,
        "settled": settle_step is not None,
    }
    settle_text = (
        "none"
        if settle_step is None
        else f"{settle_step} steps ({result['settle_time']:.2f} s)"
    )
    if settle_step is not None:
        verdict = f"held the goal pose (settled at step {settle_step})"
    elif in_band_hold_max >= args.settle_hold_steps:
        verdict = (
            f"reached the goal band and held it for {in_band_hold_max} steps, "
            "but the stop-speed criterion was not met for a full hold"
        )
    else:
        verdict = "did NOT settle"
    print(
        f"[STAGE1_PID_RANGE] run end:   mode={mode} standoff={measured_standoff:.4f} "
        f"final_dock_err={result['final_dock_error']:.4f} "
        f"min_dock_err={result['min_dock_error']:.4f} "
        f"min_att_err={result['min_attitude_error']:.4f} "
        f"overshoot={overshoot:.4f} "
        f"settle={settle_text} "
        f"finally_standoff={result['final_standoff']:.4f} "
        f"final_att_err={result['final_attitude_error']:.4f} "
        f"max_pid_err={result['max_pid_position_error']:.4f}/"
        f"{result['max_pid_attitude_error']:.4f} "
        f"max_track_lag={result['max_track_lag']:.3f} "
        f"max_arm_joint_err={result['max_arm_joint_error']:.6f} rad "
        f"in_band_hold={result['in_band_hold_max']} "
        f"saturated={result['saturation_fraction'] * 100:.0f}% "
        f"wall={wall_time:.1f}s",
        flush=True,
    )
    print(f"[STAGE1_PID_RANGE]   verdict: {verdict}", flush=True)
    return result


def main():
    args = parse_args()
    args.start_attitude_offset_deg = _parse_vec3(
        args.start_attitude_offset_deg, "--start-attitude-offset-deg"
    )
    start_specs = _parse_start_distances(args.start_distances)
    modes = _parse_modes(args.modes)

    _configure_cameras()
    _configure_viewer(args)

    env = build_stage1_env(
        num_envs=1,
        enable_visual_obs=False,
        target_position_noise=False,
    )
    task = env.task
    log_rows = []
    results = []
    try:
        env.reset()
        viewer = None
        if SCENE_SETTINGS["show_viewer"]:
            candidate = getattr(task.gsim, "viewer", None)
            try:
                if candidate is not None and candidate.is_alive():
                    viewer = candidate
            except Exception:  # noqa: BLE001 - OpenGL context failure
                viewer = None
            if viewer is None:
                print(
                    "[STAGE1_PID_RANGE] viewer window unavailable; "
                    "continuing headless",
                    flush=True,
                )

        manip = task.starlink_manipulator
        if args.assumed_inertia is None:
            args.assumed_inertia = measure_base_inertia(task)
            task.reset()
            print(
                f"[STAGE1_PID_RANGE] measured base inertia "
                f"I_yy={args.assumed_inertia:.1f} kg*m^2 (torque impulse)",
                flush=True,
            )
        grasp_pos, grasp_quat = task._get_target_grasp_pose()
        base_pos, base_quat = task._get_base_pose()
        nominal_base_pos = (
            task.nominal_base_pos
            if task.nominal_base_pos is not None
            else base_pos
        )
        nominal_base_quat = (
            task.nominal_base_quat
            if task.nominal_base_quat is not None
            else base_quat
        )
        reset_standoff = float(task.base_alignment_error[0].item())
        goal_distance = (
            task.base_goal_threshold
            if args.goal_distance is None
            else float(args.goal_distance)
        )
        mass = float(manip.robot.get_mass())

        goal_pos, goal_quat = build_legacy_stage1_goal(
            task,
            grasp_pos,
            grasp_quat,
            nominal_base_pos,
            nominal_base_quat,
            goal_distance,
        )

        if viewer is not None:
            print(
                "[STAGE1_PID_RANGE] viewer: stock camera from SCENE_SETTINGS "
                "(close the window to stop the run)",
                flush=True,
            )

        print(
            f"[STAGE1_PID_RANGE] setup: base_pos={_vec3(base_pos)} "
            f"grasp_pos={_vec3(grasp_pos)} reset_standoff={reset_standoff:.4f} m "
            f"goal_distance={goal_distance:.4f} m "
            f"goal_pos={_vec3(goal_pos)}",
            flush=True,
        )
        print(
            f"[STAGE1_PID_RANGE] controller: P={[float(v) for v in manip.pid.Kp.tolist()]} "
            f"D={[float(v) for v in manip.pid.Kd.tolist()]} "
            f"limits={[pair[1] for pair in manip.pid.output_limits]} "
            f"pid_dt={float(manip.pid.dt)} "
            f"control_dt={task.control_dt} decimation={task.base_control_decimation} "
            f"mass={mass:.1f} kg "
            f"max_accel={max(pair[1] for pair in manip.pid.output_limits[:3]) / mass:.3f} m/s^2",
            flush=True,
        )
        print(
            f"[STAGE1_PID_RANGE] task: max_steps={task.max_steps} control steps "
            f"({task.max_steps * task.control_dt:.2f} s episode budget), "
            f"goal_band={task.base_goal_band} m, "
            f"attitude_band={task.base_attitude_band} rad, "
            f"action_limit={float(task.base_translation_action_limit)} m / "
            f"{float(task.base_rotation_action_limit)} rad, "
            f"arm_fixture={'normal' if args.no_arm_fixture else 'stiff-position-servo'}",
            flush=True,
        )

        for spec in start_specs:
            standoff = reset_standoff if spec == "current" else float(spec)
            label = "current" if spec == "current" else f"{standoff:.2f}"
            for mode in modes:
                # Re-run the stock reset between cases so every run starts from
                # a clean simulator state, then move the base to the standoff.
                task.reset()
                grasp_pos, grasp_quat = task._get_target_grasp_pose()
                reset_pos, reset_quat = task._get_base_pose()
                goal_reference_pos = (
                    task.nominal_base_pos
                    if task.nominal_base_pos is not None
                    else reset_pos
                )
                goal_reference_quat = (
                    task.nominal_base_quat
                    if task.nominal_base_quat is not None
                    else reset_quat
                )
                goal_pos, goal_quat = build_legacy_stage1_goal(
                    task,
                    grasp_pos,
                    grasp_quat,
                    goal_reference_pos,
                    goal_reference_quat,
                    goal_distance,
                )
                result = run_case(
                    task,
                    mode,
                    standoff,
                    goal_pos,
                    goal_quat,
                    args,
                    log_rows,
                    viewer=viewer,
                )
                if result is None:
                    print(
                        "[STAGE1_PID_RANGE] aborted: the viewer window was closed",
                        flush=True,
                    )
                    return 0
                results.append(result)
                print(
                    f"[STAGE1_PID_RANGE] case: standoff={label} mode={mode}",
                    flush=True,
                )

        print("[STAGE1_PID_RANGE] summary", flush=True)
        header = (
            f"{'standoff':>9} {'move':>7} {'mode':>9} {'min_err':>8} {'final_err':>9} "
            f"{'pid_err':>8} {'settle':>7} {'inband':>7} {'time(s)':>8} "
            f"{'min_att':>8} {'att_err':>8} {'sat%':>5}"
        )
        print("[STAGE1_PID_RANGE] " + header, flush=True)
        for result in results:
            settle = result["settle_step"] if result["settle_step"] is not None else -1
            settle_time = result["settle_time"] if result["settle_time"] is not None else float("nan")
            print(
                "[STAGE1_PID_RANGE] "
                f"{result['start_standoff']:9.3f} {result['move_distance']:7.3f} "
                f"{result['mode']:>9} {result['min_dock_error']:8.4f} "
                f"{result['final_dock_error']:9.4f} "
                f"{result['max_pid_position_error']:8.4f} "
                f"{settle:7d} {result['in_band_hold_max']:7d} {settle_time:8.2f} "
                f"{result['min_attitude_error']:8.4f} "
                f"{result['final_attitude_error']:8.4f} "
                f"{result['saturation_fraction'] * 100:5.0f}",
                flush=True,
            )

        if args.csv:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"[STAGE1_PID_RANGE] wrote {len(log_rows)} rows to {csv_path}", flush=True)

        failures = [
            result
            for result in results
            if not result["settled"]
        ]
        if failures:
            detail = ", ".join(
                f"{result['mode']}@{result['start_standoff']:.2f}m"
                f"(final_err={result['final_dock_error']:.3f}, "
                f"min_err={result['min_dock_error']:.3f})"
                for result in failures
            )
            print(
                f"[STAGE1_PID_RANGE] FAILED: base PID did not settle inside the "
                f"{task.base_goal_band} m / {task.base_attitude_band} rad band "
                f"within {args.max_control_steps} control steps for: {detail}",
                flush=True,
            )
            return 1

        print("[STAGE1_PID_RANGE] PASSED", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
