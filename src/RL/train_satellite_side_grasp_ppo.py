import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import genesis as gs

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
sys.path.append(str(project_src_path))

from RL.policies.multimodal_extractor import MultiModalExtractor
import configs.asset_configs as asset_configs
from configs.config_starlink import _STARLINK_PATHS, to_posix
from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA
from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
from tasks.RL_task_parallel import PPO, gym, make_parallel_vec_env, spaces
from tasks.RL_task_parallel_stage1 import make_stage1_vec_env
from tasks.RL_task_parallel_stage2 import make_stage2_vec_env

try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
except Exception:
    BaseCallback = None
    CallbackList = None
    CheckpointCallback = None

try:
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:
    VecNormalize = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO for the batched satellite side-edge grasp task."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=("stage1", "stage2", "full"),
        help=(
            "Training stage. Use 'stage1' for base alignment, 'stage2' for "
            "base-held arm pre-grasp, or 'full' for side grasp."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a saved PPO .zip model to continue training from, e.g. "
            "a final_model.zip or a file under checkpoints/."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument(
        "--metric-log-freq",
        type=int,
        default=1000,
        help="Training steps between custom TensorBoard metric logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device for PPO. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument("--target-asset", type=str, default="starlink")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--target-position-noise",
        action="store_true",
        help="Enable reset-time target position noise. Target orientation noise is always disabled.",
    )
    parser.add_argument(
        "--enable-visual-obs",
        dest="enable_visual_obs",
        action="store_true",
        help="Use camera RGB observations in addition to low-dimensional state.",
    )
    parser.add_argument(
        "--disable-visual-obs",
        dest="enable_visual_obs",
        action="store_false",
        help="Disable camera RGB observations and train with low-dimensional state only.",
    )
    parser.add_argument(
        "--enable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_true",
        help=(
            "For stage-1/stage-2 non-visual training, append oracle target "
            "data to the low-dimensional state."
        ),
    )
    parser.add_argument(
        "--disable-target-relative-obs",
        dest="enable_target_relative_obs",
        action="store_false",
        help=(
            "For stage-1/stage-2 non-visual training, do not append "
            "oracle target data to the low-dimensional state."
        ),
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=3,
        metavar=("C", "H", "W"),
        default=(3, 84, 84),
        help="Image observation shape passed to the policy extractor.",
    )
    parser.add_argument(
        "--normalize-images",
        action="store_true",
        help="Normalize image observations to [0, 1] in the environment.",
    )
    parser.add_argument(
        "--image-encoder-type",
        type=str,
        default="cnn",
        choices=("cnn", "vit"),
        help="Visual backbone used by the multimodal feature extractor.",
    )
    parser.add_argument("--state-feature-dim", type=int, default=128)
    parser.add_argument("--image-feature-dim", type=int, default=256)
    parser.add_argument("--fused-feature-dim", type=int, default=256)
    parser.add_argument(
        "--stage2-base-distance",
        type=float,
        default=1.1,
        help="Stage-2 base initialization distance from the target grasp point in meters.",
    )
    parser.add_argument(
        "--stage2-virtual-grasp-height",
        type=float,
        default=0.08,
        help="Stage-2 virtual pre-grasp target height above the physical grasp point in meters.",
    )
    parser.add_argument(
        "--stage2-enable-target-collision",
        action="store_true",
        help=(
            "Enable physical target collision in stage-2. Disabled by default "
            "because stage-2 trains against a virtual pre-grasp pose."
        ),
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(10.0, 10.0),
        help="Spacing between parallel env replicas in the Genesis batched scene.",
    )
    parser.add_argument(
        "--n-envs-per-row",
        type=int,
        default=None,
        help="Optional row width for env grid layout.",
    )
    parser.add_argument(
        "--center-envs-at-origin",
        dest="center_envs_at_origin",
        action="store_true",
        help="Center the env grid around the origin.",
    )
    parser.add_argument(
        "--no-center-envs-at-origin",
        dest="center_envs_at_origin",
        action="store_false",
        help="Do not center the env grid around the origin.",
    )
    parser.set_defaults(
        center_envs_at_origin=True,
        enable_visual_obs=True,
        enable_target_relative_obs=None,
    )
    return parser.parse_args()


def resolve_stage_settings(args):
    if args.stage == "stage1":
        return {
            "make_env": make_stage1_vec_env,
            "default_save_dir": "src/outputs/rl/ppo_satellite_stage1_base",
            "checkpoint_prefix": "ppo_satellite_stage1_base",
            "description": "stage-1 base alignment",
            "env_kwargs": {
                "enable_target_relative_obs": args.enable_target_relative_obs,
            },
            "image_key": "image",
        }
    if args.stage == "stage2":
        return {
            "make_env": make_stage2_vec_env,
            "default_save_dir": "src/outputs/rl/ppo_satellite_stage2_arm",
            "checkpoint_prefix": "ppo_satellite_stage2_arm",
            "description": "stage-2 base-held arm pre-grasp",
            "env_kwargs": {
                "stage2_base_distance": args.stage2_base_distance,
                "virtual_grasp_height": args.stage2_virtual_grasp_height,
                "enable_target_relative_obs": args.enable_target_relative_obs,
            },
            "image_key": "wrist_image",
        }
    return {
        "make_env": make_parallel_vec_env,
        "default_save_dir": "src/outputs/rl/ppo_satellite_side_grasp_parallel",
        "checkpoint_prefix": "ppo_satellite_side_grasp_parallel",
        "description": "full side grasp",
        "env_kwargs": {},
        "image_key": "image",
    }


def normalize_oracle_observation_args(args):
    if args.stage not in ("stage1", "stage2"):
        return
    if args.enable_visual_obs:
        args.enable_target_relative_obs = False
    elif args.enable_target_relative_obs is None:
        args.enable_target_relative_obs = True


def resolve_training_device(device_arg: str) -> str:
    device_arg = device_arg.strip().lower()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False."
        )
    return device_arg


def ensure_dependencies():
    missing = []
    if gym is None or spaces is None:
        missing.append("gymnasium")
    if PPO is None:
        missing.append("stable-baselines3")
    if CheckpointCallback is None:
        missing.append("stable-baselines3 common utilities")
    if BaseCallback is None or CallbackList is None:
        missing.append("stable-baselines3 callback utilities")

    if missing:
        deps = ", ".join(sorted(set(missing)))
        raise ImportError(
            f"Missing training dependencies: {deps}. "
            "Please install them in the Spaceman_env environment before training."
        )


def make_run_dir(base_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = run_name if run_name else f"run_{timestamp}"
    run_dir = base_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path

    raise FileNotFoundError(f"Resume model not found: {model_path}")


def save_run_config(run_dir: Path, args):
    config = {
        "total_timesteps": args.total_timesteps,
        "resume_from": args.resume_from,
        "stage": args.stage,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "seed": args.seed,
        "checkpoint_freq": args.checkpoint_freq,
        "metric_log_freq": args.metric_log_freq,
        "target_asset": args.target_asset,
        "num_envs": args.num_envs,
        "target_position_noise": args.target_position_noise,
        "enable_visual_obs": args.enable_visual_obs,
        "normalize_observations": not args.enable_visual_obs,
        "normalize_rewards": False,
        "enable_target_relative_obs": args.enable_target_relative_obs,
        "image_shape": list(args.image_shape),
        "normalize_images": args.normalize_images,
        "image_encoder_type": args.image_encoder_type,
        "state_feature_dim": args.state_feature_dim,
        "image_feature_dim": args.image_feature_dim,
        "fused_feature_dim": args.fused_feature_dim,
        "stage2_base_distance": args.stage2_base_distance,
        "stage2_virtual_grasp_height": args.stage2_virtual_grasp_height,
        "stage2_enable_target_collision": args.stage2_enable_target_collision,
        "env_spacing": list(args.env_spacing),
        "n_envs_per_row": args.n_envs_per_row,
        "center_envs_at_origin": args.center_envs_at_origin,
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def configure_headless_training(enable_visual_obs: bool):
    if enable_visual_obs:
        return

    SCENE_SETTINGS["renderer"] = None
    TPV_CAM_SETTINGS["enable_recording"] = False

    rl_obs_config = FRANKA_S_Q_CAMERA.get("rl_obs", {})
    rl_obs_config["enabled"] = False
    rl_obs_config["enable_recording"] = False

    base_camera_config = FRANKA_S_Q_CAMERA.get("base_camera", {})
    base_camera_config["enabled"] = False
    base_camera_config["enable_recording"] = False
    base_rl_obs_config = base_camera_config.get("rl_obs", {})
    base_rl_obs_config["enabled"] = False
    base_rl_obs_config["enable_recording"] = False


def configure_stage2_target_collision(args):
    if args.stage != "stage2":
        return
    if args.target_asset != "starlink":
        print(
            "Stage-2 target collision override is only implemented for target_asset='starlink'; "
            f"got {args.target_asset!r}.",
            flush=True,
        )
        return

    enable_collision = bool(args.stage2_enable_target_collision)

    def make_starlink_for_stage2_training():
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

    asset_configs._ASSET_FACTORIES["starlink"] = make_starlink_for_stage2_training
    print(
        f"Stage-2 target collision enabled: {enable_collision}",
        flush=True,
    )


def find_vecnormalize_stats(model_path: Path) -> Path | None:
    for candidate_dir in (model_path.parent, model_path.parent.parent):
        stats_path = candidate_dir / "vecnormalize.pkl"
        if stats_path.exists():
            return stats_path
    return None


class Stage2MetricsCallback(BaseCallback if BaseCallback is not None else object):
    """Record stage-2 task metrics to TensorBoard during PPO training."""

    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = max(1, int(log_freq))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        for attr_name, tag_prefix in (
            ("tcp_to_virtual_grasp_error", "stage2/tcp_to_virtual_grasp_error"),
            ("pose_position_error", "stage2/pose_position_error"),
            ("pose_quat_similarity", "stage2/pose_quat_similarity"),
            ("tcp_speed", "stage2/tcp_speed"),
            ("target_hold_count", "stage2/target_hold_count"),
        ):
            self._record_tensor_attr(attr_name, tag_prefix)
        return True

    def _record_tensor_attr(self, attr_name: str, tag_prefix: str) -> None:
        values = self.training_env.get_attr(attr_name)
        if not values:
            return

        tensor = torch.as_tensor(values[0]).detach().float()
        if tensor.numel() == 0:
            return

        self.logger.record(f"{tag_prefix}/mean", float(tensor.mean().item()))
        self.logger.record(f"{tag_prefix}/min", float(tensor.min().item()))
        self.logger.record(f"{tag_prefix}/max", float(tensor.max().item()))


def train():
    args = parse_args()
    normalize_oracle_observation_args(args)
    ensure_dependencies()
    stage_settings = resolve_stage_settings(args)
    training_device = resolve_training_device(args.device)

    save_dir = args.save_dir or stage_settings["default_save_dir"]
    base_dir = Path(save_dir)
    run_dir = make_run_dir(base_dir, args.run_name)
    save_run_config(run_dir, args)
    configure_headless_training(args.enable_visual_obs)
    configure_stage2_target_collision(args)
    resume_model_path = resolve_model_path(args.resume_from) if args.resume_from else None

    env = stage_settings["make_env"](
        num_envs=args.num_envs,
        target_asset=args.target_asset,
        env_spacing=tuple(args.env_spacing),
        n_envs_per_row=args.n_envs_per_row,
        center_envs_at_origin=args.center_envs_at_origin,
        enable_visual_obs=args.enable_visual_obs,
        image_shape=tuple(args.image_shape),
        normalize_images=args.normalize_images,
        target_position_noise=args.target_position_noise,
        **stage_settings["env_kwargs"],
    )
    if not args.enable_visual_obs:
        if VecNormalize is None:
            raise ImportError(
                "stable-baselines3 VecNormalize is required for non-visual "
                "observation normalization."
            )
        stats_path = find_vecnormalize_stats(resume_model_path) if resume_model_path else None
        if stats_path is not None:
            env = VecNormalize.load(str(stats_path), env)
            env.training = True
            env.norm_reward = False
            print(f"Loaded observation normalization stats from: {stats_path}")
        else:
            if resume_model_path is not None:
                print(
                    "No vecnormalize.pkl found next to the resume model; "
                    "starting fresh observation normalization stats.",
                    flush=True,
                )
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                gamma=args.gamma,
            )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix=stage_settings["checkpoint_prefix"],
    )
    callbacks = [checkpoint_callback]
    if args.stage == "stage2":
        callbacks.append(Stage2MetricsCallback(log_freq=args.metric_log_freq))
    training_callback = CallbackList(callbacks)

    policy = "MlpPolicy"
    policy_kwargs = None
    if args.enable_visual_obs:
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=MultiModalExtractor,
            features_extractor_kwargs=dict(
                image_key=stage_settings["image_key"],
                image_encoder_type=args.image_encoder_type,
                state_feature_dim=args.state_feature_dim,
                image_feature_dim=args.image_feature_dim,
                fused_feature_dim=args.fused_feature_dim,
                normalize_images=not args.normalize_images,
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

    model_kwargs = dict(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        verbose=1,
        seed=args.seed,
        device=training_device,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    if args.resume_from:
        model = PPO.load(
            str(resume_model_path),
            env=env,
            **model_kwargs,
        )
    else:
        model = PPO(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )

    print(f"Training stage: {args.stage} ({stage_settings['description']})")
    print(f"Using batched PPO training device: {training_device}")
    print(f"Using Genesis batched scene count: {args.num_envs}")
    print(f"Using visual observations: {args.enable_visual_obs}")
    print(f"Using target position reset noise: {args.target_position_noise}")
    if args.resume_from:
        print(f"Resuming PPO model from: {resolve_model_path(args.resume_from)}")
        print(f"Loaded model timestep count: {model.num_timesteps}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=training_callback,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume_from),
    )

    model.save(str(run_dir / "final_model"))
    if not args.enable_visual_obs and hasattr(env, "save"):
        env.save(str(run_dir / "vecnormalize.pkl"))
    env.close()

    print(f"Training completed. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    train()
