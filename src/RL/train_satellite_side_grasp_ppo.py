import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
sys.path.append(str(project_src_path))

from RL.policies.multimodal_extractor import MultiModalExtractor
from tasks.RL_task_parallel import PPO, gym, make_parallel_vec_env, spaces

try:
    from stable_baselines3.common.callbacks import CheckpointCallback
except Exception:
    CheckpointCallback = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO for the batched satellite side-edge grasp task."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/outputs/rl/ppo_satellite_side_grasp_parallel",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device for PPO. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument("--target-asset", type=str, default="starlink")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--enable-visual-obs",
        action="store_true",
        help="Use wrist camera RGB observations in addition to low-dimensional state.",
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
    parser.set_defaults(center_envs_at_origin=True)
    return parser.parse_args()


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


def save_run_config(run_dir: Path, args):
    config = {
        "total_timesteps": args.total_timesteps,
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
        "seed": args.seed,
        "checkpoint_freq": args.checkpoint_freq,
        "target_asset": args.target_asset,
        "num_envs": args.num_envs,
        "enable_visual_obs": args.enable_visual_obs,
        "image_shape": list(args.image_shape),
        "normalize_images": args.normalize_images,
        "image_encoder_type": args.image_encoder_type,
        "state_feature_dim": args.state_feature_dim,
        "image_feature_dim": args.image_feature_dim,
        "fused_feature_dim": args.fused_feature_dim,
        "env_spacing": list(args.env_spacing),
        "n_envs_per_row": args.n_envs_per_row,
        "center_envs_at_origin": args.center_envs_at_origin,
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def train():
    args = parse_args()
    ensure_dependencies()
    training_device = resolve_training_device(args.device)

    base_dir = Path(args.save_dir)
    run_dir = make_run_dir(base_dir, args.run_name)
    save_run_config(run_dir, args)

    env = make_parallel_vec_env(
        num_envs=args.num_envs,
        target_asset=args.target_asset,
        env_spacing=tuple(args.env_spacing),
        n_envs_per_row=args.n_envs_per_row,
        center_envs_at_origin=args.center_envs_at_origin,
        enable_visual_obs=args.enable_visual_obs,
        image_shape=tuple(args.image_shape),
        normalize_images=args.normalize_images,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_satellite_side_grasp_parallel",
    )

    policy = "MlpPolicy"
    policy_kwargs = None
    if args.enable_visual_obs:
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=MultiModalExtractor,
            features_extractor_kwargs=dict(
                image_encoder_type=args.image_encoder_type,
                state_feature_dim=args.state_feature_dim,
                image_feature_dim=args.image_feature_dim,
                fused_feature_dim=args.fused_feature_dim,
                normalize_images=not args.normalize_images,
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

    model = PPO(
        policy=policy,
        env=env,
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
        verbose=1,
        seed=args.seed,
        device=training_device,
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    print(f"Using batched PPO training device: {training_device}")
    print(f"Using Genesis batched scene count: {args.num_envs}")
    print(f"Using visual observations: {args.enable_visual_obs}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    env.close()

    print(f"Training completed. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    train()
