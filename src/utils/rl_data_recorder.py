import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import yaml

import sys
from pathlib import Path

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from utils.data_recorder import DataRecorder


class RLDataRecorder(DataRecorder):
    """
    Recorder specialized for RL rollouts and replay dataset collection.

    Compared with ``DataRecorder``:
    - supports reward / done / truncated logging
    - supports batched or single-image observations
    - does not require interactive terminal input on stop
    - writes a lightweight RL-oriented manifest file
    """

    def __init__(
        self,
        base_save_dir: str = "src/datasets/rl",
        episode_id: Optional[str] = None,
        image_key: str = "rgb_gripper",
    ):
        super().__init__(base_save_dir=base_save_dir, episode_id=episode_id)
        self.image_key = image_key
        self.rewards = []
        self.dones = []
        self.truncated = []
        self.extra_metadata: Dict = {}
        self.expected_image_shape = None

    def initialize(
        self,
        static_camera_config: Optional[Dict] = None,
        gripper_camera_config: Optional[Dict] = None,
        scene_config: Optional[Dict] = None,
        robot_config: Optional[Dict] = None,
        expected_image_shape=None,
        extra_metadata: Optional[Dict] = None,
    ) -> None:
        super().initialize(
            static_camera_config=static_camera_config,
            gripper_camera_config=gripper_camera_config,
            scene_config=scene_config,
            robot_config=robot_config,
        )
        self.rewards = []
        self.dones = []
        self.truncated = []
        self.expected_image_shape = (
            tuple(expected_image_shape) if expected_image_shape is not None else None
        )
        self.extra_metadata = dict(extra_metadata or {})
        self.file_paths.update(
            {
                "rewards": os.path.join(self.episode_dir, "rewards.npz"),
                "dones": os.path.join(self.episode_dir, "dones.npz"),
                "truncated": os.path.join(self.episode_dir, "truncated.npz"),
            }
        )

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        # Support single-frame CHW -> HWC.
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        # Support batched NCHW -> NHWC.
        if image.ndim == 4 and image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (0, 2, 3, 1))

        if self.expected_image_shape is not None:
            if tuple(image.shape) != self.expected_image_shape:
                raise AssertionError(
                    f"Image shape should be {self.expected_image_shape}, got {tuple(image.shape)}"
                )
        return image

    def record_rgb_gripper(self, rgb_gripper: np.ndarray) -> None:
        rgb_gripper = self._normalize_image(rgb_gripper)
        self.rgb_gripper.append(rgb_gripper)
        self.logger.debug(f"Recorded rgb_gripper: shape {rgb_gripper.shape}")

    def record_reward(self, reward) -> None:
        reward = np.asarray(reward)
        self.rewards.append(reward)
        self.logger.debug(f"Recorded reward: shape {reward.shape}")

    def record_done(self, done) -> None:
        done = np.asarray(done, dtype=bool)
        self.dones.append(done)
        self.logger.debug(f"Recorded done: shape {done.shape}")

    def record_truncated(self, truncated) -> None:
        truncated = np.asarray(truncated, dtype=bool)
        self.truncated.append(truncated)
        self.logger.debug(f"Recorded truncated: shape {truncated.shape}")

    def record_transition(
        self,
        *,
        action=None,
        robot_obs=None,
        scene_obs=None,
        rgb_gripper=None,
        reward=None,
        done=None,
        truncated=None,
    ) -> None:
        if action is not None:
            self.actions.append(np.asarray(action))
            self.step_count += 1
        if robot_obs is not None:
            self.robot_obs.append(np.asarray(robot_obs))
        if scene_obs is not None:
            self.scene_obs.append(np.asarray(scene_obs))
        if rgb_gripper is not None:
            self.record_rgb_gripper(rgb_gripper)
        if reward is not None:
            self.record_reward(reward)
        if done is not None:
            self.record_done(done)
        if truncated is not None:
            self.record_truncated(truncated)

    def stop(self) -> Dict[str, str]:
        if self.step_count == 0 and not self.actions:
            self.logger.warning("No RL data to save.")
            raise ValueError("No RL data to save.")

        self.logger.info(f"Stopping RL recording and saving episode {self.episode_id}")
        saved_files = {}

        if self.actions:
            actions_array = np.array(self.actions)
            np.savez_compressed(
                self.file_paths["actions"],
                data=actions_array,
                episode_id=self.episode_id,
                step_count=len(self.actions),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["actions"] = self.file_paths["actions"]

        if self.robot_obs:
            robot_obs_array = np.array(self.robot_obs)
            np.savez_compressed(
                self.file_paths["robot_obs"],
                data=robot_obs_array,
                episode_id=self.episode_id,
                step_count=len(self.robot_obs),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["robot_obs"] = self.file_paths["robot_obs"]

        if self.scene_obs:
            scene_obs_array = np.array(self.scene_obs)
            np.savez_compressed(
                self.file_paths["scene_obs"],
                data=scene_obs_array,
                episode_id=self.episode_id,
                step_count=len(self.scene_obs),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["scene_obs"] = self.file_paths["scene_obs"]

        if self.rgb_static:
            rgb_static_array = np.array(self.rgb_static)
            np.savez_compressed(
                self.file_paths["rgb_static"],
                data=rgb_static_array,
                episode_id=self.episode_id,
                step_count=len(self.rgb_static),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["rgb_static"] = self.file_paths["rgb_static"]

        if self.rgb_gripper:
            rgb_gripper_array = np.array(self.rgb_gripper)
            np.savez_compressed(
                self.file_paths["rgb_gripper"],
                data=rgb_gripper_array,
                episode_id=self.episode_id,
                step_count=len(self.rgb_gripper),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["rgb_gripper"] = self.file_paths["rgb_gripper"]

        if self.rewards:
            rewards_array = np.array(self.rewards)
            np.savez_compressed(
                self.file_paths["rewards"],
                data=rewards_array,
                episode_id=self.episode_id,
                step_count=len(self.rewards),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["rewards"] = self.file_paths["rewards"]

        if self.dones:
            dones_array = np.array(self.dones, dtype=bool)
            np.savez_compressed(
                self.file_paths["dones"],
                data=dones_array,
                episode_id=self.episode_id,
                step_count=len(self.dones),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["dones"] = self.file_paths["dones"]

        if self.truncated:
            truncated_array = np.array(self.truncated, dtype=bool)
            np.savez_compressed(
                self.file_paths["truncated"],
                data=truncated_array,
                episode_id=self.episode_id,
                step_count=len(self.truncated),
                timestamp=datetime.now().isoformat(),
            )
            saved_files["truncated"] = self.file_paths["truncated"]

        manifest_file = self._generate_rl_manifest()
        if manifest_file is not None:
            saved_files["manifest"] = manifest_file

        self.logger.info(
            f"RL episode {self.episode_id} saved successfully with {len(saved_files)} files"
        )
        return saved_files

    def _generate_rl_manifest(self) -> Optional[str]:
        try:
            manifest = {
                "episode_id": self.episode_id,
                "timestamp": datetime.now().isoformat(),
                "step_count": int(self.step_count),
                "num_actions": len(self.actions),
                "num_robot_obs": len(self.robot_obs),
                "num_scene_obs": len(self.scene_obs),
                "num_rewards": len(self.rewards),
                "num_dones": len(self.dones),
                "num_truncated": len(self.truncated),
                "num_rgb_static": len(self.rgb_static),
                "num_rgb_gripper": len(self.rgb_gripper),
                "expected_image_shape": list(self.expected_image_shape)
                if self.expected_image_shape is not None
                else None,
                "static_camera_config": self.static_camera_config,
                "gripper_camera_config": self._to_serializable(self.gripper_camera_config),
                "scene_config": self._to_serializable(self.scene_config),
                "robot_config": self._to_serializable(self.robot_config),
                "extra_metadata": self._to_serializable(self.extra_metadata),
                "files": {
                    key: os.path.basename(path)
                    for key, path in self.file_paths.items()
                    if os.path.exists(path)
                },
            }

            manifest_path = os.path.join(self.save_dir, "rl_manifest.yaml")
            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, indent=2)

            self.logger.info(f"Generated RL manifest file: {manifest_path}")
            return manifest_path
        except Exception as e:
            self.logger.error(f"Failed to generate RL manifest file: {e}")
            return None

    def _to_serializable(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key: self._to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                return value
        return value
