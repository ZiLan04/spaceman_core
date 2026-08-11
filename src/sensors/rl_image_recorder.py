from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

from utils.utils import generate_filename

_RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"


class RLImageRecorder:
    """Records already-rendered RL observation frames without extra camera renders."""

    def __init__(
        self,
        *,
        name: str,
        enabled: bool = False,
        env_idx: int = 0,
        fps: int = 30,
        episodes_per_video: int = 1,
        channels_first: bool = True,
        normalize: bool = False,
    ):
        self.name = str(name)
        self.enabled = bool(enabled)
        self.env_idx = max(0, int(env_idx))
        self.fps = max(1, int(fps))
        self.episodes_per_video = max(1, int(episodes_per_video))
        self.channels_first = bool(channels_first)
        self.normalize = bool(normalize)
        self._frames: List[np.ndarray] = []
        self._episode_count = 0
        self._segment_index = 0
        self._warned_missing_cv2 = False

    def record_batch(self, rgb_batch: torch.Tensor):
        if not self.enabled:
            return
        if cv2 is None:
            if not self._warned_missing_cv2:
                print("[RL_IMAGE_RECORDER] cv2 is not available; skipping RL video recording.")
                self._warned_missing_cv2 = True
            return
        if rgb_batch is None:
            return

        frame = rgb_batch[min(self.env_idx, int(rgb_batch.shape[0]) - 1)].detach()
        if self.channels_first:
            frame = frame.permute(1, 2, 0)
        frame = frame.cpu().numpy()

        if self.normalize:
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._frames.append(np.ascontiguousarray(frame))

    def save_segment(self):
        if not self.enabled or not self._frames:
            return
        if cv2 is None:
            self._frames.clear()
            return

        self._segment_index += 1
        save_file = generate_filename(
            prefix=f"{self.name}_ep_chunk_{self._segment_index:04d}",
            extension="mp4",
            folder_path=str(_RECORDINGS_DIR),
        )
        height, width = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(
            save_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            print(f"[RL_IMAGE_RECORDER] Failed to open video writer: {save_file}")
            self._frames.clear()
            return

        for frame in self._frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self._frames.clear()
        print(f"[RL_IMAGE_RECORDER] Saved RL camera recording to: {save_file}")

    def on_episode_end(self, episode_count: int = 1, env_ids: Optional[object] = None):
        if not self.enabled:
            return

        if env_ids is None:
            completed_record_env_episodes = max(0, int(episode_count))
        else:
            if torch.is_tensor(env_ids):
                env_ids = env_ids.detach().cpu().tolist()
            completed_record_env_episodes = sum(
                1 for env_id in env_ids if int(env_id) == self.env_idx
            )

        if completed_record_env_episodes <= 0:
            return

        self._episode_count += completed_record_env_episodes
        if self._episode_count >= self.episodes_per_video:
            self.save_segment()
            self._episode_count = 0

    def reset(self):
        self._episode_count = 0

    def stop(self):
        self.save_segment()
