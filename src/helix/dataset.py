"""Data pipeline: teleoperated episodes + hindsight VLM labeling.

Matches the Helix description:

  "We collect a high quality, multi-robot, multi-operator dataset of diverse
   teleoperated behaviors, ~500 hours in total. To generate natural language-
   conditioned training pairs, we use an auto-labeling VLM to generate hindsight
   instructions. The VLM processes segmented video clips from the onboard robot
   cameras, prompted with: 'What instruction would you have given the robot to get
   the action seen in this video?' All items handled during training are excluded
   from evaluations to prevent contamination."
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config


@dataclass
class Episode:
    """One teleoperated episode.

    images:   (T, C, H, W) monocular camera frames at image_fps
    states:   (T, S)       wrist pose (7) + finger positions (5)
    actions:  (T, A)       continuous upper-body actions at 200 Hz (upsampled)
    text:     string       hindsight instruction (auto-labeled by a VLM)
    """
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    text: str = ""
    ep_id: str = ""
    object_ids: set[str] = field(default_factory=set)


def load_episodes(data_dir: str) -> list[Episode]:
    """Load episodes from <data_dir>/*.npz + optional <data_dir>/*.json labels.

    .npz keys: images (uint8 T,H,W,C), states (float T,S), actions (float T,A),
    optionally text (str) and object_ids (list[str]).
    """
    episodes: list[Episode] = []
    for npz_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        data = np.load(npz_path, allow_pickle=True)
        ep = Episode(
            images=np.transpose(data["images"], (0, 3, 1, 2)).astype(np.float32) / 255.0,
            states=data["states"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            text=str(data["text"]) if "text" in data else "",
            ep_id=os.path.basename(npz_path),
            object_ids=set(data["object_ids"].tolist()) if "object_ids" in data else set(),
        )
        episodes.append(ep)
    return episodes


class TeleopDataset(Dataset):
    """Sliding-window dataset with the S1/S2 temporal offset built in.

    Returns aligned training pairs (see Batch in model.py). The S2 branch samples
    the observation delayed by `temporal_offset` steps, S1 samples the current one.
    """

    def __init__(self, episodes: list[Episode], cfg: Config):
        self.cfg = cfg
        self.episodes = episodes
        self.offset = max(1, int(cfg.data.temporal_offset_ms / 1000 * cfg.data.image_fps))
        self.chunk = 3  # prev_actions chunk length (Tq)
        self.samples: list[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            for t in range(self.offset + self.chunk, len(ep.images) - 1):
                self.samples.append((ei, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ei, t = self.samples[idx]
        ep = self.episodes[ei]
        off = self.offset
        return {
            "s1_images": torch.from_numpy(ep.images[t]),
            "s1_state": torch.from_numpy(ep.states[t]),
            "s2_images": torch.from_numpy(ep.images[t - off]),
            "s2_state": torch.from_numpy(ep.states[t - off]),
            "s2_text": ep.text,
            "prev_actions": torch.from_numpy(ep.actions[t - self.chunk : t]),
            "target_actions": torch.from_numpy(ep.actions[t : t + 1]),
        }


class HindsightLabeler:
    """Auto-labeling VLM: segments a clip and asks what instruction would produce
    the observed action (matches the paper's prompt). A stub unless a real VLM
    inference backend is plugged in."""

    PROMPT = ("What instruction would you have given the robot to get the action "
              "seen in this video?")

    def __init__(self, backend=None):
        self.backend = backend  # pluggable; e.g. an API/VLM client

    def label(self, clip_frames: np.ndarray) -> str:
        if self.backend is None:
            return "<auto-labeled by VLM: stub>"
        return self.backend(self.PROMPT, clip_frames)
