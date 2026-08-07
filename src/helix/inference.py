"""Streaming inference on the robot — the deployment design from the paper.

Matches the Helix description:

  "The inference pipeline splits across S2 (high-level latent planning) and S1
   (low-level control) models, each running on dedicated GPUs. S2 operates as an
   asynchronous background process, consuming the latest observation (onboard
   camera and robot state) and natural language commands. It continuously updates
   a shared memory latent vector that encodes the high-level behavioral intent.

   S1 executes as a separate real-time process, maintaining the critical 200Hz
   control loop required for smooth whole upper body action. It takes both the
   latest observation and the most recent S2 latent vector. The inherent speed
   difference between S2 and S1 inference naturally results in S1 operating with
   higher temporal resolution on robot observations, creating a tighter feedback
   loop for reactive control."
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config
from .model import HelixVLA


@dataclass
class SharedLatent:
    """The single shared-memory latent vector (updated by S2, read by S1)."""
    value: np.ndarray | None = None
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, latent: np.ndarray):
        with self.lock:
            self.value = latent
            self.updated_at = time.time()

    def read(self) -> np.ndarray | None:
        with self.lock:
            return self.value


class System2Process:
    """Async background loop at 7–9 Hz: observation + command -> latent vector."""

    def __init__(self, model: HelixVLA, s2_device: str, rate_hz: float = 7.5):
        self.model = model
        self.device = s2_device
        self.period = 1.0 / rate_hz
        self.model.s2.to(self.device)  # model-parallel: S2 on its own GPU

    def step(self, images, robot_state, command: str) -> np.ndarray:
        """Single S2 inference: distills semantics into the latent vector."""
        with torch.no_grad():
            img = torch.from_numpy(images).unsqueeze(0).to(self.device)
            state = torch.from_numpy(robot_state).unsqueeze(0).to(self.device)
            if self.model.cfg.system2.model_type == "vlm":
                s2_inputs = self.model.s2.prepare_inputs([command], img)
                lat = self.model.s2(s2_inputs, state)
            else:
                s2_inputs = {"images": img, "token_ids": torch.tensor(
                    [[min(255, ord(c)) for c in command[:512]]],
                    dtype=torch.long, device=self.device)}
                lat = self.model.s2(s2_inputs, state)
        return lat.squeeze(0).cpu().numpy()

    def run_forever(self, shared: SharedLatent, camera, command_provider, stop):
        """Async background process: continuously refreshes the shared latent."""
        while not stop.is_set():
            images, state = camera.latest()
            command = command_provider()
            latent = self.step(images, state, command)
            shared.write(latent)
            time.sleep(self.period)


class System1Process:
    """Real-time control loop at 200 Hz using the most recent latent vector."""

    def __init__(self, model: HelixVLA, s1_device: str, rate_hz: float = 200.0):
        self.model = model
        self.device = s1_device
        self.period = 1.0 / rate_hz
        self.model.s1.to(self.device)  # model-parallel: S1 on its own GPU

    def step(self, images, robot_state, latent: np.ndarray, prev_actions) -> np.ndarray:
        """Single S1 inference: latest observation + most recent latent -> actions."""
        with torch.no_grad():
            act = self.model.s1(
                images=torch.from_numpy(images).unsqueeze(0).to(self.device),
                robot_state=torch.from_numpy(robot_state).unsqueeze(0).to(self.device),
                latent=torch.from_numpy(latent).unsqueeze(0).to(self.device),
                prev_actions=torch.from_numpy(prev_actions).unsqueeze(0).to(self.device),
            )
        return act.squeeze(0).cpu().numpy()

    def run_forever(self, shared: SharedLatent, robot, stop):
        """Real-time 200 Hz loop; falls back to the last latent if S2 is slow."""
        prev = np.zeros((1, self.model.cfg.system1.action_dim), dtype=np.float32)
        while not stop.is_set():
            t0 = time.perf_counter()
            images, state = robot.latest()
            latent = shared.read()
            if latent is None:
                latent = np.zeros(self.model.cfg.system2.latent_dim, dtype=np.float32)
            actions = self.step(images, state, latent, prev)
            robot.send_actions(actions)     # 35-DoF upper body + termination
            prev = actions[np.newaxis, -1:] if actions.ndim == 1 else actions[-1:]
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self.period - elapsed))
