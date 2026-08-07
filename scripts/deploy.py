"""Script: deploy the streaming S2/S1 inference on two devices.

Requires: a robot interface providing `latest()` and `send_actions(actions)`.
For a smoke test, a `SimRobot` stub is used.
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import torch

from helix.config import Config
from helix.inference import SharedLatent, System1Process, System2Process
from helix.model import HelixVLA


class SimRobot:
    """Stub robot: camera + action sink (no real hardware)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._images = np.zeros((3, *cfg.system1.image_size), dtype=np.float32)
        self._state = np.zeros(12, dtype=np.float32)
        self.actions = []

    def latest(self):
        return self._images, self._state

    def send_actions(self, actions):
        self.actions.append(np.asarray(actions).reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/helix_default.yaml")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--command", default="Pick up the desert item")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    model = HelixVLA(cfg).eval()

    # Smoke test without a second GPU: fall back to CPU when fewer than 2 devices.
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        cfg.deploy.s2_device = cfg.deploy.s1_device = "cpu"

    robot = SimRobot(cfg)
    shared = SharedLatent()
    stop = threading.Event()

    s2 = System2Process(model, cfg.deploy.s2_device, cfg.system2.rate_hz)
    s1 = System1Process(model, cfg.deploy.s1_device, cfg.system1.action_rate_hz)

    t2 = threading.Thread(target=s2.run_forever, args=(shared, robot, lambda: args.command, stop))
    t1 = threading.Thread(target=s1.run_forever, args=(shared, robot, stop))
    t2.start(); t1.start()

    import time
    time.sleep(args.seconds)
    stop.set(); t1.join(); t2.join()

    print(f"Ran {args.seconds}s: S2 at {cfg.system2.rate_hz} Hz, "
          f"S1 at {cfg.system1.action_rate_hz} Hz. "
          f"Actions sent to robot: {len(robot.actions)}")
    print("Latent vector seen by S1 (shape):", None if shared.value is None else shared.value.shape)


if __name__ == "__main__":
    main()
