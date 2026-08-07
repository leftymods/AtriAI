"""Script: end-to-end training on synthetic (or real) teleop data.

`steps` defaults to a tiny run so the pipeline is verifiable without GPUs/data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from helix.config import Config
from helix.dataset import Episode, TeleopDataset
from helix.model import HelixVLA
from helix.training import train


def make_synthetic_episode(cfg: Config, seed: int = 0) -> Episode:
    """Synthetic teleop episode for smoke-testing the full training loop."""
    rng = np.random.default_rng(seed)
    t = 600
    imgs = rng.random((t, 3, *cfg.system1.image_size), dtype=np.float32)
    states = rng.random((t, 12), dtype=np.float32)
    actions = rng.random((t, cfg.system1.action_dim), dtype=np.float32)
    return Episode(images=imgs, states=states, actions=actions,
                   text="Pick up the desert item", ep_id="synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/helix_default.yaml")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--data", default=None, help="directory of .npz teleop episodes")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.train.batch_size = 2
    cfg.data.num_workers = 0

    if args.data:
        from helix.dataset import load_episodes
        episodes = load_episodes(args.data)
    else:
        episodes = [make_synthetic_episode(cfg)]
    print(f"Using {len(episodes)} episode(s), {len(episodes[0].actions)} steps each")

    ds = TeleopDataset(episodes, cfg)
    model = HelixVLA(cfg)
    train(cfg, model, ds, out_dir=args.out, steps=args.steps)


if __name__ == "__main__":
    main()
