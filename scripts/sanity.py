"""Script: run a forward pass through the full HelixVLA pipeline (sanity check)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from helix.config import Config
from helix.model import Batch, HelixVLA


def main():
    cfg = Config.from_yaml("configs/helix_default.yaml")
    model = HelixVLA(cfg)
    print("Parameters:", model.num_params(), "total:",
          sum(model.num_params().values()))

    b = cfg.train.batch_size
    batch = Batch(
        s1_images=torch.randn(b, 3, *cfg.system1.image_size),
        s1_state=torch.randn(b, 12),
        s2_images=torch.randn(b, 3, 224, 224),
        s2_text_ids=torch.randint(0, 256, (b, 64)),
        s2_state=torch.randn(b, 12),
        prev_actions=torch.randn(b, 3, cfg.system1.action_dim),
        target_actions=torch.randn(b, 1, cfg.system1.action_dim),
    )
    loss, metrics = model.loss(batch)
    loss.backward()
    print(f"Forward+backward OK. loss={loss.item():.4f} mse={metrics['mse'].item():.4f}")
    print("Gradients reach System 2 (latent_head):",
          model.s2.latent_head.mlp[0].weight.grad is not None)


if __name__ == "__main__":
    main()
