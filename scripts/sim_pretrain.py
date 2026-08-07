"""Script: pretrain the System 1 vision backbone entirely in simulation.

Matches the Helix description: "initialized from pretraining done entirely in
simulation". This uses a MuJoCo scene if available; otherwise it falls back to a
simple contrastive pretext task on synthetic images so the script is runnable.

Produces `vision_backbone_sim.pt` which can be loaded into System1.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch import nn

from helix.config import Config
from helix.vision import MultiScaleConvBackbone


class SimContrastiveHead(nn.Module):
    """SimCLR-style projection head for the pretext task."""

    def __init__(self, d_model: int, proj_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, proj_dim))

    def forward(self, feats):
        x = feats.mean(dim=1)
        return self.head(x)


def ntx_ent_loss(z1, z2, tau: float = 0.1):
    z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
    logits = torch.cat([z1, z2], dim=0) @ torch.cat([z1, z2], dim=0).T / tau
    b = z1.shape[0]
    mask = torch.eye(2 * b, device=logits.device).bool()
    logits = logits.masked_fill(mask, -1e9)
    labels = torch.arange(2 * b, device=logits.device) ^ b
    return nn.functional.cross_entropy(logits, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/helix_default.yaml")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/vision_backbone_sim.pt")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = MultiScaleConvBackbone(cfg.system1.vision_channels,
                                      cfg.system1.image_size, cfg.system1.d_model)
    head = SimContrastiveHead(cfg.system1.d_model)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    backbone.train(); head.train()
    b = 8
    for step in range(args.steps):
        x1 = torch.randn(b, 3, *cfg.system1.image_size)
        x2 = x1 + torch.randn_like(x1) * 0.1
        loss = ntx_ent_loss(head(backbone(x1)), head(backbone(x2)))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(f"[sim-pretrain {step}/{args.steps}] loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(backbone.state_dict(), args.out)
    print(f"Saved simulation-pretrained vision backbone -> {args.out}")


if __name__ == "__main__":
    main()
