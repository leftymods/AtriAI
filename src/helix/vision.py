"""Fully-convolutional multi-scale vision backbone for System 1.

Matches the Helix description:

  "S1, an 80M parameter cross-attention encoder-decoder transformer, handles
   low-level control. It relies on a fully convolutional, multi-scale vision
   backbone for visual processing, initialized from pretraining done entirely
   in simulation."

The backbone returns a *sequence* of visual tokens (flattened feature maps of the
two deepest scales), which System 1's decoder will cross-attend to and which the
S2 latent vector is concatenated with.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvStage(nn.Module):
    """A single scale: conv block downsample, doubling channels."""

    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MultiScaleConvBackbone(nn.Module):
    """Multi-scale convolutional vision backbone (no attention inside).

    Outputs tokens from two deepest scales -> (B, T_vis, d_model).
    """

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 512),
                 image_size: tuple[int, int] = (224, 224),
                 d_model: int = 512):
        super().__init__()
        self.stages = nn.ModuleList()
        cin = 3
        for i, cout in enumerate(channels):
            self.stages.append(ConvStage(cin, cout, stride=2))
            cin = cout
        h, w = image_size
        fh, fw = h // (2 ** len(channels)), w // (2 ** len(channels))
        self._small_shape = (fh, fw)
        # Each selected scale is projected independently to d_model so that
        # multi-scale features can be concatenated along the sequence dimension.
        self.projectors = nn.ModuleList(
            [nn.Linear(c, d_model) for c in channels[-2:]]
        )

    def forward(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> visual tokens (B, S_h*S_w + S2_h*S2_w, d_model)."""
        x = images
        features: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= len(self.stages) - 2:
                features.append(x)  # keep two deepest scales

        toks: list[Tensor] = []
        for f, proj in zip(features, self.projectors):
            b, c, h, w = f.shape
            toks.append(proj(f.flatten(2).transpose(1, 2)))  # (B, h*w, d_model)
        vis = torch.cat(toks, dim=1)
        return vis  # (B, T_vis, d_model)


class StateTokenizer(nn.Module):
    """Robot state (wrist pose + finger positions) -> tokens for the decoder."""

    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.mlp(state).unsqueeze(1)  # (B, 1, d_model)
