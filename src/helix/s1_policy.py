"""System 1 (S1) — the 200 Hz cross-attention visuomotor policy.

Matches the Helix description:

  "S1, an 80M parameter cross-attention encoder-decoder transformer, handles
   low-level control... While S1 receives the same image and state inputs as S2,
   it processes them at a higher frequency to enable more responsive closed-loop
   control. The latent vector from S2 is projected into S1's token space and
   concatenated with visual features from S1's vision backbone along the sequence
   dimension, providing task conditioning."

  "S1 outputs full upper body humanoid control at 200hz, including desired wrist
   poses, finger flexion and abduction control, and torso and head orientation
   targets. We append to the action space a synthetic 'percentage task completion'
   action, allowing Helix to predict its own termination condition."
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .latent import LatentInjection
from .vision import MultiScaleConvBackbone, StateTokenizer

from .config import System1Config, ACTION_DIM


class ActionDecoder(nn.Module):
    """Decoder-only transformer that cross-attends to visual + latent tokens.

    Query sequence: previous actions (autoregressive) plus state tokens.
    """

    def __init__(self, cfg: System1Config):
        super().__init__()
        self.d_model = cfg.d_model
        layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.n_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.action_dim),
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        """query: (B, Tq, d_model); memory: (B, Tm, d_model) -> actions (B, Tq, A)."""
        b, tq, _ = query.shape
        query = query + self.pos_embed[:, :tq]
        out = self.decoder(query, memory)
        return self.out_proj(out)


class System1(nn.Module):
    """End-to-end trainable S1; the S2 latent vector arrives via `latent`."""

    def __init__(self, cfg: System1Config):
        super().__init__()
        self.cfg = cfg
        self.vision = MultiScaleConvBackbone(
            cfg.vision_channels, cfg.image_size, cfg.d_model)
        self.state_tok = StateTokenizer(cfg.state_dim, cfg.d_model)
        self.latent_inject = LatentInjection(cfg.latent_dim, cfg.d_model)
        self.action_embed = nn.Linear(cfg.action_dim, cfg.d_model)
        self.decoder = ActionDecoder(cfg)

    def forward(
        self,
        images: Tensor,
        robot_state: Tensor,
        latent: Tensor,
        prev_actions: Tensor,
    ) -> Tensor:
        """images: (B, 3, H, W); robot_state: (B, S); latent: (B, L);
        prev_actions: (B, Tq, A) -> next actions (B, Tq, A)."""
        vis = self.vision(images)                        # (B, T_vis, d)
        lat = self.latent_inject(latent)                 # (B, 1, d)  -> concatenated
        state = self.state_tok(robot_state)              # (B, 1, d)
        memory = torch.cat([vis, lat, state], dim=1)     # conditioning sequence

        act = self.action_embed(prev_actions)            # (B, Tq, d)
        return self.decoder(act, memory)


def forward_s1_at_200hz(s1: System1, obs, latent, action_chunk):
    """Public helper naming the 200 Hz loop explicitly.

    The real-time controller consumes the *latest* observation and the *most
    recent* S2 latent vector at the control frequency.
    """
    return s1(obs.images, obs.robot_state, latent, action_chunk)
