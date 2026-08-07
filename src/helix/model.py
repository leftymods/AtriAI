"""HelixVLA — the full end-to-end System 2 + System 1 model.

Matches the Helix description:

  "Helix is trained fully end-to-end, mapping from raw pixels and text commands to
   continuous actions with a standard regression loss. Gradients are backpropagated
   from S1 into S2 via the latent communication vector used to condition S1's
   behavior, allowing joint optimization of both components."

  "During training, we add a temporal offset between S1 and S2 inputs. This offset
   is calibrated to match the gap between S1 and S2's deployed inference latency,
   ensuring that the real-time control requirements during deployment are
   accurately reflected in training."
"""
from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

from .config import Config
from .s1_policy import System1
from .s2_vlm import System2


@dataclasses.dataclass
class Batch:
    """A training batch aligned to the S1/S2 temporal-offset design.

    `s2_*` inputs are delayed relative to `s1_*` inputs by the calibrated offset,
    mirroring the deployed latency gap between the two systems.
    """
    s1_images: Tensor        # (B, 3, H, W)  -- fast loop observations
    s1_state: Tensor         # (B, S)
    s2_images: Tensor        # (B, 3, H, W)  -- delayed (older) observations
    s2_state: Tensor         # (B, S)
    prev_actions: Tensor     # (B, Tq, A)  -- action chunk history
    target_actions: Tensor   # (B, Tq, A)  -- regression targets
    s2_text: list[str] = dataclasses.field(default_factory=list)  # raw instructions
    s2_text_ids: Tensor | None = None  # char token ids (tiny mode)


class HelixVLA(nn.Module):
    """Single set of network weights, end-to-end, one training stage."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.s2 = System2(cfg.system2)
        self.s1 = System1(cfg.system1)

    def forward(self, batch: Batch) -> Tensor:
        """Raw pixels + text -> continuous actions; gradients flow S1 -> S2.

        The latent vector is the single communication channel between systems,
        exactly as in the paper: S2 distills semantics, S1 is conditioned on it.
        """
        if self.cfg.system2.model_type == "vlm":
            s2_inputs = self.s2.prepare_inputs(batch.s2_text, batch.s2_images)
            s2_inputs["token_ids"] = batch.s2_text_ids  # unused in vlm mode
        else:
            s2_inputs = {"images": batch.s2_images, "token_ids": batch.s2_text_ids}
        latent = self.s2(s2_inputs, batch.s2_state)  # (B, latent_dim)
        actions = self.s1(
            images=batch.s1_images,
            robot_state=batch.s1_state,
            latent=latent,
            prev_actions=batch.prev_actions,
        )  # (B, Tq, A)
        return actions

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard regression loss over continuous actions (paper: "standard
        regression loss"). The termination action (index ACTION_DIM-1) is part of
        the same regression target.

        S1 outputs one action per control tick (200 Hz); `prev_actions` are the
        query history, so the next action is the LAST predicted token. It is
        compared against the single-step `target_actions`."""

        pred = self.forward(batch)              # (B, Tq, A)
        pred_next = pred[:, -1]                 # (B, A) -> next action only
        target = batch.target_actions[:, -1]    # (B, A)
        mse = nn.functional.mse_loss(pred_next, target)
        l1 = nn.functional.l1_loss(pred_next, target)
        metrics = {"mse": mse.detach(), "l1": l1.detach()}
        return mse, metrics

    def num_params(self) -> dict[str, int]:
        return {
            "s2": sum(p.numel() for p in self.s2.parameters()),
            "s1": sum(p.numel() for p in self.s1.parameters()),
        }
