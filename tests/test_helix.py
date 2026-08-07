"""End-to-end forward/backward through the full HelixVLA pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from helix.config import Config, UPPER_BODY_DOF, ACTION_DIM, TERMINATION_ACTION_IDX
from helix.model import Batch, HelixVLA


def make_batch(cfg: Config, b: int = 2):
    return Batch(
        s1_images=torch.randn(b, 3, *cfg.system1.image_size),
        s1_state=torch.randn(b, 12),
        s2_images=torch.randn(b, 3, 224, 224),
        s2_text_ids=torch.randint(0, 256, (b, 64)),
        s2_state=torch.randn(b, 12),
        prev_actions=torch.randn(b, 3, ACTION_DIM),
        target_actions=torch.randn(b, 1, ACTION_DIM),
    )


def test_action_space_is_35_dof():
    assert len(UPPER_BODY_DOF) == 35
    assert ACTION_DIM == 36  # 35 DoF + termination
    assert TERMINATION_ACTION_IDX == 35
    assert set(dof for dof, _ in UPPER_BODY_DOF) >= {"l_wrist_yaw", "r_finger_4_flex",
                                                     "torso_pitch", "head_yaw"}


def test_full_pipeline_forward_backward():
    cfg = Config.from_yaml(os.path.join(os.path.dirname(__file__), "..",
                                        "configs", "helix_default.yaml"))
    model = HelixVLA(cfg)
    batch = make_batch(cfg)
    pred = model(batch)
    assert pred.shape == (2, 3, ACTION_DIM)  # (B, Tq, A) — next action at last token

    loss, metrics = model.loss(batch)
    loss.backward()
    assert set(metrics) == {"mse", "l1"}

    # End-to-end: gradients must reach System 2 through the latent vector.
    grad_ok = model.s2.latent_head.mlp[0].weight.grad is not None
    assert grad_ok, "no gradient in S2 -> latent vector communication broken"


def test_latent_dim_matches_s1_conditioning():
    cfg = Config.from_yaml(os.path.join(os.path.dirname(__file__), "..",
                                        "configs", "helix_default.yaml"))
    model = HelixVLA(cfg)
    batch = make_batch(cfg, b=1)
    s2_inputs = {"images": batch.s2_images, "token_ids": batch.s2_text_ids}
    lat = model.s2(s2_inputs, batch.s2_state)
    assert lat.shape == (1, cfg.system2.latent_dim)
