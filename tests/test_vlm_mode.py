"""Tests the VLM code path with a fake processor/model (no downloads needed).

Monkeypatches the transformers entry points inside helix.s2_vlm to verify that
prepare_inputs -> forward -> latent wiring is correct and end-to-end gradients
still reach the VLM when model_type='vlm'.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

import helix.s2_vlm as s2v
from helix.config import Config, System2Config


class FakeConfig:
    hidden_size = 64


class FakeOut:
    last_hidden_state = None


class FakeVLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = torch.nn.Linear(16, config.hidden_size)
        self.device = torch.device("cpu")

    def forward(self, **kw):
        input_ids = kw.get("input_ids", torch.zeros(1, 8, dtype=torch.long))
        b, t = input_ids.shape
        x = torch.randn(b, t, 16)
        out = FakeOut()
        out.last_hidden_state = self.backbone(x)
        return out

    def generate(self, **kw):
        return torch.zeros(kw["input_ids"].shape[0], 8, dtype=torch.long)


class FakeProcessor:
    def __init__(self):
        self.tok = {"a": 1}

    def __call__(self, text=None, images=None, padding=True, return_tensors=None, **kw):
        n = len(text) if text else 1
        return {"input_ids": torch.full((n, 8), 1, dtype=torch.long),
                "attention_mask": torch.ones(n, 8, dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "fake prompt"

    def decode(self, ids, skip_special_tokens=True):
        return "fake reply"


def _patch():
    s2v._HF_AVAILABLE = True
    s2v.AutoModel = type("AutoModel", (), {"from_pretrained": staticmethod(
        lambda *a, **k: FakeVLM(FakeConfig()))})
    s2v.AutoProcessor = type("AutoProcessor", (), {"from_pretrained": staticmethod(
        lambda *a, **k: FakeProcessor())})


def _vlm_cfg():
    cfg = Config()
    cfg.system2 = System2Config(model_type="vlm", vlm_name="fake", latent_dim=32)
    cfg.system1.latent_dim = 32  # latent dim must match S2 output
    return cfg


def test_vlm_prepare_inputs_and_latent_shape():
    _patch()
    from helix.model import HelixVLA
    model = HelixVLA(_vlm_cfg())
    inputs = model.s2.prepare_inputs(["pick up the cup"], torch.randn(1, 3, 224, 224))
    assert "input_ids" in inputs and "attention_mask" in inputs
    lat = model.s2(inputs, torch.randn(1, 12))
    assert lat.shape == (1, 32)


def test_vlm_end_to_end_grad_reaches_vlm():
    _patch()
    from helix.model import HelixVLA
    model = HelixVLA(_vlm_cfg())
    from helix.model import Batch
    batch = Batch(
        s1_images=torch.randn(1, 3, 224, 224),
        s1_state=torch.randn(1, 12),
        s2_images=torch.randn(1, 3, 224, 224),
        s2_state=torch.randn(1, 12),
        prev_actions=torch.randn(1, 3, 36),
        target_actions=torch.randn(1, 1, 36),
        s2_text=["pick up the cup"],
    )
    loss, _ = model.loss(batch)
    loss.backward()
    assert model.s2.vlm.backbone.weight.grad is not None, \
        "end-to-end gradient did not reach the VLM"
