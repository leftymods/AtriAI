"""End-to-end training of HelixVLA.

Matches the Helix description:

  "Helix is trained fully end-to-end, mapping from raw pixels and text commands to
   continuous actions with a standard regression loss. Gradients are backpropagated
   from S1 into S2 via the latent communication vector used to condition S1's
   behavior, allowing joint optimization of both components. Helix requires no
   task-specific adaptation; it maintains a single training stage and single set
   of neural network weights without separate action heads or per-task fine-tuning
   stages."

  "During training, we add a temporal offset between S1 and S2 inputs."
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import TeleopDataset
from .model import Batch, HelixVLA


def collate_fn(batch: list[dict]):
    def stack(key):
        return torch.stack([b[key] for b in batch])

    return Batch(
        s1_images=stack("s1_images"),
        s1_state=stack("s1_state"),
        s2_images=stack("s2_images"),
        s2_text=[b["s2_text"] for b in batch],
        s2_text_ids=torch.tensor(  # char-level tokenizer for tiny mode
            [[min(255, ord(c)) for c in b["s2_text"][:512]] for b in batch],
            dtype=torch.long,
        ),
        s2_state=stack("s2_state"),
        prev_actions=stack("prev_actions"),
        target_actions=stack("target_actions"),
    )


def maybe_apply_lora(model: HelixVLA, cfg: Config) -> HelixVLA:
    """QLoRA: wrap only the VLM (S2) in PEFT adapters, keep S1 and the
    latent/state heads fully trainable. Returns the same model for tiny mode."""
    if cfg.system2.model_type != "vlm" or not cfg.system2.lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("install 'peft' to use system2.lora=True")
    targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora = LoraConfig(
        r=cfg.system2.lora_r, lora_alpha=cfg.system2.lora_alpha,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    model.s2.vlm = get_peft_model(model.s2.vlm, lora)
    # The rest of the graph (latent_head, state_proj, S1) stays trainable, so
    # end-to-end gradients still flow S1 -> S2 through the latent vector.
    return model


class LRScheduler:
    def __init__(self, lr: float, warmup: int, max_steps: int):
        self.lr, self.warmup, self.max_steps = lr, warmup, max_steps

    def at(self, step: int) -> float:
        if step < self.warmup:
            return self.lr * step / max(1, self.warmup)
        p = (step - self.warmup) / max(1, self.max_steps - self.warmup)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * p))


def train(
    cfg: Config,
    model: HelixVLA,
    dataset: TeleopDataset,
    out_dir: str = "checkpoints",
    steps: int | None = None,
):
    """Single training stage, single set of weights, regression loss only."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maybe_apply_lora(model, cfg)
    model.to(device)

    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = LRScheduler(cfg.train.lr, cfg.train.warmup_steps, cfg.train.max_steps)

    model.train()
    total = steps or cfg.train.max_steps
    step, epoch = 0, 0
    while step < total:
        for batch in loader:
            batch = move(batch, device)
            lr = sched.at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, metrics = model.loss(batch)
            opt.zero_grad()
            loss.backward()  # gradients flow back into S2 through the latent vector
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if step % cfg.train.log_every == 0:
                print(f"[step {step}/{total}] loss={loss.item():.4f} "
                      f"l1={metrics['l1'].item():.4f} lr={lr:.2e}")
            step += 1
            if step >= total:
                break
        epoch += 1

    save_model(model, cfg, out_dir)
    return model


def save_model(model: HelixVLA, cfg: Config, out_dir: str) -> None:
    """Saves S1 + the S2 heads/adapters. A quantized 7B VLM's full state_dict is
    not persisted; PEFT adapters are saved instead."""
    import os

    s1 = os.path.join(out_dir, "s1.pt")
    torch.save(model.s1.state_dict(), s1)
    if cfg.system2.model_type == "vlm":
        if cfg.system2.lora and hasattr(model.s2.vlm, "save_pretrained"):
            model.s2.vlm.save_pretrained(os.path.join(out_dir, "s2_lora"))
        torch.save(
            {"latent_head": model.s2.latent_head.state_dict(),
             "state_proj": model.s2.state_proj.state_dict()},
            os.path.join(out_dir, "s2_heads.pt"),
        )
    else:
        torch.save(model.s2.state_dict(), os.path.join(out_dir, "s2.pt"))
    print(f"Saved checkpoints -> {out_dir}")


def move(batch: Batch, device):
    for f in ("s1_images", "s1_state", "s2_images", "s2_text_ids",
              "s2_state", "prev_actions", "target_actions"):
        v = getattr(batch, f)
        if v is not None:
            setattr(batch, f, v.to(device))
    return batch
