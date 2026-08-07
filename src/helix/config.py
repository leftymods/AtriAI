"""Конфигурация и пространство действий 35-DoF верхнего тела.

Соответствует описанию Helix: «Helix coordinates a 35-DoF action space at 200Hz,
controlling everything from individual finger movements to end-effector trajectories,
head gaze, and torso posture».
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class System2Config:
    model_type: str = "tiny"          # "vlm" | "tiny"
    vlm_name: str | None = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_freeze: bool = False
    lora: bool = False                # QLoRA fine-tune of the VLM (needs peft+bitsandbytes)
    lora_r: int = 16
    lora_alpha: int = 32
    latent_dim: int = 512
    rate_hz: float = 7.5


@dataclass
class System1Config:
    latent_dim: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 4096
    action_dim: int = 36             # 35 DoF + % completion
    action_rate_hz: float = 200.0
    vision_channels: tuple = (64, 128, 256, 512)
    image_size: tuple = (224, 224)
    pretrained_in_sim: bool = True
    state_dim: int = 12              # wrist pose (7) + finger positions (5)


@dataclass
class DataConfig:
    teleop_hours: float = 500.0
    obs_history: int = 3
    temporal_offset_ms: float = 120.0
    image_fps: int = 30
    num_workers: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    max_steps: int = 250_000
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50


@dataclass
class DeployConfig:
    s2_device: str = "cuda:0"
    s1_device: str = "cuda:1"
    shared_memory_key: str = "helix_latent"


@dataclass
class Config:
    system2: System2Config = field(default_factory=System2Config)
    system1: System1Config = field(default_factory=System1Config)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            system2=_make(System2Config, raw.get("system2", {})),
            system1=_make(System1Config, raw.get("system1", {})),
            data=_make(DataConfig, raw.get("data", {})),
            train=_make(TrainConfig, raw.get("train", {})),
            deploy=_make(DeployConfig, raw.get("deploy", {})),
        )


def _make(cls: type, data: dict[str, Any]) -> Any:
    data = {k: v for k, v in data.items() if k != "vision"}
    return cls(**data)


# --- Пространство действий верхнего тела (35 DoF) ---

#: Подробное описание каждой степени свободы — для читаемости кода и логов.
UPPER_BODY_DOF = [
    # torso: 3 (yaw, pitch, roll)
    ("torso_yaw", "rad"),
    ("torso_pitch", "rad"),
    ("torso_roll", "rad"),
    # head: 2 (yaw, pitch)
    ("head_yaw", "rad"),
    ("head_pitch", "rad"),
    # left arm: 6 DoF (shoulder x3, elbow, wrist x2) + wrist pose targets
    ("l_shoulder_roll", "rad"), ("l_shoulder_pitch", "rad"), ("l_shoulder_yaw", "rad"),
    ("l_elbow", "rad"),
    ("l_wrist_yaw", "rad"), ("l_wrist_pitch", "rad"),
    # right arm: 6 DoF
    ("r_shoulder_roll", "rad"), ("r_shoulder_pitch", "rad"), ("r_shoulder_yaw", "rad"),
    ("r_elbow", "rad"),
    ("r_wrist_yaw", "rad"), ("r_wrist_pitch", "rad"),
    # fingers: 5 per hand, flexion
    *[(f"l_finger_{i}_flex", "rad") for i in range(5)],
    *[(f"r_finger_{i}_flex", "rad") for i in range(5)],
    # thumb abduction per hand
    ("l_thumb_abd", "rad"), ("r_thumb_abd", "rad"),
    # end-effector positions (desired wrist pose targets, cartesian + quat)
    ("l_ee_x", "m"), ("l_ee_y", "m"), ("l_ee_z", "m"),
    ("r_ee_x", "m"), ("r_ee_y", "m"), ("r_ee_z", "m"),
]

assert len(UPPER_BODY_DOF) == 35, f"ожидалось 35 DoF, получено {len(UPPER_BODY_DOF)}"

#: Синтетическое действие «% завершения задачи» — позволяет модели самой
#: решать, когда прекратить поведение (описание Helix, "prediction of its own
#: termination condition").
TERMINATION_ACTION_IDX = 35
ACTION_DIM = 36

#: Состояние робота: wrist pose (7 = position 3 + quat 4) + finger positions (5).
WRIST_POSE_DIM = 7
FINGER_POS_DIM = 5
STATE_DIM = WRIST_POSE_DIM + FINGER_POS_DIM
