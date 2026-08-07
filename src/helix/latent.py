"""Единый непрерывный латентный вектор — канал связи S2 -> S1.

По описанию Helix: «S2 distills all semantic task-relevant information into a
single continuous latent vector, passed to S1 to condition its low-level actions.
The latent vector from S2 is projected into S1's token space and concatenated with
visual features from S1's vision backbone along the sequence dimension».

Здесь — торрent/функции вокруг этого вектора + шум для регуляризации связи.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentProjector(nn.Module):
    """Головка System 2: последний токен VLM -> один непрерывный вектор L.

    В оригинале S2 «distills» семантику в единый вектор; мы учим этот вектор
    нести всю информацию о задаче (текст + сцена), чтобы S1 мог по нему действовать.
    """

    def __init__(self, vlm_hidden: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(vlm_hidden),
            nn.Linear(vlm_hidden, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, last_hidden: Tensor) -> Tensor:
        """last_hidden: (B, H) — скрытое состояние последнего токена VLM."""
        return self.mlp(last_hidden)


class LatentInjection(nn.Module):
    """Инжекция латентного вектора в System 1.

    Воспроизводит строку: «The latent vector from S2 is projected into S1's token
    space and concatenated with visual features from S1's vision backbone along the
    sequence dimension, providing task conditioning».
    """

    def __init__(self, latent_dim: int, d_model: int):
        super().__init__()
        self.project = nn.Linear(latent_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, latent: Tensor) -> Tensor:
        """Возвращает conditioning-токены (B, 1, d_model) для конкатенации."""
        return self.norm(self.project(latent)).unsqueeze(1)


def make_communication_noise(latent: Tensor, std: float = 0.01) -> Tensor:
    """Шум на латентном векторе (drop-out на связи) — повышает устойчивость
    к асинхронности S2 на деплое (неактуальный латентный вектор)."""
    return latent + torch.randn_like(latent) * std
