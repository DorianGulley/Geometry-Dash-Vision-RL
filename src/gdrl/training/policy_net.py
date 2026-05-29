from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PolicyNetConfig:
    height: int = 96
    width: int = 96
    actions: int = 2


class TinyJumpCNN(nn.Module):
    """Small one-frame policy: grayscale image -> no-jump/jump logits."""

    def __init__(self, config: PolicyNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyNetConfig()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Linear(self._feature_dim(), self.config.actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.head(self.features(x))

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = obs_to_tensor(obs)
            logits = self(x)
            return int(torch.argmax(logits, dim=1).item())

    def _feature_dim(self) -> int:
        sample = torch.zeros(1, 1, self.config.height, self.config.width)
        return int(self.features(sample).shape[1])


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected observation shape (1,H,W) or (N,1,H,W), got {tuple(x.shape)}.")
    return x


def save_policy(model: TinyJumpCNN, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_policy(path: str | Path, config: PolicyNetConfig | None = None) -> TinyJumpCNN:
    model = TinyJumpCNN(config)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
