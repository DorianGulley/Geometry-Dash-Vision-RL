from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .simulator import State


def lowdim_observation(state: State) -> dict[str, Any]:
    # Stable, explicit dict for debugging and simple baselines.
    return asdict(state)


def surface_to_rgb_array(surface: "Any") -> np.ndarray:
    # Returns HxWx3 uint8
    import pygame

    arr = pygame.surfarray.array3d(surface)  # WxHx3
    arr = np.transpose(arr, (1, 0, 2))  # HxWx3
    return arr.astype(np.uint8, copy=False)

