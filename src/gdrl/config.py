from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    fixed_dt: float = 1.0 / 60.0
    display_fps: int = 60
    capture_every_n_steps: int = 0  # 0 disables capture

    gravity: float = 2200.0
    jump_velocity: float = -750.0
    scroll_speed: float = 300.0

    tile_size: int = 32
    screen_width: int = 960
    screen_height: int = 540

