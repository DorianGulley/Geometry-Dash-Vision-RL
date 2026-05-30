from __future__ import annotations

import numpy as np


class SpikeWindowPolicy:
    """Simple visual baseline: jump when a spike-like blob is near the player."""

    def __init__(
        self,
        *,
        row_start: int = 60,
        row_end: int = 72,
        col_min: float = 30.0,
        col_max: float = 36.0,
        threshold: int = 80,
        min_pixels: int = 12,
    ) -> None:
        self.row_start = row_start
        self.row_end = row_end
        self.col_min = col_min
        self.col_max = col_max
        self.threshold = threshold
        self.min_pixels = min_pixels

    def __call__(self, obs: np.ndarray, info: dict | None = None) -> int:
        return self.act(obs)

    def act(self, obs: np.ndarray) -> int:
        if obs.ndim != 3 or obs.shape[0] != 1:
            raise ValueError(f"Expected observation shape (1,H,W), got {obs.shape}.")
        window = obs[0, self.row_start : self.row_end, :]
        mask = window > self.threshold
        for left, right in bright_column_groups(mask):
            center = (left + right) / 2.0
            pixels = int(mask[:, left : right + 1].sum())
            if self.col_min <= center <= self.col_max and pixels >= self.min_pixels:
                return 1
        return 0


class SpikeTimingPolicy:
    """Estimate spike columns from the first grayscale frame and schedule jumps."""

    def __init__(
        self,
        *,
        row_start: int = 60,
        row_end: int = 72,
        threshold: int = 80,
        min_pixels: int = 10,
        screen_to_tile_scale: float = 3.1,
        screen_to_tile_offset: float = 4.0,
        jump_steps_per_tile: float = 6.25,
        jump_step_offset: float = -32.5,
        pulse_steps: int = 3,
    ) -> None:
        self.row_start = row_start
        self.row_end = row_end
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.screen_to_tile_scale = screen_to_tile_scale
        self.screen_to_tile_offset = screen_to_tile_offset
        self.jump_steps_per_tile = jump_steps_per_tile
        self.jump_step_offset = jump_step_offset
        self.pulse_steps = pulse_steps
        self._jump_steps: list[int] = []

    def __call__(self, obs: np.ndarray, info: dict | None = None) -> int:
        timestep = int((info or {}).get("timestep", 0))
        if timestep == 0:
            self._jump_steps = self._schedule(obs)
        return int(any(start <= timestep < start + self.pulse_steps for start in self._jump_steps))

    def _schedule(self, obs: np.ndarray) -> list[int]:
        centers = obstacle_like_centers(
            obs,
            row_start=self.row_start,
            row_end=self.row_end,
            threshold=self.threshold,
            min_pixels=self.min_pixels,
        )
        tile_xs = [
            (center - self.screen_to_tile_offset) / self.screen_to_tile_scale
            for center in centers
            if center > 20.0
        ]
        jump_steps = [
            max(0, int(round(self.jump_steps_per_tile * tile_x + self.jump_step_offset)))
            for tile_x in sorted(tile_xs)
        ]
        return dedupe_close_steps(jump_steps)


def obstacle_like_centers(
    obs: np.ndarray,
    *,
    row_start: int,
    row_end: int,
    threshold: int,
    min_pixels: int,
) -> list[float]:
    if obs.ndim != 3 or obs.shape[0] != 1:
        raise ValueError(f"Expected observation shape (1,H,W), got {obs.shape}.")
    window = obs[0, row_start:row_end, :]
    mask = window > threshold
    centers: list[float] = []
    for left, right in bright_column_groups(mask):
        group = mask[:, left : right + 1]
        heights = group.sum(axis=0)
        width = right - left + 1
        pixels = int(heights.sum())
        if 2 <= width <= 10 and pixels >= min_pixels:
            centers.append((left + right) / 2.0)
    return centers


def dedupe_close_steps(steps: list[int], *, min_gap: int = 12) -> list[int]:
    out: list[int] = []
    for step in steps:
        if not out or step - out[-1] >= min_gap:
            out.append(step)
    return out


def bright_column_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(mask.sum(axis=0) > 0)
    if len(cols) == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        groups.append((start, prev))
        start = prev = col
    groups.append((start, prev))
    return groups
