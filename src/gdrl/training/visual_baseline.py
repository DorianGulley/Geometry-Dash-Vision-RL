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
