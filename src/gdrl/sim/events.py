from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    done: bool
    completed: bool
    died: bool
    progress: float  # [0,1]
    reason: str  # "completed" | "death" | "timeout" | "quit" | ...

