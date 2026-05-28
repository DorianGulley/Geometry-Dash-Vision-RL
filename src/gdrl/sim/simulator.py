from __future__ import annotations

import math
from dataclasses import dataclass

from gdrl.levels import Level

from .events import EpisodeResult
from .physics import PlayerBody, StepPhysicsResult, jump, step_physics


@dataclass(frozen=True)
class Action:
    jump_pressed: bool = False


@dataclass(frozen=True)
class State:
    timestep: int
    player_x: float
    player_y: float
    player_vy: float
    grounded: bool
    progress: float
    dead: bool
    completed: bool


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class Simulator:
    """
    Deterministic fixed-timestep simulator.

    - `reset()` initializes player state.
    - `step(action)` advances exactly one physics step of size `level.physics.fixed_dt`.
    """

    def __init__(self, level: Level, *, max_steps: int | None = None) -> None:
        self.level = level
        self.dt = float(level.physics.fixed_dt)
        self.max_steps = max_steps if max_steps is not None else self._default_max_steps()
        self.jump_height_tiles = 2.0
        self._jump_velocity = self._compute_jump_velocity()

        self._timestep = 0
        self._done = False
        self._dead = False
        self._completed = False
        self._reason = ""

        self.player = self._spawn_player()

    def _compute_jump_velocity(self) -> float:
        # Enforce a deterministic "exactly N tiles" ballistic apex height:
        # v^2 = 2 g h  ->  v = -sqrt(2 g h)
        g = float(self.level.physics.gravity)
        h = float(self.jump_height_tiles) * float(self.level.tile_size)
        return -math.sqrt(max(0.0, 2.0 * g * h))

    def _default_max_steps(self) -> int:
        # Conservative timeout: enough time to traverse level at scroll speed plus slack.
        world_w_px = self.level.width * self.level.tile_size
        seconds = max(1.0, world_w_px / max(1e-6, self.level.physics.scroll_speed))
        seconds *= 2.5
        return int(seconds / max(1e-9, self.dt))

    def _spawn_player(self) -> PlayerBody:
        s = self.level.tile_size
        # Slightly smaller than a tile to avoid edge-stickiness.
        w = 0.8 * s
        h = 0.9 * s
        x = self.level.start.x * s + (s - w) * 0.5
        y = self.level.start.y * s + (s - h) * 0.5
        return PlayerBody(
            x=float(x),
            y=float(y),
            vx=float(self.level.physics.scroll_speed),
            vy=0.0,
            w=float(w),
            h=float(h),
            grounded=False,
        )

    def reset(self) -> State:
        self._timestep = 0
        self._done = False
        self._dead = False
        self._completed = False
        self._reason = ""
        self.player = self._spawn_player()
        # Ensure we start "grounded" if placed on top of floor.
        _ = step_physics(self.level, self.player, 0.0)
        return self.state()

    def progress(self) -> float:
        s = self.level.tile_size
        start_x = self.level.start.x * s
        end_x = self.level.end.x * s
        denom = max(1.0, end_x - start_x)
        return _clamp01((self.player.x - start_x) / denom)

    def state(self) -> State:
        return State(
            timestep=self._timestep,
            player_x=self.player.x,
            player_y=self.player.y,
            player_vy=self.player.vy,
            grounded=self.player.grounded,
            progress=self.progress(),
            dead=self._dead,
            completed=self._completed,
        )

    def episode_result(self) -> EpisodeResult | None:
        if not self._done:
            return None
        return EpisodeResult(
            done=True,
            completed=self._completed,
            died=self._dead,
            progress=self.progress(),
            reason=self._reason,
        )

    def _check_completion(self) -> bool:
        # Complete when player's leading edge reaches the end tile column.
        end_x_px = self.level.end.x * self.level.tile_size
        return (self.player.x + self.player.w) >= end_x_px

    def step(self, action: Action) -> tuple[State, EpisodeResult | None]:
        if self._done:
            return self.state(), self.episode_result()

        if action.jump_pressed:
            jump(self.player, self._jump_velocity)

        phy: StepPhysicsResult = step_physics(self.level, self.player, self.dt)

        self._timestep += 1

        if phy.died:
            self._done = True
            self._dead = True
            self._reason = "death"
        elif self._check_completion():
            self._done = True
            self._completed = True
            self._reason = "completed"
        elif self._timestep >= self.max_steps:
            self._done = True
            self._reason = "timeout"

        return self.state(), self.episode_result()

