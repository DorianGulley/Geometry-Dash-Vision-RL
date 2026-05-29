from __future__ import annotations

from dataclasses import dataclass

from gdrl.levels import Level, TileType

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
        self._jump_velocity = float(level.physics.jump_velocity)

        self._timestep = 0
        self._done = False
        self._dead = False
        self._completed = False
        self._reason = ""

        self.player = self._spawn_player()

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
        self._snap_player_to_ground()
        return self.state()

    def progress(self) -> float:
        s = self.level.tile_size
        start_x = self._spawn_x()
        complete_x = self.level.end.x * s - self.player.w
        denom = max(1.0, complete_x - start_x)
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

    def _spawn_x(self) -> float:
        s = self.level.tile_size
        w = 0.8 * s
        return float(self.level.start.x * s + (s - w) * 0.5)

    def _snap_player_to_ground(self) -> None:
        s = self.level.tile_size
        bottom = self.player.y + self.player.h
        candidates = [self.level.floor_y() * s]
        candidates.extend(t.y * s for t in self.level.tiles if t.type == TileType.BLOCK and self._overlaps_tile_x(t.x))
        top = nearest_ground_top(candidates, bottom, max_snap=s * 2.0)
        if top is None:
            return
        self.player.y = top - self.player.h
        self.player.vy = 0.0
        self.player.grounded = True

    def _overlaps_tile_x(self, tx: int) -> bool:
        s = self.level.tile_size
        left = tx * s
        right = left + s
        return self.player.x < right and self.player.x + self.player.w > left

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


def nearest_ground_top(candidates: list[float], bottom: float, *, max_snap: float) -> float | None:
    below = [top for top in candidates if 0.0 <= top - bottom <= max_snap]
    if not below:
        return None
    return min(below, key=lambda top: top - bottom)
