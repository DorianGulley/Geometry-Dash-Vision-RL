from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gdrl.levels import Level, load_level, validate_level
from gdrl.levels import SimpleLevelConfig, make_simple_level
from gdrl.sim import Action, EpisodeResult, Simulator, State
from gdrl.sim.renderer import RenderConfig, Renderer

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only in minimal installs
    gym = None
    spaces = None

ActionLike = int | np.integer | bool | Action
Policy = Callable[[np.ndarray, dict[str, Any]], ActionLike]


@dataclass(frozen=True)
class ObservationConfig:
    """One grayscale policy frame, returned as (1, height, width)."""

    width: int = 96
    height: int = 96
    crop_top: int = 96
    crop_bottom: int = 96
    smooth_scale: bool = False

    def shape(self) -> tuple[int, ...]:
        return (1, self.height, self.width)


@dataclass(frozen=True)
class RewardConfig:
    """Default reward shaping for first-pass RL training."""

    progress_weight: float = 10.0
    completion_bonus: float = 50.0
    death_penalty: float = -25.0
    timeout_penalty: float = -5.0
    step_penalty: float = -0.01


@dataclass(frozen=True)
class EpisodeStats:
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def load_checked_level(level: str | Path | Level, *, validate: bool = True) -> Level:
    out = load_level(level) if isinstance(level, str | Path) else level
    if validate:
        require_valid_level(out)
    return out


def require_valid_level(level: Level) -> None:
    errors = validate_level(level)
    if errors:
        joined = "; ".join(f"{e.code}: {e.message}" for e in errors)
        raise ValueError(f"Invalid level: {joined}")


def make_spaces(config: ObservationConfig) -> tuple[Any, Any] | tuple[None, None]:
    if spaces is None:
        return None, None
    return spaces.Discrete(2), spaces.Box(0, 255, config.shape(), np.uint8)


def coerce_action(action: ActionLike) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, bool):
        return Action(jump_pressed=action)
    value = int(action)
    if value not in (0, 1):
        raise ValueError(f"Expected action 0 or 1, got {value}.")
    return Action(jump_pressed=bool(value))


def terminal_flags(episode: EpisodeResult | None) -> tuple[bool, bool]:
    terminated = bool(episode and (episode.completed or episode.died))
    truncated = bool(episode and episode.reason == "timeout")
    return terminated, truncated


def reward_delta(
    prev_progress: float,
    state: State,
    episode: EpisodeResult | None,
    config: RewardConfig,
) -> float:
    reward = (state.progress - prev_progress) * config.progress_weight
    reward += config.step_penalty
    if episode is None:
        return float(reward)
    if episode.completed:
        reward += config.completion_bonus
    if episode.died:
        reward += config.death_penalty
    if episode.reason == "timeout":
        reward += config.timeout_penalty
    return float(reward)


def state_info(state: State, episode: EpisodeResult | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestep": state.timestep,
        "progress": state.progress,
        "player_x": state.player_x,
        "player_y": state.player_y,
        "player_vy": state.player_vy,
        "grounded": state.grounded,
    }
    if episode is not None:
        info["episode"] = episode_info(episode)
    return info


def episode_info(episode: EpisodeResult) -> dict[str, Any]:
    return {
        "done": episode.done,
        "completed": episode.completed,
        "died": episode.died,
        "progress": episode.progress,
        "reason": episode.reason,
    }


class GDRLEnv(gym.Env if gym is not None else object):
    """
    Gymnasium-compatible wrapper around the deterministic simulator.

    Actions:
    - 0: no jump
    - 1: press jump this timestep

    Observations are one-frame uint8 grayscale image tensors.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        level: str | Path | Level,
        *,
        observation_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int | None = None,
        validate: bool = True,
    ) -> None:
        if gym is not None:
            super().__init__()

        self.level = load_checked_level(level, validate=validate)
        self.observation_config = observation_config or ObservationConfig()
        self.reward_config = reward_config or RewardConfig()
        self.sim = Simulator(self.level, max_steps=max_steps)
        self.renderer = Renderer(self.level, config=RenderConfig(draw_grid=False, draw_hud=False))
        self._init_pygame()
        self.action_space, self.observation_space = make_spaces(self.observation_config)

        self._done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        state = self.sim.reset()
        self._done = False
        return self._observation(), state_info(state, None)

    def step(self, action: ActionLike) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")

        prev_progress = self.sim.progress()
        state, episode = self.sim.step(coerce_action(action))
        reward = reward_delta(prev_progress, state, episode, self.reward_config)
        terminated, truncated = terminal_flags(episode)
        self._done = terminated or truncated

        return self._observation(), reward, terminated, truncated, state_info(state, episode)

    def render(self) -> np.ndarray:
        """Return an RGB array at the renderer's native camera resolution."""
        surf = self.renderer.render_to_surface(player=self.sim.player, timestep=self.sim.state().timestep)
        return surface_to_rgb(surf)

    def close(self) -> None:
        pass

    def _observation(self) -> np.ndarray:
        surf = self.renderer.render_to_surface(player=self.sim.player, timestep=self.sim.state().timestep)
        return observe_surface(surf, self.observation_config)

    @staticmethod
    def _init_pygame() -> None:
        import pygame

        if not pygame.get_init():
            pygame.init()


def observe_surface(surface: Any, config: ObservationConfig) -> np.ndarray:
    rgb = surface_to_rgb(scale_surface(crop_surface(surface, config), config))
    obs = np.transpose(grayscale(rgb), (2, 0, 1))
    return np.ascontiguousarray(obs, dtype=np.uint8)


def crop_surface(surface: Any, config: ObservationConfig) -> Any:
    height = surface.get_height()
    top = max(0, min(config.crop_top, height - 1))
    bottom = max(top + 1, height - max(0, config.crop_bottom))
    if top == 0 and bottom == height:
        return surface
    return surface.subsurface((0, top, surface.get_width(), bottom - top))


def scale_surface(surface: Any, config: ObservationConfig) -> Any:
    import pygame

    size = (config.width, config.height)
    if (surface.get_width(), surface.get_height()) == size:
        return surface
    scaler = pygame.transform.smoothscale if config.smooth_scale else pygame.transform.scale
    return scaler(surface, size)


def surface_to_rgb(surface: Any) -> np.ndarray:
    import pygame

    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8, copy=False)


def grayscale(rgb: np.ndarray) -> np.ndarray:
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray.astype(np.uint8)[:, :, None]


def make_simple_env(
    *,
    seed: int | None = None,
    observation_config: ObservationConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> GDRLEnv:
    level = make_simple_level(SimpleLevelConfig(seed=seed))
    return GDRLEnv(level, observation_config=observation_config, reward_config=reward_config)


def run_episode(env: GDRLEnv, policy: Policy | None = None, *, max_steps: int | None = None) -> EpisodeStats:
    obs, info = env.reset()
    total = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if max_steps is not None and steps >= max_steps:
            truncated = True
            break
        action = 0 if policy is None else policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1

    return EpisodeStats(steps, total, terminated, truncated, info)
