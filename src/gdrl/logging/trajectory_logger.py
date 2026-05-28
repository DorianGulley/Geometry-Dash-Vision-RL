from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gdrl.levels import Level
from gdrl.sim import Action, EpisodeResult, State
from gdrl.sim.physics import PlayerBody


LOGGING_VERSION = 1

STATES_CSV_COLUMNS = [
    "episode_id",
    "level_id",
    "timestep",
    "action_jump_pressed",
    "player_x",
    "player_y",
    "player_vy",
    "grounded",
    "progress",
    "done",
    "completed",
    "died",
    "reward",
    "frame_path",
]


@dataclass
class TrajectoryLogger:
    """
    Writes one episode under `dataset_root/episodes/episode_XXXXXX/`.

    See PROGRESS.md (Milestone 3) for field definitions in metadata.json and states.csv.
    """

    level: Level
    dataset_root: Path
    provider: str = "unknown"
    level_path: str | None = None
    capture_every_n_steps: int = 0
    episode_id: str | None = None

    episode_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    _rows: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.episode_id = self.episode_id or self._next_episode_id(self.dataset_root)
        self.episode_dir = self.dataset_root / "episodes" / self.episode_id
        self.frames_dir = self.episode_dir / "frames"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if self.capture_every_n_steps > 0:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _next_episode_id(dataset_root: Path) -> str:
        episodes_dir = dataset_root / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        max_n = 0
        for p in episodes_dir.iterdir():
            if p.is_dir() and p.name.startswith("episode_"):
                try:
                    max_n = max(max_n, int(p.name.split("_", 1)[1]))
                except ValueError:
                    continue
        return f"episode_{max_n + 1:06d}"

    def log_step(
        self,
        state: State,
        action: Action,
        *,
        frame_path: str | None = None,
        episode_done: bool = False,
        episode_result: EpisodeResult | None = None,
    ) -> None:
        """Record one (state, action) pair. Call once per simulation step."""
        done = episode_done
        completed = bool(episode_result.completed) if episode_done and episode_result else False
        died = bool(episode_result.died) if episode_done and episode_result else False
        progress = float(episode_result.progress) if episode_done and episode_result else state.progress

        self._rows.append(
            {
                "episode_id": self.episode_id,
                "level_id": self.level.meta.level_id,
                "timestep": state.timestep,
                "action_jump_pressed": int(action.jump_pressed),
                "player_x": state.player_x,
                "player_y": state.player_y,
                "player_vy": state.player_vy,
                "grounded": int(state.grounded),
                "progress": progress,
                "done": int(done),
                "completed": int(completed),
                "died": int(died),
                "reward": 0.0,
                "frame_path": frame_path or "",
            }
        )

    def finish(self, episode: EpisodeResult) -> Path:
        """Write metadata.json and states.csv. Returns episode directory."""
        if self._rows:
            last = self._rows[-1]
            if not last["done"]:
                last["done"] = 1
                last["completed"] = int(episode.completed)
                last["died"] = int(episode.died)
                last["progress"] = float(episode.progress)

        metadata = self._build_metadata(episode)
        meta_path = self.episode_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=False)
            f.write("\n")

        csv_path = self.episode_dir / "states.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STATES_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

        return self.episode_dir

    def _build_metadata(self, episode: EpisodeResult) -> dict[str, Any]:
        return {
            "logging_version": LOGGING_VERSION,
            "episode_id": self.episode_id,
            "level": {
                "level_id": self.level.meta.level_id,
                "name": self.level.meta.name,
                "path": self.level_path,
                "split": self.level.meta.split,
                "tags": list(self.level.meta.tags),
                "schema_version": self.level.meta.schema_version,
                "width": self.level.width,
                "height": self.level.height,
                "tile_size": self.level.tile_size,
            },
            "rollout": {
                "provider": self.provider,
                "num_steps": len(self._rows),
                "capture_every_n_steps": self.capture_every_n_steps,
            },
            "episode": {
                "done": episode.done,
                "completed": episode.completed,
                "died": episode.died,
                "progress": episode.progress,
                "reason": episode.reason,
            },
            "physics": {
                "fixed_dt": self.level.physics.fixed_dt,
                "gravity": self.level.physics.gravity,
                "jump_velocity": self.level.physics.jump_velocity,
                "scroll_speed": self.level.physics.scroll_speed,
            },
            "camera": {
                "screen_width": self.level.camera.screen_width,
                "screen_height": self.level.camera.screen_height,
                "player_screen_x": self.level.camera.player_screen_x,
            },
        }

    def should_capture_frame(self, timestep: int) -> bool:
        n = self.capture_every_n_steps
        return n > 0 and (timestep % n == 0)

    def frame_relpath(self, timestep: int) -> str:
        return f"frames/{timestep:06d}.png"

    def frame_abspath(self, timestep: int) -> Path:
        return self.episode_dir / self.frame_relpath(timestep)

    def write_from_rollout(
        self,
        result: "RolloutResult",
        *,
        renderer: Any | None = None,
    ) -> Path:
        """
        Persist a completed rollout. If `renderer` is given and capture is enabled,
        saves PNG frames (no raw arrays in JSON).
        """
        if result.episode is None:
            raise ValueError("Cannot log rollout without an EpisodeResult.")

        for i, (state, action) in enumerate(result.trajectory):
            is_last = i == len(result.trajectory) - 1
            frame_path: str | None = None
            if renderer is not None and self.should_capture_frame(state.timestep):
                relpath = self.frame_relpath(state.timestep)
                self._capture_frame(renderer, state, relpath)
                frame_path = relpath
            self.log_step(
                state,
                action,
                frame_path=frame_path,
                episode_done=is_last,
                episode_result=result.episode if is_last else None,
            )

        return self.finish(result.episode)

    def _capture_frame(self, renderer: Any, state: State, relpath: str) -> str:
        import pygame

        body = _player_body_for_render(self.level, state)
        surf = renderer.render_to_surface(player=body, timestep=state.timestep)
        out = self.episode_dir / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surf, out)
        return relpath


def _player_body_for_render(level: Level, state: State) -> PlayerBody:
    s = level.tile_size
    w = 0.8 * s
    h = 0.9 * s
    return PlayerBody(
        x=state.player_x,
        y=state.player_y,
        vx=0.0,
        vy=state.player_vy,
        w=float(w),
        h=float(h),
        grounded=state.grounded,
    )
