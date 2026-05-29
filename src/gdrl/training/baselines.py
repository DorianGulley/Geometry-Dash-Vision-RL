from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from gdrl.envs import GDRLEnv, run_episode
from gdrl.levels import load_level


@dataclass(frozen=True)
class EvalRow:
    level_id: str
    episodes: int
    success_rate: float
    avg_progress: float
    avg_reward: float


def random_policy(seed: int | None = None):
    rng = random.Random(seed)

    def policy(obs, info) -> int:
        return rng.randint(0, 1)

    return policy


def eval_random(level_paths: list[str | Path], *, episodes: int = 20, seed: int = 0) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for level_path in level_paths:
        rows.append(eval_random_level(level_path, episodes=episodes, seed=seed))
    return rows


def eval_random_level(level_path: str | Path, *, episodes: int, seed: int) -> EvalRow:
    level = load_level(level_path)
    completed = 0
    progress = 0.0
    reward = 0.0

    for i in range(episodes):
        env = GDRLEnv(level)
        stats = run_episode(env, random_policy(seed + i))
        episode = stats.info.get("episode", {})
        completed += int(bool(episode.get("completed", False)))
        progress += float(episode.get("progress", stats.info.get("progress", 0.0)))
        reward += stats.reward

    n = max(1, episodes)
    return EvalRow(level.meta.level_id, episodes, completed / n, progress / n, reward / n)
