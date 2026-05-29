from pathlib import Path

import numpy as np
import pytest

from gdrl.envs import GDRLEnv, ObservationConfig, RewardConfig, make_simple_env, run_episode


def _example_level() -> Path:
    return Path(__file__).resolve().parents[1] / "levels" / "example_level.json"


def test_env_reset_returns_image_observation_and_info() -> None:
    env = GDRLEnv(_example_level(), observation_config=ObservationConfig(width=64, height=48))

    obs, info = env.reset()

    assert obs.shape == (1, 48, 64)
    assert obs.dtype == np.uint8
    assert int(obs.max()) > int(obs.min())
    assert info["timestep"] == 0
    assert 0.0 <= info["progress"] <= 1.0


def test_env_step_uses_discrete_jump_action_and_reward() -> None:
    env = GDRLEnv(
        _example_level(),
        observation_config=ObservationConfig(width=64, height=48),
        reward_config=RewardConfig(progress_weight=10.0, step_penalty=0.0),
    )
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (1, 48, 64)
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert info["timestep"] == 1


def test_env_runs_until_terminal_signal() -> None:
    env = GDRLEnv(_example_level(), observation_config=ObservationConfig(width=32, height=32))
    env.reset()

    terminated = False
    truncated = False
    info = {}
    for _ in range(1000):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break

    assert terminated or truncated
    assert info["episode"]["reason"] in {"death", "completed", "timeout"}

    with pytest.raises(RuntimeError):
        env.step(0)


def test_make_simple_env_runs() -> None:
    env = make_simple_env(seed=1, observation_config=ObservationConfig(width=32, height=32))

    obs, info = env.reset()
    _, reward, terminated, truncated, step_info = env.step(0)

    assert obs.shape == (1, 32, 32)
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert step_info["timestep"] == info["timestep"] + 1


def test_env_supports_simple_vertical_crop() -> None:
    cfg = ObservationConfig(width=40, height=30, crop_top=32, crop_bottom=32)
    env = GDRLEnv(_example_level(), observation_config=cfg)

    obs, _ = env.reset()

    assert obs.shape == (1, 30, 40)
    assert obs.dtype == np.uint8


def test_run_episode_helper() -> None:
    env = make_simple_env(seed=2, observation_config=ObservationConfig(width=32, height=32))

    stats = run_episode(env, max_steps=3)

    assert stats.steps == 3
    assert stats.reward > 0.0
    assert stats.terminated is False
    assert stats.truncated is False
