from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical

from gdrl.envs import GDRLEnv

from .policy_net import TinyJumpCNN, obs_to_tensor


@dataclass(frozen=True)
class ReinforceConfig:
    episodes: int = 200
    gamma: float = 0.99
    lr: float = 1e-3
    entropy_weight: float = 0.01
    max_steps: int | None = None
    seed: int = 0
    log_every: int = 0


@dataclass
class EpisodeTrace:
    rewards: list[float] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))


@dataclass(frozen=True)
class TrainResult:
    rewards: list[float]
    completed: list[bool]


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    success_rate: float
    avg_reward: float
    avg_progress: float


ProgressCallback = Callable[[int, float, bool], None]


def train_reinforce(
    env: GDRLEnv,
    model: TinyJumpCNN | None = None,
    config: ReinforceConfig | None = None,
    on_episode: ProgressCallback | None = None,
) -> tuple[TinyJumpCNN, TrainResult]:
    cfg = config or ReinforceConfig()
    set_seed(cfg.seed)
    model = model or TinyJumpCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rewards: list[float] = []
    completed: list[bool] = []

    for episode_idx in range(1, cfg.episodes + 1):
        trace = run_policy_episode(env, model, max_steps=cfg.max_steps)
        loss = reinforce_loss(trace, cfg.gamma, cfg.entropy_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        done = trace_completed(trace)
        rewards.append(trace.total_reward)
        completed.append(done)
        if on_episode is not None:
            on_episode(episode_idx, trace.total_reward, done)

    return model, TrainResult(rewards, completed)


def evaluate_policy(env: GDRLEnv, model: TinyJumpCNN, *, episodes: int = 10) -> EvalResult:
    rewards: list[float] = []
    progress: list[float] = []
    completed = 0
    for _ in range(episodes):
        trace = run_greedy_episode(env, model)
        rewards.append(trace.total_reward)
        progress.append(trace_progress(trace))
        completed += int(trace_completed(trace))
    n = max(1, episodes)
    return EvalResult(episodes, completed / n, sum(rewards) / n, sum(progress) / n)


def run_policy_episode(env: GDRLEnv, model: TinyJumpCNN, *, max_steps: int | None = None) -> EpisodeTrace:
    obs, info = env.reset()
    trace = EpisodeTrace()
    done = False

    while not done:
        if max_steps is not None and len(trace.rewards) >= max_steps:
            break
        action, log_prob, entropy = sample_action(model, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.rewards.append(float(reward))
        trace.log_probs.append(log_prob)
        trace.entropies.append(entropy)
        trace.actions.append(action)
        done = terminated or truncated

    trace.info = info
    return trace


def run_greedy_episode(env: GDRLEnv, model: TinyJumpCNN, *, max_steps: int | None = None) -> EpisodeTrace:
    obs, info = env.reset()
    trace = EpisodeTrace()
    done = False

    while not done:
        if max_steps is not None and len(trace.rewards) >= max_steps:
            break
        action = model.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.rewards.append(float(reward))
        trace.actions.append(action)
        done = terminated or truncated

    trace.info = info
    return trace


def sample_action(model: TinyJumpCNN, obs: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
    logits = model(obs_to_tensor(obs))
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0)


def reinforce_loss(trace: EpisodeTrace, gamma: float, entropy_weight: float) -> torch.Tensor:
    returns = normalize(discounted_returns(trace.rewards, gamma))
    log_probs = torch.stack(trace.log_probs)
    entropies = torch.stack(trace.entropies)
    policy_loss = -(log_probs * returns).sum()
    entropy_bonus = entropies.sum() * entropy_weight
    return policy_loss - entropy_bonus


def trace_completed(trace: EpisodeTrace) -> bool:
    return bool(trace.info.get("episode", {}).get("completed", False))


def trace_progress(trace: EpisodeTrace) -> float:
    episode = trace.info.get("episode", {})
    return float(episode.get("progress", trace.info.get("progress", 0.0)))


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def normalize(x: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x
    std = x.std()
    if float(std) < 1e-6:
        return x - x.mean()
    return (x - x.mean()) / (std + 1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
