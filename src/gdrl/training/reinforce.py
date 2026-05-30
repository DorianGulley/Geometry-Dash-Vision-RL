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
    entropy_weight: float = 0.0
    value_weight: float = 0.01
    batch_episodes: int = 8
    max_steps: int | None = None
    seed: int = 0
    log_every: int = 0


@dataclass
class EpisodeTrace:
    rewards: list[float] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    decision_mask: list[bool] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    truncated: bool = False

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def jump_rate(self) -> float:
        if not self.actions:
            return 0.0
        return sum(self.actions) / len(self.actions)


@dataclass(frozen=True)
class TrainResult:
    rewards: list[float]
    completed: list[bool]
    jump_rates: list[float]
    truncated: list[bool]


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    success_rate: float
    avg_reward: float
    avg_progress: float
    avg_jump_rate: float


ProgressCallback = Callable[[int, float, bool, float], None]


def train_reinforce(
    env: GDRLEnv | list[GDRLEnv],
    model: TinyJumpCNN | None = None,
    config: ReinforceConfig | None = None,
    on_episode: ProgressCallback | None = None,
) -> tuple[TinyJumpCNN, TrainResult]:
    cfg = config or ReinforceConfig()
    set_seed(cfg.seed)
    envs = env if isinstance(env, list) else [env]
    if not envs:
        raise ValueError("Expected at least one training environment.")
    env_rng = random.Random(cfg.seed)
    model = model or TinyJumpCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rewards: list[float] = []
    completed: list[bool] = []
    jump_rates: list[float] = []
    truncated: list[bool] = []

    episode_idx = 0
    while episode_idx < cfg.episodes:
        batch: list[EpisodeTrace] = []
        for _ in range(max(1, cfg.batch_episodes)):
            if episode_idx >= cfg.episodes:
                break
            train_env = env_rng.choice(envs)
            trace = run_policy_episode(train_env, model, max_steps=cfg.max_steps)
            if not trace.rewards:
                raise ValueError("REINFORCE requires at least one step per episode.")
            batch.append(trace)
            episode_idx += 1

            done = trace_completed(trace)
            rewards.append(trace.total_reward)
            completed.append(done)
            jump_rates.append(trace.jump_rate)
            truncated.append(trace.truncated)
            if on_episode is not None:
                on_episode(episode_idx, trace.total_reward, done, trace.jump_rate)

        loss = reinforce_batch_loss(batch, cfg.gamma, cfg.entropy_weight, cfg.value_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, TrainResult(rewards, completed, jump_rates, truncated)


def evaluate_policy(env: GDRLEnv, model: TinyJumpCNN, *, episodes: int = 10) -> EvalResult:
    rewards: list[float] = []
    progress: list[float] = []
    jump_rates: list[float] = []
    completed = 0
    for _ in range(episodes):
        trace = run_greedy_episode(env, model)
        rewards.append(trace.total_reward)
        progress.append(trace_progress(trace))
        jump_rates.append(trace.jump_rate)
        completed += int(trace_completed(trace))
    n = max(1, episodes)
    return EvalResult(episodes, completed / n, sum(rewards) / n, sum(progress) / n, sum(jump_rates) / n)


def run_policy_episode(env: GDRLEnv, model: TinyJumpCNN, *, max_steps: int | None = None) -> EpisodeTrace:
    obs, info = env.reset()
    trace = EpisodeTrace()
    done = False

    while not done:
        if max_steps is not None and len(trace.rewards) >= max_steps:
            trace.truncated = True
            break
        can_affect_state = bool(info.get("grounded", False))
        action, log_prob, entropy, value = sample_action(model, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.rewards.append(float(reward))
        trace.log_probs.append(log_prob)
        trace.entropies.append(entropy)
        trace.values.append(value)
        trace.decision_mask.append(can_affect_state)
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
            trace.truncated = True
            break
        action = model.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.rewards.append(float(reward))
        trace.actions.append(action)
        done = terminated or truncated

    trace.info = info
    return trace


def sample_action(model: TinyJumpCNN, obs: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = obs_to_tensor(obs)
    logits = model(x)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return (
        int(action.item()),
        dist.log_prob(action).squeeze(0),
        dist.entropy().squeeze(0),
        model.value(x).squeeze(0),
    )


def reinforce_loss(
    trace: EpisodeTrace,
    gamma: float,
    entropy_weight: float,
    value_weight: float = 0.01,
) -> torch.Tensor:
    if not trace.log_probs:
        raise ValueError("Cannot compute REINFORCE loss for an empty episode trace.")
    return reinforce_batch_loss([trace], gamma, entropy_weight, value_weight)


def reinforce_batch_loss(
    traces: list[EpisodeTrace],
    gamma: float,
    entropy_weight: float,
    value_weight: float = 0.01,
) -> torch.Tensor:
    if not traces:
        raise ValueError("Cannot compute REINFORCE loss for an empty batch.")
    for trace in traces:
        if not trace.log_probs:
            raise ValueError("Cannot compute REINFORCE loss for an empty episode trace.")

    returns = torch.cat([discounted_returns(trace.rewards, gamma) for trace in traces])
    log_probs = torch.cat([torch.stack(trace.log_probs) for trace in traces])
    entropies = torch.cat([torch.stack(trace.entropies) for trace in traces])
    values = torch.cat([torch.stack(trace.values) for trace in traces])
    decision_mask = torch.cat([
        torch.tensor(trace.decision_mask, dtype=torch.bool) for trace in traces
    ])
    advantages = normalize(returns - values.detach())
    if bool(decision_mask.any()):
        policy_loss = -(log_probs[decision_mask] * advantages[decision_mask]).sum()
        entropy_bonus = entropies[decision_mask].sum() * entropy_weight
    else:
        policy_loss = log_probs.sum() * 0.0
        entropy_bonus = entropies.sum() * 0.0
    value_loss = torch.nn.functional.mse_loss(values, returns)
    return policy_loss + value_weight * value_loss - entropy_bonus


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
