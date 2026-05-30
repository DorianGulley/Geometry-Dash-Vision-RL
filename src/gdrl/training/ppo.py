from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from gdrl.envs import GDRLEnv

from .policy_net import TinyJumpCNN, obs_to_tensor
from .reinforce import EvalResult, discounted_returns, evaluate_policy, normalize


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 20
    batch_episodes: int = 8
    epochs: int = 4
    gamma: float = 0.99
    lr: float = 3e-4
    clip_ratio: float = 0.2
    value_weight: float = 0.01
    entropy_weight: float = 0.0
    max_grad_norm: float = 0.5
    max_steps: int | None = None
    seed: int = 0
    eval_episodes: int = 5


@dataclass(frozen=True)
class PPORolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    decision_mask: torch.Tensor
    rewards: list[float]
    completed: list[bool]
    jump_rates: list[float]


@dataclass
class PPOTrainResult:
    rewards: list[float] = field(default_factory=list)
    completed: list[bool] = field(default_factory=list)
    jump_rates: list[float] = field(default_factory=list)
    eval_results: list[EvalResult] = field(default_factory=list)


def train_ppo(
    envs: GDRLEnv | list[GDRLEnv],
    *,
    model: TinyJumpCNN | None = None,
    config: PPOConfig | None = None,
) -> tuple[TinyJumpCNN, PPOTrainResult]:
    cfg = config or PPOConfig()
    set_seed(cfg.seed)
    train_envs = envs if isinstance(envs, list) else [envs]
    if not train_envs:
        raise ValueError("Expected at least one training environment.")

    model = model or TinyJumpCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)
    result = PPOTrainResult()

    for _ in range(cfg.updates):
        batch = collect_rollouts(model, train_envs, cfg, rng)
        ppo_update(model, optimizer, batch, cfg)
        result.rewards.extend(batch.rewards)
        result.completed.extend(batch.completed)
        result.jump_rates.extend(batch.jump_rates)
        result.eval_results.append(evaluate_policy(train_envs[0], model, episodes=cfg.eval_episodes))

    return model, result


def collect_rollouts(
    model: TinyJumpCNN,
    envs: list[GDRLEnv],
    cfg: PPOConfig,
    rng: random.Random,
) -> PPORolloutBatch:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    decision_mask: list[bool] = []
    rewards: list[float] = []
    completed: list[bool] = []
    jump_rates: list[float] = []

    for _ in range(max(1, cfg.batch_episodes)):
        env = rng.choice(envs)
        obs, info = env.reset()
        episode_rewards: list[float] = []
        episode_actions: list[int] = []
        episode_values: list[torch.Tensor] = []
        episode_done = False

        while not episode_done:
            if cfg.max_steps is not None and len(episode_rewards) >= cfg.max_steps:
                break
            x = obs_to_tensor(obs)
            with torch.no_grad():
                logits = model(x)
                dist = Categorical(logits=logits)
                action = dist.sample()
                value = model.value(x).squeeze(0)

            observations.append(x.squeeze(0))
            actions.append(action.squeeze(0))
            old_log_probs.append(dist.log_prob(action).squeeze(0))
            old_values.append(value)
            decision_mask.append(bool(info.get("grounded", False)))

            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            episode_rewards.append(float(reward))
            episode_actions.append(int(action.item()))
            episode_values.append(value)
            episode_done = terminated or truncated

        returns.extend(discounted_returns(episode_rewards, cfg.gamma))
        rewards.append(float(sum(episode_rewards)))
        completed.append(trace_completed_info(info))
        jump_rates.append(sum(episode_actions) / max(1, len(episode_actions)))

    return PPORolloutBatch(
        observations=torch.stack(observations),
        actions=torch.stack(actions).long(),
        old_log_probs=torch.stack(old_log_probs).detach(),
        returns=torch.stack(returns),
        old_values=torch.stack(old_values).detach(),
        decision_mask=torch.tensor(decision_mask, dtype=torch.bool),
        rewards=rewards,
        completed=completed,
        jump_rates=jump_rates,
    )


def ppo_update(
    model: TinyJumpCNN,
    optimizer: torch.optim.Optimizer,
    batch: PPORolloutBatch,
    cfg: PPOConfig,
) -> None:
    advantages = normalize(batch.returns - batch.old_values)
    mask = batch.decision_mask
    for _ in range(max(1, cfg.epochs)):
        logits = model(batch.observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch.actions)
        values = model.value(batch.observations)
        ratio = torch.exp(log_probs - batch.old_log_probs)

        if bool(mask.any()):
            clipped = torch.clamp(ratio[mask], 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            objective = torch.min(ratio[mask] * advantages[mask], clipped * advantages[mask])
            policy_loss = -objective.mean()
            entropy_bonus = dist.entropy()[mask].mean() * cfg.entropy_weight
        else:
            policy_loss = log_probs.sum() * 0.0
            entropy_bonus = dist.entropy().sum() * 0.0

        value_loss = torch.nn.functional.mse_loss(values, batch.returns)
        loss = policy_loss + cfg.value_weight * value_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()


def trace_completed_info(info: dict) -> bool:
    return bool(info.get("episode", {}).get("completed", False))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def level_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        out = sorted(p.glob("*.json"))
    else:
        out = [p]
    if not out:
        raise FileNotFoundError(f"No level JSON files found at {p}.")
    return out
