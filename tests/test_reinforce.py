import torch
import pytest

from gdrl.envs import GDRLEnv
from gdrl.training import (
    ReinforceConfig,
    TinyJumpCNN,
    evaluate_policy,
    run_greedy_episode,
    run_policy_episode,
    train_reinforce,
)
from gdrl.training.reinforce import discounted_returns, reinforce_batch_loss, reinforce_loss


def test_discounted_returns() -> None:
    out = discounted_returns([1.0, 1.0, 1.0], gamma=0.9)

    assert torch.allclose(out, torch.tensor([2.71, 1.9, 1.0]))


def test_run_policy_episode_collects_rewards() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    trace = run_policy_episode(env, TinyJumpCNN(), max_steps=5)

    assert len(trace.rewards) == 5
    assert len(trace.log_probs) == 5
    assert len(trace.values) == 5
    assert len(trace.decision_mask) == 5
    assert trace.decision_mask[0]
    assert len(trace.actions) == 5
    assert 0.0 <= trace.jump_rate <= 1.0
    assert trace.truncated


def test_run_greedy_episode_marks_local_truncation() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    trace = run_greedy_episode(env, TinyJumpCNN(), max_steps=5)

    assert len(trace.rewards) == 5
    assert trace.truncated


def test_reinforce_loss_rejects_empty_trace() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    trace = run_policy_episode(env, TinyJumpCNN(), max_steps=0)

    with pytest.raises(ValueError, match="empty episode trace"):
        reinforce_loss(trace, 0.99, 0.01)


def test_reinforce_batch_loss_accepts_multiple_traces() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    model = TinyJumpCNN()
    traces = [
        run_policy_episode(env, model, max_steps=3),
        run_policy_episode(env, model, max_steps=3),
    ]

    loss = reinforce_batch_loss(traces, 0.99, 0.0)

    assert loss.ndim == 0


def test_train_reinforce_smoke() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    cfg = ReinforceConfig(episodes=2, max_steps=5, seed=1)

    model, result = train_reinforce(env, config=cfg)

    assert isinstance(model, TinyJumpCNN)
    assert len(result.rewards) == 2
    assert len(result.completed) == 2
    assert len(result.jump_rates) == 2
    assert len(result.truncated) == 2


def test_train_reinforce_accepts_multiple_envs() -> None:
    envs = [
        GDRLEnv("levels/curriculum/flat_empty.json"),
        GDRLEnv("levels/tiny_spikes.json"),
    ]
    cfg = ReinforceConfig(episodes=2, max_steps=5, seed=1, batch_episodes=2)

    _, result = train_reinforce(envs, config=cfg)

    assert len(result.rewards) == 2


def test_greedy_eval_smoke() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    model = TinyJumpCNN()

    trace = run_greedy_episode(env, model, max_steps=5)
    result = evaluate_policy(GDRLEnv("levels/tiny_spikes.json"), model, episodes=2)

    assert len(trace.rewards) == 5
    assert result.episodes == 2
    assert 0.0 <= result.success_rate <= 1.0
    assert 0.0 <= result.avg_jump_rate <= 1.0
