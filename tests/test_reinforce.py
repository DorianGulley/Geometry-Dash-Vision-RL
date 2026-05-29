import torch

from gdrl.envs import GDRLEnv
from gdrl.training import (
    ReinforceConfig,
    TinyJumpCNN,
    evaluate_policy,
    run_greedy_episode,
    run_policy_episode,
    train_reinforce,
)
from gdrl.training.reinforce import discounted_returns


def test_discounted_returns() -> None:
    out = discounted_returns([1.0, 1.0, 1.0], gamma=0.9)

    assert torch.allclose(out, torch.tensor([2.71, 1.9, 1.0]))


def test_run_policy_episode_collects_rewards() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    trace = run_policy_episode(env, TinyJumpCNN(), max_steps=5)

    assert len(trace.rewards) == 5
    assert len(trace.log_probs) == 5
    assert len(trace.actions) == 5


def test_train_reinforce_smoke() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    cfg = ReinforceConfig(episodes=2, max_steps=5, seed=1)

    model, result = train_reinforce(env, config=cfg)

    assert isinstance(model, TinyJumpCNN)
    assert len(result.rewards) == 2
    assert len(result.completed) == 2


def test_greedy_eval_smoke() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    model = TinyJumpCNN()

    trace = run_greedy_episode(env, model, max_steps=5)
    result = evaluate_policy(GDRLEnv("levels/tiny_spikes.json"), model, episodes=2)

    assert len(trace.rewards) == 5
    assert result.episodes == 2
    assert 0.0 <= result.success_rate <= 1.0
