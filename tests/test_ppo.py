import random

from gdrl.envs import GDRLEnv
from gdrl.training import PPOConfig, TinyJumpCNN, train_ppo
from gdrl.training.ppo import collect_rollouts


def test_collect_rollouts_shapes() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    model = TinyJumpCNN()
    cfg = PPOConfig(batch_episodes=2, max_steps=3, seed=1)

    batch = collect_rollouts(model, [env], cfg, random.Random(1))

    assert batch.observations.shape[0] == 6
    assert batch.actions.shape[0] == 6
    assert batch.returns.shape[0] == 6
    assert len(batch.rewards) == 2


def test_train_ppo_smoke() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    cfg = PPOConfig(updates=1, batch_episodes=2, epochs=1, max_steps=3, seed=1)

    model, result = train_ppo(env, config=cfg)

    assert isinstance(model, TinyJumpCNN)
    assert len(result.rewards) == 2
    assert len(result.eval_results) == 1
