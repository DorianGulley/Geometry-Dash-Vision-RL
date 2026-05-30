from pathlib import Path

from gdrl.envs import GDRLEnv, run_episode
from gdrl.training import SpikeTimingPolicy, SpikeWindowPolicy


def test_spike_window_policy_completes_curriculum() -> None:
    policy = SpikeWindowPolicy()

    for path in sorted(Path("levels/curriculum").glob("*.json")):
        stats = run_episode(GDRLEnv(path), policy)

        assert stats.info["episode"]["completed"], path


def test_spike_timing_policy_completes_generated_levels() -> None:
    policy = SpikeTimingPolicy()

    for path in sorted(Path("levels/generated").rglob("*.json")):
        stats = run_episode(GDRLEnv(path), policy)

        assert stats.info["episode"]["completed"], path
