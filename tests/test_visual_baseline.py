from pathlib import Path

from gdrl.envs import GDRLEnv, run_episode
from gdrl.training import SpikeWindowPolicy


def test_spike_window_policy_completes_curriculum() -> None:
    policy = SpikeWindowPolicy()

    for path in sorted(Path("levels/curriculum").glob("*.json")):
        stats = run_episode(GDRLEnv(path), policy)

        assert stats.info["episode"]["completed"], path
