from pathlib import Path

from gdrl.training import eval_random, random_policy


def test_random_policy_returns_binary_actions() -> None:
    policy = random_policy(seed=0)

    assert {policy(None, {}) for _ in range(20)} <= {0, 1}


def test_eval_random_reports_one_row_per_level() -> None:
    paths = [
        Path("levels/curriculum/flat_empty.json"),
        Path("levels/curriculum/one_spike.json"),
    ]

    rows = eval_random(paths, episodes=2, seed=0)

    assert [row.level_id for row in rows] == ["flat_empty", "one_spike"]
    assert all(row.episodes == 2 for row in rows)
    assert all(0.0 <= row.success_rate <= 1.0 for row in rows)
