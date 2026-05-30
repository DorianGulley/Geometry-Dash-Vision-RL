from pathlib import Path

from gdrl.levels import (
    SimpleLevelConfig,
    load_level,
    make_curriculum_levels,
    make_pillar_sweep,
    make_simple_level,
    make_single_spike_sweep,
    make_stair_sweep,
    make_three_spike_sweep,
    make_tiny_spike_level,
    make_two_spike_sweep,
    validate_level,
)


def test_simple_level_is_valid_and_seeded() -> None:
    cfg = SimpleLevelConfig(seed=7)
    a = make_simple_level(cfg)
    b = make_simple_level(cfg)

    assert validate_level(a) == []
    assert a.to_dict() == b.to_dict()
    assert len(a.tiles) > 0
    assert all(t.y == a.floor_y() - 1 for t in a.tiles)


def test_tiny_spike_level_is_fixed_and_valid() -> None:
    level = make_tiny_spike_level()

    assert validate_level(level) == []
    assert level.meta.level_id == "tiny_spikes"
    assert [(t.x, t.y, t.type.value) for t in level.tiles] == [
        (12, 10, "spike"),
        (20, 10, "spike"),
    ]


def test_curriculum_levels_are_fixed_and_valid() -> None:
    levels = make_curriculum_levels()

    assert [level.meta.level_id for level in levels] == [
        "flat_empty",
        "one_spike",
        "two_spikes",
        "late_spike",
        "mixed_spikes",
    ]
    assert all(validate_level(level) == [] for level in levels)


def test_checked_in_curriculum_matches_generator() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_curriculum_levels()}

    for path in sorted(Path("levels/curriculum").glob("*.json")):
        level = load_level(path)
        assert level.to_dict() == generated[level.meta.level_id]


def test_checked_in_levels_are_valid() -> None:
    for path in sorted(Path("levels").rglob("*.json")):
        assert validate_level(load_level(path)) == []


def test_single_spike_sweep_is_valid_and_checked_in() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_single_spike_sweep()}
    seen: set[str] = set()

    for path in sorted(Path("levels/generated/single_spike").glob("*.json")):
        level = load_level(path)
        seen.add(level.meta.level_id)
        assert validate_level(level) == []
        assert level.to_dict() == generated[level.meta.level_id]
    assert seen == set(generated)


def test_two_spike_sweep_is_valid_and_checked_in() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_two_spike_sweep()}
    seen: set[str] = set()

    for path in sorted(Path("levels/generated/two_spike").glob("*.json")):
        level = load_level(path)
        seen.add(level.meta.level_id)
        assert validate_level(level) == []
        assert level.to_dict() == generated[level.meta.level_id]
    assert seen == set(generated)


def test_three_spike_sweep_is_valid_and_checked_in() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_three_spike_sweep()}
    seen: set[str] = set()

    for path in sorted(Path("levels/generated/three_spike").glob("*.json")):
        level = load_level(path)
        seen.add(level.meta.level_id)
        assert validate_level(level) == []
        assert level.to_dict() == generated[level.meta.level_id]
    assert seen == set(generated)


def test_pillar_sweep_is_valid_and_checked_in() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_pillar_sweep()}
    seen: set[str] = set()

    for path in sorted(Path("levels/generated/pillar").glob("*.json")):
        level = load_level(path)
        seen.add(level.meta.level_id)
        assert validate_level(level) == []
        assert level.to_dict() == generated[level.meta.level_id]
    assert seen == set(generated)


def test_stair_sweep_is_valid_and_checked_in() -> None:
    generated = {level.meta.level_id: level.to_dict() for level in make_stair_sweep()}
    seen: set[str] = set()

    for path in sorted(Path("levels/generated/stair").glob("*.json")):
        level = load_level(path)
        seen.add(level.meta.level_id)
        assert validate_level(level) == []
        assert level.to_dict() == generated[level.meta.level_id]
    assert seen == set(generated)
