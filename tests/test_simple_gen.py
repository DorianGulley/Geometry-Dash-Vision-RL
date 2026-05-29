from gdrl.levels import (
    SimpleLevelConfig,
    make_curriculum_levels,
    make_simple_level,
    make_tiny_spike_level,
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
