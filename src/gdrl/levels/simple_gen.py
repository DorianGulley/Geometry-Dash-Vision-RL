from __future__ import annotations

import random
from dataclasses import dataclass

from .level_schema import (
    CameraSpec,
    Level,
    LevelMetadata,
    PhysicsSpec,
    Tile,
    TileType,
    Vec2i,
)


@dataclass(frozen=True)
class SimpleLevelConfig:
    width: int = 80
    height: int = 12
    tile_size: int = 32
    obstacle_count: int = 8
    min_gap: int = 5
    max_gap: int = 9
    seed: int | None = None
    level_id: str = "generated_simple"


def make_tiny_spike_level() -> Level:
    return make_fixed_spike_level("tiny_spikes", "Tiny Spikes", [12, 20], width=36)


def make_curriculum_levels() -> list[Level]:
    return [
        make_fixed_spike_level("flat_empty", "Flat Empty", [], width=28),
        make_fixed_spike_level("one_spike", "One Spike", [14], width=32),
        make_fixed_spike_level("two_spikes", "Two Spikes", [12, 20], width=36),
        make_fixed_spike_level("late_spike", "Late Spike", [24], width=40),
        make_fixed_spike_level("mixed_spikes", "Mixed Spikes", [12, 22, 29], width=44),
    ]


def make_fixed_spike_level(level_id: str, name: str, spike_xs: list[int], *, width: int) -> Level:
    floor_y = 11
    player_y = floor_y - 2
    spike_y = floor_y - 1
    return Level(
        meta=LevelMetadata(
            level_id=level_id,
            name=name,
            author="local",
            tags=["train", "curriculum"],
        ),
        width=width,
        height=12,
        tile_size=32,
        start=Vec2i(2, player_y),
        end=Vec2i(width - 4, player_y),
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=[Tile(x=x, y=spike_y, type=TileType.SPIKE) for x in spike_xs],
    )


def make_simple_level(config: SimpleLevelConfig | None = None) -> Level:
    cfg = config or SimpleLevelConfig()
    rng = random.Random(cfg.seed)
    floor_y = cfg.height - 1
    obstacle_y = floor_y - 1
    start = Vec2i(2, obstacle_y - 1)
    end = Vec2i(cfg.width - 5, obstacle_y - 1)

    return Level(
        meta=metadata(cfg),
        width=cfg.width,
        height=cfg.height,
        tile_size=cfg.tile_size,
        start=start,
        end=end,
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=spikes(cfg, rng, obstacle_y),
    )


def metadata(config: SimpleLevelConfig) -> LevelMetadata:
    return LevelMetadata(
        level_id=config.level_id,
        name="Generated Simple",
        author="local",
        tags=["generated", "simple"],
    )


def spikes(config: SimpleLevelConfig, rng: random.Random, y: int) -> list[Tile]:
    x = 12
    out: list[Tile] = []
    for _ in range(max(0, config.obstacle_count)):
        if x >= config.width - 8:
            break
        out.append(Tile(x=x, y=y, type=TileType.SPIKE))
        if rng.random() < 0.25 and x + 1 < config.width - 8:
            out.append(Tile(x=x + 1, y=y, type=TileType.SPIKE))
        x += rng.randint(config.min_gap, config.max_gap)
    return out
