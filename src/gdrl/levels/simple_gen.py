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


def make_single_spike_sweep(spike_xs: list[int] | None = None) -> list[Level]:
    xs = spike_xs or [10, 12, 14, 16, 18, 20, 22, 24]
    return [
        make_fixed_spike_level(
            f"single_spike_x{x}",
            f"Single Spike X{x}",
            [x],
            width=max(32, x + 12),
            tags=["train", "generated", "single_spike"],
        )
        for x in xs
    ]


def make_two_spike_sweep(
    first_xs: list[int] | None = None,
    gaps: list[int] | None = None,
) -> list[Level]:
    starts = first_xs or [10, 12, 14]
    spacing = gaps or [6, 8, 10, 12]
    levels: list[Level] = []
    for first_x in starts:
        for gap in spacing:
            second_x = first_x + gap
            levels.append(
                make_fixed_spike_level(
                    f"two_spikes_x{first_x}_gap{gap}",
                    f"Two Spikes X{first_x} Gap {gap}",
                    [first_x, second_x],
                    width=max(36, second_x + 14),
                    tags=["train", "generated", "two_spike"],
                )
            )
    return levels


def make_three_spike_sweep() -> list[Level]:
    spike_sets = [
        [10, 16, 24],
        [10, 18, 24],
        [10, 18, 26],
        [10, 18, 28],
        [10, 20, 28],
        [12, 18, 26],
        [12, 20, 26],
        [12, 20, 28],
    ]
    return [
        make_fixed_spike_level(
            f"three_spikes_{'_'.join(str(x) for x in xs)}",
            f"Three Spikes {' '.join(str(x) for x in xs)}",
            xs,
            width=max(44, xs[-1] + 14),
            tags=["train", "generated", "three_spike"],
        )
        for xs in spike_sets
    ]


def make_pillar_sweep() -> list[Level]:
    levels: list[Level] = []
    for x in [12, 14, 16, 18, 20]:
        levels.append(
            make_fixed_tile_level(
                f"pillar_x{x}",
                f"Pillar X{x}",
                [Tile(x=x, y=10, type=TileType.BLOCK)],
                width=max(36, x + 14),
                tags=["train", "generated", "pillar"],
            )
        )
        levels.append(
            make_fixed_tile_level(
                f"tall_pillar_x{x}",
                f"Tall Pillar X{x}",
                [
                    Tile(x=x, y=10, type=TileType.BLOCK),
                    Tile(x=x, y=9, type=TileType.BLOCK),
                ],
                width=max(36, x + 14),
                tags=["train", "generated", "pillar"],
            )
        )
    return levels


def make_stair_sweep() -> list[Level]:
    stair_tiles = [
        [Tile(x=12, y=10, type=TileType.BLOCK), Tile(x=13, y=9, type=TileType.BLOCK)],
        [Tile(x=14, y=10, type=TileType.BLOCK), Tile(x=15, y=9, type=TileType.BLOCK)],
        [
            Tile(x=12, y=10, type=TileType.BLOCK),
            Tile(x=13, y=9, type=TileType.BLOCK),
            Tile(x=14, y=8, type=TileType.BLOCK),
        ],
        [
            Tile(x=14, y=10, type=TileType.BLOCK),
            Tile(x=15, y=9, type=TileType.BLOCK),
            Tile(x=16, y=8, type=TileType.BLOCK),
        ],
    ]
    return [
        make_fixed_tile_level(
            f"stair_{i}",
            f"Stair {i}",
            tiles,
            width=40,
            tags=["train", "generated", "stair"],
        )
        for i, tiles in enumerate(stair_tiles, start=1)
    ]


def make_fixed_tile_level(
    level_id: str,
    name: str,
    tiles: list[Tile],
    *,
    width: int,
    tags: list[str],
) -> Level:
    floor_y = 11
    player_y = floor_y - 2
    return Level(
        meta=LevelMetadata(
            level_id=level_id,
            name=name,
            author="local",
            tags=tags,
        ),
        width=width,
        height=12,
        tile_size=32,
        start=Vec2i(2, player_y),
        end=Vec2i(width - 4, player_y),
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=tiles,
    )


def make_fixed_spike_level(
    level_id: str,
    name: str,
    spike_xs: list[int],
    *,
    width: int,
    tags: list[str] | None = None,
) -> Level:
    floor_y = 11
    player_y = floor_y - 2
    spike_y = floor_y - 1
    return Level(
        meta=LevelMetadata(
            level_id=level_id,
            name=name,
            author="local",
            tags=tags or ["train", "curriculum"],
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
