from gdrl.levels import (
    CameraSpec,
    Level,
    LevelMetadata,
    PhysicsSpec,
    Tile,
    TileType,
    Vec2i,
)
from gdrl.sim import Action, Simulator
from gdrl.sim.physics import PlayerBody, step_physics


def test_reset_snaps_player_to_ground() -> None:
    sim = Simulator(flat_test_level())
    state = sim.reset()

    assert state.grounded
    assert sim.player.y + sim.player.h == sim.level.floor_y() * sim.level.tile_size


def test_jump_uses_level_configured_velocity() -> None:
    level = Level(
        meta=LevelMetadata(level_id="jump_velocity_test"),
        width=20,
        height=12,
        tile_size=32,
        start=Vec2i(2, 9),
        end=Vec2i(18, 9),
        physics=PhysicsSpec(jump_velocity=-900.0, gravity=0.0),
        camera=CameraSpec(),
        tiles=[],
    )
    sim = Simulator(level)
    sim.reset()

    state, _ = sim.step(Action(jump_pressed=True))

    assert state.player_vy == -900.0


def test_completed_episode_reports_full_progress() -> None:
    sim = Simulator(flat_test_level())
    sim.reset()
    sim.player.x = sim.level.end.x * sim.level.tile_size - sim.player.w

    state, result = sim.step(Action())

    assert result is not None
    assert result.completed
    assert result.progress == 1.0
    assert state.progress == 1.0


def test_hitting_block_side_is_death() -> None:
    level = Level(
        meta=LevelMetadata(level_id="block_side_test"),
        width=20,
        height=12,
        tile_size=32,
        start=Vec2i(2, 9),
        end=Vec2i(18, 9),
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=[Tile(x=5, y=10, type=TileType.BLOCK)],
    )
    sim = Simulator(level)
    sim.reset()

    result = None
    for _ in range(40):
        _, result = sim.step(Action())
        if result is not None:
            break

    assert result is not None
    assert result.died
    assert result.reason == "death"


def test_reset_can_snap_to_block_platform() -> None:
    level = Level(
        meta=LevelMetadata(level_id="platform_spawn_test"),
        width=20,
        height=12,
        tile_size=32,
        start=Vec2i(5, 6),
        end=Vec2i(18, 6),
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=[Tile(x=5, y=8, type=TileType.BLOCK)],
    )
    sim = Simulator(level)
    state = sim.reset()

    assert state.grounded
    assert sim.player.y + sim.player.h == 8 * level.tile_size


def test_spike_collision_uses_triangle_shape() -> None:
    level = Level(
        meta=LevelMetadata(level_id="spike_triangle_test"),
        width=20,
        height=12,
        tile_size=32,
        start=Vec2i(2, 9),
        end=Vec2i(18, 9),
        physics=PhysicsSpec(gravity=0.0, scroll_speed=0.0),
        camera=CameraSpec(),
        tiles=[Tile(x=5, y=10, type=TileType.SPIKE)],
    )

    near_left_edge = PlayerBody(x=160.0, y=320.0, vx=0.0, vy=0.0, w=4.0, h=20.0)
    centered_on_spike = PlayerBody(x=170.0, y=323.2, vx=0.0, vy=0.0, w=12.0, h=28.8)

    assert not step_physics(level, near_left_edge, 0.0).died
    assert step_physics(level, centered_on_spike, 0.0).died


def flat_test_level() -> Level:
    return Level(
        meta=LevelMetadata(level_id="flat_physics_test"),
        width=20,
        height=12,
        tile_size=32,
        start=Vec2i(2, 9),
        end=Vec2i(18, 9),
        physics=PhysicsSpec(),
        camera=CameraSpec(),
        tiles=[],
    )
