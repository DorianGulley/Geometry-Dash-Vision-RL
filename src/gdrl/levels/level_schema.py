from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class TileType(str, Enum):
    AIR = "air"
    BLOCK = "block"
    SPIKE = "spike"


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Vec2i:
    x: int
    y: int


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    type: TileType


@dataclass(frozen=True)
class PhysicsSpec:
    fixed_dt: float = 1.0 / 60.0
    gravity: float = 2200.0
    jump_velocity: float = -750.0
    scroll_speed: float = 300.0


@dataclass(frozen=True)
class CameraSpec:
    screen_width: int = 960
    screen_height: int = 540
    player_screen_x: int = 240


@dataclass(frozen=True)
class LevelMetadata:
    schema_version: int = 1
    level_id: str = "unknown"
    name: str = "Untitled"
    author: str = "unknown"
    split: Split = "train"
    tags: list[str] = field(default_factory=list)
    version: str = "0.0.1"


@dataclass(frozen=True)
class Level:
    meta: LevelMetadata
    width: int
    height: int
    tile_size: int
    start: Vec2i
    end: Vec2i
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    tiles: list[Tile] = field(default_factory=list)

    def floor_y(self) -> int:
        return self.height - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.meta.schema_version,
            "level_id": self.meta.level_id,
            "name": self.meta.name,
            "author": self.meta.author,
            "split": self.meta.split,
            "tags": list(self.meta.tags),
            "version": self.meta.version,
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "start": {"x": self.start.x, "y": self.start.y},
            "end": {"x": self.end.x, "y": self.end.y},
            "physics": {
                "fixed_dt": self.physics.fixed_dt,
                "gravity": self.physics.gravity,
                "jump_velocity": self.physics.jump_velocity,
                "scroll_speed": self.physics.scroll_speed,
            },
            "camera": {
                "screen_width": self.camera.screen_width,
                "screen_height": self.camera.screen_height,
                "player_screen_x": self.camera.player_screen_x,
            },
            "tiles": [
                {"x": t.x, "y": t.y, "type": t.type.value}
                for t in sorted(self.tiles, key=lambda tt: (tt.y, tt.x, tt.type.value))
            ],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Level":
        meta = LevelMetadata(
            schema_version=int(d.get("schema_version", 1)),
            level_id=str(d.get("level_id", "unknown")),
            name=str(d.get("name", "Untitled")),
            author=str(d.get("author", "unknown")),
            split=d.get("split", "train"),
            tags=list(d.get("tags", [])),
            version=str(d.get("version", "0.0.1")),
        )
        physics_d = d.get("physics", {}) or {}
        camera_d = d.get("camera", {}) or {}
        tiles_d = d.get("tiles", []) or []

        lvl = Level(
            meta=meta,
            width=int(d["width"]),
            height=int(d["height"]),
            tile_size=int(d.get("tile_size", 32)),
            start=Vec2i(int(d["start"]["x"]), int(d["start"]["y"])),
            end=Vec2i(int(d["end"]["x"]), int(d["end"]["y"])),
            physics=PhysicsSpec(
                fixed_dt=float(physics_d.get("fixed_dt", 1.0 / 60.0)),
                gravity=float(physics_d.get("gravity", 2200.0)),
                jump_velocity=float(physics_d.get("jump_velocity", -750.0)),
                scroll_speed=float(physics_d.get("scroll_speed", 300.0)),
            ),
            camera=CameraSpec(
                screen_width=int(camera_d.get("screen_width", 960)),
                screen_height=int(camera_d.get("screen_height", 540)),
                player_screen_x=int(camera_d.get("player_screen_x", 240)),
            ),
            tiles=[
                Tile(x=int(t["x"]), y=int(t["y"]), type=TileType(str(t["type"])))
                for t in tiles_d
            ],
        )
        return lvl

