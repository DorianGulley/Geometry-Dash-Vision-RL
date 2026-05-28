from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from gdrl.levels import (
    CameraSpec,
    Level,
    LevelMetadata,
    PhysicsSpec,
    Tile,
    TileType,
    Vec2i,
)


class Brush(str, Enum):
    ERASE = "erase"
    BLOCK = "block"
    SPIKE = "spike"
    START = "start"
    END = "end"


def bump_version(version: str) -> str:
    """Increment patch component: 0.0.1 -> 0.0.2."""
    parts = version.strip().split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        major, minor, patch = (int(p) for p in parts)
        return f"{major}.{minor}.{patch + 1}"
    return f"{version}.1"


def default_new_level() -> "EditorState":
    return EditorState.new(width=80, height=12)


@dataclass
class EditorState:
    width: int
    height: int
    tile_size: int
    start: Vec2i
    end: Vec2i
    meta: LevelMetadata
    tiles: dict[tuple[int, int], TileType] = field(default_factory=dict)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    loaded_path: Path | None = None
    dirty: bool = False
    camera_tx: int = 0

    @classmethod
    def new(
        cls,
        *,
        width: int = 80,
        height: int = 12,
        tile_size: int = 32,
    ) -> "EditorState":
        floor_y = height - 1
        return cls(
            width=width,
            height=height,
            tile_size=tile_size,
            start=Vec2i(2, floor_y - 1),
            end=Vec2i(width - 5, floor_y - 1),
            meta=LevelMetadata(
                level_id="new_level",
                name="Untitled",
                author="",
                version="0.0.1",
            ),
        )

    @classmethod
    def from_level(cls, level: Level, *, path: Path | None = None) -> "EditorState":
        tiles = {(t.x, t.y): t.type for t in level.tiles}
        return cls(
            width=level.width,
            height=level.height,
            tile_size=level.tile_size,
            start=level.start,
            end=level.end,
            meta=level.meta,
            tiles=tiles,
            physics=level.physics,
            camera=level.camera,
            loaded_path=path,
            dirty=False,
        )

    def floor_y(self) -> int:
        return self.height - 1

    def is_floor_row(self, ty: int) -> bool:
        return ty == self.floor_y()

    def in_bounds(self, tx: int, ty: int) -> bool:
        return 0 <= tx < self.width and 0 <= ty < self.height

    def can_paint(self, tx: int, ty: int) -> bool:
        return self.in_bounds(tx, ty) and not self.is_floor_row(ty)

    def mark_dirty(self) -> None:
        self.dirty = True

    def apply_brush(self, tx: int, ty: int, brush: Brush) -> bool:
        if not self.can_paint(tx, ty):
            return False

        if brush == Brush.START:
            self.start = Vec2i(tx, ty)
            self.tiles.pop((tx, ty), None)
            self.mark_dirty()
            return True
        if brush == Brush.END:
            self.end = Vec2i(tx, ty)
            self.tiles.pop((tx, ty), None)
            self.mark_dirty()
            return True
        if brush == Brush.ERASE:
            if self.tiles.pop((tx, ty), None) is not None:
                self.mark_dirty()
                return True
            return False
        if brush == Brush.BLOCK:
            if (tx, ty) == (self.start.x, self.start.y) or (tx, ty) == (self.end.x, self.end.y):
                return False
            self.tiles[(tx, ty)] = TileType.BLOCK
            self.mark_dirty()
            return True
        if brush == Brush.SPIKE:
            if (tx, ty) == (self.start.x, self.start.y) or (tx, ty) == (self.end.x, self.end.y):
                return False
            self.tiles[(tx, ty)] = TileType.SPIKE
            self.mark_dirty()
            return True
        return False

    def to_level(self) -> Level:
        tile_list = [
            Tile(x=x, y=y, type=tt)
            for (x, y), tt in sorted(self.tiles.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        ]
        return Level(
            meta=self.meta,
            width=self.width,
            height=self.height,
            tile_size=self.tile_size,
            start=self.start,
            end=self.end,
            physics=self.physics,
            camera=self.camera,
            tiles=tile_list,
        )

    def prepare_for_save(self) -> None:
        """Bump version when saving an edited level that was loaded from disk."""
        if self.dirty and self.loaded_path is not None:
            self.meta = LevelMetadata(
                schema_version=self.meta.schema_version,
                level_id=self.meta.level_id,
                name=self.meta.name,
                author=self.meta.author,
                split=self.meta.split,
                tags=list(self.meta.tags),
                version=bump_version(self.meta.version),
            )
