from __future__ import annotations

from dataclasses import dataclass

from .level_schema import Level, TileType


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str


def validate_level(level: Level) -> list[ValidationError]:
    errs: list[ValidationError] = []

    if level.width <= 0 or level.height <= 1:
        errs.append(ValidationError("bad_dims", "Level width/height must be positive and height > 1."))
        return errs

    floor_y = level.floor_y()

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < level.width and 0 <= y < level.height

    if not in_bounds(level.start.x, level.start.y):
        errs.append(ValidationError("start_oob", "Start position is out of bounds."))
    if not in_bounds(level.end.x, level.end.y):
        errs.append(ValidationError("end_oob", "End position is out of bounds."))

    if level.end.x <= level.start.x:
        errs.append(ValidationError("end_left_of_start", "End position must be to the right of start position."))

    if level.start.y == floor_y:
        errs.append(ValidationError("start_on_floor_row", "Start cannot be on implicit floor row."))
    if level.end.y == floor_y:
        errs.append(ValidationError("end_on_floor_row", "End cannot be on implicit floor row."))

    seen: set[tuple[int, int]] = set()
    for t in level.tiles:
        if t.type == TileType.AIR:
            errs.append(ValidationError("air_in_tiles", "Tiles list should not contain air tiles."))
        if not in_bounds(t.x, t.y):
            errs.append(ValidationError("tile_oob", f"Tile out of bounds at ({t.x},{t.y})."))
            continue
        if t.y == floor_y:
            errs.append(ValidationError("tile_on_floor", f"Editable tile cannot be on implicit floor at ({t.x},{t.y})."))
        key = (t.x, t.y)
        if key in seen:
            errs.append(ValidationError("dup_tile", f"Duplicate tile at ({t.x},{t.y})."))
        seen.add(key)

    solid = {(t.x, t.y) for t in level.tiles if t.type == TileType.BLOCK}
    if (level.start.x, level.start.y) in solid:
        errs.append(ValidationError("start_in_block", "Start position must not be inside a block."))
    if (level.end.x, level.end.y) in solid:
        errs.append(ValidationError("end_in_block", "End position must not be inside a block."))

    return errs

