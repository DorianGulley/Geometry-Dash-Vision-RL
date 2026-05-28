from __future__ import annotations

from dataclasses import dataclass

from gdrl.levels import Level, TileType

from .collision import RectF, intersects


@dataclass
class PlayerBody:
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    grounded: bool = False

    def rect(self) -> RectF:
        return RectF(self.x, self.y, self.w, self.h)


def _tile_rect(level: Level, tx: int, ty: int) -> RectF:
    s = level.tile_size
    return RectF(tx * s, ty * s, s, s)


def _solid_at(level: Level, solid_tiles: set[tuple[int, int]], tx: int, ty: int) -> bool:
    if ty == level.floor_y():
        return True
    return (tx, ty) in solid_tiles


def _kill_at(level: Level, spike_tiles: set[tuple[int, int]], tx: int, ty: int) -> bool:
    return (tx, ty) in spike_tiles


def _iter_tiles_overlapping(level: Level, rect: RectF) -> list[tuple[int, int]]:
    s = level.tile_size
    min_tx = int(rect.left() // s) - 1
    max_tx = int(rect.right() // s) + 1
    min_ty = int(rect.top() // s) - 1
    max_ty = int(rect.bottom() // s) + 1
    out: list[tuple[int, int]] = []
    for ty in range(min_ty, max_ty + 1):
        for tx in range(min_tx, max_tx + 1):
            if 0 <= tx < level.width and 0 <= ty < level.height:
                out.append((tx, ty))
    return out


def jump(body: PlayerBody, jump_velocity: float) -> None:
    if body.grounded:
        body.vy = float(jump_velocity)
        body.grounded = False


@dataclass(frozen=True)
class StepPhysicsResult:
    died: bool


def step_physics(level: Level, body: PlayerBody, dt: float) -> StepPhysicsResult:
    solid_tiles = {(t.x, t.y) for t in level.tiles if t.type == TileType.BLOCK}
    spike_tiles = {(t.x, t.y) for t in level.tiles if t.type == TileType.SPIKE}

    # Integrate velocity -> position with simple separation on solids.
    body.vy += level.physics.gravity * dt

    died = False

    # Horizontal movement
    new_x = body.x + body.vx * dt
    rect_x = RectF(new_x, body.y, body.w, body.h)
    for tx, ty in _iter_tiles_overlapping(level, rect_x):
        if _solid_at(level, solid_tiles, tx, ty):
            tr = _tile_rect(level, tx, ty)
            if intersects(rect_x, tr):
                if body.vx > 0:
                    new_x = tr.left() - body.w
                elif body.vx < 0:
                    new_x = tr.right()
                rect_x = RectF(new_x, body.y, body.w, body.h)
        if _kill_at(level, spike_tiles, tx, ty):
            tr = _tile_rect(level, tx, ty)
            if intersects(rect_x, tr):
                died = True

    body.x = new_x

    # Vertical movement
    new_y = body.y + body.vy * dt
    rect_y = RectF(body.x, new_y, body.w, body.h)
    grounded = False
    for tx, ty in _iter_tiles_overlapping(level, rect_y):
        if _solid_at(level, solid_tiles, tx, ty):
            tr = _tile_rect(level, tx, ty)
            if intersects(rect_y, tr):
                if body.vy > 0:
                    new_y = tr.top() - body.h
                    grounded = True
                elif body.vy < 0:
                    # Head-bonk into a block is death (Geometry Dash-like).
                    new_y = tr.bottom()
                    died = True
                body.vy = 0.0
                rect_y = RectF(body.x, new_y, body.w, body.h)
        if _kill_at(level, spike_tiles, tx, ty):
            tr = _tile_rect(level, tx, ty)
            if intersects(rect_y, tr):
                died = True

    body.y = new_y
    body.grounded = grounded

    # Keep within world vertically (simple clamp)
    max_y = (level.height * level.tile_size) - body.h
    if body.y > max_y:
        body.y = max_y
        body.vy = 0.0
        body.grounded = True

    if body.y < 0:
        body.y = 0
        body.vy = 0.0

    return StepPhysicsResult(died=died)

