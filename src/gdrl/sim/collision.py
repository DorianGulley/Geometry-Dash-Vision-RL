from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectF:
    x: float
    y: float
    w: float
    h: float

    def left(self) -> float:
        return self.x

    def right(self) -> float:
        return self.x + self.w

    def top(self) -> float:
        return self.y

    def bottom(self) -> float:
        return self.y + self.h


def intersects(a: RectF, b: RectF) -> bool:
    return not (
        a.right() <= b.left()
        or a.left() >= b.right()
        or a.bottom() <= b.top()
        or a.top() >= b.bottom()
    )

