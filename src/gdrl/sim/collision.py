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


def rect_intersects_triangle(rect: RectF, tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> bool:
    if any(point_in_rect(p, rect) for p in tri):
        return True
    if any(point_in_triangle(p, tri) for p in rect_points(rect)):
        return True
    return any(
        segments_intersect(a1, a2, b1, b2)
        for a1, a2 in rect_edges(rect)
        for b1, b2 in triangle_edges(tri)
    )


def rect_points(rect: RectF) -> list[tuple[float, float]]:
    return [
        (rect.left(), rect.top()),
        (rect.right(), rect.top()),
        (rect.right(), rect.bottom()),
        (rect.left(), rect.bottom()),
    ]


def rect_edges(rect: RectF) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = rect_points(rect)
    return list(zip(pts, pts[1:] + pts[:1]))


def triangle_edges(
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = list(tri)
    return list(zip(pts, pts[1:] + pts[:1]))


def point_in_rect(p: tuple[float, float], rect: RectF) -> bool:
    x, y = p
    return rect.left() < x < rect.right() and rect.top() < y < rect.bottom()


def point_in_triangle(
    p: tuple[float, float],
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    eps = 1e-9
    a, b, c = tri
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    if abs(d1) <= eps or abs(d2) <= eps or abs(d3) <= eps:
        return False
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def sign(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    return o1 * o2 < 0 and o3 * o4 < 0


def orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
