from __future__ import annotations

import json
from pathlib import Path

from .level_schema import Level


def load_level(path: str | Path) -> Level:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return Level.from_dict(d)


def save_level(level: Level, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(level.to_dict(), f, indent=2, sort_keys=False)
        f.write("\n")

