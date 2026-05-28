from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    from gdrl.editor import EditorApp

    root = Path(__file__).resolve().parents[1]
    levels_dir = root / "levels"
    EditorApp(levels_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
