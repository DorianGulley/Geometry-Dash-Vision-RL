from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    from gdrl.levels import make_single_spike_sweep, save_level

    ap = argparse.ArgumentParser(description="Generate simple fixed training levels.")
    ap.add_argument("--out-dir", default="levels/generated/single_spike")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for level in make_single_spike_sweep():
        save_level(level, out_dir / f"{level.meta.level_id}.json")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
