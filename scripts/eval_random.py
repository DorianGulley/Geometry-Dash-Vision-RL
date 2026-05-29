from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def _level_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*.json"))


def main() -> int:
    _add_src_to_path()

    from gdrl.training import eval_random

    ap = argparse.ArgumentParser(description="Evaluate random jump/no-jump policy.")
    ap.add_argument("--levels-dir", default="levels/curriculum")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = eval_random(_level_paths(Path(args.levels_dir)), episodes=args.episodes, seed=args.seed)
    print("level,episodes,success_rate,avg_progress,avg_reward")
    for row in rows:
        print(
            f"{row.level_id},{row.episodes},{row.success_rate:.3f},"
            f"{row.avg_progress:.3f},{row.avg_reward:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
