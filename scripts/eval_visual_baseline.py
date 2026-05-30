from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    from gdrl.envs import GDRLEnv, run_episode
    from gdrl.training import SpikeWindowPolicy

    ap = argparse.ArgumentParser(description="Evaluate the simple visual spike-window baseline.")
    ap.add_argument("levels_path", nargs="?", default="levels/curriculum")
    args = ap.parse_args()

    policy = SpikeWindowPolicy()
    paths = collect_level_paths(Path(args.levels_path))
    print("level,completed,progress,reward,steps")
    for path in paths:
        stats = run_episode(GDRLEnv(path), policy)
        episode = stats.info.get("episode", {})
        print(
            f"{path.stem},{int(bool(episode.get('completed', False)))},"
            f"{float(episode.get('progress', stats.info['progress'])):.3f},"
            f"{stats.reward:.3f},{stats.steps}"
        )
    return 0


def collect_level_paths(path: Path) -> list[Path]:
    if path.is_dir():
        out = sorted(path.glob("*.json"))
    else:
        out = [path]
    if not out:
        raise FileNotFoundError(f"No level JSON files found at {path}.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
