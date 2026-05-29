from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    from gdrl.envs import GDRLEnv
    from gdrl.training import evaluate_policy, load_policy

    ap = argparse.ArgumentParser(description="Evaluate a trained TinyJumpCNN checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--level", default="levels/tiny_spikes.json")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    model = load_policy(args.checkpoint)
    result = evaluate_policy(GDRLEnv(args.level), model, episodes=args.episodes)
    print(f"episodes={result.episodes}")
    print(f"success_rate={result.success_rate:.3f}")
    print(f"avg_reward={result.avg_reward:.3f}")
    print(f"avg_progress={result.avg_progress:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
