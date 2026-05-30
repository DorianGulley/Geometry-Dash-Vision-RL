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
    from gdrl.training import PPOConfig, evaluate_policy, save_policy, train_ppo
    from gdrl.training.ppo import level_paths

    ap = argparse.ArgumentParser(description="Train TinyJumpCNN with a small PPO loop.")
    ap.add_argument("levels_path", nargs="?", default="levels/curriculum")
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--value-weight", type=float, default=0.01)
    ap.add_argument("--entropy-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--checkpoint", default="checkpoints/tiny_ppo.pt")
    args = ap.parse_args()

    paths = level_paths(args.levels_path)
    envs = [GDRLEnv(path) for path in paths]
    cfg = PPOConfig(
        updates=args.updates,
        batch_episodes=args.batch_episodes,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
    )

    model, result = train_ppo(envs, config=cfg)
    save_policy(model, args.checkpoint)

    print(f"updates={args.updates}")
    print(f"episodes={len(result.rewards)}")
    print(f"train_success_rate={sum(result.completed) / max(1, len(result.completed)):.3f}")
    print(f"train_avg_jump_rate={sum(result.jump_rates) / max(1, len(result.jump_rates)):.3f}")
    for path in paths:
        eval_result = evaluate_policy(GDRLEnv(path), model, episodes=args.eval_episodes)
        print(
            f"eval_level={path.stem} success_rate={eval_result.success_rate:.3f} "
            f"avg_progress={eval_result.avg_progress:.3f} "
            f"jump_rate={eval_result.avg_jump_rate:.3f}"
        )
    print(f"checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
