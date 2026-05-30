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
    from gdrl.training import ReinforceConfig, evaluate_policy, save_policy, train_reinforce

    ap = argparse.ArgumentParser(description="Train TinyJumpCNN with REINFORCE.")
    ap.add_argument("level_path", nargs="?", default="levels/tiny_spikes.json")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy-weight", type=float, default=0.0)
    ap.add_argument("--value-weight", type=float, default=0.01)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/tiny_reinforce.pt")
    args = ap.parse_args()

    level_paths = collect_level_paths(Path(args.level_path))
    envs = [GDRLEnv(path) for path in level_paths]
    cfg = ReinforceConfig(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        entropy_weight=args.entropy_weight,
        value_weight=args.value_weight,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
        log_every=args.log_every,
    )
    recent_rewards: list[float] = []
    recent_completed: list[bool] = []
    recent_jump_rates: list[float] = []

    def on_episode(i: int, reward: float, completed: bool, jump_rate: float) -> None:
        recent_rewards.append(reward)
        recent_completed.append(completed)
        recent_jump_rates.append(jump_rate)
        if args.log_every <= 0 or i % args.log_every != 0:
            return
        window = min(args.log_every, len(recent_rewards))
        avg_reward = sum(recent_rewards[-window:]) / window
        success = sum(recent_completed[-window:]) / window
        avg_jump = sum(recent_jump_rates[-window:]) / window
        print(
            f"episode={i} avg_reward={avg_reward:.3f} "
            f"success={success:.3f} jump_rate={avg_jump:.3f}"
        )

    model, result = train_reinforce(envs, config=cfg, on_episode=on_episode)
    save_policy(model, args.checkpoint)
    eval_results = [(path, evaluate_policy(GDRLEnv(path), model, episodes=args.eval_episodes)) for path in level_paths]

    last = result.rewards[-1] if result.rewards else 0.0
    success_rate = sum(result.completed) / max(1, len(result.completed))
    print(f"episodes={len(result.rewards)}")
    print(f"last_reward={last:.3f}")
    print(f"train_success_rate={success_rate:.3f}")
    for path, eval_result in eval_results:
        print(
            f"eval_level={path.stem} success_rate={eval_result.success_rate:.3f} "
            f"avg_progress={eval_result.avg_progress:.3f} "
            f"jump_rate={eval_result.avg_jump_rate:.3f}"
        )
    print(f"checkpoint={args.checkpoint}")
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
