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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/tiny_reinforce.pt")
    args = ap.parse_args()

    env = GDRLEnv(args.level_path)
    cfg = ReinforceConfig(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        log_every=args.log_every,
    )
    recent_rewards: list[float] = []
    recent_completed: list[bool] = []

    def on_episode(i: int, reward: float, completed: bool) -> None:
        recent_rewards.append(reward)
        recent_completed.append(completed)
        if args.log_every <= 0 or i % args.log_every != 0:
            return
        window = min(args.log_every, len(recent_rewards))
        avg_reward = sum(recent_rewards[-window:]) / window
        success = sum(recent_completed[-window:]) / window
        print(f"episode={i} avg_reward={avg_reward:.3f} success={success:.3f}")

    model, result = train_reinforce(env, config=cfg, on_episode=on_episode)
    save_policy(model, args.checkpoint)
    eval_result = evaluate_policy(GDRLEnv(args.level_path), model, episodes=args.eval_episodes)

    last = result.rewards[-1] if result.rewards else 0.0
    success_rate = sum(result.completed) / max(1, len(result.completed))
    print(f"episodes={len(result.rewards)}")
    print(f"last_reward={last:.3f}")
    print(f"train_success_rate={success_rate:.3f}")
    print(f"eval_success_rate={eval_result.success_rate:.3f}")
    print(f"eval_avg_progress={eval_result.avg_progress:.3f}")
    print(f"checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
