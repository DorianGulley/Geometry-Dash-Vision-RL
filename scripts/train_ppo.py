from __future__ import annotations

import argparse
import csv
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
    ap.add_argument("--metrics-dir", default="artifacts/ppo")
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
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_training_metrics(result, metrics_dir / "training_episodes.csv")

    print(f"updates={args.updates}")
    print(f"episodes={len(result.rewards)}")
    print(f"train_success_rate={sum(result.completed) / max(1, len(result.completed)):.3f}")
    print(f"train_avg_jump_rate={sum(result.jump_rates) / max(1, len(result.jump_rates)):.3f}")
    eval_rows = []
    for path in paths:
        eval_result = evaluate_policy(GDRLEnv(path), model, episodes=args.eval_episodes)
        eval_rows.append((path.stem, eval_result))
        print(
            f"eval_level={path.stem} success_rate={eval_result.success_rate:.3f} "
            f"avg_progress={eval_result.avg_progress:.3f} "
            f"jump_rate={eval_result.avg_jump_rate:.3f}"
        )
    write_eval_metrics(eval_rows, metrics_dir / "eval_by_level.csv")
    write_plots(result, eval_rows, metrics_dir)
    print(f"metrics_dir={metrics_dir}")
    print(f"checkpoint={args.checkpoint}")
    return 0


def write_training_metrics(result, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "completed", "jump_rate"])
        for i, (reward, completed, jump_rate) in enumerate(
            zip(result.rewards, result.completed, result.jump_rates),
            start=1,
        ):
            writer.writerow([i, reward, int(completed), jump_rate])


def write_eval_metrics(rows, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "episodes", "success_rate", "avg_progress", "avg_reward", "avg_jump_rate"])
        for level, result in rows:
            writer.writerow([
                level,
                result.episodes,
                result.success_rate,
                result.avg_progress,
                result.avg_reward,
                result.avg_jump_rate,
            ])


def write_plots(result, eval_rows, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_training_series(result.rewards, "Reward", out_dir / "training_reward.png")
    plot_training_series(rolling_mean([float(x) for x in result.completed], 10), "Rolling success", out_dir / "training_success.png")
    plot_training_series(rolling_mean(result.jump_rates, 10), "Rolling jump rate", out_dir / "training_jump_rate.png")

    levels = [level for level, _ in eval_rows]
    success = [res.success_rate for _, res in eval_rows]
    progress = [res.avg_progress for _, res in eval_rows]
    x = range(len(levels))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.18 for i in x], success, width=0.36, label="Success")
    plt.bar([i + 0.18 for i in x], progress, width=0.36, label="Progress")
    plt.xticks(list(x), levels, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title("PPO eval by level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eval_by_level.png", dpi=160)
    plt.close()


def plot_training_series(values: list[float], ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        segment = values[start : i + 1]
        out.append(sum(segment) / len(segment))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
