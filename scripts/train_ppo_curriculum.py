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
    from gdrl.training import (
        PPOConfig,
        TinyJumpCNN,
        evaluate_policy,
        evaluate_stochastic_policy,
        save_policy,
        train_ppo,
    )
    from gdrl.training.ppo import level_paths

    ap = argparse.ArgumentParser(description="Train PPO through generated spike-level stages.")
    ap.add_argument(
        "--stages",
        nargs="+",
        default=[
            "levels/generated/single_spike",
            "levels/generated/two_spike",
            "levels/generated/three_spike",
        ],
    )
    ap.add_argument("--updates-per-stage", type=int, default=2)
    ap.add_argument("--batch-episodes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--metrics-dir", default="artifacts/ppo_curriculum")
    ap.add_argument("--checkpoint", default="checkpoints/curriculum_ppo.pt")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model = TinyJumpCNN()
    rows = []

    for stage_idx, stage in enumerate(args.stages, start=1):
        paths = level_paths(stage)
        envs = [GDRLEnv(path) for path in paths]
        cfg = PPOConfig(
            updates=args.updates_per_stage,
            batch_episodes=args.batch_episodes,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed + stage_idx - 1,
            eval_episodes=args.eval_episodes,
        )
        model, result = train_ppo(envs, model=model, config=cfg)
        train_success = sum(result.completed) / max(1, len(result.completed))
        train_jump = sum(result.jump_rates) / max(1, len(result.jump_rates))

        for path in paths:
            greedy = evaluate_policy(GDRLEnv(path), model, episodes=args.eval_episodes)
            sampled = evaluate_stochastic_policy(GDRLEnv(path), model, episodes=args.eval_episodes)
            rows.append([
                stage_idx,
                Path(stage).name,
                path.stem,
                train_success,
                train_jump,
                greedy.success_rate,
                sampled.success_rate,
                greedy.avg_progress,
                sampled.avg_progress,
                greedy.avg_jump_rate,
                sampled.avg_jump_rate,
            ])

        print(
            f"stage={stage_idx} name={Path(stage).name} levels={len(paths)} "
            f"train_success={train_success:.3f} train_jump={train_jump:.3f}"
        )

    save_policy(model, args.checkpoint)
    write_stage_metrics(rows, metrics_dir / "stage_eval.csv")
    print(f"metrics_dir={metrics_dir}")
    print(f"checkpoint={args.checkpoint}")
    return 0


def write_stage_metrics(rows: list[list], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "stage_name",
            "level",
            "train_success",
            "train_jump_rate",
            "greedy_success",
            "sampled_success",
            "greedy_progress",
            "sampled_progress",
            "greedy_jump_rate",
            "sampled_jump_rate",
        ])
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
