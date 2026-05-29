from __future__ import annotations

import csv
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    import matplotlib.pyplot as plt

    from gdrl.envs import GDRLEnv
    from gdrl.training import ReinforceConfig, evaluate_policy, eval_random, save_policy, train_reinforce

    root = Path("artifacts/milestone3")
    plots = root / "plots"
    tables = root / "tables"
    ckpts = root / "checkpoints"
    for p in (plots, tables, ckpts):
        p.mkdir(parents=True, exist_ok=True)

    level_paths = sorted(Path("levels/curriculum").glob("*.json"))
    random_rows = eval_random(level_paths, episodes=30, seed=0)
    write_random_csv(random_rows, tables / "random_baseline.csv")

    train_level = Path("levels/curriculum/one_spike.json")
    env = GDRLEnv(train_level)
    rewards: list[float] = []
    completed: list[bool] = []

    def on_episode(i: int, reward: float, done: bool) -> None:
        rewards.append(reward)
        completed.append(done)

    cfg = ReinforceConfig(episodes=80, lr=1e-3, seed=0, entropy_weight=0.02)
    model, train_result = train_reinforce(env, config=cfg, on_episode=on_episode)
    save_policy(model, ckpts / "one_spike_reinforce.pt")
    write_training_csv(train_result.rewards, train_result.completed, tables / "cnn_training.csv")

    eval_rows = []
    for level_path in level_paths:
        result = evaluate_policy(GDRLEnv(level_path), model, episodes=20)
        eval_rows.append((level_path.stem, result))
    write_cnn_csv(eval_rows, tables / "cnn_eval.csv")
    write_comparison_csv(random_rows, eval_rows, tables / "comparison.csv")

    plot_training(train_result.rewards, plots / "training_reward.png", "CNN REINFORCE reward")
    plot_training(rolling_success(train_result.completed), plots / "training_success.png", "CNN rolling success")
    plot_comparison(tables / "comparison.csv", plots / "comparison_success.png")
    print(root)
    return 0


def write_random_csv(rows, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "level", "episodes", "success_rate", "avg_progress", "avg_reward"])
        for row in rows:
            writer.writerow(["random", row.level_id, row.episodes, row.success_rate, row.avg_progress, row.avg_reward])


def write_training_csv(rewards: list[float], completed: list[bool], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "completed"])
        for i, (reward, done) in enumerate(zip(rewards, completed), start=1):
            writer.writerow([i, reward, int(done)])


def write_cnn_csv(rows, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "level", "episodes", "success_rate", "avg_progress", "avg_reward"])
        for level, result in rows:
            writer.writerow(["cnn_reinforce", level, result.episodes, result.success_rate, result.avg_progress, result.avg_reward])


def write_comparison_csv(random_rows, cnn_rows, path: Path) -> None:
    cnn_by_level = {level: result for level, result in cnn_rows}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "random_success", "cnn_success", "random_progress", "cnn_progress"])
        for row in random_rows:
            cnn = cnn_by_level[row.level_id]
            writer.writerow([row.level_id, row.success_rate, cnn.success_rate, row.avg_progress, cnn.avg_progress])


def rolling_success(completed: list[bool], window: int = 10) -> list[float]:
    out = []
    for i in range(len(completed)):
        start = max(0, i + 1 - window)
        vals = completed[start : i + 1]
        out.append(sum(vals) / len(vals))
    return out


def plot_training(values: list[float], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_comparison(csv_path: Path, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    levels = [r["level"] for r in rows]
    random_vals = [float(r["random_success"]) for r in rows]
    cnn_vals = [float(r["cnn_success"]) for r in rows]
    x = range(len(levels))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.18 for i in x], random_vals, width=0.36, label="Random")
    plt.bar([i + 0.18 for i in x], cnn_vals, width=0.36, label="CNN RL")
    plt.xticks(list(x), levels, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Completion rate")
    plt.title("Random baseline vs CNN RL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
