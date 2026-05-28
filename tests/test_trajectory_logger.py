import csv
import json
import shutil
from pathlib import Path

from gdrl.experiments import run_rollout
from gdrl.input import NullInputProvider
from gdrl.levels import load_level
from gdrl.logging import TrajectoryLogger
from gdrl.sim import Simulator


def test_trajectory_logger_writes_metadata_and_states(tmp_path: Path) -> None:
    level = load_level(Path(__file__).resolve().parents[1] / "levels" / "example_level.json")
    sim = Simulator(level)
    result = run_rollout(sim, NullInputProvider())
    assert result.episode is not None

    logger = TrajectoryLogger(
        level=level,
        dataset_root=tmp_path,
        provider="null",
        level_path="levels/example_level.json",
        capture_every_n_steps=0,
        episode_id="episode_000099",
    )
    out = logger.write_from_rollout(result)
    assert out.name == "episode_000099"
    assert (out / "metadata.json").is_file()
    assert (out / "states.csv").is_file()

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["episode_id"] == "episode_000099"
    assert meta["level"]["level_id"] == level.meta.level_id
    assert meta["rollout"]["provider"] == "null"
    assert meta["episode"]["reason"] in {"death", "completed", "timeout", "quit"}

    with (out / "states.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.trajectory)
    assert rows[0]["timestep"] == "0"
    assert rows[-1]["done"] == "1"
    assert "frame_path" in rows[0]

    shutil.rmtree(tmp_path)
