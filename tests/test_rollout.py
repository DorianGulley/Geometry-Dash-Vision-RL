from pathlib import Path

from gdrl.experiments import run_rollout
from gdrl.input import NullInputProvider, ScriptedInputProvider
from gdrl.levels import load_level
from gdrl.sim import Simulator


def _example_level() -> Path:
    return Path(__file__).resolve().parents[1] / "levels" / "example_level.json"


def test_rollout_null_provider_terminates() -> None:
    level = load_level(_example_level())
    sim = Simulator(level)
    result = run_rollout(sim, NullInputProvider())
    assert result.episode is not None
    assert result.episode.done
    assert len(result.trajectory) > 0
    assert result.final_state is not None


def test_rollout_scripted_jumps_recorded() -> None:
    level = load_level(_example_level())
    sim = Simulator(level)
    result = run_rollout(sim, ScriptedInputProvider(jump_at_timesteps={0, 10}))
    assert result.episode is not None
    jumped = [a.jump_pressed for _, a in result.trajectory]
    assert jumped[0] is True
    assert jumped[10] is True
    assert jumped[5] is False
