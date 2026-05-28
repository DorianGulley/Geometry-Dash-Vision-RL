from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from gdrl.input import InputProvider
from gdrl.sim import Action, EpisodeResult, Simulator, State


@dataclass(frozen=True)
class RolloutResult:
    """One completed (or stopped) episode rollout."""

    trajectory: list[tuple[State, Action]] = field(default_factory=list)
    final_state: State | None = None
    episode: EpisodeResult | None = None


def run_rollout(
    sim: Simulator,
    input_provider: InputProvider,
    *,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[State, Action], None] | None = None,
) -> RolloutResult:
    """
    Run one episode until the simulator terminates or `should_stop()` returns True.

    The simulator advances with a fixed `dt` per step. `input_provider` is queried
    once per simulation step; the simulator never reads pygame keyboard state.
    """
    sim.reset()
    input_provider.reset()

    trajectory: list[tuple[State, Action]] = []

    while True:
        if should_stop is not None and should_stop():
            final = sim.state()
            return RolloutResult(
                trajectory=trajectory,
                final_state=final,
                episode=EpisodeResult(
                    done=True,
                    completed=final.completed,
                    died=final.dead,
                    progress=sim.progress(),
                    reason="quit",
                ),
            )

        state = sim.state()
        action = input_provider.get_action(state)
        trajectory.append((state, action))

        if on_step is not None:
            on_step(state, action)

        _, result = sim.step(action)
        if result is not None:
            return RolloutResult(
                trajectory=trajectory,
                final_state=sim.state(),
                episode=result,
            )
