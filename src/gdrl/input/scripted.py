from __future__ import annotations

from collections.abc import Callable

from gdrl.sim import Action, State

from .base import InputProvider


class NullInputProvider(InputProvider):
    def get_action(self, state: State) -> Action:
        return Action(jump_pressed=False)


class ScriptedInputProvider(InputProvider):
    """
    Very small scripted provider for debugging.

    You can pass either:
    - `jump_at_timesteps`: set of integer timesteps when jump should be pressed
    - `policy_fn`: Callable mapping state->Action
    """

    def __init__(
        self,
        *,
        jump_at_timesteps: set[int] | None = None,
        policy_fn: Callable[[State], Action] | None = None,
    ) -> None:
        self.jump_at_timesteps = jump_at_timesteps or set()
        self.policy_fn = policy_fn

    def get_action(self, state: State) -> Action:
        if self.policy_fn is not None:
            return self.policy_fn(state)
        return Action(jump_pressed=(state.timestep in self.jump_at_timesteps))

