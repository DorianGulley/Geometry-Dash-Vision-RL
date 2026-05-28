from __future__ import annotations

from abc import ABC, abstractmethod

from gdrl.sim import Action, State


class InputProvider(ABC):
    """
    Produces an `Action` each simulation step.

    The simulator never reads pygame keyboard state directly; the human provider does.
    """

    def reset(self) -> None:
        return None

    @abstractmethod
    def get_action(self, state: State) -> Action: ...

