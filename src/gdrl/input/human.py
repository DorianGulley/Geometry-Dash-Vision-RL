from __future__ import annotations

from gdrl.sim import Action, State

from .base import InputProvider


class HumanInputProvider(InputProvider):
    """
    Keyboard mapping:
    - Jump: Space or Up
    """

    def __init__(self) -> None:
        self._prev_jump = False

    def reset(self) -> None:
        self._prev_jump = False

    def get_action(self, state: State) -> Action:
        import pygame

        keys = pygame.key.get_pressed()
        jump = bool(keys[pygame.K_SPACE] or keys[pygame.K_UP])

        # Treat as "pressed" (not held) to avoid repeated mid-air jump attempts.
        jump_pressed = jump and not self._prev_jump
        self._prev_jump = jump
        return Action(jump_pressed=jump_pressed)

