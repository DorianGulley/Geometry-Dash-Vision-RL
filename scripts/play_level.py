from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    import pygame

    from gdrl.input import HumanInputProvider
    from gdrl.levels import load_level, validate_level
    from gdrl.sim import Action, Simulator
    from gdrl.sim.renderer import Renderer, init_pygame_window

    ap = argparse.ArgumentParser()
    ap.add_argument("level_path", type=str, help="Path to level JSON")
    args = ap.parse_args()

    level = load_level(args.level_path)
    errs = validate_level(level)
    if errs:
        print("Level validation errors:")
        for e in errs:
            print(f"- {e.code}: {e.message}")
        return 2

    screen = init_pygame_window(level, title=f"GDRL Play — {level.meta.level_id}")
    clock = pygame.time.Clock()

    sim = Simulator(level)
    renderer = Renderer(level)
    input_provider = HumanInputProvider()

    sim.reset()
    input_provider.reset()

    fixed_dt = float(level.physics.fixed_dt)
    accumulator = 0.0
    last = time.perf_counter()
    render_fps = int(level.camera.screen_width)  # dummy init to silence lints
    render_fps = 60

    paused_on_done = False

    running = True
    while running:
        now = time.perf_counter()
        frame_dt = now - last
        last = now
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    sim.reset()
                    input_provider.reset()
                    paused_on_done = False

        # Step fixed-timestep simulation. If episode done, stop stepping until restart.
        if not paused_on_done:
            # Action sampled once per rendered frame; reused for any catch-up steps.
            action: Action = input_provider.get_action(sim.state())
            while accumulator >= fixed_dt:
                _, result = sim.step(action)
                accumulator -= fixed_dt
                if result is not None:
                    print(f"Episode done: {result}")
                    paused_on_done = True
                    break

        surf = renderer.render_to_surface(player=sim.player, timestep=sim.state().timestep)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(render_fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

