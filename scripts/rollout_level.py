from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def _parse_jump_timesteps(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def main() -> int:
    _add_src_to_path()

    from gdrl.experiments import run_rollout
    from gdrl.input import HumanInputProvider, NullInputProvider, ScriptedInputProvider
    from gdrl.levels import load_level, validate_level
    from gdrl.sim import Simulator
    from gdrl.sim.renderer import Renderer, init_pygame_window

    ap = argparse.ArgumentParser(description="Roll out one episode with any InputProvider.")
    ap.add_argument("level_path", type=str, help="Path to level JSON")
    ap.add_argument(
        "--provider",
        choices=["null", "scripted", "human"],
        default="null",
        help="Input provider (default: null)",
    )
    ap.add_argument(
        "--jump-at",
        type=str,
        default="",
        help="Comma-separated timesteps to jump (scripted provider only)",
    )
    ap.add_argument("--render", action="store_true", help="Show pygame window while rolling out")
    ap.add_argument("--fps", type=int, default=60, help="Render FPS when --render is set")
    ap.add_argument("--log", action="store_true", help="Save trajectory to dataset/episodes/")
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Dataset root (default: dataset)",
    )
    ap.add_argument(
        "--capture-every",
        type=int,
        default=0,
        help="Save a PNG frame every N steps (0 = disabled; requires --log)",
    )
    args = ap.parse_args()

    level = load_level(args.level_path)
    errs = validate_level(level)
    if errs:
        print("Level validation errors:")
        for e in errs:
            print(f"- {e.code}: {e.message}")
        return 2

    if args.provider == "null":
        input_provider = NullInputProvider()
    elif args.provider == "scripted":
        input_provider = ScriptedInputProvider(jump_at_timesteps=_parse_jump_timesteps(args.jump_at))
    else:
        input_provider = HumanInputProvider()

    sim = Simulator(level)

    screen = None
    renderer = None
    clock = None
    quit_requested = False

    if args.render:
        import pygame

        screen = init_pygame_window(level, title=f"GDRL Rollout — {level.meta.level_id}")
        renderer = Renderer(level)
        clock = pygame.time.Clock()

        def should_stop() -> bool:
            nonlocal quit_requested
            import pygame

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True
            return quit_requested

        def on_step(state, action) -> None:
            assert renderer is not None and screen is not None and clock is not None
            import pygame

            surf = renderer.render_to_surface(player=sim.player, timestep=state.timestep)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)

        result = run_rollout(sim, input_provider, should_stop=should_stop, on_step=on_step)
        import pygame

        pygame.quit()
    else:
        result = run_rollout(sim, input_provider)

    print(f"Steps: {len(result.trajectory)}")
    if result.episode is not None:
        print(f"Episode: {result.episode}")
    else:
        print("Episode: (no result)")

    if args.log:
        if result.episode is None:
            print("Skipping log: episode did not finish with a result.")
            return 1

        from gdrl.logging import TrajectoryLogger

        import pygame

        pygame.init()
        capture_renderer = None
        if args.capture_every > 0:
            capture_renderer = Renderer(level)

        logger = TrajectoryLogger(
            level=level,
            dataset_root=Path(args.dataset_dir),
            provider=args.provider,
            level_path=str(Path(args.level_path)),
            capture_every_n_steps=args.capture_every,
        )
        episode_dir = logger.write_from_rollout(result, renderer=capture_renderer)
        print(f"Logged episode to: {episode_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
