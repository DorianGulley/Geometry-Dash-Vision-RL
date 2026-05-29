from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    _add_src_to_path()

    import pygame

    from gdrl.envs import GDRLEnv
    from gdrl.levels import load_level

    out = Path("artifacts/milestone3/frames")
    out.mkdir(parents=True, exist_ok=True)
    levels = sorted(Path("levels/curriculum").glob("*.json"))

    pygame.init()
    for path in levels:
        env = GDRLEnv(path)
        obs, _ = env.reset()
        save_rgb(env.render(), out / f"full_render_{path.stem}.png")
        save_gray(obs[0], out / f"model_obs_{path.stem}.png")
        save_rgb(debug_overlay(env), out / f"debug_overlay_{path.stem}.png")

    make_contact_sheet(out, "full_render", out / "full_render_contact.png")
    make_contact_sheet(out, "model_obs", out / "model_obs_contact.png")
    make_contact_sheet(out, "debug_overlay", out / "debug_overlay_contact.png")
    print(out)
    return 0


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    import pygame

    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    pygame.image.save(surf, path)


def save_gray(gray: np.ndarray, path: Path) -> None:
    save_rgb(np.repeat(gray[:, :, None], 3, axis=2), path)


def debug_overlay(env) -> np.ndarray:
    import pygame

    level = env.level
    surf = pygame.surfarray.make_surface(np.transpose(env.render(), (1, 0, 2)))
    cam_x = env.renderer.compute_camera_x(env.sim.player.x)
    draw_player_box(surf, env.sim.player, cam_x)
    for tile in level.tiles:
        if tile.type.value == "spike":
            draw_spike_triangle(surf, level, tile.x, tile.y, cam_x)
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))


def draw_player_box(surf, player, cam_x: float) -> None:
    import pygame

    rect = pygame.Rect(int(player.x - cam_x), int(player.y), int(player.w), int(player.h))
    pygame.draw.rect(surf, (40, 255, 120), rect, width=2)


def draw_spike_triangle(surf, level, tx: int, ty: int, cam_x: float) -> None:
    import pygame

    s = level.tile_size
    left = tx * s - cam_x
    top = ty * s
    points = [
        (int(left), int(top + s)),
        (int(left + s / 2), int(top)),
        (int(left + s), int(top + s)),
    ]
    pygame.draw.polygon(surf, (255, 240, 40), points, width=2)


def make_contact_sheet(folder: Path, prefix: str, out_path: Path) -> None:
    import pygame

    paths = sorted(folder.glob(f"{prefix}_*.png"))
    if not paths:
        return
    images = [pygame.image.load(p) for p in paths]
    thumb_w, thumb_h = 240, 135
    sheet = pygame.Surface((thumb_w * len(images), thumb_h))
    for i, img in enumerate(images):
        sheet.blit(pygame.transform.scale(img, (thumb_w, thumb_h)), (i * thumb_w, 0))
    pygame.image.save(sheet, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
