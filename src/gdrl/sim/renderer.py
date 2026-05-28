from __future__ import annotations

from dataclasses import dataclass

from gdrl.levels import Level, TileType
from gdrl.sim.physics import PlayerBody


@dataclass(frozen=True)
class RenderConfig:
    draw_grid: bool = False
    draw_hud: bool = True


class Renderer:
    def __init__(self, level: Level, *, config: RenderConfig | None = None) -> None:
        self.level = level
        self.config = config or RenderConfig()

        self.screen_w = int(level.camera.screen_width)
        self.screen_h = int(level.camera.screen_height)
        self.player_screen_x = int(level.camera.player_screen_x)

        self._surface = None

    def ensure_surface(self) -> "object":
        import pygame

        if self._surface is None:
            self._surface = pygame.Surface((self.screen_w, self.screen_h))
        return self._surface

    def compute_camera_x(self, player_x: float) -> float:
        world_w_px = self.level.width * self.level.tile_size
        cam_x = player_x - float(self.player_screen_x)
        cam_x = max(0.0, min(cam_x, max(0.0, world_w_px - self.screen_w)))
        return cam_x

    def render_to_surface(self, *, player: PlayerBody, timestep: int = 0) -> "object":
        import pygame

        surf = self.ensure_surface()
        s = self.level.tile_size

        cam_x = self.compute_camera_x(player.x)

        # Colors: simple + high-contrast
        bg = (18, 18, 22)
        floor_c = (60, 60, 70)
        block_c = (210, 210, 220)
        spike_c = (220, 80, 80)
        player_c = (80, 180, 255)
        start_c = (80, 220, 120)
        end_c = (240, 210, 80)

        surf.fill(bg)

        # Floor (implicit)
        floor_y_px = self.level.floor_y() * s
        pygame.draw.rect(
            surf,
            floor_c,
            pygame.Rect(0, floor_y_px, self.screen_w, self.screen_h - floor_y_px),
        )

        # Visible tile range
        min_tx = int(cam_x // s) - 1
        max_tx = int((cam_x + self.screen_w) // s) + 1
        min_tx = max(0, min_tx)
        max_tx = min(self.level.width - 1, max_tx)

        blocks = {(t.x, t.y): t.type for t in self.level.tiles}

        for ty in range(0, self.level.height - 1):  # exclude implicit floor row
            for tx in range(min_tx, max_tx + 1):
                tt = blocks.get((tx, ty))
                if tt is None:
                    continue
                x = tx * s - cam_x
                y = ty * s
                r = pygame.Rect(int(x), int(y), int(s), int(s))
                if tt == TileType.BLOCK:
                    pygame.draw.rect(surf, block_c, r)
                elif tt == TileType.SPIKE:
                    # Triangle spike
                    p1 = (r.left, r.bottom)
                    p2 = (r.centerx, r.top)
                    p3 = (r.right, r.bottom)
                    pygame.draw.polygon(surf, spike_c, [p1, p2, p3])

        # Start/end markers
        for (pos, col) in ((self.level.start, start_c), (self.level.end, end_c)):
            x = pos.x * s - cam_x
            y = pos.y * s
            r = pygame.Rect(int(x), int(y), int(s), int(s))
            pygame.draw.rect(surf, col, r, width=2)

        # Player
        pr = pygame.Rect(int(player.x - cam_x), int(player.y), int(player.w), int(player.h))
        pygame.draw.rect(surf, player_c, pr)

        if self.config.draw_grid:
            grid_c = (35, 35, 42)
            for tx in range(min_tx, max_tx + 1):
                gx = int(tx * s - cam_x)
                pygame.draw.line(surf, grid_c, (gx, 0), (gx, self.screen_h), 1)
            for ty in range(0, self.level.height + 1):
                gy = int(ty * s)
                pygame.draw.line(surf, grid_c, (0, gy), (self.screen_w, gy), 1)

        if self.config.draw_hud:
            font = pygame.font.Font(None, 22)
            prog = 0.0
            # Avoid importing Simulator to keep renderer small; compute rough progress by x position.
            start_x = self.level.start.x * s
            end_x = self.level.end.x * s
            denom = max(1.0, end_x - start_x)
            prog = max(0.0, min(1.0, (player.x - start_x) / denom))
            text = f"t={timestep}  x={player.x:.1f}  y={player.y:.1f}  vy={player.vy:.1f}  prog={prog:.3f}"
            img = font.render(text, True, (230, 230, 240))
            surf.blit(img, (8, 8))

        return surf


def init_pygame_window(level: Level, title: str = "GDRL Simulator") -> "object":
    import pygame

    pygame.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((int(level.camera.screen_width), int(level.camera.screen_height)))
    return screen

