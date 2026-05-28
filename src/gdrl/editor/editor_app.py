from __future__ import annotations

import re
from pathlib import Path

from gdrl.editor.editor_state import Brush, EditorState, default_new_level
from gdrl.levels import load_level, save_level, validate_level


def _safe_filename(level_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", level_id.strip())
    return s.strip("_") or "untitled"


class TextField:
    def __init__(self, label: str, value: str = "", max_len: int = 64) -> None:
        self.label = label
        self.value = value
        self.max_len = max_len
        self.active = False

    def handle_key(self, key: int, unicode: str, *, pygame_mod) -> bool:
        import pygame

        if not self.active:
            return False
        if key == pygame.K_BACKSPACE:
            self.value = self.value[:-1]
            return True
        if key == pygame.K_RETURN:
            self.active = False
            return True
        if unicode and len(self.value) < self.max_len and unicode.isprintable():
            self.value += unicode
            return True
        return False


class EditorApp:
  """Minimal pygame level editor."""

  PANEL_H = 96
  TILE_SIZE = 32

  def __init__(self, levels_dir: Path) -> None:
      self.levels_dir = Path(levels_dir)
      self.levels_dir.mkdir(parents=True, exist_ok=True)
      self.state = default_new_level()
      self.brush = Brush.BLOCK
      self.message = ""
      self.load_menu_open = False
      self.load_menu_index = 0
      self._level_files: list[Path] = []

      self.field_name = TextField("Name")
      self.field_author = TextField("Author")
      self.field_id = TextField("Level ID")
      self._fields = [self.field_name, self.field_author, self.field_id]
      self._sync_fields_from_state()

  def _sync_fields_from_state(self) -> None:
      self.field_name.value = self.state.meta.name
      self.field_author.value = self.state.meta.author
      self.field_id.value = self.state.meta.level_id

  def _sync_state_from_fields(self) -> None:
      from gdrl.levels import LevelMetadata

      self.state.meta = LevelMetadata(
          schema_version=self.state.meta.schema_version,
          level_id=self.field_id.value.strip() or "untitled",
          name=self.field_name.value.strip() or "Untitled",
          author=self.field_author.value.strip(),
          split=self.state.meta.split,
          tags=list(self.state.meta.tags),
          version=self.state.meta.version,
      )

  def _list_level_files(self) -> list[Path]:
      return sorted(self.levels_dir.glob("*.json"))

  def _refresh_load_menu(self) -> None:
      self._level_files = self._list_level_files()
      self.load_menu_index = min(self.load_menu_index, max(0, len(self._level_files) - 1))

  def _load_file(self, path: Path) -> None:
      level = load_level(path)
      self.state = EditorState.from_level(level, path=path.resolve())
      self._sync_fields_from_state()
      self.message = f"Loaded {path.name}"
      self.load_menu_open = False

  def _save(self) -> None:
      self._sync_state_from_fields()
      name = self.state.meta.name.strip()
      author = self.state.meta.author.strip()
      if not name:
          self.message = "Save failed: name is required."
          return
      if not author:
          self.message = "Save failed: author is required."
          return

      if self.state.end.x <= self.state.start.x:
          self.message = "Save failed: end must be to the right of start."
          return

      self.state.prepare_for_save()
      level = self.state.to_level()
      errs = validate_level(level)
      if errs:
          self.message = f"Save failed: {errs[0].message}"
          return

      fname = _safe_filename(self.state.meta.level_id) + ".json"
      out = self.levels_dir / fname
      save_level(level, out)
      self.state.loaded_path = out.resolve()
      self.state.dirty = False
      self.state.meta = level.meta
      self._sync_fields_from_state()
      self.message = f"Saved {out.name} (v{level.meta.version})"

  def _new_level(self) -> None:
      self.state = default_new_level()
      self._sync_fields_from_state()
      self.message = "New level"

  def run(self) -> None:
      import pygame

      pygame.init()
      screen_w, screen_h = 960, 540 + self.PANEL_H
      screen = pygame.display.set_mode((screen_w, screen_h))
      pygame.display.set_caption("GDRL Level Editor")
      clock = pygame.time.Clock()
      font = pygame.font.Font(None, 22)
      font_lg = pygame.font.Font(None, 28)

      grid_h = screen_h - self.PANEL_H
      tiles_visible_x = screen_w // self.TILE_SIZE

      dragging = False
      drag_erase = False

      running = True
      while running:
          clock.tick(60)
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  running = False
              elif event.type == pygame.KEYDOWN:
                  if self._handle_keydown(event, pygame) is False:
                      running = False
                      break
              elif event.type == pygame.MOUSEBUTTONDOWN:
                  if event.button in (1, 3) and not self.load_menu_open:
                      if event.pos[1] < grid_h:
                          dragging = True
                          drag_erase = event.button == 3
                          self._paint_at_mouse(event.pos, grid_h, tiles_visible_x, erase=drag_erase)
              elif event.type == pygame.MOUSEBUTTONUP:
                  if event.button in (1, 3):
                      dragging = False
              elif event.type == pygame.MOUSEMOTION:
                  if dragging and not self.load_menu_open and event.pos[1] < grid_h:
                      self._paint_at_mouse(event.pos, grid_h, tiles_visible_x, erase=drag_erase)

          self._draw(screen, font, font_lg, grid_h, tiles_visible_x)
          pygame.display.flip()

      pygame.quit()

  def _handle_keydown(self, event, pygame) -> bool:
      if self.load_menu_open:
          if event.key == pygame.K_ESCAPE:
              self.load_menu_open = False
              return True
          if event.key == pygame.K_UP:
              self.load_menu_index = max(0, self.load_menu_index - 1)
              return True
          if event.key == pygame.K_DOWN:
              self.load_menu_index = min(len(self._level_files) - 1, self.load_menu_index + 1)
              return True
          if event.key in (pygame.K_RETURN, pygame.K_l):
              if self._level_files:
                  self._load_file(self._level_files[self.load_menu_index])
              return True
          return True

      for f in self._fields:
          if f.handle_key(event.key, event.unicode, pygame_mod=pygame):
              self.state.mark_dirty()
              return True

      key = event.key
      if key == pygame.K_ESCAPE:
          return False
      if key == pygame.K_TAB:
          self._cycle_field()
          return True
      if key == pygame.K_1:
          self.brush = Brush.ERASE
          self.message = "Brush: erase"
      elif key == pygame.K_2:
          self.brush = Brush.BLOCK
          self.message = "Brush: block"
      elif key == pygame.K_3:
          self.brush = Brush.SPIKE
          self.message = "Brush: spike"
      elif key == pygame.K_4:
          self.brush = Brush.START
          self.message = "Brush: start"
      elif key == pygame.K_5:
          self.brush = Brush.END
          self.message = "Brush: end"
      elif key == pygame.K_LEFT or key == pygame.K_a:
          self.state.camera_tx = max(0, self.state.camera_tx - 1)
      elif key == pygame.K_RIGHT or key == pygame.K_d:
          max_cam = max(0, self.state.width - 1)
          self.state.camera_tx = min(max_cam, self.state.camera_tx + 1)
      elif key == pygame.K_s:
          self._save()
      elif key == pygame.K_l:
          self._refresh_load_menu()
          self.load_menu_open = True
          self.message = "Load menu (↑↓ Enter Esc)"
      elif key == pygame.K_n:
          self._new_level()
      elif key == pygame.K_v:
          self._sync_state_from_fields()
          errs = validate_level(self.state.to_level())
          if errs:
              self.message = "; ".join(e.message for e in errs[:3])
          else:
              self.message = "Validation OK"
      return True

  def _cycle_field(self) -> None:
      idx = next((i for i, f in enumerate(self._fields) if f.active), -1)
      for f in self._fields:
          f.active = False
      self._fields[(idx + 1) % len(self._fields)].active = True

  def _paint_at_mouse(self, pos, grid_h: int, tiles_visible_x: int, *, erase: bool) -> None:
      mx, my = pos
      if my >= grid_h:
          return
      tx = int(mx // self.TILE_SIZE) + self.state.camera_tx
      ty = int(my // self.TILE_SIZE)
      brush = Brush.ERASE if erase else self.brush
      if self.state.apply_brush(tx, ty, brush):
          self.message = ""

  def _draw(self, screen, font, font_lg, grid_h: int, tiles_visible_x: int) -> None:
      import pygame

      screen.fill((18, 18, 22))
      s = self.TILE_SIZE
      cam = self.state.camera_tx
      floor_y = self.state.floor_y()

      block_c = (210, 210, 220)
      spike_c = (220, 80, 80)
      floor_c = (60, 60, 70)
      grid_c = (35, 35, 42)
      start_c = (80, 220, 120)
      end_c = (240, 210, 80)

      for ty in range(self.state.height):
          for vx in range(tiles_visible_x):
              tx = cam + vx
              if tx >= self.state.width:
                  continue
              x = vx * s
              y = ty * s
              r = pygame.Rect(x, y, s, s)
              if ty == floor_y:
                  pygame.draw.rect(screen, floor_c, r)
              else:
                  pygame.draw.rect(screen, grid_c, r, 1)

      for (tx, ty), tt in self.state.tiles.items():
          if not (cam <= tx < cam + tiles_visible_x):
              continue
          x = (tx - cam) * s
          y = ty * s
          r = pygame.Rect(x, y, s, s)
          if tt.value == "block":
              pygame.draw.rect(screen, block_c, r)
          elif tt.value == "spike":
              pygame.draw.polygon(
                  screen,
                  spike_c,
                  [(r.left, r.bottom), (r.centerx, r.top), (r.right, r.bottom)],
              )

      for pos, col in ((self.state.start, start_c), (self.state.end, end_c)):
          if cam <= pos.x < cam + tiles_visible_x:
              x = (pos.x - cam) * s
              y = pos.y * s
              pygame.draw.rect(screen, col, pygame.Rect(x, y, s, s), 2)

      pygame.draw.line(screen, (50, 50, 60), (0, grid_h), (screen.get_width(), grid_h), 2)

      panel_y = grid_h + 8
      hints = (
          "1=erase 2=block 3=spike 4=start 5=end | LMB paint RMB erase | "
          "A/D scroll | S save L load N new V validate | Tab edit fields"
      )
      screen.blit(font.render(hints, True, (180, 180, 190)), (8, panel_y))

      y = panel_y + 22
      for f in self._fields:
          color = (255, 255, 120) if f.active else (200, 200, 210)
          txt = f"{f.label}: {f.value}{'_' if f.active else ''}"
          screen.blit(font.render(txt, True, color), (8, y))
          y += 20

      ver = self.state.meta.version
      dirty = "*" if self.state.dirty else ""
      path = self.state.loaded_path.name if self.state.loaded_path else "(unsaved)"
      info = f"Brush={self.brush.value}  v={ver}{dirty}  file={path}  cam_x={cam}"
      screen.blit(font.render(info, True, (160, 160, 170)), (8, y + 4))

      if self.message:
          screen.blit(font_lg.render(self.message, True, (240, 200, 100)), (8, grid_h - 28))

      if self.load_menu_open:
          self._draw_load_menu(screen, font, font_lg)

  def _draw_load_menu(self, screen, font, font_lg) -> None:
      import pygame

      overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
      overlay.fill((0, 0, 0, 180))
      screen.blit(overlay, (0, 0))
      screen.blit(font_lg.render("Load level (↑↓ Enter, Esc cancel)", True, (240, 240, 250)), (40, 40))
      if not self._level_files:
          screen.blit(font.render("No .json files in levels/", True, (200, 200, 200)), (40, 80))
          return
      for i, p in enumerate(self._level_files):
          col = (255, 255, 120) if i == self.load_menu_index else (220, 220, 230)
          screen.blit(font.render(p.name, True, col), (56, 80 + i * 24))
