# Implementation Progress

This file tracks what’s implemented and what you (human) have verified locally.

## Status legend

- [ ] not started
- [~] in progress
- [x] implemented
- [t] you tested locally

## Milestone 0 — Level format + IO + validation

- [x] JSON schema v1 dataclasses (`src/gdrl/levels/level_schema.py`)
- [x] Save/load JSON (`src/gdrl/levels/level_io.py`)
- [x] Validation rules (floor implicit, tiles not on floor, end right of start, etc.) (`src/gdrl/levels/validation.py`)
- [x] **You test**: `python -c "from gdrl.levels import load_level, validate_level; lvl=load_level('levels/example_level.json'); print(validate_level(lvl))"`

## Milestone 1 — Simulator core (deterministic step API)

- [x] `Action` + `State` objects (`src/gdrl/sim/simulator.py`)
- [x] Deterministic fixed `dt` step loop (`Simulator.step`)
- [x] Basic physics + collision with blocks/spikes + implicit floor (`src/gdrl/sim/physics.py`)
- [x] Episode termination + result (`EpisodeResult`)
- [x] Pygame renderer (camera + tiles + player) (`src/gdrl/sim/renderer.py`)
- [x] Human play script (`scripts/play_level.py`)
- [x] **You test** (after play script lands): `python scripts/play_level.py levels/example_level.json`

## Milestone 2 — Input providers

- [x] `InputProvider` interface (`src/gdrl/input/base.py`)
- [x] `HumanInputProvider` (`src/gdrl/input/human.py`)
- [x] `NullInputProvider` + `ScriptedInputProvider` (`src/gdrl/input/scripted.py`)
- [x] Rollout loop with any provider (`src/gdrl/experiments/rollout.py`, `scripts/rollout_level.py`)
- [x] **You test**: `python scripts/rollout_level.py levels/example_level.json`

## Milestone 3 — Dataset logging

- [x] Episode directory structure under `dataset/episodes/`
- [x] `metadata.json` + `states.csv` (`src/gdrl/logging/trajectory_logger.py`)
- [x] Optional PNG frame capture (no raw arrays in JSON)
- [x] **You test**: `python scripts/rollout_level.py levels/example_level.json --log --capture-every 4`

### Dataset layout

```
dataset/
  episodes/
    episode_000001/
      metadata.json    # episode-level summary (one file per episode)
      states.csv       # one row per simulation timestep
      frames/          # optional PNGs (only when capture_every_n_steps > 0)
        000000.png
        000004.png
```

### `metadata.json` (episode-level, written once at end)

| Field | Description |
|-------|-------------|
| `logging_version` | Logger schema version (currently `1`) |
| `episode_id` | e.g. `episode_000001` |
| `level.level_id` | From level JSON |
| `level.name` | Human-readable level name |
| `level.path` | Path used to load the level (if provided) |
| `level.split` | `train` / `val` / `test` |
| `level.tags` | List of tags from level JSON |
| `level.schema_version` | Level JSON schema version |
| `level.width`, `level.height`, `level.tile_size` | Level geometry |
| `rollout.provider` | Input provider name (`null`, `scripted`, `human`) |
| `rollout.num_steps` | Number of rows in `states.csv` |
| `rollout.capture_every_n_steps` | Frame capture interval (`0` = off) |
| `episode.done` | Whether the episode terminated |
| `episode.completed` | Reached end flag |
| `episode.died` | Died on spike / head-bonk |
| `episode.progress` | Final progress in \([0, 1]\) |
| `episode.reason` | `completed`, `death`, `timeout`, or `quit` |
| `physics.*` | `fixed_dt`, `gravity`, `jump_velocity`, `scroll_speed` used for this run |
| `camera.*` | Screen size and `player_screen_x` used for rendering |

No pixel arrays or per-step data in `metadata.json`.

### `states.csv` (one row per timestep)

| Column | Description |
|--------|-------------|
| `episode_id` | Matches directory name |
| `level_id` | From level JSON |
| `timestep` | Simulation step index (0-based) |
| `action_jump_pressed` | `1` if jump was pressed this step, else `0` |
| `player_x`, `player_y` | Player position (pixels) **before** the step is applied |
| `player_vy` | Vertical velocity before the step |
| `grounded` | `1` if grounded before the step |
| `progress` | Progress along level before the step (last row uses final progress) |
| `done` | `1` only on the **last** row when the episode ended |
| `completed` | `1` on last row if level was completed |
| `died` | `1` on last row if player died |
| `reward` | Placeholder (`0.0` for now) |
| `frame_path` | Relative path e.g. `frames/000012.png`, or empty if no frame saved |

Frames are PNG files on disk only; `frame_path` is a string reference, not embedded image data.

## Milestone 4 — Editor

- [x] Pygame level editor UI (`src/gdrl/editor/`, `scripts/run_editor.py`)
- [x] Enforce start/end (single position each), end.x > start.x, no tiles on implicit floor; save requires name + author
- [x] Load/save under `levels/`; version auto-bumps on save after loading a file
- [x] **You test**: `python scripts/run_editor.py` → edit → save → `python scripts/play_level.py levels/your_level.json`

