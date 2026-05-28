# GDRL: Geometry-Dash-like Editor + Simulator (Scaffold)

This repo contains a simplified Geometry Dash-like environment for:

- **Level editing** (grid editor, save/load JSON)
- **Playable simulation** (pygame; deterministic fixed-timestep physics)
- **Dataset logging** (per-episode metadata + states CSV + optional PNG frames)

The code is organized to be easy to extend for scripted agents, RL policies, imitation learning, evaluation, and experimentation.

## Install

Create a virtual environment (recommended), then:

```bash
pip install -r requirements.txt
```

## Run

### Level editor

```bash
python scripts/run_editor.py
```

- Left click: paint selected tile
- Right click: erase to air
- Keys: `1` air, `2` block, `3` spike, `4` start, `5` end
- `S`: save to `levels/edited_level.json`
- `L`: load from `levels/edited_level.json`
- `V`: validate and print errors

### Play a level (human)

```bash
python scripts/play_level.py levels/example_level.json
```

- `Space` / `Up`: jump
- `R`: restart
- `Esc`: quit

### Rollout a level (any InputProvider) + log trajectory

```bash
python scripts/rollout_level.py levels/example_level.json
```

Outputs episodes to:

```
dataset/episodes/episode_000001/
  metadata.json
  states.csv
  frames/000000.png ...
```

## Notes

- **Tile coordinates**: JSON uses tile coordinates; internal sim uses pixel coordinates.
- **Implicit floor**: solid floor exists at `y = height - 1` and is not stored in `tiles`.
- **Determinism**: simulation uses a fixed `dt` (default 60 Hz) independent of rendering FPS.

