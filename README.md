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

- Left click: paint selected brush; right click: erase tile
- Keys: `1` erase, `2` block, `3` spike, `4` start, `5` end
- `Tab`: cycle Name / Author / Level ID fields (type to edit)
- `A` / `D`: scroll camera
- `S`: save to `levels/<level_id>.json` (requires name, author, valid start/end)
- `L`: load menu (pick a file from `levels/`)
- `N`: new level
- `V`: validate

### Play a level (human)

```bash
python scripts/play_level.py levels/example_level.json
```

- `Space` / `Up`: jump
- `R`: restart
- `Esc`: quit

### Rollout a level (any InputProvider) + log trajectory

```bash
python scripts/rollout_level.py levels/example_level.json --log --capture-every 4
```

See `PROGRESS.md` (Milestone 3) for `metadata.json` and `states.csv` field definitions.

Outputs episodes to:

```
dataset/episodes/episode_000001/
  metadata.json
  states.csv
  frames/000000.png ...
```

### Gymnasium-style environment

```python
from gdrl.envs import GDRLEnv, ObservationConfig

env = GDRLEnv("levels/example_level.json", observation_config=ObservationConfig(width=96, height=96))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)  # 1 = jump, 0 = no jump
```

Observations are one-frame `uint8` grayscale image tensors shaped `(1, H, W)`.
Rewards use simple progress shaping plus terminal bonuses and penalties.

For quick training smoke tests, use a deterministic generated level:

```python
from gdrl.envs import make_simple_env
from gdrl.levels import make_tiny_spike_level

env = make_simple_env(seed=0)
level = make_tiny_spike_level()
```

For a no-jump smoke run:

```python
from gdrl.envs import make_simple_env, run_episode

stats = run_episode(make_simple_env(seed=0), max_steps=200)
```

Create the tiny one-frame CNN policy:

```python
from gdrl.training import TinyJumpCNN

model = TinyJumpCNN()
```

Train the CNN directly with a minimal REINFORCE loop:

```bash
python scripts/train_reinforce.py levels/tiny_spikes.json --episodes 200
```

Evaluate a saved policy checkpoint:

```bash
python scripts/eval_policy.py checkpoints/tiny_reinforce.pt --level levels/tiny_spikes.json
```

Evaluate the random baseline on the starter curriculum:

```bash
python scripts/eval_random.py --episodes 20
```

## Notes

- **Tile coordinates**: JSON uses tile coordinates; internal sim uses pixel coordinates.
- **Implicit floor**: solid floor exists at `y = height - 1` and is not stored in `tiles`.
- **Determinism**: simulation uses a fixed `dt` (default 60 Hz) independent of rendering FPS.
