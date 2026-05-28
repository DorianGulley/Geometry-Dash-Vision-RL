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
- [t] **You test** (after play script lands): `python scripts/play_level.py levels/example_level.json`

## Milestone 2 — Input providers

- [x] `InputProvider` interface (`src/gdrl/input/base.py`)
- [x] `HumanInputProvider` (`src/gdrl/input/human.py`)
- [x] `NullInputProvider` + `ScriptedInputProvider` (`src/gdrl/input/scripted.py`)
- [ ] Rollout loop with any provider
- [t] **You test**: `python scripts/rollout_level.py levels/example_level.json`

## Milestone 3 — Dataset logging

- [ ] Episode directory structure under `dataset/episodes/`
- [ ] `metadata.json` + `states.csv`
- [ ] Optional PNG frame capture (no raw arrays in JSON)
- [t] **You test**: verify files exist and open correctly after rollout

## Milestone 4 — Editor

- [ ] Pygame level editor UI
- [ ] Enforce exactly one start/end, end right of start, don’t overwrite implicit floor
- [t] **You test**: create/save/load a level and play it

