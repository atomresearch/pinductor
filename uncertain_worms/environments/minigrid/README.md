# `uncertain_worms/environments/minigrid/` — Custom MiniGrid POMDPs

The four base environments evaluated in the paper (Corners, Lava, Four
Rooms, Unlock) plus their stochastic variants. Built on top of the public
`minigrid` package (v2.3.1) but wrapped behind the project's `Environment`
interface so the same agent code drives both the Pinductor and the baseline
pipelines.

## Layout

```
minigrid/
├── __init__.py
├── api.py                # Environment factory + Gymnasium registration
├── minigrid_env.py       # Core wrapper: partial-FOV observations, soft distance kernel
├── sample_state.py       # Belief-state sampling helpers used by the agent
├── collect_demos.py      # Interactive demo collection (pygame UI)
├── collect_demos_random_agent.py  # Optional: random-policy demo seeding
├── model_templates/      # Plain-text Hydra-injectable initial/observation/transition templates
├── custom_environments/  # The actual env classes (one file per env family)
└── trajectory_data/      # Pre-collected offline demonstrations (.pkl) — see its own README
```

## Environments registered (paper-grade)

| Gym id | File | Notes |
| --- | --- | --- |
| `CornerGoalRandom-Empty-10x10-v0` | `custom_environments/corners_large_random.py` | Goal in a random corner (paper §5 §App. G.1) |
| `MyMiniGrid-LavaWall-v0` | `custom_environments/lavagap.py` | Lava wall with one gap; deterministic by default |
| `MyMiniGrid-StochasticLavaWall-v0` | same | Stochastic variant (agent start + goal vary) |
| `MyMiniGrid-FourRooms-v0` | `custom_environments/four_rooms.py` | 4 rooms connected by 4 narrow doorways |
| `MyMiniGrid-StochasticFourRooms-v0` | same | Stochastic variant |
| `MyUnlockEnv-v0` | `custom_environments/unlock.py` | Locked door + matching key |
| `MyMiniGrid-StochasticUnlock-v0` | same | Stochastic variant |

The natural-language descriptions injected into LLM prompts for each env
live in `env_descriptions.txt` at the repository root (level L3 of the
prompt-information sweep, paper Fig. 7).

## Observation model

`MinigridEnvironment` returns a `MinigridObservation`
(`uncertain_worms/structs.py`) at every step:

* `grid` — 3×3 partial field of view in front of the agent (paper §3, §5).
* `direction` — agent's facing direction (0–3).
* `carrying` — index of the object currently held, if any.

The soft distance used by the particle filter (paper Eq. 7) is implemented
on this dataclass via `MinigridObservation.distance_soft`. Optional Numba
acceleration is loaded lazily through `particle_filtering.field_weights`.

## Adding a new MiniGrid environment

1. Create a class inheriting from `minigrid.MiniGridEnv` under
   `custom_environments/` (use `four_rooms.py` as a template).
2. Register it in `custom_environments/__init__.py` with
   `gym.register(...)`.
3. Add an L3 description block to `env_descriptions.txt`.
4. Collect demonstrations via `python -m
   uncertain_worms.environments.minigrid.collect_demos --env <new_id>`
   (or use the random-policy script).
5. Slice them into `_paper_N{N}.pkl` files with
   `scripts/paper/regenerate_datasets.py`.
6. Add `scripts/paper/configs/<cond>/<env>.yaml` files (one per condition)
   by editing `scripts/paper/build_configs.py` to include the new env in
   its environment list, then running it.
