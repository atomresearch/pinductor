# `trajectory_data/` — Offline demonstration buffers

Pre-collected offline demonstrations used by both Pinductor and the POMDP
Coder baseline. The same files are mirrored under
`curtis_baseline/uncertain_worms/environments/minigrid/trajectory_data/`
so each codebase can read demos relative to its own working directory.

## File naming

```
<gym_id>_paper_N<N>.pkl
<gym_id>_pure_mixed.pkl
```

* `*_paper_N<N>.pkl` — a slice of `N` trajectories used by the offline
  sample-efficiency sweep (paper Fig. 4 / experiment `E2_offline`). Valid
  `N` values are `{1, 2, 3, 4, 5, 6, 8, 10, 12}`; `N = 10` is the default
  used by `E1`.
* `*_pure_mixed.pkl` — the unsliced source buffer (≈ 20–50 trajectories,
  mixing successes, failures and truncations; see App. E.2 of the paper).
  Consumed by `scripts/paper/regenerate_datasets.py`.

## Pickle format

Each file is a pickled
`uncertain_worms.structs.ReplayBuffer[MinigridState, int,
MinigridObservation]`. A buffer's `episodes` attribute is a list of
`Episode` objects with the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `previous_states` | `List[MinigridState]` | Ground-truth states (only read by the oracle baseline; Pinductor never accesses them) |
| `previous_observations` | `List[MinigridObservation]` | Agent observations at each step |
| `actions` | `List[int]` | Actions taken |
| `next_states` | `List[MinigridState]` | Ground-truth next states |
| `next_observations` | `List[MinigridObservation]` | Resulting observations |
| `rewards` | `List[float]` | Immediate rewards |
| `terminated` | `List[bool]` | Episode-end flag for each step |
| `length` | `int` | Episode length (≤ horizon) |

A short inspection (run from the repository root):

```python
import pickle
buf = pickle.load(open("uncertain_worms/environments/minigrid/trajectory_data/MyMiniGrid-LavaWall-v0_paper_N10.pkl", "rb"))
print("episodes:", len(buf.episodes))
ep = buf.episodes[0]
print("horizon:", len(ep.actions),
      "terminal_reward:", ep.rewards[-1],
      "terminated:", ep.terminated[-1])
```

## Regeneration

`scripts/paper/regenerate_datasets.py` re-slices the `*_paper_N<N>.pkl`
files from the corresponding `*_pure_mixed.pkl` sources, mirroring the
output into both `uncertain_worms/...` and `curtis_baseline/...`. The
script is idempotent (skips slices whose pickle checksum is already
correct).

To **add** a new `N` value, edit `_VALID_N_DEMOS` in
`scripts/paper/hyperparams.py` and re-run `regenerate_datasets.py`.

## Why two locations?

The two `uncertain_worms` namespaces (top-level vs. `curtis_baseline/`)
hold structurally different `MinigridState` / `MinigridObservation`
dataclasses. Pickle resolves classes by their qualified module name, so
each codebase needs its own pickled copy. The sliced files share the
trajectory **content** but are deserialised into the local dataclass
flavour.
