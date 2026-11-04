# Provenance

Vendored from https://github.com/NM512/dreamerv3-torch at commit `6ef8646`.
Upstream `LICENSE` (MIT, NM512) and `README.md` are the original files.

## What we added

- `envs/uw.py` — bridge from this paper's environments to the DreamerV3 env
  API. Exposes `UWEnv` with the `reset`/`step`/`observation_space`/
  `action_space` this learner expects and converts our observations into its
  dict format. It adds a dummy 8x8 image the encoder ignores, because the world
  model's preprocessing divides an image by 255 whether or not one is used.
- `envs/uw_seed_demos.py` — the same bridge with the replay buffer pre-loaded
  from our demonstrations, which is the demonstration-seeded arm.
- `campaign_offline.sh`, `campaign_ms_seeded.sh`, `campaign_completion.sh` —
  the run campaigns behind the reported numbers.

## What we changed upstream

`configs.yaml` and `dreamer.py`, 94 lines in total, kept as `OUR_CHANGES.diff`.
Apply that patch to a fresh clone of `6ef8646` and you have this directory.

## Why the bridge encodes observations itself

`envs/uw.py` carries its own three observation encoders rather than importing
`uncertain_worms.environments.obs_encoding`, which the other baselines share.
That is deliberate: for MineSweeper it talks to `popgym.envs.minesweeper`
directly rather than to this repository's PopGym wrapper, so its input is a
Discrete scalar where the shared encoder expects the already-flattened vector
our wrapper emits. Unifying them would mean changing what the bridge talks to,
which would change the arm rather than tidy it.

## Dependencies

`pip install -e '.[dreamer-torch]'`. Upstream's `requirements.txt` also pins mujoco,
dm_control, memory_maze, crafter and opencv for its own benchmark suites; none
of those are reachable from these bridges, so the extra lists only what
`dreamer.py` and the two bridges import.

Verified in a clean virtual environment: `pip install -e '.[dreamer-torch]'` succeeds
and the bridge builds all nine environments with the right observation widths
and action counts, and steps on MineSweeper and Blocks.
