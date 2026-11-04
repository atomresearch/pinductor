# Provenance

Vendored from https://github.com/danijar/dreamerv3 at commit `e3f0224`. The
upstream `LICENSE` (MIT, Danijar Hafner) and `README.md` are the original files,
and `OUR_CHANGES.diff` is empty: nothing upstream is modified.

This is the vanilla online arm, learning from scratch with no demonstrations.
The demonstration-seeded arm is the sibling `dreamerv3-torch/`.

## What we added

Three bridges under `dreamerv3/`, none of which touch upstream code:

- `uw_envs.py` — this paper's environments behind the legacy gym API that
  `embodied.FromGym` expects. Registers `UWBlocks-v0` and `UWMiniGrid-v0`;
  the environment and step cap are read from `UW_ENV_NAME` and `UW_MAX_STEPS`.
- `uw_popgym.py` — the same for PopGym MineSweeper, which is gymnasium rather
  than gym.
- `mgflat.py` — flattens the MiniGrid observation to the vector the world
  model's MLP head consumes.

## Running it

Install `.[dreamer-jax]` first — this arm needs gym 0.25, the PyTorch one 0.22.

```bash
UW_ENV_NAME=MyMiniGrid-LavaWall-v0 UW_MAX_STEPS=40 UW_TRAIN_SEED=0 \
python dreamerv3/main.py --configs size1m \
    --task gym_uw_envs:UWMiniGrid-v0 \
    --run.steps 1000000 --jax.platform cpu \
    --batch_length 16 --report_length 8
```

The paper's runs used `size1m` on CPU. The queueing scripts that sequenced them
overnight are not shipped: they carried absolute paths to one machine and a
`pgrep` wait loop, neither of which is part of the method.
