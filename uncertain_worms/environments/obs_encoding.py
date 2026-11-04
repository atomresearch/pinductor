"""Observations to flat float vectors, one encoder per suite.

Neural baselines need a fixed-width vector; our environments emit structured
observations. These three functions are the whole of that conversion, and they
are shared rather than duplicated: the DreamerV3 bridge under
`baselines/dreamerv3-torch/envs/uw.py` grew its own copies first, which is how
two baselines ended up encoding the same observation two different ways.

Nothing here is learned or tuned. It is a layout choice, and it has to stay
identical across baselines for the comparison between them to mean anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_blocks(env: Any) -> np.ndarray:
    """Blocks: the stack at the gripper's column plus the held block."""
    return np.asarray(env.get_last_obs().image, np.float32)


def encode_minigrid(env: Any) -> np.ndarray:
    """MiniGrid: the egocentric view, the facing direction, what is carried."""
    from minigrid.core.constants import OBJECT_TO_IDX

    o = env.get_last_obs()
    carried = o.carrying
    carry = 0.0
    if carried is not None:
        carry = float(OBJECT_TO_IDX.get(
            getattr(carried, "type", str(carried)), 1))
    return np.concatenate([
        np.asarray(o.image, np.float32).ravel(),
        [float(o.agent_dir), carry],
    ]).astype(np.float32)


def encode_popgym(env: Any) -> np.ndarray:
    """PopGym: our wrappers already deliver a flat binary vector.

    Not a one-hot of a Discrete index, which is what a bare PopGym environment
    emits. `PopGymEnvironment` applies the Flatten and Markovian wrappers, so
    the observation space is a Box, and for MineSweeperEasy the 13 entries are
    the one-hot adjacent-mine count, the one-hot previous action row and
    column, and two antialias bits.
    """
    return np.asarray(env.get_last_obs().value[2], np.float32)


def encoder_for(env: Any):
    """The encoder matching this environment, and the width it produces."""
    name = type(env).__name__
    if name == "BlocksEnvironment":
        fn = encode_blocks
    elif name == "MinigridEnvironment":
        fn = encode_minigrid
    elif name == "PopGymEnvironment":
        fn = encode_popgym
    else:
        raise ValueError(
            f"no observation encoder registered for {name}. Add one here rather "
            f"than encoding inline in a baseline, so every baseline sees the "
            f"same vector.")
    env.reset(seed=0)
    return fn, int(fn(env).shape[0])
