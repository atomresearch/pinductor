"""Environment-agnostic state sampler for Minigrid.

Given a reference grid, samples a random valid MinigridState
(agent placed on an empty cell, random direction).
"""
import random
from typing import Optional

import numpy as np

from uncertain_worms.environments.minigrid.minigrid_env import (
    MinigridState,
    ObjectTypes,
)


def make_minigrid_state_sampler(reference_state: MinigridState):
    """Return a callable that samples random MinigridStates.

    The grid layout is taken from *reference_state* (walls, doors, etc.
    stay fixed).  The agent is placed on a uniformly random empty cell
    with a uniformly random direction.

    Usage::

        sampler = make_minigrid_state_sampler(empty_state)
        random_state = sampler()
    """
    grid = reference_state.grid
    # Pre-compute walkable positions (empty, goal, open_door — anything not wall/lava)
    walkable = set()
    blocked = {int(ObjectTypes.wall), int(ObjectTypes.lava)}
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if int(grid[x, y]) not in blocked:
                walkable.add((x, y))
    walkable = list(walkable)
    if not walkable:
        raise ValueError("No walkable cells in reference grid")

    def _sample() -> MinigridState:
        pos = random.choice(walkable)
        direction = random.randint(0, 3)
        return MinigridState(
            grid=grid.copy(),
            agent_pos=pos,
            agent_dir=direction,
            carrying=None,
        )

    return _sample
