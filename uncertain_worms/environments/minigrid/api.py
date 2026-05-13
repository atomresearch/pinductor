# type: ignore
# ─────────────────────────────────────────────────────────────────────
# LLM context template — NOT an importable module.
#
# This file is read as plain text and injected into the LLM prompt so
# the model understands the Minigrid state/observation API.  It is
# never imported by Python code.  The authoritative implementations
# live in minigrid_env.py.
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

SEE_THROUGH_WALLS = True


class ObjectTypes(IntEnum):
    unseen = 0
    empty = 1
    wall = 2
    open_door = 4
    closed_door = 5
    locked_door = 6
    key = 7
    ball = 8
    box = 9
    goal = 10
    lava = 11
    agent = 12


class Direction(IntEnum):
    facing_right = 0
    facing_down = 1
    facing_left = 2
    facing_up = 3


class Actions(IntEnum):
    left = 0  # Turn left
    right = 1  # Turn right
    forward = 2  # Move forward
    pickup = 3  # Pick up an object
    drop = 4  # Drop an object
    toggle = 5  # Toggle/activate an object
    done = 6  # Done completing the task


@dataclass
class MinigridObservation(Observation):
    """POMDP-clean observation — three perceptual fields the agent reads
    each step.

    Args:
        `image`: 3x3 field of view in front of the agent.
        `agent_dir`: compass — facing direction
                     (0=right, 1=down, 2=left, 3=up). Proprioception.
        `carrying`: object the agent holds (`None` if empty).
                    Proprioception.

    ``reward`` and ``terminated`` are NOT fields of the observation:
    they are *outcomes of the transition* computed by ``reward_func``,
    not perceptual signals that ``obs_func`` should be asked to
    reproduce. The agent still sees them per step (they appear in the
    OBSERVED DATA block alongside each obs), but ``obs_func`` should
    fill only ``image`` (and may set ``agent_dir`` / ``carrying`` if
    the underlying state carries them).

    Absolute position and the full grid are NOT part of the observation
    — the agent never sees them. Any code that reads or writes
    ``obs.agent_pos`` will crash.
    """

    image: NDArray[np.int8]
    agent_dir: int = 0
    carrying: Optional[int] = None


@dataclass
class MinigridState(State):
    """An agent exists in an indoor multi-room environment represented by a
    grid."""

    grid: NDArray[np.int8]  # Indexed as grid[x, y] where x=column, y=row
    agent_pos: Tuple[int, int]  # (x, y) position in the grid
    agent_dir: int  # 0=right, 1=down, 2=left, 3=up
    carrying: Optional[int]  # ObjectType being carried, or None

    @property
    def front_pos(self) -> Tuple[int, int]:
        """Get the (x, y) position of the cell directly in front of the agent."""
        return (
            np.array(self.agent_pos) + np.array(DIR_TO_VEC[self.agent_dir])
        ).tolist()

    @property
    def width(self) -> int:
        return self.grid.shape[0]  # grid[x, y]: shape[0] = width (x-axis)

    @property
    def height(self) -> int:
        return self.grid.shape[1]  # grid[x, y]: shape[1] = height (y-axis)

    def get_type_indices(self, obj_type: int) -> List[Tuple[int, int]]:
        """Return list of (x, y) positions where grid contains obj_type."""
        idxs = np.where(self.grid == obj_type)
        return list(zip(idxs[0], idxs[1]))  # List of (x, y) tuples


    def get_field_of_view(self, view_size: int) -> NDArray[np.int8]:
        """Returns the field of view in front of the agent.

        DO NOT modify this function.
        """

        # Get the extents of the square set of tiles visible to the agent
        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - view_size + 1
            topY = self.agent_pos[1] - view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - view_size // 2
            topY = self.agent_pos[1] - view_size + 1
        else:
            assert False, "invalid agent direction"

        fov = np.full((view_size, view_size), ObjectTypes.wall, dtype=self.grid.dtype)

        # Compute the overlapping region in the grid.
        gx0 = max(topX, 0)
        gy0 = max(topY, 0)
        gx1 = min(topX + view_size, self.grid.shape[0])
        gy1 = min(topY + view_size, self.grid.shape[1])

        # Determine where the overlapping region goes in the padded array.
        px0 = max(0, -topX)
        py0 = max(0, -topY)

        # Copy the overlapping slice.
        fov[px0 : px0 + (gx1 - gx0), py0 : py0 + (gy1 - gy0)] = self.grid[
            gx0:gx1, gy0:gy1
        ]

        for _ in range(self.agent_dir + 1):
            # Rotate left
            fov = np.rot90(fov.T, k=1).T

        agent_pos = (fov.shape[0] // 2, fov.shape[1] - 1)
        fov[agent_pos] = ObjectTypes.agent

        return fov
