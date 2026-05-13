# type:ignore
from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class LavaGapEnv(MiniGridEnv):
    """## Description

    The agent has to reach the green goal square at the opposite corner of the
    room, and must pass through a narrow gap in a vertical strip of deadly lava.
    Touching the lava terminate the episode with a zero reward. This environment
    is useful for studying safety and safe exploration.

    ## Mission Space

    Depending on the `obstacle_type` parameter:
    - `Lava`: "avoid the lava and get to the green goal square"
    - otherwise: "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of map SxS.

    - `MiniGrid-LavaGapS5-v0`
    - `MiniGrid-LavaGapS6-v0`
    - `MiniGrid-LavaGapS7-v0`
    """

    def __init__(
        self,
        size,
        obstacle_type=Lava,
        max_steps: int | None = None,
        randomize_gap: bool = True,
        randomize_agent_pos: bool = False,
        randomize_goal_pos: bool = False,
        randomize_agent_dir: bool = False,
        **kwargs,
    ):
        self.obstacle_type = obstacle_type
        self.size = size
        # New parameter to toggle randomness of the gap position
        self.randomize_gap = randomize_gap
        self.randomize_agent_pos = randomize_agent_pos
        self.randomize_goal_pos = randomize_goal_pos
        self.randomize_agent_dir = randomize_agent_dir

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _sample_position(self, positions: list[tuple[int, int]]) -> tuple[int, int]:
        if not positions:
            raise ValueError("Cannot sample a position from an empty set")
        return positions[self._rand_int(0, len(positions))]

    @staticmethod
    def _positions_left_of_gap(gap_x: int, height: int) -> list[tuple[int, int]]:
        return [(x, y) for x in range(1, gap_x) for y in range(1, height - 1)]

    @staticmethod
    def _positions_right_of_gap(
        gap_x: int, width: int, height: int
    ) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(gap_x + 1, width - 1)
            for y in range(1, height - 1)
        ]

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Determine gap position either randomly or fixed (if randomness is disabled)
        if self.randomize_gap:
            gap_x = self._rand_int(2, width - 2)
            gap_y = self._rand_int(1, height - 1)
        else:
            gap_x = width // 2
            gap_y = height // 2
        self.gap_pos = np.array((gap_x, gap_y))

        # Place the obstacle wall and create the gap
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)
        self.grid.set(*self.gap_pos, None)

        agent_candidates = self._positions_left_of_gap(gap_x, height)
        goal_candidates = self._positions_right_of_gap(gap_x, width, height)

        # Place the agent on the left side of the lava wall.
        agent_pos = (
            self._sample_position(agent_candidates)
            if self.randomize_agent_pos
            else (1, 1)
        )
        self.agent_pos = np.array(agent_pos)
        self.agent_dir = self._rand_int(0, 4) if self.randomize_agent_dir else 0

        # Place the goal on the right side so the gap remains relevant.
        goal_pos = (
            self._sample_position(goal_candidates)
            if self.randomize_goal_pos
            else (width - 2, height - 2)
        )
        self.goal_pos = np.array(goal_pos)
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


class StochasticLavaGapEnv(LavaGapEnv):
    """Lava gap environment with stochastic initial agent and goal positions."""

    def __init__(self, size: int = 10, **kwargs):
        kwargs.setdefault("randomize_agent_pos", True)
        kwargs.setdefault("randomize_goal_pos", True)
        kwargs.setdefault("randomize_agent_dir", True)
        super().__init__(size=size, **kwargs)
