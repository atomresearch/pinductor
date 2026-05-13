# type: ignore
from __future__ import annotations

import random
from typing import Any, Tuple

from minigrid.core.grid import Grid  # type:ignore
from minigrid.core.mission import MissionSpace  # type:ignore
from minigrid.core.world_object import Goal  # type:ignore
from minigrid.minigrid_env import MiniGridEnv  # type:ignore


class CornerGoalRandomEmptyEnv(MiniGridEnv):
    """## Description This environment is an empty room where the agent must
    reach one of the green goal squares. Goals are placed at all corners except
    the one where the agent starts.

    ## Mission Space
    "get to any of the green goal squares"

    ## Rewards
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination
    The episode ends if any one of the following conditions is met:
    1. The agent reaches any goal.
    2. Timeout (see `max_steps`).
    """

    def __init__(
        self,
        size: int = 8,
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "get to any of the green goal squares"

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Define all corner positions
        corners = [
            (1, 1),
            (1, height - 2),
            (width - 2, 1),
            (width - 2, height - 2),
        ]

        # Place goal squares in all corners except the agent start position
        x, y = random.choice(corners)
        self.put_obj(Goal(), x, y)

        while True:
            # Place the agent
            self.agent_pos = (
                random.choice(list(range(1, height - 1))),
                random.choice(list(range(1, height - 1))),
            )
            if (self.agent_pos) not in corners:
                break

        self.agent_dir = random.choice([0, 1, 2, 3])

        self.mission = "get to any of the green goal squares"

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = 1
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
