from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import minigrid.core.world_object as world_object  # type: ignore
import numpy as np
from gymnasium import Env
from minigrid.core.constants import IDX_TO_OBJECT  # type: ignore
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid  # type: ignore
from minigrid.minigrid_env import MiniGridEnv  # type: ignore
from minigrid.utils.rendering import downsample  # type: ignore
from minigrid.utils.rendering import fill_coords  # type: ignore
from minigrid.utils.rendering import highlight_img  # type: ignore
from minigrid.utils.rendering import point_in_rect  # type: ignore
from minigrid.utils.rendering import point_in_triangle  # type: ignore
from minigrid.utils.rendering import rotate_fn  # type: ignore
from minigrid.wrappers import FullyObsWrapper  # type: ignore
from minigrid.wrappers import ViewSizeWrapper  # type: ignore
from numpy.typing import NDArray

from uncertain_worms.structs import (
    ActType,
    Environment,
    Heuristic,
    InitialModel,
    Observation,
    ObservationModel,
    Optional,
    RewardModel,
    State,
    TransitionModel,
)
from uncertain_worms.utils import get_log_dir


class AgentPositionWrapper(ViewSizeWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {**obs, "agent_pos": self.unwrapped.agent_pos}


class GridWrapper(AgentPositionWrapper):
    def __init__(self, env: Env, width: int, height: int):
        self.width, self.height = width, height
        super().__init__(env)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **obs,
            "agent_pos": self.unwrapped.agent_pos,
            "grid": self.env.unwrapped.grid,
        }


log = logging.getLogger(__name__)
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


# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_COLOR_MAP = {
    ObjectTypes.wall: COLOR_TO_IDX["grey"],
    ObjectTypes.goal: COLOR_TO_IDX["green"],
    ObjectTypes.key: COLOR_TO_IDX["blue"],
    ObjectTypes.ball: COLOR_TO_IDX["blue"],
    ObjectTypes.lava: COLOR_TO_IDX["blue"],
    ObjectTypes.locked_door: COLOR_TO_IDX["blue"],
    ObjectTypes.open_door: COLOR_TO_IDX["blue"],
    ObjectTypes.closed_door: COLOR_TO_IDX["blue"],
}


# Needed for parsing the state/observation
def minigrid_to_local(obj: world_object.WorldObj) -> int:
    if obj is None:
        return ObjectTypes.empty
    elif obj.type == "goal":
        return ObjectTypes.goal
    elif obj.type == "wall":
        return ObjectTypes.wall
    elif obj.type == "key":
        return ObjectTypes.key
    elif obj.type == "door" and obj.is_open:
        return ObjectTypes.open_door
    elif obj.type == "door" and (not obj.is_open) and (not obj.is_locked):
        return ObjectTypes.closed_door
    elif obj.type == "door" and (not obj.is_open) and (obj.is_locked):
        return ObjectTypes.locked_door
    elif obj.type == "box":
        return ObjectTypes.box
    elif obj.type == "ball":
        return ObjectTypes.ball
    elif obj.type == "lava":
        return ObjectTypes.lava
    else:
        raise TypeError(f"Unknown object {obj.type}")


# Needed for rendering
def local_to_minigrid(lobj: int) -> world_object.WorldObj:
    if lobj == ObjectTypes.empty or lobj == ObjectTypes.unseen:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["empty"], color_idx=0, state=0
        )
    elif lobj == ObjectTypes.goal:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["goal"], color_idx=0, state=0
        )
    elif lobj == ObjectTypes.wall:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["wall"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.wall],
            state=0,
        )
    elif lobj == ObjectTypes.key:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["key"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.key],
            state=0,
        )
    elif lobj == ObjectTypes.open_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.open_door],
            state=0,
        )
    elif lobj == ObjectTypes.closed_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.closed_door],
            state=1,
        )
    elif lobj == ObjectTypes.locked_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.locked_door],
            state=2,
        )
    elif lobj == ObjectTypes.ball:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["ball"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.ball],
            state=0,
        )
    elif lobj == ObjectTypes.lava:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["lava"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.lava],
            state=0,
        )

    return None


@dataclass
class MinigridObservation(Observation):
    """
    Represents the non-centered field of view of the agent.
    The agent is NOT in the center of the observation grid.
    Observation grids are always square-sizes (i.e. 3x3, 5x5, 7x7).
    The width and height of the observation grid are called view size.
    The agent is ALWAYS in the observation and ALWAYS at the same spot
    in the observation `image`, independent of the observation.
    The experiences are printed through the `__repr__` function.
    Args:
        `image`: field of view in front of the agent.

        `agent_pos`: agent's position in the real world. It differs from the position
                     in the observation grid.
        `agent_dir`: agent's direction in the real world. It differs from the direction
                     of the agent in the observation grid.
        `carrying`: what the agent is carrying at the moment.
    """

    image: NDArray[np.int8]
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying: Optional[int] = None

    def encode(self) -> Any:
        """Converts the state to a representation that the language model is
        expecting."""
        return self

    @staticmethod
    def decode(obj: MinigridObservation) -> MinigridObservation:
        """Converts the encoding back to the state representation."""
        return MinigridObservation(
            image=np.array(obj.image, dtype=np.int8),
            agent_pos=obj.agent_pos,
            agent_dir=obj.agent_dir,
            carrying=obj.carrying,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MinigridObservation)
            and isinstance(self.image, np.ndarray)
            and isinstance(other.image, np.ndarray)
            and self.image.shape == other.image.shape
            and np.allclose(self.image, other.image)
            and tuple(self.agent_pos) == tuple(other.agent_pos)
            and self.agent_dir == other.agent_dir
            and self.carrying == other.carrying
        )

    def __hash__(self) -> int:
        return hash(
            (self.image.tobytes(), tuple(self.agent_pos), self.agent_dir, self.carrying)
        )

    def __repr__(self) -> str:
        return str(
            MinigridState(
                self.image,
                agent_pos=self.agent_pos,
                agent_dir=self.agent_dir,
                carrying=self.carrying,
            )
        )


@dataclass
class MinigridState(State):
    grid: NDArray[np.int8]
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying: Optional[int]

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.agent_pos),
                self.agent_dir,
                self.carrying,
                self.grid.tobytes(),
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MinigridState)
            and isinstance(self.grid, np.ndarray)
            and isinstance(other.grid, np.ndarray)
            and self.grid.shape == other.grid.shape
            and np.allclose(self.grid, other.grid)
            and tuple(self.agent_pos) == tuple(other.agent_pos)
            and self.agent_dir == other.agent_dir
            and self.carrying == other.carrying
        )

    @property
    def front_pos(self) -> Tuple[int, int]:
        """Get the position of the cell that is right in front of the agent."""

        return tuple(
            (np.array(self.agent_pos) + np.array(DIR_TO_VEC[self.agent_dir])).tolist()
        )

    @property
    def width(self) -> int:
        return self.grid.shape[0]

    @property
    def height(self) -> int:
        return self.grid.shape[1]

    def get_type_indices(self, type: int) -> List[Tuple[int, int]]:
        idxs = np.where(self.grid == type)  # Returns (row_indices, col_indices)
        return list(zip(idxs[0], idxs[1]))  # Combine row and column indices

    def encode(self) -> MinigridState:
        return self

    @staticmethod
    def decode(obj: MinigridState) -> MinigridState:
        return MinigridState(
            np.array(obj.grid, dtype=np.int8),
            agent_pos=obj.agent_pos,
            agent_dir=obj.agent_dir,
            carrying=obj.carrying,
        )

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

    def __repr__(self) -> str:
        """Returns a string representation of the grid with agent position."""

        # print_agent = False
        print_state = "agent_pos={}\n".format(self.agent_pos)
        print_state += "agent_dir={}\n".format(self.agent_dir)
        print_state += "carrying={}\n".format(self.carrying)
        print_state += "grid=[\n"
        for x in range(self.width):
            row = "["
            for y in range(self.height):
                row += f"{self.grid[x, y]:2d}, "
            row += "],\n"
            print_state += row
        print_state += "]\n"
        return print_state


def minigrid_empty_observation_gen(**kwargs: Any) -> MinigridObservation:
    return MinigridObservation(
        image=np.full((3, 3), ObjectTypes.empty), agent_pos=(1, 1), agent_dir=0
    )


def minigrid_empty_state_gen(env_name: str = "MiniGrid-Empty-5x5-v0") -> MinigridState:
    env: MiniGridEnv = gym.make(env_name, agent_view_size=3, max_steps=1)
    _ = env.reset()
    grid = env.unwrapped.grid
    state = grid_to_state(
        grid=grid,
        agent_pos=(1, 1),
        agent_dir=0,
    )
    # Remove all non-walls from the grid and make them empty:
    state.grid[state.grid != ObjectTypes.wall] = ObjectTypes.empty
    return state


def goal_heuristic_gen() -> Heuristic:
    def goal_heuristic(state: MinigridState) -> float:
        if len(state.get_type_indices(ObjectTypes.goal)) == 0:
            return 0

        min_dist = min(
            [
                abs(s[0] - state.agent_pos[0]) + abs(s[1] - state.agent_pos[1])
                for s in state.get_type_indices(ObjectTypes.goal)
            ]
        )
        return min_dist / (state.width * state.height)

    return goal_heuristic


def get_empty_room(width: int, height: int) -> MinigridState:
    grid = Grid(width, height)

    # Place walls around the entire edge of the map
    for x in range(width):
        grid.set(x, 0, world_object.Wall())  # Top edge
        grid.set(x, height - 1, world_object.Wall())  # Bottom edge

    for y in range(height):
        grid.set(0, y, world_object.Wall())  # Left edge
        grid.set(width - 1, y, world_object.Wall())  # Right edge

    # Place the agent at (1, 1) with an initial direction of 0
    return grid_to_state(grid, agent_pos=(1, 1), agent_dir=0)


def minigrid_hardcoded_initial_model_gen(
    env_name: str = "MiniGrid-Empty-5x5-v0",
) -> InitialModel:
    env: MiniGridEnv = gym.make(env_name, agent_view_size=3, max_steps=1)
    env.unwrapped.see_through_walls = SEE_THROUGH_WALLS

    def minigrid_hardcoded_initial_model(_: MinigridState) -> MinigridState:
        _ = env.reset(seed=np.random.randint(10000, 20000))
        grid = env.unwrapped.grid
        return copy.deepcopy(
            grid_to_state(
                grid=grid,
                agent_pos=env.unwrapped.agent_pos,
                agent_dir=env.unwrapped.agent_dir,
            )
        )

    return minigrid_hardcoded_initial_model


def minigrid_hardcoded_reward_model_gen_unlock() -> RewardModel:
    def minigrid_hardcoded_reward_model(
        state: MinigridState, action: ActType, next_state: MinigridState
    ) -> Tuple[float, bool]:
        if np.any(next_state.grid == ObjectTypes.open_door):
            return 1.0, True
        return 0.0, False

    return minigrid_hardcoded_reward_model


def minigrid_hardcoded_reward_model_gen_memory() -> RewardModel:
    def minigrid_hardcoded_reward_model_memory(
        state: MinigridState, action: ActType, next_state: MinigridState
    ) -> Tuple[float, bool]:
        # Determine the target object by looking at the initial area.
        # In MemoryEnv, the starting room object is placed at (1, grid.height // 2 - 1)

        if state.carrying is not None:
            return 0.0, False

        target_x = 1
        target_y = state.grid.shape[1] // 2 - 1
        target_obj = state.grid[target_x, target_y]
        # Get the agent's front position in the next state.
        front_x, front_y = next_state.front_pos
        front_obj = state.grid[front_x, front_y]

        if front_obj == target_obj and not (
            tuple(next_state.front_pos) == (target_x, target_y)
        ):
            return 1.0, True

        return 0.0, False

    return minigrid_hardcoded_reward_model_memory


def minigrid_hardcoded_reward_model_gen() -> RewardModel:
    def minigrid_hardcoded_reward_model(
        state: MinigridState, action: ActType, next_state: MinigridState
    ) -> Tuple[float, bool]:
        if (
            state.grid[next_state.agent_pos[0], next_state.agent_pos[1]]
            == ObjectTypes.goal
        ):
            return 1.0, True
        if (
            state.grid[next_state.agent_pos[0], next_state.agent_pos[1]]
            == ObjectTypes.lava
        ):
            return 0.0, True
        return 0.0, False

    return minigrid_hardcoded_reward_model


def process_vis(grid: NDArray[np.int8], agent_pos: tuple[int, int]) -> NDArray[np.int8]:
    # Initialize the boolean mask.
    mask = np.zeros(grid.shape, dtype=bool)
    mask[agent_pos] = True
    left_shift = np.roll(mask, shift=-1, axis=0)
    right_shift = np.roll(mask, shift=1, axis=0)
    down_shift = np.roll(mask, shift=-1, axis=0)
    down_shift[:, 0] = mask[:, 0]  # For the first column, do not roll.
    visible = mask & (grid == None)
    mask = mask | visible | left_shift | right_shift | down_shift
    return mask.astype(np.int8)


def minigrid_hardcoded_observation_model_gen() -> ObservationModel:
    def minigrid_hardcoded_observation_model(
        state: MinigridState,
        action: int,
        empty_obs: MinigridObservation,
        agent_view_size: int = 3,
    ) -> MinigridObservation:
        grid = state.get_field_of_view(view_size=agent_view_size)

        if not SEE_THROUGH_WALLS:
            vis_mask = process_vis(
                grid.T, agent_pos=(agent_view_size // 2, agent_view_size - 1)
            ).T
            grid[~vis_mask] = ObjectTypes.empty


        return MinigridObservation(
            image=np.array(grid, dtype=np.int8),
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            carrying=state.carrying,
        )

    return minigrid_hardcoded_observation_model


def minigrid_hardcoded_transition_model_gen() -> TransitionModel:
    def minigrid_hardcoded_transition_model(
        state: MinigridState, action: Actions
    ) -> MinigridState:
        """Applies a hardcoded transition model for Minigrid."""

        state = copy.deepcopy(state)
        # Create a copy of the current state
        new_state = MinigridState(
            grid=state.grid.copy(),
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            carrying=state.carrying,
        )

        # Rotate left
        if action == Actions.left:
            new_state.agent_dir = (state.agent_dir - 1) % 4

        # Rotate right
        elif action == Actions.right:
            new_state.agent_dir = (state.agent_dir + 1) % 4

        # Move forward
        elif action == Actions.forward:
            fwd_pos = state.front_pos  # Compute forward position

            # Check if the forward position is within bounds
            if 0 <= fwd_pos[0] < state.width and 0 <= fwd_pos[1] < state.height:
                # Check if there's a wall at the new position
                if (
                    state.grid[fwd_pos] != ObjectTypes.wall
                    and state.grid[fwd_pos] != ObjectTypes.ball
                    and state.grid[fwd_pos] != ObjectTypes.key
                    and state.grid[fwd_pos] != ObjectTypes.locked_door
                    and state.grid[fwd_pos] != ObjectTypes.closed_door
                ):
                    new_state.agent_pos = fwd_pos

        elif action == Actions.pickup:
            if state.grid[state.front_pos] == ObjectTypes.key:
                new_state.carrying = ObjectTypes.key
                new_state.grid[state.front_pos] = ObjectTypes.empty
            if state.grid[state.front_pos] == ObjectTypes.ball:
                new_state.carrying = ObjectTypes.ball
                new_state.grid[state.front_pos] = ObjectTypes.empty
        elif action == Actions.toggle:
            if (
                state.grid[state.front_pos] == ObjectTypes.locked_door
                and state.carrying == ObjectTypes.key
            ):
                new_state.grid[state.front_pos] = ObjectTypes.open_door
        elif action == Actions.drop:
            if (
                state.carrying is not None
                and state.grid[state.front_pos] == ObjectTypes.empty
            ):
                new_state.grid[state.front_pos] = state.carrying
                new_state.carrying = None
        else:
            raise ValueError(f"Unknown action: {action}")

        return new_state

    return minigrid_hardcoded_transition_model


def grid_to_state(
    grid: Grid,
    agent_pos: Tuple[int, int],
    agent_dir: int,
    carrying: world_object.WorldObj | None = None,
) -> MinigridState:
    grid_array = np.zeros((grid.width, grid.height), dtype=np.int8)
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            grid_array[x, y] = minigrid_to_local(obj)
    if carrying is not None:
        carrying = ObjectTypes[carrying.type]
    return MinigridState(
        np.array(grid_array, dtype=np.int8), agent_pos, agent_dir, carrying
    )


def build_state(
    obs_dict: Dict,
    carrying: world_object.WorldObj | None,
) -> MinigridState:
    return grid_to_state(
        grid=obs_dict["grid"],
        agent_pos=obs_dict["agent_pos"],
        agent_dir=obs_dict["direction"],
        carrying=carrying,
    )


def build_obs(obs_dict: Dict, fully: bool = True) -> MinigridObservation:
    W, H, _ = obs_dict["image"].shape

    obs_array = np.zeros((W, H), dtype=np.int8)
    carrying = None
    for i in range(W):
        for j in range(H):
            if i == W // 2 and j == H - 1 and not fully:
                # Agent location
                obs_array[i, j] = ObjectTypes.agent
                if IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] != "empty":
                    carrying = minigrid_to_local(
                        world_object.WorldObj.decode(
                            *(obs_dict["image"][i, j, :].tolist())
                        )
                    )
            elif fully and list(obs_dict["agent_pos"]) == [i, j]:
                if IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] != "agent":
                    carrying = minigrid_to_local(
                        world_object.WorldObj.decode(
                            *(obs_dict["image"][i, j, :].tolist())
                        )
                    )
                obs_array[i, j] = ObjectTypes.empty
            elif IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] == "empty":
                obs_array[i, j] = ObjectTypes.empty
            else:
                wobj = world_object.WorldObj.decode(
                    *(obs_dict["image"][i, j, :].tolist())
                )
                obs_array[i, j] = int(minigrid_to_local(wobj))

    return MinigridObservation(
        image=obs_array,
        agent_pos=obs_dict["agent_pos"],
        agent_dir=obs_dict["direction"],
        carrying=carrying,
    )


def create_gif(
    width: int,
    height: int,
    observations: List[MinigridObservation],
    gif_filename: str = "output.gif",
    duration: float = 100.0,
    fully_obs: bool = True,
) -> None:
    """Create a GIF showing the change in color gradients over time.

    Parameters:
    probabilities_over_time (numpy array): A N x H x W x C  array of probabilities over N steps.
    class_to_color (dict): A dictionary mapping class indices to RGB colors.
    gif_filename (str): The name of the output GIF file.
    duration (float): Duration of each frame in the GIF.

    Returns:
    None
    """
    current_grid = PartialGrid(np.zeros((width, height)))
    partial_grid_trajectory = []
    for observation in observations:
        if fully_obs:
            current_grid = update_fo_grid(current_grid, observation)
        else:
            current_grid = update_po_grid(current_grid, observation)
        partial_grid_trajectory.append(current_grid)

    frames = []
    for step in range(len(partial_grid_trajectory)):
        color_grid = render_partial_grid(
            partial_grid_trajectory[step], fully_obs=fully_obs
        )

        # Convert color grid to image and store in memory
        buffer = BytesIO()
        plt.imshow(color_grid)
        plt.title(f"Step {step + 1}")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Append the image to the frame list
        frames.append(imageio.v2.imread(buffer))
        buffer.close()

    # Save frames as a GIF
    imageio.v2.mimwrite(gif_filename, frames, format="GIF", duration=duration)  # type: ignore
    print(f"GIF saved as {gif_filename}")


# -------- Belief related functions
@dataclass
class PartialGrid:
    grid: NDArray
    agent_pos: Tuple[int, int] = (0, 0)
    agent_dir: int = 0
    in_view: List[Tuple[int, int]] = field(default_factory=list)
    has_viewed: List[Tuple[int, int]] = field(default_factory=list)


def get_view_exts(
    agent_pos: Tuple[int, int], agent_dir: int, agent_view_size: int
) -> Tuple[int, int, int, int]:
    """
    Get the extents of the square set of tiles visible to the agent
    Note: the bottom extent indices are not included in the set
    if agent_view_size is None, use self.agent_view_size
    """

    # Facing right
    if agent_dir == 0:
        topX = agent_pos[0]
        topY = agent_pos[1] - agent_view_size // 2
    # Facing down
    elif agent_dir == 1:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1]
    # Facing left
    elif agent_dir == 2:
        topX = agent_pos[0] - agent_view_size + 1
        topY = agent_pos[1] - agent_view_size // 2
    # Facing up
    elif agent_dir == 3:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1] - agent_view_size + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + agent_view_size
    botY = topY + agent_view_size

    return topX, topY, botX, botY


def get_right_vector(agent_dir_vec: Tuple[int, int]) -> Tuple[int, int]:
    """Returns vector pointing to the right of the agent."""
    dx, dy = agent_dir_vec
    return np.array((-dy, dx)).tolist()


def update_fo_grid(partial_grid: PartialGrid, obs: MinigridObservation) -> PartialGrid:
    new_partial_grid = copy.deepcopy(partial_grid)
    new_partial_grid.agent_pos = obs.agent_pos
    new_partial_grid.agent_dir = obs.agent_dir
    new_partial_grid.grid[obs.image != ObjectTypes.unseen] = obs.image[
        obs.image != ObjectTypes.unseen
    ]
    return new_partial_grid


def update_po_grid(partial_grid: PartialGrid, obs: MinigridObservation) -> PartialGrid:
    new_partial_grid = copy.deepcopy(partial_grid)
    new_partial_grid.agent_pos = obs.agent_pos
    new_partial_grid.agent_dir = obs.agent_dir

    agent_dir = DIR_TO_VEC[obs.agent_dir]
    partially_observable_view = obs.image
    view_size_x, view_size_y = partially_observable_view.shape[0:2]
    f_vec = agent_dir
    r_vec = np.array(get_right_vector(tuple(f_vec.tolist())))

    top_left = (
        new_partial_grid.agent_pos
        + f_vec * (view_size_x - 1)
        - r_vec * (view_size_y // 2)
    )

    new_partial_grid.in_view = []
    for i in range(0, view_size_x):
        for j in range(0, view_size_y):
            obj_idx = partially_observable_view[i, j]
            abs_i, abs_j = top_left - (f_vec * j) + (r_vec * i)
            new_partial_grid.has_viewed.append((abs_i, abs_j))

            if abs_i < 0 or abs_i >= new_partial_grid.grid.shape[0]:
                continue
            if abs_j < 0 or abs_j >= new_partial_grid.grid.shape[1]:
                continue

            new_partial_grid.in_view.append((abs_i, abs_j))

            if obj_idx == ObjectTypes.unseen:
                continue

            if obj_idx == ObjectTypes.agent:
                continue

            new_partial_grid.grid[abs_i, abs_j] = obj_idx

    return new_partial_grid


def render_tile(
    obj: int,
    agent_dir: int | None = None,
    highlight: bool = False,
    tile_size: int = 32,
    subdivs: int = 3,
    unseen: bool = False,
) -> NDArray[np.uint8]:
    # Hash map lookup key for the cache
    key: tuple[Any, ...] = (obj, agent_dir, highlight, tile_size, unseen)

    if key in Grid.tile_cache:
        return Grid.tile_cache[key]

    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    if obj == ObjectTypes.unseen and unseen:
        super_size = tile_size * subdivs
        stripe_thickness = subdivs * 2  # ~1px in final image
        stripe_spacing = 8 * subdivs  # e.g. every 8 final pixels
        I, J = np.indices((super_size, super_size))
        mask = ((I - J) % stripe_spacing) < stripe_thickness
        img[mask] = np.array([0, 0, 255], dtype=img.dtype)

    # Draw the grid lines (top and left edges)
    fill_coords(img, world_object.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    wobj = local_to_minigrid(obj)
    if wobj is not None:
        wobj.render(img)

    # Overlay the agent on top
    if agent_dir is not None:
        assert (
            wobj is None
            or obj == ObjectTypes.goal
            or obj == ObjectTypes.open_door
            or obj == ObjectTypes.lava
        ), f"Agent direction should only be set for goal objects, not {obj}"
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        fill_coords(img, tri_fn, (255, 0, 0))

    # Highlight the cell if needed
    if highlight:
        highlight_img(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)
    # Cache the rendered tile
    Grid.tile_cache[key] = img

    return img


def render_partial_grid(
    partial_grid: PartialGrid, tile_size: int = 32, fully_obs: bool = False
) -> NDArray[np.uint8]:
    """Create a color gradient based on class probabilities.

    Parameters:
    probabilities (numpy array): A H x W x C array of class probabilities.
    class_to_color (dict): A dictionary mapping class indices to RGB colors.

    Returns:
    numpy array: A H x W x 3 array of colors representing the gradient.
    """

    W, H = partial_grid.grid.shape[0], partial_grid.grid.shape[1]

    # Compute the total grid size
    width_px = W * tile_size
    height_px = H * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for i in range(0, W):
        for j in range(0, H):
            agent_here = np.array_equal(partial_grid.agent_pos, (i, j))
            if (i, j) in partial_grid.in_view:
                highlight = True
            else:
                highlight = False

            tile_img = render_tile(
                partial_grid.grid[i, j],
                agent_dir=partial_grid.agent_dir if agent_here else None,
                highlight=highlight,
                tile_size=tile_size,
                unseen=(not ((i, j) in partial_grid.has_viewed)) and (not fully_obs),
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img


class MinigridEnvironment(Environment):
    def __init__(
        self,
        *args: Any,
        env_name: str = "MiniGrid-Empty-5x5-v0",
        fully_obs: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.env: MiniGridEnv = gym.make(
            env_name, agent_view_size=3, max_steps=self.max_steps
        )
        self.env.unwrapped.see_through_walls = SEE_THROUGH_WALLS
        self.fully_obs = fully_obs
        self.width, self.height = self.env.width, self.env.height
        self.env = AgentPositionWrapper(self.env)
        if fully_obs:
            self.env = FullyObsWrapper(self.env)
        self.env = GridWrapper(self.env, width=self.width, height=self.height)

    def step(
        self, action: ActType
    ) -> Tuple[Observation, State, float, bool, bool, Dict[str, Any]]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        next_obs = build_obs(obs_dict, fully=self.fully_obs)
        next_state = build_state(obs_dict, carrying=self.env.unwrapped.carrying)
        return next_obs, next_state, int(reward > 0.0), terminated, truncated, info

    def reset(self, seed: int = 0) -> State:
        obs_dict, _ = self.env.reset(seed=seed)
        initial_obs = build_obs(obs_dict, fully=self.fully_obs)
        intial_state = build_state(obs_dict, carrying=self.env.unwrapped.carrying)
        return intial_state

    def visualize_episode(
        self,
        _: List[MinigridState],
        observations: List[MinigridObservation],
        actions: List[int],
        episode_num: int,
    ) -> None:
        """An environment speciic belief trajectory visualization function."""
        filename = os.path.join(get_log_dir(), f"episode{episode_num}_output.gif")
        create_gif(
            self.width, self.height, observations, filename, fully_obs=self.fully_obs
        )
