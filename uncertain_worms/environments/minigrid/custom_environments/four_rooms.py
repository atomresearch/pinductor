# type:ignore
from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

ROOM_MAPPING = {
    "top_left": (0, 0),
    "top_right": (1, 0),
    "bottom_left": (0, 1),
    "bottom_right": (1, 1),
}


class MyFourRoomsEnv(MiniGridEnv):
    """Classic four room RL environment.

    The agent must navigate in a maze composed of four rooms. The goal is
    now restricted to appear only in the rooms specified in `goal_rooms`.

    The allowed room names are:
      - "top_left"
      - "top_right"
      - "bottom_left"
      - "bottom_right"

    For example, to spawn the goal only in the bottom right room, initialize with:

        MyFourRoomsEnv(goal_rooms=["bottom_right"])
    """

    def __init__(
        self,
        agent_pos=(1, 1),
        max_steps=100,
        goal_rooms: list[str] | None = None,
        agent_rooms: list[str] | None = None,
        randomize_agent_pos: bool = False,
        randomize_agent_dir: bool = False,
        randomize_goal_pos: bool = True,
        **kwargs,
    ):
        # Fixed agent starting position.
        self._agent_default_pos = agent_pos
        # If no goal_rooms are provided, default to bottom right.
        if goal_rooms is None:
            goal_rooms = ["bottom_right"]
        if agent_rooms is None:
            agent_rooms = ["top_left"]
        self.goal_rooms = goal_rooms
        self.agent_rooms = agent_rooms
        self.randomize_agent_pos = randomize_agent_pos
        self.randomize_agent_dir = randomize_agent_dir
        self.randomize_goal_pos = randomize_goal_pos

        self.size = 15
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    @staticmethod
    def _room_area(room_name: str, room_w: int, room_h: int):
        if room_name not in ROOM_MAPPING:
            raise ValueError(
                f"Invalid room name: {room_name}. "
                f"Valid options are: {list(ROOM_MAPPING.keys())}"
            )

        i, j = ROOM_MAPPING[room_name]
        top_x = i * room_w + 1
        top_y = j * room_h + 1
        size_x = room_w - 1  # interior width (excludes walls)
        size_y = room_h - 1  # interior height (excludes walls)
        return (top_x, top_y), (size_x, size_y)

    def _sample_room(self, rooms: list[str]) -> str:
        if not rooms:
            raise ValueError("Cannot sample a room from an empty list")
        return rooms[self._rand_int(0, len(rooms))]

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # Create the four rooms with connecting doors.
        for j in range(2):
            for i in range(2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Vertical wall and door between rooms
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    # Fixed door position: middle of the wall
                    pos = (xR, yT + room_h // 2)
                    self.grid.set(*pos, None)

                # Horizontal wall and door between rooms
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    # Fixed door position: middle of the wall
                    pos = (xL + room_w // 2, yB)
                    self.grid.set(*pos, None)

        # Now restrict the goal spawn to one of the allowed rooms.
        goal = Goal()
        allowed_room = self._sample_room(self.goal_rooms)
        goal_top, goal_size = self._room_area(allowed_room, room_w, room_h)
        if self.randomize_goal_pos:
            goal_pos = self.place_obj(goal, top=goal_top, size=goal_size)
        else:
            goal_pos = (
                goal_top[0] + goal_size[0] // 2,
                goal_top[1] + goal_size[1] // 2,
            )
            self.grid.set(goal_pos[0], goal_pos[1], goal)
        self.goal = goal
        self.goal_pos = tuple(goal_pos)

        if self.randomize_agent_pos:
            agent_room = self._sample_room(self.agent_rooms)
            agent_top, agent_size = self._room_area(agent_room, room_w, room_h)
            self.place_agent(
                top=agent_top,
                size=agent_size,
                rand_dir=self.randomize_agent_dir,
                max_tries=1000,
            )
        else:
            # Set fixed agent starting position.
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 0


class StochasticFourRoomsEnv(MyFourRoomsEnv):
    """Four rooms with stochastic initial agent and goal positions."""

    def __init__(self, **kwargs):
        kwargs.setdefault("agent_rooms", ["top_left"])
        kwargs.setdefault("goal_rooms", ["bottom_right"])
        kwargs.setdefault("randomize_agent_pos", True)
        kwargs.setdefault("randomize_agent_dir", True)
        kwargs.setdefault("randomize_goal_pos", True)
        super().__init__(**kwargs)
