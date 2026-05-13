# type:ignore
from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal  # type:ignore
from minigrid.core.world_object import Ball, Box, Door, Key, WorldObj
from minigrid.minigrid_env import MiniGridEnv


def reject_next_to(env: MiniGridEnv, pos: tuple[int, int]):
    """Filter out object positions that are right next to the agent's start."""
    sx, sy = env.agent_pos
    x, y = pos
    return abs(sx - x) + abs(sy - y) < 2


class Room:
    def __init__(self, top: tuple[int, int], size: tuple[int, int]):
        self.top = top
        self.size = size
        # Doors: order is right, down, left, up.
        self.doors: list[bool | Door | None] = [None] * 4
        self.door_pos: list[tuple[int, int] | None] = [None] * 4
        self.neighbors: list[Room | None] = [None] * 4
        self.locked: bool = False
        self.objs: list[WorldObj] = []

    def rand_pos(self, env: MiniGridEnv) -> tuple[int, int]:
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._randPos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)

    def pos_inside(self, x: int, y: int) -> bool:
        topX, topY = self.top
        sizeX, sizeY = self.size
        return topX <= x < topX + sizeX and topY <= y < topY + sizeY


class MyRoomGrid(MiniGridEnv):
    """Environment with multiple rooms and random objects.

    Serves as a base class for other environments.
    """

    def __init__(
        self,
        room_size: int = 7,
        num_rows: int = 3,
        num_cols: int = 3,
        max_steps: int = 100,
        agent_view_size: int = 7,
        **kwargs,
    ):
        assert room_size >= 3
        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        height = (room_size - 1) * num_rows + 1
        width = (room_size - 1) * num_cols + 1
        self.mission = ""

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def room_from_pos(self, x: int, y: int) -> Room:
        i = x // (self.room_size - 1)
        j = y // (self.room_size - 1)
        return self.room_grid[j][i]

    def get_room(self, i: int, j: int) -> Room:
        return self.room_grid[j][i]

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.room_grid = []

        for j in range(self.num_rows):
            row = []
            for i in range(self.num_cols):
                room = Room(
                    (i * (self.room_size - 1), j * (self.room_size - 1)),
                    (self.room_size, self.room_size),
                )
                row.append(room)
                self.grid.wall_rect(*room.top, *room.size)
            self.room_grid.append(row)

        for j in range(self.num_rows):
            for i in range(self.num_cols):
                room = self.room_grid[j][i]
                x_l, y_l = room.top[0] + 1, room.top[1] + 1
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )
                # Set door positions: right, down, left, up.
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        self.agent_pos = np.array(
            (
                (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
                (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
            )
        )
        self.agent_dir = 0

    def place_in_room(
        self, i: int, j: int, obj: WorldObj
    ) -> tuple[WorldObj, tuple[int, int]]:
        room = self.get_room(i, j)
        pos = self.place_obj(
            obj, room.top, room.size, reject_fn=reject_next_to, max_tries=1000
        )
        room.objs.append(obj)
        return obj, pos

    def add_object(
        self, i: int, j: int, kind: str | None = None, color: str | None = None
    ) -> tuple[WorldObj, tuple[int, int]]:
        if kind is None:
            kind = self._rand_elem(["key", "ball", "box"])
        if color is None:
            color = self._rand_color()
        assert kind in ["key", "ball", "box"]
        if kind == "key":
            obj = Key(color)
        elif kind == "ball":
            obj = Ball(color)
        elif kind == "box":
            obj = Box(color)
        else:
            raise ValueError(
                f"{kind} object kind is not available in this environment."
            )
        return self.place_in_room(i, j, obj)

    def add_door(
        self,
        i: int,
        j: int,
        door_idx: int | None = None,
        color: str | None = None,
        locked: bool | None = None,
    ) -> tuple[Door, tuple[int, int]]:
        room = self.get_room(i, j)
        if door_idx is None:
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break
        if color is None:
            color = self._rand_color()
        if locked is None:
            locked = self._rand_bool()
        assert room.doors[door_idx] is None, "door already exists"
        room.locked = locked
        door = Door(color, is_locked=locked)
        pos = room.door_pos[door_idx]
        self.grid.set(pos[0], pos[1], door)
        door.cur_pos = pos
        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx + 2) % 4] = door
        return door, pos

    def remove_wall(self, i: int, j: int, wall_idx: int):
        room = self.get_room(i, j)
        assert 0 <= wall_idx < 4
        assert room.doors[wall_idx] is None, "door exists on this wall"
        assert room.neighbors[wall_idx]
        neighbor = room.neighbors[wall_idx]
        tx, ty = room.top
        w, h = room.size
        if wall_idx == 0:
            for i in range(1, h - 1):
                self.grid.set(tx + w - 1, ty + i, None)
        elif wall_idx == 1:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty + h - 1, None)
        elif wall_idx == 2:
            for i in range(1, h - 1):
                self.grid.set(tx, ty + i, None)
        elif wall_idx == 3:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty, None)
        room.doors[wall_idx] = True
        neighbor.doors[(wall_idx + 2) % 4] = True

    def place_agent(
        self, i: int | None = None, j: int | None = None, rand_dir: bool = True
    ) -> np.ndarray:
        if i is None:
            i = self._rand_int(0, self.num_cols)
        if j is None:
            j = self._rand_int(0, self.num_rows)
        room = self.room_grid[j][i]
        while True:
            super().place_agent(room.top, room.size, rand_dir, max_tries=1000)
            front_cell = self.grid.get(*self.front_pos)
            if front_cell is None or front_cell.type == "wall":
                break
        return self.agent_pos

    def connect_all(
        self, door_colors: list[str] = COLOR_NAMES, max_itrs: int = 5000
    ) -> list[Door]:
        start_room = self.room_from_pos(*self.agent_pos)
        added_doors = []

        def find_reach():
            reach = set()
            stack = [start_room]
            while stack:
                room = stack.pop()
                if room in reach:
                    continue
                reach.add(room)
                for i in range(4):
                    if room.doors[i]:
                        stack.append(room.neighbors[i])
            return reach

        num_itrs = 0
        while True:
            if num_itrs > max_itrs:
                raise RecursionError("connect_all failed")
            num_itrs += 1
            reach = find_reach()
            if len(reach) == self.num_rows * self.num_cols:
                break
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            k = self._rand_int(0, 4)
            room = self.get_room(i, j)
            if not room.door_pos[k] or room.doors[k]:
                continue
            neighbor_room = room.neighbors[k]
            if room.locked or neighbor_room.locked:
                continue
            color = self._rand_elem(door_colors)
            door, _ = self.add_door(i, j, k, color, False)
            added_doors.append(door)
        return added_doors

    def add_distractors(
        self,
        i: int | None = None,
        j: int | None = None,
        num_distractors: int = 10,
        all_unique: bool = True,
    ) -> list[WorldObj]:
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))
        dists = []
        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            typ = self._rand_elem(["key", "ball", "box"])
            obj = (typ, color)
            if all_unique and obj in objs:
                continue
            room_i = i if i is not None else self._rand_int(0, self.num_cols)
            room_j = j if j is not None else self._rand_int(0, self.num_rows)
            dist, pos = self.add_object(room_i, room_j, *obj)
            objs.append(obj)
            dists.append(dist)
        return dists


class MyUnlockEnv(MyRoomGrid):
    """
    Modified environment: The agent must retrieve a key, unlock the door in the first room,
    then move into the second room and reach the goal. The mission is "go to the goal".
    The reward is only given when the agent reaches the goal.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        randomize_agent_pos: bool = True,
        randomize_agent_dir: bool = True,
        randomize_goal_pos: bool = False,
        **kwargs,
    ):
        room_size = 6
        self.randomize_agent_pos = randomize_agent_pos
        self.randomize_agent_dir = randomize_agent_dir
        self.randomize_goal_pos = randomize_goal_pos
        # Update mission to reflect the new goal
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 8 * room_size**2
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "go to the goal"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # --- Configure Room 0 (left room) ---
        room = self.get_room(0, 0)
        # Fix the door position (right wall) to the center.
        fixed_door_pos = (
            room.top[0] + room.size[0] - 1,
            room.top[1] + room.size[1] // 2,
        )
        room.door_pos[0] = fixed_door_pos
        # Add a locked door on the right wall.
        self.door, _ = self.add_door(0, 0, 0, locked=True)
        # Place the key in Room 0 (the key matches the door's color).
        self.add_object(0, 0, "key", self.door.color)
        # Place the agent in Room 0.
        if self.randomize_agent_pos:
            self.place_agent(0, 0, rand_dir=self.randomize_agent_dir)
        else:
            agent_pos = (
                room.top[0] + room.size[0] // 2,
                room.top[1] + room.size[1] // 2,
            )
            self.agent_pos = np.array(agent_pos)
            self.agent_dir = 0

        # --- Place the Goal in Room 1 (the second room) ---
        goal = Goal()
        room = self.get_room(1, 0)
        if self.randomize_goal_pos:
            goal_pos = self.place_obj(
                goal,
                top=(room.top[0] + 1, room.top[1] + 1),
                size=(room.size[0] - 2, room.size[1] - 2),
                max_tries=1000,
            )
        else:
            # Use a fixed location for the goal: center of the room.
            goal_pos = (
                room.top[0] + room.size[0] // 2,
                room.top[1] + room.size[1] // 2,
            )
            self.grid.set(goal_pos[0], goal_pos[1], goal)
        self.goal = goal
        self.goal_pos = tuple(goal_pos)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Check if the agent has reached the goal position.
        if tuple(self.agent_pos) == self.goal_pos:
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info


class StochasticUnlockEnv(MyUnlockEnv):
    """Unlock task with stochastic goal position in the right room."""

    def __init__(self, **kwargs):
        kwargs.setdefault("randomize_agent_pos", True)
        kwargs.setdefault("randomize_agent_dir", True)
        kwargs.setdefault("randomize_goal_pos", True)
        super().__init__(**kwargs)
