from __future__ import annotations

from gymnasium.envs.registration import register

__version__ = "2.5.0"


def register_minigrid_envs() -> None:
    register(
        id="CustomMiniGrid-Empty-10x10-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.empty_large:CustomEmptyEnv",
        kwargs={"size": 10},
    )
    register(
        id="CornerGoal-Empty-10x10-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.corners_large:CornerGoalEmptyEnv",
        kwargs={"size": 10},
    )
    register(
        id="CornerGoalRandom-Empty-10x10-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.corners_large_random:CornerGoalRandomEmptyEnv",
        kwargs={"size": 10},
    )
    register(
        id="TinyEmpty-4x4-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.empty_large:CustomEmptyEnv",
        kwargs={"size": 4, "agent_start_dir": 3},
    )

    register(
        id="MyMiniGrid-FourRooms-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.four_rooms:MyFourRoomsEnv",
        kwargs={},
    )

    register(
        id="MyMiniGrid-StochasticFourRooms-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.four_rooms:StochasticFourRoomsEnv",
        kwargs={},
    )

    register(
        id="MyMiniGrid-MemoryEnv-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.memory:MemoryEnv",
        kwargs={},
    )

    register(
        id="MyMiniGrid-LavaWall-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.lavagap:LavaGapEnv",
        kwargs={"size": 10},
    )

    register(
        id="MyMiniGrid-StochasticLavaWall-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.lavagap:StochasticLavaGapEnv",
        kwargs={"size": 10},
    )

    register(
        id="MyUnlockEnv-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.unlock:MyUnlockEnv",
        kwargs={},
    )

    register(
        id="MyMiniGrid-StochasticUnlock-v0",
        entry_point="uncertain_worms.environments.minigrid.custom_environments.unlock:StochasticUnlockEnv",
        kwargs={},
    )


register_minigrid_envs()
