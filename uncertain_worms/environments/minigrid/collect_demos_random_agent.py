# type:ignore
#!/usr/bin/env python
"""collect_demos.py.

This script allows a user to manually control a Minigrid agent via keyboard input to collect demos.
The agent is initialized with a random position and direction.
The controls are:
    w: move forward
    a: turn left
    d: turn right
    p: pickup
    t: toggle (e.g. open door)
    q: drop
    f: finish the current episode

After collecting 5 demo episodes, the demos are saved to a replay buffer file.
"""

import os

import numpy as np

from uncertain_worms.environments import *

# Import the MinigridEnvironment and Actions enum.
# (Assumes these are available in your uncertain_worms package.)
from uncertain_worms.environments.minigrid.minigrid_env import (
    Actions,
    MinigridEnvironment,
    minigrid_hardcoded_observation_model_gen,
    minigrid_empty_observation_gen,
    minigrid_empty_state_gen,
    minigrid_hardcoded_initial_model_gen,
    minigrid_hardcoded_initial_model_fixed_grid_random_agent_gen,
    state_to_grid,
)
from uncertain_worms.structs import Observation, ReplayBuffer, State
from uncertain_worms.utils import PROJECT_ROOT


def main():
    num_demos = 10
    max_steps = 50  # Maximum steps per episode; adjust as needed.

    env_name = "MiniGrid-Empty-5x5-v0"
    # env_name = "CornerGoalRandom-Empty-10x10-v0"
    # env_name = "MyMiniGrid-LavaWall-v0"
    # env_name = "MyMiniGrid-FourRooms-v0"
    # env_name = "MyUnlockEnv-v0"

    # Initialize an empty replay buffer.
    replay_buffer = ReplayBuffer[State, int, Observation]()

    # Create the environment (here using default env_name "MiniGrid-Empty-5x5-v0" and full observability).
    env = MinigridEnvironment(env_name=env_name, max_steps=max_steps, fully_obs=False, pure_obs=True)

    obs_model = minigrid_hardcoded_observation_model_gen()
    empty_obs = minigrid_empty_observation_gen()

    # Define key-to-action mapping.
    key_action_mapping = {
        "w": Actions.forward,
        "a": Actions.left,
        "d": Actions.right,
        "p": Actions.pickup,
        "q": Actions.drop,
        "t": Actions.toggle,
        "f": "finish",
    }

    print("Demo collection started.")
    print("Controls:")
    print("  w: move forward")
    print("  a: turn left")
    print("  d: turn right")
    print("  p: pickup")
    print("  t: toggle")
    print("  q: drop")
    print("  f: finish the current episode")

    initial_model = minigrid_hardcoded_initial_model_fixed_grid_random_agent_gen()
    empty_state = minigrid_empty_state_gen(env_name=env_name)

    for demo in range(num_demos):
        print(f"\n=== Starting demo episode {demo + 1}/{num_demos} ===")
        #initial_state = env.reset(seed=random.randint(0, 10000))
        initial_state = initial_model(empty_state)
        _ = env.reset(seed=np.random.randint(0, 10000))
        base_env = env.env.unwrapped
        state_to_grid(initial_state, base_env.grid)
        base_env.agent_pos = tuple(initial_state.agent_pos)
        base_env.agent_dir = int(initial_state.agent_dir)
        env.env.unwrapped.render_mode = "human"
        env.env.render()

        initial_obs = obs_model(
            state=initial_state,
            action=int(Actions.forward),  # Dummy action for initial observation
            empty_obs=empty_obs,
            agent_view_size=3,
        )

        print("Initial state:")
        print(initial_state)
        print("Initial observation:")
        print(initial_obs)

        # Initialize the episode tracking.
        previous_state = initial_state
        previous_obs = initial_obs
        terminated = False
        step = 0

        while not terminated:
            user_input = input("Enter action (w/a/d/p/t/q/f): ").strip().lower()
            if user_input not in key_action_mapping:
                print("Invalid input. Please try again.")
                continue
            if key_action_mapping[user_input] == "finish":
                print("Finishing current episode.")
                break

            action = key_action_mapping[user_input]
            next_obs, next_state, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step {step}: Action {action}, Reward: {reward}, Terminated: {terminated}"
            )
            print("Next state:")
            print(next_state)
            print("Next observation:")
            print(next_obs)

            # Append the transition to the replay buffer.
            replay_buffer.append_episode_step(
                previous_state,
                next_state,
                action,
                previous_obs,
                next_obs,
                reward,
                terminated,
            )
            previous_state = next_state
            previous_obs = next_obs
            step += 1

            if terminated or truncated:
                break

        # Finish the current episode in the replay buffer.
        replay_buffer.wrap_up_episode()
        print(f"Demo episode {demo + 1} finished.\n")
    
    # Ask for a filename and save the replay buffer.
    environment_name = f"{env_name}_pure_randominit_agent" if env.pure_obs else f"{env_name}"
    filename = os.path.join(
        PROJECT_ROOT, "environments/minigrid/trajectory_data", environment_name + ".pkl"
    )
    replay_buffer.save_to_file(filename)
    print(f"Demos saved to {filename}")


if __name__ == "__main__":
    main()