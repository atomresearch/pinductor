# type:ignore
#!/usr/bin/env python
"""collect_demos.py.

This script allows a user to manually control a Minigrid agent via keyboard input to collect demos.
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
import random

from uncertain_worms.environments import *

# Import the MinigridEnvironment and Actions enum.
# (Assumes these are available in your uncertain_worms package.)
from uncertain_worms.environments.minigrid.minigrid_env import (
    Actions,
    MinigridEnvironment,
)
from uncertain_worms.structs import Observation, ReplayBuffer, State
from uncertain_worms.utils import PROJECT_ROOT


def main():
    num_demos = 10
    max_steps = 50  # Maximum steps per episode; adjust as needed.

    # env_name = "MiniGrid-Empty-5x5-v0"
    # env_name = "CornerGoalRandom-Empty-10x10-v0"
    # env_name = "MyMiniGrid-LavaWall-v0"
    # env_name = "MyMiniGrid-FourRooms-v0"
    env_name = "MyUnlockEnv-v0"

    # Initialize an empty replay buffer.
    replay_buffer = ReplayBuffer[State, int, Observation]()

    # Create the environment (here using default env_name "MiniGrid-Empty-5x5-v0" and full observability).
    env = MinigridEnvironment(env_name=env_name, max_steps=max_steps, fully_obs=False)

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

    for demo in range(num_demos):
        print(f"\n=== Starting demo episode {demo + 1}/{num_demos} ===")
        # Reset environment with a random seed.
        initial_state = env.reset(seed=random.randint(0, 10000))
        env.env.unwrapped.render_mode = "human"
        env.env.render()
        print("Initial state:")
        print(initial_state)

        # Initialize the episode tracking.
        previous_state = initial_state
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
                next_obs,
                reward,
                terminated,
            )
            previous_state = next_state
            step += 1

            if terminated or truncated:
                break

        # Finish the current episode in the replay buffer.
        replay_buffer.wrap_up_episode()
        print(f"Demo episode {demo + 1} finished.\n")

    # Ask for a filename and save the replay buffer.
    filename = os.path.join(
        PROJECT_ROOT, "environments/minigrid/trajectory_data", env_name + ".pkl"
    )
    replay_buffer.save_to_file(filename)
    print(f"Demos saved to {filename}")


if __name__ == "__main__":
    main()
