#!/usr/bin/env python
"""Train the recurrent PPO baseline on any of the paper's environments.

One entry point for all three suites, replacing the two implementations the
rebuttal used. See uncertain_worms/policies/recurrent_ppo.py for why.

    python scripts/train_recurrent_ppo.py --env blocks --episodes 1000
    python scripts/train_recurrent_ppo.py --env lava --episodes 1000 --seed 1
    python scripts/train_recurrent_ppo.py --env minesweeper_easy --smoke

Training and evaluation use disjoint seed ranges, so a run is never scored on
an instance it trained on.

Budget. The paper's neural baselines get 1000 online episodes and that is the
right order here too. MiniGrid exposes six actions, two of which do nothing in
several environments, so exploration is slower than the action count suggests:
on Empty-5x5 the return stays at zero for the first six hundred episodes and
only then climbs, reaching 0.27 on held-out instances by episode 1200. A few
hundred episodes prove nothing about this implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertain_worms.environments.obs_encoding import encoder_for  # noqa: E402
from uncertain_worms.policies.recurrent_ppo import RecurrentPPO  # noqa: E402

TRAIN_SEED_BASE = 1_000_000
EVAL_SEED_BASE = 0  # the held-out instances the paper evaluates on


def make_env(name: str):
    """(environment, max_steps, n_actions) for a suite key.

    The action count is stated here rather than read off the environment,
    because our environments do not expose one: the paper passes it through the
    `actions:` list in each config. These values mirror those configs exactly
    (MiniGrid 0-5, Blocks 0-3). Guessing it is not harmless, since dropping
    MiniGrid's actions 4 and 5 removes pick-up and toggle, which makes Unlock
    and KeyCorridor unsolvable while leaving Empty and Lava apparently fine.
    """
    if name == "blocks":
        from uncertain_worms.environments.blocksworld.blocksworld_env import (
            BlocksEnvironment)
        return BlocksEnvironment(instance="rand6", max_steps=120), 120, 4
    if name.startswith("minesweeper_"):
        from uncertain_worms.environments.popgym.popgym_env import (
            PopGymEnvironment)
        diff = name.split("_", 1)[1].capitalize()
        env = PopGymEnvironment(env_id=f"MineSweeper{diff}", max_steps=14)
        return env, 14, int(env.env.action_space.n)
    from uncertain_worms.environments.minigrid.minigrid_env import (
        MinigridEnvironment)
    envs = {"lava": "MyMiniGrid-LavaWall-v0",
            "corners": "CornerGoalRandom-Empty-10x10-v0",
            "four_rooms": "MyMiniGrid-FourRooms-v0",
            "unlock": "MyUnlockEnv-v0",
            "empty": "MiniGrid-Empty-5x5-v0"}
    steps = {"four_rooms": 60, "empty": 10}
    return (MinigridEnvironment(env_name=envs[name], fully_obs=False,
                                max_steps=steps.get(name, 40)),
            steps.get(name, 40), 6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16, help="episodes per update")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="32 episodes, just to prove the loop runs")
    args = p.parse_args()

    if args.smoke:
        args.episodes, args.batch, args.eval_episodes = 32, 8, 5

    env, max_steps, n_actions = make_env(args.env)
    encode, obs_dim = encoder_for(env)
    agent = RecurrentPPO(obs_dim, n_actions, seed=args.seed)
    print(f"[ppo] env={args.env} obs_dim={obs_dim} actions={n_actions} "
          f"max_steps={max_steps} episodes={args.episodes}")

    rows, done = [], 0
    while done < args.episodes:
        n = min(args.batch, args.episodes - done)
        batch, train_ret = agent.collect(
            env, encode, n, max_steps,
            seed_of=lambda i: TRAIN_SEED_BASE + args.seed * 1_000_000 + done + i)
        loss = agent.update(batch)
        done += n
        rows.append({"episodes": done, "train_return": round(train_ret, 4),
                     "loss": round(loss, 4)})
        print(f"  {done:5d} ep  train_return {train_ret:.4f}  loss {loss:.4f}",
              flush=True)

    _, eval_ret = agent.collect(env, encode, args.eval_episodes, max_steps,
                                seed_of=lambda i: EVAL_SEED_BASE + i)
    print(f"[ppo] held-out return over {args.eval_episodes} episodes: {eval_ret:.4f}")

    out = args.out or f"outputs/ppo_{args.env}_seed{args.seed}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episodes", "train_return", "loss"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[ppo] -> {out}")


if __name__ == "__main__":
    main()
