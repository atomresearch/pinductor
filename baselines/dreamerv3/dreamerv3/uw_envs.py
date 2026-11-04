"""Bridges: OUR paper environments -> legacy gym API for embodied.FromGym.

Two envs, exact alignment with the Pinductor paper benches:

- ``uwmg:UWMiniGrid-v0``  — our MinigridEnvironment (3x3 partial FOV +
  direction + carrying), same observability as Pinductor. Env id via
  UW_ENV_NAME, max steps via UW_MAX_STEPS.
- ``uwmg:UWBlocks-v0``    — our BlocksEnvironment rand6 (obs = 7-int
  image: current stack + held), same as the PPO baseline.

Seeding protocol (identical to scripts/rand6_ppo_baseline.py):
train resets use TRAIN_SEED_BASE + train_seed*1e6 + counter (disjoint from
eval); eval uses ep_seeds 0..29 via set_eval_seed().
Configure with env vars: UW_TRAIN_SEED (int), UW_ENV_NAME, UW_MAX_STEPS.
"""

import os
import pathlib
import sys

import gym
import numpy as np

# Repository root: three levels up from baselines/dreamerv3/dreamerv3/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

TRAIN_SEED_BASE = 100_000
EVAL_EP_SEEDS = list(range(30))


def _train_seed():
    return int(os.environ.get("UW_TRAIN_SEED", "0"))


class UWMiniGrid(gym.Env):
    """Our MinigridEnvironment, 3x3 FOV, flattened to an 11-d float vector:
    [fov(9), agent_dir(1), carrying_idx(1)]."""

    def __init__(self):
        from uncertain_worms.environments.minigrid.minigrid_env import (
            MinigridEnvironment)
        env_name = os.environ.get("UW_ENV_NAME", "CornerGoalRandom-Empty-10x10-v0")
        max_steps = int(os.environ.get("UW_MAX_STEPS", "40"))
        self._env = MinigridEnvironment(
            env_name=env_name, fully_obs=False, max_steps=max_steps)
        self.observation_space = gym.spaces.Box(-1.0, 32.0, (11,), np.float32)
        self.action_space = gym.spaces.Discrete(6)
        self._counter = 0
        self._eval_seed = None

    def set_eval_seed(self, s):
        self._eval_seed = s

    def _flat(self, o):
        from minigrid.core.constants import OBJECT_TO_IDX
        carry = 0.0
        if o.carrying is not None:
            carry = float(OBJECT_TO_IDX.get(
                getattr(o.carrying, "type", str(o.carrying)), 1))
        return np.concatenate(
            [np.asarray(o.image, np.float32).ravel(),
             [float(o.agent_dir), carry]]).astype(np.float32)

    def reset(self):
        if self._eval_seed is not None:
            seed = self._eval_seed
        else:
            seed = TRAIN_SEED_BASE + _train_seed() * 1_000_000 + self._counter
            self._counter += 1
        self._env.reset(seed=seed)
        return self._flat(self._env.get_last_obs())

    def step(self, a):
        obs, state, r, term, trunc, info = self._env.step(int(a))
        info = dict(info or {})
        info["is_terminal"] = bool(term)
        return self._flat(obs), float(r), bool(term or trunc), info


class UWBlocks(gym.Env):
    """Our BlocksEnvironment rand6 — obs = 7-int image (stack + held)."""

    def __init__(self):
        from uncertain_worms.environments.blocksworld.blocksworld_env import (
            BlocksEnvironment)
        self._env = BlocksEnvironment(instance="rand6", max_steps=120)
        cap = self._env.cap
        self.observation_space = gym.spaces.Box(
            0.0, float(cap + 1), (cap + 1,), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._counter = 0
        self._eval_seed = None
        self._steps = 0

    def set_eval_seed(self, s):
        self._eval_seed = s

    def _obs(self):
        return np.asarray(self._env.get_last_obs().image, np.float32)

    def reset(self):
        if self._eval_seed is not None:
            seed = self._eval_seed
        else:
            seed = TRAIN_SEED_BASE + _train_seed() * 1_000_000 + self._counter
            self._counter += 1
        self._env.reset(seed=seed)
        self._steps = 0
        return self._obs()

    def step(self, a):
        obs, state, r, term, trunc, info = self._env.step(int(a))
        self._steps += 1
        done = bool(term) or self._steps >= 120
        info = dict(info or {})
        info["is_terminal"] = bool(term)
        return self._obs(), float(r), done, info


gym.envs.registration.register(id="UWMiniGrid-v0", entry_point=UWMiniGrid)
gym.envs.registration.register(id="UWBlocks-v0", entry_point=UWBlocks)
