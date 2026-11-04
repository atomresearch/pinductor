"""Bridge: MiniGrid (gymnasium) -> legacy gym API expected by embodied FromGym.

Registers ``mgflat:MiniGridFlat-v0``-style ids. Observation = flattened
symbolic image (7x7x3) + direction, as a float32 vector (Dreamer encodes
vectors with a symlog MLP). Task selection via the ``env_id`` kwarg or the
MGFLAT_ENV environment variable.
"""

import os

import gym
import gymnasium
import minigrid  # noqa: F401  (registers MiniGrid envs with gymnasium)
import numpy as np


class MiniGridFlat(gym.Env):

    def __init__(self, env_id=None):
        env_id = env_id or os.environ.get("MGFLAT_ENV", "MiniGrid-Empty-8x8-v0")
        self._env = gymnasium.make(env_id)
        img = self._env.observation_space["image"]
        n = int(np.prod(img.shape)) + 1  # + direction
        self.observation_space = gym.spaces.Box(0.0, 255.0, (n,), np.float32)
        self.action_space = gym.spaces.Discrete(int(self._env.action_space.n))

    def _flat(self, o):
        return np.concatenate(
            [o["image"].ravel(), [o["direction"]]]).astype(np.float32)

    def reset(self):
        o, _ = self._env.reset()
        return self._flat(o)

    def step(self, a):
        o, r, term, trunc, info = self._env.step(int(a))
        info = dict(info)
        info["is_terminal"] = bool(term)
        return self._flat(o), float(r), bool(term or trunc), info


gym.envs.registration.register(
    id="MiniGridFlat-v0", entry_point=MiniGridFlat)
