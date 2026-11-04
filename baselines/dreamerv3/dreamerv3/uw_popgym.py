"""Bridge: POPGym MineSweeper (gymnasium) -> legacy gym API for FromGym.

Encoding choices, declared here because the PyTorch arm's adapter was not
available when this bridge was written:
- observation Discrete(3) -> one-hot float32 vector (3,)
- action MultiDiscrete([w,h]) -> flattened Discrete(w*h), decoded as divmod
Env selection via UW_POPGYM_ENV in {MineSweeperEasy, MineSweeperMedium,
MineSweeperHard}. Seeding: episode i uses seed UW_TRAIN_SEED*10^6 + i.
"""

import os

import gym
import numpy as np


class POPGymMineSweeper(gym.Env):

    def __init__(self):
        import popgym  # noqa: F401
        from popgym.envs import minesweeper as ms
        name = os.environ.get("UW_POPGYM_ENV", "MineSweeperEasy")
        self._env = getattr(ms, name)()
        n_obs = int(self._env.observation_space.n)
        dims = [int(d) for d in self._env.action_space.nvec]
        self._dims = dims
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n_obs,), np.float32)
        self.action_space = gym.spaces.Discrete(int(np.prod(dims)))
        self._n_obs = n_obs
        self._counter = 0
        self._base = int(os.environ.get("UW_TRAIN_SEED", "0")) * 1_000_000

    def _onehot(self, o):
        v = np.zeros(self._n_obs, np.float32)
        v[int(o)] = 1.0
        return v

    def reset(self):
        o, _ = self._env.reset(seed=self._base + self._counter)
        self._counter += 1
        return self._onehot(o)

    def step(self, a):
        a = int(a)
        x, y = divmod(a, self._dims[1])
        o, r, term, trunc, info = self._env.step(np.array([x, y]))
        info = dict(info or {})
        info["is_terminal"] = bool(term)
        return self._onehot(o), float(r), bool(term or trunc), info


gym.envs.registration.register(
    id="POPGymMineSweeper-v0", entry_point=POPGymMineSweeper)
