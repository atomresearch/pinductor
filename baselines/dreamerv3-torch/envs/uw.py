"""Bridge: OUR paper environments (uncertain_worms) -> NM512/dreamerv3-torch env API.

NEW module (does not modify existing repo files). Provides discrete-action,
vector-observation POMDP envs in the dict-obs format DreamerV3 expects:

    obs = {
        "vector":     float32 (D,)   real observation (encoded by the MLP head)
        "image":      uint8   (8,8,3) DUMMY, ignored by encoder/decoder
                              (cnn_keys='$^'); required only by
                              WorldModel.preprocess (divides image by 255) and
                              tools.simulate video logging.
        "is_first":   bool
        "is_last":    bool
        "is_terminal":bool
    }

Use with encoder/decoder ``mlp_keys='vector', cnn_keys='$^'`` and
``video_pred_log: false`` (see configs uw_* below / configs_uw.yaml).

Six envs, selected by task string ``uw_<name>`` (see UW_ENVS):
  uw_blocks_rand6, uw_cornergoal, uw_empty5x5, uw_fourrooms, uw_lavawall, uw_unlock

Observation / action / seeding conventions mirror EXACTLY
the sibling JAX bridge, baselines/dreamerv3/dreamerv3/uw_envs.py:
  * MiniGrid: 3x3 FOV (9) + agent_dir (1) + carrying_idx (1) -> 11-d float vector.
  * Blocks rand6: 7-int image vector (stack + held).
  * train reset seed = 100000 + train_seed*1_000_000 + counter (disjoint from eval).
  * binary terminal reward (whatever the underlying env emits; is_terminal = env term).
"""

import os
import pathlib
import sys

import gym
import numpy as np

# uncertain_worms lives in the main repo; add to path (do NOT touch its venv).
# Repository root: three levels up from baselines/dreamerv3-torch/envs/.
UW_REPO = os.environ.get(
    "UW_REPO", str(pathlib.Path(__file__).resolve().parents[3])
)
if UW_REPO not in sys.path:
    sys.path.insert(0, UW_REPO)

TRAIN_SEED_BASE = 100_000

# task suffix -> (kind, config).  kind in {"blocks", "minigrid"}.
UW_ENVS = {
    "blocks_rand6": ("blocks", {"instance": "rand6", "max_steps": 120}),
    "cornergoal": ("minigrid", {"env_name": "CornerGoalRandom-Empty-10x10-v0", "max_steps": 40}),
    "empty5x5": ("minigrid", {"env_name": "MiniGrid-Empty-5x5-v0", "max_steps": 10}),
    "fourrooms": ("minigrid", {"env_name": "MyMiniGrid-FourRooms-v0", "max_steps": 60}),
    "lavawall": ("minigrid", {"env_name": "MyMiniGrid-LavaWall-v0", "max_steps": 40}),
    "unlock": ("minigrid", {"env_name": "MyUnlockEnv-v0", "max_steps": 40}),
    # POPGym MineSweeper — observation and action spaces replicate the vanilla
    # JAX wrapper (dreamerv3/dreamerv3/uw_popgym.py) exactly: one-hot of the
    # Discrete(n) clue, action = flattened MultiDiscrete grid cell.
    "ms_easy": ("popgym", {"env_name": "MineSweeperEasy", "n_obs": 3, "n_actions": 16}),
    "ms_medium": ("popgym", {"env_name": "MineSweeperMedium", "n_obs": 7, "n_actions": 36}),
    "ms_hard": ("popgym", {"env_name": "MineSweeperHard", "n_obs": 10, "n_actions": 64}),
}

# demo pickle files (15-episode paper demos), relative to the uncertain_worms package.
UW_DEMO_FILES = {
    "blocks_rand6": "environments/blocksworld/trajectory_data/rand6_demos.pkl",
    "cornergoal": "environments/minigrid/trajectory_data/CornerGoalRandom-Empty-10x10-v0_paper_N10.pkl",
    "empty5x5": "environments/minigrid/trajectory_data/MiniGrid-Empty-5x5-v0_paper_N10.pkl",
    "fourrooms": "environments/minigrid/trajectory_data/MyMiniGrid-FourRooms-v0_paper_N10.pkl",
    "lavawall": "environments/minigrid/trajectory_data/MyMiniGrid-LavaWall-v0_paper_N10.pkl",
    "unlock": "environments/minigrid/trajectory_data/MyUnlockEnv-v0_paper_N10.pkl",
    "ms_easy": "environments/popgym/trajectory_data/MineSweeperEasy_random_N50.pkl",
    "ms_medium": "environments/popgym/trajectory_data/MineSweeperMedium_random_N50.pkl",
    "ms_hard": "environments/popgym/trajectory_data/MineSweeperHard_random_N50.pkl",
}

_DUMMY_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


def _popgym_onehot(scalar, n_obs):
    """One-hot of the POPGym Discrete(n) observation — same as the JAX wrapper."""
    v = np.zeros(n_obs, np.float32)
    v[int(scalar)] = 1.0
    return v


def _minigrid_flat(o):
    """MiniGridObservation -> 11-d float32 vector (matches JAX uw_envs)."""
    from minigrid.core.constants import OBJECT_TO_IDX

    carry = 0.0
    if o.carrying is not None:
        carry = float(
            OBJECT_TO_IDX.get(getattr(o.carrying, "type", str(o.carrying)), 1)
        )
    return np.concatenate(
        [np.asarray(o.image, np.float32).ravel(), [float(o.agent_dir), carry]]
    ).astype(np.float32)


def _blocks_obs(env):
    return np.asarray(env.get_last_obs().image, np.float32)


class UWEnv:
    """DreamerV3-torch env for one uncertain_worms env.  Not a gym.Env subclass:
    it exposes the minimal duck-typed API used by envs.wrappers / tools.simulate:
    observation_space, action_space, reset() -> obs_dict, step(action) -> (obs, r, done, info).
    """

    metadata = {}

    def __init__(self, name, seed=0):
        assert name in UW_ENVS, f"unknown uw env {name!r}; choices={list(UW_ENVS)}"
        self._name = name
        self._train_seed = int(seed)
        self._counter = 0
        self._eval_seed = None
        self.reward_range = [-np.inf, np.inf]

        kind, cfg = UW_ENVS[name]
        self._kind = kind
        if kind == "blocks":
            from uncertain_worms.environments.blocksworld.blocksworld_env import (
                BlocksEnvironment,
            )

            self._env = BlocksEnvironment(instance=cfg["instance"], max_steps=cfg["max_steps"])
            self._max_steps = cfg["max_steps"]
            self._dim = self._env.cap + 1
            self._n_actions = 4
            self._flat = lambda: _blocks_obs(self._env)
        elif kind == "minigrid":
            from uncertain_worms.environments.minigrid.minigrid_env import (
                MinigridEnvironment,
            )

            self._env = MinigridEnvironment(
                env_name=cfg["env_name"], fully_obs=False, max_steps=cfg["max_steps"]
            )
            self._max_steps = cfg["max_steps"]
            self._dim = 11
            self._n_actions = 6
            self._flat = lambda: _minigrid_flat(self._env.get_last_obs())
        if kind == "popgym":
            from popgym.envs import minesweeper as _ms

            self._env = getattr(_ms, cfg["env_name"])()
            self._dims = [int(d) for d in self._env.action_space.nvec]
            self._n_obs = cfg["n_obs"]
            self._n_actions = cfg["n_actions"]
            self._dim = cfg["n_obs"]
            self._max_steps = None          # POPGym sets its own episode length
            self._last_scalar = 0
            self._flat = lambda: _popgym_onehot(self._last_scalar, self._n_obs)
        self._steps = 0

    # ---- spaces -----------------------------------------------------------
    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "vector": gym.spaces.Box(-np.inf, np.inf, (self._dim,), dtype=np.float32),
                "image": gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._n_actions)
        space.discrete = True
        return space

    # ---- seeding ----------------------------------------------------------
    def set_eval_seed(self, s):
        self._eval_seed = s

    def _next_seed(self):
        if self._eval_seed is not None:
            return self._eval_seed
        seed = TRAIN_SEED_BASE + self._train_seed * 1_000_000 + self._counter
        self._counter += 1
        return seed

    # ---- obs dict ---------------------------------------------------------
    def _obs(self, is_first, is_last, is_terminal):
        return {
            "vector": self._flat(),
            "image": _DUMMY_IMAGE,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    # ---- API --------------------------------------------------------------
    def reset(self):
        if self._kind == "popgym":
            o, _ = self._env.reset(seed=self._next_seed())
            self._last_scalar = int(o)
            self._steps = 0
            return self._obs(True, False, False)
        self._env.reset(seed=self._next_seed())
        self._steps = 0
        return self._obs(True, False, False)

    def step(self, action):
        if self._kind == "popgym":
            a = int(action)
            x, y = divmod(a, self._dims[1])
            o, r, term, trunc, _info = self._env.step(np.array([x, y]))
            self._last_scalar = int(o)
            self._steps += 1
            done = bool(term or trunc)
            return self._obs(False, done, bool(term)), float(r), done, {}
        obs, state, r, term, trunc, info = self._env.step(int(action))
        self._steps += 1
        done = bool(term) or bool(trunc) or self._steps >= self._max_steps
        info = dict(info or {})
        info["is_terminal"] = bool(term)
        return self._obs(False, done, bool(term)), float(r), done, info

    def render(self, *args, **kwargs):
        return _DUMMY_IMAGE

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
