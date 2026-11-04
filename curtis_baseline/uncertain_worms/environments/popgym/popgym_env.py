from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from uncertain_worms.structs import ActType, Environment, Observation, State

try:
    from popgym.envs.minesweeper import MineSweeper as _BasePopGymMineSweeper
except ImportError:
    _BasePopGymMineSweeper = object  # type: ignore[assignment,misc]


HashableValue = Tuple[Any, ...] | int | float | str | bool | None


def freeze_value(value: Any) -> HashableValue:
    """Convert Gym/NumPy values into compact hashable Python values."""
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return (
            "ndarray",
            str(arr.dtype),
            tuple(int(x) if np.issubdtype(arr.dtype, np.integer) else float(x)
                  for x in arr.reshape(-1).tolist()),
            tuple(int(x) for x in arr.shape),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (repr(k), freeze_value(v))
                for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(freeze_value(v) for v in value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _as_numeric_array(value: Any) -> Optional[np.ndarray]:
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and value[0] == "ndarray"
        and isinstance(value[2], tuple)
    ):
        try:
            return np.asarray(value[2], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        try:
            return np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
    if isinstance(value, np.generic):
        return np.asarray([value.item()], dtype=float)
    if isinstance(value, (int, float, bool)):
        return np.asarray([value], dtype=float)
    if isinstance(value, (tuple, list)):
        try:
            return np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
    return None


def _strip_ndarray_wrapper(value: Any) -> Any:
    """Replace freeze_value's ndarray metadata tuples with plain value tuples.

    Converts ('ndarray', dtype, values_tuple, shape_tuple) → values_tuple,
    recursively. Used to produce readable repr strings for prompt display
    without affecting the stored frozen value.
    """
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and value[0] == "ndarray"
        and isinstance(value[2], tuple)
    ):
        return value[2]
    if isinstance(value, tuple):
        return tuple(_strip_ndarray_wrapper(v) for v in value)
    return value


def _unfreeze_ndarray(value: Any) -> Optional[np.ndarray]:
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and value[0] == "ndarray"
        and isinstance(value[2], tuple)
        and isinstance(value[3], tuple)
    ):
        try:
            return np.asarray(value[2], dtype=value[1]).reshape(value[3])
        except (TypeError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return None


def _cartpole_state_array(state: "PopGymState") -> np.ndarray:
    arr = _unfreeze_ndarray(state.value)
    if arr is None:
        raise ValueError(f"Expected PopGym CartPole ndarray state, got {state!r}")
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.shape[0] != 4:
        raise ValueError(f"Expected 4 CartPole state values, got {flat.shape[0]}")
    return flat


@dataclass(frozen=True)
class PopGymState(State):
    """Hidden Markov state exposed by PopGym's Markovian info wrapper."""

    value: HashableValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))

    @classmethod
    def from_raw(cls, value: Any) -> "PopGymState":
        return cls(freeze_value(value))

    def encode(self) -> "PopGymState":
        return self

    @classmethod
    def decode(cls, encoded: Any) -> "PopGymState":
        if isinstance(encoded, PopGymState):
            return cls.from_raw(encoded.value)
        if hasattr(encoded, "value"):
            return cls.from_raw(getattr(encoded, "value"))
        return cls.from_raw(encoded)

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"PopGymState(value={self.value!r})"


@dataclass(frozen=True)
class PopGymObservation(Observation):
    """Hashable observation wrapper for flattened/discrete PopGym observations."""

    value: HashableValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))

    @classmethod
    def from_raw(cls, value: Any) -> "PopGymObservation":
        return cls(freeze_value(value))

    def encode(self) -> "PopGymObservation":
        return self

    @classmethod
    def decode(cls, encoded: Any) -> "PopGymObservation":
        if isinstance(encoded, PopGymObservation):
            return cls.from_raw(encoded.value)
        if hasattr(encoded, "value"):
            return cls.from_raw(getattr(encoded, "value"))
        return cls.from_raw(encoded)

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"PopGymObservation(value={_strip_ndarray_wrapper(self.value)!r})"

    def distance(self, other: Observation) -> float:
        """Exact-match distance (0.0 or inf), same semantics as the base
        ``Observation.distance`` that curtis_baseline's particle-filter
        belief update calls. The only change from the inherited default is
        that both sides are canonicalized to a plain numeric array before
        comparing, so a real env observation (frozen from an np.ndarray,
        wrapped as ('ndarray', dtype, data, shape) by freeze_value) and an
        LLM-generated observation (a plain Python tuple, never wrapped by
        freeze_value since it isn't an np.ndarray instance) are compared on
        their actual values rather than on which serialization path
        produced them. Still strictly binary — no partial credit, no
        kernel weighting; use distance_soft for that."""
        if not isinstance(other, PopGymObservation):
            return float("inf")
        if self.value == other.value:
            return 0.0
        a = _as_numeric_array(self.value)
        b = _as_numeric_array(other.value)
        if a is None or b is None or a.shape != b.shape:
            return float("inf")
        return 0.0 if np.array_equal(a, b) else float("inf")

    def distance_soft(self, other: Observation) -> float:
        if not isinstance(other, PopGymObservation):
            return float("inf")
        if self.value == other.value:
            return 0.0

        a = _as_numeric_array(self.value)
        b = _as_numeric_array(other.value)
        if a is None or b is None or a.shape != b.shape:
            return -math.log(1e-6)
        return float(np.mean((a - b) ** 2))


def _load_popgym_env_class(env_id: str) -> type:
    try:
        import popgym
    except ImportError as exc:
        raise ImportError(
            "PopGymEnvironment requires the optional 'popgym' package. "
            "Install it in the active environment before running PopGym configs."
        ) from exc

    registry = getattr(popgym.envs, "ALL", {})
    if env_id in registry:
        value = registry[env_id]
        return value if isinstance(env_id, str) else env_id

    for key, value in registry.items():
        candidate = value if isinstance(key, str) else key
        if getattr(candidate, "__name__", "") == env_id:
            return candidate

    if "." in env_id:
        module_name, class_name = env_id.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    for module_name in (
        "repeat_previous",
        "repeat_first",
        "higher_lower",
        "multiarmed_bandit",
        "minesweeper",
    ):
        try:
            module = importlib.import_module(f"popgym.envs.{module_name}")
        except ImportError:
            continue
        if hasattr(module, env_id):
            return getattr(module, env_id)

    raise ValueError(
        f"Unknown PopGym env_id {env_id!r}. Use a key from popgym.envs.ALL "
        "or a full class path such as 'popgym.envs.repeat_previous.RepeatPreviousEasy'."
    )


def _state_from_info(info: Dict[str, Any], fallback_obs: Any) -> PopGymState:
    try:
        from popgym.core.observability import STATE
    except ImportError:
        STATE = "state"

    for key in (STATE, "state", "STATE"):
        if key in info:
            return PopGymState.from_raw(info[key])
    return PopGymState.from_raw(fallback_obs)


def _apply_wrappers(env: Any, wrappers: Sequence[str]) -> Any:
    if not wrappers:
        return env

    from popgym.core.observability import Observability
    from popgym.wrappers import Antialias, DiscreteAction, Flatten, Markovian, PreviousAction

    wrapper_map = {
        "previous_action": lambda e: PreviousAction(e),
        "antialias": lambda e: Antialias(e),
        "markovian_info": lambda e: Markovian(e, Observability.FULL_IN_INFO_DICT),
        "flatten": lambda e: Flatten(e),
        "discrete_action": lambda e: DiscreteAction(e),
    }
    for name in wrappers:
        key = name.lower()
        if key not in wrapper_map:
            raise ValueError(
                f"Unsupported PopGym wrapper {name!r}. "
                f"Supported wrappers: {sorted(wrapper_map)}"
            )
        env = wrapper_map[key](env)
    return env


class PopGymEnvironment(Environment):
    """Adapter from PopGym's Gymnasium API to uncertain_worms.Environment."""

    def __init__(
        self,
        *args: Any,
        env_id: str = "RepeatPreviousEasy",
        env_kwargs: Optional[Dict[str, Any]] = None,
        wrappers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        env_cls = _load_popgym_env_class(env_id)
        self.env_id = env_id
        self.env = env_cls(**(env_kwargs or {}))
        self.env = _apply_wrappers(
            self.env,
            wrappers or ["previous_action", "antialias", "markovian_info", "flatten", "discrete_action"],
        )
        self._last_obs: Optional[PopGymObservation] = None
        self._last_state: Optional[PopGymState] = None
        self._step_count = 0

    @property
    def action_space(self) -> spaces.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space

    def get_last_obs(self) -> PopGymObservation:
        if self._last_obs is None:
            raise RuntimeError("No observation available yet. Call reset() first.")
        return self._last_obs

    def step(
        self, action: ActType
    ) -> Tuple[Observation, State, float, bool, bool, Dict[str, Any]]:
        obs_raw, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        obs = PopGymObservation.from_raw(obs_raw)
        state = _state_from_info(info, fallback_obs=obs_raw)
        self._last_obs = obs
        self._last_state = state
        if self.max_steps > 0 and self._step_count >= self.max_steps:
            truncated = True
        return obs, state, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: int = 0) -> State:
        obs_raw, info = self.env.reset(seed=seed)
        self._step_count = 0
        obs = PopGymObservation.from_raw(obs_raw)
        state = _state_from_info(info, fallback_obs=obs_raw)
        self._last_obs = obs
        self._last_state = state
        return state


def popgym_empty_state_gen() -> PopGymState:
    return PopGymState.from_raw(None)


def popgym_empty_observation_gen() -> PopGymObservation:
    return PopGymObservation.from_raw(None)


def popgym_cartpole_initial_model_gen() -> Any:
    """Ground-truth-ish initial model for POPGym PositionOnlyCartPoleEasy."""

    def initial_model(_: PopGymState) -> PopGymState:
        return PopGymState.from_raw(np.random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32))

    return initial_model


def popgym_cartpole_transition_model_gen() -> Any:
    """Gymnasium CartPole dynamics used by POPGym's position-only variant."""
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02

    def transition_model(state: PopGymState, action: ActType) -> PopGymState:
        x, x_dot, theta, theta_dot = [float(v) for v in _cartpole_state_array(state)]
        force = force_mag if int(action) == 1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        next_state = np.asarray(
            [
                x + tau * x_dot,
                x_dot + tau * xacc,
                theta + tau * theta_dot,
                theta_dot + tau * thetaacc,
            ],
            dtype=np.float32,
        )
        return PopGymState.from_raw(next_state)

    return transition_model


def popgym_cartpole_reward_model_gen() -> Any:
    """Survival reward and termination rule for PositionOnlyCartPoleEasy."""
    x_threshold = 2.4
    theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
    survival_reward = 0.005

    def reward_model(
        state: PopGymState, action: ActType, next_state: PopGymState
    ) -> Tuple[float, bool]:
        del state, action
        x, _, theta, _ = [float(v) for v in _cartpole_state_array(next_state)]
        terminated = bool(
            x < -x_threshold
            or x > x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )
        return survival_reward, terminated

    return reward_model


def popgym_cartpole_shaped_reward_model_gen() -> Any:
    """Dense planning reward for CartPole while preserving true termination.

    This is intended for GT-planning diagnostics only. Evaluation still uses
    the real POPGym reward emitted by the environment.
    """
    x_threshold = 2.4
    theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
    survival_reward = 0.005

    def reward_model(
        state: PopGymState, action: ActType, next_state: PopGymState
    ) -> Tuple[float, bool]:
        del state, action
        x, x_dot, theta, theta_dot = [float(v) for v in _cartpole_state_array(next_state)]
        terminated = bool(
            x < -x_threshold
            or x > x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )
        # Keep magnitudes near the real per-step reward scale, but make
        # unstable continuous states visibly worse before terminal failure.
        stability_penalty = (
            0.020 * abs(theta)
            + 0.002 * abs(theta_dot)
            + 0.001 * abs(x)
            + 0.001 * abs(x_dot)
        )
        terminal_penalty = 1.0 if terminated else 0.0
        return survival_reward - stability_penalty - terminal_penalty, terminated

    return reward_model


def popgym_cartpole_observation_model_gen() -> Any:
    """Flattened PositionOnlyCartPole observation under current wrappers."""

    def observation_model(
        state: PopGymState, action: Optional[ActType], empty_obs: PopGymObservation
    ) -> PopGymObservation:
        del empty_obs
        x, _, theta, _ = [float(v) for v in _cartpole_state_array(state)]
        if action is None:
            previous_action = [1.0, 0.0]
            antialias = [0.0, 1.0]
        else:
            previous_action = [1.0, 0.0] if int(action) == 0 else [0.0, 1.0]
            antialias = [1.0, 0.0]
        obs = np.asarray([x, theta, *previous_action, *antialias], dtype=np.float64)
        return PopGymObservation.from_raw(obs)

    return observation_model


def _battleship_state_parts(state: "PopGymState") -> Tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(state.value, tuple) or len(state.value) != 3:
        raise ValueError(f"Expected Battleship state tuple, got {state!r}")
    board = _unfreeze_ndarray(state.value[0])
    guesses = _unfreeze_ndarray(state.value[1])
    if board is None or guesses is None:
        raise ValueError(f"Expected Battleship ndarray state parts, got {state!r}")
    return (
        np.asarray(board, dtype=np.int8).reshape(8, 8),
        np.asarray(guesses, dtype=np.int8).reshape(8, 8),
        int(state.value[2]),
    )


def _battleship_place_ship(
    board: np.ndarray, ship_size: int, rng: Any
) -> np.ndarray:
    while True:
        if hasattr(rng, "integers"):
            idx_start = rng.integers(8, size=2)
            direction = rng.integers(2, size=2)
        else:
            idx_start = rng.randint(8, size=2)
            direction = rng.randint(2, size=2)
        idx_end = idx_start.copy()
        idx_end[direction[0]] = idx_start[direction[0]] + (-1) ** int(
            direction[1]
        ) * (ship_size - 1)
        if np.any(idx_end < 0) or np.any(idx_end >= 8):
            continue
        start = np.minimum(idx_start, idx_end)
        end = np.maximum(idx_start, idx_end)
        if np.sum(board[start[0] : end[0] + 1, start[1] : end[1] + 1]) == 0:
            board[start[0] : end[0] + 1, start[1] : end[1] + 1] = 1
            return board


def popgym_battleship_initial_model_gen() -> Any:
    """Ground-truth initial model for POPGym BattleshipEasy."""
    ship_sizes = [2, 3, 3, 4]

    def initial_model(_: PopGymState) -> PopGymState:
        rng = np.random
        board = np.zeros((8, 8), dtype=np.int8)
        for ship_size in ship_sizes:
            board = _battleship_place_ship(board, ship_size, rng)
        guesses = np.zeros((8, 8), dtype=np.int8)
        return PopGymState.from_raw((board, guesses, 0))

    return initial_model


def popgym_battleship_transition_model_gen() -> Any:
    """Ground-truth transition model for POPGym BattleshipEasy."""

    def transition_model(state: PopGymState, action: ActType) -> PopGymState:
        board, guesses, _ = _battleship_state_parts(state)
        row = int(action) // 8
        col = int(action) % 8
        next_guesses = guesses.copy()
        is_ship = bool(board[row, col])
        guessed_before = bool(next_guesses[row, col])
        next_guesses[row, col] = 1
        hit = int(is_ship and not guessed_before)
        return PopGymState.from_raw((board.copy(), next_guesses, hit))

    return transition_model


def popgym_battleship_reward_model_gen() -> Any:
    """Ground-truth reward and termination model for POPGym BattleshipEasy."""
    needed_hits = 12
    max_episode_length = 64
    hit_reward = 1.0 / needed_hits
    miss_reward = -1.0 / (max_episode_length - needed_hits)

    def reward_model(
        state: PopGymState, action: ActType, next_state: PopGymState
    ) -> Tuple[float, bool]:
        del state, action
        board, guesses, last_obs = _battleship_state_parts(next_state)
        hits = int(np.sum((board == 1) & (guesses == 1)))
        terminated = hits == needed_hits
        reward = hit_reward if int(last_obs) == 1 else miss_reward
        return reward, terminated

    return reward_model


def popgym_battleship_observation_model_gen() -> Any:
    """Flattened BattleshipEasy observation under current wrappers."""

    def observation_model(
        state: PopGymState, action: Optional[ActType], empty_obs: PopGymObservation
    ) -> PopGymObservation:
        del empty_obs
        _, _, last_obs = _battleship_state_parts(state)
        if action is None:
            row = 0
            col = 0
            antialias = [0, 1]
        else:
            row = int(action) // 8
            col = int(action) % 8
            antialias = [1, 0]

        hit_obs = [1, 0] if int(last_obs) == 0 else [0, 1]
        row_one_hot = [1 if i == row else 0 for i in range(8)]
        col_one_hot = [1 if i == col else 0 for i in range(8)]
        obs = np.asarray(
            [*hit_obs, *row_one_hot, *col_one_hot, *antialias],
            dtype=np.int64,
        )
        return PopGymObservation.from_raw(obs)

    return observation_model


def _minesweeper_params(difficulty: str = "easy") -> Tuple[Tuple[int, int], int]:
    difficulty = difficulty.lower()
    if difficulty == "easy":
        return (4, 4), 2
    if difficulty == "medium":
        return (6, 6), 6
    if difficulty == "hard":
        return (8, 8), 10
    raise ValueError(f"Unknown MineSweeper difficulty {difficulty!r}")


def _minesweeper_state_grid(
    state: "PopGymState", dims: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    arr = _unfreeze_ndarray(state.value)
    if arr is None:
        arr = _as_numeric_array(state.value)
    if arr is None:
        raise ValueError(f"Expected MineSweeper ndarray-like state, got {state!r}")
    flat = np.asarray(arr, dtype=np.int8).reshape(-1)
    expected = int(np.prod(dims))
    if flat.shape[0] != expected:
        raise ValueError(
            f"Expected MineSweeper state with {expected} cells, got {flat.shape[0]}"
        )
    return flat.reshape(dims)


def _minesweeper_action_to_row_col(
    action: ActType, dims: Tuple[int, int] = (4, 4)
) -> Tuple[int, int]:
    row_count, col_count = dims
    action_int = int(action)
    if action_int < 0 or action_int >= row_count * col_count:
        raise ValueError(
            f"MineSweeper action {action_int} outside [0, {row_count * col_count})"
        )
    return action_int // col_count, action_int % col_count


def _minesweeper_neighbor_count(grid: np.ndarray, row: int, col: int) -> int:
    mine_grid = (np.asarray(grid, dtype=np.int8) == 1).astype(np.int8)
    r0 = max(0, row - 1)
    r1 = min(mine_grid.shape[0], row + 2)
    c0 = max(0, col - 1)
    c1 = min(mine_grid.shape[1], col + 2)
    return int(np.sum(mine_grid[r0:r1, c0:c1]))


def _minesweeper_observation_from_parts(
    neighbor_count: int,
    row: int,
    col: int,
    *,
    is_t0: bool,
    dims: Tuple[int, int],
    num_mines: int,
) -> PopGymObservation:
    row_count, col_count = dims
    obs_size = min(num_mines + 1, 10)
    clipped_count = min(max(int(neighbor_count), 0), obs_size - 1)
    count_one_hot = [1 if i == clipped_count else 0 for i in range(obs_size)]
    row_one_hot = [1 if i == row else 0 for i in range(row_count)]
    col_one_hot = [1 if i == col else 0 for i in range(col_count)]
    antialias = [0, 1] if is_t0 else [1, 0]
    obs = np.asarray(
        [*count_one_hot, *row_one_hot, *col_one_hot, *antialias],
        dtype=np.int64,
    )
    return PopGymObservation.from_raw(obs)


def popgym_minesweeper_initial_model_gen(difficulty: str = "easy") -> Any:
    """Ground-truth initial model for POPGym MineSweeper variants.

    POPGym encodes the hidden grid as 0=CLEAR, 1=MINE, 2=VIEWED. The
    Markovian info state does not include timestep, so the GT state here
    mirrors the state exposed by the environment.
    """
    dims, num_mines = _minesweeper_params(difficulty)

    def initial_model(_: PopGymState) -> PopGymState:
        flat = np.zeros(int(np.prod(dims)), dtype=np.int8)
        mine_indices = np.random.choice(flat.size, size=num_mines, replace=False)
        flat[mine_indices] = 1
        return PopGymState.from_raw(flat)

    return initial_model


def popgym_minesweeper_transition_model_gen(difficulty: str = "easy") -> Any:
    """Ground-truth transition model for POPGym MineSweeper variants."""
    dims, _ = _minesweeper_params(difficulty)

    def transition_model(state: PopGymState, action: ActType) -> PopGymState:
        grid = _minesweeper_state_grid(state, dims).copy()
        row, col = _minesweeper_action_to_row_col(action, dims)
        if int(grid[row, col]) == 0:
            grid[row, col] = 2
        return PopGymState.from_raw(grid.reshape(-1))

    return transition_model


def popgym_minesweeper_reward_model_gen(difficulty: str = "easy") -> Any:
    """Ground-truth reward and task-termination model for MineSweeper variants."""
    dims, num_mines = _minesweeper_params(difficulty)
    max_episode_length = int(np.prod(dims)) - num_mines
    success_reward = 1.0 / max_episode_length
    fail_reward = -0.5 - success_reward
    bad_action_reward = -0.5 / (max_episode_length - 2)

    def reward_model(
        state: PopGymState, action: ActType, next_state: PopGymState
    ) -> Tuple[float, bool]:
        grid = _minesweeper_state_grid(state, dims)
        next_grid = _minesweeper_state_grid(next_state, dims)
        row, col = _minesweeper_action_to_row_col(action, dims)

        if int(grid[row, col]) == 1:
            return fail_reward, True
        if int(grid[row, col]) == 2:
            return bad_action_reward, False

        terminated = bool(np.all(next_grid != 0))
        return success_reward, terminated

    return reward_model


def popgym_minesweeper_observation_model_gen(difficulty: str = "easy") -> Any:
    """Flattened MineSweeper observation under the standard wrappers."""
    dims, num_mines = _minesweeper_params(difficulty)

    def observation_model(
        state: PopGymState, action: Optional[ActType], empty_obs: PopGymObservation
    ) -> PopGymObservation:
        del empty_obs
        if action is None:
            return _minesweeper_observation_from_parts(
                0, 0, 0, is_t0=True, dims=dims, num_mines=num_mines
            )

        grid = _minesweeper_state_grid(state, dims)
        row, col = _minesweeper_action_to_row_col(action, dims)
        neighbor_count = _minesweeper_neighbor_count(grid, row, col)
        return _minesweeper_observation_from_parts(
            neighbor_count,
            row,
            col,
            is_t0=False,
            dims=dims,
            num_mines=num_mines,
        )

    return observation_model


def popgym_velocity_cartpole_observation_model_gen() -> Any:
    """Flattened VelocityOnlyCartPole observation under current wrappers.

    POPGym's VelocityOnlyCartPole emits [x, theta] at reset and
    [x_dot, theta_dot] after environment steps. The PreviousAction and
    Antialias wrappers then append two one-hot fields each.
    """

    def observation_model(
        state: PopGymState, action: Optional[ActType], empty_obs: PopGymObservation
    ) -> PopGymObservation:
        del empty_obs
        x, x_dot, theta, theta_dot = [float(v) for v in _cartpole_state_array(state)]
        if action is None:
            core_obs = [x, theta]
            previous_action = [1.0, 0.0]
            antialias = [0.0, 1.0]
        else:
            core_obs = [x_dot, theta_dot]
            previous_action = [1.0, 0.0] if int(action) == 0 else [0.0, 1.0]
            antialias = [1.0, 0.0]
        obs = np.asarray([*core_obs, *previous_action, *antialias], dtype=np.float32)
        return PopGymObservation.from_raw(obs)


# ---------------------------------------------------------------------------
# MineSweeperGridObs: natural POMDP observation (full visible grid)
# ---------------------------------------------------------------------------

class MineSweeperGridObs(_BasePopGymMineSweeper):
    """MineSweeper where the observation is the full visible grid, not just the last click.

    Each element of the flat observation array is:
      -1   — cell is still hidden (agent does not know if it is CLEAR or MINE)
      0..N — cell has been revealed; value is the count of adjacent mines

    State encoding (from the base env) is unchanged: 0=CLEAR, 1=MINE, 2=VIEWED.
    Game rules, rewards, and actions are identical to the original.
    """

    def __init__(self, difficulty: str = "easy") -> None:
        super().__init__(difficulty)
        n_cells = int(np.prod(self.dims))
        self.observation_space = spaces.Box(
            low=np.full(n_cells, -1, dtype=np.int8),
            high=np.full(n_cells, self.num_mines, dtype=np.int8),
            dtype=np.int8,
        )
        self._visible: np.ndarray = np.full(n_cells, np.int8(-1), dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self._visible = np.full(int(np.prod(self.dims)), np.int8(-1), dtype=np.int8)
        return self._visible.copy(), info

    def step(self, action):
        obs_single, reward, terminated, truncated, info = super().step(action)
        a = np.asarray(action).flatten()
        row, col = int(a[0]), int(a[1])
        self._visible[row * self.dims[1] + col] = np.int8(int(obs_single))
        return self._visible.copy(), reward, terminated, truncated, info


class MineSweeperGridObsEasy(MineSweeperGridObs):
    def __init__(self) -> None:
        super().__init__("easy")


class MineSweeperGridObsMedium(MineSweeperGridObs):
    def __init__(self) -> None:
        super().__init__("medium")


class MineSweeperGridObsHard(MineSweeperGridObs):
    def __init__(self) -> None:
        super().__init__("hard")


def popgym_minesweeper_grid_obs_observation_model_gen(difficulty: str = "easy") -> Any:
    """GT observation model for MineSweeperGridObs.

    Reconstructs the full visible grid from the hidden state: VIEWED cells expose
    their neighbor mine count; CLEAR and MINE cells remain hidden (-1).
    """
    dims, _ = _minesweeper_params(difficulty)
    n_cells = int(np.prod(dims))

    def observation_model(
        state: PopGymState, action: Optional[ActType], empty_obs: PopGymObservation
    ) -> PopGymObservation:
        del action, empty_obs
        if state.value is None:
            return PopGymObservation.from_raw(np.full(n_cells, np.int8(-1), dtype=np.int8))
        grid = _minesweeper_state_grid(state, dims)
        visible = np.full(n_cells, np.int8(-1), dtype=np.int8)
        for i in range(n_cells):
            row, col = i // dims[1], i % dims[1]
            if int(grid[row, col]) == 2:  # VIEWED
                visible[i] = np.int8(_minesweeper_neighbor_count(grid, row, col))
        return PopGymObservation.from_raw(visible)

    return observation_model

    return observation_model
