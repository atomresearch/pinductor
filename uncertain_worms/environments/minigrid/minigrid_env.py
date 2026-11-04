"""MiniGrid navigation domains: states, observations, models, rendering.

This module adapts the MiniGrid gridworlds to the interfaces in
:mod:`uncertain_worms.structs`, and contains four things.

**The representations.** :class:`MinigridState` is the full world — grid,
agent pose, carried object — and is what the environment and the logs
manipulate. :class:`MinigridObservation` is what an agent is allowed to
perceive: a small field of view in front of it, the direction it faces
and what it holds. Nothing else. The gap between the two is the problem
this repository studies, so the boundary is enforced by the observation
class itself rather than by convention.

**The reference models.** The ``minigrid_hardcoded_*_model_gen``
functions are hand-written, correct implementations of the four POMDP
components for this domain. They serve as the ground truth a generated
model is compared against, and as the models used when a method is run
with its dynamics given rather than learned.

**The environment.** :class:`MinigridEnvironment` wraps a gym MiniGrid
instance and returns both the observation and the true state; the state
goes to the logs and the replay buffer, never to a partially observable
agent.

**Rendering.** The remaining helpers reconstruct a map from a sequence of
partial views and turn it into a GIF. They exist for inspection only and
run outside the evaluation path.

Several code paths here are optimised versions of an obvious
implementation, kept because they sit in the inner loop of the particle
filter, which calls them millions of times per run. Each carries the
plain formulation it must stay identical to.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import minigrid.core.world_object as world_object  # type: ignore
import numpy as np
from gymnasium import Env
from minigrid.core.constants import IDX_TO_OBJECT  # type: ignore
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid  # type: ignore
from minigrid.minigrid_env import MiniGridEnv  # type: ignore
from minigrid.utils.rendering import downsample  # type: ignore
from minigrid.utils.rendering import fill_coords  # type: ignore
from minigrid.utils.rendering import highlight_img  # type: ignore
from minigrid.utils.rendering import point_in_rect  # type: ignore
from minigrid.utils.rendering import point_in_triangle  # type: ignore
from minigrid.utils.rendering import rotate_fn  # type: ignore
from minigrid.wrappers import FullyObsWrapper  # type: ignore
from minigrid.wrappers import ViewSizeWrapper  # type: ignore
from numpy.typing import NDArray

from uncertain_worms.structs import (
    ActType,
    Environment,
    Heuristic,
    InitialModel,
    Observation,
    ObservationModel,
    Optional,
    RewardModel,
    State,
    TransitionModel,
)
from uncertain_worms.utils import get_log_dir


class AgentPositionWrapper(ViewSizeWrapper):
    """Adds the agent's absolute position to the gym observation dict.

    This is how the environment recovers the true state for logging and
    for the replay buffer; the value is stripped before anything reaches
    an agent (see :func:`build_obs`).
    """

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {**obs, "agent_pos": self.unwrapped.agent_pos}


class GridWrapper(AgentPositionWrapper):
    """Adds the complete grid to the gym observation dict.

    Same purpose as the position above: :func:`build_state` needs the
    whole world to reconstruct a :class:`MinigridState`, which only the
    evaluation loop is given.
    """

    def __init__(self, env: Env, width: int, height: int):
        self.width, self.height = width, height
        super().__init__(env)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **obs,
            "agent_pos": self.unwrapped.agent_pos,
            "grid": self.env.unwrapped.grid,
        }


log = logging.getLogger(__name__)
AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

# Occlusion switch, applied both to the gym environment and to the field
# of view this module computes. With it on, a cell hidden behind a wall is
# still reported, so partial observability comes from the limited extent
# of the view rather than from line of sight.
SEE_THROUGH_WALLS = True

# Weights of the non-image fields when two observations are compared (see
# `MinigridObservation.distance_soft`). They are read from the
# environment at import time, so a run can be re-scored under different
# weights without touching a configuration file. The reward and
# terminated weights below are left over from a version in which those
# were observation fields; nothing reads them today.
OBS_DIR_WEIGHT = float(os.environ.get("OBS_DIR_WEIGHT", "1.0"))
OBS_REWARD_WEIGHT = float(os.environ.get("OBS_REWARD_WEIGHT", "0.0"))
OBS_TERMINATED_WEIGHT = float(os.environ.get("OBS_TERMINATED_WEIGHT", "0.0"))

# Units in which a field mismatch is charged.
#
# ``"arbitrary"`` (the default) charges the raw weight, which mixes units:
# the image term of the distance is a log-likelihood in nats, the field
# penalties are not, and the two are added together.
#
# ``"nats"`` charges ``-log(eps)`` instead, that is the negative log
# probability of observing a mismatched field under an error rate of
# ``eps``, which puts the whole distance in nats and makes the weights
# dimensionless multipliers.
OBS_DIST_MODE = os.environ.get("OBS_DIST_MODE", "arbitrary").lower()
OBS_DIR_EPS = float(os.environ.get("OBS_DIR_EPS", "0.01"))
OBS_TERM_EPS = float(os.environ.get("OBS_TERM_EPS", "0.01"))
OBS_REWARD_SIGMA = float(os.environ.get("OBS_REWARD_SIGMA", "0.5"))

# Optional per-field weighting, not shipped here: without the module the
# comparison falls back to the flat weights above. The import is resolved
# once at import time rather than inside `distance_soft`, because that
# function runs millions of times per evaluation and a repeated module
# lookup there was measured to cost several percent of a run.
try:
    from particle_filtering.field_weights import (
        get_active_adapter as _get_active_adapter,
    )
    _HAS_FIELD_WEIGHTS = True
except ImportError:
    _get_active_adapter = None
    _HAS_FIELD_WEIGHTS = False


# Compiled versions of the two inner-loop routines: the field-of-view
# computation and the batched image distance. They are defined only when
# numba is installed; both have a pure-Python equivalent below that must
# produce identical results, and the module works either way. Numba is
# therefore a speed dependency, never a correctness one.
try:
    import numba as _numba_mod  # type: ignore

    @_numba_mod.njit(cache=True)
    def _fov_compute_numba(
        grid: NDArray[np.int8],
        ax: int,
        ay: int,
        agent_dir: int,
        view_size: int,
        wall: np.int8,
        agent_value: np.int8,
    ) -> NDArray[np.int8]:
        """Field of view in front of the agent, in compiled int8 code.

        Must stay bit-identical to the plain implementation in
        :meth:`MinigridState.get_field_of_view`, which it replaces when
        numba is available: cells outside the grid read as wall, the view
        is rotated so that the agent always looks "up", and the agent
        marker is written at the fixed bottom-centre slot.
        """
        W = grid.shape[0]
        H = grid.shape[1]
        # Compute top-left of overlap window in grid coords
        if agent_dir == 0:        # right
            topX = ax
            topY = ay - view_size // 2
        elif agent_dir == 1:      # down
            topX = ax - view_size // 2
            topY = ay
        elif agent_dir == 2:      # left
            topX = ax - view_size + 1
            topY = ay - view_size // 2
        else:                     # up (3)
            topX = ax - view_size // 2
            topY = ay - view_size + 1

        fov = np.empty((view_size, view_size), dtype=np.int8)
        for i in range(view_size):
            for j in range(view_size):
                fov[i, j] = wall

        gx0 = topX if topX > 0 else 0
        gy0 = topY if topY > 0 else 0
        gx1 = (topX + view_size) if (topX + view_size) < W else W
        gy1 = (topY + view_size) if (topY + view_size) < H else H
        px0 = -topX if topX < 0 else 0
        py0 = -topY if topY < 0 else 0

        for x in range(gx1 - gx0):
            for y in range(gy1 - gy0):
                fov[px0 + x, py0 + y] = grid[gx0 + x, gy0 + y]

        # Same rotation as the plain path, k = -(agent_dir+1) mod 4, but
        # written into a fresh contiguous array: numba will not let a
        # strided view be reassigned afterwards.
        n_rot = (-(agent_dir + 1)) % 4
        if n_rot != 0:
            rotated = np.empty((view_size, view_size), dtype=np.int8)
            if n_rot == 1:
                # fov.T[::-1]: rotated[i,j] = fov[j, view_size-1-i]
                for i in range(view_size):
                    for j in range(view_size):
                        rotated[i, j] = fov[j, view_size - 1 - i]
            elif n_rot == 2:
                # fov[::-1, ::-1]
                for i in range(view_size):
                    for j in range(view_size):
                        rotated[i, j] = fov[view_size - 1 - i, view_size - 1 - j]
            else:  # n_rot == 3
                # fov[::-1].T: rotated[i,j] = fov[view_size-1-j, i]
                for i in range(view_size):
                    for j in range(view_size):
                        rotated[i, j] = fov[view_size - 1 - j, i]
            fov = rotated

        # Agent placement at the canonical centre-bottom slot
        apos_x = view_size // 2
        apos_y = view_size - 1
        fov[apos_x, apos_y] = agent_value
        return fov

    @_numba_mod.njit(fastmath=True, cache=True)
    def _distance_soft_batch_image_numba(
        predicted: NDArray[np.int8],
        actual: NDArray[np.int8],
        mask: np.int8,
    ) -> NDArray[np.float64]:
        """K predicted (K, n) vs 1 actual (n,) → (K,) distances.

        Image-only base distance (matches the parent
        ``Observation.distance_soft`` formula bit-for-bit). The caller
        adds agent_dir / carrying penalties separately.
        """
        K, n = predicted.shape
        distances = np.empty(K, dtype=np.float64)
        for k in range(K):
            total = 0
            matches = 0
            for i in range(n):
                v = predicted[k, i]
                if v != mask:
                    total += 1
                    if v == actual[i]:
                        matches += 1
            if total == 0:
                distances[k] = 0.0
            elif matches == 0:
                distances[k] = np.inf
            else:
                r = matches / total
                distances[k] = -math.log(1e-6 + (1.0 - 1e-6) * r * r * r)
        return distances

    _HAS_NUMBA = True
except ImportError:
    _distance_soft_batch_image_numba = None
    _HAS_NUMBA = False


def distance_soft_batch_image(
    predicted_images: NDArray[np.int8],
    actual_image: NDArray[np.int8],
    mask: int = 12,
) -> NDArray[np.float64]:
    """Compute the image-only ``distance_soft`` for K predicted images
    against a single actual image — vectorised over K.

    Args:
        predicted_images: shape (K, H, W) or (K, H*W) int8.
        actual_image:     shape (H, W) or (H*W,) int8.
        mask:             cell value to ignore (default 12 = agent cell).

    Returns:
        ``np.ndarray`` of shape (K,) float64. Identical (bit-equal) to
        applying the per-particle :meth:`MinigridObservation.distance_soft`
        image branch K times. Field penalties (agent_dir, carrying)
        are *not* included — callers add those in a per-particle loop
        because they are cheap and only need scalar comparisons.

    The numba kernel is compiled on first call (~430ms one-time JIT)
    and the result is disk-cached. Falls back to a pure-Python loop
    when numba is not installed (zero behaviour change, slower).
    """
    if predicted_images.ndim == 3:
        K = predicted_images.shape[0]
        pred_flat = np.ascontiguousarray(predicted_images.reshape(K, -1))
    else:
        pred_flat = np.ascontiguousarray(predicted_images)
    actual_flat = np.ascontiguousarray(actual_image.ravel())
    mask_i8 = np.int8(mask)

    if _HAS_NUMBA:
        return _distance_soft_batch_image_numba(pred_flat, actual_flat, mask_i8)

    # Fallback : pure Python loop (slow, but correct)
    K, n = pred_flat.shape
    distances = np.empty(K, dtype=np.float64)
    for k in range(K):
        total = 0
        matches = 0
        for i in range(n):
            v = int(pred_flat[k, i])
            if v != mask:
                total += 1
                if v == int(actual_flat[i]):
                    matches += 1
        if total == 0:
            distances[k] = 0.0
        elif matches == 0:
            distances[k] = math.inf
        else:
            r = matches / total
            distances[k] = -math.log(1e-6 + (1.0 - 1e-6) * r * r * r)
    return distances


class ObjectTypes(IntEnum):
    """Cell values used in state grids and observation images.

    This encoding is local to the repository, not MiniGrid's own: doors
    are split by their state so that a single integer per cell describes
    the world completely, which is what lets a grid be a plain int8 array
    and a state be hashable. :func:`minigrid_to_local` and
    :func:`local_to_minigrid` convert to and from MiniGrid objects.

    ``agent`` marks the agent's own cell in a field of view. It sits at a
    fixed slot and is excluded from observation comparisons.
    """

    unseen = 0
    empty = 1
    wall = 2
    open_door = 4
    closed_door = 5
    locked_door = 6
    key = 7
    ball = 8
    box = 9
    goal = 10
    lava = 11
    agent = 12


class Direction(IntEnum):
    """Agent headings, in the order :data:`DIR_TO_VEC` indexes."""

    facing_right = 0
    facing_down = 1
    facing_left = 2
    facing_up = 3


class Actions(IntEnum):
    """Action codes, in MiniGrid's own order.

    The integers are passed straight through to the gym environment, so
    the order here has to match MiniGrid's and cannot be rearranged.
    """

    left = 0  # Turn left
    right = 1  # Turn right
    forward = 2  # Move forward
    pickup = 3  # Pick up an object
    drop = 4  # Drop an object
    toggle = 5  # Toggle/activate an object
    done = 6  # Done completing the task


# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_COLOR_MAP = {
    ObjectTypes.wall: COLOR_TO_IDX["grey"],
    ObjectTypes.goal: COLOR_TO_IDX["green"],
    ObjectTypes.key: COLOR_TO_IDX["blue"],
    ObjectTypes.ball: COLOR_TO_IDX["blue"],
    ObjectTypes.lava: COLOR_TO_IDX["blue"],
    ObjectTypes.locked_door: COLOR_TO_IDX["blue"],
    ObjectTypes.open_door: COLOR_TO_IDX["blue"],
    ObjectTypes.closed_door: COLOR_TO_IDX["blue"],
}


def minigrid_to_local(obj: world_object.WorldObj) -> int:
    """Map a MiniGrid object to its :class:`ObjectTypes` code.

    An empty cell is ``None`` in MiniGrid and becomes ``empty`` here.
    Doors map to three different codes depending on whether they are
    open, closed or locked. Unknown object types raise rather than being
    silently encoded as something else.
    """
    if obj is None:
        return ObjectTypes.empty
    elif obj.type == "goal":
        return ObjectTypes.goal
    elif obj.type == "wall":
        return ObjectTypes.wall
    elif obj.type == "key":
        return ObjectTypes.key
    elif obj.type == "door" and obj.is_open:
        return ObjectTypes.open_door
    elif obj.type == "door" and (not obj.is_open) and (not obj.is_locked):
        return ObjectTypes.closed_door
    elif obj.type == "door" and (not obj.is_open) and (obj.is_locked):
        return ObjectTypes.locked_door
    elif obj.type == "box":
        return ObjectTypes.box
    elif obj.type == "ball":
        return ObjectTypes.ball
    elif obj.type == "lava":
        return ObjectTypes.lava
    else:
        raise TypeError(f"Unknown object {obj.type}")


def local_to_minigrid(lobj: int) -> world_object.WorldObj:
    """Inverse of :func:`minigrid_to_local`, used to render a grid.

    Colours come from :data:`IDX_COLOR_MAP`: the local encoding drops
    colour, so it has to be reintroduced from the object type. Codes this
    function does not handle, the agent marker among them, come back as
    ``None``.
    """
    if lobj == ObjectTypes.empty or lobj == ObjectTypes.unseen:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["empty"], color_idx=0, state=0
        )
    elif lobj == ObjectTypes.goal:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["goal"], color_idx=0, state=0
        )
    elif lobj == ObjectTypes.wall:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["wall"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.wall],
            state=0,
        )
    elif lobj == ObjectTypes.key:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["key"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.key],
            state=0,
        )
    elif lobj == ObjectTypes.open_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.open_door],
            state=0,
        )
    elif lobj == ObjectTypes.closed_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.closed_door],
            state=1,
        )
    elif lobj == ObjectTypes.locked_door:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["door"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.locked_door],
            state=2,
        )
    elif lobj == ObjectTypes.ball:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["ball"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.ball],
            state=0,
        )
    elif lobj == ObjectTypes.lava:
        return world_object.WorldObj.decode(
            type_idx=OBJECT_TO_IDX["lava"],
            color_idx=IDX_COLOR_MAP[ObjectTypes.lava],
            state=0,
        )

    return None


@dataclass
class MinigridObservation(Observation):
    """POMDP-clean Minigrid observation — the three perceptual fields the
    agent is entitled to read each step:

    - ``image``: 3×3 FOV grid in front of the agent (minigrid FOV, with
      the agent cell at the fixed slot).
    - ``agent_dir``: compass — the direction the agent is facing
      (0=right, 1=down, 2=left, 3=up). Proprioception.
    - ``carrying``: object the agent is currently holding
      (``None`` if empty). Proprioception.

    ``reward`` and ``terminated`` are deliberately NOT fields here:
    they are *outcomes of the transition* computed by ``reward_func``,
    not perceptual signals that ``obs_func`` should be asked to
    reproduce. The agent still sees them per step (they appear in the
    OBSERVED DATA block of the prompt, supplied by the buffer) and the
    scorer compares ``reward_func``'s prediction against the buffer
    rewards/terminated separately — no need to duplicate them inside
    the observation. Absolute position and the full grid are also
    excluded.
    """

    image: NDArray[np.int8]
    agent_dir: int = 0
    carrying: Optional[int] = None

    def copy(self) -> "MinigridObservation":
        """Copy the image and share nothing else worth copying.

        Written out rather than delegating to ``copy.deepcopy``, which
        would walk the whole object for no benefit: the other two fields
        are immutable scalars.
        """
        return MinigridObservation(
            image=self.image.copy(),
            agent_dir=self.agent_dir,
            carrying=self.carrying,
        )

    def encode(self) -> Any:
        return self

    @staticmethod
    def decode(obj: "MinigridObservation") -> "MinigridObservation":
        """Rebuild an observation, copying its image.

        The image is coerced to int8 unless it already is, which is the
        common case; the fields are read defensively so that an object
        that merely looks like an observation can be decoded too.
        """
        img = obj.image
        if isinstance(img, np.ndarray) and img.dtype == np.int8:
            new_img = img.copy()
        else:
            new_img = np.array(img, dtype=np.int8)
        return MinigridObservation(
            image=new_img,
            agent_dir=int(getattr(obj, "agent_dir", 0)),
            carrying=getattr(obj, "carrying", None),
        )

    def __eq__(self, other: object) -> bool:
        # Images are compared through their bytes rather than with
        # np.array_equal: same result, since shape and dtype are already
        # known to agree, and an order of magnitude cheaper on arrays this
        # small. Observations are compared constantly by the filter.
        if not isinstance(other, MinigridObservation):
            return False
        if not isinstance(self.image, np.ndarray) or not isinstance(other.image, np.ndarray):
            return False
        if self.image.shape != other.image.shape:
            return False
        if self.image.tobytes() != other.image.tobytes():
            return False
        # Direct attribute access: both are dataclass fields with
        # defaults, so they exist on every instance and no fallback is
        # needed.
        return (
            self.agent_dir == other.agent_dir
            and self.carrying == other.carrying
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.image.shape,
                self.image.tobytes(),
                self.agent_dir,
                self.carrying,
            )
        )

    def __repr__(self) -> str:
        return (
            "\nMinigridObservation(image=\n"
            f"{self.image}, dir={self.agent_dir}, "
            f"carrying={self.carrying})"
        )

    def distance_soft(self, other: "Observation") -> float:
        """How badly this observation disagrees with ``other``.

        This is what weights particles: an imagined observation far from
        the real one makes its particle unlikely. The image contributes
        the base distance defined by
        :meth:`uncertain_worms.structs.Observation.distance_soft`; a
        mismatched direction or carried object adds a fixed penalty on
        top, in the units selected by :data:`OBS_DIST_MODE`. Direction is
        only charged when :data:`OBS_DIR_WEIGHT` is positive; a mismatched
        carried object always is.

        When the optional weighting module is installed, its per-field
        weights replace the flat ones and also scale the image term.
        Without it the two branches agree.
        """
        adapter = _get_active_adapter() if _HAS_FIELD_WEIGHTS else None
        if not isinstance(other, MinigridObservation):
            return Observation.distance_soft(self, other)
        # Same formula as the base implementation, counted over raw bytes
        # rather than with numpy: identical result, and appreciably
        # cheaper on views this small at the volume the filter calls it.
        a_b = self.image.tobytes()
        b_b = other.image.tobytes()
        if len(a_b) != len(b_b) or self.image.shape != other.image.shape:
            return float("inf")
        total = 0
        matches = 0
        for i in range(len(a_b)):
            if a_b[i] != 12:  # ObjectTypes.agent
                total += 1
                if a_b[i] == b_b[i]:
                    matches += 1
        if total == 0:
            return 0.0
        if matches == 0:
            return float("inf")
        ratio = matches / total
        base = -math.log(1e-6 + (1.0 - 1e-6) * ratio * ratio * ratio)
        if math.isinf(base):
            return base

        nats = (OBS_DIST_MODE == "nats")
        dir_nll_mismatch = -math.log(max(OBS_DIR_EPS, 1e-12))

        if adapter is not None:
            w_img = adapter.weight_of("image", default=1.0)
            w_dir = adapter.weight_of("agent_dir", default=1.0)
            w_carry = adapter.weight_of("carrying", default=1.0)
            scaled_base = w_img * base
            extra = 0.0
            if self.agent_dir != other.agent_dir:
                extra += w_dir * (dir_nll_mismatch if nats else 1.0)
            if self.carrying != other.carrying:
                extra += w_carry * (dir_nll_mismatch if nats else 1.0)
            return scaled_base + extra

        # Flat weights, used when no weighting module is installed.
        extra = 0.0
        if OBS_DIR_WEIGHT > 0.0:
            if self.agent_dir != other.agent_dir:
                extra += OBS_DIR_WEIGHT * (dir_nll_mismatch if nats else 1.0)
        if self.carrying != other.carrying:
            extra += (dir_nll_mismatch if nats else 1.0)
        return base + extra

@dataclass
class MinigridState(State):
    """The complete world: grid, agent pose, and carried object.

    The grid is indexed ``[x, y]`` with ``y`` growing downwards, matching
    MiniGrid's own convention, and holds :class:`ObjectTypes` codes. A
    state is fully described by these four fields, so equality and hash
    cover all of them and states can be counted inside a belief.

    Instances are handled by the environment, the evaluation loop and the
    planners' internal simulation; a partially observable agent only ever
    sees the states it imagines, never the environment's.
    """

    grid: NDArray[np.int8]
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying: Optional[int]

    def copy(self) -> MinigridState:
        """Copy the grid, share the rest.

        Overrides the deep copy inherited from :class:`State`: the grid is
        the only mutable field, and this is called tens of millions of
        times per run by the planners.
        """
        return MinigridState(
            grid=self.grid.copy(),
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            carrying=self.carrying,
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.agent_pos),
                self.agent_dir,
                self.carrying,
                self.grid.tobytes(),
            )
        )

    def __eq__(self, other: object) -> bool:
        # Overrides the hash-based equality of `State` with a field-by-
        # field comparison. Grids are compared through their bytes, which
        # is unambiguous once the shapes are known to match and much
        # cheaper than np.array_equal at this call volume.
        if not isinstance(other, MinigridState):
            return False
        if not isinstance(self.grid, np.ndarray) or not isinstance(other.grid, np.ndarray):
            return False
        if self.grid.shape != other.grid.shape:
            return False
        if self.grid.tobytes() != other.grid.tobytes():
            return False
        return (
            tuple(self.agent_pos) == tuple(other.agent_pos)
            and self.agent_dir == other.agent_dir
            and self.carrying == other.carrying
        )

    @property
    def front_pos(self) -> Tuple[int, int]:
        """Get the position of the cell that is right in front of the agent.

        May fall outside the grid; callers check the bounds themselves.
        """
        # Plain tuple arithmetic rather than numpy: this runs inside the
        # transition model, on every simulated step.
        dx, dy = DIR_TO_VEC[self.agent_dir]
        return (self.agent_pos[0] + int(dx), self.agent_pos[1] + int(dy))

    @property
    def width(self) -> int:
        return self.grid.shape[0]

    @property
    def height(self) -> int:
        return self.grid.shape[1]

    def get_type_indices(self, obj_type: int) -> List[Tuple[int, int]]:
        """Positions of every cell holding ``obj_type``, as (x, y) pairs."""
        idxs = np.where(self.grid == obj_type)  # Returns (row_indices, col_indices)
        return list(zip(idxs[0], idxs[1]))  # Combine row and column indices

    def encode(self) -> MinigridState:
        """Identity: generated code manipulates the state object itself.

        Domains whose states need a different representation for the
        language model do the conversion here; this one does not, so what
        a model reads and what the planner holds are the same object.
        """
        return self

    @staticmethod
    def decode(obj: MinigridState) -> MinigridState:
        """Rebuild a state from an encoded one.

        The grid is copied rather than shared, so code that writes into
        the returned state cannot alter the one it was built from.
        """
        # Skip the dtype coercion when the array already has the expected
        # type, as it does on the hot path.
        grid = obj.grid
        if isinstance(grid, np.ndarray) and grid.dtype == np.int8:
            new_grid = grid.copy()
        else:
            new_grid = np.array(grid, dtype=np.int8)
        return MinigridState(
            new_grid,
            agent_pos=obj.agent_pos,
            agent_dir=obj.agent_dir,
            carrying=obj.carrying,
        )

    def get_field_of_view(self, view_size: int) -> NDArray[np.int8]:
        """Returns the field of view in front of the agent.

        A ``view_size`` square anchored on the agent, rotated so that the
        result is always read as if the agent were facing up, with the
        agent itself at the bottom-centre cell. Cells outside the grid
        come back as wall, so the view has a fixed shape wherever the
        agent stands. This is the geometry the observation model and the
        particle filter both assume.

        DO NOT modify this function.
        """
        # Compiled path; falls through to the identical Python one below
        # when numba is unavailable.
        if _HAS_NUMBA:
            return _fov_compute_numba(
                self.grid,
                self.agent_pos[0],
                self.agent_pos[1],
                int(self.agent_dir),
                int(view_size),
                np.int8(ObjectTypes.wall),
                np.int8(ObjectTypes.agent),
            )

        # Get the extents of the square set of tiles visible to the agent
        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - view_size + 1
            topY = self.agent_pos[1] - view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - view_size // 2
            topY = self.agent_pos[1] - view_size + 1
        else:
            assert False, "invalid agent direction"

        # Everything starts as wall; the slice below then overwrites every
        # cell that falls inside the grid, so what remains is exactly the
        # part of the view that lies outside it.
        fov = np.empty((view_size, view_size), dtype=self.grid.dtype)
        fov.fill(ObjectTypes.wall)

        # Compute the overlapping region in the grid.
        gx0 = max(topX, 0)
        gy0 = max(topY, 0)
        gx1 = min(topX + view_size, self.grid.shape[0])
        gy1 = min(topY + view_size, self.grid.shape[1])

        # Determine where the overlapping region goes in the padded array.
        px0 = max(0, -topX)
        py0 = max(0, -topY)

        # Copy the overlapping slice.
        fov[px0 : px0 + (gx1 - gx0), py0 : py0 + (gy1 - gy0)] = self.grid[
            gx0:gx1, gy0:gy1
        ]

        # Rotate the view into the agent's frame, k = -(agent_dir+1) mod 4.
        # Written as explicit slicing rather than np.rot90: identical
        # result on all four cases, but strided views instead of the
        # copies rot90 makes, which matters at this call volume.
        n_rot = (-(self.agent_dir + 1)) % 4
        if n_rot == 1:
            fov = fov.T[::-1]
        elif n_rot == 2:
            fov = fov[::-1, ::-1]
        elif n_rot == 3:
            fov = fov[::-1].T
        # n_rot == 0: leave fov unchanged

        agent_pos = (fov.shape[0] // 2, fov.shape[1] - 1)
        fov[agent_pos] = ObjectTypes.agent

        return fov

    def __repr__(self) -> str:
        """Returns a string representation of the grid with agent position."""
        print_state = "agent_pos={}\n".format(self.agent_pos)
        print_state += "agent_dir={}\n".format(self.agent_dir)
        print_state += "carrying={}\n".format(self.carrying)
        print_state += "grid=[\n"
        for x in range(self.width):
            row = "["
            for y in range(self.height):
                row += f"{self.grid[x, y]:2d}, "
            row += "],\n"
            print_state += row
        print_state += "]\n"
        return print_state


def minigrid_empty_observation_gen(**kwargs: Any) -> MinigridObservation:
    """A blank observation of the right shape for this domain.

    Observation models receive one of these as a template and return it
    unchanged when they have nothing to report.
    """
    return MinigridObservation(
        image=np.full((3, 3), ObjectTypes.empty),
        agent_dir=0,
        carrying=None,
    )

def minigrid_empty_state_gen(env_name: str = "MiniGrid-Empty-5x5-v0") -> MinigridState:
    """A blank state with the walls of ``env_name`` and nothing else.

    The environment is instantiated only to read its layout, then every
    non-wall cell is cleared: the result has the right dimensions and
    boundaries but no goal, key or door. Configurations pass it to an
    agent as its ``empty_state`` template.
    """
    env: MiniGridEnv = gym.make(env_name, agent_view_size=3, max_steps=1)
    _ = env.reset()
    grid = env.unwrapped.grid
    state = grid_to_state(
        grid=grid,
        agent_pos=(1, 1),
        agent_dir=0,
    )
    # Remove all non-walls from the grid and make them empty:
    state.grid[state.grid != ObjectTypes.wall] = ObjectTypes.empty
    return state


def goal_heuristic_gen() -> Heuristic:
    """Search heuristic: normalised Manhattan distance to the nearest goal.

    Divided by the number of cells so the value stays in [0, 1] and does
    not overwhelm the costs it is added to. A state with no visible goal
    scores zero, which leaves such states indistinguishable to the search.
    """

    def goal_heuristic(state: MinigridState) -> float:
        if len(state.get_type_indices(ObjectTypes.goal)) == 0:
            return 0

        min_dist = min(
            [
                abs(s[0] - state.agent_pos[0]) + abs(s[1] - state.agent_pos[1])
                for s in state.get_type_indices(ObjectTypes.goal)
            ]
        )
        return min_dist / (state.width * state.height)

    return goal_heuristic


def get_empty_room(width: int, height: int) -> MinigridState:
    """Build a bare walled room of the given size, agent in the corner."""
    grid = Grid(width, height)

    # Place walls around the entire edge of the map
    for x in range(width):
        grid.set(x, 0, world_object.Wall())  # Top edge
        grid.set(x, height - 1, world_object.Wall())  # Bottom edge

    for y in range(height):
        grid.set(0, y, world_object.Wall())  # Left edge
        grid.set(width - 1, y, world_object.Wall())  # Right edge

    # Place the agent at (1, 1) with an initial direction of 0
    return grid_to_state(grid, agent_pos=(1, 1), agent_dir=0)


def minigrid_hardcoded_initial_model_gen(
    env_name: str = "MiniGrid-Empty-5x5-v0",
) -> InitialModel:
    """Reference initial-state model: reset a real environment and read it.

    Sampling a start means resetting a private copy of the environment,
    so the distribution over starts is exactly the one the agent will
    face. The reset seeds are drawn from a fixed high range, disjoint
    from the small seeds the evaluation loop uses for its episodes.
    """
    env: MiniGridEnv = gym.make(env_name, agent_view_size=3, max_steps=1)
    env.unwrapped.see_through_walls = SEE_THROUGH_WALLS

    def minigrid_hardcoded_initial_model(_: MinigridState) -> MinigridState:
        _ = env.reset(seed=np.random.randint(10000, 20000))
        grid = env.unwrapped.grid
        return copy.deepcopy(
            grid_to_state(
                grid=grid,
                agent_pos=env.unwrapped.agent_pos,
                agent_dir=env.unwrapped.agent_dir,
            )
        )

    return minigrid_hardcoded_initial_model


def minigrid_hardcoded_initial_model_fixed_grid_random_agent_gen(
    env_name: str = "MiniGrid-Empty-5x5-v0",
    max_position_distance: int = 2,
    fully_random: bool = True,
) -> InitialModel:
    """Initial-state model over agent poses only, on one fixed layout.

    The layout is read once from a seed-0 reset and reused for every
    sample: walls and goal never move, and only the agent's pose is
    drawn. This isolates uncertainty about where the agent is from
    uncertainty about what the world contains.

    - ``fully_random=True``: position and direction uniform over the
      cells the agent may occupy, which excludes walls, lava and the goal.
    - ``fully_random=False``: position within ``max_position_distance``
      of the layout's own start, keeping its original direction.
    """
    env: MiniGridEnv = gym.make(env_name, agent_view_size=3, max_steps=1)
    env.unwrapped.see_through_walls = SEE_THROUGH_WALLS
    _ = env.reset(seed=0)
    grid = env.unwrapped.grid
    fixed_state = grid_to_state(
        grid=grid,
        agent_pos=env.unwrapped.agent_pos,
        agent_dir=env.unwrapped.agent_dir,
    )
    original_pos = tuple(fixed_state.agent_pos)
    original_dir = fixed_state.agent_dir
    goal_positions = set(fixed_state.get_type_indices(ObjectTypes.goal))
    # Valid positions: any non-wall, non-lava, non-goal cell (agent cannot start on goal)
    valid_all_positions: List[Tuple[int, int]] = []
    valid_agent_positions: List[Tuple[int, int]] = []
    for x in range(fixed_state.width):
        for y in range(fixed_state.height):
            cell = fixed_state.grid[x, y]
            pos = (x, y)
            if cell != ObjectTypes.wall and cell != ObjectTypes.lava and pos not in goal_positions:
                valid_all_positions.append(pos)
                if not fully_random:
                    dist = abs(x - original_pos[0]) + abs(y - original_pos[1])
                    if dist <= max_position_distance:
                        valid_agent_positions.append(pos)
    if not valid_agent_positions:
        valid_agent_positions = valid_all_positions if valid_all_positions else [original_pos]
    if not valid_all_positions:
        valid_all_positions = [original_pos]

    def minigrid_initial_fixed_grid_random_agent(_: MinigridState) -> MinigridState:
        grid_copy = np.copy(fixed_state.grid)
        # Goal stays fixed. Only randomize agent position and direction.
        agent_pos = (
            valid_all_positions[np.random.randint(len(valid_all_positions))]
            if fully_random
            else valid_agent_positions[np.random.randint(len(valid_agent_positions))]
        )
        agent_dir = np.random.randint(4) if fully_random else original_dir
        return MinigridState(
            grid=grid_copy,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            carrying=fixed_state.carrying,
        )

    return minigrid_initial_fixed_grid_random_agent


def minigrid_hardcoded_reward_model_gen() -> RewardModel:
    """Reference reward model: 1 for reaching a goal, 0 otherwise.

    Both goal and lava end the episode, and neither is charged a cost, so
    the discounted return only rewards reaching the goal quickly. The
    cell is read from ``state``, the grid before the move, at the
    position the agent ends up in.
    """

    def minigrid_hardcoded_reward_model(
        state: MinigridState, action: ActType, next_state: MinigridState
    ) -> Tuple[float, bool]:
        if (
            state.grid[next_state.agent_pos[0], next_state.agent_pos[1]]
            == ObjectTypes.goal
        ):
            return 1.0, True
        if (
            state.grid[next_state.agent_pos[0], next_state.agent_pos[1]]
            == ObjectTypes.lava
        ):
            return 0.0, True
        return 0.0, False

    return minigrid_hardcoded_reward_model


def process_vis(grid: NDArray[np.int8], agent_pos: tuple[int, int]) -> NDArray[np.int8]:
    """Visibility mask over a field of view, for occluded observations.

    Only reachable when :data:`SEE_THROUGH_WALLS` is False; with the
    module's default nothing calls it.
    """
    # Initialize the boolean mask.
    mask = np.zeros(grid.shape, dtype=bool)
    mask[agent_pos] = True
    left_shift = np.roll(mask, shift=-1, axis=0)
    right_shift = np.roll(mask, shift=1, axis=0)
    down_shift = np.roll(mask, shift=-1, axis=0)
    down_shift[:, 0] = mask[:, 0]  # For the first column, do not roll.
    visible = mask & (grid == None)
    mask = mask | visible | left_shift | right_shift | down_shift
    return mask.astype(np.int8)


def minigrid_hardcoded_observation_model_gen() -> ObservationModel:
    """Single POMDP-clean observation model.

    Renders the partial FOV and carries through the fields the agent
    is entitled to observe: ``agent_dir`` (compass / proprioception)
    and ``carrying`` (proprioception). ``reward`` and ``terminated``
    stay at their defaults here — they are populated by the env
    (``MinigridEnvironment.step``) because they describe the step
    outcome, not the internal rendering.
    """

    def minigrid_hardcoded_observation_model(
        state: MinigridState,
        action: int,
        empty_obs: MinigridObservation,
        agent_view_size: int = 3,
    ) -> MinigridObservation:
        grid = state.get_field_of_view(view_size=agent_view_size)

        if not SEE_THROUGH_WALLS:
            vis_mask = process_vis(
                grid.T, agent_pos=(agent_view_size // 2, agent_view_size - 1)
            ).T
            grid[~vis_mask] = ObjectTypes.empty

        return MinigridObservation(
            image=np.array(grid, dtype=np.int8),
            agent_dir=state.agent_dir,
            carrying=state.carrying,
        )

    return minigrid_hardcoded_observation_model


def minigrid_hardcoded_transition_model_gen() -> TransitionModel:
    """Reference transition model: MiniGrid's dynamics, written out.

    Deterministic. Turning always succeeds; moving forward is blocked by
    walls, doors that are not open, and objects lying on the floor;
    pickup, drop and toggle act on the cell directly in front. An action
    with no effect leaves the state as it was, and an unknown action code
    raises.
    """

    def minigrid_hardcoded_transition_model(
        state: MinigridState, action: Actions
    ) -> MinigridState:
        """Return the state that follows ``action``, leaving the input alone."""

        state = copy.deepcopy(state)
        # Create a copy of the current state
        new_state = MinigridState(
            grid=state.grid.copy(),
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            carrying=state.carrying,
        )

        # Rotate left
        if action == Actions.left:
            new_state.agent_dir = (state.agent_dir - 1) % 4

        # Rotate right
        elif action == Actions.right:
            new_state.agent_dir = (state.agent_dir + 1) % 4

        # Move forward
        elif action == Actions.forward:
            fwd_pos = state.front_pos  # Compute forward position

            # Check if the forward position is within bounds
            if 0 <= fwd_pos[0] < state.width and 0 <= fwd_pos[1] < state.height:
                # Check if there's a wall at the new position
                if (
                    state.grid[fwd_pos] != ObjectTypes.wall
                    and state.grid[fwd_pos] != ObjectTypes.ball
                    and state.grid[fwd_pos] != ObjectTypes.key
                    and state.grid[fwd_pos] != ObjectTypes.locked_door
                    and state.grid[fwd_pos] != ObjectTypes.closed_door
                ):
                    new_state.agent_pos = fwd_pos

        elif action == Actions.pickup:
            if state.grid[state.front_pos] == ObjectTypes.key:
                new_state.carrying = ObjectTypes.key
                new_state.grid[state.front_pos] = ObjectTypes.empty
            if state.grid[state.front_pos] == ObjectTypes.ball:
                new_state.carrying = ObjectTypes.ball
                new_state.grid[state.front_pos] = ObjectTypes.empty
        elif action == Actions.toggle:
            if (
                state.grid[state.front_pos] == ObjectTypes.locked_door
                and state.carrying == ObjectTypes.key
            ):
                new_state.grid[state.front_pos] = ObjectTypes.open_door
        elif action == Actions.drop:
            if (
                state.carrying is not None
                and state.grid[state.front_pos] == ObjectTypes.empty
            ):
                new_state.grid[state.front_pos] = state.carrying
                new_state.carrying = None
        else:
            raise ValueError(f"Unknown action: {action}")

        return new_state

    return minigrid_hardcoded_transition_model


def state_to_grid(state: MinigridState, grid: Grid) -> None:
    """Apply a MinigridState's grid to a minigrid Grid in-place."""
    for x in range(state.width):
        for y in range(state.height):
            cell = state.grid[x, y]
            if cell == ObjectTypes.empty or cell == ObjectTypes.unseen:
                grid.set(x, y, None)
            else:
                grid.set(x, y, local_to_minigrid(cell))


def grid_to_state(
    grid: Grid,
    agent_pos: Tuple[int, int],
    agent_dir: int,
    carrying: world_object.WorldObj | None = None,
) -> MinigridState:
    """Convert a MiniGrid ``Grid`` and an agent pose into a MinigridState.

    Cell by cell, MiniGrid objects become :class:`ObjectTypes` codes; the
    carried object, given as a MiniGrid object, becomes a code too. This
    is the only path by which the environment's internals enter the
    representation used everywhere else.
    """
    grid_array = np.zeros((grid.width, grid.height), dtype=np.int8)
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            grid_array[x, y] = minigrid_to_local(obj)
    if carrying is not None:
        carrying = ObjectTypes[carrying.type]
    return MinigridState(
        np.array(grid_array, dtype=np.int8), agent_pos, agent_dir, carrying
    )


def build_state(
    obs_dict: Dict,
    carrying: world_object.WorldObj | None,
) -> MinigridState:
    """Reconstruct the true state from a wrapped gym observation dict.

    Relies on the position and grid the wrappers at the top of this
    module add to that dict, which is why the environment installs them.
    """
    return grid_to_state(
        grid=obs_dict["grid"],
        agent_pos=obs_dict["agent_pos"],
        agent_dir=obs_dict["direction"],
        carrying=carrying,
    )


def build_obs(obs_dict: Dict, fully: bool = True) -> MinigridObservation:
    """Build the observation an agent receives from a gym ``obs_dict``.

    Re-encodes the image into :class:`ObjectTypes` codes and picks up the
    carried object and the facing direction. Absolute position and the
    full grid are dropped here: this function is the boundary past which
    the state does not travel.

    ``fully`` selects which image the dict holds. In the partially
    observable case it is the field of view, whose bottom-centre cell is
    the agent's own; in the fully observable case it is the whole grid
    and the agent's cell is located through ``agent_pos``. Either way the
    agent's cell is where the carried object is read from, since MiniGrid
    encodes it there.
    """
    W, H, _ = obs_dict["image"].shape

    obs_array = np.zeros((W, H), dtype=np.int8)
    carrying = None
    for i in range(W):
        for j in range(H):
            if i == W // 2 and j == H - 1 and not fully:
                # Agent location in the FOV (partial observation mode).
                obs_array[i, j] = ObjectTypes.agent
                if IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] != "empty":
                    carrying = minigrid_to_local(
                        world_object.WorldObj.decode(
                            *(obs_dict["image"][i, j, :].tolist())
                        )
                    )
            elif fully and list(obs_dict["agent_pos"]) == [i, j]:
                if IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] != "agent":
                    carrying = minigrid_to_local(
                        world_object.WorldObj.decode(
                            *(obs_dict["image"][i, j, :].tolist())
                        )
                    )
                obs_array[i, j] = ObjectTypes.empty
            elif IDX_TO_OBJECT[obs_dict["image"][i, j, 0]] == "empty":
                obs_array[i, j] = ObjectTypes.empty
            else:
                wobj = world_object.WorldObj.decode(
                    *(obs_dict["image"][i, j, :].tolist())
                )
                obs_array[i, j] = int(minigrid_to_local(wobj))

    return MinigridObservation(
        image=obs_array,
        agent_dir=int(obs_dict["direction"]),
        carrying=carrying,
    )

def create_gif(
    width: int,
    height: int,
    observations: List[MinigridObservation],
    gif_filename: str = "output.gif",
    duration: float = 100.0,
    fully_obs: bool = True,
    states: List[MinigridState] | None = None,
) -> None:
    """Render an episode as a GIF of the map the agent gradually uncovers.

    Successive views are stitched onto a shared grid, so the animation
    shows what had been seen by each step rather than the true world.

    ``states`` is required because observations carry no position: the
    poses needed to place each view on the map have to come from the
    recorded states. Reading them is legitimate here — this is an
    inspection utility that runs after an episode, never on the path that
    produces a result.
    """
    if states is None:
        raise ValueError(
            "create_gif now requires `states` for every call: the "
            "observation class no longer exposes agent_pos."
        )

    current_grid = PartialGrid(np.zeros((width, height)))
    partial_grid_trajectory: List[PartialGrid] = []

    # ``states`` is typically [s0] + next_states while ``observations``
    # is ``next_observations`` (one element shorter by one). Align them
    # so ``state_idx`` points to the state at which ``observation`` was
    # emitted.
    if len(states) == len(observations) + 1:
        state_offset = 1
    elif len(states) == len(observations):
        state_offset = 0
    else:
        raise ValueError(
            f"create_gif: states/observations length mismatch "
            f"(states={len(states)}, observations={len(observations)})"
        )

    for t, observation in enumerate(observations):
        image = observation.image
        state_idx = t + state_offset
        agent_pos = states[state_idx].agent_pos
        agent_dir = states[state_idx].agent_dir

        if fully_obs:
            current_grid = update_fo_grid_with_pose(
                current_grid, image=image, agent_pos=agent_pos, agent_dir=agent_dir
            )
        else:
            current_grid = update_po_grid_with_pose(
                current_grid, image=image, agent_pos=agent_pos, agent_dir=agent_dir
            )
    
        partial_grid_trajectory.append(current_grid)

    frames = []
    for step in range(len(partial_grid_trajectory)):
        color_grid = render_partial_grid(
            partial_grid_trajectory[step], fully_obs=fully_obs
        )

        buffer = BytesIO()
        plt.imshow(color_grid)
        plt.title(f"Step {step + 1}")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        frames.append(imageio.v2.imread(buffer))
        buffer.close()

    imageio.v2.mimwrite(gif_filename, frames, format="GIF", duration=duration)  # type: ignore
    print(f"GIF saved as {gif_filename}")

# -------- Belief related functions
@dataclass
class PartialGrid:
    """A map built up from partial views, for rendering.

    ``grid`` accumulates every cell seen so far; ``in_view`` is what the
    current step reveals, drawn highlighted, and ``has_viewed`` is
    everything seen since the start — cells absent from it are rendered
    as unknown rather than as empty.
    """

    grid: NDArray
    agent_pos: Tuple[int, int] = (0, 0)
    agent_dir: int = 0
    in_view: List[Tuple[int, int]] = field(default_factory=list)
    has_viewed: List[Tuple[int, int]] = field(default_factory=list)


def get_view_exts(
    agent_pos: Tuple[int, int], agent_dir: int, agent_view_size: int
) -> Tuple[int, int, int, int]:
    """
    Get the extents of the square set of tiles visible to the agent
    Note: the bottom extent indices are not included in the set
    if agent_view_size is None, use self.agent_view_size
    """

    # Facing right
    if agent_dir == 0:
        topX = agent_pos[0]
        topY = agent_pos[1] - agent_view_size // 2
    # Facing down
    elif agent_dir == 1:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1]
    # Facing left
    elif agent_dir == 2:
        topX = agent_pos[0] - agent_view_size + 1
        topY = agent_pos[1] - agent_view_size // 2
    # Facing up
    elif agent_dir == 3:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1] - agent_view_size + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + agent_view_size
    botY = topY + agent_view_size

    return topX, topY, botX, botY


def get_right_vector(agent_dir_vec: Tuple[int, int]) -> Tuple[int, int]:
    """Returns vector pointing to the right of the agent."""
    dx, dy = agent_dir_vec
    return np.array((-dy, dx)).tolist()


def update_fo_grid_with_pose(
    partial_grid: PartialGrid,
    image: NDArray[np.int8],
    agent_pos: Tuple[int, int] | NDArray[Any],
    agent_dir: int,
) -> PartialGrid:
    """Fold a fully observable view into the rendered map.

    The image is already in world coordinates, so it overwrites the map
    directly, except where it is marked unseen.
    """
    new_partial_grid = copy.deepcopy(partial_grid)
    new_partial_grid.agent_pos = agent_pos
    new_partial_grid.agent_dir = agent_dir
    new_partial_grid.grid[image != ObjectTypes.unseen] = image[image != ObjectTypes.unseen]
    return new_partial_grid


def update_po_grid_with_pose(
    partial_grid: PartialGrid,
    image: NDArray[np.int8],
    agent_pos: Tuple[int, int] | NDArray[Any],
    agent_dir: int,
) -> PartialGrid:
    """Fold a partial view into the rendered map.

    The view is expressed relative to the agent, so each of its cells is
    mapped back to world coordinates through the forward and right
    vectors of the current heading before being written. Cells that fall
    outside the map are skipped, and the agent's own cell is not written,
    since it says nothing about what the world holds there.
    """
    new_partial_grid = copy.deepcopy(partial_grid)
    new_partial_grid.agent_pos = agent_pos
    new_partial_grid.agent_dir = agent_dir

    agent_pos_arr = np.array(agent_pos)
    f_vec = DIR_TO_VEC[agent_dir]
    r_vec = np.array(get_right_vector(tuple(f_vec.tolist())))

    partially_observable_view = image
    view_size_x, view_size_y = partially_observable_view.shape[0:2]

    top_left = agent_pos_arr + f_vec * (view_size_x - 1) - r_vec * (view_size_y // 2)

    new_partial_grid.in_view = []
    for i in range(0, view_size_x):
        for j in range(0, view_size_y):
            obj_idx = partially_observable_view[i, j]
            abs_i, abs_j = top_left - (f_vec * j) + (r_vec * i)
            new_partial_grid.has_viewed.append((abs_i, abs_j))

            if abs_i < 0 or abs_i >= new_partial_grid.grid.shape[0]:
                continue
            if abs_j < 0 or abs_j >= new_partial_grid.grid.shape[1]:
                continue

            new_partial_grid.in_view.append((abs_i, abs_j))

            if obj_idx == ObjectTypes.unseen or obj_idx == ObjectTypes.agent:
                continue

            new_partial_grid.grid[abs_i, abs_j] = obj_idx

    return new_partial_grid

def render_tile(
    obj: int,
    agent_dir: int | None = None,
    highlight: bool = False,
    tile_size: int = 32,
    subdivs: int = 3,
    unseen: bool = False,
) -> NDArray[np.uint8]:
    """Render one cell, caching the result.

    Tiles are drawn oversampled and downsampled at the end, which is how
    the edges come out smooth. Identical tiles recur constantly across a
    frame and across frames, so rendered tiles are memoised in MiniGrid's
    own cache, keyed by everything that affects the drawing.
    """
    # Hash map lookup key for the cache
    key: tuple[Any, ...] = (obj, agent_dir, highlight, tile_size, unseen)

    if key in Grid.tile_cache:
        return Grid.tile_cache[key]

    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    if obj == ObjectTypes.unseen and unseen:
        super_size = tile_size * subdivs
        stripe_thickness = subdivs * 2  # ~1px in final image
        stripe_spacing = 8 * subdivs  # e.g. every 8 final pixels
        I, J = np.indices((super_size, super_size))
        mask = ((I - J) % stripe_spacing) < stripe_thickness
        img[mask] = np.array([0, 0, 255], dtype=img.dtype)

    # Draw the grid lines (top and left edges)
    fill_coords(img, world_object.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    wobj = local_to_minigrid(obj)
    if wobj is not None:
        wobj.render(img)

    # Overlay the agent on top
    if agent_dir is not None:
        assert (
            wobj is None
            or obj == ObjectTypes.goal
            or obj == ObjectTypes.open_door
            or obj == ObjectTypes.lava
        ), f"Agent direction should only be set for goal objects, not {obj}"
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        fill_coords(img, tri_fn, (255, 0, 0))

    # Highlight the cell if needed
    if highlight:
        highlight_img(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)
    # Cache the rendered tile
    Grid.tile_cache[key] = img

    return img


def render_partial_grid(
    partial_grid: PartialGrid, tile_size: int = 32, fully_obs: bool = False
) -> NDArray[np.uint8]:
    """Render a :class:`PartialGrid` into an RGB image.

    Each cell is drawn through :func:`render_tile`; the cells currently in
    view are highlighted, and cells never seen are hatched unless the
    caller asks for the fully observable rendering.

    Returns an array of shape ``(height * tile_size, width * tile_size, 3)``:
    image rows are the grid's ``y``, so the axes are transposed with
    respect to the state's ``[x, y]`` indexing.
    """

    W, H = partial_grid.grid.shape[0], partial_grid.grid.shape[1]

    # Compute the total grid size
    width_px = W * tile_size
    height_px = H * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for i in range(0, W):
        for j in range(0, H):
            agent_here = np.array_equal(partial_grid.agent_pos, (i, j))
            if (i, j) in partial_grid.in_view:
                highlight = True
            else:
                highlight = False

            tile_img = render_tile(
                partial_grid.grid[i, j],
                agent_dir=partial_grid.agent_dir if agent_here else None,
                highlight=highlight,
                tile_size=tile_size,
                unseen=(not ((i, j) in partial_grid.has_viewed)) and (not fully_obs),
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img


class MinigridEnvironment(Environment):
    """A MiniGrid level behind the interface the evaluation loop expects.

    Wraps a gym environment named by ``env_name`` and returns, at every
    step, both the observation an agent may see and the true state, which
    goes to the logs and the replay buffer.

    The agent's view is three cells wide, and walls do not block sight
    (see :data:`SEE_THROUGH_WALLS`), so what limits the agent is the
    extent of its view. ``fully_obs`` switches the whole grid in as the
    observation, which is used for the fully observable comparisons.
    """

    def __init__(
        self,
        *args: Any,
        env_name: str = "MiniGrid-Empty-5x5-v0",
        fully_obs: bool = True,
        # Accepted and ignored: these once selected between observation
        # variants that a single observation class now replaces. They stay
        # in the signature so that configurations and scripts still
        # passing them keep loading.
        pure_obs: bool = False,
        compass: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.env: MiniGridEnv = gym.make(
            env_name, agent_view_size=3, max_steps=self.max_steps
        )
        self.env.unwrapped.see_through_walls = SEE_THROUGH_WALLS
        self.fully_obs = fully_obs
        self.pure_obs = pure_obs
        self.compass = compass
        self.width, self.height = self.env.width, self.env.height
        self.env = AgentPositionWrapper(self.env)
        if fully_obs:
            self.env = FullyObsWrapper(self.env)
        self.env = GridWrapper(self.env, width=self.width, height=self.height)
        self._last_obs: Observation | None = None

    def get_last_obs(self) -> Observation:
        """Return the most recent observation, for callers that stepped elsewhere."""
        if self._last_obs is None:
            raise RuntimeError("No observation available yet. Call reset() first.")
        return self._last_obs

    def _build_obs(self, obs_dict: Dict) -> MinigridObservation:
        """Reduce a gym observation dict to what an agent may perceive."""
        return build_obs(obs_dict, fully=self.fully_obs)

    def step(
        self, action: ActType
    ) -> Tuple[Observation, State, float, bool, bool, Dict[str, Any]]:
        """Take one step and return the observation, the state and the outcome.

        Reward and termination are returned beside the observation and are
        not written into it: the observation stays purely perceptual, and
        the rest of the pipeline reads outcomes from the replay buffer's
        own arrays.
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        next_obs = self._build_obs(obs_dict)
        next_state = build_state(obs_dict, carrying=self.env.unwrapped.carrying)
        # MiniGrid's success reward decays with the number of steps taken.
        # It is reduced to a plain success flag here, so that how long an
        # episode took shows up through discounting only.
        binary_reward = int(reward > 0.0)
        self._last_obs = next_obs
        return next_obs, next_state, binary_reward, terminated, truncated, info

    def reset(self, seed: int = 0) -> State:
        """Start a new episode and return its true initial state.

        The seed fixes the layout and the starting pose, so a given seed
        always produces the same episode. The initial observation is kept
        for :meth:`get_last_obs` rather than returned.
        """
        obs_dict, _ = self.env.reset(seed=seed)
        initial_obs = self._build_obs(obs_dict)
        initial_state = build_state(obs_dict, carrying=self.env.unwrapped.carrying)
        self._last_obs = initial_obs
        return initial_state
    
    def visualize_episode(
        self,
        states: List[MinigridState],
        observations: List[MinigridObservation],
        actions: List[int],
        episode_num: int,
    ) -> None:
        """Write the episode's GIF into the run directory."""
        filename = os.path.join(get_log_dir(), f"episode{episode_num}_output.gif")
        create_gif(
            self.width,
            self.height,
            observations,
            filename,
            fully_obs=self.fully_obs,
            states=states,
        )
