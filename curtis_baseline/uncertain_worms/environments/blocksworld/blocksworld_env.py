"""Partially-observable Blocks World for uncertain_worms.

A genuinely non-grid, non-MiniGrid domain: the state is a tuple of *stacks*
(lists of distinct block ids) plus a gripper position and a held block — NOT a
2D image. The task is manipulation (rearrange blocks into a target
configuration), goal-reaching and sparse-terminal, exactly the regime where the
submitted best-case A* planner is sound:

  * the ONLY hidden variable is the gripper's starting position (passive
    localization), the static block layout is known;
  * the agent observes only the stack it is currently at (partial), never its
    absolute position — so it must localize by acting, info revealed by
    goal-directed motion (no dedicated sense action);
  * reward is +1 exactly when the configuration matches the target.

Actions: 0=move left (pos-1), 1=move right (pos+1), 2=pick (take the top block
of the current stack into the gripper, if empty-handed and the stack is
non-empty), 3=drop (place the held block on top of the current stack).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from uncertain_worms.structs import Environment, Observation, State

# Actions
A_LEFT, A_RIGHT, A_PICK, A_DROP = 0, 1, 2, 3
NO_BLOCK = 0  # gripper holds nothing

Stacks = Tuple[Tuple[int, ...], ...]


def _norm_stacks(stacks) -> Stacks:
    return tuple(tuple(int(b) for b in st) for st in stacks)


# =========================================================================
# State
# =========================================================================
@dataclass(eq=False)
class BlocksState(State):
    """stacks[i] = blocks at position i, bottom -> top. ``pos`` is the gripper
    column, ``held`` the carried block (0 = none). ``target`` is the static goal
    configuration (carried in the state so reward/termination are self-contained)."""

    stacks: Stacks
    pos: int
    held: int
    target: Stacks

    def __post_init__(self) -> None:
        self.stacks = _norm_stacks(self.stacks)
        self.target = _norm_stacks(self.target)
        self.pos = int(self.pos)
        self.held = int(self.held)

    @property
    def num_pos(self) -> int:
        return len(self.stacks)

    def is_solved(self) -> bool:
        return self.held == NO_BLOCK and self.stacks == self.target

    def copy(self) -> "BlocksState":
        return BlocksState(stacks=self.stacks, pos=self.pos, held=self.held, target=self.target)

    def __repr__(self) -> str:
        return f"BlocksState(stacks={self.stacks}, pos={self.pos}, held={self.held})"

    def encode(self) -> "BlocksState":
        return self

    @classmethod
    def decode(cls, encoded: Any) -> "BlocksState":
        if isinstance(encoded, BlocksState):
            return encoded.copy()
        if isinstance(encoded, dict):
            return cls(stacks=encoded["stacks"], pos=encoded["pos"],
                       held=encoded["held"], target=encoded["target"])
        raise TypeError(f"cannot decode {type(encoded)}")

    def __hash__(self) -> int:
        return hash((self.stacks, self.pos, self.held, self.target))


# =========================================================================
# Observation
# =========================================================================
@dataclass(eq=False)
class BlocksObservation(Observation):
    """What the gripper perceives: the contents of the CURRENT stack (bottom ->
    top) and the block it holds. It does NOT include the absolute position.

    ``image`` is a fixed-length int vector (current stack padded to ``cap`` with
    zeros, followed by the held block) so the particle filter has a soft
    distance to work with."""

    current_stack: Tuple[int, ...]
    held: int
    image: NDArray[np.int8] = field(default_factory=lambda: np.zeros(1, dtype=np.int8))

    def __post_init__(self) -> None:
        self.current_stack = tuple(int(b) for b in self.current_stack)
        self.held = int(self.held)

    def __hash__(self) -> int:
        return hash((self.current_stack, self.held))

    def __eq__(self, other: object) -> bool:
        # Value equality, consistent with __hash__. Required by the base
        # Observation.distance() default (0 iff equal, else inf), which the
        # original Curtis particle filter relies on; ``image`` is a pure
        # function of (current_stack, held) so it is not compared.
        if not isinstance(other, BlocksObservation):
            return NotImplemented
        return (self.current_stack, self.held) == (other.current_stack, other.held)

    def __repr__(self) -> str:
        return f"BlocksObservation(current_stack={self.current_stack}, held={self.held})"

    def distance_soft(self, other: "Observation") -> float:
        a = np.asarray(self.image)
        b = np.asarray(getattr(other, "image", None))
        if a.shape != b.shape:
            return float("inf")
        total = a.size
        matches = int(np.count_nonzero(a == b))
        if matches == 0:
            return float("inf")
        ratio = matches / total
        eps = 1e-6
        return -math.log(eps + (1.0 - eps) * ratio ** 3)

    def encode(self) -> "BlocksObservation":
        return self

    @classmethod
    def decode(cls, encoded: Any) -> "BlocksObservation":
        if isinstance(encoded, BlocksObservation):
            return encoded
        raise TypeError(f"cannot decode {type(encoded)}")


def _obs_image(current_stack: Tuple[int, ...], held: int, cap: int) -> NDArray[np.int8]:
    vec = np.zeros(cap + 1, dtype=np.int8)
    for i, b in enumerate(current_stack[:cap]):
        vec[i] = b
    vec[cap] = held
    return vec


# =========================================================================
# Deterministic dynamics (shared by env + GT transition model)
# =========================================================================
def blocks_apply_action(state: BlocksState, action: int) -> BlocksState:
    """Apply one gripper action. Returns a NEW state; never mutates ``state``."""
    stacks = [list(s) for s in state.stacks]
    pos, held = state.pos, state.held
    n = len(stacks)

    if action == A_LEFT:
        if pos > 0:
            pos -= 1
    elif action == A_RIGHT:
        if pos < n - 1:
            pos += 1
    elif action == A_PICK:
        if held == NO_BLOCK and stacks[pos]:
            held = stacks[pos].pop()
    elif action == A_DROP:
        if held != NO_BLOCK:
            stacks[pos].append(held)
            held = NO_BLOCK
    # any invalid action / blocked move -> no-op (handled by the guards above)
    return BlocksState(stacks=tuple(tuple(s) for s in stacks), pos=pos,
                       held=held, target=state.target)


# =========================================================================
# Fixed instances (like a Sokoban layout)
# =========================================================================
INSTANCES: Dict[str, Dict[str, Any]] = {
    # 3 columns, blocks 1/2/3 each alone -> stack them all on column 0 as (1,2,3).
    "stack3": {
        "init": ((1,), (2,), (3,)),
        "target": ((1, 2, 3), (), ()),
        "cap": 3,
    },
    # 4 columns, harder: blocks spread, target is two stacks.
    "swap4": {
        "init": ((1,), (2,), (3,), (4,)),
        "target": ((1, 2), (), (3, 4), ()),
        "cap": 4,
    },
    # --- Complex instances (small state space, but longer / trickier plans) ---
    # Reversal: a tower must be turned upside-down using the empty columns as
    # buffers. Long plan (~12-16) but small state space.
    "reversal3": {
        "init": ((1, 2, 3), (), ()),
        "target": ((3, 2, 1), (), ()),
        "cap": 3,
    },
    # Two given towers must be SWAPPED between columns (needs a buffer column).
    "swaptowers": {
        "init": ((1, 2), (), (3, 4)),
        "target": ((3, 4), (), (1, 2)),
        "cap": 4,
    },
    # 5 blocks into one ordered tower from scattered singles.
    "stack5": {
        "init": ((1,), (2,), (3,), (4,), (5,)),
        "target": ((1, 2, 3, 4, 5), (), (), (), ()),
        "cap": 5,
    },
    # Interleave: build two towers with a specific non-adjacent ordering.
    "interleave4": {
        "init": ((4, 1), (), (2, 3)),
        "target": ((1, 2), (), (3, 4)),
        "cap": 4,
    },
    # --- Scaling block COUNT on few columns (branching stays low) -------------
    # Reverse a 4-tower on 3 columns (longer plan than reversal3, same 3 cols).
    "reversal4": {
        "init": ((1, 2, 3, 4), (), ()),
        "target": ((4, 3, 2, 1), (), ()),
        "cap": 4,
    },
    # Reverse a 5-tower on 3 columns.
    "reversal5": {
        "init": ((1, 2, 3, 4, 5), (), ()),
        "target": ((5, 4, 3, 2, 1), (), ()),
        "cap": 5,
    },
    # Gather 6 scattered blocks into one ordered tower on 3 columns.
    "gather6": {
        "init": ((2, 1), (4, 3), (6, 5)),
        "target": ((1, 2, 3, 4, 5, 6), (), ()),
        "cap": 6,
    },
    # ===== SCALING LADDER =====================================================
    # Length axis (fixed 3 columns, reverse an N-tower -> plan grows with N).
    "reversal6": {"init": ((1, 2, 3, 4, 5, 6), (), ()),
                  "target": ((6, 5, 4, 3, 2, 1), (), ()), "cap": 6},
    "reversal7": {"init": ((1, 2, 3, 4, 5, 6, 7), (), ()),
                  "target": ((7, 6, 5, 4, 3, 2, 1), (), ()), "cap": 7},
    # Column / branching axis (N singles on N columns -> one ordered N-tower).
    "stack4c": {"init": ((1,), (2,), (3,), (4,)),
                "target": ((1, 2, 3, 4), (), (), ()), "cap": 4},
    "stack6c": {"init": ((1,), (2,), (3,), (4,), (5,), (6,)),
                "target": ((1, 2, 3, 4, 5, 6), (), (), (), (), ()), "cap": 6},
    "stack7c": {"init": ((1,), (2,), (3,), (4,), (5,), (6,), (7,)),
                "target": ((1, 2, 3, 4, 5, 6, 7), (), (), (), (), (), ()), "cap": 7},
    # ===== RANDOMIZED-INIT FAMILY ============================================
    # Analog of MiniGrid's randomized layouts: each reset(seed) samples a fresh
    # initial configuration (see _sample_random_stacks for the exact
    # distribution), target fixed (sorted tower on column 0). Support ~2·10^4
    # configurations x 3 gripper positions, so demos cannot cover the eval
    # distribution -> discriminates generalization (tabular/BC replay dies).
    # "init" here is only a SHAPE PLACEHOLDER (ncols/decode template); the real
    # init is sampled in reset() / the hardcoded initial model.
    "rand6": {"init": ((6, 5, 4, 3, 2, 1), (), ()),
              "target": ((1, 2, 3, 4, 5, 6), (), ()), "cap": 6,
              "random_init": True},
}


def _sample_random_stacks(nblocks: int, ncols: int) -> Tuple[Tuple[int, ...], ...]:
    """Sample an initial configuration: blocks 1..nblocks are dealt in a
    uniformly random order, each onto a uniformly random column (stack order =
    deal order). Uses the module `random` RNG, seeded by env.reset(seed)."""
    import random
    order = list(range(1, nblocks + 1))
    random.shuffle(order)
    cols: List[List[int]] = [[] for _ in range(ncols)]
    for b in order:
        cols[random.randrange(ncols)].append(b)
    return tuple(tuple(c) for c in cols)


def _sample_initial_stacks(cfg: Dict[str, Any]) -> Tuple[Tuple[int, ...], ...]:
    """Fixed init for classic instances; freshly sampled (never already equal
    to the target) for random_init instances."""
    if not cfg.get("random_init", False):
        return cfg["init"]
    nblocks = sum(len(c) for c in cfg["init"])
    ncols = len(cfg["init"])
    while True:
        stacks = _sample_random_stacks(nblocks, ncols)
        if stacks != cfg["target"]:
            return stacks


# =========================================================================
# Empty generators + GT model generators
# =========================================================================
def blocks_empty_observation_gen(cap: int = 3) -> BlocksObservation:
    return BlocksObservation(current_stack=(), held=NO_BLOCK,
                             image=_obs_image((), NO_BLOCK, cap))


def blocks_empty_state_gen(instance: str = "stack3") -> BlocksState:
    cfg = INSTANCES[instance]
    return BlocksState(stacks=cfg["init"], pos=0, held=NO_BLOCK, target=cfg["target"])


def _observe(state: BlocksState, cap: int) -> BlocksObservation:
    cur = state.stacks[state.pos]
    return BlocksObservation(current_stack=cur, held=state.held,
                             image=_obs_image(cur, state.held, cap))


def blocks_hardcoded_transition_model_gen():
    def transition_model(state: BlocksState, action: int) -> BlocksState:
        return blocks_apply_action(state, action)
    return transition_model


def blocks_hardcoded_reward_model_gen():
    def reward_model(state: BlocksState, action: int, next_state: BlocksState) -> Tuple[float, bool]:
        if next_state.is_solved():
            return 1.0, True
        return 0.0, False
    return reward_model


def blocks_hardcoded_observation_model_gen(cap: int = 3):
    def observation_model(state: BlocksState, action: Optional[int],
                          empty_obs: BlocksObservation) -> BlocksObservation:
        return _observe(state, cap)
    return observation_model


def blocks_hardcoded_initial_model_gen(instance: str = "stack3"):
    """Ground-truth rho_0 (ORACLE use only): the true layout distribution with
    UNKNOWN gripper start position. Static instances -> fixed stacks + uniform
    position; random_init instances -> the same sampler the env uses."""
    cfg = INSTANCES[instance]

    def initial_model(empty_state: BlocksState) -> BlocksState:
        import random
        n = len(cfg["init"])
        stacks = _sample_initial_stacks(cfg)
        pos = random.randrange(n)
        return BlocksState(stacks=stacks, pos=pos, held=NO_BLOCK, target=cfg["target"])

    return initial_model


# =========================================================================
# Environment
# =========================================================================
class BlocksEnvironment(Environment[BlocksState, int, BlocksObservation]):
    def __init__(self, instance: str = "stack3", max_steps: int = 60, **kwargs: Any) -> None:
        super().__init__(max_steps=max_steps)
        self.instance = instance
        self.cap = INSTANCES[instance]["cap"]
        self._state: Optional[BlocksState] = None
        self._last_obs: Optional[BlocksObservation] = None
        self._step_count = 0

    def reset(self, seed: int = 0) -> BlocksState:
        import random
        random.seed(seed)
        cfg = INSTANCES[self.instance]
        n = len(cfg["init"])
        stacks = _sample_initial_stacks(cfg)
        pos = random.randrange(n)  # unknown start position
        self._state = BlocksState(stacks=stacks, pos=pos, held=NO_BLOCK, target=cfg["target"])
        self._step_count = 0
        self._last_obs = _observe(self._state, self.cap)
        return self._state

    def _observe(self, state: BlocksState) -> BlocksObservation:
        return _observe(state, self.cap)

    def get_last_obs(self) -> BlocksObservation:
        assert self._last_obs is not None
        return self._last_obs

    def step(self, action: int) -> Tuple[BlocksObservation, BlocksState, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None
        next_state = blocks_apply_action(self._state, action)
        reward = 1.0 if next_state.is_solved() else 0.0
        terminated = next_state.is_solved()
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        self._state = next_state
        self._last_obs = _observe(next_state, self.cap)
        return self._last_obs, next_state, reward, terminated, truncated, {}
