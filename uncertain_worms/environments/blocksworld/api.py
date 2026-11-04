# type: ignore
# ─────────────────────────────────────────────────────────────────────
# LLM context template — NOT an importable module. Read as plain text and
# injected into the prompt so the model knows the Blocks World state/observation
# API. The authoritative implementations live in blocksworld_env.py.
#
# IMPORTANT FOR GENERATED CODE
# - transition_func / initial_func MUST return BlocksState(stacks=..., pos=..., held=..., target=...).
# - observation_func MUST return BlocksObservation(current_stack=..., held=..., image=...).
# - reward_func MUST return (float_reward, bool_done).
# - Never mutate the input state; build new tuples and return a NEW BlocksState.
# - observation_func must handle action=None for the reset observation.
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# --- Actions (an action either moves the gripper or manipulates a block) ---
A_LEFT = 0
A_RIGHT = 1
A_PICK = 2
A_DROP = 3
NO_BLOCK = 0  # the gripper holds nothing


@dataclass
class BlocksObservation(Observation):
    """What the gripper perceives each step: the contents of the stack it is
    CURRENTLY at (bottom -> top) and the block it holds. It does NOT reveal the
    gripper's absolute position nor the other stacks.

    Args:
        current_stack: tuple of block ids at the current column (bottom -> top).
        held: the carried block id, or 0 (NO_BLOCK) if empty-handed.
        image: a fixed-length int vector (the current stack padded to ``cap``
            with zeros, then the held block) used for soft matching.
    """

    current_stack: Tuple[int, ...]
    held: int
    image: NDArray[np.int8]


@dataclass
class BlocksState(State):
    """A Blocks-World configuration plus the gripper.

    ``stacks[i]`` is the tuple of block ids at column i (bottom -> top); blocks
    are distinct positive integers. ``pos`` is the gripper column, ``held`` the
    carried block (0 = none), ``target`` the static goal configuration. The goal
    is reached when ``stacks == target`` and nothing is held."""

    stacks: Tuple[Tuple[int, ...], ...]
    pos: int
    held: int
    target: Tuple[Tuple[int, ...], ...]

    @property
    def num_pos(self) -> int:
        return len(self.stacks)

    def is_solved(self) -> bool:
        return self.held == NO_BLOCK and self.stacks == self.target

# The constants A_LEFT, A_RIGHT, A_PICK, A_DROP, NO_BLOCK and the classes
# BlocksState / BlocksObservation are available to your generated code.
