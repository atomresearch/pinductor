"""State sampler for Blocks World (mirrors the Sokoban one)."""
from __future__ import annotations

from typing import Callable

from uncertain_worms.environments.blocksworld.blocksworld_env import BlocksState


def make_blocks_state_sampler(reference_state: BlocksState) -> Callable[[], BlocksState]:
    def _sample() -> BlocksState:
        return reference_state.copy()

    return _sample
