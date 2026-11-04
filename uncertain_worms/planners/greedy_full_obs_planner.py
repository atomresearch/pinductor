"""Optimal reference policy for PopGym MineSweeper under full observability.

This is the `Optimal` row of the paper's reference table: what the task is worth
to an agent that sees the hidden grid and plays perfectly. It is not a baseline
anyone could deploy, it is the upper bound the partially observable methods are
measured against.

Optimality is exact here rather than empirical, and follows from the reward
structure in `popgym/envs/minesweeper.py`:

    max_episode_length   = rows * cols - num_mines
    success_reward_scale = 1 / max_episode_length          # a fresh safe click
    fail_reward_scale    = -0.5 - success_reward_scale     # a mine, terminal
    bad_action_reward_scale = -0.5 / (max_episode_length - 2)   # already viewed

Every fresh safe click pays exactly `1 / max_episode_length` and turns one CLEAR
square into VIEWED. The episode ends when no CLEAR square is left. A policy that
only ever clicks CLEAR squares therefore plays exactly `max_episode_length`
steps and collects `max_episode_length * (1 / max_episode_length) = 1.0`. The
two other action classes both pay negative and neither creates a new CLEAR
square, so no policy can beat 1.0 and any deviation is strictly worse. One-step
lookahead is enough: there is nothing to trade off across time.

`assert_optimal_return` checks the implementation against that constant instead
of against a remembered number.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# popgym encodes the hidden grid as 0 = CLEAR, 1 = MINE, 2 = VIEWED.
CLEAR, MINE, VIEWED = 0, 1, 2


class GreedyFullObsPlanner:
    """Clicks a CLEAR square, never a mine, never one already viewed."""

    def __init__(self, actions: Optional[list] = None, **_: Any) -> None:
        self.actions = actions

    def __call__(self, state: Any, *_: Any, **__: Any) -> int:
        return self.plan(state)

    def plan(self, state: Any, *_: Any, **__: Any) -> int:
        grid = np.asarray(_hidden_grid(state)).reshape(-1)
        clear = np.flatnonzero(grid == CLEAR)
        if clear.size:
            return int(clear[0])
        # No CLEAR square left. The environment terminates on this condition, so
        # this is only reached if a caller keeps planning past the end; return a
        # valid action rather than raising.
        return 0

    def reset(self) -> None:
        pass


def _hidden_grid(state: Any) -> Any:
    """The hidden grid, whatever wrapper the state arrives in."""
    for attr in ("hidden_grid", "grid", "state", "obs"):
        if hasattr(state, attr):
            return getattr(state, attr)
    return state


def optimal_return(rows: int, cols: int, num_mines: int) -> float:
    """The closed-form full-clear return. Exactly 1.0 for every difficulty."""
    max_len = rows * cols - num_mines
    return max_len * (1.0 / max_len)


def assert_optimal_return(difficulty: str = "easy", episodes: int = 30,
                          seed: int = 0, tol: float = 1e-12) -> float:
    """Play the policy on the real environment and check it hits the constant.

    Returns the mean undiscounted return. Raises if it differs from the
    closed-form value by more than `tol`.
    """
    from popgym.envs.minesweeper import MineSweeper

    env = MineSweeper(difficulty=difficulty)
    target = optimal_return(*env.dims, env.num_mines)
    policy = GreedyFullObsPlanner()

    totals = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            a = policy.plan(env.hidden_grid)
            _, r, terminated, truncated, _ = env.step(
                np.unravel_index(a, env.dims))
            total += float(r)
            done = terminated or truncated
        totals.append(total)

    mean = float(np.mean(totals))
    if abs(mean - target) > tol:
        raise AssertionError(
            f"{difficulty}: mean return {mean!r} differs from the closed-form "
            f"{target!r} by {abs(mean - target):.3e}")
    return mean


if __name__ == "__main__":
    for d in ("easy", "medium", "hard"):
        m = assert_optimal_return(d)
        print(f"  {d:7s} mean undiscounted return {m:.15f}  matches closed form")
