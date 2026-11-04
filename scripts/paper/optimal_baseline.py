"""Computes the "Optimal (full observability)" upper bound per environment.

This is the reference row the learned agents are compared against: the
return an agent would obtain if it knew the hidden state exactly. It is
solved for, not learned, so it depends only on the environment and the
evaluation grid.

The grid must match the one the runs use, or the bound is not comparable
to them. Episodes are seeded as ``seed * num_episodes + episode`` (as in
``main.py``), over seeds 0..9 and 3 episodes, giving 30 instances.

Usage:
    python scripts/paper/optimal_baseline.py --part minigrid
    python scripts/paper/optimal_baseline.py --part minesweeper
    python scripts/paper/optimal_baseline.py --part blocks
    python scripts/paper/optimal_baseline.py --part all

Part 1 — MiniGrid
-----------------
Breadth-first search over the state space, using the hand-written
transition model, for the shortest action sequence reaching the goal.
Reaching the goal terminates with reward 1, lava terminates with 0, and
states that terminate without reward are not expanded further.

Part 2 — MineSweeper
--------------------
No search needed: with full observability the optimal policy probes every
safe cell, in any order, hitting neither a mine nor a repeat. The return
then follows from the environment's reward scale, spelled out in
``_compute_minesweeper_optimal``.

Part 3 — Blocks
---------------
Breadth-first search again, but over the fully observed Blocks state; see
the comment above ``BLOCKS_ACTIONS`` for why the depth bound is what it is.
"""

from __future__ import annotations

import argparse
import collections
import copy
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Setup path so uncertain_worms is importable
# ---------------------------------------------------------------------------
# scripts/paper/ -> repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def discounted_reward(rewards: list[float], gamma: float) -> float:
    """Compute discounted return (matches main.py exactly)."""
    discounted_sum = 0.0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + gamma * discounted_sum
    return discounted_sum


# ---------------------------------------------------------------------------
# Part 1: MiniGrid BFS
# ---------------------------------------------------------------------------

MINIGRID_ENVS = [
    dict(env_name="CornerGoalRandom-Empty-10x10-v0", max_steps=40,  gamma=0.98, num_episodes=3),
    dict(env_name="MiniGrid-Empty-5x5-v0",           max_steps=10,  gamma=0.98, num_episodes=3),
    dict(env_name="MyMiniGrid-FourRooms-v0",         max_steps=60,  gamma=0.98, num_episodes=3),
    dict(env_name="MyMiniGrid-LavaWall-v0",          max_steps=40,  gamma=0.98, num_episodes=3),
    dict(env_name="MyUnlockEnv-v0",                  max_steps=40,  gamma=0.98, num_episodes=3),
]

VALID_ACTIONS = list(range(6))  # 0..5; action 6 raises


def _register_custom_envs():
    """Register custom MiniGrid envs (idempotent)."""
    import uncertain_worms.environments.minigrid.custom_environments  # noqa: F401 — registers on import


def _reset_env_and_get_state(env_name: str, ep_seed: int):
    """Reset MinigridEnvironment and return the initial MinigridState."""
    import gymnasium as gym
    from uncertain_worms.environments.minigrid.minigrid_env import (
        MinigridEnvironment,
    )

    env = MinigridEnvironment(env_name=env_name, fully_obs=False, max_steps=9999)
    state = env.reset(seed=ep_seed)
    env.env.close()
    return state


def _bfs_minigrid(initial_state, max_steps: int, transition_fn, reward_fn):
    """BFS over full MinigridState space.

    Returns the shortest path length L* (number of actions) to reach the
    goal, or None if no solution within max_steps.

    Dead states (lava termination without reward) are NOT expanded further.
    """
    from uncertain_worms.environments.minigrid.minigrid_env import MinigridState

    # BFS: (state, depth)
    queue = collections.deque()
    queue.append((initial_state, 0))
    visited = {initial_state}

    while queue:
        state, depth = queue.popleft()

        if depth >= max_steps:
            continue

        for action in VALID_ACTIONS:
            next_state = transition_fn(state, action)
            reward, terminated = reward_fn(state, action, next_state)

            if terminated:
                if reward > 0:
                    # Goal reached
                    return depth + 1
                else:
                    # Lava / dead end — do not expand
                    continue

            # Non-terminal step
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, depth + 1))

    return None  # no solution found within max_steps


def run_minigrid_part(seeds=range(10), num_episodes=3, verbose=True):
    """Run Part 1: MiniGrid BFS upper-bound computation."""
    _register_custom_envs()

    from uncertain_worms.environments.minigrid.minigrid_env import (
        minigrid_hardcoded_transition_model_gen,
        minigrid_hardcoded_reward_model_gen,
    )

    transition_fn = minigrid_hardcoded_transition_model_gen()
    reward_fn = minigrid_hardcoded_reward_model_gen()

    rows = []  # (env_name, seed, episode, ep_seed, L_star, optimal_return)

    for cfg in MINIGRID_ENVS:
        env_name = cfg["env_name"]
        max_steps = cfg["max_steps"]
        gamma = cfg["gamma"]
        n_ep = cfg["num_episodes"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"ENV: {env_name}  max_steps={max_steps}  gamma={gamma}")

        seed_means = []
        no_solution_count = 0
        total_instances = 0

        for seed in seeds:
            ep_returns = []
            for episode in range(n_ep):
                ep_seed = seed * n_ep + episode

                # Seed Python random + numpy (mirrors main.py exactly)
                random.seed(ep_seed)
                np.random.seed(ep_seed)

                initial_state = _reset_env_and_get_state(env_name, ep_seed)

                L_star = _bfs_minigrid(
                    initial_state, max_steps, transition_fn, reward_fn
                )
                total_instances += 1

                if L_star is None:
                    no_solution_count += 1
                    opt_return = 0.0
                    if verbose:
                        print(
                            f"  seed={seed} ep={episode} ep_seed={ep_seed}  "
                            f"L*=NO_SOLUTION -> return=0.0  [FLAG: investigate!]"
                        )
                else:
                    # Discounted return for binary-reward episode:
                    # rewards = [0, 0, ..., 0, 1] with L_star steps
                    rewards = [0.0] * (L_star - 1) + [1.0]
                    opt_return = discounted_reward(rewards, gamma)
                    if verbose:
                        print(
                            f"  seed={seed} ep={episode} ep_seed={ep_seed}  "
                            f"L*={L_star}  return={opt_return:.4f}"
                        )

                rows.append(
                    (env_name, seed, episode, ep_seed, L_star, opt_return)
                )
                ep_returns.append(opt_return)

            seed_means.append(np.mean(ep_returns))

        mean_return = float(np.mean(seed_means))
        std_return = float(np.std(seed_means))

        # Sanity checks
        no_sol_frac = no_solution_count / max(total_instances, 1)
        if verbose:
            print(f"\n  SUMMARY {env_name}:")
            print(f"    mean_return (over 10 seed-means) = {mean_return:.4f}")
            print(f"    std_return                        = {std_return:.4f}")
            print(f"    no-solution fraction              = {no_sol_frac:.2%}  "
                  f"(flag if > 5%: {'FLAG' if no_sol_frac > 0.05 else 'OK'})")

    # Print hand-check for Empty-5x5
    if verbose:
        print("\n" + "="*60)
        print("HAND-CHECK: MiniGrid-Empty-5x5-v0")
        print("  5x5 grid = 3x3 interior. Agent starts deterministically at (1,1)")
        print("  facing right (dir=0). Goal always at (3,3) (bottom-right interior).")
        print("  Shortest path: forward 2 steps + turn right + forward 2 steps = 4 steps?")
        print("  Actually: turn right (face down, dir=1), forward, forward, turn left,")
        print("  forward, forward -> 6 steps (Manhattan = 4, but need turns).")
        print("  BFS will give exact answer. Checking ep_seed=0 (seed=0, episode=0):")
        random.seed(0)
        np.random.seed(0)
        state_e5 = _reset_env_and_get_state("MiniGrid-Empty-5x5-v0", 0)
        print(f"  initial state:\n{state_e5}")
        from uncertain_worms.environments.minigrid.minigrid_env import ObjectTypes
        goal_pos = state_e5.get_type_indices(int(ObjectTypes.goal))
        print(f"  goal_pos={goal_pos}  agent_pos={state_e5.agent_pos}  agent_dir={state_e5.agent_dir}")

    return rows


# ---------------------------------------------------------------------------
# Part 2: MineSweeper optimal
# ---------------------------------------------------------------------------

MINESWEEPER_CONFIGS = [
    dict(env_id="MineSweeperEasy",   max_steps=16, gamma=0.98, num_episodes=3,
         dims=(4, 4), num_mines=2),
    dict(env_id="MineSweeperMedium", max_steps=30, gamma=0.98, num_episodes=3,
         dims=(6, 6), num_mines=6),
    dict(env_id="MineSweeperHard",   max_steps=54, gamma=0.98, num_episodes=3,
         dims=(8, 8), num_mines=10),
]


def _compute_minesweeper_optimal(env_id: str, ep_seed: int, dims, num_mines, gamma: float):
    """
    Reset the MineSweeper env with ep_seed, read mine layout, simulate
    probing all safe tiles in step order (row-major), compute discounted return.

    Reward scheme (from minesweeper.py):
      max_ep_len = dims[0]*dims[1] - num_mines  (= number of safe tiles)
      success_reward_scale = 1 / max_ep_len       (per safe tile probed)
      bad_action_reward_scale = -0.5 / (max_ep_len - 2)  (repeated click)
      fail_reward_scale = -0.5 - success_reward_scale     (mine click)

    Optimal: click all safe tiles exactly once, no mines, no repeats.
    Rewards = [success_reward_scale] * max_ep_len
    Order: row-major (index 0 first) — doesn't matter since step count is
    always max_ep_len and rewards are identical per step.
    """
    from popgym.envs.minesweeper import MineSweeper

    env = MineSweeper.__new__(MineSweeper)
    env.__init__(difficulty={
        "MineSweeperEasy": "easy",
        "MineSweeperMedium": "medium",
        "MineSweeperHard": "hard",
    }[env_id])

    # Reset with the ep_seed (mirrors PopGymEnvironment.reset)
    env.reset(seed=ep_seed)

    # Get mine layout from the hidden grid
    # hidden_grid: shape dims, values 0=CLEAR, 1=MINE, 2=VIEWED
    hidden_grid = env.hidden_grid.copy()  # (dims[0], dims[1])

    n_cells = dims[0] * dims[1]
    n_safe = n_cells - num_mines  # = max_episode_length
    success_reward = env.success_reward_scale  # = 1/n_safe

    # Optimal: probe n_safe safe cells, each gives success_reward
    rewards = [success_reward] * n_safe

    # Compute discounted return
    opt_return = discounted_reward(rewards, gamma)

    # Cross-check mine count
    mine_count = int(np.sum(hidden_grid == 1))
    assert mine_count == num_mines, (
        f"Expected {num_mines} mines, got {mine_count} for ep_seed={ep_seed}"
    )

    return opt_return, n_safe


def run_minesweeper_part(seeds=range(10), num_episodes=3, verbose=True):
    """Optimal MineSweeper return, discounted.

    This function reports the discounted return, like the other two families.
    The quantity the paper reports for MineSweeper is the undiscounted return,
    where a full clear is exactly 1.0 by construction of the reward; it is
    computed by `uncertain_worms/planners/greedy_full_obs_planner.py`, which
    checks it against the closed form.
    """
    rows = []  # (env_id, seed, episode, ep_seed, L_star, optimal_return)

    for cfg in MINESWEEPER_CONFIGS:
        env_id = cfg["env_id"]
        dims = cfg["dims"]
        num_mines = cfg["num_mines"]
        gamma = cfg["gamma"]
        n_ep = cfg["num_episodes"]

        n_safe = dims[0] * dims[1] - num_mines
        success_reward = 1.0 / n_safe

        if verbose:
            print(f"\n{'='*60}")
            print(f"ENV: {env_id}  dims={dims}  num_mines={num_mines}")
            print(f"  n_safe={n_safe}  success_reward_per_step={success_reward:.6f}")
            print(f"  gamma={gamma}")

        seed_means = []

        for seed in seeds:
            ep_returns = []
            for episode in range(n_ep):
                ep_seed = seed * n_ep + episode

                # Mirror main.py seeding
                random.seed(ep_seed)
                np.random.seed(ep_seed)

                opt_return, n_steps = _compute_minesweeper_optimal(
                    env_id, ep_seed, dims, num_mines, gamma
                )

                if verbose:
                    print(
                        f"  seed={seed} ep={episode} ep_seed={ep_seed}  "
                        f"n_safe={n_steps}  return={opt_return:.4f}"
                    )

                rows.append(
                    (env_id, seed, episode, ep_seed, n_steps, opt_return)
                )
                ep_returns.append(opt_return)

            seed_means.append(np.mean(ep_returns))

        mean_return = float(np.mean(seed_means))
        std_return = float(np.std(seed_means))

        if verbose:
            print(f"\n  SUMMARY {env_id}:")
            print(f"    mean_return (over 10 seed-means) = {mean_return:.4f}")
            print(f"    std_return                        = {std_return:.4f}")
            print(f"    NOTE: MineSweeper optimal return is DETERMINISTIC per env_id")
            print(f"          because n_safe is fixed and all safe-tile rewards are equal.")
            # Verify: rewards are always [1/n_safe]*n_safe, so return is fixed
            rewards_check = [success_reward] * n_safe
            fixed_return = discounted_reward(rewards_check, gamma)
            print(f"    Fixed return (independent of seed) = {fixed_return:.6f}")

    return rows


# ---------------------------------------------------------------------------
# Output CSV
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Part 3 — Blocks
# ---------------------------------------------------------------------------
# Breadth-first search over the fully observed Blocks state, from the state the
# episode actually starts in. rand6 scatters the blocks afresh every episode, so
# the optimal length is a property of the instance, not of the layout: it ranges
# from 34 to 62 moves over the evaluation grid. The depth bound is 120 rather
# than something tighter because two of the thirty instances need 62 moves, and
# cutting the search short scores them zero and drags the mean down by 0.02.
BLOCKS_ACTIONS = [0, 1, 2, 3]
BLOCKS_DEPTH_CAP = 120
BLOCKS_NODE_CAP = 5_000_000


def _bfs_blocks(start, apply_action):
    seen = {(start.stacks, start.pos, start.held)}
    queue = collections.deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if state.is_solved():
            return depth
        if depth > BLOCKS_DEPTH_CAP or len(seen) > BLOCKS_NODE_CAP:
            return None
        for action in BLOCKS_ACTIONS:
            nxt = apply_action(state, action)
            key = (nxt.stacks, nxt.pos, nxt.held)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, depth + 1))
    return None


def run_blocks_part(seeds=range(10), num_episodes=3, gamma=0.98, verbose=True):
    """Optimal discounted return on Blocks rand6 over the evaluation grid."""
    from uncertain_worms.environments.blocksworld.blocksworld_env import (
        BlocksEnvironment, blocks_apply_action)

    env = BlocksEnvironment(instance="rand6", max_steps=120)
    rows = []
    for seed in seeds:
        for episode in range(num_episodes):
            ep_seed = seed * num_episodes + episode
            env.reset(seed=ep_seed)
            length = _bfs_blocks(env._state, blocks_apply_action)
            ret = gamma ** (length - 1) if length else 0.0
            rows.append(("Blocks-rand6", seed, episode, ep_seed, length, ret))
            if verbose:
                print(f"  seed={seed} ep={episode} ep_seed={ep_seed:<3} "
                      f"L*={length} return={ret:.6f}")
    return rows


def write_csv(rows_minigrid, rows_minesweeper, out_path: str):
    """Write combined CSV with all results."""
    import csv

    header = ["env", "seed", "episode", "ep_seed", "L_star", "optimal_return"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows_minigrid:
            env_name, seed, episode, ep_seed, L_star, opt_return = row
            writer.writerow([env_name, seed, episode, ep_seed,
                             "" if L_star is None else L_star, f"{opt_return:.6f}"])
        for row in rows_minesweeper:
            env_id, seed, episode, ep_seed, n_steps, opt_return = row
            writer.writerow([env_id, seed, episode, ep_seed, n_steps, f"{opt_return:.6f}"])

    print(f"\nCSV written to {out_path}")


def print_summary(rows_minigrid, rows_minesweeper, rows_blocks=()):
    """Print per-env mean ± std summary table."""
    print("\n" + "="*70)
    print("OPTIMAL UPPER-BOUND SUMMARY  (mean ± std over 10 seed-means)")
    print("MineSweeper rows are discounted here; the paper reports them")
    print("undiscounted, where a full clear is exactly 1.0.")
    print("="*70)
    print(f"{'Env':<42} {'Mean':>8}  {'Std':>8}  {'gamma':>6}")
    print("-"*70)

    # MiniGrid
    gamma_map = {c["env_name"]: c["gamma"] for c in MINIGRID_ENVS}
    n_ep_map = {c["env_name"]: c["num_episodes"] for c in MINIGRID_ENVS}

    env_groups: dict = {}
    for env_name, seed, episode, ep_seed, L_star, opt_return in rows_minigrid:
        if env_name not in env_groups:
            env_groups[env_name] = {}
        if seed not in env_groups[env_name]:
            env_groups[env_name][seed] = []
        env_groups[env_name][seed].append(opt_return)

    for cfg in MINIGRID_ENVS:
        env_name = cfg["env_name"]
        gamma = cfg["gamma"]
        if env_name not in env_groups:
            continue
        seed_means = [np.mean(v) for v in env_groups[env_name].values()]
        mean_r = float(np.mean(seed_means))
        std_r = float(np.std(seed_means))
        print(f"  {env_name:<40} {mean_r:>8.4f}  {std_r:>8.4f}  {gamma:>6.3f}")

    # MineSweeper
    ms_groups: dict = {}
    for env_id, seed, episode, ep_seed, n_steps, opt_return in rows_minesweeper:
        if env_id not in ms_groups:
            ms_groups[env_id] = {}
        if seed not in ms_groups[env_id]:
            ms_groups[env_id][seed] = []
        ms_groups[env_id][seed].append(opt_return)

    gamma_ms = {c["env_id"]: c["gamma"] for c in MINESWEEPER_CONFIGS}
    for cfg in MINESWEEPER_CONFIGS:
        env_id = cfg["env_id"]
        gamma = cfg["gamma"]
        if env_id not in ms_groups:
            continue
        seed_means = [np.mean(v) for v in ms_groups[env_id].values()]
        mean_r = float(np.mean(seed_means))
        std_r = float(np.std(seed_means))
        print(f"  {env_id:<40} {mean_r:>8.4f}  {std_r:>8.4f}  {gamma:>6.3f}")

    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

    if rows_blocks:
        returns = [r[5] for r in rows_blocks]
        lengths = [r[4] for r in rows_blocks if r[4]]
        mean = sum(returns) / len(returns)
        std = (sum((x - mean) ** 2 for x in returns) / len(returns)) ** 0.5
        print(f"{'Blocks-rand6':<42} {mean:>8.4f}  {std:>8.4f}  {0.98:>6.3f}")
        if len(lengths) < len(returns):
            print(f"{'':<42} {len(returns) - len(lengths)} instance(s) unsolved "
                  f"within the depth cap, scored 0")
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Compute optimal upper-bound baselines")
    parser.add_argument(
        "--part",
        choices=["minigrid", "minesweeper", "blocks", "all"],
        default="all",
        help="Which part to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seeds to evaluate (default 0..9)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "outputs", "optimal_baseline.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-instance output"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    seeds = args.seeds

    rows_minigrid = []
    rows_minesweeper = []

    rows_blocks = []
    if args.part in ("minigrid", "all"):
        print("\n" + "#"*60)
        print("# PART 1: MiniGrid BFS optimal upper-bound")
        print("#"*60)
        rows_minigrid = run_minigrid_part(seeds=seeds, verbose=verbose)

    if args.part in ("minesweeper", "all"):
        print("\n" + "#"*60)
        print("# PART 2: MineSweeper optimal upper-bound")
        print("#"*60)
        rows_minesweeper = run_minesweeper_part(seeds=seeds, verbose=verbose)

    if args.part in ("blocks", "all"):
        print("\n" + "#"*60)
        print("# PART 3: Blocks BFS optimal upper-bound")
        print("#"*60)
        rows_blocks = run_blocks_part(seeds=seeds, verbose=verbose)
        mean = sum(r[5] for r in rows_blocks) / len(rows_blocks)
        print(f"\n  Blocks-rand6  mean optimal return {mean:.4f} "
              f"over {len(rows_blocks)} instances")

    print_summary(rows_minigrid, rows_minesweeper, rows_blocks)
    write_csv(rows_minigrid, rows_minesweeper + rows_blocks, args.out)


if __name__ == "__main__":
    main()
