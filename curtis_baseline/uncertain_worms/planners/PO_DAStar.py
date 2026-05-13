from __future__ import annotations

import heapq
import itertools
import logging
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple

import networkx as nx  # type: ignore
import numpy as np
from pyvis.network import Network  # type: ignore

from uncertain_worms.planners.base_planner import PartiallyObservablePlanner
from uncertain_worms.structs import (
    ActType,
    CategoricalBelief,
    ObsType,
    ParticleBelief,
    StateType,
)
from uncertain_worms.utils import get_log_dir

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Helper utilities
# -----------------------------------------------------------------------------#


def normalize(count_dict: Dict[Any, int]) -> Dict[Any, float]:
    total = sum(count_dict.values())
    return {item: count / total for item, count in count_dict.items()}


def rollout_fn(fn: Callable, inputs: List[Any], num_rollouts: int) -> Counter[Any]:
    """Call a stochastic function fn(*inputs) num_rollouts times and count
    outcomes."""
    return Counter(fn(*inputs) for _ in range(num_rollouts))


# -----------------------------------------------------------------------------#
# Belief-only search-space node
# -----------------------------------------------------------------------------#


@dataclass(frozen=True)
class BeliefNode(Generic[StateType]):
    belief: CategoricalBelief
    terminal: bool
    expected_reward: float
    # last_belief: Optional[CategoricalBelief] = None
    # last_action: Optional[ActType] = None

    def __hash__(self) -> int:
        return hash((self.belief, self.terminal))

    def __repr__(self) -> str:
        return f"BN{hash(self)}"


# -----------------------------------------------------------------------------#
# Planner implementation
# -----------------------------------------------------------------------------#

def _is_descendant(child, potential_parent, came_from,
                   hard_cap: int = 10_000) -> bool:
    """Return True iff *child* is reachable from *potential_parent*.
    Traversal is cycle- and depth-safe."""
    cur = potential_parent
    visited = set()
    steps = 0

    while cur in came_from:
        if cur == child:          # found the child → descendant
            return True
        if cur in visited:        # loop detected
            return True           # treat as descendant to stay safe
        visited.add(cur)

        cur, _ = came_from[cur]   # climb one level
        steps += 1
        if steps >= hard_cap:     # absurdly deep chain → bail out
            logging.warning(
                "came_from depth > %d; assuming descendant to stay safe", hard_cap
            )
            return True

    return False  

class PO_DAStar(
    PartiallyObservablePlanner[StateType, ActType, ObsType, ParticleBelief]
):
    r"""Determinized‑A* (DA*) under partial observability, using combined
    reward-cost-entropy heuristic."""

    def __init__(
        self,
        empty_observation: ObsType,
        *args: Any,
        max_expansions: Optional[int] = None,
        visualize_graph: bool = False,
        entropy_coeff: float = 1.0,
        lambda_coeff: float = 1.0,
        num_rollouts: int = 1,
        action_cost: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.empty_observation = empty_observation
        self.max_expansions = max_expansions
        self.visualize_graph = visualize_graph
        self.entropy_coeff = entropy_coeff
        self.lambda_coeff = lambda_coeff
        self.num_rollouts = num_rollouts
        self.action_cost = action_cost
        super().__init__(*args, **kwargs)

    def save_search_graph(
        self,
        came_from: Dict[BeliefNode[ActType], Tuple[BeliefNode[ActType], ActType]],
        start_node: BeliefNode[ActType],
        expanded_steps: Dict[BeliefNode[ActType], int],
        cost_values: Dict[BeliefNode[ActType], float],
        all_edges: List[Tuple[BeliefNode[ActType], BeliefNode[ActType], ActType]],
        cost_so_far: Dict[BeliefNode[ActType], float],
    ) -> None:
        G = nx.DiGraph()
        parents = {parent for parent, _ in came_from.values()}
        nodes = set(came_from.keys()) | parents | {start_node}
        for node in nodes:
            node_id = str(node)
            step_info = expanded_steps.get(node, "N/A")
            reward_val = node.expected_reward
            cost_comp = cost_so_far.get(node, 0.0)
            ent_comp = self.entropy_coeff * node.belief.get_entropy()
            total = cost_values.get(node, reward_val - cost_comp + ent_comp)
            label = (
                f"{node_id}\n"
                f"Step: {step_info}\n"
                f"Obj: {total:.2f}\n"
                f"(R={reward_val:.2f}, λ·C={cost_comp:.2f}, E={ent_comp:.2f})"
            )
            color = (
                "red" if node.terminal else "purple" if node is start_node else "blue"
            )
            G.add_node(node_id, label=label, title=label, color=color)
        for parent, child, action in all_edges:
            G.add_edge(
                str(parent),
                str(child),
                label=str(action),
                title=f"Action: {action}",
            )
        nt = Network(height="800px", width="800px", directed=True)
        nt.from_nx(G)
        path = os.path.join(get_log_dir(), f"search_graph_{time.time()}.html")
        nt.write_html(path)
        log.info("Search graph saved to %s", path)

    def plan_next_action(
        self, belief_state: ParticleBelief, max_steps: int
    ) -> Tuple[ActType, Dict]:
        counter = itertools.count()
        open_set: List[Tuple[float, int, float, BeliefNode[ActType], int]] = []
        expanded_steps: Dict[BeliefNode[ActType], int] = {}
        cost_values: Dict[BeliefNode[ActType], float] = {}
        all_edges: List[Tuple[BeliefNode[ActType], BeliefNode[ActType], ActType]] = []

        start_belief = CategoricalBelief[StateType, ObsType](
            normalize(belief_state.particles)
        )
        start_node = BeliefNode[ActType](
            belief=start_belief,
            terminal=False,
            expected_reward=0.0,
        )

        print(f"Start entropy: {start_node.belief.get_entropy():.2f}")
        heapq.heappush(open_set, (0.0, next(counter), 0.0, start_node, 0))
        cost_values[start_node] = float("inf")

        came_from: Dict[BeliefNode[ActType], Tuple[BeliefNode[ActType], ActType]] = {}
        cost_so_far: Dict[BeliefNode[ActType], float] = {start_node: 0.0}
        closed = set()
        num_expansions = 0
        best_priority = cost_values[start_node]

        # ==================================================================#
        # A* loop
        # ==================================================================#
        while open_set:
            if (
                self.max_expansions is not None
                and num_expansions >= self.max_expansions
            ):
                break

            priority, _, current_g, current_node, steps = heapq.heappop(open_set)
            best_priority = (
                min(best_priority, priority) if num_expansions > 0 else priority
            )

            if current_node in closed:
                continue

            closed.add(current_node)

            # print(
            #     f"Best priority: {best_priority:.2f} | "
            #     f"num_expansions: {num_expansions} | "
            #     f"current priority: {priority:.2f} | "
            #     f"Num nodes: {len(cost_so_far)}"
            # )

            num_expansions += 1
            if steps >= max_steps or current_node.terminal:
                continue

            for action in self.actions:
                try:
                    total_outcome = defaultdict(float)
                    for state, p_s in current_node.belief.dist.items():
                        counts = rollout_fn(
                            self.transition_model, [state, action], self.num_rollouts
                        )
                        tot = sum(counts.values())
                        for s2, cnt in counts.items():
                            total_outcome[s2] += p_s * (cnt / tot)
                    merged = CategoricalBelief[StateType, ObsType](
                        dist=dict(total_outcome)
                    )
                except Exception:
                    log.info(traceback.format_exc())
                    continue

                branches: Dict[ObsType, Dict[StateType, float]] = defaultdict(
                    lambda: defaultdict(float)
                )
                try:
                    for s2, p_m in merged.dist.items():
                        obs_counts = rollout_fn(
                            self.observation_model,
                            [s2, action, self.empty_observation],
                            self.num_rollouts,
                        )
                        tot_o = sum(obs_counts.values())
                        for obs, cnt in obs_counts.items():
                            branches[obs][s2] += p_m * (cnt / tot_o)
                except Exception:
                    log.info(traceback.format_exc())
                    continue

                for obs, dist in branches.items():
                    prob = sum(dist.values())
                    if prob == 0.0:
                        continue

                    exp_r = 0.0
                    term_flags: List[bool] = []
                    for s_prev, p_prev in current_node.belief.dist.items():
                        for s_next, p_sn in dist.items():
                            p_joint = p_prev * (p_sn / prob)
                            try:
                                r, term = self.reward_model(s_prev, action, s_next)
                                exp_r += r * p_joint
                                term_flags.append(term)
                            except Exception:
                                log.info(traceback.format_exc())
                    is_term = all(term_flags)

                    norm_dist = {s: p / prob for s, p in dist.items()}
                    child_belief = CategoricalBelief[StateType, ObsType](dist=norm_dist)

                    new_cost = (
                        current_g
                        - exp_r
                        - self.lambda_coeff * np.log(prob)
                        + self.action_cost
                    )
                    ent = child_belief.get_entropy()

                    child = BeliefNode[ActType](
                        belief=child_belief, terminal=is_term, expected_reward=exp_r
                    )

                    priority = new_cost + self.entropy_coeff * ent

                    all_edges.append((current_node, child, action))
                    if child not in cost_so_far \
                        or new_cost < cost_so_far[child] \
                        and not _is_descendant(child, current_node, came_from):
                        cost_so_far[child] = new_cost
                        heapq.heappush(
                            open_set,
                            (priority, next(counter), new_cost, child, steps + 1),
                        )
                        came_from[child] = (current_node, action)
                        expanded_steps[child] = num_expansions
                        cost_values[child] = priority

        # ==================================================================#
        # Search finished – pick best candidate by same objective
        # ==================================================================#
        best = min(cost_values.keys(), key=lambda n: cost_values[n])
        plan = self._reconstruct_plan(best, came_from, start_node)

        if self.visualize_graph:
            self.save_search_graph(
                came_from,
                start_node,
                expanded_steps,
                cost_values,
                all_edges,
                cost_so_far,
            )
        if plan:
            return plan[0], {}

        log.info("No path discovered, defaulting to random action")
        return random.choice(self.actions), {}

    @staticmethod
    def _reconstruct_plan(
        node: BeliefNode[ActType],
        came_from: Dict[BeliefNode[ActType], Tuple[BeliefNode[ActType], ActType]],
        start_node: BeliefNode[ActType],
    ) -> List[ActType]:
        plan: List[ActType] = []
        cur = node
        while cur != start_node and cur in came_from:
            parent, action = came_from[cur]
            plan.append(action)
            cur = parent
        plan.reverse()
        return plan
