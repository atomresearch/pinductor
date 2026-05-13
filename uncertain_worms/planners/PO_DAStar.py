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

    _VALID_INFO_GAIN_MODES = (
        "entropy",
        "entropy_reduction",
        "expected_ig",
        "kl_divergence",
        "voi",
    )

    def __init__(
        self,
        empty_observation: ObsType,
        *args: Any,
        max_expansions: Optional[int] = None,
        visualize_graph: bool = False,
        entropy_coeff: float = 1.0,
        lambda_coeff: float = 1.0,
        num_rollouts: int = 10,
        action_cost: float = 0.01,
        info_gain_mode: str = "entropy",
        **kwargs: Any,
    ) -> None:
        if info_gain_mode not in self._VALID_INFO_GAIN_MODES:
            raise ValueError(
                f"info_gain_mode must be one of {self._VALID_INFO_GAIN_MODES}, "
                f"got {info_gain_mode!r}."
            )
        if max_expansions is not None and max_expansions < 1:
            raise ValueError(
                f"max_expansions must be ≥ 1 (or None), got {max_expansions}."
            )
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be ≥ 1, got {num_rollouts}.")

        self.empty_observation = empty_observation
        self.max_expansions = max_expansions
        self.visualize_graph = visualize_graph
        self.entropy_coeff = entropy_coeff
        self.lambda_coeff = lambda_coeff
        self.num_rollouts = num_rollouts
        self.action_cost = action_cost
        self.info_gain_mode = info_gain_mode
        super().__init__(*args, **kwargs)
        # The base class assigns ``self.actions``; validate now that it
        # is a non-empty list (audit MIN8 — empty list silently degraded
        # to a random action choice).
        if not getattr(self, "actions", None):
            raise ValueError(
                "PO_DAStar requires a non-empty action list."
            )

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

    def _priority_hook(
        self,
        *,
        new_cost: float,
        ig_term: float,
        mi_term: float,
        child: Any,
        parent_belief: Any,
        action: Any,
        exp_r: float,
    ) -> float:
        return new_cost + ig_term + mi_term

    def plan_next_action(
        self, belief_state: ParticleBelief, max_steps: int,
        mi_bonus: Optional[Dict[ActType, float]] = None,
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

        log.info(f"Start entropy: {start_node.belief.get_entropy():.2f}")
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

            if num_expansions <= 50 and steps <= 2:
                # Log first expansions for debugging A* behavior
                pos_summary = {str(tuple(s.agent_pos)): f"{p:.2f}" for s, p in list(current_node.belief.dist.items())[:5]}
                log.info(
                    f"[A*] exp={num_expansions} depth={steps} priority={priority:.3f} "
                    f"g={current_g:.3f} ent={current_node.belief.get_entropy():.3f} "
                    f"E[r]={current_node.expected_reward:.3f} "
                    f"pos={pos_summary}"
                )

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

                # Collect all observation branches for this action
                parent_ent = current_node.belief.get_entropy()
                branch_children: List[Tuple] = []  # (child, new_cost, ent, prob, exp_r)

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
                    branch_children.append((child, new_cost, ent, prob, exp_r))

                # Compute expected entropy across all branches (for modes 2/4)
                total_branch_prob = sum(bc[3] for bc in branch_children)
                expected_child_ent = (
                    sum(bc[3] * bc[2] for bc in branch_children) / total_branch_prob
                    if total_branch_prob > 0 else parent_ent
                )
                # Expected info gain = H(parent) - E[H(child|obs)]
                expected_ig = parent_ent - expected_child_ent

                for child, new_cost, ent, prob, exp_r in branch_children:
                    # ── Info gain modes ──
                    mode = self.info_gain_mode
                    if mode == "entropy":
                        # Original: penalise high child entropy
                        ig_term = self.entropy_coeff * ent
                    elif mode == "entropy_reduction":
                        # Mode 1: reward entropy reduction (parent - child)
                        ig_term = -self.entropy_coeff * (parent_ent - ent)
                    elif mode == "expected_ig":
                        # Mode 2/4: expected info gain across all obs branches
                        ig_term = -self.entropy_coeff * expected_ig
                    elif mode == "kl_divergence":
                        # Mode 3: KL(child || parent)
                        kl = 0.0
                        for s, p_c in child.belief.dist.items():
                            p_p = current_node.belief.dist.get(s, 1e-15)
                            if p_c > 1e-15:
                                kl += p_c * (np.log(p_c) - np.log(max(p_p, 1e-15)))
                        ig_term = -self.entropy_coeff * kl
                    elif mode == "voi":
                        # Mode 5: Value of Information approximation
                        # VoI ≈ max(E[r] over actions given child belief)
                        #      - max(E[r] over actions given parent belief)
                        # Approximated by: exp_r improvement from belief refinement
                        # Use expected_ig as proxy scaled by best available reward
                        ig_term = -self.entropy_coeff * expected_ig * max(abs(exp_r), 0.1)
                    else:
                        ig_term = self.entropy_coeff * ent  # fallback

                    # Model uncertainty bonus (BALD): reward actions where
                    # candidate models disagree, weighted inversely by
                    # reward confidence.  When reward is high (goal nearby),
                    # the bonus vanishes; when reward ≈ 0 (lost), it dominates.
                    mi_term = 0.0
                    if mi_bonus is not None and action in mi_bonus:
                        reward_confidence = max(abs(exp_r), 0.05)
                        mi_term = -mi_bonus[action] / reward_confidence

                    priority = self._priority_hook(
                        new_cost=new_cost, ig_term=ig_term, mi_term=mi_term,
                        child=child, parent_belief=current_node.belief,
                        action=action, exp_r=exp_r,
                    )

                    all_edges.append((current_node, child, action))
                    if num_expansions <= 20 and steps == 0:
                        action_names = ['LEFT','RIGHT','FWD','PICKUP','DROP','TOGGLE']
                        a_name = action_names[action] if action < 6 else str(action)
                        n_branches_for_action = len(branches)
                        mi_str = f" mi_term={mi_term:.3f}" if mi_term != 0.0 else ""
                        log.info(
                            f"[A*] depth=0 action={a_name} obs_branch: "
                            f"prob={prob:.3f} E[r]={exp_r:.4f} "
                            f"child_ent={ent:.3f} new_cost={new_cost:.3f} "
                            f"ig={ig_term:.3f}{mi_str} "
                            f"priority={priority:.3f} n_branches={n_branches_for_action}"
                        )
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

        # Log A* decision summary
        if plan:
            action_names = ['LEFT','RIGHT','FWD','PICKUP','DROP','TOGGLE']
            chosen = action_names[plan[0]] if plan[0] < 6 else str(plan[0])
            # Compute per-action best cost at depth 0
            depth0_actions = {}
            for child, (parent, act) in came_from.items():
                if parent == start_node:
                    a_name = action_names[act] if act < 6 else str(act)
                    if a_name not in depth0_actions or cost_values.get(child, float('inf')) < depth0_actions[a_name]:
                        depth0_actions[a_name] = cost_values.get(child, float('inf'))
            action_costs_str = " ".join([f"{a}={c:.2f}" for a, c in sorted(depth0_actions.items(), key=lambda x: x[1])])
            max_depth = max((s for _, (_, _, _, _, s) in zip(range(len(cost_so_far)),
                           [(0,0,0,n,expanded_steps.get(n,0)) for n in cost_so_far])
                           if True), default=0) if cost_so_far else 0
            mi_str = ""
            if mi_bonus is not None:
                mi_vals = " ".join(f"{action_names[a] if a < 6 else a}={mi_bonus.get(a,0):.3f}" for a in sorted(mi_bonus))
                mi_str = f" mi_bonus=[{mi_vals}]"
            log.info(
                f"[A*_DECISION] chosen={chosen} expansions={num_expansions} "
                f"best_cost={cost_values.get(best, 0):.3f} plan_len={len(plan)} "
                f"action_costs=[{action_costs_str}]{mi_str}"
            )

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
