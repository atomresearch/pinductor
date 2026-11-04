"""QMDP cost-to-go heuristic derived exclusively from the planner's own models.

h(s) = scale * d*(s), where d*(s) is the optimal action count from s to a
terminal (goal) state in the fully-observed relaxation of the POMDP defined by
the models CURRENTLY plugged into the planner (transition_model +
reward_model). Since the planner in this pipeline only ever holds induced /
learned models, the heuristic never touches ground-truth environment code:
it is exactly as correct as the learned model it is derived from.

QMDP relaxation (Littman, Cassandra & Kaelbling 1995): d*(s) lower-bounds the
number of actions any POMDP policy needs from state s, so
E_{s~b}[h(s)] with scale == action_cost is an admissible estimate of the
remaining path cost in the belief-space search (with a sparse terminal-only
reward, the -E[r] term of the path cost is a constant on every solution path
and does not affect ranking).

Construction: forward BFS over the reachable state graph from the query state
(successors sampled from transition_model; terminal edges detected via the
reward model's terminal flag and treated as absorbing), then multi-source
reverse BFS from the goal states. Distances are cached per model pair and
invalidated whenever the planner's models are swapped (online re-learning).

Assumes a deterministic transition model when num_transition_samples == 1
(the default). For stochastic models, num_transition_samples > 1 approximates
the successor SUPPORT, and d* becomes the optimistic (best-case) count — still
a lower bound, hence still admissible.

Fallbacks (both deterministic, no tunable penalty):
  * component larger than node_cap  -> h == 0 everywhere in it (uninformed A*,
    trivially admissible);
  * state that cannot reach a goal under the model -> h == scale * |component|,
    an upper bound on any shortest path inside the enumerated component, so
    such particles are dominated by any goal-reaching particle.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class ModelQMDPHeuristic:
    def __init__(
        self,
        scale: float = 0.01,
        node_cap: int = 800_000,
        num_transition_samples: int = 1,
    ) -> None:
        if scale < 0.0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        if node_cap < 1:
            raise ValueError(f"node_cap must be >= 1, got {node_cap}")
        if num_transition_samples < 1:
            raise ValueError(
                f"num_transition_samples must be >= 1, got {num_transition_samples}"
            )
        self.scale = scale
        self.node_cap = node_cap
        self.num_transition_samples = num_transition_samples
        self._planner: Optional[Any] = None
        # Cache state: valid only for the model pair identified by _model_ids.
        self._model_ids: Optional[Tuple[int, int]] = None
        self._dist: Dict[Any, int] = {}
        self._explored: Set[Any] = set()  # states already covered by a build
        self._component_size = 0
        self._capped = False

    # ------------------------------------------------------------------ #
    # Wiring: the planner binds itself so the heuristic always reads the  #
    # models CURRENTLY deployed (they are replaced during online learning) #
    # ------------------------------------------------------------------ #
    def bind_planner(self, planner: Any) -> None:
        self._planner = planner

    def _refresh_cache(self) -> None:
        assert self._planner is not None
        ids = (id(self._planner.transition_model), id(self._planner.reward_model))
        if ids != self._model_ids:
            self._model_ids = ids
            self._dist.clear()
            self._explored.clear()
            self._component_size = 0
            self._capped = False

    # ------------------------------------------------------------------ #
    # Reachable-graph construction from the induced models                #
    # ------------------------------------------------------------------ #
    def _successors(self, state: Any) -> List[Tuple[Any, bool]]:
        planner = self._planner
        assert planner is not None
        out: List[Tuple[Any, bool]] = []
        seen: Set[Any] = set()
        for action in planner.actions:
            for _ in range(self.num_transition_samples):
                try:
                    nxt = planner.transition_model(state, action)
                except Exception:
                    continue
                if nxt in seen:
                    continue
                seen.add(nxt)
                try:
                    _, terminal = planner.reward_model(state, action, nxt)
                except Exception:
                    terminal = False
                out.append((nxt, bool(terminal)))
        return out

    def _build_from(self, start: Any) -> None:
        """Enumerate the component reachable from `start`, then reverse-BFS
        distances from every goal state found. Merges into the global cache."""
        adj: Dict[Any, List[Any]] = {start: []}
        goals: Set[Any] = set()
        queue: Deque[Any] = deque([start])
        expanded: Set[Any] = set()
        while queue:
            if len(adj) + self._component_size > self.node_cap:
                self._capped = True
                log.warning(
                    "ModelQMDPHeuristic: component exceeded node_cap=%d; "
                    "falling back to h=0 (uninformed search).",
                    self.node_cap,
                )
                self._explored.update(adj)
                return
            state = queue.popleft()
            if state in expanded:
                continue
            expanded.add(state)
            for nxt, terminal in self._successors(state):
                adj[state].append(nxt)
                if terminal:
                    # Absorbing goal: distance 0, never expanded further —
                    # matches the planner's handling of terminal nodes.
                    goals.add(nxt)
                    adj.setdefault(nxt, [])
                elif nxt not in adj:
                    adj[nxt] = []
                    queue.append(nxt)

        rev: Dict[Any, List[Any]] = {k: [] for k in adj}
        for state, succs in adj.items():
            for nxt in succs:
                rev[nxt].append(state)

        dist: Dict[Any, int] = {g: 0 for g in goals}
        dq: Deque[Any] = deque(goals)
        while dq:
            state = dq.popleft()
            for prev in rev[state]:
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    dq.append(prev)

        self._component_size += len(adj)
        self._explored.update(adj)
        self._dist.update(dist)

    # ------------------------------------------------------------------ #
    # Heuristic interface: Callable[[State], float]                       #
    # ------------------------------------------------------------------ #
    def __call__(self, state: Any) -> float:
        if self._planner is None:
            raise RuntimeError(
                "ModelQMDPHeuristic is not bound to a planner. It must be "
                "passed as the `heuristic` of a Planner (base_planner binds it)."
            )
        self._refresh_cache()
        if self._capped:
            return 0.0
        if state not in self._explored:
            self._build_from(state)
            if self._capped:
                return 0.0
        d = self._dist.get(state)
        if d is None:
            # Cannot reach any goal under the current model: dominated by any
            # goal-reaching particle via the component-size upper bound.
            return self.scale * self._component_size
        return self.scale * d
