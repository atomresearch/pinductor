"""Helpers for the REx loop that are NOT specific to the agent class.

Four small, stateless utilities extracted so the agent file stays
readable and the algorithmic choices can be A/B-tested in isolation.

1. ``ucb1_select`` — standard UCB1 node selection for tree refinement
   (Kocsis & Szepesvári 2006). Replaces the ad-hoc Thompson-on-Beta
   sampler which structurally starves new nodes because the root
   accumulates wins before its children exist.

2. ``compute_reward_reachability`` — sanity check that the LLM's
   reward_func is *alive*: replay each trajectory's own actions on the
   learned model and count episodes where any non-zero reward or
   terminal step is predicted. A model whose ``reward_func`` returns
   ``(0, False)`` everywhere scores RR=0, which an ELBO-only signal
   cannot detect because obs-explanation can be fine even with a dead
   reward head.

3. ``normalize_to_nll`` — convert ``obs_acc`` / ``trans_acc`` /
   ``rew_acc`` into a common negative-log-likelihood unit so the
   ``weakest component`` identification stops being dominated by the
   component with the smallest native scale.

4. ``rotated_strat_seed`` — derive a per-iteration seed from a base
   seed + iter number so the stratified sample rotates across REx
   iterations instead of showing the same 3 episodes forever.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional


# ---------------------------------------------------------------------------
# D. UCB1 selection over REx tree nodes
# ---------------------------------------------------------------------------


def ucb1_select(
    nodes: List[Any],
    *,
    c: float = 1.0,
    active_key: Callable[[Any], bool] = lambda n: getattr(n, "to_update", True),
    value_key: Callable[[Any], float] = lambda n: float(getattr(n, "q_mean", 0.0)),
    visit_key: Callable[[Any], int] = lambda n: int(getattr(n, "n_visit", 1)),
) -> Optional[Any]:
    """Return the node with the largest UCB1 score among active nodes.

    UCB1 (Kocsis & Szepesvári, ECML 2006):

        score(n) = Q̂(n) + c * sqrt(ln(N_total) / n_visit(n))

    Implementation note — the original ``if v == 0: score = +inf``
    shortcut degenerated the search: every REx iteration appends a
    fresh node (α=β=1 → visit=0), so that node always won via the
    +inf bonus and Q̂ was *never* consulted. Logs confirmed this:
    every UCB1 pick had ``n_visit=0, Q̂=0.000``. We replace the
    +inf shortcut with a ``(v + 1)`` denominator: fresh nodes still
    get the largest exploration bonus (no prior visits) but the
    bonus is finite, so a sibling with a good Q̂ can overtake it.

    Behaviour notes:
    - Ties are broken by the first node in iteration order (caller is
      responsible for any stability guarantee).
    - ``active_key`` lets the caller mask frozen nodes
      (``to_update=False``) without rebuilding the list.
    """
    active = [n for n in nodes if active_key(n)]
    if not active:
        return None

    total_visits = sum(max(visit_key(n), 0) for n in active)
    # Use ``+1`` inside the log so ``ln(N_total)`` stays finite and
    # monotonically increasing from the very first call (when no node
    # has been visited yet, ``total_visits = 0``).
    log_total = math.log(total_visits + 1)

    best_node = None
    best_score = -math.inf
    for n in active:
        v = max(visit_key(n), 0)
        # Add one to ``v`` so fresh nodes get the largest — but finite —
        # exploration bonus. Prevents the +inf sink that drowned Q̂.
        score = value_key(n) + c * math.sqrt(log_total / (v + 1))
        if score > best_score:
            best_score = score
            best_node = n
    return best_node


# ---------------------------------------------------------------------------
# B. NLL normalization for MODEL QUALITY feedback
# ---------------------------------------------------------------------------


def normalize_to_nll(
    obs_acc: float,
    trans_acc: float,
    rew_acc: float,
    *,
    eps: float = 0.01,
) -> Tuple[float, float, float]:
    """Map three [0,1] accuracies to negative log-likelihoods (nats).

    The three components are on heterogeneous native scales:
        obs_acc   = exp(-mean_dist)        bound away from 0 by kernel
        trans_acc = mean ESS / K           degenerates near 0 when
                                           particles all die
        rew_acc   = fraction of events     effectively binary on small
                                           event counts

    Taking ``min`` of these to name a ``weakest component`` is biased
    by the native scale, not by the truth. Converting to NLL puts them
    in a common additive unit so ``argmax NLL`` picks the truly worst
    component.

    Each accuracy ``p`` is mapped to ``-log(max(p, eps))``. The ``eps``
    floor keeps the NLL bounded even when a component scores 0 (which
    would otherwise give +inf and always win the argmax trivially).

    Returns ``(nll_obs, nll_trans, nll_rew)`` in nats. Larger = worse.
    """
    floor = max(float(eps), 1e-12)

    def _nll(p: float) -> float:
        if math.isnan(p):
            return float("nan")
        return -math.log(max(float(p), floor))

    return _nll(obs_acc), _nll(trans_acc), _nll(rew_acc)


# ---------------------------------------------------------------------------
# A. Mini-batch rotation seed
# ---------------------------------------------------------------------------


def rotated_strat_seed(base_seed: int, iter_num: int) -> int:
    """Compose a base seed with the current REx iter number.

    Rotating the stratified-sample seed per iteration ensures the
    prompt shows a different sampled triplet each time, so over the
    course of 20+ REx iterations the LLM has effectively seen the
    whole train set without blowing up the prompt budget.

    Kept as a tiny function for three reasons: (i) the composition
    convention is easy to get wrong (simple ``+`` collides across
    envs at iter 0), (ii) unit-testable, (iii) callers can mock it.
    """
    # (base, iter) pair mixed through a large prime to decorrelate
    # consecutive iterations and across base seeds. Deterministic.
    return int((int(base_seed) * 2654435761 + int(iter_num) * 1103515245) & 0x7FFFFFFF)
