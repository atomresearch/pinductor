"""Pure helpers for the LLM REx feedback pipeline.

Extracted from ``partially_obs_planning_agent.py`` so the agent module
can stay focused on orchestration. These helpers are *stateless* — none
of them depend on agent state — and are unit-testable in isolation.

Module contents:

- ``LIKELIHOOD_FLOOR``: finite floor used to clamp ``-inf`` per-episode scores
  so that np.mean of a partially-collapsed episode set stays meaningful.
- ``likelihood_from_episode_metrics``: aggregate per-episode pf_scores into a
  single number for the REx selection step.
- ``jensen_shannon_divergence``: symmetric distribution distance used by
  the legacy single-model REx loop and a couple of analysis scripts.

Private convenience aliases (same module): ``_LIKELIHOOD_FLOOR``,
``_likelihood_from_episode_metrics``, ``jsd``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


# A pf_score of -inf indicates total particle collapse on a single
# episode. Aggregating with np.mean would propagate -inf and lose all
# discriminative power between models that fail on 1/N episodes vs N/N.
# We clamp to a finite floor that is "very bad but distinguishable".
LIKELIHOOD_FLOOR: float = -100.0


def likelihood_from_episode_metrics(
    episode_metrics: Optional[List[Dict[str, Any]]],
    msc_weight: float = 0.0,
    aggregation: str = "mean",
) -> float:
    """Aggregate per-episode pf_scores into a single comparable number.

    score = reducer(clamp(pf_score)) + msc_weight * reducer(clamp(msc_score))

    where ``reducer`` is ``np.mean`` (default) or ``np.median`` when
    ``aggregation="median"`` (Fix F2 / R#2-A, Huber 1964 robust stats).

    Both pf_score and msc_score are clamped to :data:`LIKELIHOOD_FLOOR` so that
    a single particle collapse on one episode does not destroy the score
    of the entire run. Returns :data:`LIKELIHOOD_FLOOR` when no usable
    episodes are available.
    """
    return likelihood_components_from_episode_metrics(
        episode_metrics, msc_weight=msc_weight, aggregation=aggregation
    )[0]


def likelihood_components_from_episode_metrics(
    episode_metrics: Optional[List[Dict[str, Any]]],
    msc_weight: float = 0.0,
    aggregation: str = "mean",
) -> tuple[float, Dict[str, float]]:
    """Aggregate + expose raw/clamped components side by side (M6).

    Args:
        episode_metrics: per-episode metrics dicts (with ``pf_score`` key).
        msc_weight: weight of the optional MSC side-channel (default 0.0,
            disabled).
        aggregation: reducer across episodes. ``"mean"`` (default, backward
            compatible) or ``"median"`` (robust to a single collapsed
            episode — Fix F2 a.k.a. R#2-A). Robust statistics (Huber 1964)
            justify the median choice when the dataset contains outliers
            caused by ``pf_score=-inf`` collapses that the M6 clamp brings
            back to ``LIKELIHOOD_FLOOR`` but still pulls the mean down.

    Returns ``(clamped_score, diagnostics)`` where ``diagnostics`` has
    keys: ``raw_mean``, ``clamped_mean``, ``n_collapsed``, ``n_valid``,
    ``min_raw``, ``max_raw``, ``aggregation`` and (if ``msc_weight > 0``)
    ``msc_mean``.

    Callers that only need the scalar use :func:`likelihood_from_episode_metrics`.
    The diagnostics dict is intended for logging so the user can see
    whether the clamp is masking catastrophic models — the audit's M6
    concern: a hard floor at -100 hides the gap between "bad" and
    "extremely bad" and makes convergence look smoother than it is.
    """
    if aggregation not in ("mean", "median"):
        raise ValueError(
            f"aggregation must be 'mean' or 'median', got {aggregation!r}"
        )
    reducer = np.median if aggregation == "median" else np.mean
    diag: Dict[str, float] = {
        "raw_mean": float(LIKELIHOOD_FLOOR),
        "clamped_mean": float(LIKELIHOOD_FLOOR),
        "n_collapsed": 0.0,
        "n_valid": 0.0,
        "min_raw": float(LIKELIHOOD_FLOOR),
        "max_raw": float(LIKELIHOOD_FLOOR),
        "aggregation": aggregation,
    }
    if not episode_metrics:
        return LIKELIHOOD_FLOOR, diag
    valid = [m for m in episode_metrics if m and "pf_score" in m]
    if not valid:
        return LIKELIHOOD_FLOOR, diag
    raw = [float(m["pf_score"]) for m in valid]
    clamped = [max(v, LIKELIHOOD_FLOOR) for v in raw]
    base = float(reducer(clamped))
    finite_raw = [v for v in raw if math.isfinite(v)]
    diag["clamped_mean"] = base
    diag["raw_mean"] = (
        float(reducer(finite_raw)) if finite_raw else float(LIKELIHOOD_FLOOR)
    )
    diag["n_valid"] = float(len(valid))
    diag["n_collapsed"] = float(
        sum(
            1
            for v in raw
            if not math.isfinite(v) or v <= LIKELIHOOD_FLOOR
        )
    )
    diag["min_raw"] = float(min(raw))
    diag["max_raw"] = float(max(raw))
    if msc_weight > 0.0:
        msc_vals = [m.get("msc_score", LIKELIHOOD_FLOOR) for m in valid]
        finite_msc = [max(v, LIKELIHOOD_FLOOR) for v in msc_vals]
        if finite_msc:
            msc_agg = float(reducer(finite_msc))
            base += msc_weight * msc_agg
            diag["msc_mean"] = msc_agg
    return base, diag


def term_cross_entropy(
    pred_term_prob: float,
    actual_terminated: bool,
    eps: float = 1e-6,
) -> float:
    """Cross-entropy of the observed terminated flag under the belief.

    ``pred_term_prob`` is the belief-averaged probability that the LLM
    candidate predicts ``terminated=True`` for this step (from
    ``get_score_metrics._step_score``). The CE is the negative log of
    the probability assigned to the *observed* event, clamped by ``eps``
    to bound the penalty on an exactly-wrong prediction (otherwise one
    catastrophically wrong step would dominate every other channel).

    This helper exists so F5's (R#5-A) Pareto terminated channel can
    be unit-tested without mocking the whole particle pipeline.
    Mathematical content: ``-log p`` where ``p = P_belief(t_obs)``.

    Args:
        pred_term_prob: float in [0, 1] — belief mass on ``terminated=True``.
        actual_terminated: observed terminated flag from the replay buffer.
        eps: floor used for the log argument; default ``1e-6`` bounds the
            worst-case penalty at ``-log(1e-6) ≈ 13.82 nats``.

    Returns the nonnegative CE in nats.
    """
    p = float(pred_term_prob)
    p_correct = p if bool(actual_terminated) else (1.0 - p)
    return -math.log(max(p_correct, eps))


def jensen_shannon_divergence(
    p: Dict[Any, float],
    q: Dict[Any, float],
    base: float = 2.0,
) -> float:
    """Symmetric Jensen–Shannon divergence between two discrete distributions.

    JSD(P, Q) = 0.5 * (KL(P||M) + KL(Q||M))  where  M = 0.5*(P+Q)

    Bounded in [0, log_base(2)] (so [0, 1] when base=2). Used by the
    legacy single-model REx coverage scoring.
    """
    all_keys = set(p.keys()) | set(q.keys())
    midpoint: Dict[Any, float] = {}
    for k in all_keys:
        midpoint[k] = (p.get(k, 0.0) + q.get(k, 0.0)) / 2.0

    kl_pm = 0.0
    kl_qm = 0.0
    for k in all_keys:
        p_val = p.get(k, 0.0)
        q_val = q.get(k, 0.0)
        m_val = midpoint[k]
        if p_val > 0 and m_val > 0:
            kl_pm += p_val * math.log(p_val / m_val, base)
        if q_val > 0 and m_val > 0:
            kl_qm += q_val * math.log(q_val / m_val, base)

    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Private aliases used by partially_obs_planning_agent and analysis helpers.
# ---------------------------------------------------------------------------
_LIKELIHOOD_FLOOR = LIKELIHOOD_FLOOR
_likelihood_from_episode_metrics = likelihood_from_episode_metrics
jsd = jensen_shannon_divergence
