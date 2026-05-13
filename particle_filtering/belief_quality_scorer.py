"""Belief-quality scorer — ORACLE diagnostic, NOT FOR DEPLOYMENT.

This scorer reads ground-truth states from ``Episode.next_states``.
It violates the paper's obs-only constraint and must **never** be wired
into the production scoring loop. It exists for two strictly-offline
purposes:

1. **Canary/sanity validation** : confirm that a non-trivial signal
   exists in principle (GT > iter1 > iter0 ordering when the scorer
   has oracle access).
2. **Ceiling calibration** : report the upper bound achievable on
   correlation with deployment reward, to be compared against
   obs-only proxies (``pf_self_consistency``, ``event_scoring``).

Measures how well the particle-filter belief induced by the candidate
models concentrates on the ground-truth agent position at each step of
the offline replay buffer. A candidate whose ``transition_func`` and
``reward_func`` are correct produces a PF that tracks the true agent
through the trajectory; a bugged candidate (e.g. lava not terminal,
wall traversal allowed) produces a PF that diverges and loses mass on
the true position.

Empirical motivation
--------------------
During deployment the PF belief can concentrate on the true agent
position while aggregate ELBO scoring still dilutes the belief-vs-real
signal across observationally redundant steps. This scorer reads the
belief-vs-real match at every step directly.

Consumes ``per_step_decomp`` dicts produced at
``get_score_metrics.py:431`` and annotated with
``belief_vs_real_pos_weight`` (mass on particles whose ``agent_pos``
matches the true state) and ``belief_vs_real_weight`` (mass on
particles that exactly equal the true state).

Env-agnostic: requires only that the replay buffer's state type
exposes an ``agent_pos`` attribute. When absent (non-minigrid envs),
``_compute_episode_score`` sets the two fields to ``None`` and this
scorer reports ``n_valid_steps=0`` with ``pf_score=0.0``.

Score contract:
- ``pf_score ∈ [0, 1]`` where 1 = PF collé sur la vérité, 0 = PF
  uniformly spread or concentrated on a wrong position.
- Uses position-level match by default (``weight_on_pos_mean``) since
  the PF cannot resolve agent_dir / carrying from image observations
  alone on Minigrid. The exact-state match is reported as a secondary
  diagnostic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


log = logging.getLogger(__name__)


PF_SCORE_FLOOR = 0.0
PF_SCORE_CEIL = 1.0


@dataclass
class BeliefQualityReport:
    """Summary of belief-quality over one trajectory.

    Attributes
    ----------
    pf_score:
        Scalar score injected into the scoring pipeline. Defaults to the
        mean position-level match (``weight_on_pos_mean``). Clipped to
        ``[0, 1]``.
    weight_on_pos_mean:
        Mean over valid steps of the belief mass on particles whose
        ``agent_pos`` matches the true state.
    weight_on_state_mean:
        Mean over valid steps of the belief mass on particles that
        exactly equal the true state (stricter, harder to satisfy).
    n_valid_steps:
        Number of steps with a non-``None`` belief-vs-real annotation.
    n_total_steps:
        Total number of steps in the trajectory decomposition.
    """

    pf_score: float
    weight_on_pos_mean: float
    weight_on_state_mean: float
    n_valid_steps: int
    n_total_steps: int


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float or return None if not finite."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def score_trajectory(
    per_step_decomp: List[Dict[str, Any]],
) -> BeliefQualityReport:
    """Score one trajectory by mean belief mass on the true state.

    Parameters
    ----------
    per_step_decomp:
        List of per-step dicts produced by ``ScoreEvaluator._step_score``
        and post-annotated in ``_compute_episode_score``. Each dict
        should contain ``belief_vs_real_pos_weight`` (float in [0, 1]
        or ``None``) and ``belief_vs_real_weight`` (idem, stricter).
        Steps with ``None`` for either field are skipped.

    Returns
    -------
    :class:`BeliefQualityReport`
        ``pf_score`` defaults to ``weight_on_pos_mean`` clipped to
        ``[0, 1]``. If no valid steps, returns ``pf_score=0.0`` and
        ``n_valid_steps=0`` with a warning.
    """
    n_total = len(per_step_decomp) if per_step_decomp else 0

    if not per_step_decomp:
        log.warning(
            "[belief_quality] empty per_step_decomp — returning "
            "pf_score=0.0 (no trajectory data)."
        )
        return BeliefQualityReport(
            pf_score=0.0,
            weight_on_pos_mean=0.0,
            weight_on_state_mean=0.0,
            n_valid_steps=0,
            n_total_steps=0,
        )

    pos_weights: List[float] = []
    state_weights: List[float] = []
    for d in per_step_decomp:
        w_pos = _as_float(d.get("belief_vs_real_pos_weight"))
        w_state = _as_float(d.get("belief_vs_real_weight"))
        if w_pos is None and w_state is None:
            continue
        if w_pos is not None:
            pos_weights.append(max(0.0, min(1.0, w_pos)))
        if w_state is not None:
            state_weights.append(max(0.0, min(1.0, w_state)))

    n_valid = max(len(pos_weights), len(state_weights))
    if n_valid == 0:
        log.warning(
            "[belief_quality] no step has belief_vs_real annotations "
            "(%d total steps) — returning pf_score=0.0. Check that "
            "episode.next_states exposes agent_pos.",
            n_total,
        )
        return BeliefQualityReport(
            pf_score=0.0,
            weight_on_pos_mean=0.0,
            weight_on_state_mean=0.0,
            n_valid_steps=0,
            n_total_steps=n_total,
        )

    pos_mean = (
        float(sum(pos_weights)) / float(len(pos_weights))
        if pos_weights
        else 0.0
    )
    state_mean = (
        float(sum(state_weights)) / float(len(state_weights))
        if state_weights
        else 0.0
    )
    pf_score = max(PF_SCORE_FLOOR, min(PF_SCORE_CEIL, pos_mean))

    return BeliefQualityReport(
        pf_score=pf_score,
        weight_on_pos_mean=pos_mean,
        weight_on_state_mean=state_mean,
        n_valid_steps=n_valid,
        n_total_steps=n_total,
    )
