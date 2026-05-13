"""Particle-filter score evaluator with soft observation distance.

Used by the LLM REx loop to score candidate ``(transition, observation,
reward)`` models against trajectories collected from the real env. The
flow mirrors :class:`PartiallyObsPlanningAgent.update_belief`:

1. Propagate particles with the transition model.
2. Weight particles using ``exp(-distance_soft / kernel_bandwidth)``.
3. Rejuvenate from the initial model when particle mass collapses.

Per-step score (energy form):

    score_t = -E_q[distance_soft(predicted_obs, actual_obs)]

where *q* is the normalised particle distribution at step *t*. The
"prequential" mode computes the FIVO log-marginal estimate
``log((1/K) Σ_k w_k)`` instead (Dawid 1984; Maddison et al. 2017).

Optional diagnostics:

- ``decompose=True`` enables FIVO-style per-component scoring (obs / rew /
  trans) and event-level reward accuracy (truly informative since most
  steps trivially have reward 0).
- ``msc_horizon>0`` enables Multi-Step Consistency: refilters from the
  particle belief and predicts the observation *d* steps ahead, scoring
  the cumulative drift of the transition model.

References:
- Dawid, A. P. (1984), "Statistical theory: the prequential approach"
- Maddison, Lawson, et al. (2017), "Filtering Variational Objectives" (FIVO)
- Chopin & Papaspiliopoulos (2020), "An Introduction to SMC", §10.3
- Bishop (2006), PRML §11.3 for ESS and degeneracy
"""

from __future__ import annotations

import copy
import logging
import math
import os
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uncertain_worms.structs import (
    ActType,
    InitialModel,
    RewardModel,
    TransitionModel,
    ObservationModel,
    ObsType,
    ParticleBelief,
    ReplayBuffer,
    StateType,
)

log = logging.getLogger(__name__)


# Tolerance used to decide when a predicted reward value counts as
# matching the observed one. Kept at 1e-3 (consistent with the prior
# local definition) so all existing accuracy counts stay comparable.
REWARD_TOL = 1e-3


def _soft_weight_from_distance(
    dist: float,
    kernel_bandwidth: float,
    obs_noise_epsilon: float = 0.0,
) -> float:
    """Convert distance to weight with optional likelihood tempering.

    w = (1 - ε) · exp(-dist / κ) + ε

    When ε > 0 every particle retains a minimum weight, preventing total
    collapse when the model has small prediction errors.  This is the
    standard mixture-kernel approach from the SMC literature.

    Ref: Chopin & Papaspiliopoulos (2020), §10.3 "Tempering"
    """
    if math.isinf(dist) or dist < 0:
        return obs_noise_epsilon
    base_w = math.exp(-dist / max(kernel_bandwidth, 1e-12))
    if obs_noise_epsilon > 0.0:
        return (1.0 - obs_noise_epsilon) * base_w + obs_noise_epsilon
    return base_w


def _effective_sample_size(weights: List[float]) -> float:
    """ESS = 1 / sum(w_norm^2) for normalized weights."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    w_norm = np.array(weights) / total
    return float(1.0 / (np.sum(w_norm**2) + 1e-12))




class ScoreEvaluator:
    """Evaluate LLM-generated models via particle-filter scoring.

    Mirrors the PF in PartiallyObsPlanningAgent.update_belief, but
    designed for *offline evaluation* of model quality on replay data.

    Two scoring modes:
      - "energy" (default): score_t = -E_q[distance_soft(pred_obs, actual_obs)]
      - "prequential": score_t = log( (1/K) Σ_k w_k )  (Dawid 1984 / FIVO)

    Optional diagnostics:
      - decompose=True → per-step obs/rew/trans breakdown (FIVO-style)
      - msc_horizon>0  → Multi-Step Consistency penalty for transition drift

    Args:
        transition_model: T(s, a) → s'
        observation_model: O(s, a, empty_obs) → obs
        initial_model: I(empty_state) → s_0
        reward_model: R(s, a, s') → (reward, terminated)
        num_particles: target particle count for belief (also ESS reference)
        kernel_bandwidth (κ): controls weight sharpness: w = exp(-d / κ).

            The 1.0 default here is a *loose* prior intended for
            analysis-only calls that score held-out episodes; the
            runtime agent deliberately overrides it with a tighter
            ``kernel_bandwidth = 0.2`` (`ours.yaml` configs) after
            ablations showed that 0.2 preserves particle diversity
            without letting catastrophic-obs particles dominate.
            Any caller is free to override via the constructor.
        max_rejuvenation_attempts: hard stop on the
            re-sample-from-initial loop when particles cannot be
            matched to the trajectory. 100k is generous enough that
            well-specified models never hit it; mis-specified models
            hit it quickly and fall back to the empty-belief path.
            Exposed so that sensitivity studies can tighten it.
        scoring_mode: one of "energy" (default), "prequential" (FIVO
            log-marginal), "whiteness" (Ljung-Box + Hosking + Hansen J
            + McNemar on residuals), "betting" (ONS-betting e-value
            multi-channel, Waudby-Smith & Ramdas 2024), or "bisim"
            (Ferns-2011 bisimulation distance with absorbing-state
            penalty on terminated). The three non-likelihood scorers
            require ``decompose=True``; on empty ``per_step_decomp``
            they log a warning and fall back to the energy score.
        obs_noise_epsilon (ε): likelihood floor for particle survival (E4)
    """

    _VALID_SCORING_MODES = (
        "energy",
        "prequential",
        "whiteness",
        "betting",
        "event",
        "bisim",
        "belief_quality",
        "pf_self_consistency",
        "prediction_persistence",
        "hybrid_pf_event",
    )

    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        initial_model: InitialModel,
        reward_model: RewardModel,
        empty_state: StateType,
        empty_observation: ObsType,
        num_particles: int = 1000,
        kernel_bandwidth: float = 1.0,
        max_rejuvenation_attempts: int = 100000,
        decompose: bool = False,
        msc_horizon: int = 0,
        scoring_mode: str = "energy",
        obs_noise_epsilon: float = 0.0,
        event_weighted_scoring: bool = False,
        reward_weight: float = 0.0,
    ):
        # --- Input validation (fail fast on misconfiguration) ---
        if num_particles < 2:
            raise ValueError(
                f"num_particles must be ≥ 2, got {num_particles}. "
                "ESS becomes meaningless with a single particle."
            )
        if kernel_bandwidth <= 0:
            raise ValueError(
                f"kernel_bandwidth must be > 0, got {kernel_bandwidth}."
            )
        if max_rejuvenation_attempts < 1:
            raise ValueError(
                f"max_rejuvenation_attempts must be ≥ 1, got {max_rejuvenation_attempts}."
            )
        if msc_horizon < 0:
            raise ValueError(
                f"msc_horizon must be ≥ 0 (0 disables MSC), got {msc_horizon}."
            )
        if not 0.0 <= obs_noise_epsilon <= 1.0:
            raise ValueError(
                f"obs_noise_epsilon must be in [0, 1], got {obs_noise_epsilon}."
            )
        if scoring_mode not in self._VALID_SCORING_MODES:
            raise ValueError(
                f"scoring_mode must be one of {self._VALID_SCORING_MODES}, "
                f"got {scoring_mode!r}."
            )

        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.empty_state = empty_state
        self.empty_observation = empty_observation
        self.num_particles = num_particles
        self.decompose = decompose
        self.msc_horizon = msc_horizon
        self.scoring_mode = scoring_mode
        self.kernel_bandwidth = kernel_bandwidth
        self.max_rejuvenation_attempts = max_rejuvenation_attempts
        self.obs_noise_epsilon = obs_noise_epsilon
        # I3 — inverse-frequency event weighting (Hong et al. NeurIPS 2023).
        # When True, each step's contribution to ``total_score`` is scaled
        # by the inverse of its empirical class frequency in the episode,
        # so rare "event" steps (reward != 0 or terminated) are upweighted
        # relative to the ~90% trivially-zero middle steps. Sum of weights
        # is normalised to equal episode length T so the total_score scale
        # stays comparable to the baseline (important for the Beta update
        # delta and ELBO clamping). Default False = byte-identical to the
        # pre-I3 formula.
        self.event_weighted_scoring = event_weighted_scoring
        # Preliminary #8 — explicit reward MSE term in ``pf_score``.
        # When ``reward_weight > 0`` and ``decompose=True``, the per-step
        # ``score_rew`` (-E[(r_pred - r_actual)^2]) is summed and added
        # to ``total_score`` as ``+ reward_weight * Σ_t score_rew_t``.
        # ``score_rew`` is already per-step ≤ 0 (MSE is negative of
        # error), so positive ``reward_weight`` increases the penalty
        # for reward-prediction errors, making the scorer reward-aware
        # on top of the observation-only ELBO. Requires ``decompose=True``
        # to have the per-step decomposition — silently ignored otherwise.
        # Default 0.0 = byte-identical to pre-feature formula.
        self.reward_weight = float(reward_weight)

    def evaluate_score(
        self,
        replay_buffer: ReplayBuffer,
        output_particle_distance_metrics: bool = False,
    ) -> Tuple[
        List[List[ActType]],
        List[List[ObsType]],
        List[List[ObsType]],
        List[List[float]],
        List[List[bool]],
        Optional[str],
        List[Dict[str, Any]],
    ]:
        """
        Evaluate particle-filter score traces for each episode.

        Args:
            replay_buffer: Replay buffer with episodes to evaluate.
            output_particle_distance_metrics: If True, compute and include per-episode metrics in
                episode_metrics. Default False for backward compatibility.

        Returns:
            episode_actions: List of action lists per episode
            episode_observations: List of next_observation lists per episode
            episode_previous_observations: List of previous_observation lists per episode (initial obs first)
            episode_rewards: List of reward lists per episode (reward at each step)
            episode_terminated: List of terminated lists per episode (True if episode ended at that step)
            err_str: Error message if any
            episode_metrics: List of dicts, one per episode. When output_particle_distance_metrics=True,
                each dict contains: best_particle_distance, median_particle_distance, variance_distance,
                mean_ESS, min_ESS (particle degeneracy). Otherwise each dict is empty.
        """
        log.info("[ScoreEvaluator] Evaluating score using particle filtering with distance_soft")

        episode_actions: List[List[ActType]] = []
        episode_observations: List[List[ObsType]] = []
        episode_previous_observations: List[List[ObsType]] = []
        episode_rewards: List[List[float]] = []
        episode_terminated: List[List[bool]] = []
        episode_metrics: List[Dict[str, Any]] = []
        err_str: Optional[str] = None

        # Score episodes in parallel when there are enough to justify overhead.
        # Uses threads (not processes) because model closures aren't picklable.
        n_episodes = len(replay_buffer.episodes)
        use_parallel = n_episodes >= 3

        # === EARLY-REJECT PRE-SCREEN ===
        # When the caller has set `self._reject_threshold` (typically
        # `best_ever_score - margin` from REx), we evaluate demo 0 first
        # in sequential mode. If its pf_score is already below the threshold
        # the candidate cannot recover on the remaining demos (running mean
        # would need to be > threshold, but we already start far below) — so
        # we short-circuit with pf_score=-inf for every episode. Downstream
        # MONOTONE/Pareto/etc. all handle -inf as "reject".
        # Net effect: a clearly-bad candidate is rejected after one demo
        # (~30 s) instead of evaluating all demos in parallel (~5 min).
        # When `_reject_threshold` is None the path is byte-identical to
        # the previous behaviour.
        reject_threshold = getattr(self, "_reject_threshold", None)
        early_first_score = None
        early_first_actions = None
        early_first_observations = None
        early_first_metrics = None
        if reject_threshold is not None and n_episodes >= 1:
            try:
                pre_actions, pre_obs, pre_metrics = self._compute_episode_score(
                    replay_buffer.episodes[0],
                    collect_distances=output_particle_distance_metrics,
                )
                pf0 = (
                    float(pre_metrics.get("pf_score", 0.0))
                    if pre_metrics is not None
                    else 0.0
                )
                if pf0 < float(reject_threshold):
                    log.info(
                        "[EARLY_REJECT] pf_score demo 0 = %.4f < threshold %.4f "
                        "→ skipping %d remaining demos (saving ~5 min)",
                        pf0, float(reject_threshold), n_episodes - 1,
                    )
                    # Build a result tuple that signals rejection :
                    # pf_score=-inf for every episode → downstream clamping
                    # catches it as a collapsed candidate.
                    early_episode_metrics: List[Dict[str, Any]] = []
                    for i in range(n_episodes):
                        if i == 0 and pre_metrics is not None:
                            # Keep real demo-0 metrics so the diagnostic log
                            # still reports the score that triggered the cut.
                            m = dict(pre_metrics)
                            m["pf_score"] = float("-inf")
                            early_episode_metrics.append(m)
                        else:
                            early_episode_metrics.append(
                                {
                                    "pf_score": float("-inf"),
                                    "best_particle_distance": float("inf"),
                                    "median_particle_distance": float("inf"),
                                    "variance_distance": 0.0,
                                    "mean_ESS": 0.0,
                                    "min_ESS": 0.0,
                                }
                            )
                    early_actions: List[List[ActType]] = []
                    early_observations: List[List[ObsType]] = []
                    early_prev_obs: List[List[ObsType]] = []
                    early_rewards: List[List[float]] = []
                    early_terminated: List[List[bool]] = []
                    for i, ep in enumerate(replay_buffer.episodes):
                        if i == 0:
                            early_actions.append(pre_actions)
                            early_observations.append(pre_obs)
                        else:
                            early_actions.append([])
                            early_observations.append([])
                        prev = getattr(ep, "previous_observations", None) or []
                        early_prev_obs.append(prev)
                        early_rewards.append(getattr(ep, "rewards", []) or [])
                        early_terminated.append(getattr(ep, "terminated", []) or [])
                    return (
                        early_actions,
                        early_observations,
                        early_prev_obs,
                        early_rewards,
                        early_terminated,
                        None,
                        early_episode_metrics,
                    )
                # Pre-screen passed → cache demo 0 result for re-use below.
                early_first_score = pf0
                early_first_actions = pre_actions
                early_first_observations = pre_obs
                early_first_metrics = pre_metrics
            except Exception as _exc:
                log.debug(
                    "[EARLY_REJECT] pre-screen failed (%s) → falling back to "
                    "full eval", _exc,
                )
        # === END EARLY-REJECT PRE-SCREEN ===

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _score_one(index_episode):
                idx, ep = index_episode
                return idx, self._compute_episode_score(
                    ep, collect_distances=output_particle_distance_metrics
                )

            scored = [None] * n_episodes
            with ThreadPoolExecutor(max_workers=min(n_episodes, 8)) as pool:
                futures = {
                    pool.submit(_score_one, (i, ep)): i
                    for i, ep in enumerate(replay_buffer.episodes)
                }
                for future in as_completed(futures):
                    try:
                        idx, result = future.result()
                        scored[idx] = result
                    except Exception as e:
                        err_str = str(e)
                        log.info(f"Parallel scoring error: {e}")

            for index, episode in enumerate(replay_buffer.episodes):
                if scored[index] is None:
                    continue
                actions, observations, metrics = scored[index]
                episode_actions.append(actions)
                episode_observations.append(observations)
                previous_observations = getattr(episode, "previous_observations", None)
                if previous_observations is None:
                    previous_observations = []
                episode_previous_observations.append(previous_observations)
                episode_rewards.append(episode.rewards)
                episode_terminated.append(episode.terminated)

                episode_metrics_dict: Dict[str, Any] = {}
                if output_particle_distance_metrics and metrics is not None:
                    episode_metrics_dict = metrics
                    # Decomposition log: pf_score = obs_only - d_reward - d_heur
                    _obs_only = metrics.get("pf_score_obs_only")
                    _dr = metrics.get("d_reward")
                    _dh = metrics.get("d_heur")
                    _extra = ""
                    if _obs_only is not None and (_dr is not None or _dh is not None):
                        _extra += f" [obs_only={float(_obs_only):.4f}"
                        if _dr is not None:
                            _ne = metrics.get("d_reward_n_eval", "?")
                            _extra += f" −d_reward={float(_dr):.4f}({_ne})"
                        if _dh is not None:
                            _ne = metrics.get("d_heur_n_eval", "?")
                            _extra += f" −d_heur={float(_dh):.4f}({_ne})"
                        _extra += "]"
                    log.info(
                        f"[ScoreEvaluator] Episode {index} metrics: "
                        f"pf_score={metrics.get('pf_score', 0):.4f}, "
                        f"best_dist={metrics.get('best_particle_distance', 0):.4f}, "
                        f"mean_ESS={metrics.get('mean_ESS', 0):.4f}, "
                        f"min_ESS={metrics.get('min_ESS', 0):.4f}"
                        + _extra
                    )
                episode_metrics.append(episode_metrics_dict)

            return episode_actions, episode_observations, episode_previous_observations, episode_rewards, episode_terminated, err_str, episode_metrics

        # Sequential fallback for small replay buffers
        for index, episode in enumerate(replay_buffer.episodes):
            try:
                actions, observations, metrics = self._compute_episode_score(
                    episode, collect_distances=output_particle_distance_metrics
                )
                episode_actions.append(actions)
                episode_observations.append(observations)
                previous_observations = getattr(episode, "previous_observations", None)
                if previous_observations is None:
                    previous_observations = []
                episode_previous_observations.append(previous_observations)
                episode_rewards.append(episode.rewards)
                episode_terminated.append(episode.terminated)

                episode_metrics_dict: Dict[str, Any] = {}
                if output_particle_distance_metrics and metrics is not None:
                    episode_metrics_dict = metrics
                    # Decomposition log: pf_score = obs_only - d_reward - d_heur
                    _obs_only = metrics.get("pf_score_obs_only")
                    _dr = metrics.get("d_reward")
                    _dh = metrics.get("d_heur")
                    _extra = ""
                    if _obs_only is not None and (_dr is not None or _dh is not None):
                        _extra += f" [obs_only={float(_obs_only):.4f}"
                        if _dr is not None:
                            _ne = metrics.get("d_reward_n_eval", "?")
                            _extra += f" −d_reward={float(_dr):.4f}({_ne})"
                        if _dh is not None:
                            _ne = metrics.get("d_heur_n_eval", "?")
                            _extra += f" −d_heur={float(_dh):.4f}({_ne})"
                        _extra += "]"
                    log.info(
                        f"[ScoreEvaluator] Episode {index} metrics: "
                        f"pf_score={metrics.get('pf_score', 0):.4f}, "
                        f"best_dist={metrics.get('best_particle_distance', 0):.4f}, "
                        f"mean_ESS={metrics.get('mean_ESS', 0):.4f}, "
                        f"min_ESS={metrics.get('min_ESS', 0):.4f}"
                        + _extra
                    )
                episode_metrics.append(episode_metrics_dict)
            except Exception as e:
                log.info(f"Error during score evaluation: {e}")
                err_str = str(traceback.format_exc())
                log.info(err_str)
                break

        return episode_actions, episode_observations, episode_previous_observations, episode_rewards, episode_terminated, err_str, episode_metrics


    def evaluate_elbo(
        self,
        replay_buffer: ReplayBuffer,
        use_log_space: bool = True,
        output_particle_distance_metrics: bool = False,
    ) -> Tuple[
        List[List[ActType]],
        List[List[ObsType]],
        List[List[ObsType]],
        List[List[float]],
        List[List[bool]],
        Optional[str],
        List[Dict[str, Any]],
    ]:
        """Backward-compatible alias for ``evaluate_score``.

        ``use_log_space`` is accepted for compatibility and intentionally ignored.
        """
        return self.evaluate_score(
            replay_buffer, output_particle_distance_metrics=output_particle_distance_metrics
        )

    def _batch_distance_or_fallback(
        self,
        predicted_obs: List[ObsType],
        actual_obs: ObsType,
    ) -> List[float]:
        """Vectorised distance ``predicted_obs[k].distance_soft(actual_obs)``
        across the K particles in one call.

        For ``MinigridObservation`` we batch the image-base distance via
        the numba-compiled ``distance_soft_batch_image`` (~7-12x faster
        than K sequential calls on 3x3 FOV) then add the cheap scalar
        agent_dir / carrying penalties in a Python loop. For other obs
        types or when numba is missing, fall back to the per-particle
        ``distance_soft`` path — bit-equal, just slower.
        """
        if not predicted_obs:
            return []
        # Detect MinigridObservation fast-path (avoid name import here
        # to keep the module env-agnostic; rely on duck-typing).
        first = predicted_obs[0]
        has_image = (
            hasattr(first, "image")
            and isinstance(getattr(first, "image", None), np.ndarray)
            and hasattr(actual_obs, "image")
            and isinstance(getattr(actual_obs, "image", None), np.ndarray)
        )
        if not has_image:
            return [p.distance_soft(actual_obs) for p in predicted_obs]
        try:
            from uncertain_worms.environments.minigrid.minigrid_env import (
                distance_soft_batch_image, OBS_DIR_WEIGHT, OBS_DIR_EPS,
                OBS_DIST_MODE,
            )
        except ImportError:
            return [p.distance_soft(actual_obs) for p in predicted_obs]

        # Stack predicted images. All particles produced by the same
        # observation_func have the same FOV shape — `np.stack` will
        # raise if not, in which case we degrade to the slow path.
        try:
            stacked = np.stack([p.image for p in predicted_obs])
        except (ValueError, AttributeError):
            return [p.distance_soft(actual_obs) for p in predicted_obs]

        base_distances = distance_soft_batch_image(stacked, actual_obs.image)
        # Per-particle field penalties (cheap scalars). Mirrors the
        # tail of ``MinigridObservation.distance_soft``. Direct
        # attribute access — agent_dir/carrying are dataclass fields
        # always present on MinigridObservation.
        nats = (OBS_DIST_MODE == "nats")
        dir_nll_mismatch = -math.log(max(OBS_DIR_EPS, 1e-12))
        dir_pen_unit = dir_nll_mismatch if nats else 1.0
        actual_dir = actual_obs.agent_dir
        actual_carry = actual_obs.carrying

        result: List[float] = []
        for i, p in enumerate(predicted_obs):
            d = float(base_distances[i])
            if math.isinf(d):
                result.append(d)
                continue
            extra = 0.0
            if OBS_DIR_WEIGHT > 0.0:
                if p.agent_dir != actual_dir:
                    extra += OBS_DIR_WEIGHT * dir_pen_unit
            if p.carrying != actual_carry:
                extra += dir_pen_unit
            result.append(d + extra)
        return result

    def _compute_episode_score(
        self, episode, collect_distances: bool = False
    ) -> Tuple[List[ActType], List[ObsType], Optional[Dict[str, Any]]]:
        """
        Compute per-episode traces and optional PF diagnostics.

        Internal score:
            score = sum_t score_t
            score_t = -E_q[distance_soft(predicted_obs, actual_obs)]

        At step t: belief over s_t (state after transition with a_t, where we observe o_t).
        - t=0: no prior; rejuvenate with (a_0, o_0) from initial model
        - t>0: propagate from belief_{t-1} with (a_t, o_t)

        Returns:
            actions, observations, and optionally metrics dict
            (None if ``collect_distances`` is False).
        """
        actions = episode.actions
        observations = episode.next_observations

        if not observations:
            metrics: Optional[Dict[str, Any]] = None
            if collect_distances:
                metrics = {
                    "pf_score": float("-inf"),
                    "best_particle_distance": float("inf"),
                    "median_particle_distance": float("inf"),
                    "variance_distance": 0.0,
                    "mean_ESS": 0.0,
                    "min_ESS": 0.0,
                }
            return actions, observations, metrics

        total_score = 0.0
        current_belief: Optional[ParticleBelief] = None
        prev_weighted: Optional[Dict[StateType, float]] = None
        all_distances: List[float] = []
        ess_list: List[float] = []
        # Auxiliary diagnostic dict populated by reward/heuristic and
        # scorer-mode specific blocks below. Merged into ``metrics`` at
        # the bottom of the function when ``collect_distances`` is True.
        scorer_aux: Dict[str, Any] = {}

        # Per-step decomposition accumulators (only when self.decompose)
        per_step_decomp: List[Dict[str, float]] = []
        rewards = getattr(episode, "rewards", None)
        terminated_list = getattr(episode, "terminated", None)
        # Ground-truth trajectory states (for the belief-quality scorer).
        # When absent (non-minigrid envs or ReplayBuffer without state
        # logging) the belief-vs-real annotations stay None and the
        # belief_quality scorer degrades gracefully to pf_score=0.0.
        next_states_gt = getattr(episode, "next_states", None)
        # Tolerance for "reward correct": predictions within ε of actual.
        # Promoted to module-level ``REWARD_TOL`` (imported above) so the
        # ``_step_score`` helper can use the same threshold. Local alias
        # kept for readability of the surrounding block.

        # --- I3: inverse-frequency event weights ---
        # When ``self.event_weighted_scoring`` is True, precompute a per-step
        # weight ``event_weights[t]`` from the ground-truth trajectory. A step
        # is an "event" iff its reward is non-zero or it terminated. In sparse
        # reward POMDPs ~1 step per episode is an event, so the event class
        # frequency is ~5% — I3 upweights it proportionally so the scorer is
        # no longer dominated by the trivially-zero middle steps (scout B4).
        #
        # The weights are normalised so that ``sum(weights) == T``: event and
        # non-event classes each get half of the total weight budget. When
        # either class is empty (e.g., the episode contains no event, or is
        # trivially a single event step) all weights fall back to 1.0, making
        # the flag a no-op. When the flag is False all weights are 1.0, so
        # the weighted path is byte-identical to the pre-I3 formula.
        T = len(observations)
        event_weights: List[float] = [1.0] * T
        if self.event_weighted_scoring and T > 0 and rewards is not None and terminated_list is not None:
            classes: List[bool] = []
            for t in range(T):
                r = float(rewards[t]) if t < len(rewards) else 0.0
                term = bool(terminated_list[t]) if t < len(terminated_list) else False
                classes.append(bool(abs(r) > REWARD_TOL or term))
            n_events = sum(1 for c in classes if c)
            n_non = T - n_events
            if n_events > 0 and n_non > 0:
                w_event = T / (2.0 * n_events)
                w_non = T / (2.0 * n_non)
                event_weights = [w_event if c else w_non for c in classes]

        # Observation-conditioned scoring: group scores by (obs, action)
        obs_action_scores: Dict[Tuple[bytes, int], List[float]] = {}
        prev_obs_bytes: Optional[bytes] = None

        n_model_exceptions = 0  # track transition/observation model crashes
        # Capture the first traceback so the LLM sees the actual runtime error
        # (e.g. ``NameError: random is not defined``) instead of just a count.
        # Without this, silently-counted exceptions surface as "mauvais score"
        # in the feedback prompt with no actionable info.
        first_exception_trace: Optional[str] = None
        # Reset the init-model trace capture on this episode; ``_rejuvenate``
        # stores the first ``initial_model`` / ``transition_model`` crash
        # traceback on ``self._last_init_exception_trace`` while running.
        self._last_init_exception_trace = None

        for t, o_t in enumerate(observations):
            a_t = actions[t]

            if current_belief is None:
                # t=0: no prior; rejuvenate from scratch with (a_0, o_0)
                weighted_particles = self._rejuvenate_from_scratch(
                    action_history=[a_t],
                    observation_history=[o_t],
                )
            else:
                # t>0: propagate from current belief.
                # Two-phase batch path. Phase 1 collects all valid
                # (next_state, model_obs) pairs from
                # the LLM-generated transition+observation models (these
                # cannot be batched — the LLM code accepts one state at
                # a time). Phase 2 batches the image distance for the K
                # successful particles via ``distance_soft_batch_image``
                # (numba JIT, ~7-12x faster than K sequential calls on
                # 3x3 FOV). The per-particle agent_dir / carrying
                # penalty stays sequential — it's a scalar comparison,
                # negligible cost.
                weighted_particles = {}
                # Phase 1: collect valid particle propagations.
                phase1_next_states: List[StateType] = []
                phase1_model_obs: List[ObsType] = []
                for particle in current_belief.to_list():
                    try:
                        next_state = self.transition_model(particle.copy(), a_t)
                        model_obs = self.observation_model(
                            next_state.copy(), a_t, self.empty_observation
                        )
                    except Exception:
                        if first_exception_trace is None:
                            first_exception_trace = traceback.format_exc()
                        n_model_exceptions += 1
                        continue
                    phase1_next_states.append(next_state)
                    phase1_model_obs.append(model_obs)

                # Phase 2: batched distance + per-particle weighting.
                if phase1_model_obs:
                    distances_arr = self._batch_distance_or_fallback(
                        phase1_model_obs, o_t,
                    )
                    for i, (next_state, model_obs) in enumerate(zip(
                        phase1_next_states, phase1_model_obs,
                    )):
                        dist = float(distances_arr[i])
                        w = _soft_weight_from_distance(
                            dist, self.kernel_bandwidth, self.obs_noise_epsilon,
                        )
                        if w > 0.0:
                            weighted_particles[next_state] = (
                                weighted_particles.get(next_state, 0.0) + w
                            )

                # Rejuvenate if needed
                if sum(weighted_particles.values()) < self.num_particles:
                    weighted_particles = self._rejuvenate_step(
                        current_weighted=weighted_particles,
                        action_history=actions[: t + 1],
                        observation_history=observations[: t + 1],
                    )

            if not weighted_particles:
                metrics = None
                if collect_distances:
                    metrics = {
                        "pf_score": float("-inf"),
                        "best_particle_distance": float("inf"),
                        "median_particle_distance": float("inf"),
                        "variance_distance": 0.0,
                        "mean_ESS": 0.0,
                        "min_ESS": 0.0,
                        "collapse_step": t,
                        "n_model_exceptions": n_model_exceptions,
                        "first_exception_trace": (
                            first_exception_trace
                            or getattr(self, "_last_init_exception_trace", None)
                        ),
                    }
                return actions, observations, metrics

            if collect_distances:
                weights = list(weighted_particles.values())
                ess_list.append(_effective_sample_size(weights))

            # --- Per-step score ---
            # Prequential log score (Dawid 1984 / FIVO):
            #   score_t = log( (1/K) Σ_k w_k )
            # where w_k are the unnormalized particle weights from the
            # observation likelihood.  These are already in weighted_particles
            # (computed at lines 256-259 above, BEFORE rejuvenation).
            #
            # Energy score (original):
            #   score_t = -E_q[d_soft(o_pred, o_actual)]
            # computed by _step_score which re-evaluates the observation model.
            if self.scoring_mode == "prequential":
                raw_weights = list(weighted_particles.values())
                K = len(raw_weights)
                if K > 0 and sum(raw_weights) > 0:
                    score_t = math.log(sum(raw_weights) / max(K, 1))
                else:
                    score_t = float("-inf")
                step_distances = None
                step_decomp = None
            else:
                actual_r = rewards[t] if (rewards is not None and t < len(rewards)) else None
                actual_t = (
                    bool(terminated_list[t])
                    if (terminated_list is not None and t < len(terminated_list))
                    else None
                )
                score_t, step_distances, step_decomp = self._step_score(
                    weighted_particles, o_t, a_t,
                    return_distances=collect_distances,
                    actual_reward=actual_r,
                    prev_particles=prev_weighted,
                    actual_terminated=actual_t,
                )
            # I3: apply inverse-frequency event weight before accumulation.
            # event_weights[t] == 1.0 in the default path (flag False or
            # degenerate class counts), so the weighted line is byte-identical
            # to `total_score += score_t` when the flag is off.
            total_score += event_weights[t] * score_t
            if collect_distances and step_distances is not None:
                all_distances.extend(step_distances)
            if step_decomp is not None:
                step_decomp["t"] = float(t)
                # Annotate event-quality flags so we can later separate
                # informative reward steps (non-zero or terminal) from the
                # trivially-zero majority. The actual reward is the ground
                # truth from the trajectory.
                actual_r_now = float(rewards[t]) if (rewards is not None and t < len(rewards)) else 0.0
                terminal_now = bool(terminated_list[t]) if (terminated_list is not None and t < len(terminated_list)) else False
                step_decomp["actual_reward"] = actual_r_now
                step_decomp["actual_reward_nonzero"] = abs(actual_r_now) > REWARD_TOL
                step_decomp["terminal_step"] = terminal_now
                # Score_rew is -E[(r_pred - r_actual)^2]; if the squared
                # error is small the prediction matches actual.
                rew_err = -float(step_decomp.get("score_rew", 0.0))  # ≥0
                value_ok = rew_err < (REWARD_TOL ** 2)
                term_pred = bool(step_decomp.get("term_pred", False))
                term_ok = (term_pred == terminal_now)
                step_decomp["reward_value_correct"] = value_ok
                step_decomp["terminated_correct"] = term_ok
                # A step is "reward_correct" only if BOTH the reward
                # value AND the terminated flag match — a reward_func
                # that returns ``(0, True)`` for a non-terminal step
                # must not score as perfect.
                step_decomp["reward_correct"] = value_ok and term_ok

                # MODEL FIT trace data — stash the human-readable fields
                # the LLM-facing formatter needs to display per-step
                # mismatches (predicted obs vs observed obs, predicted
                # reward/term vs observed reward/term, action name).
                # All values stay belief-based: ``pred_obs_image`` and
                # ``pred_reward/pred_terminated`` are queried on the MAP
                # particle of the current belief, which itself is built
                # from observations only — no state-oracle lookup. The
                # observed image / reward / terminated fields come from
                # the buffer (legitimate observations).
                step_decomp["action_id"] = int(a_t) if a_t is not None else None
                step_decomp["observed_reward"] = actual_r_now
                step_decomp["observed_terminated"] = terminal_now
                if hasattr(o_t, "image"):
                    try:
                        step_decomp["obs_image"] = o_t.image.tolist()
                    except Exception:
                        step_decomp["obs_image"] = None
                else:
                    step_decomp["obs_image"] = None
                if weighted_particles:
                    try:
                        map_state = max(
                            weighted_particles, key=weighted_particles.get
                        )
                        map_obs_pred = self.observation_model(
                            map_state.copy(), a_t,
                            self.empty_observation.copy(),
                        )
                        if hasattr(map_obs_pred, "image"):
                            step_decomp["pred_obs_image"] = (
                                map_obs_pred.image.tolist()
                            )
                        else:
                            step_decomp["pred_obs_image"] = None
                        try:
                            map_next = self.transition_model(
                                map_state.copy(), a_t
                            )
                            r_pred_map, t_pred_map = self.reward_model(
                                map_state.copy(), a_t,
                                map_next.copy(),
                            )
                            step_decomp["pred_reward"] = float(r_pred_map)
                            step_decomp["pred_terminated"] = bool(t_pred_map)
                        except Exception:
                            step_decomp["pred_reward"] = None
                            step_decomp["pred_terminated"] = None
                    except Exception:
                        step_decomp["pred_obs_image"] = None
                        step_decomp["pred_reward"] = None
                        step_decomp["pred_terminated"] = None
                else:
                    step_decomp["pred_obs_image"] = None
                    step_decomp["pred_reward"] = None
                    step_decomp["pred_terminated"] = None

                # Belief-quality annotations (ORACLE: reads true state).
                # Weight of particles whose ``agent_pos`` (resp. exact
                # state) matches the ground-truth next state. Consumed
                # only by the ``belief_quality`` scorer. Gated on
                # scoring_mode to avoid reading true states in production.
                step_decomp["belief_vs_real_pos_weight"] = None
                step_decomp["belief_vs_real_weight"] = None
                if (
                    self.scoring_mode == "belief_quality"
                    and next_states_gt is not None
                    and t < len(next_states_gt)
                    and weighted_particles
                ):
                    actual_next_state = next_states_gt[t]
                    try:
                        total_w = float(sum(weighted_particles.values()))
                        if total_w > 0.0:
                            true_pos = getattr(
                                actual_next_state, "agent_pos", None
                            )
                            if true_pos is not None:
                                true_pos_t = tuple(true_pos)
                                w_on_pos = 0.0
                                for s_i, w_i in weighted_particles.items():
                                    sp = getattr(s_i, "agent_pos", None)
                                    if sp is not None and tuple(sp) == true_pos_t:
                                        w_on_pos += float(w_i)
                                step_decomp["belief_vs_real_pos_weight"] = (
                                    w_on_pos / total_w
                                )
                            w_on_state = float(
                                weighted_particles.get(actual_next_state, 0.0)
                            )
                            step_decomp["belief_vs_real_weight"] = (
                                w_on_state / total_w
                            )
                    except Exception:
                        # Keep None on any comparison failure (e.g.
                        # non-hashable state, missing agent_pos).
                        pass

                per_step_decomp.append(step_decomp)

            # Accumulate obs-conditioned scores.
            # MIN1: fall back to ``repr(o_t)`` instead of ``hash(o_t)`` when
            # the observation has no numpy image — repr is deterministic
            # across processes whereas hash() depends on PYTHONHASHSEED.
            if not math.isinf(score_t):
                obs_bytes = (
                    o_t.image.tobytes() if hasattr(o_t, "image")
                    else repr(o_t).encode("utf-8")
                )
                obs_changed = (prev_obs_bytes is not None and obs_bytes != prev_obs_bytes)
                key = (obs_bytes, a_t)
                entry = {"score": score_t, "t": t, "obs_changed": obs_changed}
                # Store the actual observation grid for readable feedback
                if hasattr(o_t, "image"):
                    entry["obs_grid"] = o_t.image.tolist()
                obs_action_scores.setdefault(key, []).append(entry)
                prev_obs_bytes = obs_bytes

            prev_weighted = weighted_particles
            current_belief = self._weighted_to_belief(weighted_particles)

        # Preliminary #8 — fold explicit reward MSE into total_score.
        # ``per_step_decomp`` is only populated when ``self.decompose=True``
        # and the scoring mode is not prequential; otherwise the list is
        # empty and the addition is a no-op. Each ``score_rew`` entry is
        # already ≤ 0 (it is ``-E[(r_pred - r_actual)^2]``), so multiplying
        # by ``reward_weight`` and adding keeps the per-step sign
        # convention: better reward prediction → less-negative pf_score.
        if self.reward_weight > 0.0 and per_step_decomp:
            reward_term = float(np.sum(
                [d.get("score_rew", 0.0) for d in per_step_decomp]
            ))
            total_score += self.reward_weight * reward_term

        # Strictly proper logarithmic scoring rule on observed rewards
        # (Gneiting & Raftery 2007). Computes
        # ``-Σ_t log p(r_t | model)`` where ``p(r_t | model)`` is the
        # empirical fraction of rolled-out particles whose predicted
        # reward matches the observed one. Uses Laplace smoothing so the
        # term stays finite. Subtracts from total_score (convention
        # F = -score, so larger D_reward → smaller score).
        #
        # Pure POMDP: only observable signals (``rewards[t]`` is the
        # reward the agent received at step t, available via
        # ``env.step()``); particles are rolled out from the candidate's
        # own initial_func / transition_func / reward_func — no peek at
        # env ground-truth state.
        #
        # Activation: REX_DREWARD=1 (default off — byte-identical to the
        # pre-flag scorer). Tunable knobs:
        #   REX_DREWARD_N        — particles per rollout (default 10)
        #   REX_DREWARD_LAPLACE  — Laplace smoothing alpha (default 1.0)
        import os as _dr_os
        # Snapshot the obs-only score before D_reward / D_heur subtract
        # any penalty — exposed in metrics so the breakdown is visible.
        scorer_aux["pf_score_obs_only"] = float(total_score)

        _dr_flag = _dr_os.environ.get("REX_DREWARD", "0").strip().lower()
        if (
            _dr_flag in ("1", "true", "yes", "on")
            and rewards is not None
            and len(rewards) > 0
            and len(actions) > 0
        ):
            try:
                _dr_n = max(2, int(_dr_os.environ.get("REX_DREWARD_N", "10")))
                _dr_alpha = float(_dr_os.environ.get("REX_DREWARD_LAPLACE", "1.0"))
                # Sample N rollout particles from candidate's initial_model
                _dr_particles: List[Any] = []
                for _ in range(_dr_n):
                    try:
                        sp = self.initial_model(self.empty_state.copy())
                        _dr_particles.append(sp)
                    except Exception:
                        continue
                # Roll out forward, log score per step
                _dr_total = 0.0
                _dr_n_eval = 0
                for t_dr, a_dr in enumerate(actions):
                    if t_dr >= len(rewards):
                        break
                    r_real = float(rewards[t_dr])
                    n_match = 0
                    n_eval = 0
                    new_particles: List[Any] = []
                    for p_dr in _dr_particles:
                        try:
                            s_next_dr = self.transition_model(p_dr.copy(), a_dr)
                            r_pred_dr, _ = self.reward_model(
                                p_dr, a_dr, s_next_dr
                            )
                            if abs(float(r_pred_dr) - r_real) < REWARD_TOL:
                                n_match += 1
                            n_eval += 1
                            new_particles.append(s_next_dr)
                        except Exception:
                            continue
                    _dr_particles = new_particles
                    if n_eval > 0:
                        # Laplace-smoothed log score (binary success/fail
                        # in matching the reward signal).
                        p_correct = (n_match + _dr_alpha) / (
                            n_eval + 2.0 * _dr_alpha
                        )
                        _dr_total += -math.log(max(p_correct, 1e-12))
                        _dr_n_eval += 1
                    if not _dr_particles:
                        # All particles crashed; remaining steps would
                        # be empty — penalize the rest as worst-case.
                        remaining = max(0, len(rewards) - t_dr - 1)
                        if remaining > 0 and _dr_n > 0:
                            worst = -math.log(_dr_alpha / (_dr_n + 2.0 * _dr_alpha))
                            _dr_total += worst * remaining
                            _dr_n_eval += remaining
                        break
                # Subtract D_reward from total_score (F = -score so a
                # larger D_reward shrinks the score).
                total_score -= _dr_total
                # Stash for diagnostics — scorer_aux is initialised at
                # the top of the function so this propagates to metrics.
                scorer_aux["d_reward"] = float(_dr_total)
                scorer_aux["d_reward_n_eval"] = int(_dr_n_eval)
            except Exception:
                # Belt-and-suspenders: never let the new term break the
                # eval. On any internal error, log and continue with the
                # baseline total_score.
                log.warning(
                    "[D_REWARD] log-score computation raised; "
                    "skipping the term for this episode (no penalty)",
                    exc_info=True,
                )

        # Admissibility log-score for the heuristic.
        # On winning episodes (``r_terminal > 0``), the true cost-to-go
        # at step t is exactly ``T - t - 1`` (number of actions remaining
        # before the terminal reward) — observable, no peek at state.
        # An admissible heuristic must satisfy ``h(s) ≤ remaining`` for
        # every particle of the belief. Each violation costs a fixed
        # log-loss penalty per particle, normalised so the term stays on
        # the same nats scale as D_reward / D_obs.
        #
        # Pure POMDP: only uses ``rewards[-1]``, ``len(actions)``, and
        # the candidate's own ``heuristic_model`` evaluated on rolled-out
        # particles. No env state access.
        #
        # Activation: REX_DHEUR=1 (default off). Knobs:
        #   REX_DHEUR_N      — particles per rollout (default 10)
        #   REX_DHEUR_LAPLACE — Laplace alpha (default 1.0)
        _dh_flag = _dr_os.environ.get("REX_DHEUR", "0").strip().lower()
        if (
            _dh_flag in ("1", "true", "yes", "on")
            and rewards is not None
            and len(rewards) > 0
            and len(actions) > 0
            and hasattr(self, "heuristic_model")
            and self.heuristic_model is not None
        ):
            try:
                # Only score on winning trajectories — fails have no
                # observable cost-to-go, scoring there would inject
                # noise (the agent's failure path may have arbitrary
                # remaining-step counts that the heuristic shouldn't
                # match).
                final_r = float(rewards[-1])
                if final_r > REWARD_TOL:
                    _dh_n = max(2, int(_dr_os.environ.get("REX_DHEUR_N", "10")))
                    _dh_alpha = float(_dr_os.environ.get("REX_DHEUR_LAPLACE", "1.0"))
                    # Sample N rollout particles from initial_func
                    _dh_particles: List[Any] = []
                    for _ in range(_dh_n):
                        try:
                            sp = self.initial_model(self.empty_state.copy())
                            _dh_particles.append(sp)
                        except Exception:
                            continue
                    _dh_total = 0.0
                    _dh_n_eval = 0
                    T_full = len(actions)
                    for t_dh, a_dh in enumerate(actions):
                        remaining = T_full - t_dh
                        if not _dh_particles:
                            break
                        n_admissible = 0
                        n_eval = 0
                        new_dh: List[Any] = []
                        for p_dh in _dh_particles:
                            try:
                                # Heuristic must be ≤ remaining for
                                # admissibility on a winning trajectory.
                                h_val = float(self.heuristic_model(p_dh))
                                if h_val <= remaining + REWARD_TOL:
                                    n_admissible += 1
                                n_eval += 1
                                # Propagate forward for the next step
                                s_next_dh = self.transition_model(
                                    p_dh.copy(), a_dh
                                )
                                new_dh.append(s_next_dh)
                            except Exception:
                                continue
                        _dh_particles = new_dh
                        if n_eval > 0:
                            p_admissible = (
                                (n_admissible + _dh_alpha)
                                / (n_eval + 2.0 * _dh_alpha)
                            )
                            _dh_total += -math.log(max(p_admissible, 1e-12))
                            _dh_n_eval += 1
                    total_score -= _dh_total
                    scorer_aux["d_heur"] = float(_dh_total)
                    scorer_aux["d_heur_n_eval"] = int(_dh_n_eval)
            except Exception:
                log.warning(
                    "[D_HEUR] admissibility log-score raised; "
                    "skipping the term for this episode (no penalty)",
                    exc_info=True,
                )

        # Alternative scorers — opt-in via ``scoring_mode``. Each replaces
        # ``total_score`` with a scalar that shares the *higher is better*
        # convention of ``energy`` / ``prequential`` so that
        # ``_elbo_components_from_episode_metrics`` can consume it
        # Experimental scoring modes (whiteness, betting, bisim,
        # pf_self_consistency, prediction_persistence, event, hybrid_pf_event)
        # were ablation probes during development and depend on modules that
        # are not shipped with this paper release. The only modes wired in
        # this bundle are:
        #   * "energy"          — paper default (distance-kernel
        #                         pseudo-likelihood, Eq. 7 / Eq. 8).
        #   * "belief_quality"  — oracle reference, requires ground-truth
        #                         states from the replay buffer. Used for
        #                         calibration only.
        _EXPERIMENTAL = (
            "whiteness", "betting", "bisim",
            "pf_self_consistency", "prediction_persistence",
            "event", "hybrid_pf_event",
        )
        if self.scoring_mode in _EXPERIMENTAL:
            raise NotImplementedError(
                f"scoring_mode={self.scoring_mode!r} is an experimental "
                "probe not shipped with this release. Use 'energy' (paper "
                "default) or 'belief_quality' (oracle reference)."
            )

        if self.scoring_mode == "belief_quality":
            # ORACLE DIAGNOSTIC: reads ground-truth states from the replay
            # buffer (``episode.next_states``). Must NOT be used in the
            # production scoring loop because it violates the obs-only
            # constraint.
            # Kept wired for offline canary scripts only.
            log.warning(
                "[ScoreEvaluator] scoring_mode=belief_quality is an ORACLE "
                "diagnostic (reads true states). Do NOT deploy in online/ "
                "production scoring. Use only in canary/calibration scripts."
            )
            if not per_step_decomp:
                log.warning(
                    "[ScoreEvaluator] scoring_mode=belief_quality requires "
                    "decompose=True with per_step_decomp populated; falling "
                    "back to energy."
                )
            else:
                from particle_filtering.belief_quality_scorer import (
                    score_trajectory as belief_quality_score,
                )
                report = belief_quality_score(per_step_decomp)
                total_score = float(report.pf_score)
                scorer_aux = {
                    "belief_quality_pf_score": float(report.pf_score),
                    "belief_quality_weight_on_pos_mean": float(
                        report.weight_on_pos_mean
                    ),
                    "belief_quality_weight_on_state_mean": float(
                        report.weight_on_state_mean
                    ),
                    "belief_quality_n_valid_steps": int(report.n_valid_steps),
                    "belief_quality_n_total_steps": int(report.n_total_steps),
                }

        metrics: Optional[Dict[str, Any]] = None
        if collect_distances:
            metrics = {
                "pf_score": total_score,
                "best_particle_distance": float(np.min(all_distances)) if all_distances else 0.0,
                "median_particle_distance": float(np.median(all_distances)) if all_distances else 0.0,
                "variance_distance": float(np.var(all_distances)) if all_distances else 0.0,
                "mean_ESS": float(np.mean(ess_list)) if ess_list else 0.0,
                "min_ESS": float(np.min(ess_list)) if ess_list else 0.0,
                "n_model_exceptions": n_model_exceptions,
                "first_exception_trace": (
                    first_exception_trace
                    or getattr(self, "_last_init_exception_trace", None)
                ),
            }
            if scorer_aux:
                metrics.update(scorer_aux)

        # Observation-conditioned group summary
        if obs_action_scores and metrics is not None:
            action_names = {0: "LEFT", 1: "RIGHT", 2: "FWD", 4: "DROP", 5: "TOGGLE"}
            group_summaries = []
            for (obs_b, act), entries in obs_action_scores.items():
                scores = [e["score"] for e in entries]
                has_change = any(e["obs_changed"] for e in entries)
                # Get the actual grid from the first entry that has it
                obs_grid = None
                for e in entries:
                    if "obs_grid" in e:
                        obs_grid = e["obs_grid"]
                        break
                group_summaries.append({
                    "obs_hash": obs_b[:8].hex(),
                    "obs_grid": obs_grid,
                    "action": action_names.get(act, str(act)),
                    "action_id": act,
                    "count": len(scores),
                    "mean_score": float(np.mean(scores)),
                    "has_obs_change": has_change,
                })
            # Sort by mean_score ascending (worst first)
            group_summaries.sort(key=lambda g: g["mean_score"])
            metrics["obs_action_groups"] = group_summaries

        # Append FIVO decomposition when enabled
        if self.decompose and per_step_decomp and metrics is not None:
            obs_scores = [d["score_obs"] for d in per_step_decomp]
            rew_scores = [d.get("score_rew", 0.0) for d in per_step_decomp]
            trans_scores = [d["score_trans"] for d in per_step_decomp]
            ess_vals = [d["ess"] for d in per_step_decomp]

            metrics["fivo_score_obs"] = float(np.sum(obs_scores))
            metrics["fivo_score_rew"] = float(np.sum(rew_scores))
            metrics["fivo_score_trans"] = float(np.mean(trans_scores))
            metrics["fivo_mean_ess"] = float(np.mean(ess_vals))
            metrics["fivo_min_ess"] = float(np.min(ess_vals))
            metrics["fivo_per_step"] = per_step_decomp

            # F5 (R#5-A) — episode-level terminated cross-entropy.
            # Summing per-step ``term_nats`` gives -Σ log p(t_actual_t)
            # i.e. the negative log-likelihood of the observed terminated
            # sequence under the belief; the sign is flipped to match
            # the other ``fivo_score_*`` conventions where higher = better
            # (so ``fivo_score_term = -Σ term_nats``). Steps whose
            # decomposition lacked ``term_nats`` (missing
            # ``actual_terminated`` at scoring time) are skipped rather
            # than replaced with zeros — silently filling in zero would
            # confound an honest "good flag predictions" signal with
            # "scoring never saw the flag".
            term_nats_vals = [
                d["term_nats"]
                for d in per_step_decomp
                if d.get("term_nats") is not None
            ]
            if term_nats_vals:
                metrics["fivo_score_term"] = -float(np.sum(term_nats_vals))
                metrics["fivo_mean_term_nats"] = float(np.mean(term_nats_vals))
            else:
                metrics["fivo_score_term"] = float("nan")
                metrics["fivo_mean_term_nats"] = float("nan")

            # --- Reward-event focused metrics ---
            # In sparse-reward MDPs, reward = 0 for ~90% of steps. A model
            # that always predicts 0 looks "perfect" on plain MSE. We need
            # to separate trivial-zero predictions from meaningful events.
            #
            # Definitions (env-agnostic):
            #   nonzero step  = step where actual reward != 0 OR terminated
            #   reward_correct(step) = pred matches actual (with tol)
            n_steps = len(per_step_decomp)
            nonzero_indices = []
            for i, d in enumerate(per_step_decomp):
                if d.get("actual_reward_nonzero") or d.get("terminal_step"):
                    nonzero_indices.append(i)
            metrics["n_reward_events"] = len(nonzero_indices)
            metrics["n_total_steps"] = n_steps

            # Per-step reward correctness (within ε)
            n_correct = sum(1 for d in per_step_decomp
                            if d.get("reward_correct"))
            metrics["reward_accuracy_overall"] = n_correct / max(n_steps, 1)

            n_nonzero_correct = sum(
                1 for d in per_step_decomp
                if (d.get("actual_reward_nonzero") or d.get("terminal_step"))
                and d.get("reward_correct")
            )
            metrics["reward_accuracy_on_events"] = (
                n_nonzero_correct / len(nonzero_indices)
                if nonzero_indices else float("nan")
            )

            # Per-dimension breakdown so the LLM can see which dim fails.
            # The headline ``reward_accuracy_on_events`` requires BOTH the
            # reward value AND the terminated flag to match; these two
            # metrics isolate each dimension across ALL steps so a drop in
            # one does not get hidden by success in the other.
            n_value_correct = sum(
                1 for d in per_step_decomp if d.get("reward_value_correct")
            )
            n_term_correct = sum(
                1 for d in per_step_decomp if d.get("terminated_correct")
            )
            metrics["reward_value_accuracy"] = n_value_correct / max(n_steps, 1)
            metrics["terminated_accuracy"] = n_term_correct / max(n_steps, 1)

            # MSE restricted to non-zero events (the truly informative ones)
            if nonzero_indices:
                event_errs = [
                    -per_step_decomp[i]["score_rew"]  # score is -MSE
                    for i in nonzero_indices
                    if "score_rew" in per_step_decomp[i]
                ]
                metrics["fivo_score_rew_events"] = (
                    -float(np.mean(event_errs)) if event_errs else 0.0
                )
                # Joint (value + terminated) per-event energy. Same
                # convention as ``fivo_score_rew_events``: higher
                # (closer to 0) is better. Consumed by the Pareto
                # selector as its reward channel so that candidates
                # differing only on ``terminated`` are distinguishable.
                # Falls back to ``score_rew`` when the step decomp does
                # not carry ``score_reward_joint`` (legacy scoring
                # path), so old runs stay comparable.
                joint_errs = [
                    -per_step_decomp[i].get(
                        "score_reward_joint",
                        per_step_decomp[i].get("score_rew", 0.0),
                    )
                    for i in nonzero_indices
                ]
                metrics["fivo_score_reward_events_joint"] = (
                    -float(np.mean(joint_errs)) if joint_errs else 0.0
                )
                # Continuous likelihood version of the headline
                # ``reward + terminated correct on X/Y events``: the
                # legacy headline is a binary AND on a majority-vote
                # over particles, so a belief where 40% of the mass
                # predicts the correct (r, t) tuple registers as a
                # flat zero. ``fivo_event_log_likelihood`` keeps that
                # graduality by averaging log(P(match | belief)) across
                # events, with a small floor so log(0) is finite.
                # POMDP-pure: consumes ``explain_likelihood`` computed
                # from particle weights, no new state information.
                likelihoods = [
                    per_step_decomp[i].get("explain_likelihood", float("nan"))
                    for i in nonzero_indices
                ]
                valid = [
                    float(p) for p in likelihoods
                    if p is not None
                    and not (isinstance(p, float) and math.isnan(p))
                ]
                if valid:
                    # 1e-6 matches a ``fivo_event_log_likelihood`` floor
                    # of ≈ −13.8 for a candidate that cannot explain
                    # any event — enough to sink Pareto's reward
                    # channel relative to any candidate with even
                    # marginal coverage. ε is large enough to stay
                    # numerically safe with float32 agents in the wild.
                    metrics["fivo_event_log_likelihood"] = float(
                        np.mean([math.log(max(p, 1e-6)) for p in valid])
                    )
                else:
                    metrics["fivo_event_log_likelihood"] = float("nan")
            else:
                metrics["fivo_score_rew_events"] = float("nan")
                metrics["fivo_score_reward_events_joint"] = float("nan")
                metrics["fivo_event_log_likelihood"] = float("nan")

        # ----- Multi-Step Consistency (MSC) scoring -----
        # Measures whether T^d(s_t, a_{t:t+d-1}) still explains o_{t+d}.
        # A model with compounding transition errors will score poorly here
        # even if 1-step observation prediction is acceptable.
        d = self.msc_horizon
        if d > 0 and metrics is not None and len(observations) > d:
            msc_errors: List[float] = []
            # Re-run filtering to collect beliefs at each step
            belief_t: Optional[ParticleBelief] = None
            beliefs: List[Optional[ParticleBelief]] = []
            for t in range(len(observations)):
                a_t, o_t = actions[t], observations[t]
                if belief_t is None:
                    wp = self._rejuvenate_from_scratch([a_t], [o_t])
                else:
                    wp = {}
                    for p in belief_t.to_list():
                        try:
                            ns = self.transition_model(p.copy(), a_t)
                            mo = self.observation_model(
                                ns.copy(), a_t, self.empty_observation
                            )
                        except Exception:
                            continue
                        dist = mo.distance_soft(o_t)
                        w = _soft_weight_from_distance(dist, self.kernel_bandwidth, self.obs_noise_epsilon)
                        if w > 0:
                            wp[ns] = wp.get(ns, 0.0) + w
                    if sum(wp.values()) < self.num_particles:
                        wp = self._rejuvenate_step(
                            wp, actions[:t+1], observations[:t+1]
                        )
                beliefs.append(
                    self._weighted_to_belief(wp) if wp else None
                )
                belief_t = beliefs[-1]

            # For each valid starting step t, propagate d steps open-loop
            for t in range(len(observations) - d):
                if beliefs[t] is None:
                    continue
                particles = beliefs[t].to_list()
                if not particles:
                    continue

                step_errors: List[float] = []
                for p in particles[:20]:  # cap to avoid expensive loop
                    state = p.copy()
                    failed = False
                    for k in range(d):
                        try:
                            state = self.transition_model(state, actions[t + k])
                        except Exception:
                            failed = True
                            break
                    if failed:
                        continue
                    try:
                        pred_obs = self.observation_model(
                            state.copy(), actions[t + d - 1],
                            self.empty_observation,
                        )
                        dist = pred_obs.distance_soft(observations[t + d - 1])
                        if not math.isinf(dist) and dist >= 0:
                            step_errors.append(dist)
                    except Exception:
                        continue

                if step_errors:
                    msc_errors.append(float(np.mean(step_errors)))

            if msc_errors:
                metrics["msc_score"] = -float(np.mean(msc_errors))
                metrics["msc_std"] = float(np.std(msc_errors))
                metrics["msc_horizon"] = d
            else:
                metrics["msc_score"] = float("-inf")
                metrics["msc_std"] = 0.0
                metrics["msc_horizon"] = d

        return actions, observations, metrics

    def _step_score(
        self,
        weighted_particles: Dict[StateType, float],
        observation: ObsType,
        action: ActType,
        return_distances: bool = False,
        actual_reward: Optional[float] = None,
        prev_particles: Optional[Dict[StateType, float]] = None,
        actual_terminated: Optional[bool] = None,
    ) -> Tuple[float, Optional[List[float]], Optional[Dict[str, float]]]:
        """Compute per-step observation score and optional FIVO decomposition.

        score_obs_t = -E_q[distance_soft(predicted_obs, actual_obs)]

        When ``self.decompose`` is True and ``actual_reward`` is provided,
        additionally computes:
            score_rew_t         = -E_q[(r_pred - r_actual)^2]
            score_trans_t       = log(ESS_t / K)
            score_rew_joint_t   = -E_q[(r_pred - r_actual)^2
                                       + w_term * (t_pred - t_actual)^2]

        ``score_rew_joint`` (only populated when ``actual_terminated`` is
        provided) is the same energy form as ``score_rew`` but augmented
        with the terminated-flag mismatch. This keeps ``score_rew``
        byte-identical for callers that rely on FIVO-style MSE-only
        reward decomposition, while exposing a reward channel that is
        not blind to ``terminated`` (a common failure mode: the LLM
        correctly predicts ``reward=0`` on both a safe step and a fatal
        one, but only the fatal one has ``terminated=True``). The weight
        ``w_term`` is read from the ``REW_TERM_WEIGHT`` env var (default
        5.0) so it can be swept without code change.

        POMDP-pure: both ``actual_reward`` and ``actual_terminated`` are
        outcomes emitted by ``env.step`` and stored in the replay buffer
        — no state-oracle lookup.

        Returns:
            score_obs_t,
            distances list (or None),
            decomposition dict (or None when decompose=False)
        """
        total_w = sum(weighted_particles.values())
        if total_w <= 0:
            return float("-inf"), None, None

        distances: List[float] = []
        weights_list: List[float] = []

        for state, w in weighted_particles.items():
            if w <= 0:
                continue
            try:
                model_obs = self.observation_model(
                    state.copy(), action, self.empty_observation
                )
                # ``agent_dir`` is a perceptual signal (compass /
                # proprioception). The LLM's ``obs_func`` may or may
                # not stamp it, so we backfill from the particle state
                # when available so ``distance_soft`` always compares
                # the same field. ``reward`` and ``terminated`` are
                # NOT fields of the observation any more — they are
                # outcomes of ``reward_func`` and scored separately.
                if hasattr(model_obs, "agent_dir"):
                    state_dir = getattr(state, "agent_dir", None)
                    if state_dir is not None:
                        try:
                            model_obs.agent_dir = int(state_dir)
                        except Exception:
                            pass
                dist = model_obs.distance_soft(observation)
                if not math.isinf(dist) and dist >= 0:
                    distances.append(dist)
                    weights_list.append(w)
            except Exception:
                continue

        if not weights_list:
            return float("-inf"), None, None

        weights_arr = np.array(weights_list)
        distances_arr = np.array(distances)
        Z = np.sum(weights_arr)
        probs = weights_arr / Z

        # E_q[distance] — reuse distances_arr below for hard_match_rate.
        expected_dist = float(np.sum(probs * distances_arr))
        score_obs = -expected_dist

        decomp: Optional[Dict[str, float]] = None
        if self.decompose:
            decomp = {"score_obs": score_obs}

            # --- score_rew: reward prediction error ---
            # Capture BOTH reward value AND terminated flag from the LLM's
            # reward_func so downstream metrics can honestly combine the
            # two dimensions. Before this change, ``_`` silently dropped
            # ``t_pred`` and the "reward_func accuracy" headline only
            # reflected the reward value — a model that predicted
            # ``(0, terminated=True)`` for a non-terminal step scored
            # perfect on the headline despite mispredicting the flag.
            if actual_reward is not None and prev_particles is not None:
                rew_errors: List[float] = []
                r_preds_raw: List[float] = []
                term_preds: List[int] = []
                rew_weights: List[float] = []
                # Hoist prev_state lookup out of the per-particle loop —
                # the MAP is a property of prev_particles, not of the
                # current state, so it is constant across the inner
                # iteration. Saves O(K) max-scan × K particles.
                if prev_particles:
                    prev_state_map = max(prev_particles, key=prev_particles.get)
                else:
                    prev_state_map = None
                for state, w in weighted_particles.items():
                    if w <= 0:
                        continue
                    prev_state = prev_state_map if prev_state_map is not None else state
                    try:
                        r_pred, t_pred = self.reward_model(
                            prev_state.copy(), action, state.copy()
                        )
                        rew_errors.append((r_pred - actual_reward) ** 2)
                        r_preds_raw.append(float(r_pred))
                        term_preds.append(int(bool(t_pred)))
                        rew_weights.append(w)
                    except Exception:
                        continue
                if rew_weights:
                    # Pre-build numpy arrays once and reuse across all
                    # downstream metrics — saves ~5 redundant np.array
                    # allocations per step (rw, rew_errors, term_preds,
                    # r_preds_raw, term_errors, matches).
                    rw = np.array(rew_weights)
                    rw = rw / np.sum(rw)
                    rew_errors_arr = np.array(rew_errors)
                    term_preds_arr = np.array(term_preds)
                    r_preds_raw_arr = np.array(r_preds_raw)
                    decomp["score_rew"] = -float(np.sum(rw * rew_errors_arr))
                    # Weighted majority vote for the terminated flag
                    # (each particle contributes 0 or 1, weighted by w).
                    decomp["term_pred"] = bool(
                        float(np.sum(rw * term_preds_arr)) >= 0.5
                    )
                    # Belief-averaged continuous summaries used by the
                    # non-likelihood scorers (whiteness/betting/bisim).
                    # These complement — rather than replace — the
                    # MAP-particle ``pred_reward`` / ``pred_terminated``
                    # that the caller writes a few lines later: the
                    # scorers prefer the continuous form when present
                    # (``d.get("pred_term_prob", ...)``) and otherwise
                    # fall back to the MAP bool. POMDP-pure: only the
                    # particle weights and the LLM-generated
                    # ``reward_model`` outputs are consumed — no state
                    # oracle.
                    decomp["pred_term_prob"] = float(
                        np.sum(rw * term_preds_arr)
                    )
                    decomp["pred_reward_exp"] = float(
                        np.sum(rw * r_preds_raw_arr)
                    )
                    # --- score_reward_joint: (value, terminated) energy ---
                    # Identical form to score_rew but the energy adds a
                    # term for the terminated mismatch. When
                    # ``actual_terminated`` is not supplied (legacy
                    # callers) we leave the joint score equal to the
                    # MSE-only reward score — no silent regression.
                    if actual_terminated is not None:
                        w_term = float(os.environ.get("REW_TERM_WEIGHT", "5.0"))
                        t_actual_int = int(bool(actual_terminated))
                        # Vectorised term_errors instead of list-comp +
                        # np.array — single allocation, no Python loop.
                        term_errors_arr = (term_preds_arr - t_actual_int) ** 2
                        joint_err = rew_errors_arr + w_term * term_errors_arr
                        decomp["score_reward_joint"] = -float(
                            np.sum(rw * joint_err)
                        )

                        # F5 (R#5-A) — cross-entropy terminated channel.
                        # Scoring the TERMINATED flag on its own axis lets
                        # the Pareto selector keep a candidate that
                        # correctly predicts `done=True` on lava even
                        # when its reward values wash out in the
                        # squared-error aggregate. pred_term_prob is
                        # already the belief-weighted probability of
                        # predicting terminated; the CE of the observed
                        # flag against that probability is a proper
                        # log-loss that composes cleanly with the
                        # obs/trans/rew log channels in the Pareto
                        # front. Clamped to avoid -log(0) when the
                        # belief is fully wrong.
                        from uncertain_worms.policies.llm_feedback import (
                            term_cross_entropy,
                        )
                        decomp["term_nats"] = float(
                            term_cross_entropy(
                                float(decomp["pred_term_prob"]),
                                bool(t_actual_int),
                            )
                        )
                        # --- explain_likelihood: P(exact match | belief) ---
                        # Continuous alternative to the binary
                        # ``reward_correct = value_ok AND term_ok`` flag
                        # that dominates the feedback headline. Instead
                        # of testing whether the *weighted majority* of
                        # particles predicts the exact (r, t) pair, we
                        # sum the weight of particles that predict the
                        # exact tuple. The output lives in [0, 1] and
                        # preserves graduality: if 40% of the particle
                        # mass explains the event, ``explain_likelihood
                        # = 0.4`` instead of the ``majority vote = 0``
                        # that the legacy binary returns. No new belief
                        # is computed; we reuse ``rew_errors /
                        # term_errors`` already weighted by the same
                        # ``rw``. POMDP-pure: only (reward, terminated)
                        # from the buffer and particle weights are
                        # consumed — no state oracle, no observation
                        # decode.
                        # Vectorised exact-match mask: per-particle bool of
                        # (rew within tol AND term flag exact). Saves the
                        # Python list-comp + np.array trip.
                        matches_arr = (
                            (rew_errors_arr < (REWARD_TOL ** 2))
                            & (term_errors_arr == 0)
                        ).astype(float)
                        decomp["explain_likelihood"] = float(
                            np.sum(rw * matches_arr)
                        )
                    else:
                        decomp["score_reward_joint"] = decomp["score_rew"]
                        decomp["explain_likelihood"] = float("nan")
                        # F5 — term_nats only makes sense when we have
                        # the actual terminated flag. Absent otherwise.
                        decomp["term_nats"] = None
                else:
                    decomp["score_rew"] = 0.0
                    decomp["score_reward_joint"] = 0.0
                    decomp["explain_likelihood"] = float("nan")
                    # Continuous belief-averaged fields: absent (via
                    # ``None``) rather than silently 0.0 — scorers that
                    # ``.get()`` with a fallback handle this cleanly.
                    decomp["pred_term_prob"] = None
                    decomp["pred_reward_exp"] = None
                    decomp["term_nats"] = None

            # --- score_trans: log(ESS / K) ---
            ess = _effective_sample_size(list(weighted_particles.values()))
            K = len(weighted_particles)
            decomp["score_trans"] = math.log(max(ess, 1e-12) / max(K, 1))
            decomp["ess"] = ess
            decomp["belief_n_particles"] = int(K)
            # Shannon entropy of the normalised belief (nats). Used by
            # the obs-only pf_self_consistency scorer to reward
            # concentrated beliefs. Normalised later by log(K).
            w_vals = list(weighted_particles.values())
            total_w = float(sum(w_vals))
            if total_w > 0.0 and K > 1:
                ent = 0.0
                for w in w_vals:
                    p = float(w) / total_w
                    if p > 0.0:
                        ent -= p * math.log(p)
                decomp["belief_entropy"] = float(ent)
            else:
                decomp["belief_entropy"] = 0.0

            # --- hard_match_rate: P_q[distance == 0] ---
            # Obs-only. Fraction of belief mass on particles whose
            # predicted observation is an EXACT match of the observed
            # one (distance_soft == 0). Complementary to score_obs
            # which is a soft L2-like penalty; hard_match ignores
            # "near misses" and only rewards particles that match
            # exactly. Consumed by prediction_persistence scorer.
            if distances and weights_list:
                try:
                    # `probs` already a numpy array from L1437; reuse
                    # `distances_arr` allocated above for expected_dist.
                    hard_mask = (distances_arr <= 1e-9).astype(float)
                    hard_match = float(np.sum(probs * hard_mask))
                except Exception:
                    hard_match = 0.0
                decomp["hard_match_rate"] = hard_match
            else:
                decomp["hard_match_rate"] = 0.0

        return score_obs, distances if return_distances else None, decomp

    def _rejuvenate(
        self,
        action_history: List[ActType],
        observation_history: List[ObsType],
        seed_weighted: Optional[Dict[StateType, float]] = None,
    ) -> Dict[StateType, float]:
        """Sample trajectories from initial_model and weight by observation likelihood.

        Each candidate is sampled from the initial model, then propagated
        through the action/observation history.  The trajectory weight is
        the product of exp(-d_soft / κ) at each step.

        Args:
            action_history: actions taken so far.
            observation_history: observations received so far.
            seed_weighted: existing weighted particles to augment (optional).
                When None, starts from scratch (used at t=0).

        Returns:
            Weighted particle dict mapping state → cumulative weight.
        """
        weighted: Dict[StateType, float] = dict(seed_weighted) if seed_weighted else {}
        # Maintain a running weight sum to avoid O(n) recomputation per attempt
        running_sum = sum(weighted.values())
        num_attempts = 0
        # Let rejuvenation walk the full budget so hard environments can
        # still find weak-match particles before any learned model exists.

        while running_sum < self.num_particles and num_attempts < self.max_rejuvenation_attempts:
            num_attempts += 1
            if num_attempts % 10000 == 0:
                log.debug(f"Rejuvenation attempts: {num_attempts}/{self.max_rejuvenation_attempts}")

            try:
                candidate = self.initial_model(self.empty_state.copy())
            except Exception:
                # Capture first init_model traceback so that the REx loop
                # can surface the real error (e.g. ``NameError: random is
                # not defined``) to the LLM in the next refinement prompt.
                if getattr(self, "_last_init_exception_trace", None) is None:
                    self._last_init_exception_trace = traceback.format_exc()
                continue

            traj_weight = 1.0
            state = candidate

            for a, obs in zip(action_history, observation_history):
                try:
                    state = self.transition_model(state.copy(), a)
                    model_obs = self.observation_model(
                        state.copy(), a, self.empty_observation
                    )
                except Exception:
                    # Same idea for transition/observation_model during rejuv.
                    if getattr(self, "_last_init_exception_trace", None) is None:
                        self._last_init_exception_trace = traceback.format_exc()
                    traj_weight = 0.0
                    break

                dist = model_obs.distance_soft(obs)
                w = _soft_weight_from_distance(dist, self.kernel_bandwidth, self.obs_noise_epsilon)
                traj_weight *= w
                if traj_weight <= 0:
                    break

            if traj_weight > 0:
                weighted[state] = weighted.get(state, 0.0) + traj_weight
                running_sum += traj_weight

        # Mode-aware diversity preservation. Optional, opt-in via env var
        # REX_PF_MIN_MODES. POMDP-pure: reseed candidates are
        # filtered by the same trajectory weight (= obs likelihood) as
        # the standard rejuvenation, so we never inject hypotheses that
        # contradict observations the agent has actually seen.
        # Rationale: the standard rejuvenation re-samples from
        # initial_model but always renormalises to ``num_particles`` —
        # a dominant mode at sample time can soak up all the budget,
        # leaving minor modes (one of which may be the true mode)
        # unrepresented. We force a minimum spread on user-declared
        # protected latents (default minigrid: door/key/goal types).
        weighted = self._enforce_mode_diversity(
            weighted, action_history, observation_history,
        )

        return weighted

    def _enforce_mode_diversity(
        self,
        weighted: Dict[StateType, float],
        action_history: List[ActType],
        observation_history: List[ObsType],
    ) -> Dict[StateType, float]:
        """Reseed missing modes on protected latents via ``initial_model``,
        each candidate filtered by trajectory consistency (POMDP-pure).

        Activation: ``REX_PF_MIN_MODES=N`` (default 0 = no-op, byte-
        identical with previous behaviour). Knobs:
          REX_PF_PROTECTED_TYPES : comma-separated ObjectType ids to
              protect (default ``"6,7,10"`` = door/key/goal for minigrid).
              Falls back gracefully if the state lacks
              ``get_type_indices``.
          REX_PF_RESEED_ATTEMPTS : per-latent reseed budget (default 30).
        """
        import os as _div_os
        min_modes = int(_div_os.environ.get("REX_PF_MIN_MODES", "0"))
        if min_modes <= 0 or not weighted:
            return weighted

        protected_types_env = _div_os.environ.get(
            "REX_PF_PROTECTED_TYPES", "6,7,10"
        ).strip()
        try:
            protected_types = [
                int(x) for x in protected_types_env.split(",") if x.strip()
            ]
        except Exception:
            return weighted
        if not protected_types:
            return weighted

        # Probe a representative state for the API (env-agnostic graceful skip).
        rep_state = next(iter(weighted.keys()))
        if not hasattr(rep_state, "get_type_indices"):
            return weighted

        active_types: List[int] = []
        for t_id in protected_types:
            try:
                if rep_state.get_type_indices(t_id):
                    active_types.append(t_id)
            except Exception:
                continue
        if not active_types:
            return weighted

        max_reseed_attempts = int(
            _div_os.environ.get("REX_PF_RESEED_ATTEMPTS", "30")
        )

        for t_id in active_types:
            modes: Dict[Tuple[int, int], float] = {}
            for state, w in weighted.items():
                try:
                    positions = state.get_type_indices(t_id)
                    if positions:
                        pos = tuple(positions[0])
                        modes[pos] = modes.get(pos, 0.0) + float(w)
                except Exception:
                    continue
            if len(modes) >= min_modes:
                continue  # already diverse enough on this latent

            attempts = 0
            n_added = 0
            while len(modes) < min_modes and attempts < max_reseed_attempts:
                attempts += 1
                try:
                    candidate = self.initial_model(self.empty_state.copy())
                    cand_positions = candidate.get_type_indices(t_id)
                    if not cand_positions:
                        continue
                    cand_mode = tuple(cand_positions[0])
                    if cand_mode in modes:
                        continue  # already represented
                except Exception:
                    continue

                # Trajectory-weight filter (POMDP-pure consistency check)
                try:
                    traj_weight = 1.0
                    state = candidate
                    for a, obs in zip(action_history, observation_history):
                        state = self.transition_model(state.copy(), a)
                        model_obs = self.observation_model(
                            state.copy(), a, self.empty_observation,
                        )
                        dist = model_obs.distance_soft(obs)
                        w_step = _soft_weight_from_distance(
                            dist, self.kernel_bandwidth, self.obs_noise_epsilon,
                        )
                        traj_weight *= w_step
                        if traj_weight <= 0:
                            break
                except Exception:
                    continue

                if traj_weight > 0:
                    weighted[state] = weighted.get(state, 0.0) + traj_weight
                    modes[cand_mode] = traj_weight
                    n_added += 1
            if n_added > 0:
                log.debug(
                    f"[PF_DIVERSITY] type={t_id} modes={len(modes)}/{min_modes} "
                    f"reseed added n={n_added} after {attempts} attempts"
                )
        return weighted

    # Backward-compatible aliases
    def _rejuvenate_from_scratch(
        self, action_history: List[ActType], observation_history: List[ObsType],
    ) -> Dict[StateType, float]:
        return self._rejuvenate(action_history, observation_history)

    def _rejuvenate_step(
        self, current_weighted: Dict[StateType, float],
        action_history: List[ActType], observation_history: List[ObsType],
    ) -> Dict[StateType, float]:
        return self._rejuvenate(action_history, observation_history, seed_weighted=current_weighted)

    def _weighted_to_belief(self, weighted: Dict[StateType, float]) -> ParticleBelief:
        """Convert weighted particle dict to ParticleBelief with integer counts."""
        total_w = sum(weighted.values())
        if total_w <= 0:
            return ParticleBelief(particles={})

        int_particles: Dict[StateType, int] = Counter()
        for s, w in weighted.items():
            count = max(1, int(round(w / total_w * self.num_particles)))
            int_particles[s] += count

        return ParticleBelief(particles=dict(int_particles))


# Backward compatibility with code paths importing ``ELBOEvaluator``.
ELBOEvaluator = ScoreEvaluator
