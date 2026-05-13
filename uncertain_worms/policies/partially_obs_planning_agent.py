"""LLM-guided POMDP induction agent — implements Algorithm 1 of the paper.

The core class :class:`LLMPartiallyObsPlanningAgent` proposes executable
POMDP components (initial / observation / transition / reward) from an LLM,
scores them under a particle filter, and refines them through a REx
(Refinement EXploration) search tree until an episode-budgeted termination.

High-level pseudocode (paper §4, Algorithm 1)
---------------------------------------------
::

    1.  Load offline trajectories  D = {(o_0, a_0, r_1, ..., o_H)_n}
    2.  for j in 1..J:                                # REx rounds
    3.      parent  <- UCB1Select(pool)               # see rex_helpers.ucb1_select
    4.      for k in 1..M:                            # candidates per round
    5.          m_jk  <- LLM(parent prompt + diagnostics)
    6.          S_jk, B_jk, D_jk <- ParticleFilterKernelScore(m_jk, D)
    7.          C  <- C ∪ {T^{m_jk}}                  # disagreement committee
    8.          Q_jk <- QBCDisagreement(C, B_jk, D)   # vote entropy, Eq. 9
    9.          pool <- pool ∪ {m_jk}
    10. m*  <- softmax-sample from near-best set (Eq. 11–12)
    11. return m*

Mapping to code
---------------
- Step 1 (offline load)            ``LLMPartiallyObsPlanningAgent.__init__``
                                   reads ``dataset_path`` and replays
                                   trajectories.
- Step 2-4 (REx loop)              ``_run_rex`` / ``joint_update_models_rex``
- Step 5 (LLM proposal)            ``base_policy.requery_joint`` (joint mode,
                                   Pinductor) or ``base_policy.requery``
                                   (per-component, POMDP Coder baseline)
- Step 6 (Particle filter score)   :func:`particle_filtering.get_score_metrics
                                   .ELBOEvaluator.evaluate_elbo` (Eq. 7-8)
- Step 8 (QBC disagreement)        :func:`particle_filtering.model_disagreement
                                   .committee_prediction_entropy` (Eq. 9)
                                   + ``DisagreementDetector`` for per-context
                                   summaries fed back into the LLM prompt.
- Step 10 (final selection)        Inline near-best softmax inside
                                   ``joint_update_models_rex`` (search for
                                   ``elbo_softmax_temperature`` / ``near_best``).
                                   Implements Eq. 11-12.

Online deployment loop
----------------------
Once a model :math:`m^*` is selected, :meth:`update_belief` runs a particle
filter forward at each step, and :meth:`get_action` queries the planner
(``PartiallyObservablePlanner``) over the current particle belief. After
each episode, the new trajectory is appended to the replay buffer and a
fresh REx round (Step 2) is triggered for online refinement.

Hyperparameters (paper Table 1 / App. D)
----------------------------------------
- ``kernel_bandwidth`` (κ, default 0.2) — distance-kernel sharpness
- ``num_particles`` (K, default 10)     — belief size
- ``num_model_attempts`` (M, default 5) — candidates per REx round
- ``ucb1_c`` (c, default 1.0)           — UCB1 exploration constant
- ``elbo_softmax_temperature`` (T)      — final selection temperature
- ``num_input_examples`` (N_D)          — number of offline trajectories
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pyvis.network import Network  # type:ignore

from particle_filtering.get_score_metrics import ELBOEvaluator
from uncertain_worms.planners.base_planner import PartiallyObservablePlanner
from uncertain_worms.policies.base_policy import (
    PROMPT_DIR,
    Policy,
    initial_model_translator,
    observation_model_translator,
    requery,
    requery_joint,
    reward_model_translator,
    transition_model_translator,
)
# Pure helpers live in llm_feedback to keep this module focused on
# orchestration. We re-export under the historical names so any in-file
# reference (and any external import) keeps working.
from uncertain_worms.policies.llm_feedback import (
    ELBO_FLOOR as _ELBO_FLOOR,
    elbo_components_from_episode_metrics as _elbo_components_from_episode_metrics,
    elbo_from_episode_metrics as _elbo_from_episode_metrics,
    jensen_shannon_divergence as jsd,
)
from uncertain_worms.policies.rex_helpers import (
    normalize_to_nll,
    rotated_strat_seed,
    ucb1_select,
)
from uncertain_worms.policies.sfs_directions import (
    DirectionHistory,
    build_scattering_prompt,
    format_direction_prompt,
    get_first_iter_directions,
    needs_llm_scattering,
    parse_directions,
)
from uncertain_worms.structs import (
    ActType,
    InitialModel,
    ObsType,
    ParticleBelief,
    ReplayBuffer,
    StateType,
)
from uncertain_worms.utils import PROJECT_ROOT, get_log_dir

log = logging.getLogger(__name__)


# Define type variables for conditions and outcomes.
#
# Note on ``*Outcome = Tuple[X]`` (one-element tuple):
# these aliases intentionally use a fixed-arity tuple — the outcome
# of e.g. a transition is always "a single next state" — so type
# checkers can distinguish them from the reward outcome which carries
# ``(reward, terminated)``. The audit (CQ2) flagged this as lossy but
# callers rely on the one-vs-two-tuple shape to route into
# :meth:`LLMPartiallyObsPlanningAgent.print_io`; widening it to
# ``Tuple[StateType, ...]`` would silently change that routing.

RewardCondition = Tuple[StateType, ActType, StateType]
RewardOutcome = Tuple[float, bool]
TransitionCondition = Tuple[StateType, ActType]
TransitionOutcome = Tuple[StateType]
ObservationCondition = Tuple[StateType]
ObservationOutcome = Tuple[ObsType]
InitialModelCondition = Tuple
InitialModelOutcome = Tuple[StateType]

Condition = Union[
    RewardCondition,
    TransitionCondition,
    ObservationCondition,
    InitialModelCondition,
]
Outcome = Union[
    RewardOutcome, TransitionOutcome, ObservationOutcome, InitialModelOutcome
]


def distribution_to_hashable(dist: Dict[StateType, float]) -> frozenset:
    """Convert a state distribution to a hashable frozenset of (state, prob) tuples.
    
    Args:
        dist: Dictionary mapping states to probabilities
        
    Returns:
        A frozenset of (state, prob) tuples that can be used as dictionary keys
    """
    return frozenset((state, prob) for state, prob in dist.items())


def hashable_to_distribution(hashable: frozenset) -> Dict[StateType, float]:
    """Convert a hashable frozenset back to a distribution dictionary.
    
    Args:
        hashable: A frozenset of (state, prob) tuples
        
    Returns:
        Dictionary mapping states to probabilities
    """
    return dict(hashable)


def normalize_outcome_list(outcome_list: List[Outcome]) -> Dict[Outcome, float]:
    """Converts a list of outcomes into a normalized frequency distribution."""
    counts = Counter(outcome_list)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _episode_to_io(
    obs_list: List[ObsType],
    action_list: Optional[List[ActType]] = None,
    initial_obs: Optional[ObsType] = None,
    reward_list: Optional[List[float]] = None,
    terminated_list: Optional[List[bool]] = None,
) -> List[Tuple[Condition, Outcome]]:
    """Convert one episode's rollout to the ``(condition, outcome)`` list
    consumed by :meth:`LLMPartiallyObsPlanningAgent.print_io`.

    Each step ``t`` becomes ``((action[t],), (obs[t], "Reward: r, Terminated: d"))``
    when ``reward_list`` and ``terminated_list`` are supplied, or
    ``((action[t],), (obs[t],))`` otherwise. Steps past the length of
    ``action_list`` (or when ``action_list is None``) fall back to
    ``((), (obs,))``, preserving the semantics of the two inlined
    helpers that previously lived inside
    :meth:`get_joint_starting_prompt` and
    :meth:`get_joint_feedback_prompt`. The optional ``initial_obs`` is
    prepended as a leading ``((), (initial_obs,))`` entry so the LLM
    prompt can show the state before the first action.

    The helper has been extracted verbatim from the two nested
    closures it replaces; the per-step tuple construction and the
    iteration order are unchanged, so the generated prompt string is
    byte-identical.
    """
    base: List[Tuple[Condition, Outcome]] = []
    for t, obs in enumerate(obs_list):
        if action_list is not None and t < len(action_list):
            if (
                reward_list is not None
                and t < len(reward_list)
                and terminated_list is not None
                and t < len(terminated_list)
            ):
                outcome = (obs, f"Reward: {reward_list[t]}, Terminated: {terminated_list[t]}")
            else:
                outcome = (obs,)
            base.append(((action_list[t],), outcome))
        else:
            base.append(((), (obs,)))
    if initial_obs is not None:
        return [((), (initial_obs,))] + base
    return base


def conditional_jsd(
    p_dict: Dict[Condition, List[Outcome]], q_dict: Dict[Condition, List[Outcome]]
) -> float:
    """Computes the conditional Jensen-Shannon divergence, taking expectation
    over P(X).

    Args:
        p_dict: Dictionary mapping conditions (X) to lists of observed outcomes.
        q_dict: Dictionary mapping conditions (X) to lists of predicted outcomes.

    Returns:
        The expectation of JSD under P(X), i.e.,
        E_{X ~ P(X)}[D_JS(P(Y|X) || Q(Y|X))].
    """
    js_divs: List[float] = []
    # Compute P(X) from the empirical counts (each condition's weight)
    total_count = sum(len(outcomes) for outcomes in p_dict.values())
    p_x_distribution: Dict[Condition, float] = {
        condition: len(outcomes) / total_count for condition, outcomes in p_dict.items()
    }

    for condition, p_x in p_x_distribution.items():
        if condition in q_dict:
            # Convert outcome lists into normalized frequency distributions
            p_dist = normalize_outcome_list(p_dict[condition])
            q_dist = normalize_outcome_list(q_dict[condition])

            # Compute JSD for this condition
            js_div = jsd(p_dist, q_dist, base=2)

            # Weight the JSD by P(X)
            js_divs.append(p_x * js_div)

    return float(np.sum(js_divs)) if js_divs else 0.0


class PartiallyObsPlanningAgent(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        planner: PartiallyObservablePlanner[
            StateType, ActType, ObsType, ParticleBelief
        ],
        empty_state: StateType,
        empty_observation: ObsType,
        *args: Any,
        ground_truth_initial_model: Optional[InitialModel] = None,
        num_initial_model_samples: int = 20000,
        num_particles: int = 1000,
        resampling: bool = True,
        max_attempts: int = 1000000,
        num_bug_fix_attempts: int = 5,
        log_belief_diagnostics: bool = False,
        root_sampling_k: int = 0,
        kernel_bandwidth: float = 1.0,
        adaptive_bandwidth: bool = False,
        adaptive_bw_tight: float = 0.05,
        adaptive_bw_loose: float = 1.0,
        decompose: bool = False,
        msc_horizon: int = 0,
        msc_weight: float = 0.0,
        scoring_mode: str = "energy",
        obs_noise_epsilon: float = 0.0,
        output_particle_distance_metrics: bool = False,
        model_averaging_top_k: int = 1,
        max_rejuvenation_attempts: int = 1000,
        qbc_max_particles_per_step: int = 10,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.empty_state = empty_state
        self.empty_observation = empty_observation
        # Explicit storage so ``_make_elbo_evaluator`` can propagate the
        # value instead of hard-coding (the former ``num_particles=100``
        # / ``kernel_bandwidth=1.0`` hardcodes caused the "10 at plan,
        # 100 at score" divergence that polluted early runs). The QBC
        # limit was likewise hardcoded at 5 — the class-level default
        # is 10, restored here.
        self.max_rejuvenation_attempts = int(max_rejuvenation_attempts)
        self.qbc_max_particles_per_step = int(qbc_max_particles_per_step)
        # Build an env-agnostic state sampler from the empty_state.
        # Works for any env that provides a valid empty_state.
        self._state_sampler = self._make_state_sampler(empty_state)
        self.num_particles = num_particles
        self.num_initial_model_samples = num_initial_model_samples
        self.ground_truth_initial_model = ground_truth_initial_model
        self.planner = planner
        self.action_hist: List[ActType] = []
        self.observation_hist: List[ObsType] = []
        self.steps_taken = 0
        self.resampling = resampling
        self.num_bug_fix_attempts = num_bug_fix_attempts
        self.max_attempts = max_attempts
        self.num_attempts = 0
        self.current_belief: Optional[ParticleBelief] = None
        self.log_belief_diagnostics = log_belief_diagnostics
        self.root_sampling_k = root_sampling_k
        self.kernel_bandwidth = kernel_bandwidth
        self.adaptive_bandwidth = adaptive_bandwidth
        self.adaptive_bw_tight = adaptive_bw_tight
        self.adaptive_bw_loose = adaptive_bw_loose
        self.decompose = decompose
        self.msc_horizon = msc_horizon
        self.msc_weight = msc_weight
        self.scoring_mode = scoring_mode
        self.obs_noise_epsilon = obs_noise_epsilon
        self.output_particle_distance_metrics = output_particle_distance_metrics
        self.model_averaging_top_k = model_averaging_top_k

    @staticmethod
    def _make_state_sampler(empty_state):
        """Build a state sampler appropriate for the environment.

        Dispatches based on the type of *empty_state* so that the rest of
        the agent code is env-agnostic.
        """
        cls_name = type(empty_state).__name__
        if "Minigrid" in cls_name:
            from uncertain_worms.environments.minigrid.sample_state import (
                make_minigrid_state_sampler,
            )
            return make_minigrid_state_sampler(empty_state)
        else:
            raise NotImplementedError(
                f"No state sampler for {cls_name}. "
                f"Implement one in the environment's package."
            )

    def _effective_bandwidth(self, obs: Optional[ObsType] = None) -> float:
        """Placeholder — adaptive bandwidth is computed inside update_belief."""
        if not self.adaptive_bandwidth:
            return self.kernel_bandwidth
        # When adaptive, we pass bw_loose to update_belief; it will tighten
        # internally after computing particle distances.
        return self.adaptive_bw_loose

    def reset(self) -> None:
        self.observation_hist = []
        self.action_hist = []
        self.steps_taken = 0
        self.num_attempts = 0
        self.current_belief = None
        # Log active transition code for reproducibility diagnostics.
        # Identifies which LLM-generated code is loaded at episode start.
        if hasattr(self, "current_code"):
            trans_code = self.current_code.get("transition_model", "")
            code_hash = hash(trans_code) if trans_code else 0
            log.info("[DIAG] reset: transition_code_hash=%d code_len=%d", code_hash, len(trans_code))

    # Legacy MiniGrid action vocabulary kept for backwards-compatibility
    # of the opt-in [BELIEF_DIAG] log lines. New environments should
    # expose ``action.name`` (IntEnum-style) so ``_resolve_action_name``
    # can pick it up dynamically without touching this mapping.
    _LEGACY_MINIGRID_ACTION_NAMES: Dict[int, str] = {
        0: "LEFT", 1: "RIGHT", 2: "FWD", 3: "PICKUP", 4: "DROP", 5: "TOGGLE",
    }

    def _resolve_action_name(self, action: ActType) -> str:
        """Return a human-readable name for ``action``.

        Resolution order:
        1. ``action.name`` when the action type exposes it (IntEnum,
           namedtuple, ...). This is the preferred env-agnostic path.
        2. The legacy MiniGrid lookup
           :data:`_LEGACY_MINIGRID_ACTION_NAMES` for int actions, so
           the opt-in ``[BELIEF_DIAG]`` log lines keep their historical
           ``LEFT/RIGHT/FWD/PICKUP/DROP/TOGGLE`` display names when
           running the current MiniGrid configs.
        3. ``str(action)``.

        The function is only consulted from the human-readable
        diagnostic log; it does not affect any computation.
        """
        name = getattr(action, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(action, int) and action in self._LEGACY_MINIGRID_ACTION_NAMES:
            return self._LEGACY_MINIGRID_ACTION_NAMES[action]
        return str(action)

    def _log_full_belief_diagnostic(self, belief: ParticleBelief) -> None:
        """Exhaustive belief diagnostic — logs position distribution, direction
        distribution, per-action expected reward with decomposition explaining
        WHY each action gets its reward.  Enabled by ``log_belief_diagnostics=True``."""
        particles = belief.particles  # Dict[State, int]
        total = sum(particles.values())
        if total == 0:
            log.info("[BELIEF_DIAG] step=%d EMPTY BELIEF", self.steps_taken)
            return

        self._log_position_and_direction_distributions(particles, total)

        # --- Per-action expected reward with decomposition ---
        cat = {s: c / total for s, c in particles.items()}

        for a in self.actions:
            er = 0.0
            contributing_states: List[str] = []  # states where reward > 0
            next_positions: Dict[tuple, float] = {}  # where particles end up

            for s, p in cat.items():
                try:
                    ns = self.planner.transition_model(s.copy(), a)
                    r, term = self.planner.reward_model(s, a, ns)
                    er += r * p
                    if r > 0:
                        s_pos = tuple(s.agent_pos) if hasattr(s, "agent_pos") else "?"
                        ns_pos = tuple(ns.agent_pos) if hasattr(ns, "agent_pos") else "?"
                        contributing_states.append(
                            f"from={s_pos}→to={ns_pos} r={r:.1f} p={p:.3f} contrib={r*p:.4f}"
                        )
                    if hasattr(ns, "agent_pos"):
                        np_key = tuple(ns.agent_pos)
                        next_positions[np_key] = next_positions.get(np_key, 0.0) + p
                except Exception:
                    pass

            # Top 5 next positions
            next_pos_sorted = sorted(next_positions.items(), key=lambda x: -x[1])[:5]
            next_str = " ".join(f"{p}:{prob:.3f}" for p, prob in next_pos_sorted)

            log.info(
                "[BELIEF_DIAG]   action=%s E[r]=%.4f | next_pos_top5: %s",
                self._resolve_action_name(a), er, next_str,
            )
            if contributing_states:
                for cs in contributing_states[:3]:
                    log.info("[BELIEF_DIAG]     reward_source: %s", cs)

        # --- Observation info ---
        if self.observation_hist:
            last_obs = self.observation_hist[-1]
            img = getattr(last_obs, "image", None)
            if img is not None:
                has_wall = bool(np.any(np.asarray(img) == 2))
                has_goal = bool(np.any(np.asarray(img) == 10))
                log.info(
                    "[BELIEF_DIAG]   last_obs: wall=%s goal=%s",
                    has_wall, has_goal,
                )

    def _log_position_and_direction_distributions(
        self,
        particles: Dict[Any, int],
        total: int,
    ) -> None:
        """Emit the position / direction / entropy / goal-positions log
        block shared by ``_log_full_belief_diagnostic``.

        Pure side-effect helper: it owns three INFO log lines and
        depends on ``self.steps_taken``. Factoring it out keeps the
        bigger diagnostic method focused on the per-action expected
        reward decomposition.
        """
        pos_counts: Dict[tuple, int] = {}
        dir_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        goal_positions_in_belief: Set[Any] = set()
        for state, count in particles.items():
            if hasattr(state, "agent_pos"):
                p = tuple(state.agent_pos)
                pos_counts[p] = pos_counts.get(p, 0) + count
            if hasattr(state, "agent_dir"):
                dir_counts[state.agent_dir] = dir_counts.get(state.agent_dir, 0) + count
            if hasattr(state, "get_type_indices"):
                for gp in state.get_type_indices(10):  # goal=10
                    goal_positions_in_belief.add(gp)

        pos_probs = sorted(
            [(p, c / total) for p, c in pos_counts.items()],
            key=lambda x: -x[1],
        )
        dir_probs = {d: c / total for d, c in dir_counts.items()}
        entropy = -sum(
            (c / total) * math.log(c / total)
            for c in particles.values() if c > 0
        )

        log.info(
            "[BELIEF_DIAG] step=%d | %d unique states, %d unique positions, "
            "entropy=%.3f nats",
            self.steps_taken, len(particles), len(pos_counts), entropy,
        )
        log.info(
            "[BELIEF_DIAG]   Position distribution (top 10): %s",
            "  ".join(f"{p}:{prob:.3f}" for p, prob in pos_probs[:10]),
        )
        dir_str = " ".join(f"d{d}={prob:.2f}" for d, prob in sorted(dir_probs.items()))
        log.info("[BELIEF_DIAG]   Direction distribution: %s", dir_str)
        if goal_positions_in_belief:
            log.info(
                "[BELIEF_DIAG]   Goal positions in belief particles: %s",
                goal_positions_in_belief,
            )

    def update_belief(
        self,
        current_belief: ParticleBelief,
        observation_history: List[ObsType],
        action_history: List[ActType],
        *,
        distance_threshold: float | None = None,
        kernel_bandwidth: float = 1.0,
    ) -> Tuple[Optional[ParticleBelief], Dict[str, str]]:
        """Soft-particle belief update using object-defined distances.

        • dist == ∞  ➜ reject particle  (old exact-match behaviour) 
        • distance_threshold given ➜ keep particles with dist ≤ threshold
        (equal weight) 
        • else ➜ weight = exp(-dist / kernel_bandwidth)
        """
        assert len(observation_history) == len(action_history)

        eps = self.obs_noise_epsilon  # E4 likelihood tempering
        weighted_particles: Dict[StateType, float] = {}

        # ------------------------------------------------------------------ #
        # 1) Propagate existing particles and weight them                    #
        # ------------------------------------------------------------------ #
        if current_belief:
            a_last = action_history[-1]
            o_last = observation_history[-1]

            # First pass: compute distances (reused for adaptive bandwidth).
            # P3: a single failing particle no longer aborts the whole update —
            # we skip it, record the exception for diagnostics, and keep going
            # with the rest of the belief.
            propagated: list = []  # [(next_state, dist)]
            n_particle_errors = 0
            last_particle_error: Optional[str] = None
            for particle in current_belief.to_list():
                try:
                    next_state = self.planner.transition_model(
                        particle.copy(), a_last
                    )
                    model_obs = self.planner.observation_model(
                        next_state.copy(), a_last, self.empty_observation
                    )
                except Exception:
                    n_particle_errors += 1
                    last_particle_error = traceback.format_exc()
                    continue

                dist = model_obs.distance_soft(o_last)
                if not math.isinf(dist):
                    propagated.append((next_state, dist))

            if n_particle_errors > 0:
                log.info(
                    "[P3] %d/%d particles raised during propagation — skipped",
                    n_particle_errors, len(current_belief.to_list()),
                )
                # If every single particle failed, we cannot recover — return
                # an error so the caller falls back to rejuvenation/random.
                if not propagated and last_particle_error is not None:
                    return None, {"model": last_particle_error}

            # Adaptive bandwidth: tighten when distances vary across particles
            if self.adaptive_bandwidth and propagated:
                dists = [d for _, d in propagated]
                spread = max(dists) - min(dists)
                # If particles disagree on how well they match the obs,
                # the obs is discriminative → use tight bandwidth.
                if spread > 1e-6:
                    kernel_bandwidth = self.adaptive_bw_tight
                else:
                    kernel_bandwidth = self.adaptive_bw_loose
                log.info(f"ADAPTIVE_BW: bw={kernel_bandwidth} "
                         f"spread={spread:.4f} n={len(dists)}")

            # Second pass: assign weights using effective bandwidth
            # E4 likelihood tempering: w = (1-ε)·exp(-d/κ) + ε
            for next_state, dist in propagated:
                if distance_threshold is not None:
                    if dist <= distance_threshold:
                        weighted_particles[next_state] = (
                            weighted_particles.get(next_state, 0) + 1.0
                        )
                else:
                    base_w = math.exp(-dist / max(kernel_bandwidth, 1e-12))
                    w = (1.0 - eps) * base_w + eps if eps > 0.0 else base_w
                    if w > 0.0:
                        weighted_particles[next_state] = (
                            weighted_particles.get(next_state, 0) + w
                        )

        log.info(
            "Num valid initial particles pre-rejuvenation: "
            f"{len(weighted_particles)}"
        )

        # ------------------------------------------------------------------ #
        # 2) Rejuvenate / initialise until we hit self.num_particles         #
        # ------------------------------------------------------------------ #
        running_weight_sum = sum(weighted_particles.values())
        while (
            running_weight_sum < self.num_particles
            and self.num_attempts < self.max_attempts
        ):
            self.num_attempts += 1
            if self.num_attempts % 1000 == 0:
                log.debug(f"Rejuvenation attempt {self.num_attempts}/{self.max_attempts}")
            try:
                candidate = self.planner.initial_model(self.empty_state.copy())
            except Exception:
                return None, {"initial_model": traceback.format_exc()}

            ok = True
            next_state = candidate
            total_dist = 0.0
            for a, obs in zip(action_history, observation_history):
                try:
                    next_state = self.planner.transition_model(
                        next_state, a
                    )
                    cand_obs = self.planner.observation_model(
                        next_state, a, self.empty_observation
                    )
                except Exception:
                    # P3: a single rejuvenation candidate failing on the
                    # learned models is just "that particle is infeasible" —
                    # reject it and draw a new one instead of aborting the
                    # whole belief update.
                    ok = False
                    break

                dist = cand_obs.distance_soft(obs)
                if math.isinf(dist) or (
                    distance_threshold is not None and dist > distance_threshold
                ):
                    ok = False
                    break
                total_dist += dist

            if ok:
                base_w = math.exp(-total_dist / max(kernel_bandwidth, 1e-12))
                w = (1.0 - eps) * base_w + eps if eps > 0.0 else base_w
                if w > 0.0:
                    weighted_particles[next_state] = (
                        weighted_particles.get(next_state, 0) + w
                    )
                    running_weight_sum += w

        log.info(f"Num attempts: {self.num_attempts}/{self.max_attempts}")
        log.info("Num valid initial particles: " + str(len(weighted_particles)))

        if not weighted_particles:
            return None, {
                "observation_model": "No particles consistent with observation"
            }

        # ------------------------------------------------------------------ #
        # 3) Resample to exactly self.num_particles (proper weighted sampling) #
        # ------------------------------------------------------------------ #
        total_w = sum(weighted_particles.values())
        if total_w == 0.0:
            return None, {"observation_model": "All particle weights zero"}

        states = list(weighted_particles.keys())
        weights = [weighted_particles[s] for s in states]
        sampled = random.choices(states, weights=weights, k=self.num_particles)
        int_particles: Dict[StateType, int] = Counter(sampled)

        log.info("Num unique initial particles: " + str(len(int_particles)))

        new_belief = ParticleBelief[StateType, ObsType](particles=int_particles)
        return new_belief, {}

    def get_next_action(self, obs: Optional[ObsType]) -> ActType:
        if obs is not None:
            self.observation_hist.append(obs)

        # Diagnostic: log RNG state hash and active transition model at step 0
        # to detect belief divergence caused by stochastic LLM code generation.
        if self.steps_taken == 0:
            rng_hash = hash(np.random.get_state()[1].tobytes())
            trans_id = id(self.planner.transition_model)
            log.info(
                "[DIAG] step=0 np.random_state_hash=%d transition_model_id=%d",
                rng_hash, trans_id,
            )

        eff_bw = self._effective_bandwidth(obs)

        # If no belief has been established yet, initialize it
        if self.current_belief is None:
            log.info("Initializing belief")
            belief, error = self.update_belief(
                ParticleBelief(), observation_history=[], action_history=[],
                kernel_bandwidth=eff_bw,
            )
            if belief is None:
                log.info("Belief initialization error: " + str(error))
                action = random.choice(self.actions)
                self.action_hist.append(action)
                return action
            self.current_belief = belief
        else:
            log.info("Updating belief")
            # Otherwise update the belief using the last action and the new observation
            belief, error = self.update_belief(
                self.current_belief,
                observation_history=self.observation_hist,
                action_history=self.action_hist,
                kernel_bandwidth=eff_bw,
            )
            if belief is None:
                log.info("Belief update error: " + str(error))
                action = random.choice(self.actions)
                self.action_hist.append(action)
                return action
            self.current_belief = belief

        # --- Belief diagnostic logging (opt-in via config) ---
        if getattr(self, "log_belief_diagnostics", False) and belief is not None:
            self._log_full_belief_diagnostic(belief)

        steps_left = self.max_steps - self.steps_taken

        # Pre-compute exploration bonus per action, combining:
        #   1) BALD MI: entropy of committee's state predictions (epistemic)
        #   2) Reward variance: disagreement on predicted rewards (DreamerV3-XP)
        # Both are cheap: 5 particles × N_models × N_actions, done ONCE.
        # Passed to the planner as a constant bonus — zero per-node cost.
        mi_bonus: Optional[Dict[int, float]] = None
        qbc_candidates = getattr(self, "_qbc_candidates", [])
        if len(qbc_candidates) >= 2 and belief is not None:
            try:
                from particle_filtering.model_disagreement import (
                    mutual_information_committee,
                    reward_variance_committee,
                )
                mi_bonus = {}
                for a in self.actions:
                    mi = mutual_information_committee(
                        qbc_candidates, belief, a
                    )
                    rv = reward_variance_committee(
                        qbc_candidates, self.planner.reward_model,
                        belief, a,
                    )
                    # Combine: MI captures state disagreement,
                    # reward variance captures reward-relevant disagreement.
                    # Both in [0, ∞), additive.
                    mi_bonus[a] = mi + rv
                has_signal = max(mi_bonus.values()) > 0
                if has_signal:
                    log.info(
                        "[BALD+RV] exploration bonus per action: %s",
                        {a: f"{v:.3f}" for a, v in mi_bonus.items() if v > 0},
                    )
                else:
                    mi_bonus = None
            except Exception:
                mi_bonus = None

        if self.root_sampling_k > 0 and belief is not None and belief.particles:
            # Root sampling (Silver & Veness 2010, POMCP):
            votes: Counter = Counter()
            for _ in range(self.root_sampling_k):
                sampled_state = belief.sample()
                delta_belief = ParticleBelief(particles={sampled_state: 1})
                try:
                    a, _ = self.planner.plan_next_action(
                        delta_belief, max_steps=steps_left, mi_bonus=mi_bonus,
                    )
                    votes[a] += 1
                except Exception:
                    pass
            if votes:
                action = votes.most_common(1)[0][0]
                log.info(
                    f"[ROOT_SAMPLING] K={self.root_sampling_k} votes={dict(votes)} "
                    f"selected={action}"
                )
            else:
                action, _ = self.planner.plan_next_action(
                    belief, max_steps=steps_left, mi_bonus=mi_bonus,
                )
        else:
            action, error = self.planner.plan_next_action(
                belief, max_steps=steps_left, mi_bonus=mi_bonus,
            )
            if len(error) != 0:
                log.info("Planning error, taking random action")
                action = random.choice(self.actions)

        self.action_hist.append(action)
        self.steps_taken += 1
        return action


def save_rex_tree_html(
    model_name: str,
    generated_nodes: List[RexNode],
    parent_mapping: Dict[RexNode, RexNode],
    log_dir: str,
    iteration: int,
) -> None:
    """Saves the Rex tree visualization as an HTML file in the specified log
    directory.

    Args:
        generated_nodes: A list of generated RexNode objects.
        parent_mapping: Mapping from a child RexNode to its parent's RexNode.
        log_dir: Directory where HTML files will be saved.
        iteration: Current iteration number for unique file naming.
    """
    # Create a directed network graph.
    net = Network(height="750px", width="100%", directed=True)

    # Add nodes for each RexNode.
    for node in generated_nodes:
        label = (
            f"Iter: {node.iter_num}\n"
            f"Depth: {node.depth}\n"
            f"Train/Test: {node.train_coverage:.2f}, {node.test_coverage:.2f}\n"
            f"Alpha,Beta: ({node.alpha:.2f}, {node.beta:.2f})\n"
        )

        # Compute total coverage as the average of train and test coverage.
        total_coverage = (node.train_coverage + node.test_coverage) / 2.0

        # Create a red-to-green gradient:
        # When total_coverage is 0, color is red (#FF0000);
        # when it is 1, color is green (#00FF00).
        red = int((1 - total_coverage) * 255)
        green = int(total_coverage * 255)
        blue = 0
        color = f"#{red:02x}{green:02x}{blue:02x}"

        net.add_node(str(node.iter_num), label=label, color=color)

    # Add edges from parent to child.
    for child, parent in parent_mapping.items():
        net.add_edge(str(parent.iter_num), str(child.iter_num))

    # Construct the file name and save the HTML file in log_dir.
    # filename = os.path.join(log_dir, f"rex_tree_iter_{model_name}_{iteration}.html")
    filename = os.path.join(log_dir, f"rex_tree_iter_{model_name}.html")

    net.write_html(filename)
    log.info(f"Rex tree for iteration {iteration} saved to {filename}")


@dataclass
class RexNode:
    """One node in the REx search tree.

    Each node remembers the score of a candidate ``(observation_model,
    transition_model, reward_model)`` triple evaluated on a held-out
    train / test split. Fields ``train_coverage`` and ``test_coverage``
    are fossils from the old single-model "coverage" scoring path.
    They now hold ELBO aggregates (i.e.
    the mean clamped pf_score of the train / test replay buffers).

    The new-style names :attr:`train_score` / :attr:`test_score` are
    read-only ``@property`` aliases returning exactly the same numbers.
    Introduced here so new code can stop referring to the misleading
    "coverage" fossil; the fossil field names stay in place because
    ``plots/plot_all_overnight.py`` parses ``test_coverage=`` strings
    out of the log files via a regex, so renaming the fields in
    ``__repr__`` would silently break the benchmark plots. A follow-up
    will update both sides together.
    """

    to_update: bool
    train_coverage: float
    test_coverage: float
    train_empirical_dist: Dict[Condition, List[Outcome]]
    train_model_dist: Dict[Condition, List[Outcome]]
    test_empirical_dist: Dict[Condition, List[Outcome]]
    test_model_dist: Dict[Condition, List[Outcome]]
    alpha: float
    beta: float
    depth: int
    iter_num: int
    messages: List[Dict[str, str]] = field(default_factory=list)
    previous_code: Optional[str] = None
    train_actions: Optional[List[List[ActType]]] = None
    train_previous_observations: Optional[List[List[ObsType]]] = None
    train_rewards: Optional[List[List[float]]] = None
    train_terminated: Optional[List[List[bool]]] = None
    train_episode_metrics: Optional[List[Dict[str, Any]]] = None

    # -- Fossil-renaming compat: ``train_score`` / ``test_score`` are the
    # new names. They are strict aliases for ``train_coverage`` /
    # ``test_coverage`` — same values, same type, same dtype. Intended
    # as the stable public API during the deprecation window.
    @property
    def train_score(self) -> float:
        """Canonical name for :attr:`train_coverage` (unchanged value)."""
        return self.train_coverage

    @property
    def test_score(self) -> float:
        """Canonical name for :attr:`test_coverage` (unchanged value)."""
        return self.test_coverage

    def __repr__(self) -> str:
        return (
            f"RexNode(iter={self.iter_num}, depth={self.depth},\n"
            f"        train_coverage={self.train_coverage:.4f}, test_coverage={self.test_coverage:.4f},\n"
            f"        alpha={self.alpha:.4f}, beta={self.beta:.4f}, to_update={self.to_update})"
        )

    def __hash__(self) -> int:
        return hash(self.iter_num)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RexNode):
            return False
        return self.iter_num == other.iter_num


class LLMPartiallyObsPlanningAgent(
    PartiallyObsPlanningAgent[StateType, ActType, ObsType]
):
    JOINT_MODEL_ORDER: Tuple[str, ...] = (
        "initial_model",
        "observation_model",
        "transition_model",
        "reward_model",
        "initial_model",
    )
    JOINT_MODEL_TO_FUNC: Dict[str, str] = {
        "initial_model": "initial_func",
        "observation_model": "observation_func",
        "transition_model": "transition_func",
        "reward_model": "reward_func",
        "initial_model": "initial_func",
    }

    @staticmethod
    def _resolve_joint_model_inserts(
        inserts: Dict[str, Dict[str, Any]], env_name: str
    ) -> Dict[str, Any]:
        """Select joint prompt inserts: tiger env -> tiger prompt, else minigrid prompt."""
        env_name_lower = env_name.lower()
        if "tiger" in env_name_lower and "joint_model_tiger" in inserts:
            return inserts["joint_model_tiger"]

        # Default to minigrid for every non-tiger environment.
        if "joint_model_minigrid" in inserts:
            return inserts["joint_model_minigrid"]
        if "joint_model_tiger" in inserts:
            return inserts["joint_model_tiger"]
        return {}

    @staticmethod
    def _human_join(parts: List[str]) -> str:
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _joint_prompt_inserts_for(
        self, active_joint_models: List[str]
    ) -> Dict[str, Any]:
        inserts = dict(self.joint_model_inserts)
        targets: List[str] = []
        if "initial_model" in active_joint_models:
            targets.append("initial states")
        if "observation_model" in active_joint_models:
            targets.append("observations")
        if "transition_model" in active_joint_models:
            targets.append("transitions")
        if "reward_model" in active_joint_models:
            targets.append("rewards")
        if targets:
            inserts["what_to_model"] = (
                f"the distribution of {self._human_join(targets)}"
            )
        if "initial_model" in active_joint_models:
            samples_info = str(inserts.get("samples_info", "")).rstrip()
            initial_note = (
                "You do not observe true initial states directly. Infer the "
                "initial-state distribution from the environment description and "
                "from the first observation of each episode. `initial_func` should "
                "sample a plausible hidden state consistent with those clues."
            )
            inserts["samples_info"] = (
                f"{samples_info}\n\n{initial_note}" if samples_info else initial_note
            )
        return inserts

    def _seed_joint_code(self, active_joint_models: List[str]) -> Optional[str]:
        for model_name in active_joint_models:
            code = self.current_code.get(model_name, "") or ""
            if code:
                return code
        return None

    def _sync_evaluator_models(
        self,
        elbo_evaluator: "ELBOEvaluator",
        active_joint_models: List[str],
    ) -> None:
        for model_name in active_joint_models:
            setattr(elbo_evaluator, model_name, self._get_model(model_name))

    def __init__(
        self,
        *args: Any,
        env_code_path: str = "",
        env_name: str = "",
        env_description: str = "",
        goal_description: str = "",
        num_model_attempts: int = 25,
        num_online_model_attempts: int = 25,
        num_input_examples: int = 5,
        learn_transition: bool = False,
        learn_reward: bool = False,
        learn_observation: bool = False,
        learn_initial: bool = False,
        use_offline: bool = True,
        use_online: bool = False,
        dataset_path: str = "",
        num_state_samples: int = 100,
        output_particle_distance_metrics: bool = False,
        decompose: bool = False,
        msc_horizon: int = 0,
        msc_weight: float = 0.0,
        scoring_mode: str = "energy",
        obs_noise_epsilon: float = 0.0,
        model_averaging_top_k: int = 1,
        qbc_explore_prob: float = 0.0,
        qbc_max_episodes: int = 3,
        convergence_patience: Optional[int] = None,
        improvement_eps: float = 1e-3,
        event_weighted_scoring: bool = False,
        reward_weight: float = 0.0,
        episode_outcome_labels: bool = False,
        refining_smallest_edit: bool = False,
        force_refining_after_iter0: bool = False,
        edit_budget_decay: bool = False,
        edit_budget_init: int = 20,
        edit_budget_min: int = 3,
        use_ucb1_selection: bool = False,
        ucb1_c: float = 1.0,
        use_nll_normalization: bool = False,
        rotate_stratified_per_iter: bool = False,
        use_pareto_selector: bool = False,
        pareto_delta: float = 0.5,
        pareto_eps_large: float = 0.1,
        pareto_rew_signal: Optional[str] = None,
        num_initial_candidates: int = 1,
        initial_sampling_temperatures: Optional[List[float]] = None,
        adaptive_field_weights: bool = False,
        adaptive_field_weights_temperature: float = 1.0,
        adaptive_field_weights_low_pass: float = 0.5,
        sfs_enabled: bool = False,
        sfs_k: int = 3,
        event_gate: bool = False,
        pf_aggregation: str = "mean",
        use_sanity_check: bool = False,
        sanity_check_num_samples: int = 10,
        use_pareto_term_channel: bool = False,
        train_on_demos_only: bool = False,
        elbo_softmax_temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Q2-3 fix: forward kwargs that are ALSO declared on the parent
        # `PartiallyObsPlanningAgent.__init__`. Without this forwarding
        # the parent falls back to its own defaults for these fields, and
        # the subclass reassignments below only paper over the problem
        # at runtime — any parent-side initialisation logic relying on
        # the Hydra value (or a future method that reads `self.decompose`
        # between `super().__init__` and the reassignments) would see
        # the wrong value. Forwarding explicitly makes the subclass
        # reassignments redundant but harmless (idempotent).
        super().__init__(
            *args,
            decompose=decompose,
            msc_horizon=msc_horizon,
            msc_weight=msc_weight,
            scoring_mode=scoring_mode,
            obs_noise_epsilon=obs_noise_epsilon,
            output_particle_distance_metrics=output_particle_distance_metrics,
            model_averaging_top_k=model_averaging_top_k,
            **kwargs,
        )
        self.num_state_samples = num_state_samples
        self.qbc_explore_prob = qbc_explore_prob
        self.qbc_max_episodes = qbc_max_episodes
        self.sfs_enabled = sfs_enabled
        self.sfs_k = sfs_k
        self.event_gate = event_gate
        # F2 (R#2-A) — opt-in robust median aggregation across episodes.
        # Default "mean" preserves backward compatibility. Setting
        # "median" mitigates the contamination from a single collapsed
        # episode (pf_score=-inf clamped to ELBO_FLOOR) that otherwise
        # pulls the aggregate down. Huber (1964) robust statistics
        # foundation. Affects both REx scoring (`_score_models`) and
        # diagnostic `[ELBO]` line.
        if pf_aggregation not in ("mean", "median"):
            raise ValueError(
                f"pf_aggregation must be 'mean' or 'median', got {pf_aggregation!r}"
            )
        self.pf_aggregation = pf_aggregation

        # F3 (MX-1-A) — opt-in pre-deployment sanity check on LLM
        # candidates. When True, each freshly-compiled joint model set is
        # exercised on 10 real replay-buffer transitions (see
        # `uncertain_worms.policies.sanity_check`) before being accepted
        # into the REx tree. Rejects candidates that would crash the
        # planner at deployment time with the canonical
        # `ValueError: truth value of an array…` class of bugs.
        # Default False = byte-identical with the pre-F3 agent.
        self.use_sanity_check = bool(use_sanity_check)
        self.sanity_check_num_samples = int(sanity_check_num_samples)

        # F5 (R#5-A) — opt-in fourth Pareto channel for the terminated
        # cross-entropy. When True, `_pareto_accept` tightens the rule
        # so candidates that regress the step-averaged ``term_nats`` by
        # more than ``pareto_delta`` are rejected even if obs/trans/rew
        # are unchanged. Lava-terminal blind spot fix (paper §3.3).
        # Default False = byte-identical with pre-F5 selector.
        self.use_pareto_term_channel = bool(use_pareto_term_channel)

        # Keep the REx training set stable across online
        # episodes. When True, ``online_update_models`` refreshes its
        # scoring signals on newly collected episodes but does NOT
        # append them to ``train_replay_buffer``; the demonstration
        # set remains the ground for all model fit so convergence can
        # be measured against a fixed reference.
        # Default False = byte-identical with pre-F6 behaviour (online
        # data always added to train).
        self.train_on_demos_only = bool(train_on_demos_only)

        # Softmax ELBO sampling of
        # `best_node` at the end of `joint_update_models_rex` instead
        # of a hard argmax. When `elbo_softmax_temperature > 0`, the
        # final deployed candidate is drawn from
        # ``p_k ∝ exp(score_k / T)`` over all generated nodes, where
        # ``score_k = (train_coverage + test_coverage) / 2`` (the same
        # scalar the argmax path uses). T → 0 recovers argmax; T → ∞
        # gives uniform sampling. The motivation is that a hard argmax
        # on a noisy scorer can leave us stuck 1 % below the correct
        # model; Boltzmann exploration (classical RL) restores a
        # chance to deploy the "second-best" candidate. Default 0.0 =
        # byte-identical with pre-F9 argmax behaviour.
        self.elbo_softmax_temperature = float(elbo_softmax_temperature)
        if self.elbo_softmax_temperature < 0.0:
            raise ValueError(
                f"elbo_softmax_temperature must be ≥ 0, got "
                f"{self.elbo_softmax_temperature!r}"
            )

        self.model_translators = {
            "transition_model": transition_model_translator,
            "reward_model": reward_model_translator,
            "observation_model": observation_model_translator,
            "initial_model": initial_model_translator,
        }
        self.should_learn = {
            "transition_model": learn_transition,
            "reward_model": learn_reward,
            "observation_model": learn_observation,
            "initial_model": learn_initial,
        }

        self.model_names = list(self.should_learn.keys())
        self.num_input_examples = num_input_examples
        self.env_description = env_description
        self.goal_description = goal_description
        self.replay_buffer = ReplayBuffer[StateType, ActType, ObsType]()
        self.previous_state: Optional[StateType] = None
        self.num_model_attempts = num_model_attempts
        self.num_online_model_attempts = num_online_model_attempts
        self.current_code = {model_name: "" for model_name in self.model_names}
        self.use_offline = use_offline
        self.use_online = use_online
        self.output_particle_distance_metrics = output_particle_distance_metrics
        self.decompose = decompose
        self.msc_horizon = msc_horizon
        self.msc_weight = msc_weight
        self.scoring_mode = scoring_mode
        self.obs_noise_epsilon = obs_noise_epsilon
        self.model_averaging_top_k = model_averaging_top_k
        # M5: REx convergence knobs (Hydra-configurable).
        self.convergence_patience = convergence_patience
        self.improvement_eps = improvement_eps
        # I3: inverse-frequency event weighting in the score evaluator.
        # Default False preserves byte-identical output against the
        # pre-I3 formula; flag threads through _make_elbo_evaluator()
        # so both offline and online REx instantiations pick it up.
        self.event_weighted_scoring = event_weighted_scoring
        # Preliminary #8: explicit reward MSE term in pf_score.
        # Default 0.0 = byte-identical, positive values add
        # ``+ reward_weight * Σ score_rew`` to total_score during PF
        # scoring (requires decompose=True to have per-step score_rew).
        self.reward_weight = float(reward_weight)
        # I_LABELS: annotate `--- Episode N ---` headers with the episode
        # outcome (SUCCESS / FAILURE / TRUNCATED). Default False keeps
        # the prompt byte-identical against the pre-flag template; opt-in
        # via ``++agent.episode_outcome_labels=true`` on the Hydra cli.
        self.episode_outcome_labels = bool(episode_outcome_labels)
        # I_SMALL_EDIT: append a "smallest possible edit" instruction to
        # the refining prompt. Targets the LR-too-high pathology where
        # the LLM rewrites large unrelated sections at every refinement
        # step instead of fixing only the diagnosed errors. Default False
        # keeps the prompt byte-identical; opt-in via Hydra
        # ``++agent.refining_smallest_edit=true``.
        self.refining_smallest_edit = bool(refining_smallest_edit)
        # I_OBS_EXTEND: post-process loaded pickles so the obs objects
        # carry their parallel-array reward / terminated / agent_dir
        # values on the new optional dataclass fields. Default False
        # leaves the obs at their dataclass defaults (reward=0,
        # terminated=False, agent_dir=0) → byte-identical with the
        # pre-extension behaviour. Combine with the OBS_*_WEIGHT env
        # vars (read by minigrid_env at module load) to enable the
        # weighted distance contributions.
        # I_FORCE_REFINING: after iter 0 of a REx loop produces a
        # successful child node, freeze the root by setting its
        # ``to_update=False``. This biases the Thompson sampler toward
        # non-root nodes for iter 1+, which routes through the refining
        # prompt path instead of the starting one. Justification: the
        # default sampler keeps picking the root because the
        # Beta-Bernoulli update boosts root's α each time iter 0 is an
        # improvement, but the refining path is where the
        # ``smallest_edit`` constraint and the ``term_hint`` reach the
        # LLM as concrete edits to its previous code rather than as
        # context for a fresh attempt. Default False keeps the sampler
        # behaviour byte-identical with the pre-flag REx loop.
        self.force_refining_after_iter0 = bool(force_refining_after_iter0)
        # I_EDIT_BUDGET: linearly decaying per-function insertion cap
        # across REx iterations. The LLM may REMOVE any number of lines
        # but may only INSERT up to ``edit_budget_init`` new lines at
        # iter 0, decaying to ``edit_budget_min`` at the last iter.
        # Rationale: late iterations should prune and generalise, not
        # pile on new special cases. Unlimited removal keeps the LLM
        # free to shorten its hypothesis. Default False keeps the
        # refining prompt byte-identical; opt-in via Hydra
        # ``++agent.edit_budget_decay=true``.
        self.edit_budget_decay = bool(edit_budget_decay)
        self.edit_budget_init = int(edit_budget_init)
        self.edit_budget_min = int(edit_budget_min)
        # D: UCB1 replaces the Thompson-on-Beta parent selector when
        # enabled. The Thompson choice is non-standard for tree search
        # (REx NeurIPS 2024 documents the same root-dominance failure
        # mode) — UCB1 gives fresh nodes an automatic exploration bonus
        # via ``c * sqrt(ln N / n_visit)`` and removes the need for the
        # ``force_refining_after_iter0`` band-aid. Default False keeps
        # the sampler byte-identical to the pre-flag behaviour.
        self.use_ucb1_selection = bool(use_ucb1_selection)
        self.ucb1_c = float(ucb1_c)
        # B: convert the three MODEL QUALITY components (obs / trans /
        # reward accuracies) to a common NLL unit before picking the
        # ``weakest component`` — otherwise the native-scale bias picks
        # obs_acc virtually every time because it is ``exp(-dist)``.
        self.use_nll_normalization = bool(use_nll_normalization)
        # A: rotate the stratified-sample seed by iter_num so the
        # prompt does not re-show the same 3/8 episodes forever.
        self.rotate_stratified_per_iter = bool(rotate_stratified_per_iter)
        # I_PARETO: multi-criteria acceptance in the REx monotone bound
        # update. When enabled, a candidate is accepted if its per-event
        # reward likelihood strictly improves AND neither obs nor trans
        # per-step NLL regress by more than ``pareto_delta`` nats, OR if
        # the aggregate ``combined_elbo`` clears the old ``improvement_eps``
        # path by ``pareto_eps_large`` nats. Default False keeps the
        # monotone comparator byte-identical. Targets the documented
        # "good reward code rejected sub-noise" failure mode: the 2%
        # signal contribution of reward_func to pf_score is drowned in
        # obs/trans variance under the scalar monotone criterion.
        self.use_pareto_selector = bool(use_pareto_selector)
        self.pareto_delta = float(pareto_delta)
        self.pareto_eps_large = float(pareto_eps_large)
        # ``pareto_rew_signal`` picks which metric key feeds the Pareto
        # reward channel. ``None`` keeps the default cascade (joint
        # energy, then value-only fallback). Set to
        # ``"fivo_event_log_likelihood"`` to use the continuous
        # ``P(exact match | belief)`` score. This is a graduated version
        # of the legacy binary ``reward_correct``
        # and recovers signal when the weighted majority disagrees with
        # the observed event but a non-zero belief fraction still
        # matches. POMDP-pure: the metric it reads comes from
        # ``explain_likelihood`` per step, which only consumes particle
        # weights and observed ``(reward, terminated)`` tuples.
        self.pareto_rew_signal: Optional[str] = (
            str(pareto_rew_signal) if pareto_rew_signal else None
        )
        # I_MULTI_INIT: draw K initial candidates at the start of the
        # REx loop instead of a single one. Each candidate becomes a
        # child of the bootstrap root and is evaluated with the full
        # scoring stack, so the bandit (UCB1 / Thompson) can pick among
        # them. Tied together with ``initial_sampling_temperatures``:
        # when that list is provided, candidate ``k`` uses temperature
        # ``initial_sampling_temperatures[k % len(list)]`` instead of
        # the agent default, giving the K samples a real chance to
        # disagree (same-temperature resamples tend to collapse to
        # consensus — empirically ``[QBC] vote_entropy≈0.000``). When
        # the list is ``None``, all K candidates use ``self.llm_temperature``.
        # ``num_initial_candidates=1`` is byte-identical to the
        # pre-flag behaviour. Requires ``num_model_attempts >=
        # num_initial_candidates`` so the refining phase has at least
        # one step left — enforced at call site.
        self.num_initial_candidates = max(1, int(num_initial_candidates))
        self.initial_sampling_temperatures: Optional[List[float]] = (
            [float(t) for t in initial_sampling_temperatures]
            if initial_sampling_temperatures
            else None
        )
        # I_ADAPTIVE_FIELD_WEIGHTS: per-field scorer weights calibrated
        # from the demo buffer and adapted per REx iter. Opt-in, read
        # back from ``self.adaptive_field_weights`` during buffer load.
        self.adaptive_field_weights = bool(adaptive_field_weights)
        self.adaptive_field_weights_temperature = float(
            adaptive_field_weights_temperature
        )
        self.adaptive_field_weights_low_pass = float(adaptive_field_weights_low_pass)
        # MIN2: explicit RNG for stratified episode sampling, seeded from
        # the policy's llm_seed so results reproduce at the same seed.
        # Falls back to 0 when no seed was supplied.
        _strat_seed = self.llm_seed if self.llm_seed is not None else 0
        self._strat_rng = random.Random(_strat_seed)
        self.elbo = float('-inf')
        self.previous_elbo = float("-inf")
        with open(os.path.join(PROMPT_DIR, "po_inserts.json")) as f:
            self.inserts = json.load(f)
        self.joint_model_inserts = self._resolve_joint_model_inserts(
            self.inserts, env_name
        )

        self.templates: Dict[str, str] = {}
        for model_name in self.model_names:
            with open(
                os.path.join(
                    PROJECT_ROOT,
                    env_code_path,
                    f"model_templates/po_{model_name}_template.txt",
                )
            ) as f:
                self.templates[model_name] = f.read().strip()

        with open(os.path.join(PROJECT_ROOT, env_code_path, "api.py")) as f:
            self.code_api = f.read().strip()

        if self.use_offline:
            # Train
            log.info("Loading offline dataset: %s", os.path.join(PROJECT_ROOT, dataset_path))
            train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
                os.path.join(PROJECT_ROOT, dataset_path)
            ) # we cretae a replay buffer containing the offline-training data
            self.init_types(train_replay_buffer)
            self.offline_replay_buffer = train_replay_buffer
            # I_ADAPTIVE_FIELD_WEIGHTS: change-rate-inverse per-field
            # weighting for the scorer. Flat OBS_*_WEIGHT env vars
            # under-penalise rare-but-informative fields (e.g.
            # ``carrying`` on Unlock changes on ~6% of transitions, so
            # a buggy transition_func that makes pickup a no-op is
            # barely distinguishable from a correct one in ELBO).
            # Installing the adapter here — after buffer load and
            # before the first REx iteration — makes the subsequent
            # ``distance_soft`` calls use per-field weights auto-
            # calibrated from the demos. Opt-in: default-off keeps the
            # legacy path byte-identical.
            if getattr(self, "adaptive_field_weights", False):
                from particle_filtering.field_weights import (
                    FieldWeightAdapter,
                    set_active_adapter,
                )
                adapter = FieldWeightAdapter.calibrated_from_buffer(
                    train_replay_buffer,
                    temperature=float(
                        getattr(self, "adaptive_field_weights_temperature", 1.0)
                    ),
                    low_pass=float(
                        getattr(self, "adaptive_field_weights_low_pass", 0.5)
                    ),
                )
                set_active_adapter(adapter)
                log.info(
                    "[I_ADAPTIVE_FIELD_WEIGHTS] installed adapter with weights=%s",
                    {k: round(v, 3) for k, v in adapter.weights.items()},
                )
            self.offline_update_models() # this is where we do the intial training of the model

    def _active_joint_models(self) -> List[str]:
        """Models that should be jointly updated in this run."""
        return [
            model_name
            for model_name in self.JOINT_MODEL_ORDER
            if self.should_learn.get(model_name, False)
        ]

    def _format_joint_function_names(self, active_joint_models: List[str]) -> str:
        function_names = [
            self.JOINT_MODEL_TO_FUNC[model_name] for model_name in active_joint_models
        ]
        if not function_names:
            return ""
        if len(function_names) == 1:
            return function_names[0]
        if len(function_names) == 2:
            return f"{function_names[0]} and {function_names[1]}"
        return ", ".join(function_names[:-1]) + f", and {function_names[-1]}"

    def gpt_update_model(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        iter_num: int,
        exec_attempt: int,
        episode: int = 0,
    ) -> Optional[str]:
        model_func_name = model_name.replace("model", "func")

        code_str, local_scope = requery(
            messages,
            model_func_name,
            iter_num,
            exec_attempt,
            replay_path=self.replay_path,
            episode=episode,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            llm_top_p=self.llm_top_p,
            llm_seed=self.llm_seed,
        )

        if code_str is not None:
            assert self.type_tuple is not None
            new_model = self.model_translators[model_name](
                model_func_name, local_scope, type_tuple=self.type_tuple
            )
            if model_name == "initial_model":
                self.planner.initial_model = new_model  # type:ignore
            elif model_name == "transition_model":
                self.planner.transition_model = new_model  # type:ignore
            elif model_name == "reward_model":
                self.planner.reward_model = new_model  # type:ignore
            elif model_name == "observation_model":
                self.planner.observation_model = new_model  # type:ignore

            return code_str
        return None

    def gpt_joint_update_model(
        self,
        messages: List[Dict[str, str]],
        active_joint_models: List[str],
        iter_num: int,
        exec_attempt: int,
        episode: int = 0,
        temperature_override: Optional[float] = None,
    ) -> Optional[str]:
        if not active_joint_models:
            log.info("[Joint] No active models selected for joint update.")
            return None

        # When ``temperature_override`` is supplied (multi-init phase),
        # it takes precedence over ``self.llm_temperature`` for this
        # single call only. None = fall back to the agent default.
        temp = (
            float(temperature_override)
            if temperature_override is not None
            else self.llm_temperature
        )
        # Here it will output three functions at once, one for each model
        code_str, local_scope = requery_joint(
            messages,
            "joint_models",
            iter_num,
            exec_attempt,
            replay_path=self.replay_path,
            episode=episode,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            required_functions=[
                self.JOINT_MODEL_TO_FUNC[model_name]
                for model_name in active_joint_models
            ],
            llm_temperature=temp,
            llm_top_p=self.llm_top_p,
            llm_seed=self.llm_seed,
        )

        if code_str is not None:
            assert self.type_tuple is not None

            if self.use_sanity_check:
                from uncertain_worms.policies.sanity_check import (
                    sanity_check_candidate,
                )

                required = tuple(
                    self.JOINT_MODEL_TO_FUNC[m] for m in active_joint_models
                )
                # At iter=0 / episode=-1 the online replay buffer is still
                # empty. Prefer offline demos whenever available; fall back
                # to online only if the offline buffer is empty.
                buffer = (
                    self.offline_replay_buffer
                    if getattr(self.offline_replay_buffer, "episodes", None)
                    else self.replay_buffer
                )
                ok, err = sanity_check_candidate(
                    local_scope,
                    buffer,
                    self.empty_observation,
                    num_samples=self.sanity_check_num_samples,
                    required=required,
                )
                if not ok:
                    log.info(
                        f"[SANITY_CHECK_REJECT] iter={iter_num} "
                        f"exec_attempt={exec_attempt} active="
                        f"{active_joint_models}: {err.splitlines()[0][:200]}"
                    )
                    return None

            for model_name in active_joint_models:
                function_name = self.JOINT_MODEL_TO_FUNC[model_name]
                new_model = self.model_translators[model_name](
                    function_name, local_scope, type_tuple=self.type_tuple
                )
                self._set_model(model_name, new_model)

            return code_str
        return None

    def print_io(self, input_outputs: List[Tuple[Condition, Outcome]]) -> str:
        io_str = ""
        for condition, outcome in input_outputs:
            if len(condition) > 0:
                for condition_item in list(condition):
                    action_display = condition_item.name if hasattr(condition_item, "name") else str(condition_item)
                    io_str += f"Action: {action_display}\n"
            for outcome_item in list(outcome):
                io_str += f"{str(outcome_item)}\n"
            io_str += "\n\n"
        return io_str

    def _episode_outcome_label(
        self,
        rewards_i: Optional[List[float]],
        terminated_i: Optional[List[bool]],
    ) -> str:
        """Return an outcome annotation for an `--- Episode N ---` header.

        Returns an empty string when:
        - the ``episode_outcome_labels`` flag is off (default), or
        - either rewards_i / terminated_i is missing/empty.

        Otherwise returns a leading space + parenthesised outcome tag, e.g.
        ``" (FAILURE — terminated with reward=0, done=True at step 7)"``.
        Env-agnostic: only inspects the (rewards, terminated) tuple.
        """
        if not getattr(self, "episode_outcome_labels", False):
            return ""
        if not rewards_i or not terminated_i:
            return ""
        n_steps = len(rewards_i)
        final_r = float(rewards_i[-1])
        final_t = bool(terminated_i[-1])
        if final_t and final_r > 0.0:
            tag = (
                f"SUCCESS — terminated with reward={final_r:g}, done=True "
                f"at step {n_steps}"
            )
        elif final_t:
            tag = (
                f"FAILURE — terminated with reward={final_r:g}, done=True "
                f"at step {n_steps} (the last action led to a terminal "
                f"non-goal state)"
            )
        else:
            tag = (
                f"TRUNCATED — episode cut at step {n_steps}, done=False "
                f"(no terminal reached)"
            )
        return f" ({tag})"

    def _stratified_episode_indices(
        self,
        rewards: Optional[List[List[float]]],
        terminated: Optional[List[List[bool]]],
        n_episodes: int,
        n_target: int = 3,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        """Pick episode indices that cover both successes and failures.

        Plain ``random.sample`` of ``n_target`` episodes can return three
        wins or three losses, hiding half of the signal from the LLM.
        This helper guarantees that whenever both classes exist in the
        dataset, the returned subset contains at least one of each.
        Otherwise it falls back to a uniform random sample.

        MIN2: the RNG is explicit. Callers should pass ``self._strat_rng``
        (a seeded ``random.Random`` owned by the agent) so the stratified
        sample is reproducible across re-runs at the same seed, rather
        than depending on the process-global ``random`` state.

        Returns up to ``n_target`` distinct indices into ``range(n_episodes)``.
        """
        n_target = min(n_target, n_episodes)
        if n_target <= 0:
            return []

        _rng = rng if rng is not None else self._strat_rng

        successes: List[int] = []
        failures: List[int] = []
        unknown: List[int] = []
        for i in range(n_episodes):
            r_list = rewards[i] if rewards is not None and i < len(rewards) else None
            t_list = terminated[i] if terminated is not None and i < len(terminated) else None
            if r_list and t_list:
                final_r = float(r_list[-1])
                final_t = bool(t_list[-1])
                if final_t and final_r > 0.0:
                    successes.append(i)
                elif final_r <= 0.0:
                    # Either terminated with reward 0 (hazard / locked door)
                    # or truncated at max_steps without reaching the goal.
                    # Both count as "failure" so non-hazard envs still get
                    # stratified against their timeouts.
                    failures.append(i)
                else:
                    unknown.append(i)
            else:
                unknown.append(i)

        # When both classes exist, force at least one of each.
        chosen: List[int] = []
        if successes and failures and n_target >= 2:
            chosen.append(_rng.choice(successes))
            chosen.append(_rng.choice(failures))
            pool = [i for i in range(n_episodes) if i not in chosen]
            _rng.shuffle(pool)
            chosen.extend(pool[: n_target - 2])
        else:
            pool = list(range(n_episodes))
            _rng.shuffle(pool)
            chosen = pool[:n_target]
        chosen.sort()
        return chosen

    def _reward_coverage_hint(
        self,
        previous_code: Optional[str],
        rewards: Optional[List[List[float]]],
        terminated: Optional[List[List[bool]]],
    ) -> str:
        """Generate an env-agnostic hint about unexercised terminal semantics.

        Key insight from diagnostic analysis: expert demonstrations typically
        contain only successful trajectories. The LLM therefore cannot infer
        alternative terminal conditions (hazard tiles, locked doors, ...) from
        trajectory data alone — it only sees ``Reward=0`` in the middle and
        ``Reward=1, Terminated=True`` at the end. This hint surfaces that
        structural gap by comparing the identifiers referenced by the current
        reward_func against the type vocabulary declared in the API file.

        Returns an empty string when:
        - ``use_reward_coverage_hint`` flag is False on this agent
        - demos already contain failure cases
        - no rewards data was supplied
        - code_api has no enumerable type vocabulary
        """
        # Opt-in: only activate when explicitly enabled via a config flag.
        # This keeps the default behaviour "dataset-only" (theoretically clean)
        # while the hint remains available as an ablation comparison point.
        if not getattr(self, "use_reward_coverage_hint", False):
            return ""
        if rewards is None or terminated is None or not rewards:
            return ""

        # 1. Detect whether demos are "all-success" (every episode ends with
        #    a strictly positive reward). A failure here means any episode
        #    ending with reward <= 0, whether the environment terminated
        #    naturally (hazard, locked door) or the rollout was truncated
        #    at max_steps without reaching the goal — both signals are
        #    useful to the LLM reward model.
        all_success = True
        has_any_failure = False
        for r_list, t_list in zip(rewards, terminated):
            if not r_list:
                continue
            final_r = float(r_list[-1])
            if final_r <= 0.0:
                has_any_failure = True
                all_success = False
        if has_any_failure:
            # Demos show failures, the LLM has the signal it needs.
            return ""
        if not all_success:
            return ""

        # 2. Extract enum/constant identifiers from code_api. We look for
        #    IntEnum members: ``name = <int>`` inside a ``class <X>(IntEnum):``
        #    block. This is env-agnostic — any enum-like API surface works.
        import re
        enum_names: List[str] = []
        if getattr(self, "code_api", None):
            current_class: Optional[str] = None
            for line in self.code_api.splitlines():
                m_class = re.match(r"\s*class\s+(\w+)\s*\(\s*IntEnum\s*\)\s*:", line)
                if m_class:
                    current_class = m_class.group(1)
                    continue
                if current_class is None:
                    continue
                m_member = re.match(r"\s{4}(\w+)\s*=\s*-?\d+", line)
                if m_member:
                    enum_names.append(f"{current_class}.{m_member.group(1)}")
                elif line.strip() and not line.startswith(" "):
                    current_class = None  # left the class scope

        # 3. Which identifiers does the current reward code reference?
        referenced: Set[str] = set()
        if previous_code:
            for name in enum_names:
                if name in previous_code:
                    referenced.add(name)

        unhandled = [n for n in enum_names if n not in referenced]
        if not unhandled:
            return ""

        lines = [
            "REWARD COVERAGE HINT",
            "-" * 20,
            "Your offline demonstrations ALL end successfully "
            "(final reward > 0 on every episode). This means the trajectory "
            "data alone cannot reveal alternative terminal conditions — "
            "i.e. state configurations that could end the episode without "
            "a positive reward.",
            "",
            "Your current reward_func references these API identifiers: "
            + (", ".join(sorted(referenced)) if referenced else "(none)")
            + ".",
            "",
            "The API declares additional identifiers your reward_func does "
            "NOT reference yet: "
            + ", ".join(unhandled[:12])
            + ("" if len(unhandled) <= 12 else f", ... (+{len(unhandled)-12} more)")
            + ".",
            "",
            "For EACH unreferenced identifier, reason step by step:",
            "  1. Could the agent ever end up on / interact with this type?",
            "  2. If yes, does interacting with it terminate the episode?",
            "  3. Should that terminal transition carry a non-zero reward?",
            "Typical omissions are identifiers that trigger terminal states "
            "— either rewarding goal-like outcomes or non-rewarding failure "
            "outcomes — which the demonstrations never exercise.",
            "Do NOT guess blindly — only add cases you can justify from the "
            "API semantics or environment description.",
        ]
        return "\n".join(lines)

    def get_joint_starting_prompt(
        self,
        observations: List[List[ObsType]],
        active_joint_models: List[str],
        previous_observations: Optional[List[List[ObsType]]] = None,
        actions: Optional[List[List[ActType]]] = None,
        rewards: Optional[List[List[float]]] = None,
        terminated: Optional[List[List[bool]]] = None,
    ) -> str:
        starting_prompt_fn = os.path.join(PROMPT_DIR, "po_joint_model_prompt_nostate.txt")

        sample_indices = self._stratified_episode_indices(
            rewards=rewards,
            terminated=terminated,
            n_episodes=len(observations),
            n_target=5,
        )
        sample_observations = [observations[i] for i in sample_indices]
        sample_previous_observations = (
            [
                previous_observations[i] if i < len(previous_observations) else []
                for i in sample_indices
            ]
            if previous_observations is not None
            else None
        )
        sample_actions = [actions[i] for i in sample_indices] if actions is not None else None
        sample_rewards = [rewards[i] for i in sample_indices] if rewards is not None else None
        sample_terminated = [terminated[i] for i in sample_indices] if terminated is not None else None

        # Convert each episode to List[Tuple[Condition, Outcome]] for print_io.
        # With actions: each step t is ((action[t],), (obs[t],)) - action taken, observation received.
        # With rewards/terminated: outcome includes "Reward: X, Terminated: Y" after each obs.
        # The conversion itself lives in the module-level ``_episode_to_io``
        # helper — keeping the per-step tuple layout identical to the
        # previous inline closure so the emitted prompt stays byte-identical.
        episode_strs = [
            self.print_io(
                _episode_to_io(
                    obs_list,
                    sample_actions[i] if sample_actions is not None else None,
                    initial_obs=(
                        sample_previous_observations[i][0]
                        if sample_previous_observations is not None
                        and i < len(sample_previous_observations)
                        and len(sample_previous_observations[i]) > 0
                        else None
                    ),
                    reward_list=sample_rewards[i] if sample_rewards is not None and i < len(sample_rewards) else None,
                    terminated_list=sample_terminated[i] if sample_terminated is not None and i < len(sample_terminated) else None,
                )
            )
            for i, obs_list in enumerate(sample_observations)
        ]
        episode_separator = "\n\n--- Episode {}{} ---\n\n"
        sample_observations_str = "".join(
            (
                episode_separator.format(
                    i + 1,
                    self._episode_outcome_label(
                        sample_rewards[i] if sample_rewards is not None and i < len(sample_rewards) else None,
                        sample_terminated[i] if sample_terminated is not None and i < len(sample_terminated) else None,
                    ),
                )
                + ep_str
            )
            for i, ep_str in enumerate(episode_strs)
        )
        # initial_exp_str = self.print_io(empirical_io)

        with open(starting_prompt_fn, "r", encoding="utf-8") as file:
            starting_prompt = file.read()

        joint_model_inserts = self._joint_prompt_inserts_for(active_joint_models)
        model_names_str = self._format_joint_function_names(active_joint_models)

        initial_template = (
            self.templates["initial_model"]
            if "initial_model" in active_joint_models
            else "# initial_func is disabled in this run."
        )
        observation_template = (
            self.templates["observation_model"]
            if "observation_model" in active_joint_models
            else "# observation_func is disabled in this run."
        )
        transition_template = (
            self.templates["transition_model"]
            if "transition_model" in active_joint_models
            else "# transition_func is disabled in this run."
        )
        reward_template = (
            self.templates["reward_model"]
            if "reward_model" in active_joint_models
            else "# reward_func is disabled in this run."
        )

        starting_templates = {
            **joint_model_inserts,
            "observations": sample_observations_str,
            "initial_code_template": initial_template,
            "observation_code_template": observation_template,
            "transition_code_template": transition_template,
            "reward_code_template": reward_template,
            "code_api": self.code_api,
            "env_description": self.env_description,
            "goal_description": self.goal_description,
            "model_names": model_names_str,
        }
        starting_prompt = starting_prompt.format(**starting_templates)

        # --- Env-agnostic reward coverage hint ---
        # When reward_func is a learning target and demos are all-success,
        # surface the unreferenced API identifiers so the LLM can reason
        # about unexercised terminal conditions on its first attempt.
        if "reward_model" in active_joint_models:
            hint = self._reward_coverage_hint(
                previous_code=None,  # first attempt, no prior code → list all enum names
                rewards=rewards,
                terminated=terminated,
            )
            if hint:
                starting_prompt = (
                    starting_prompt.rstrip() + "\n\n" + hint + "\n"
                )

        return starting_prompt

    # Original feedback prompt
    # Joint feedback prompt ; for now do a very simple thing where we only provide
    def get_joint_feedback_prompt(
        self,
        previous_code: str,
        active_joint_models: List[str],
        model_dist: Dict[Condition, List[Outcome]],
        empirical_dist: Dict[Condition, List[Outcome]],
        episode_metrics: Optional[List[Dict[str, Any]]] = None,
        observations: Optional[List[List[ObsType]]] = None,
        actions: Optional[List[List[ActType]]] = None,
        previous_observations: Optional[List[List[ObsType]]] = None,
        rewards: Optional[List[List[float]]] = None,
        terminated: Optional[List[List[bool]]] = None,
    ) -> Optional[str]:

        # Load the initial prompt template.
        initial_file_path = os.path.join(PROMPT_DIR, "po_joint_model_refining.txt")
        with open(initial_file_path, "r", encoding="utf-8") as file:
            initial_prompt = file.read()

        joint_model_inserts = self._joint_prompt_inserts_for(active_joint_models)
        model_names_str = self._format_joint_function_names(active_joint_models)

        # Format trajectory data for the refinement prompt (same logic as
        # get_joint_starting_prompt).  The LLM needs the raw data to verify
        # its understanding, not just aggregated metrics.
        observations_str = "(No trajectory data available for this refinement step.)"
        if observations is not None and len(observations) > 0:
            # A (opt-in): rotate the stratified sample across REx
            # iterations so the LLM does not keep re-reading the same
            # 3 episodes forever. The default ``self._strat_rng`` is
            # seeded once at __init__, which gives a deterministic but
            # static triplet when called per-iter.
            strat_rng = None
            if getattr(self, "rotate_stratified_per_iter", False):
                base_seed = int(self.llm_seed) if self.llm_seed is not None else 0
                strat_rng = random.Random(
                    rotated_strat_seed(base_seed, int(getattr(self, "rex_iter", 0)))
                )
            sample_indices = self._stratified_episode_indices(
                rewards=rewards,
                terminated=terminated,
                n_episodes=len(observations),
                n_target=3,
                rng=strat_rng,
            )

            # Shares the module-level ``_episode_to_io`` helper with
            # get_joint_starting_prompt so both prompts emit the exact
            # same per-episode layout.
            episode_strs = []
            for i in sample_indices:
                init_obs = None
                if previous_observations and i < len(previous_observations) and previous_observations[i]:
                    init_obs = previous_observations[i][0]
                episode_strs.append(
                    self.print_io(
                        _episode_to_io(
                            observations[i],
                            actions[i] if actions else None,
                            init_obs,
                            rewards[i] if rewards and i < len(rewards) else None,
                            terminated[i] if terminated and i < len(terminated) else None,
                        )
                    )
                )
            observations_str = "".join(
                "\n\n--- Episode {}{} ---\n\n{}".format(
                    j + 1,
                    self._episode_outcome_label(
                        rewards[sample_indices[j]] if rewards and sample_indices[j] < len(rewards) else None,
                        terminated[sample_indices[j]] if terminated and sample_indices[j] < len(terminated) else None,
                    ),
                    ep,
                )
                for j, ep in enumerate(episode_strs)
            )
        # Format metrics section from episode_metrics (structured sections)
        particle_metrics_section = ""
        if episode_metrics is not None and len(episode_metrics) > 0:
            valid = [m for m in episode_metrics if m]
            if valid:
                sections = []
                # Count collapsed episodes (ESS=0 or pf_score=-inf)
                n_collapsed = sum(
                    1 for m in valid
                    if m.get("mean_ESS", 1) == 0 or math.isinf(m.get("pf_score", 0))
                )
                if "pf_score" in valid[0]:
                    clamped_scores = [max(m["pf_score"], _ELBO_FLOOR) for m in valid]
                    avg_score = float(np.mean(clamped_scores))
                    fit_text = (
                        "MODEL FIT\n"
                        "---------\n"
                        f"Overall PF score: {avg_score:.2f}\n"
                        "(0 = perfect prediction, more negative = worse)"
                    )
                    if n_collapsed > 0:
                        fit_text += (
                            f"\n\nCRITICAL: {n_collapsed}/{len(valid)} episodes had "
                            "TOTAL PARTICLE COLLAPSE (ESS=0). This means your "
                            "transition_func produced states whose observations "
                            "never matched reality. Common causes (generic):\n"
                            "- Actions that should be rejected by the environment "
                            "still advance the agent's position.\n"
                            "- Pose- or orientation-changing actions applied with "
                            "the wrong transform (off-by-one, wrong sign).\n"
                            "- Failed / blocked actions not mapped back to the "
                            "'state unchanged' branch (so the particle drifts "
                            "into an impossible configuration).\n"
                            "- State indices swapped or transposed relative to "
                            "the code_api contract (check the axes of every "
                            "array access)."
                        )
                    sections.append(fit_text)
                if "best_particle_distance" in valid[0]:
                    mean_best = np.mean([m["best_particle_distance"] for m in valid])
                    mean_median = np.mean([m["median_particle_distance"] for m in valid])
                    mean_var = np.mean([m["variance_distance"] for m in valid])
                    sections.append(
                        "OBSERVATION EXPLANATION\n"
                        "-----------------------\n"
                        f"Best particle distance: {mean_best:.3f}\n"
                        f"Median particle distance: {mean_median:.3f}\n"
                        f"Distance variance: {mean_var:.3f}\n\n"
                        "Interpretation:\n"
                        "- Best particle distance measures whether ANY state explains the observation.\n"
                        "- Median distance measures how well the particle belief explains observations overall.\n"
                        "- Large gap between best and median indicates transition or inference problems."
                    )
                if "mean_ESS" in valid[0]:
                    mean_ess = np.mean([m["mean_ESS"] for m in valid])
                    min_ess = np.min([m["min_ESS"] for m in valid])
                    total_exceptions = sum(m.get("n_model_exceptions", 0) for m in valid)
                    pf_text = (
                        "PARTICLE FILTER HEALTH\n"
                        "----------------------\n"
                        f"Mean effective sample size (ESS): {mean_ess:.2f}\n"
                        f"Minimum ESS during episode: {min_ess:.2f}\n"
                        "(Max possible ESS ≈ number of particles)\n\n"
                        "Interpretation:\n"
                        "- Low ESS indicates particle degeneracy (few particles explain observations).\n"
                        "- Persistent low ESS suggests transition model mismatch or insufficient particles."
                    )
                    if total_exceptions > 0:
                        pf_text += (
                            f"\n\nWARNING: Your code raised {total_exceptions} runtime "
                            "exceptions during evaluation. Each exception kills a "
                            "particle. Check for: NameError (missing import such "
                            "as ``random``), IndexError (out-of-bounds grid "
                            "access), TypeError (wrong argument types), KeyError "
                            "(missing state attributes)."
                        )
                        # Surface the first real traceback so the LLM can fix
                        # the actual error instead of guessing from the count.
                        first_traces = [
                            m.get("first_exception_trace")
                            for m in valid
                            if m.get("first_exception_trace")
                        ]
                        if first_traces:
                            pf_text += (
                                "\n\nFIRST EXCEPTION (one of many, "
                                "representative):\n```\n"
                                f"{first_traces[0].rstrip()}\n```"
                            )
                    collapse_steps = [m.get("collapse_step") for m in valid
                                      if m.get("collapse_step") is not None]
                    if collapse_steps:
                        pf_text += (
                            f"\n\nCOLLAPSE at step(s): {collapse_steps}. "
                            "Your model lost all particles at these steps — "
                            "the predicted observations diverged completely "
                            "from reality."
                        )
                    sections.append(pf_text)
                # --- Decomposed score feedback (opt-in via decompose=True) ---
                if "fivo_score_obs" in valid[0]:
                    # Normalised, comparable, pedagogical scoring section.
                    #
                    # Each component is reported as a 0-100% accuracy that
                    # the LLM can interpret directly. The raw FIVO scores
                    # (negative log-likelihoods or MSEs) are kept for
                    # debugging only — they were the source of confusion
                    # in earlier feedback runs.

                    # Observation: average distance is small ⇒ good. We
                    # transform via exp(-mean_dist) to get a value in (0,1].
                    obs_dists = [
                        m.get("median_particle_distance", 0.0)
                        for m in valid if "median_particle_distance" in m
                    ]
                    obs_acc = (
                        float(np.mean([math.exp(-d) for d in obs_dists]))
                        if obs_dists else 0.0
                    )

                    # Transition health: ESS / num_particles ratio.
                    ess_ratios = []
                    for m in valid:
                        ess = m.get("mean_ESS", 0.0)
                        # num_particles config — fall back to 100 if unknown
                        K = max(getattr(self.planner, "num_particles", 100), 1)
                        ess_ratios.append(min(ess / K, 1.0))
                    trans_acc = float(np.mean(ess_ratios)) if ess_ratios else 0.0

                    # Reward accuracy: % of steps where prediction matches.
                    # The OVERALL number is misleading because ~90% of
                    # steps have reward=0. The EVENTS number (steps with
                    # actual reward != 0 or terminal) is the truly
                    # informative metric.
                    rew_overall = [
                        m.get("reward_accuracy_overall", float("nan"))
                        for m in valid
                    ]
                    rew_overall = [r for r in rew_overall if not math.isnan(r)]
                    rew_overall_acc = float(np.mean(rew_overall)) if rew_overall else 0.0

                    rew_events = [
                        m.get("reward_accuracy_on_events", float("nan"))
                        for m in valid
                    ]
                    rew_events = [r for r in rew_events if not math.isnan(r)]
                    rew_events_acc = float(np.mean(rew_events)) if rew_events else float("nan")

                    # Per-dimension accuracies (value match vs terminated
                    # flag match), averaged across ALL steps. These break
                    # the headline down so the LLM can see which dim is
                    # driving the drop.
                    value_accs = [
                        m.get("reward_value_accuracy", float("nan")) for m in valid
                    ]
                    value_accs = [r for r in value_accs if not math.isnan(r)]
                    value_acc = float(np.mean(value_accs)) if value_accs else float("nan")
                    term_accs = [
                        m.get("terminated_accuracy", float("nan")) for m in valid
                    ]
                    term_accs = [r for r in term_accs if not math.isnan(r)]
                    term_acc = float(np.mean(term_accs)) if term_accs else float("nan")

                    n_events_total = sum(
                        m.get("n_reward_events", 0) for m in valid
                    )
                    n_steps_total = sum(
                        m.get("n_total_steps", 0) for m in valid
                    )

                    # === MODEL FIT — concrete, action-localised, trace-based ===
                    # Replaces the older "MODEL QUALITY" / per-context blocks.
                    # Goal: assert what works first, then surface a few
                    # concrete mismatches as 7-step windowed traces so the
                    # LLM can see WHERE the belief breaks rather than be
                    # told the score is "low". Every field shown is either
                    # belief-MAP-derived or buffer-observable — no state
                    # oracle anywhere.
                    fit_lines = self._build_model_fit_section(
                        valid_metrics=valid,
                        n_steps_total=n_steps_total,
                        n_events_total=n_events_total,
                        rew_events_acc=rew_events_acc,
                    )
                    sections.append("\n".join(fit_lines))
                # --- Multi-Step Consistency feedback ---
                if "msc_score" in valid[0]:
                    avg_msc = float(np.mean([
                        m["msc_score"] for m in valid
                        if not math.isinf(m.get("msc_score", float("-inf")))
                    ] or [float("-inf")]))
                    h = valid[0].get("msc_horizon", "?")
                    msc_lines = [
                        f"MULTI-STEP TRANSITION CONSISTENCY (horizon={h})",
                        "-" * 50,
                        f"Score: {avg_msc:.4f}",
                        "(0 = perfect d-step prediction, more negative = compounding transition errors)\n",
                        "Interpretation:",
                        f"- This measures whether your transition model is accurate over {h} consecutive steps.",
                        "- If this score is much worse than the 1-step observation score, your transition model",
                        "  drifts over multiple steps — small errors accumulate.",
                        "- Fix: ensure transition_func handles blocked moves, pose changes, and edge-case transitions precisely (whatever 'blocked' and 'pose change' mean in this environment).",
                    ]
                    sections.append("\n".join(msc_lines))
                # --- Observation-conditioned scoring (worst contexts) ---
                if "obs_action_groups" in valid[0]:
                    all_groups = []
                    for m in valid:
                        all_groups.extend(m.get("obs_action_groups", []))
                    if all_groups:
                        # Aggregate by (obs_hash, action)
                        from collections import defaultdict
                        agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
                        change_flag: Dict[Tuple[str, str], bool] = {}
                        for g in all_groups:
                            key = (g["obs_hash"], g["action"])
                            agg[key].extend([g["mean_score"]] * g["count"])
                            if g["has_obs_change"]:
                                change_flag[key] = True

                        # Also collect grids for display
                        grid_lookup: Dict[Tuple[str, str], Any] = {}
                        for g in all_groups:
                            key = (g["obs_hash"], g["action"])
                            if g.get("obs_grid") is not None and key not in grid_lookup:
                                grid_lookup[key] = g["obs_grid"]

                        sorted_groups = sorted(
                            agg.items(), key=lambda x: np.mean(x[1])
                        )
                        worst_3 = sorted_groups[:3]
                        if worst_3:
                            oc_lines = [
                                "PREDICTION ACCURACY BY CONTEXT (observation, action)",
                                "-" * 55,
                                "Contexts where your model predicts poorly:",
                            ]
                            for (oh, act), scores in worst_3:
                                mean_s = float(np.mean(scores))
                                n = len(scores)
                                changed = change_flag.get((oh, act), False)
                                change_str = " [obs changes here]" if changed else ""
                                grid = grid_lookup.get((oh, act))
                                if grid is not None:
                                    grid_str = str(grid)
                                else:
                                    grid_str = oh + ".."
                                oc_lines.append(
                                    f"  observation={grid_str} + action={act}: "
                                    f"avg_score={mean_s:.4f} (n={n}){change_str}"
                                )
                            oc_lines.append(
                                "\nContexts marked [obs changes here] are steps where the "
                                "agent's observation shifts between consecutive time-steps "
                                "(something entered or left the field of view, or the pose "
                                "changed). Prediction errors at these steps usually mean your "
                                "transition_func mishandles the underlying state change — "
                                "check the rule that decides whether an action is blocked, "
                                "succeeds, or alters the agent's pose."
                            )
                            sections.append("\n".join(oc_lines))
                if sections:
                    particle_metrics_section = "\n\n".join(sections)

        # --- QBC: Model disagreement feedback (if computed) ---
        if getattr(self, "_qbc_feedback", ""):
            if particle_metrics_section:
                particle_metrics_section += "\n\n" + self._qbc_feedback
            else:
                particle_metrics_section = self._qbc_feedback

        # --- Reward coverage hint (env-agnostic) ---
        # When demos are all-success and reward_func is a learned target,
        # surface the list of API identifiers not yet referenced by the code.
        # This addresses the demonstrated bottleneck where the LLM misses
        # terminal conditions (e.g., hazard tiles) unobserved in expert demos.
        if "reward_model" in active_joint_models:
            reward_hint = self._reward_coverage_hint(
                previous_code=previous_code,
                rewards=rewards,
                terminated=terminated,
            )
            if reward_hint:
                if particle_metrics_section:
                    particle_metrics_section += "\n\n" + reward_hint
                else:
                    particle_metrics_section = reward_hint

        if not particle_metrics_section:
            particle_metrics_section = "N/A (no metrics)"

        initial_templates = {
            **joint_model_inserts,
            "code_api": self.code_api,
            "code": previous_code,
            "observations": observations_str,
            "particle_metrics_section": particle_metrics_section,
            "env_description": self.env_description,
            "goal_description": self.goal_description,
            "model_names": model_names_str,
        }

        try:
            rendered = initial_prompt.format(**initial_templates)
        except KeyError as e:
            log.error(
                "format() KeyError %s in get_joint_feedback_prompt. "
                "Template keys: %s, dict keys: %s",
                e,
                [m.group(1) for m in __import__('re').finditer(r'\{(\w+)\}', initial_prompt)],
                list(initial_templates.keys()),
            )
            raise

        # I_SMALL_EDIT (opt-in): append a "smallest edit" rule to the
        # refining prompt. Default False → byte-identical with the
        # pre-flag template. Targets the documented LR-too-high
        # pathology of the REx loop (large rewrites at every iter).
        if getattr(self, "refining_smallest_edit", False):
            smallest_edit_rule = (
                "SMALLEST EDIT (REQUIRED)\n"
                "------------------------\n"
                "Your refinement must be the smallest possible diff to your "
                "previous code that addresses the errors diagnosed above.\n"
                "- Do NOT rewrite or restructure code paths that are not "
                "implicated in the reported errors.\n"
                "- Keep the same function signatures, helper functions, "
                "variable names, and control flow wherever possible.\n"
                "- A targeted three-line fix is preferable to a confident "
                "rewrite. If a single conditional branch is wrong, change "
                "only that branch.\n"
                "- Do not delete branches that already worked just because "
                "you would write them differently from scratch."
            )
            rendered = rendered.rstrip() + "\n\n" + smallest_edit_rule + "\n"

        # I_EDIT_BUDGET (opt-in): linearly decaying per-function insertion
        # cap. The budget shrinks as REx iterates so the LLM is forced to
        # generalise (fewer new lines) rather than pile on FOV-specific
        # special cases. Removals remain unlimited so the LLM can keep
        # shortening its hypothesis.
        if getattr(self, "edit_budget_decay", False):
            max_iter = max(int(getattr(self, "num_model_attempts", 1)), 1)
            cur_iter = max(int(getattr(self, "rex_iter", 0)), 0)
            K0 = int(getattr(self, "edit_budget_init", 20))
            Kmin = int(getattr(self, "edit_budget_min", 3))
            if max_iter <= 1:
                budget = K0
            else:
                progress = min(cur_iter, max_iter - 1) / (max_iter - 1)
                budget = max(Kmin, int(round(K0 - (K0 - Kmin) * progress)))
            edit_budget_rule = (
                "EDIT BUDGET (REQUIRED)\n"
                "----------------------\n"
                f"This is refinement iteration {cur_iter + 1} of {max_iter}. "
                "To force generalisation, your edit budget decays with iterations.\n"
                f"- You may INSERT at most {budget} NEW line(s) in EACH of "
                f"{model_names_str} this iteration.\n"
                "- You may REMOVE any number of lines — pruning is always free.\n"
                "- Prefer shorter, more general rules over longer enumerations "
                "of specific FOV patterns. If you cannot fit a fix within the "
                "insertion budget, shorten an existing branch instead."
            )
            rendered = rendered.rstrip() + "\n\n" + edit_budget_rule + "\n"

        return rendered

    # ------------------------------------------------------------------ #
    # Shared helpers for offline / online / REx update entry points      #
    # ------------------------------------------------------------------ #

    def _make_elbo_evaluator(self) -> "ELBOEvaluator":
        """Build an :class:`ELBOEvaluator` bound to the planner's current
        transition / observation / reward / initial models.

        Every numeric parameter now propagates from the agent's own
        attributes (themselves populated from Hydra config) so the
        scoring stage uses the same knobs the user configured for
        planning. The previous implementation hard-coded ``num_particles=
        100`` and ``kernel_bandwidth=1.0``, silently diverging from
        ``++agent.num_particles`` / ``++agent.kernel_bandwidth``
        overrides and creating "10 at plan, 100 at score" mismatches
        in the logs. ``max_rejuvenation_attempts`` now reads from
        ``self.max_rejuvenation_attempts`` (plumbed through the
        constructor) so it too can be configured or at least logged.
        """
        return ELBOEvaluator(
            transition_model=self.planner.transition_model,
            observation_model=self.planner.observation_model,
            reward_model=self.planner.reward_model,
            initial_model=self.planner.initial_model,
            empty_observation=self.empty_observation,
            empty_state=self.empty_state,
            num_particles=int(getattr(self, "num_particles", 100)),
            kernel_bandwidth=float(getattr(self, "kernel_bandwidth", 1.0)),
            max_rejuvenation_attempts=int(
                getattr(self, "max_rejuvenation_attempts", 1000)
            ),
            decompose=self.decompose,
            msc_horizon=self.msc_horizon,
            scoring_mode=self.scoring_mode,
            obs_noise_epsilon=self.obs_noise_epsilon,
            event_weighted_scoring=self.event_weighted_scoring,
            reward_weight=self.reward_weight,
        )

    @staticmethod
    def _stratified_success_failure_split(
        episodes: List[Any],
        seed: int = 0,
    ) -> Tuple[List[Any], List[Any], int, int, int, int]:
        """Split ``episodes`` into a (train, test) pair stratified by
        success / failure terminal status.

        Roughly half of the successes and half of the failures go to
        the test set so the PF score on test stays a meaningful
        out-of-sample signal (the old "first half / second half"
        split left test = all-successes when datasets were ordered,
        which halved the effective discrimination signal).

        Returns ``(train_eps, test_eps, n_succ_total, n_fail_total,
        n_succ_test, n_fail_test)``. Uses a fresh
        ``random.Random(seed)`` so the same episodes land in the same
        buckets across re-runs.
        """
        successes: List[Any] = []
        failures: List[Any] = []
        for ep in episodes:
            if ep.rewards and ep.terminated and ep.terminated[-1] and ep.rewards[-1] > 0:
                successes.append(ep)
            else:
                failures.append(ep)
        rng = random.Random(seed)
        rng.shuffle(successes)
        rng.shuffle(failures)
        n_succ_test = len(successes) // 2
        n_fail_test = len(failures) // 2
        test_eps = successes[:n_succ_test] + failures[:n_fail_test]
        train_eps = successes[n_succ_test:] + failures[n_fail_test:]
        rng.shuffle(test_eps)
        rng.shuffle(train_eps)
        return (
            train_eps,
            test_eps,
            len(successes),
            len(failures),
            n_succ_test,
            n_fail_test,
        )

    def online_update_models(
        self,
        replay_buffer: ReplayBuffer,
        episode: int,
    ) -> None:
        log.info("Online updating models")

        if not self.use_online:
            log.warning("Online learning is disabled, skipping online update.")
            return

        if self.use_offline:
            # Stratified split (see comment in offline_update_models): mix
            # successes and failures evenly between train and test so the
            # test pf_score remains a meaningful out-of-sample signal.
            total_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                self.offline_replay_buffer.episodes + replay_buffer.episodes
            )
            (
                offline_train,
                offline_test,
                _n_succ,
                _n_fail,
                _n_succ_test,
                _n_fail_test,
            ) = self._stratified_success_failure_split(
                list(self.offline_replay_buffer.episodes), seed=0
            )
            # Online collected data is always added to train UNLESS the
            # F6 / R#6-A flag `train_on_demos_only` is set — then the
            # demonstration train set is frozen across episodes so the
            # REx selector can measure honest convergence against a
            # stable reference (instead of the moving target that the
            # online replay injects).
            if self.train_on_demos_only:
                online_train = []
                log.info(
                    "[F6] train_on_demos_only=True → online episodes NOT "
                    "appended to train (frozen reference set)."
                )
            else:
                online_train = replay_buffer.episodes
            test_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](offline_test)
            train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                offline_train + online_train
            )
        else:
            total_replay_buffer = replay_buffer
            train_replay_buffer = replay_buffer
            test_replay_buffer = replay_buffer

        eps = 0.001

        # Use planner.initial_model (same as joint_update) for consistency with the ELBO
        # values shown in prompts. ground_truth_initial_model can cause -inf when mixed
        # with learned transition/observation on offline+online data.
        elbo_evaluator = self._make_elbo_evaluator()

        _, _, _, _, _, error, episode_metrics = elbo_evaluator.evaluate_elbo(
            total_replay_buffer, output_particle_distance_metrics=True
        )
        if error is not None:
            log.warning(f"Error evaluating ELBO: {error}")
            return
        elbo = _elbo_from_episode_metrics(episode_metrics, msc_weight=self.msc_weight, aggregation=getattr(self, "pf_aggregation", "mean"))
        log.info(f"ELBO: {elbo:.4f}")
        # Scores are energy-based: closer to 0 is better, more negative is worse.
        # Only refine when fit on newly collected data gets worse.
        if elbo >= self.previous_elbo - eps:
            log.info(
                f"ELBO on new data is not worse than previous ELBO, skipping update"
            )
            return

        active_joint_models = self._active_joint_models()
        if not active_joint_models:
            log.info("No joint models enabled; skipping online joint update.")
            return

        node, num_rex_nodes = self.joint_update_models_rex(
            train_replay_buffer,
            test_replay_buffer,
            active_joint_models=active_joint_models,
            num_model_attempts=self.num_online_model_attempts,
            episode=episode,
        )

        self.writer.add_scalar(
            "online_num_rex_nodes/" + "joint", num_rex_nodes, episode
        )  # type:ignore
        self.writer.add_scalar(
            "online_test/" + "joint", node.test_coverage, episode
        )  # type:ignore
        self.writer.add_scalar(
            "online_train/" + "joint", node.train_coverage, episode
        )  # type:ignore

        self.reset()

        elbo_evaluator = self._make_elbo_evaluator()

        _, _, _, _, _, error, episode_metrics = elbo_evaluator.evaluate_elbo(
            total_replay_buffer, output_particle_distance_metrics=True
        )
        if error is not None:
            log.warning(f"Error evaluating ELBO : {error}")
            return
        elbo = _elbo_from_episode_metrics(episode_metrics, msc_weight=self.msc_weight, aggregation=getattr(self, "pf_aggregation", "mean"))
        self.previous_elbo = elbo
        log.info(f"ELBO: {elbo:.4f}")
        self.writer.add_scalar(
            "online_total/" + "joint", elbo, episode
        )  # type:ignore



    def offline_update_models(
        self,
    ) -> None:
        log.info("Offline updating models")

        # Stratified train/test split: each half should contain a similar
        # mix of successes and failures so that test scores remain a
        # meaningful out-of-sample signal. The previous "first half / second
        # half" split made test = all successes, train = all failures
        # whenever the dataset was ordered (which generate_mixed_demos.py
        # produces). The result was that the test pf_score was always
        # excellent (easy episodes) and only the train score discriminated
        # between models — halving the effective signal.
        (
            train_eps,
            test_eps,
            n_succ,
            n_fail,
            n_succ_test,
            n_fail_test,
        ) = self._stratified_success_failure_split(
            list(self.offline_replay_buffer.episodes), seed=0
        )
        log.info(
            f"[STRATIFIED SPLIT] train={len(train_eps)} (succ={n_succ-n_succ_test}, "
            f"fail={n_fail-n_fail_test}), test={len(test_eps)} "
            f"(succ={n_succ_test}, fail={n_fail_test})"
        )
        test_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](test_eps)
        train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](train_eps)
        active_joint_models = self._active_joint_models()
        if not active_joint_models:
            log.info("No joint models enabled; skipping offline joint update.")
            return

        node, num_rex_nodes = self.joint_update_models_rex(
            train_replay_buffer,
            test_replay_buffer,
            active_joint_models=active_joint_models,
            num_model_attempts=self.num_model_attempts,
            episode=-1,
        )
        for model_name in active_joint_models:
            self.writer.add_scalar(
                "offline_test/" + model_name, node.test_coverage, 0
            )  # type:ignore
            self.writer.add_scalar(
                "offline_train/" + model_name, node.train_coverage, 0
            )  # type:ignore
            self.writer.add_scalar(
                "offline_num_rex_nodes/" + model_name, num_rex_nodes, 0
            )  # type:ignore

        self.reset()
        elbo_evaluator = self._make_elbo_evaluator()

        _, _, _, _, _, error, episode_metrics = elbo_evaluator.evaluate_elbo(
            self.offline_replay_buffer,
            output_particle_distance_metrics=True,
        )
        elbo = _elbo_from_episode_metrics(episode_metrics, msc_weight=self.msc_weight, aggregation=getattr(self, "pf_aggregation", "mean"))

        if error is not None:
            log.warning(f"Error evaluating ELBO for joint models: {error}")
            return
        self.previous_elbo = elbo
        log.info(f"ELBO: {elbo:.4f}")
        for model_name in active_joint_models:
            self.writer.add_scalar(
                "offline_total/" + model_name, elbo, 0
            )  # type:ignore


    # ------------------------------------------------------------------ #
    # Model scoring: ELBO + optional Matrix-ELBO gradient feedback       #
    # ------------------------------------------------------------------ #

    def _score_models(
        self,
        elbo_evaluator: "ELBOEvaluator",
        replay_buffer: ReplayBuffer,
        active_joint_models: List[str],
    ) -> Tuple[float, Any, Any, Any, Any, Any, Any, Any]:
        """Evaluate current models on *replay_buffer*.

        Returns (score, actions, observations, prev_obs, rewards,
                 terminated, episode_metrics, error).

        The score is the FIVO/ELBO aggregate of per-episode particle
        filter scores (optionally with an MSC term). The legacy "coverage"
        mode was removed — it depended on a never-defined
        ``get_state_distributions_tiger`` helper and always fell back to
        ELBO via a silent ``except``, so every "coverage" experiment was
        in fact an ELBO run in disguise.
        """
        (actions, observations, prev_obs,
         rewards, terminated, error,
         episode_metrics) = elbo_evaluator.evaluate_elbo(
            replay_buffer, output_particle_distance_metrics=True
        )

        # M6: expose raw and clamped aggregates side by side. The returned
        # `score` stays clamped (used by downstream selection) but the log
        # line makes the catastrophic-episode gap visible.
        score, elbo_diag = _elbo_components_from_episode_metrics(
            episode_metrics,
            msc_weight=self.msc_weight,
            aggregation=getattr(self, "pf_aggregation", "mean"),
        )
        if elbo_diag["n_valid"] > 0:
            log.info(
                "[ELBO] clamped=%.4f raw=%.4f n_collapsed=%d/%d "
                "min_raw=%.4f max_raw=%.4f",
                elbo_diag["clamped_mean"], elbo_diag["raw_mean"],
                int(elbo_diag["n_collapsed"]), int(elbo_diag["n_valid"]),
                elbo_diag["min_raw"], elbo_diag["max_raw"],
            )

        # Surface runtime errors from silent catches so that
        # ``_joint_update_model``'s ``num_bug_fix_attempts`` retry
        # triggers within the same REx iteration instead of waiting for
        # the next one. Previously, exceptions raised inside the LLM code
        # were counted per-particle but no ``error`` was propagated, so
        # ``error=None`` kept the retry loop silent — the LLM then
        # re-generated the same buggy code each iter.
        #
        # Heuristic: if ``error`` is still None but at least 50% of
        # episodes captured a ``first_exception_trace``, treat that as a
        # systemic bug in the generated code and propagate the first
        # traceback so the bug-fix retry kicks in.
        if error is None and episode_metrics:
            traces = [
                m.get("first_exception_trace")
                for m in episode_metrics
                if m and m.get("first_exception_trace")
            ]
            n_total = sum(1 for m in episode_metrics if m)
            if traces and n_total > 0 and len(traces) >= (n_total + 1) // 2:
                error = traces[0]
                log.warning(
                    "[RUNTIME_ERROR_PROPAGATED] %d/%d episodes raised; "
                    "surfacing first traceback so num_bug_fix_attempts "
                    "triggers in the same REx iter.",
                    len(traces), n_total,
                )

        return (score, actions, observations, prev_obs,
                rewards, terminated, episode_metrics, error)


    def _run_qbc_probe(
        self,
        *,
        iter_num: int,
        train_replay_buffer: ReplayBuffer,
        active_joint_models: List[str],
        qbc_candidates: List[Tuple[str, Any]],
    ) -> None:
        """Update the QBC committee with the current transition model
        and, if enough members are present, compute vote-entropy
        feedback for the LLM refinement prompt.

        The helper preserves the side-effect semantics of the pre-refactor
        inline block in ``joint_update_models_rex``:

        1. If ``transition_model`` is part of the active joint targets,
           deep-copy the planner's current transition model and append
           it to ``qbc_candidates`` (the same list the REx loop carries
           across iterations) and to ``self._qbc_candidates``.
        2. If the committee has >= 2 members, build per-step beliefs
           from the first ``qbc_max_episodes`` training episodes
           (initial-model draw then deterministic propagation), run
           ``DisagreementDetector.detect_from_trajectory`` and store the
           resulting feedback string on ``self._qbc_feedback`` when the
           committee did not reach consensus.
        3. Any exception during step 2 is caught and reported via a
           ``[QBC] Failed`` WARNING — the REx loop keeps going.

        All magic numbers (20 rejuvenation candidates, 10-particle cap,
        first 10 belief particles, 5 particles per step in the detector)
        are preserved from the original implementation.
        """
        # Step 1: append the newly installed transition model to the
        # running committee. The attribute and local list alias each
        # other already, so the extra assignment is a no-op but we keep
        # it for readability and to match the former inline block.
        if "transition_model" in active_joint_models:
            qbc_candidates.append(
                (f"iter_{iter_num}", copy.deepcopy(self.planner.transition_model))
            )
            self._qbc_candidates = qbc_candidates

        if len(qbc_candidates) < 2:
            return

        try:
            from particle_filtering.model_disagreement import DisagreementDetector

            # MIN9: QBC was historically limited to the first 3 training
            # episodes for speed. Expose via ``qbc_max_episodes`` (Hydra-
            # configurable) so datasets with large horizons still fit in
            # memory while small datasets use them all.
            qbc_max_eps = max(1, int(getattr(self, "qbc_max_episodes", 3)))
            qbc_train_eps = train_replay_buffer.episodes[:qbc_max_eps]

            # Build a per-step belief trajectory by seeding from the
            # initial-model then propagating with the latest transition
            # model. Matches the legacy code exactly: 20 initial draws
            # capped at 10 unique particles, then 10-particle propagation.
            qbc_beliefs: List[ParticleBelief] = []
            qbc_obs_list: List[Any] = []
            for ep in qbc_train_eps:
                current_belief = None
                for t_idx in range(len(ep.actions)):
                    o = ep.next_observations[t_idx]
                    qbc_obs_list.append(o)
                    if current_belief is None:
                        seed_particles: Dict[Any, int] = {}
                        for _ in range(20):
                            seed_state = self.planner.initial_model(
                                self.empty_state.copy()
                            )
                            seed_particles[seed_state] = (
                                seed_particles.get(seed_state, 0) + 1
                            )
                            if sum(seed_particles.values()) >= 10:
                                break
                        current_belief = ParticleBelief(particles=seed_particles)
                    else:
                        next_particles: Dict[Any, int] = {}
                        for state in current_belief.to_list()[:10]:
                            try:
                                next_state = self.planner.transition_model(
                                    state.copy(), ep.actions[t_idx]
                                )
                                next_particles[next_state] = (
                                    next_particles.get(next_state, 0) + 1
                                )
                            except Exception:
                                pass
                        if next_particles:
                            current_belief = ParticleBelief(particles=next_particles)
                    qbc_beliefs.append(current_belief)

            detector = DisagreementDetector(
                qbc_candidates,
                self.planner.observation_model,
                list(self.actions),
                self.empty_observation,
                max_particles_per_step=int(self.qbc_max_particles_per_step),
            )
            all_actions: List[Any] = []
            for ep in qbc_train_eps:
                all_actions.extend(ep.actions)
            qbc_result = detector.detect_from_trajectory(
                qbc_beliefs[: len(all_actions)],
                all_actions,
                qbc_obs_list[: len(all_actions)],
            )
            if qbc_result.feedback_text:
                self._qbc_feedback = qbc_result.feedback_text
                log.info(
                    f"[QBC] N={len(qbc_candidates)} models, "
                    f"vote_entropy={qbc_result.mean_vote_entropy:.3f}, "
                    f"max_VE={qbc_result.max_vote_entropy:.3f}, "
                    f"disagree_rate={qbc_result.disagreement_rate:.2%}"
                )
            else:
                log.info(
                    f"[QBC] N={len(qbc_candidates)} models, "
                    f"vote_entropy={qbc_result.mean_vote_entropy:.3f} "
                    f"(consensus)"
                )
        except Exception:
            log.warning(f"[QBC] Failed: {traceback.format_exc()}")

    # ------------------------------------------------------------------ #
    # Continuous Beta-Bernoulli update for REx parent nodes              #
    # ------------------------------------------------------------------ #

    # Alpha / beta are capped at this posterior mass so a long string of
    # failures cannot trap a node forever (the audit's M2 concern:
    # "long-failing nodes can never recover"). Kept as a class constant
    # so the rationale and the numeric value live next to each other.
    _MAX_POSTERIOR_MASS: float = 50.0

    def _apply_continuous_beta_update(
        self,
        *,
        node: RexNode,
        child_combined_elbo: float,
        improvement_eps: float,
    ) -> None:
        """Apply one continuous Beta-Bernoulli update to ``node`` (M2).

        The "reward" fed to the Beta distribution is a soft success
        indicator

            reward_weight = sigmoid(delta / (10 * improvement_eps))

        with ``delta = child_combined_elbo - parent_combined_elbo``
        where ``parent_combined_elbo`` is ``(train_coverage +
        test_coverage) / 2``. This gives:

        - ``reward_weight ≈ 0.5`` when ``delta == 0`` (no progress);
        - ``reward_weight → 1`` when ``delta >> eps`` (large win);
        - ``reward_weight → 0`` when ``delta << -eps`` (large loss).

        Large improvements therefore earn proportionally more posterior
        mass than marginal ones. ``α`` and ``β`` are each capped at
        :data:`_MAX_POSTERIOR_MASS` so a long-failing node can still
        recover once it finally starts scoring well.

        The sigmoid is evaluated via a numerically stable two-branch
        form so that an ELBO-floored child (e.g. combined_elbo = -100)
        against a healthy parent (combined_elbo ≈ -28) no longer
        overflows ``math.exp`` — the pre-refactor inline block used
        ``1 / (1 + exp(-delta/scale))`` directly, which crashed on such
        inputs once the LLM produced a fully-collapsed candidate. The
        stable form preserves the old output byte-for-byte whenever the
        exponent stays in representable range (normal REx iterations)
        and converges to 0 or 1 on the degenerate inputs that used to
        raise ``OverflowError``.
        """
        parent_combined_elbo = (node.train_coverage + node.test_coverage) / 2
        delta = child_combined_elbo - parent_combined_elbo
        update_scale = max(10.0 * improvement_eps, 1e-9)
        x = float(delta) / update_scale
        if x >= 0.0:
            z = math.exp(-x)
            reward_weight = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            reward_weight = z / (1.0 + z)
        node.alpha = min(node.alpha + reward_weight, self._MAX_POSTERIOR_MASS)
        node.beta = min(
            node.beta + (1.0 - reward_weight), self._MAX_POSTERIOR_MASS
        )

    @staticmethod
    def _pf_self_consistency_mean(
        metrics_list: Optional[List[Dict[str, Any]]],
    ) -> Optional[float]:
        """Return the mean ``pf_self_consistency_pf_score`` across episodes.

        Populated only when ``ScoreEvaluator`` was invoked with
        ``scoring_mode='pf_self_consistency'`` (or as side-field when
        callers explicitly collect it via the aux dict). Otherwise
        returns ``None`` and the opt-in Pareto channel sits silent.
        Obs-only by construction — does not read any ground-truth
        state.
        """
        if not metrics_list:
            return None
        vals: List[float] = []
        for m in metrics_list:
            v = m.get("pf_self_consistency_pf_score")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            vals.append(fv)
        if not vals:
            return None
        return float(np.mean(vals))

    @staticmethod
    def _mean_term_nats(
        metrics_list: Optional[List[Dict[str, Any]]],
    ) -> Optional[float]:
        """Return the cross-episode mean of ``fivo_mean_term_nats``.

        F5 / R#5-A feeds the Pareto selector the terminated cross-entropy
        so a candidate that regresses on predicting ``done=True`` at the
        lava step is penalised on its own axis. NaN / None values are
        skipped; returns ``None`` if no episode populated the field
        (e.g. scorer ran without ``decompose=True``).
        """
        if not metrics_list:
            return None
        vals: List[float] = []
        for m in metrics_list:
            v = m.get("fivo_mean_term_nats")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            vals.append(fv)
        if not vals:
            return None
        return float(np.mean(vals))

    @staticmethod
    def _component_nats_vector(
        metrics_list: Optional[List[Dict[str, Any]]],
        rew_signal_key: Optional[str] = None,
    ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """Return per-component nats ``(obs_per_step, trans_per_step, rew_per_event)``.

        Used by :meth:`joint_update_models_rex` when ``use_pareto_selector``
        is enabled, so component deltas are comparable across episodes
        regardless of episode length. Any component missing from every
        supplied metric dict comes back as ``None``. Returns ``None`` when
        no metrics are supplied.

        * ``obs_per_step = mean(fivo_score_obs / n_total_steps)`` — the raw
          ``fivo_score_obs`` is a sum over steps, so normalising by the
          per-episode step count gives nats-per-step.
        * ``trans_per_step = mean(fivo_score_trans)`` — already stored as
          a per-step mean by the scorer.
        * ``rew_per_event``: resolved from ``metrics_list`` by preference
          ``rew_signal_key`` → ``fivo_score_reward_events_joint`` →
          ``fivo_score_rew_events``. ``rew_signal_key`` is the per-call
          override (e.g. ``"fivo_event_log_likelihood"`` for the
          continuous likelihood Pareto channel). When omitted, the
          default cascade preserves the legacy behaviour (joint
          value+terminated energy, value-only fallback). The
          joint signal is what makes Pareto sensitive to a candidate
          that fixes ``terminated`` without changing the reward value;
          ``fivo_event_log_likelihood`` extends that to a continuous
          P(exact match | belief) signal that preserves graduality when
          the weighted majority disagrees with the observed event
          tuple. NaN / ±inf entries are skipped in all cases.
        """
        if not metrics_list:
            return None
        preferred_keys: Tuple[str, ...] = (
            *(
                (rew_signal_key,)
                if rew_signal_key is not None
                else ()
            ),
            "fivo_score_reward_events_joint",
            "fivo_score_rew_events",
        )
        obs_vals: List[float] = []
        trans_vals: List[float] = []
        rew_vals: List[float] = []
        for m in metrics_list:
            n_steps = max(int(m.get("n_total_steps", 1) or 1), 1)
            if "fivo_score_obs" in m and not math.isinf(m["fivo_score_obs"]):
                obs_vals.append(float(m["fivo_score_obs"]) / n_steps)
            if "fivo_score_trans" in m and not math.isinf(m["fivo_score_trans"]):
                trans_vals.append(float(m["fivo_score_trans"]))
            rv: Optional[float] = None
            for key in preferred_keys:
                candidate = m.get(key)
                if (
                    candidate is not None
                    and not (isinstance(candidate, float) and math.isnan(candidate))
                    and not (isinstance(candidate, float) and math.isinf(candidate))
                ):
                    rv = float(candidate)
                    break
            if rv is not None:
                rew_vals.append(rv)
        obs_mean = float(np.mean(obs_vals)) if obs_vals else None
        trans_mean = float(np.mean(trans_vals)) if trans_vals else None
        rew_mean = float(np.mean(rew_vals)) if rew_vals else None
        if obs_mean is None and trans_mean is None and rew_mean is None:
            return None
        return (obs_mean, trans_mean, rew_mean)

    def _pareto_accept(
        self,
        new_node: "RexNode",
        best_node: Optional["RexNode"],
        combined_elbo: float,
        best_ever_score: float,
    ) -> Tuple[Optional[bool], str]:
        """Pareto-style acceptance for the monotone bound update.

        Returns ``(decision, diag_msg)`` where ``decision`` is ``None``
        when the selector cannot form an opinion (flag off, or metrics
        absent on either node — caller falls back to the scalar monotone
        check) and a ``bool`` otherwise. ``diag_msg`` is a log-friendly
        string to surface the per-component deltas or empty when the
        selector stayed silent. The decision rule is:

            accept  iff  (Δ_rew > 0  AND  Δ_obs > −δ  AND  Δ_trans > −δ)
                    OR   (combined_elbo − best_ever_score) > eps_large

        where δ = ``pareto_delta`` and eps_large = ``pareto_eps_large``.
        When a component delta cannot be computed (missing on either
        side), it is treated as 0 so the rule does not spuriously reject.
        """
        if not getattr(self, "use_pareto_selector", False):
            return None, ""
        if best_node is None:
            return None, ""
        rew_signal_key = getattr(self, "pareto_rew_signal", None)
        new_vec = self._component_nats_vector(
            new_node.train_episode_metrics, rew_signal_key=rew_signal_key
        )
        best_vec = self._component_nats_vector(
            best_node.train_episode_metrics, rew_signal_key=rew_signal_key
        )
        if new_vec is None or best_vec is None:
            return None, ""

        def _delta(a: Optional[float], b: Optional[float]) -> float:
            return float(a - b) if (a is not None and b is not None) else 0.0

        d_obs = _delta(new_vec[0], best_vec[0])
        d_trans = _delta(new_vec[1], best_vec[1])
        d_rew = _delta(new_vec[2], best_vec[2])
        delta = float(self.pareto_delta)
        eps_large = float(self.pareto_eps_large)
        pareto_crit = (d_rew > 0.0 and d_obs > -delta and d_trans > -delta)
        # Opt-in 4th channel: pf_self_consistency (obs-only, belief
        # concentration). When enabled via
        # ``agent.use_pf_self_consistency_pareto=true`` the rule tightens:
        # the candidate must also not degrade belief concentration by
        # more than ``pareto_delta``. Silent no-op when the flag is off
        # or when the metric is missing on either side (e.g. the
        # scorer wasn't run with pf_self_consistency scoring_mode).
        d_bq: float = 0.0
        bq_on = bool(getattr(self, "use_pf_self_consistency_pareto", False))
        if bq_on:
            new_bq = self._pf_self_consistency_mean(
                new_node.train_episode_metrics
            )
            best_bq = self._pf_self_consistency_mean(
                best_node.train_episode_metrics
            )
            d_bq = _delta(new_bq, best_bq)
            pareto_crit = pareto_crit and (d_bq > -delta)
        # F5 — opt-in terminated Pareto channel (see `use_pareto_term_channel`).
        # Lower ``fivo_mean_term_nats`` is better (cross-entropy), so the
        # delta we want to penalise is ``new - best > delta`` (regressed).
        # Equivalent phrasing to the other channels: the "score" here is
        # ``-term_nats`` and we require ``Δ_score > -delta`` — identical
        # rule, flipped sign.
        d_term: float = 0.0
        term_on = bool(getattr(self, "use_pareto_term_channel", False))
        if term_on:
            new_term = self._mean_term_nats(new_node.train_episode_metrics)
            best_term = self._mean_term_nats(best_node.train_episode_metrics)
            if new_term is not None and best_term is not None:
                d_term = float(best_term - new_term)
                pareto_crit = pareto_crit and (d_term > -delta)
        fallback_crit = (combined_elbo - best_ever_score) > eps_large
        accept = bool(pareto_crit or fallback_crit)
        bq_tag = f" Δ_pfsc={d_bq:+.3f}" if bq_on else ""
        term_tag = f" Δ_term={d_term:+.3f}" if term_on else ""
        diag = (
            f"[PARETO] Δ_obs={d_obs:+.3f} Δ_trans={d_trans:+.3f} "
            f"Δ_rew={d_rew:+.3f}{bq_tag}{term_tag} "
            f"(crit={pareto_crit}, fallback={fallback_crit}) "
            f"→ {'ACCEPT' if accept else 'REJECT'}"
        )
        return accept, diag

    def _build_model_fit_section(
        self,
        valid_metrics: List[Dict[str, Any]],
        n_steps_total: int,
        n_events_total: int,
        rew_events_acc: float,
    ) -> List[str]:
        """Produce the new MODEL FIT feedback block (~50-70 lines).

        Concrete, action-localised, trace-based. Always asserts what
        currently works before flagging mismatches, so the LLM is biased
        toward minimal patches instead of full rewrites. Every field is
        belief-MAP-derived or buffer-observable — no state-oracle leak.
        """
        # ---- 1. Aggregate per-step decomp across episodes --------------
        all_steps: List[Dict[str, Any]] = []
        for ep_idx, m in enumerate(valid_metrics):
            for s in (m.get("fivo_per_step") or []):
                row = dict(s)
                row["ep_idx"] = ep_idx
                all_steps.append(row)

        def _img_match(s: Dict[str, Any]) -> bool:
            pred = s.get("pred_obs_image")
            obs = s.get("obs_image")
            return pred is not None and obs is not None and pred == obs

        def _rt_match(s: Dict[str, Any]) -> bool:
            # Only meaningful on event steps; on non-event steps default
            # is (0, False) and the reward_func usually trivially matches.
            if not (s.get("actual_reward_nonzero") or s.get("terminal_step")):
                return True
            return bool(s.get("reward_correct", False))

        n_img_match = sum(1 for s in all_steps if _img_match(s))
        n_total = len(all_steps) or 1
        n_events_correct = sum(
            1 for s in all_steps
            if (s.get("actual_reward_nonzero") or s.get("terminal_step"))
            and s.get("reward_correct", False)
        )

        # ---- 2. Header + couplage note ---------------------------------
        # When using event scoring, lead with event accuracy (the signal
        # that actually correlates with deployment reward) and demote obs
        # matching to a secondary note.
        event_steps = [
            s for s in all_steps
            if s.get("actual_reward_nonzero") or s.get("terminal_step")
        ]
        event_misses = [
            s for s in event_steps if not s.get("reward_correct", False)
        ]
        use_event_header = (
            getattr(self, "scoring_mode", "energy") == "event"
            and event_steps
        )

        lines: List[str] = ["MODEL FIT", "-" * 60]
        if use_event_header:
            lines.append(
                f"EVENT ACCURACY: {n_events_correct}/{len(event_steps)} "
                f"critical steps correct "
                f"({n_events_correct / max(len(event_steps), 1) * 100:.0f}%). "
                f"This is the PRIMARY metric."
            )
            if event_misses:
                lines.append("")
                lines.append("EVENT ERRORS (fix these FIRST):")
                for s in event_misses[:5]:
                    t = int(s.get("t", -1))
                    aname = self._resolve_action_name(s.get("action_id"))
                    pred_r = s.get("pred_reward")
                    pred_t = s.get("pred_terminated")
                    obs_r = s.get("observed_reward")
                    obs_t = s.get("observed_terminated")
                    obs_img = s.get("obs_image")
                    lines.append(
                        f"  ep={s.get('ep_idx')} step={t} action={aname}:"
                    )
                    lines.append(
                        f"    predicted: reward={pred_r}, terminated={pred_t}"
                    )
                    lines.append(
                        f"    actual:    reward={obs_r}, terminated={obs_t}"
                    )
                    if obs_img is not None:
                        lines.append(f"    observation at this step: {obs_img}")
            lines.append("")
            lines.append(
                f"(Observation match: {n_img_match}/{n_total} steps "
                f"— secondary metric, do not chase obs fixes at the "
                f"expense of event accuracy.)"
            )
        else:
            lines.append(
                f"Your code explains {n_img_match}/{n_total} steps "
                f"({n_img_match / n_total * 100:.1f}%) across "
                f"{len(valid_metrics)} episode(s)."
            )
            if not math.isnan(rew_events_acc):
                lines.append(
                    f"Reward + terminated correct on "
                    f"{n_events_correct}/{n_events_total} event step(s)."
                )
        lines.extend([
            "",
            "A mismatch is a JOINT symptom of initial × trans × obs",
            "through the particle filter — belief drift accumulates.",
            "It is NOT necessarily a bug in obs_func. Read each trace",
            "as a pipeline symptom, not a per-component verdict.",
            "",
        ])

        # ---- 3. BY ACTION breakdown ------------------------------------
        action_counts: Dict[int, List[int]] = {}  # action_id -> [n_match, n_total]
        for s in all_steps:
            aid = s.get("action_id")
            if aid is None:
                continue
            slot = action_counts.setdefault(int(aid), [0, 0])
            slot[1] += 1
            if _img_match(s) and _rt_match(s):
                slot[0] += 1
        if action_counts:
            lines.append("BY ACTION (full match = obs AND reward/term):")
            for aid in sorted(action_counts):
                ok, total = action_counts[aid]
                name = self._resolve_action_name(aid)
                pct = ok / max(total, 1) * 100
                marker = "✓" if ok == total else (
                    "← focus" if pct < 95 else ""
                )
                lines.append(
                    f"  {name:8s} {ok:3d}/{total:3d} ({pct:3.0f}%)  {marker}"
                )
            lines.append("")

        # ---- 4. MATCHES — small sample, do not regress -----------------
        match_samples = [
            s for s in all_steps
            if _img_match(s) and _rt_match(s) and s.get("obs_image") is not None
        ]
        if match_samples:
            # Spread across actions so the LLM sees variety, not 4× same.
            seen_actions: set = set()
            picked: List[Dict[str, Any]] = []
            for s in match_samples:
                aid = s.get("action_id")
                if aid in seen_actions and len(picked) >= 2:
                    continue
                seen_actions.add(aid)
                picked.append(s)
                if len(picked) >= 4:
                    break
            lines.append("MATCHES (sample, do NOT regress these):")
            for s in picked:
                aname = self._resolve_action_name(s.get("action_id"))
                obs = s.get("obs_image")
                lines.append(
                    f"  ep={s['ep_idx']} s={int(s['t']):2d}  {aname:7s}  "
                    f"obs={obs}  ✓"
                )
            lines.append("")

        # ---- 5. MISMATCHES — top-K worst, with 7-step traces -----------
        # Severity = event mismatch first, then by image distance.
        def _severity(s: Dict[str, Any]) -> Tuple[int, float]:
            event_mismatch = (
                (s.get("actual_reward_nonzero") or s.get("terminal_step"))
                and not s.get("reward_correct", False)
            )
            return (
                0 if event_mismatch else 1,
                s.get("score_obs", 0.0),  # less negative = better
            )

        mismatches = [
            s for s in all_steps
            if not (_img_match(s) and _rt_match(s))
        ]
        mismatches.sort(key=_severity)
        worst = mismatches[:3]

        if worst:
            # Pre-index per-episode steps for window slicing.
            steps_by_ep: Dict[int, List[Dict[str, Any]]] = {}
            for s in all_steps:
                steps_by_ep.setdefault(s["ep_idx"], []).append(s)
            for ep_idx in steps_by_ep:
                steps_by_ep[ep_idx].sort(key=lambda r: int(r["t"]))

            lines.append(
                f"MISMATCHES ({len(worst)} worst, ±3-step traces — "
                "watch where belief drifts):"
            )
            for ws in worst:
                ep_idx = ws["ep_idx"]
                t_center = int(ws["t"])
                ep_steps = steps_by_ep.get(ep_idx, [])
                t_min = max(0, t_center - 3)
                t_max = t_center + 3
                window = [s for s in ep_steps if t_min <= int(s["t"]) <= t_max]
                lines.append("")
                lines.append(
                    f"  ep={ep_idx} window {t_min}..{t_max}"
                )
                for entry in window:
                    t = int(entry["t"])
                    aname = self._resolve_action_name(entry.get("action_id"))
                    img_ok = _img_match(entry)
                    rt_ok = _rt_match(entry)
                    if img_ok and rt_ok:
                        lines.append(
                            f"    s{t:2d}  {aname:7s}  pred/obs match  ✓"
                        )
                        continue
                    pred = entry.get("pred_obs_image")
                    obs = entry.get("obs_image")
                    lines.append(
                        f"    s{t:2d}  {aname:7s}  pred={pred}"
                    )
                    lines.append(f"               obs= {obs}")
                    if not rt_ok:
                        pred_rt = (
                            entry.get("pred_reward"),
                            entry.get("pred_terminated"),
                        )
                        obs_rt = (
                            entry.get("observed_reward"),
                            entry.get("observed_terminated"),
                        )
                        lines.append(
                            f"               pred r/t={pred_rt}  "
                            f"obs r/t={obs_rt}"
                        )
            lines.append("")

        # ---- 6. Note incertitude ---------------------------------------
        lines.extend([
            "Some mismatches are structural to POMDP inference — even a",
            "perfect model cannot always resolve belief drift without",
            "more observations. If a mismatch persists across 2+ targeted",
            "fixes, accept it rather than distort working branches to",
            "chase it.",
        ])
        return lines

    def joint_update_models_rex(
        self,
        train_replay_buffer: ReplayBuffer,
        test_replay_buffer: ReplayBuffer,
        active_joint_models: Optional[List[str]] = None,
        rex_C: float = 20,
        num_model_attempts: int = 5,
        episode: int = 0,
    ) -> Tuple[RexNode, int]:
        if active_joint_models is None:
            active_joint_models = self._active_joint_models()
        if not active_joint_models:
            raise ValueError("joint_update_models_rex called with no active joint models.")
        active_function_names = self._format_joint_function_names(active_joint_models)
        log.info(
            f"Updating models using REx jointly for: {active_function_names}"
        )

        elbo_evaluator = self._make_elbo_evaluator()

        # Evaluate initial models using the unified scorer (E1: ELBO or coverage)
        (train_elbo, train_actions, train_observations, train_previous_observations,
         train_rewards, train_terminated, train_episode_metrics, train_error
        ) = self._score_models(elbo_evaluator, train_replay_buffer, active_joint_models)

        (test_elbo, _, test_observations, _,
         _, _, test_episode_metrics, test_error
        ) = self._score_models(elbo_evaluator, test_replay_buffer, active_joint_models)

        # These are from the initial (default) model, so they should be fine.
        assert train_error is None
        assert test_error is None
        to_update = True

        # Convergence test: track best likelihood and consecutive no-improvement count.
        # Patience and improvement threshold are Hydra-configurable via
        # `convergence_patience` / `improvement_eps` on the agent; with
        # `None`, patience defaults to max(3, num_attempts // 4).
        best_combined_elbo = (train_elbo + test_elbo) / 2
        self.previous_elbo = best_combined_elbo
        no_improvement_count = 0
        convergence_patience = (
            self.convergence_patience
            if self.convergence_patience is not None
            else max(3, num_model_attempts // 4)
        )
        improvement_eps = self.improvement_eps

        # M8: Cache of identical LLM-generated code → (train_elbo, test_elbo).
        # Skips redundant PF evaluations when the LLM repeats a snippet
        # during the same REx run. Scoped to this call only (per-run cache).
        score_cache: Dict[str, Tuple[float, float]] = {}
        cache_hits = 0

        # Create the initial RexNode.
        # If the agent already has transition code (e.g. from offline REx or a
        # prior online update), seed the initial node with it so that the first
        # REx iteration sends a REFINEMENT prompt instead of starting from scratch.
        seed_code = self._seed_joint_code(active_joint_models)

        initial_node = RexNode(
            to_update=to_update,
            train_coverage=train_elbo,
            test_coverage=test_elbo,
            train_empirical_dist=train_observations,
            train_model_dist=None,
            test_empirical_dist=test_observations,
            test_model_dist=None,
            alpha=1,
            beta=1,
            depth=0,
            iter_num=-1,
            previous_code=seed_code,
            train_actions=train_actions,
            train_previous_observations=train_previous_observations,
            train_rewards=train_rewards,
            train_terminated=train_terminated,
            train_episode_metrics=train_episode_metrics,
        )
        # Use a set to store generated nodes.
        generated_nodes: Set[RexNode] = set()
        generated_nodes.add(initial_node)

        # Monotone bound: track the best node ever seen.  The final model is
        # guaranteed to be at least as good as the initial model — score(t+1)
        # >= score(t) by construction, giving a non-decreasing sequence that
        # converges to a local optimum.
        best_ever_score = best_combined_elbo
        best_ever_node: Optional[RexNode] = initial_node
        monotone_trace: List[Tuple[int, float]] = [(-1, best_ever_score)]
        log.info(f"[MONOTONE] Initial bound: {best_ever_score:.4f}")

        # Maintain a parent mapping from child RexNode to parent RexNode.
        parent_mapping: Dict[RexNode, RexNode] = {}

        # QBC: persist candidates across offline→online updates for continuity.
        # If we already have candidates from a prior REx run, keep them.
        _qbc_candidates: List[Tuple[str, Any]] = getattr(self, "_qbc_candidates", [])
        if not _qbc_candidates:
            self._qbc_feedback = ""

        # I_MULTI_INIT: the first ``num_initial_candidates`` REx steps
        # are the "initialisation phase" — each step samples a fresh
        # candidate from the bootstrap root, optionally with a
        # distinct temperature so the K samples do not collapse onto a
        # consensus. Default ``num_initial_candidates=1`` gives exactly
        # one init step, identical to the pre-flag behaviour.
        num_init = int(getattr(self, "num_initial_candidates", 1))
        init_temps = getattr(self, "initial_sampling_temperatures", None)
        if num_init > 1:
            log.info(
                "[MULTI_INIT] Running %d initial candidates from the root "
                "(temperatures=%s).",
                num_init,
                init_temps if init_temps else "default",
            )

        for iter_num in range(num_model_attempts):
            log.info("-" * 20)
            log.info(f"REx Step {iter_num}: Attempting to generate new code")
            self.rex_iter = iter_num

            # I_FORCE_REFINING (opt-in): once the initialisation phase
            # has completed (i.e. ``num_init`` candidates have been
            # sampled from the root), freeze the root by clearing its
            # ``to_update`` flag. The Thompson / UCB1 sampler multiplies
            # each draw by ``int(node.to_update)``, so the root is then
            # ignored. This biases subsequent parent selections toward
            # refining-path candidates where the smallest_edit /
            # term_hint signals reach the LLM as edits.
            if (
                getattr(self, "force_refining_after_iter0", False)
                and iter_num >= num_init
                and any(node.iter_num >= 0 for node in generated_nodes)
                and initial_node.to_update
            ):
                initial_node.to_update = False
                log.info(
                    "[I_FORCE_REFINING] Froze root node (iter=-1) at REx Step %d "
                    "to bias sampler toward refining path.",
                    iter_num,
                )

            # Parent selection — Thompson sampling (Thompson 1933):
            # draw ONE sample θ_n ~ Beta(α_n, β_n) per active node and
            # pull argmax_n θ_n. This is the textbook Bernoulli-bandit
            # formulation — it is NOT greedy: the argmax is over random
            # draws, so exploration is proportional to posterior
            # uncertainty. Inactive nodes (to_update == False) are
            # masked out by multiplying their draw by 0.
            to_update_nodes = [node for node in generated_nodes if node.to_update]
            if not to_update_nodes:
                # All active nodes have been frozen (to_update=False) —
                # typically because the feedback branch in
                # ``_joint_update_model`` decided the model already
                # has full support. Stop the REx loop cleanly instead
                # of letting ``max()`` raise ``ValueError: empty``.
                log.info(
                    "[REx] No active nodes left to update at iter %d, "
                    "breaking REx loop.", iter_num,
                )
                break
            # During the multi-init phase, short-circuit the bandit and
            # always expand from the bootstrap root so the K initial
            # candidates are genuine resamples of iter 0, not a
            # refinement of the first one. Once ``iter_num >= num_init``,
            # the standard UCB1 / Thompson path resumes.
            in_init_phase = iter_num < num_init
            if in_init_phase and initial_node in to_update_nodes:
                parent_node = initial_node
                log.info(
                    "[MULTI_INIT] Step %d/%d — sampling from root (iter=-1).",
                    iter_num, num_init,
                )
            elif getattr(self, "use_ucb1_selection", False):
                parent_node = ucb1_select(
                    to_update_nodes,
                    c=float(getattr(self, "ucb1_c", 1.0)),
                    active_key=lambda n: bool(n.to_update),
                    value_key=lambda n: (
                        (float(n.alpha) - 1.0)
                        / max(float(n.alpha) + float(n.beta) - 2.0, 1.0)
                    ),
                    visit_key=lambda n: max(
                        int(n.alpha) + int(n.beta) - 2, 0
                    ),
                ) or to_update_nodes[0]
                log.info(
                    "[UCB1] picked node iter=%d depth=%d α=%.1f β=%.1f "
                    "(n_visit=%d, Q̂≈%.3f, active=%d/%d)",
                    parent_node.iter_num, parent_node.depth,
                    parent_node.alpha, parent_node.beta,
                    max(int(parent_node.alpha) + int(parent_node.beta) - 2, 0),
                    (float(parent_node.alpha) - 1.0)
                    / max(float(parent_node.alpha) + float(parent_node.beta) - 2.0, 1.0),
                    len(to_update_nodes), len(generated_nodes),
                )
            else:
                parent_node = max(
                    to_update_nodes,
                    key=lambda node: (
                        np.random.beta(node.alpha, node.beta) * int(node.to_update)
                    ),
                )
                log.info(
                    "[THOMPSON] picked node iter=%d depth=%d α=%.1f β=%.1f "
                    "(active=%d/%d)",
                    parent_node.iter_num, parent_node.depth,
                    parent_node.alpha, parent_node.beta,
                    len(to_update_nodes), len(generated_nodes),
                )
            log.info(f"Sampling new code:\n{parent_node!r}")

            # Temperature override for the multi-init phase: if the
            # agent was configured with a list of ``initial_sampling_temperatures``,
            # pick ``init_temps[iter_num]`` so candidate k samples at a
            # distinct temperature from candidate k-1 — avoids the
            # observed collapse to consensus when K identical calls
            # are fired at the default temperature.
            iter_temperature_override: Optional[float] = None
            if in_init_phase and init_temps:
                iter_temperature_override = float(
                    init_temps[iter_num % len(init_temps)]
                )
                log.info(
                    "[MULTI_INIT] Candidate %d sampled with temperature=%.2f",
                    iter_num, iter_temperature_override,
                )

            error = None
            # Retry generation when produced code does not execute; feed runtime
            # errors back to the model as iterative feedback.
            for attempt in range(self.num_bug_fix_attempts):
                log.info(f"Attempt {attempt} to generate valid code from sampled node")
                to_update, new_messages, model_code = self._joint_update_model(
                    iter_num,
                    attempt,
                    train_replay_buffer,
                    parent_node,
                    active_joint_models=active_joint_models,
                    error=error,
                    episode=episode,
                    temperature_override=iter_temperature_override,
                )
                parent_node.messages = new_messages

                # Rebind evaluator models to the latest generated code before scoring.
                self._sync_evaluator_models(elbo_evaluator, active_joint_models)

                # M8: short-circuit PF evaluation when the LLM repeats a snippet
                # we already scored in this REx run. Keyed on the raw code string
                # (deterministic for a given LLM output).
                cache_key = model_code if isinstance(model_code, str) else ""
                cached = score_cache.get(cache_key) if cache_key else None
                if cached is not None:
                    (train_elbo, train_actions, train_observations,
                     train_previous_observations, train_rewards, train_terminated,
                     train_episode_metrics, train_error,
                     test_elbo, test_observations, test_episode_metrics,
                     test_error) = cached
                    cache_hits += 1
                    log.info(
                        f"[CACHE] hit #{cache_hits} — reusing score "
                        f"train={train_elbo:.4f} test={test_elbo:.4f}"
                    )
                else:
                    (train_elbo, train_actions, train_observations,
                     train_previous_observations, train_rewards, train_terminated,
                     train_episode_metrics, train_error
                     ) = self._score_models(elbo_evaluator, train_replay_buffer, active_joint_models)

                    (test_elbo, _, test_observations, _,
                     _, _, test_episode_metrics, test_error
                     ) = self._score_models(elbo_evaluator, test_replay_buffer, active_joint_models)

                    if cache_key and train_error is None and test_error is None:
                        score_cache[cache_key] = (
                            train_elbo, train_actions, train_observations,
                            train_previous_observations, train_rewards, train_terminated,
                            train_episode_metrics, train_error,
                            test_elbo, test_observations, test_episode_metrics,
                            test_error,
                        )

                if train_error is not None or test_error is not None:
                    error = (train_error or "") + "\n" + (test_error or "")
                else:
                    break
            else:
                log.info("Failed to generate valid code after maximum attempts.")
                continue

            # QBC: store this iteration's transition model for disagreement
            # detection and compute vote entropy across candidates so far.
            # Side effects (self._qbc_candidates / self._qbc_feedback) are
            # preserved exactly — see _run_qbc_probe for the full sequence.
            self._run_qbc_probe(
                iter_num=iter_num,
                train_replay_buffer=train_replay_buffer,
                active_joint_models=active_joint_models,
                qbc_candidates=_qbc_candidates,
            )

            combined_elbo = (train_elbo + test_elbo) / 2

            # Event gate: reject candidates that reproduce 0% of critical
            # events (reward, terminated) on the TEST set. Uses the
            # per_step_decomp already computed by the PF evaluation —
            # no extra rollouts needed.  Checks ``reward_correct`` on
            # steps where ``actual_reward_nonzero`` or ``terminal_step``
            # is True.
            event_gate_pass = True
            if getattr(self, "event_gate", False) and test_episode_metrics:
                _n_events = 0
                _n_correct = 0
                _n_steps_total = 0
                _n_eps_with_decomp = 0
                for m in test_episode_metrics:
                    steps = m.get("fivo_per_step") or []
                    _n_steps_total += len(steps)
                    if steps:
                        _n_eps_with_decomp += 1
                    for d in steps:
                        if d.get("actual_reward_nonzero") or d.get("terminal_step"):
                            _n_events += 1
                            if d.get("reward_correct", False):
                                _n_correct += 1
                log.info(
                    f"[EVENT_GATE] debug: {_n_eps_with_decomp}/{len(test_episode_metrics)} eps have decomp, "
                    f"{_n_steps_total} total steps, {_n_events} event steps"
                )
                _ev_rate = _n_correct / max(_n_events, 1)
                log.info(
                    f"[EVENT_GATE] test events: {_n_correct}/{_n_events} "
                    f"({_ev_rate:.0%}) "
                    f"{'PASS' if _n_correct > 0 else 'REJECT'}"
                )
                event_gate_pass = _n_correct > 0

            # M2: continuous Beta-Bernoulli update for the Thompson-sampled
            # parent node — see ``_apply_continuous_beta_update`` for the
            # soft-success formula, saturation behaviour and α/β cap.
            self._apply_continuous_beta_update(
                node=parent_node,
                child_combined_elbo=combined_elbo,
                improvement_eps=improvement_eps,
            )

            # Create a new RexNode for the generated code.
            new_node = RexNode(
                to_update=to_update,
                train_coverage=train_elbo,
                test_coverage=test_elbo,
                train_empirical_dist=train_observations,
                train_model_dist=None,
                test_empirical_dist=test_observations,
                test_model_dist=None,
                alpha=1,
                beta=1,
                depth=parent_node.depth + 1,
                iter_num=iter_num,
                messages=new_messages,
                previous_code=model_code,  # Store the previous code for feedback
                train_actions=train_actions,
                train_previous_observations=train_previous_observations,
                train_rewards=train_rewards,
                train_terminated=train_terminated,
                train_episode_metrics=train_episode_metrics,
            )
            # Add the new node and record its parent.
            generated_nodes.add(new_node)
            parent_mapping[new_node] = parent_node
            log.info(f"New node generated:\n{new_node!r}")

            # Monotone bound update: only replace best if strictly better.
            # I_PARETO (opt-in): when ``use_pareto_selector`` is true, the
            # scalar comparator below is *overridden* by a per-component
            # rule that accepts candidates which improve reward_func
            # without regressing obs/trans — see ``_pareto_accept``.
            # ``pareto_decision`` is ``None`` when the flag is off or the
            # selector cannot form an opinion, in which case the scalar
            # monotone rule applies unchanged (byte-identical).
            pareto_decision, pareto_msg = self._pareto_accept(
                new_node=new_node,
                best_node=best_ever_node,
                combined_elbo=combined_elbo,
                best_ever_score=best_ever_score,
            )
            if pareto_msg:
                log.info(pareto_msg)
            if not event_gate_pass:
                accept = False
                log.info("[EVENT_GATE] Candidate rejected — 0%% events on test set.")
            elif pareto_decision is not None:
                accept = pareto_decision
            else:
                accept = combined_elbo > best_ever_score + improvement_eps
            if accept:
                best_ever_score = combined_elbo
                best_ever_node = new_node
                monotone_trace.append((iter_num, best_ever_score))
                log.info(
                    f"[MONOTONE] ↑ New best at iter {iter_num}: "
                    f"{best_ever_score:.4f}"
                )
            else:
                log.info(
                    f"[MONOTONE] = Kept best: {best_ever_score:.4f} "
                    f"(candidate: {combined_elbo:.4f}, Δ={combined_elbo - best_ever_score:.4f})"
                )

            # Convergence test: break if no improvement in the last 3 attempts.
            if combined_elbo > best_combined_elbo:
                best_combined_elbo = combined_elbo
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= convergence_patience:
                    log.info(
                        f"Model Update Step {self.rex_iter}: No improvement in combined likelihood for {convergence_patience} consecutive attempts (best: {best_combined_elbo:.4f}). Converged."
                    )
                    break

            self.reset()

            # Optionally, you can now save the current Rex tree to an HTML file.
            # (See the pyvis snippet below for saving.)

            # save_rex_tree_html(
            #     model_name,
            #     list(generated_nodes),
            #     parent_mapping,
            #     get_log_dir(),
            #     iter_num,
            # )

        # Monotone selection: use the tracked best-ever node instead of
        # re-scanning all nodes.  This guarantees the output is at least as
        # good as the initial model (non-decreasing score sequence).
        best_node = best_ever_node if best_ever_node is not None else max(
            generated_nodes, key=lambda node: (node.test_coverage, node.train_coverage)
        )

        # Opt-in softmax sampling over ELBOs of generated nodes.
        # Treats `(train_coverage+test_coverage)/2`
        # as logits. At T → 0 this converges to the argmax above;
        # at T > 0 we draw ~ Categorical(softmax(score / T)), giving
        # the selector a non-zero chance to pick a candidate that is
        # structurally different but still near the best.
        #
        # Monotone floor: restrict the pool to
        # candidates whose score is within ``epsilon`` of ``best_ever_score``
        # where ``epsilon = 1 std`` of the round's scores (classic stats
        # window, fixed per round, self-contained). Without this filter
        # softmax would override MONOTONE by sampling an arbitrarily bad
        # node, breaking the non-decreasing-score guarantee. With the
        # floor we keep MONOTONE's lower bound while allowing exploration
        # between candidates that are statistically indistinguishable
        # from the best.
        softmax_T = float(getattr(self, "elbo_softmax_temperature", 0.0))
        if softmax_T > 0.0:
            candidates = [
                n for n in generated_nodes if n.previous_code is not None
            ]
            if len(candidates) >= 2:
                all_scores = np.array(
                    [
                        (float(n.train_coverage) + float(n.test_coverage)) / 2.0
                        for n in candidates
                    ]
                )
                epsilon = float(np.std(all_scores)) if len(all_scores) >= 2 else 0.0
                floor = float(best_ever_score) - epsilon
                near_best = [
                    n for n in candidates
                    if (float(n.train_coverage) + float(n.test_coverage)) / 2.0
                    >= floor
                ]
                if len(near_best) >= 2:
                    scores = np.array(
                        [
                            (float(n.train_coverage) + float(n.test_coverage)) / 2.0
                            for n in near_best
                        ]
                    )
                    stable_scores = scores - float(scores.max())
                    probs = np.exp(stable_scores / softmax_T)
                    probs = probs / float(probs.sum())
                    idx = int(np.random.choice(len(near_best), p=probs))
                    sampled = near_best[idx]
                    argmax_score = float(
                        (best_node.train_coverage + best_node.test_coverage) / 2.0
                    )
                    sampled_score = float(
                        (sampled.train_coverage + sampled.test_coverage) / 2.0
                    )
                    log.info(
                        f"[SOFTMAX_SAMPLE] T={softmax_T} eps(std)={epsilon:.3f} "
                        f"floor={floor:+.3f} pool={len(near_best)}/{len(candidates)} "
                        f"picked iter={sampled.iter_num} "
                        f"(score={sampled_score:+.3f}, prob={probs[idx]:.3f}); "
                        f"argmax iter={best_node.iter_num} "
                        f"score={argmax_score:+.3f} "
                        f"(gap={argmax_score - sampled_score:+.3f})"
                    )
                    best_node = sampled
                else:
                    log.info(
                        f"[SOFTMAX_SAMPLE] skipped — only {len(near_best)} "
                        f"candidate(s) within epsilon={epsilon:.3f} of "
                        f"best_ever={best_ever_score:.3f}; kept MONOTONE argmax."
                    )
        # Convergence summary
        n_improvements = len(monotone_trace) - 1  # exclude initial
        all_scores = [(n.iter_num, (n.train_coverage + n.test_coverage) / 2)
                      for n in generated_nodes]
        all_scores.sort(key=lambda x: x[0])
        log.info(
            f"Best code found after REx iterations:\n{best_node!r}\n"
            f"[MONOTONE] Trace: {monotone_trace}\n"
            f"[CONVERGENCE] improvements={n_improvements}/{len(generated_nodes)-1}, "
            f"initial={monotone_trace[0][1]:.4f}, final={best_ever_score:.4f}, "
            f"delta={best_ever_score - monotone_trace[0][1]:.4f}, "
            f"patience_used={no_improvement_count}/{convergence_patience}, "
            f"qbc_candidates={len(getattr(self, '_qbc_candidates', []))}, "
            f"qbc_feedback={'yes' if getattr(self, '_qbc_feedback', '') else 'no'}"
        )

        # ------------------------------------------------------------------ #
        # E2: Model Averaging via Thompson Sampling                          #
        #                                                                    #
        # When model_averaging_top_k > 1, keep the top-K nodes and build    #
        # stochastic wrapper models.  At each call the wrapper samples one   #
        # of the K models with probability ∝ exp(score_k).                  #
        #                                                                    #
        # Ref: Hoeting et al. (1999) "Bayesian Model Averaging"             #
        #      Thompson (1933) "On the likelihood that one unknown           #
        #      probability exceeds another"                                  #
        # ------------------------------------------------------------------ #
        K = self.model_averaging_top_k
        if K > 1 and len(generated_nodes) > 1:
            sorted_nodes = sorted(
                [n for n in generated_nodes if n.previous_code is not None],
                key=lambda n: (n.test_coverage, n.train_coverage),
                reverse=True,
            )[:K]
            if len(sorted_nodes) >= 2:
                log.info(f"[E2] Building ensemble from top-{len(sorted_nodes)} models")
                self._install_ensemble_models(
                    sorted_nodes, active_joint_models
                )
                return best_node, len(generated_nodes)

        # Default path: restore single best node's models into the planner.
        if best_node.previous_code is not None:
            try:
                local_scope: Dict[str, Any] = {}
                code_obj = compile(best_node.previous_code, filename="<rex_best>", mode="exec")
                import uncertain_worms.policies.base_policy as _bp
                exec(code_obj, vars(_bp), local_scope)
                assert self.type_tuple is not None
                for model_name in active_joint_models:
                    func_name = self.JOINT_MODEL_TO_FUNC[model_name]
                    if func_name in local_scope:
                        new_model = self.model_translators[model_name](
                            func_name, local_scope, type_tuple=self.type_tuple
                        )
                        self._set_model(model_name, new_model)
                # Persist code text for online continuity: the next REx run
                # can seed its initial node with this code (refinement prompt
                # instead of starting from scratch).
                for model_name in active_joint_models:
                    self.current_code[model_name] = best_node.previous_code
                log.info("[REx] Restored best node's models into planner.")
            except Exception:
                log.info(f"[REx] Failed to restore best node's code: {traceback.format_exc()}")

        return best_node, len(generated_nodes)

    # ------------------------------------------------------------------ #
    # E2: Ensemble model installation                                    #
    # ------------------------------------------------------------------ #

    def _install_ensemble_models(
        self,
        nodes: List["RexNode"],
        active_joint_models: List[str],
    ) -> None:
        """Compile top-K nodes and install Thompson-sampling wrappers.

        Each wrapper holds K callables + softmax weights.  On every call
        it randomly selects one model ∝ exp(score_k), providing implicit
        Bayesian Model Averaging through posterior sampling.
        """
        import uncertain_worms.policies.base_policy as _bp

        # Compile all K nodes into callable models
        per_model: Dict[str, List[Any]] = {m: [] for m in active_joint_models}
        scores: List[float] = []
        for node in nodes:
            if node.previous_code is None:
                continue
            try:
                ls: Dict[str, Any] = {}
                code_obj = compile(node.previous_code, filename="<rex_ensemble>", mode="exec")
                exec(code_obj, vars(_bp), ls)
                assert self.type_tuple is not None
                for model_name in active_joint_models:
                    func_name = self.JOINT_MODEL_TO_FUNC[model_name]
                    if func_name in ls:
                        model = self.model_translators[model_name](
                            func_name, ls, type_tuple=self.type_tuple
                        )
                        per_model[model_name].append(model)
                scores.append((node.test_coverage + node.train_coverage) / 2)
            except Exception:
                log.info(f"[E2] Failed to compile node: {traceback.format_exc()}")

        if len(scores) < 2:
            log.info("[E2] Not enough compiled models, falling back to single best.")
            return

        # Softmax weights: P(k) ∝ exp(score_k)
        scores_arr = np.array(scores, dtype=float)
        scores_arr -= scores_arr.max()  # numerical stability
        weights = np.exp(scores_arr)
        weights /= weights.sum()
        log.info(f"[E2] Ensemble weights: {dict(zip(range(len(weights)), [f'{w:.3f}' for w in weights]))}")

        # Install Thompson-sampling wrappers
        for model_name in active_joint_models:
            models = per_model[model_name]
            if len(models) == len(weights):
                wrapper = self._ThompsonSamplingModelInner(models, weights)
                self._set_model(model_name, wrapper)
                log.info(f"[E2] Installed ensemble for {model_name} (K={len(models)})")


    class _ThompsonSamplingModelInner:
        """Wrapper that samples one of K models ∝ weight on each call.

        Implements posterior-sampling Bayesian Model Averaging: on every
        invocation a model index is drawn from the softmax weights and
        the corresponding callable is delegated to with the caller's
        arguments. The wrapper is duck-typed to match the signature of
        whichever model it replaces (transition / observation / reward)
        — hence the untyped ``*args, **kwargs`` and ``Any`` return.

        Refs: Hoeting et al. (1999), Thompson (1933).
        """
        __slots__ = ("_models", "_weights")

        def __init__(self, models: List[Any], weights: np.ndarray) -> None:
            self._models = models
            self._weights = weights

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            idx = np.random.choice(len(self._models), p=self._weights)
            return self._models[idx](*args, **kwargs)

    def _sfs_query_directions(
        self,
        current_code: Optional[str],
        model_fit_feedback: str,
        direction_history: "DirectionHistory",
        k: int = 3,
    ) -> List[str]:
        if not needs_llm_scattering(current_code):
            directions = get_first_iter_directions(k=k)
            log.info(f"[SFS] First iteration, using {k} generic directions")
            return directions
        assert current_code is not None
        scattering_prompt = build_scattering_prompt(
            current_code=current_code,
            model_fit_feedback=model_fit_feedback,
            history=direction_history,
            k=k,
        )
        messages = [{"role": "user", "content": scattering_prompt}]
        log.info(f"[SFS] Querying LLM for {k} improvement directions")
        response = self.gpt_joint_update_model(messages)
        directions = parse_directions(response, k=k)
        for i, d in enumerate(directions):
            log.info(f"[SFS] Direction {i+1}/{k}: {d}")
        return directions

    def _joint_update_model(
        self,
        iter_num: int,
        exec_attempt: int,
        replay_buffer: ReplayBuffer,
        rex_node: RexNode,
        active_joint_models: List[str],
        error: Optional[str] = None,
        episode: int = 0,
        temperature_override: Optional[float] = None,
        sfs_direction: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, str]], Optional[str]]:
        self.init_types(replay_buffer)

        messages = copy.deepcopy(rex_node.messages)
        to_update = True
        if rex_node.to_update:
            active_function_names = self._format_joint_function_names(active_joint_models)
            log.info(f"Jointly updating models: {active_function_names}")

            # Rank the probabilities of empirical distribution under the model
            if error is not None:
                log.info(
                    f"[Joint] Episode {episode} Iter {iter_num}, step {self.rex_iter} bug: {error}"
                )

                messages.append({"role": "user", "content": error})
                code_str = self.gpt_joint_update_model(
                    messages,
                    active_joint_models=active_joint_models,
                    iter_num=iter_num,
                    exec_attempt=exec_attempt,
                    episode=episode,
                    temperature_override=temperature_override,
                )
                if code_str is None:
                    log.warning("[Joint] All code gen attempts failed on error fix, keeping previous code")
                    to_update = False
                    code_str = rex_node.previous_code
                    messages = copy.deepcopy(rex_node.messages)
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": code_str,
                        }
                    )
            else:
                log.info(f"[Joint] no execution errors")
                prompt: Optional[str] = None
                if rex_node.previous_code is None:
                    log.info(f"[Joint] First model update")
                    prompt = self.get_joint_starting_prompt(
                        observations=rex_node.train_empirical_dist,
                        active_joint_models=active_joint_models,
                        previous_observations=rex_node.train_previous_observations,
                        actions=rex_node.train_actions,
                        rewards=rex_node.train_rewards,
                        terminated=rex_node.train_terminated,
                    )
                else:
                    log.info(f"[Joint] Refining model")
                    prompt = self.get_joint_feedback_prompt(
                        previous_code=rex_node.previous_code,
                        active_joint_models=active_joint_models,
                        model_dist=rex_node.train_model_dist,
                        empirical_dist=rex_node.train_empirical_dist,
                        episode_metrics=rex_node.train_episode_metrics,
                        observations=rex_node.train_empirical_dist,
                        actions=rex_node.train_actions,
                        previous_observations=rex_node.train_previous_observations,
                        rewards=rex_node.train_rewards,
                        terminated=rex_node.train_terminated,
                    )

                if prompt is not None:
                    if sfs_direction is not None:
                        log.info(f"[SFS] Injecting direction: {sfs_direction[:80]}...")
                        prompt = format_direction_prompt(prompt, sfs_direction)
                    log.info(f"[Joint] Prompting for feedback")
                    messages = [{"role": "system", "content": prompt}]
                    code_str = self.gpt_joint_update_model(
                        messages,
                        active_joint_models=active_joint_models,
                        iter_num=iter_num,
                        exec_attempt=exec_attempt,
                        episode=episode,
                    )
                    if code_str is None:
                        log.warning("[Joint] All code gen attempts failed, keeping previous code")
                        to_update = False
                        code_str = rex_node.previous_code
                        messages = copy.deepcopy(rex_node.messages)
                    else:
                        # Preserve the LLM's response in conversation history
                        # so the next iteration has full context.
                        messages.append(
                            {"role": "assistant", "content": code_str}
                        )
                else:
                    log.info(f"[Joint] No feedback needed")
                    # Model has full support, no need to update
                    to_update = False
                    code_str = rex_node.previous_code
                    messages = copy.deepcopy(rex_node.messages)

        return to_update, messages, code_str

    def _get_model(self, model_name: str) -> Dict[str, Any]:
        return self.planner.__getattribute__(model_name)

    def _set_model(self, model_name: str, model: Any) -> None:
        self.planner.__setattr__(model_name, model)
