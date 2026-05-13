"""Query By Committee for transition model candidates.

Given N candidate transition models from REx iterations, identifies
(observation, action) contexts where models DISAGREE about what happens.
This reveals epistemic uncertainty that trajectory-based scoring misses.

Key insight: a model that allows "forward onto key" scores well on
trajectory data (more particles survive), but disagrees with models
that correctly block it.  The disagreement signal tells the LLM WHERE
its models are uncertain without encoding domain knowledge.

Two components:
  1. DisagreementDetector — finds disagreements across REx candidates
  2. Formatted feedback for the LLM refinement prompt

Reference: Query By Committee (Seung et al. 1992)
"""

from __future__ import annotations

import copy
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from uncertain_worms.structs import (
    ActType,
    ObservationModel,
    ObsType,
    ParticleBelief,
    StateType,
    TransitionModel,
)

log = logging.getLogger(__name__)


@dataclass
class Disagreement:
    """A (step, action) context where candidate models predict different next states."""

    step: int  # trajectory step index
    action: int  # the alternative action tested
    obs_context: Any  # observation image at that step (list)
    action_taken: int  # action actually taken in trajectory
    n_move: int  # n particles where models DISAGREE on outcome
    n_stay: int  # n particles where models AGREE on outcome
    n_total: int  # total particles tested


@dataclass
class DisagreementResult:
    """Aggregated disagreements across trajectory steps.

    Uses the Vote Entropy criterion of Dagan & Engelson (1995) — the
    standard query-by-committee uncertainty measure in active learning
    (see Settles, "Active Learning Literature Survey", 2009 §3.1).
    For each input x, the committee of N models produces N predictions.
    Let V(y, x) be the number of members predicting label y. Then

        VE(x) = - Σ_y (V(y,x)/N) * log(V(y,x)/N)

    Normalised by log(N) so the result is in [0, 1] independently of
    committee size: 0 = full consensus, 1 = maximum disagreement.
    """

    disagreements: List[Disagreement] = field(default_factory=list)
    total_comparisons: int = 0
    # Per-context vote entropies (already normalised in [0,1]).
    # Filled by detect_from_trajectory; stays empty for legacy calls.
    vote_entropies: List[float] = field(default_factory=list)

    @property
    def mean_vote_entropy(self) -> float:
        """Average normalised vote entropy across all probed contexts.

        Standard QBC uncertainty score from Dagan & Engelson (1995),
        Settles (2009). Independent of committee size N.
        """
        if not self.vote_entropies:
            return 0.0
        return float(sum(self.vote_entropies) / len(self.vote_entropies))

    @property
    def max_vote_entropy(self) -> float:
        """Highest per-context vote entropy seen — the most contested case."""
        if not self.vote_entropies:
            return 0.0
        return float(max(self.vote_entropies))

    @property
    def disagreement_rate(self) -> float:
        """Fraction of contexts where the committee was not unanimous.

        Reported as a sanity-check companion to vote entropy. Range [0,1].
        """
        if self.total_comparisons == 0:
            return 0.0
        return len(self.disagreements) / self.total_comparisons

    # Backwards-compat: legacy callers used `penalty`. Now an alias for the
    # canonical mean vote entropy.
    @property
    def penalty(self) -> float:
        return self.mean_vote_entropy

    @property
    def max_disagreement_ratio(self) -> float:
        """Backwards-compat alias for max_vote_entropy."""
        return self.max_vote_entropy

    def render_feedback(
        self,
        action_namer: Optional[Callable[[int], str]] = None,
        cell_namer: Optional[Callable[[int], str]] = None,
    ) -> str:
        """Render LLM-facing feedback text — env-agnostic.

        Args:
            action_namer: optional callable mapping action ids to display
                names. If None, action ids are shown as integers.
            cell_namer: optional callable mapping observation cell values
                to display names. If None, raw integers are shown.

        The hand-coded MiniGrid name lookups have been removed (audit
        finding M7); callers from environment-specific code can supply
        their own resolvers without coupling this module to MiniGrid.
        """
        if not self.disagreements:
            return ""

        a_name = action_namer or (lambda a: str(a))

        lines = [
            "MODEL UNCERTAINTY (your candidate models disagree on these cases):",
            "-" * 65,
        ]

        grouped: Dict[Tuple[int, str], List[Disagreement]] = defaultdict(list)
        for d in self.disagreements:
            key = (d.action, _obs_front_digest(d.obs_context, cell_namer))
            grouped[key].append(d)

        sorted_groups = sorted(
            grouped.items(),
            key=lambda kv: -max(d.n_move / max(d.n_total, 1) for d in kv[1]),
        )

        for (action, obs_key), disags in sorted_groups[:5]:  # top 5
            d = disags[0]
            lines.append(
                f"- When agent sees {obs_key}, action={a_name(action)}: "
                f"models DISAGREE on {d.n_move}/{d.n_total} particles "
                f"(agree on {d.n_stay}/{d.n_total}). "
                f"(seen at step {d.step}, agent took {a_name(d.action_taken)} instead)"
            )

        lines.append("")
        lines.append(
            "These disagreements indicate uncertain transition rules. "
            "Think carefully about what should happen in each case. "
            "The majority vote is not always correct."
        )

        return "\n".join(lines)

    # Backwards-compat alias for callers that still expect a property.
    @property
    def feedback_text(self) -> str:
        """Default rendering with no env-specific name resolvers."""
        return self.render_feedback()


def _obs_front_digest(
    obs_image: Any,
    cell_namer: Optional[Callable[[int], str]] = None,
) -> str:
    """Compact description of the observation's full FOV.

    Previously only the frontal column was rendered, which collapsed
    distinct FOVs that happened to share their front row into a single
    group. Two different 3x3 patterns with the same front (e.g. the
    one with lava to the side vs without) then received merged
    disagreement counts, drowning signal. Now the whole image is
    serialised row by row.

    The optional ``cell_namer`` lets the caller translate raw integer
    cell codes into human-readable strings without this module
    knowing the environment.
    """
    if obs_image is None:
        return "[?]"
    try:
        arr = np.asarray(obs_image)
        namer = cell_namer or (lambda c: str(int(c)))
        if arr.ndim == 2:
            rows = [
                "[" + ",".join(namer(int(c)) for c in row) + "]"
                for row in arr.tolist()
            ]
            return "FOV=[" + " | ".join(rows) + "]"
        return str(arr.tolist())
    except Exception:
        return str(obs_image)


class DisagreementDetector:
    """Detect disagreements between N candidate transition models.

    For each step in a trajectory, tests alternative actions on belief
    particles using all candidate models. Reports where models disagree
    about the predicted next-belief distribution (the LLM only sees
    aggregate disagreement statistics, never raw state fields).

    Args:
        candidate_models: list of (name, transition_model) pairs
        observation_model: O(s, a, empty_obs) → obs (for context)
        actions: list of available actions
        empty_observation: passed to observation model
        max_particles_per_step: cap on particles tested (for speed)
    """

    def __init__(
        self,
        candidate_models: List[Tuple[str, TransitionModel]],
        observation_model: ObservationModel,
        actions: List[ActType],
        empty_observation: ObsType,
        max_particles_per_step: int = 10,
    ):
        self.candidates = candidate_models
        self.observation_model = observation_model
        self.actions = actions
        self.empty_observation = empty_observation
        self.max_particles_per_step = max_particles_per_step

    def detect_from_trajectory(
        self,
        beliefs: List[ParticleBelief],
        actions_taken: List[ActType],
        observations: List[ObsType],
    ) -> DisagreementResult:
        """Compute Vote Entropy across the committee on alternative actions.

        For each (state, action) context probed, all N committee members
        produce a predicted next state. Let V(y, x) be the number of
        members predicting class y. The normalised Vote Entropy is

            VE(x) = -1/log(N) * Σ_y (V(y,x)/N) * log(V(y,x)/N)

        with the convention 0 * log(0) = 0. VE ∈ [0, 1] independently of
        N (Dagan & Engelson 1995; Settles 2009).

        We aggregate VE across particles drawn from the belief, then
        across (alternative action) contexts. The mean of these
        per-context vote entropies is the canonical QBC uncertainty score
        and is reported via ``DisagreementResult.mean_vote_entropy``.

        Args:
            beliefs: belief at each trajectory step (from PF scoring).
            actions_taken: action taken at each step (excluded — PF scores it).
            observations: observation received at each step (for feedback context).
        """
        result = DisagreementResult()

        N = len(self.candidates)
        if N < 2:
            return result  # need at least 2 models for disagreement
        log_N = math.log(N)  # max possible vote entropy ⇒ normaliser

        for t in range(len(beliefs)):
            belief = beliefs[t]
            action_taken = actions_taken[t] if t < len(actions_taken) else None

            # Observation context (for the LLM feedback only — never used
            # to influence the score itself).
            obs_image = None
            if t < len(observations):
                obs_image = getattr(observations[t], "image", None)
                if obs_image is not None:
                    obs_image = obs_image.tolist()

            unique_states = list(belief.particles.keys())
            if not unique_states:
                continue
            test_states = unique_states[: self.max_particles_per_step]

            for action in self.actions:
                # Probe all actions including ``action_taken``. Committee
                # disagreement on the taken action is an independent signal:
                # it reveals models that predict different next states even
                # when the PF observation likelihood accepted them. Gains
                # more probes per step at near-zero compute cost (Python-only
                # trans_model calls, no LLM).

                # Per-particle vote entropies for this (action, t) context.
                # Each particle yields one VE value; we average them to get
                # one number per context.
                particle_ves: List[float] = []
                particle_disagreed = 0
                particle_total = 0

                for state in test_states:
                    # Collect all committee predictions for this particle.
                    counts: Counter = Counter()
                    for _, trans in self.candidates:
                        try:
                            next_state = trans(copy.deepcopy(state), action)
                            # Note on hashing: Python's hash() is fine here
                            # because we only need *equivalence within a
                            # single call*, not cross-process stability.
                            counts[hash(next_state)] += 1
                        except Exception:
                            # Failed prediction = the model is unable to
                            # take this action — counts as a distinct
                            # outcome to penalise broken candidates.
                            counts[("__fail__",)] += 1
                    n_votes = sum(counts.values())
                    if n_votes < 2:
                        continue  # not enough committee members responded

                    # Normalised vote entropy for this particle.
                    ve = 0.0
                    for c in counts.values():
                        if c <= 0:
                            continue
                        p = c / n_votes
                        ve -= p * math.log(p)
                    ve /= max(math.log(n_votes), 1e-12)  # ∈ [0,1]
                    particle_ves.append(ve)

                    particle_total += 1
                    if len(counts) > 1:
                        particle_disagreed += 1

                if not particle_ves:
                    continue

                # Aggregate per particle: mean vote entropy is the canonical
                # uncertainty signal at this context.
                ctx_ve = float(sum(particle_ves) / len(particle_ves))
                result.vote_entropies.append(ctx_ve)
                result.total_comparisons += particle_total

                # Record contexts that show ANY disagreement, for feedback.
                if particle_disagreed > 0:
                    result.disagreements.append(
                        Disagreement(
                            step=t,
                            action=action,
                            obs_context=obs_image,
                            action_taken=action_taken if action_taken is not None else -1,
                            n_move=particle_disagreed,
                            n_stay=particle_total - particle_disagreed,
                            n_total=particle_total,
                        )
                    )

        # Sort feedback list by severity so the LLM sees the most
        # contested cases first.
        result.disagreements.sort(
            key=lambda d: -d.n_move / max(d.n_total, 1)
        )

        return result

    def detect_from_beliefs_only(
        self,
        belief: ParticleBelief,
    ) -> DisagreementResult:
        """Quick check on a single belief (no trajectory needed).

        Useful for online episodes where we want to identify uncertain
        transitions from the current belief state.
        """
        result = DisagreementResult()
        if len(self.candidates) < 2:
            return result

        unique_states = list(belief.particles.keys())
        if not unique_states:
            return result
        test_states = unique_states[: self.max_particles_per_step]

        for action in self.actions:
            n_agree = 0
            n_disagree = 0
            n_tested = 0
            obs_ctx = None

            for state in test_states:
                next_hashes: List[int] = []
                for _, trans in self.candidates:
                    try:
                        ns = trans(copy.deepcopy(state), action)
                        next_hashes.append(hash(ns))
                    except Exception:
                        continue

                if len(next_hashes) >= 2:
                    n_tested += 1
                    if len(set(next_hashes)) > 1:
                        n_disagree += 1
                        if obs_ctx is None:
                            try:
                                o = self.observation_model(
                                    copy.deepcopy(state), action, self.empty_observation
                                )
                                obs_ctx = getattr(o, "image", None)
                                if obs_ctx is not None:
                                    obs_ctx = obs_ctx.tolist()
                            except Exception:
                                pass
                    else:
                        n_agree += 1

            result.total_comparisons += n_tested
            if n_disagree > 0:
                result.disagreements.append(
                    Disagreement(
                        step=-1, action=action, obs_context=obs_ctx,
                        action_taken=-1, n_move=n_disagree,
                        n_stay=n_agree, n_total=n_tested,
                    )
                )

        result.disagreements.sort(
            key=lambda d: -d.n_move / max(d.n_total, 1)
        )
        return result


# ---------------------------------------------------------------------------
# Top-K Diverse Committee Selection
# ---------------------------------------------------------------------------

def select_diverse_committee(
    candidates: List[Tuple[str, TransitionModel]],
    scores: List[float],
    test_states: List[Any],
    actions: List[int],
    K: int = 5,
    score_threshold: float = 2.0,
) -> List[Tuple[str, TransitionModel]]:
    """Select K candidates that are GOOD (score filter) and DIVERSE (max disagreement).

    1. Filter: keep candidates within `score_threshold` of the best score.
    2. Greedy diversity: iteratively add the candidate that maximizes
       minimum pairwise disagreement with the current committee.

    Args:
        candidates: (name, model) pairs from REx iterations.
        scores: Likelihood score for each candidate (higher = better).
        test_states: states to test disagreement on (from belief particles).
        actions: available actions.
        K: target committee size.
        score_threshold: max score gap from best (candidates below are dropped).

    Returns:
        Subset of candidates forming the diverse committee.
    """
    if len(candidates) <= K:
        return list(candidates)

    # Step 1: score filter
    best_score = max(scores)
    filtered = [
        (i, c, s) for i, (c, s) in enumerate(zip(candidates, scores))
        if s >= best_score - score_threshold
    ]
    if len(filtered) <= K:
        return [c for _, c, _ in filtered]

    # Precompute disagreement matrix
    indices = [i for i, _, _ in filtered]
    models = {i: candidates[i][1] for i in indices}

    def pairwise_disagreement(i: int, j: int) -> int:
        count = 0
        for state in test_states:
            for action in actions:
                try:
                    h_i = hash(models[i](copy.deepcopy(state), action))
                    h_j = hash(models[j](copy.deepcopy(state), action))
                    if h_i != h_j:
                        count += 1
                except Exception:
                    pass
        return count

    # Step 2: greedy selection — start with the best-scoring candidate
    best_idx = max(filtered, key=lambda x: x[2])[0]
    selected = [best_idx]

    for _ in range(K - 1):
        remaining = [i for i, _, _ in filtered if i not in selected]
        if not remaining:
            break
        best_next = max(
            remaining,
            key=lambda i: min(pairwise_disagreement(i, s) for s in selected),
        )
        selected.append(best_next)

    return [candidates[i] for i in selected]


# ---------------------------------------------------------------------------
# Information-theoretic metrics for model uncertainty (Bishop PRML Ch.1-3)
# ---------------------------------------------------------------------------

def committee_prediction_entropy(
    candidates: List[Tuple[str, TransitionModel]],
    state: Any,
    action: int,
) -> float:
    """Entropy of the committee's prediction distribution for (state, action).

    Each model predicts a next state.  We form a categorical distribution
    over the unique predictions (weighted uniformly across models) and
    compute its Shannon entropy:

        H[s' | s, a] = -Σ_k p(s'_k) log p(s'_k)

    High entropy = high disagreement = high epistemic uncertainty.
    This is the standard QBC uncertainty measure (Bishop PRML §1.6).

    Returns 0 when all models agree, log(N) when all predictions differ.
    """
    predictions: Dict[int, int] = {}  # hash → count
    n = 0
    for _, trans in candidates:
        try:
            ns = trans(copy.deepcopy(state), action)
            h = hash(ns)
            predictions[h] = predictions.get(h, 0) + 1
            n += 1
        except Exception:
            continue
    if n == 0:
        return 0.0
    entropy = 0.0
    for count in predictions.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log(p + 1e-12)
    return entropy


def mutual_information_committee(
    candidates: List[Tuple[str, TransitionModel]],
    belief: "ParticleBelief",
    action: int,
) -> float:
    """Mutual information I(model ; s' | s, a) across the committee.

    Measures how much knowing WHICH model was used reduces uncertainty
    about the predicted next state.  High MI = models disagree = the
    action is informative for distinguishing models.

        I = H[s' | s, a] - E_model[ H[s' | s, a, model] ]

    Since each model is deterministic, H[s'|s,a,model] = 0 for each
    particle, so:  I = H[s' | s, a]  averaged over belief particles.

    This is the BALD criterion (Houlsby et al. 2011) applied to
    transition models.  It naturally weighs states by their belief
    probability and actions by their informativeness.

    Returns:
        Average prediction entropy across belief particles.
        Higher = more informative action for model discrimination.
    """
    unique_states = list(belief.particles.keys())
    weights = [belief.particles[s] for s in unique_states]
    total_w = sum(weights)
    if total_w == 0 or not unique_states:
        return 0.0

    weighted_entropy = 0.0
    for state, w in zip(unique_states, weights):
        p = w / total_w
        h = committee_prediction_entropy(candidates, state, action)
        weighted_entropy += p * h

    return weighted_entropy


def reward_variance_committee(
    candidates: List[Tuple[str, TransitionModel]],
    reward_model: Callable,
    belief: "ParticleBelief",
    action: int,
) -> float:
    """Variance of predicted rewards across the committee (DreamerV3-XP).

    For each belief particle, each candidate transition model predicts a
    next state s'.  The reward model R(s, a, s') then produces a predicted
    reward.  The variance across candidates measures reward-relevant
    model disagreement.

    Especially useful with sparse rewards (0 everywhere, 1 at goal):
    variance > 0 means some models think this action reaches a reward
    while others don't — a strong signal to explore.

        Var[R] = E_belief[ Var_model[ R(s, a, T_i(s, a)) ] ]

    Ref: DreamerV3-XP (2025), intrinsic reward = mean(R) + β·var(R)

    Returns:
        Expected reward variance across committee, averaged over belief.
    """
    unique_states = list(belief.particles.keys())
    weights = [belief.particles[s] for s in unique_states]
    total_w = sum(weights)
    if total_w == 0 or not unique_states:
        return 0.0

    weighted_var = 0.0
    for state, w in zip(unique_states, weights):
        p = w / total_w
        rewards: List[float] = []
        for _, trans in candidates:
            try:
                ns = trans(copy.deepcopy(state), action)
                r, _ = reward_model(state, action, ns)
                rewards.append(float(r))
            except Exception:
                continue
        if len(rewards) >= 2:
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            weighted_var += p * var_r

    return weighted_var


def most_informative_action(
    candidates: List[Tuple[str, TransitionModel]],
    belief: "ParticleBelief",
    actions: List[int],
) -> Tuple[int, float]:
    """Select the action that maximizes mutual information with the committee.

    This is the action that would be most informative for distinguishing
    between candidate models — ideal for directed exploration.

        a* = argmax_a I(model ; s' | s, a)

    Ref: BALD (Bayesian Active Learning by Disagreement), Houlsby et al. 2011

    Returns:
        (best_action, mutual_info) tuple.
    """
    best_a = actions[0]
    best_mi = -1.0
    for a in actions:
        mi = mutual_information_committee(candidates, belief, a)
        if mi > best_mi:
            best_mi = mi
            best_a = a
    return best_a, best_mi
