# `particle_filtering/` — Belief scoring and disagreement helpers

The "scorer" half of Pinductor. While `uncertain_worms/` owns the agent
and environments, this package owns:

* **how candidate models are scored** under the observed trajectories;
* **how candidate models disagree** with each other, which feeds back
  into the LLM refinement prompt.

## Files

| File | Role | Paper reference |
| --- | --- | --- |
| `get_score_metrics.py` | `LikelihoodEvaluator` — bootstrap particle filter + rejuvenation + distance-kernel pseudo-likelihood. The class's `evaluate_likelihood(...)` method returns the per-trajectory and aggregate kernel score used by the agent. | §4.3 + Eq. 7 + App. A |
| `model_disagreement.py` | `DisagreementDetector` (per-context aggregation) + `committee_prediction_entropy` (global QBC signal, Eq. 9). | §4.4 + Eq. 9 |
| `belief_quality_scorer.py` | Oracle diagnostic — compares particle beliefs against ground-truth states. **Not** used in the production scoring loop (would break the obs-only setting); kept for calibration scripts only. | App. E (validation) |

## Scoring modes (`LikelihoodEvaluator.scoring_mode`)

Only two values are wired in this release:

| Mode | Description | Used in the paper |
| --- | --- | --- |
| `"energy"` | Distance-kernel pseudo-likelihood. The default and the one used in every reported result. | ✅ |
| `"belief_quality"` | Oracle reference: reads `episode.next_states`. Used for offline canary scripts only. | ❌ |

Other scoring modes (whiteness / betting / bisim / pf_self_consistency /
prediction_persistence / event / hybrid_pf_event) were exploration probes
during development; they are **not** shipped — selecting one raises
`NotImplementedError` with a helpful message.

## Algorithm cheatsheet

For a candidate model `m` and an episode `(o_0, a_0, …, o_H)`:

1. Sample `K` particles `s_0^i ~ ρ_0^m`.
2. For `t = 0 … H-1`:
   * Propagate: `s_{t+1}^i ~ T^m(·|s_t^i, a_t)` (deterministic in our envs).
   * Predict observation: `ô_{t+1}^i = O^m(s_{t+1}^i, a_t)`.
   * Weight by kernel: `w_{t+1}^i ∝ exp(-d(ô_{t+1}^i, o_{t+1}) / κ)`.
   * Resample if effective-sample-size falls below threshold.
   * Optionally rejuvenate (App. B.2).
3. Accumulate the per-step log normalising constant. The model score is
   the posterior expected log-likelihood (paper Eq. 7).

`d(·, ·)` is `MinigridObservation.distance_soft` (grid Hamming + carry
flag + direction flag) and `κ` is `kernel_bandwidth` (paper hyperparam,
default `0.2`).
