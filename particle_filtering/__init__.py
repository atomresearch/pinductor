"""Particle-filtering scoring and disagreement helpers for Pinductor.

Three modules:

* :mod:`~particle_filtering.get_score_metrics` — the
  :class:`~particle_filtering.get_score_metrics.LikelihoodEvaluator` class
  implements the distance-kernel pseudo-likelihood (paper Eq. 7 / Eq. 8),
  including bootstrap particle filtering with rejuvenation and the
  posterior expected log-likelihood used as the model-ranking score.

* :mod:`~particle_filtering.model_disagreement` — Query-By-Committee vote
  entropy (paper Eq. 9). Surfaces transition contexts where the candidate
  pool disagrees, which are then fed back into the LLM refinement prompt.

* :mod:`~particle_filtering.belief_quality_scorer` — *oracle* diagnostic
  that compares particle beliefs against ground-truth hidden states.
  Intentionally **not** used in the paper's main scoring loop (would
  violate the obs-only constraint); kept for calibration scripts.

See `particle_filtering/README.md` for the algorithmic mapping to paper
sections.
"""
