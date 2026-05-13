# `uncertain_worms/` — Pinductor pipeline

This package contains the **Pinductor** implementation: the LLM-guided
POMDP induction agent, the MiniGrid environments it operates on, the
belief-space planner, and the shared data structures plus utilities.

The same package name (`uncertain_worms`) is reused by
`curtis_baseline/uncertain_worms/` for the POMDP Coder baseline. The two
implementations are kept side-by-side and disambiguated by the runner's
working-directory switch (see top-level `README.md` §8).

## Layout

```
uncertain_worms/
├── __init__.py
├── structs.py            # ReplayBuffer, Episode, Observation, State
├── utils.py              # OpenRouter client, log dirs, RNG seeding, token book-keeping
├── policies/             # Agent classes + LLM prompting helpers + REx tree
├── planners/             # Belief-space planner (PO_DAStar)
└── environments/
    └── minigrid/         # Custom MiniGrid environments and demo collection
```

## Reading order for new contributors

1. **`structs.py`** — start here. Defines `ReplayBuffer`, `Episode`,
   `State`, `Observation`. Everything else operates on these.
2. **`policies/base_policy.py`** — abstract `Policy` interface and the
   LLM proposal helpers (`requery`, `requery_joint`).
3. **`policies/partially_obs_planning_agent.py`** — the core Pinductor
   agent (`LLMPartiallyObsPlanningAgent`). The module docstring at the
   top maps each step of Algorithm 1 of the paper to a method here.
4. **`planners/PO_DAStar.py`** — belief-space planner used during online
   evaluation. Selected through Hydra (`agent.planner._target_`).
5. **`environments/minigrid/minigrid_env.py`** — wraps custom MiniGrid
   layouts behind the `Environment` interface, including the partial
   field-of-view observation and the soft-distance kernel.

## Cross-package wiring

* `LLMPartiallyObsPlanningAgent` imports
  `particle_filtering.get_score_metrics.LikelihoodEvaluator` to compute the
  paper-grade kernel pseudo-likelihood and
  `particle_filtering.model_disagreement` for QBC vote entropy.
* All Hydra `_target_` paths in `scripts/paper/configs/ours/*.yaml`
  resolve into this package; the runner ensures the subprocess CWD makes
  this the active namespace.
