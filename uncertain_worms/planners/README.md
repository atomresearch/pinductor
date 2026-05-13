# `uncertain_worms/planners/` — Belief-space planners

Decision-time planners that consume a particle belief and return the next
action. The paper uses a single planner (`PO_DAStar`) for all conditions;
the abstract `PartiallyObservablePlanner` interface is kept stable so
alternative planners can be plugged in via Hydra.

## Files

| File | Role |
| --- | --- |
| `base_planner.py` | Abstract `PartiallyObservablePlanner` and `FullyObservablePlanner` interfaces. |
| `PO_DAStar.py` | The A*-style deterministic-approximation belief-space planner described in App. B.1 of Curtis et al. (2025) and reused as the paper's planner. |

## PO_DAStar — what it does

`PO_DAStar` expands a tree of (belief, action) nodes ordered by
`f(b) = g(b) + λ · H(b)` where:

* `g(b)` is the cumulative expected reward along the path so far;
* `H(b)` is the entropy of the particle belief at the node (encourages
  exploration into ambiguous states; controlled by `entropy_coeff` in
  the YAMLs; Pinductor uses `1.0`, the baseline uses `0.0` — see paper
  §App. D);
* `λ` is `lambda_coeff` (default 0.1).

Expansion is bounded by `max_expansions` (default `5000`, matching Curtis
et al.) and per-call wall budget via the agent.

## Adding a new planner

1. Subclass `PartiallyObservablePlanner` and implement `plan(belief)
   -> action`.
2. Register the class under a new module in this directory.
3. Point a Hydra YAML at it:
   ```yaml
   agent:
     planner:
       _target_: uncertain_worms.planners.my_planner.MyPlanner
       max_expansions: 5000
   ```
4. Smoke-test with `paper_runner.py run E1 --envs lava --conditions ours
   --seeds 0` and inspect the per-episode reward in the resulting
   `result.json`.
