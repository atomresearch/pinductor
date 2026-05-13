# `curtis_baseline/` — POMDP Coder baseline subtree

The published per-component LLM-guided POMDP induction baseline reused as
the paper's main comparison method ("curtis_their" / "curtis_hardcoded"
conditions). Only the MiniGrid environments + the policies needed by the
paper are kept; the Spot / bosdyn / open3d / pybullet robotics codepaths
are removed and their dependencies stripped from
`curtis_baseline/pyproject.toml`.

## Files

```
curtis_baseline/
├── main.py            # Hydra entrypoint (twin of the top-level main.py)
├── pyproject.toml     # Standalone manifest; install is NOT required (deps come from top-level)
├── README.md          # This file
└── uncertain_worms/   # The baseline's own uncertain_worms namespace
    ├── structs.py
    ├── utils.py
    ├── metrics.py
    ├── policies/
    │   ├── base_policy.py
    │   ├── partially_obs_planning_agent.py   # Per-component TROI agent
    │   ├── tabular_learners.py               # Tabular baseline used in the paper
    │   ├── random_policy.py                  # Uniform random baseline
    │   ├── extract_code.py                   # LLM code-extraction helpers
    │   └── prompts/                          # Per-component prompt templates
    ├── planners/      # PartiallyObservablePlanner + PO_DAStar (same as top-level)
    └── environments/
        └── minigrid/  # Same custom envs as top-level, mirrored for parity
```

## Why a separate package?

Two reasons:

1. **Different `_target_` namespaces** — the baseline YAMLs point at
   `uncertain_worms.policies.partially_obs_planning_agent.LLMPartiallyObsPlanningAgent`,
   but the **baseline's** version of that class implements per-component
   refinement (a separate LLM call for `transition_func`,
   `observation_func`, `reward_func`, `initial_func`), whereas
   Pinductor's top-level version uses a single joint call. Keeping the
   two implementations side-by-side avoids polluting either codebase
   with feature flags.
2. **Different dataclass schemas** — the baseline's `MinigridObservation`
   carries an extra `agent_pos` field used by its scorer. Pickle files
   in `curtis_baseline/uncertain_worms/environments/minigrid/trajectory_data/`
   are deserialised through this version of the dataclass.

The runner switches its subprocess CWD to `curtis_baseline/` for the
`curtis_*`, `random` and `tabular` conditions, so Python's standard
CWD-on-path rule selects this `uncertain_worms` package transparently.

## Installation

You do **not** need to `pip install -e curtis_baseline`. The top-level
`pip install -e .` already provides every dependency, and the runner
loads this subtree by CWD-switching only. `curtis_baseline/pyproject.toml`
is shipped for documentation and for users who want to install/inspect
the baseline standalone.

## Manual launch

```bash
cd curtis_baseline
python main.py --config-path=$(pwd)/../scripts/paper/configs/curtis_their \
    --config-name=lava seed=0
```

In practice, prefer:

```bash
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions curtis_their --seeds 0
```
