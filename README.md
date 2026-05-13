# Pinductor — Learning POMDP World Models from Observations with Language-Model Priors

Code release for the accompanying paper *Learning POMDP World Models from
Observations with Language-Model Priors*.

Pinductor uses a large language model as a prior over executable POMDP
programs, and refines the proposals against a particle-filtered kernel
pseudo-likelihood — without ever seeing a ground-truth hidden state. The
induced model is consumed by a belief-space planner during online
interaction; the agent matches the privileged-state LLM baseline of
Curtis et al. (2025) on four MiniGrid environments while operating
strictly from observations, actions, and rewards.

```
                         ┌──────────────────────────────┐
                         │  Offline replay buffer       │
                         │  (a_t, o_{t+1}, r_{t+1}, …)  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  REx loop  (paper Algorithm 1)                               │
   │   ┌────────────────┐    ┌────────────────────────────┐       │
   │   │ UCB1 parent    │───▶│ LLM proposes m_jk          │       │
   │   │ select (Eq.10) │    │ (joint TROI, requery_joint)│       │
   │   └────────────────┘    └──────────────┬─────────────┘       │
   │            ▲                            │                    │
   │            │  S_jk, D_jk, Q_jk          ▼                    │
   │            │              ┌─────────────────────────────┐    │
   │            │              │ Particle filter score       │    │
   │            │              │ ELBOEvaluator (Eq. 7 / 8)   │    │
   │            │              │ + QBC vote entropy (Eq. 9)  │    │
   │            └──────────────┴─────────────┬───────────────┘    │
   │                                          ▼                   │
   │                              Near-best softmax m* (Eq. 11-12)│
   └─────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
       Online deployment: PO_DAStar planner over particle belief
       (paper §4.5, App. B.1)
```

📘 **Start here**: [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for the
five-minute install + smoke walkthrough.
🏗️ **Architecture & paper-to-code map**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Table of contents

1. [What's in this bundle](#1-whats-in-this-bundle)
2. [Installation](#2-installation)
3. [API keys](#3-api-keys)
4. [Smoke tests (do this first)](#4-smoke-tests-do-this-first)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Datasets](#6-datasets)
7. [How the dual-cwd model works](#7-how-the-dual-cwd-model-works)
8. [Troubleshooting](#8-troubleshooting)
9. [Repository layout](#9-repository-layout)
10. [How to extend](#10-how-to-extend)
11. [License and citation](#11-license-and-citation)

---

## 1. What's in this bundle

- The **Pinductor** pipeline (`uncertain_worms/`, `particle_filtering/`)
- The reused **POMDP Coder baseline** under `curtis_baseline/`. Only the
  MiniGrid environments are kept and heavy robotics dependencies are stripped
  from `pyproject.toml`.
- The paper experiment driver and Hydra configs (`scripts/paper/`)
- Pre-collected offline demonstration datasets (~3.7 MB) under
  `uncertain_worms/environments/minigrid/trajectory_data/*_paper_N*.pkl`
- Source `*_pure_mixed.pkl` buffers used by `regenerate_datasets.py`

A clean checkout fits under 30 MB on disk.

---

## 2. Installation

Tested on **Ubuntu 24.04**, **Python 3.11** and **Python 3.12** (a fresh
`pip install -e .` was verified end-to-end on both), single CPU
workstation, **no GPU**.

```bash
# 1. Create and activate a fresh environment
conda create -n pinductor python=3.11 -y
conda activate pinductor

# 2. Install Pinductor (this also pulls every dep needed by curtis_baseline/)
pip install -e .
```

This single install is enough for both Pinductor and the POMDP Coder
baseline. Both subtrees define a package named `uncertain_worms`; the
runner sets the working directory of each subprocess to either the
top-level repository or `curtis_baseline/`, so Python's normal CWD-on-path
behaviour selects the right implementation. `curtis_baseline/pyproject.toml`
is provided for documentation and standalone inspection — installing it is
**not** required.

The `pip install` step pulls `pymdptoolbox` from a public GitHub fork; if
your machine has no internet access at install time, vendor it manually
into your environment first.

Optional extras (not needed to reproduce the paper, but used for prettier
plots and a faster particle-filter kernel):

```bash
pip install -e '.[fast,plots]'   # numba JIT + tueplots paper-style bundle
```

---

## 3. API keys

Pinductor and the POMDP Coder baseline call LLMs through OpenRouter. Copy
`.env.example` to `.env` at the repository root and set your key:

```bash
cp .env.example .env
$EDITOR .env   # set OPEN_ROUTER_KEY=...
```

The LLM-free baselines (`tabular`, `random`, `curtis_hardcoded`) do **not**
require any API key — you can ignore this step if you only want to run
them.

If both keys (primary + backup) start returning HTTP 402/429, the runner
will mark the affected atoms as `rate_limited` in the SQLite registry; you
can resume them later with the `resume` subcommand (see §5).

---

## 4. Smoke tests (do this first)

Three tiers, ordered from fastest to slowest. Run them in order; if tier
*N* fails, **do not** spend money on tier *N+1*.

### 4.1. Imports — under one second, no API key

```bash
python -c "import uncertain_worms; print('uncertain_worms OK')"
```

Expected: `uncertain_worms OK`. A failure here means `pip install -e .`
did not complete cleanly.

### 4.2. Unit tests — under thirty seconds, no API key

```bash
python scripts/paper/tests/run_all.py
```

Expected: `OVERALL: PASS (7 suites)`. The test runner spawns one
subprocess per suite; individual suites can also be run directly with
`python scripts/paper/tests/test_<name>.py`.

### 4.3. Tabular run — about one minute, no API key

```bash
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions tabular --seeds 0
```

Expected last line: `[done] {'done': 3}`. Outputs land under
`outputs/paper_runs/runs/E1__lava__tabular__s0__<hash>/`, including
`result.json` (with a numeric `reward` field), `stdout.log`,
`stderr.log`, and a Hydra `metadata.json`.

### 4.4. End-to-end LLM smoke — five to ten minutes per atom, requires `OPEN_ROUTER_KEY`

```bash
# Pinductor (ours) — joint TROI proposal
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions ours --seeds 0

# POMDP Coder baseline — per-component proposals
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions curtis_their --seeds 0
```

Each atom does three episodes (configurable via the YAML); the first one
takes longer because the LLM proposes the initial model. You can interrupt
with `Ctrl-C` once you see at least one `episode_<k>_iter_<j>_step_*` file
in the run directory — the pipeline is then confirmed to be wired
end-to-end (dataset → particle filter → LLM proposal → planner).

---

## 5. Reproducing the paper

`scripts/paper/paper_runner.py` is the canonical driver. It enumerates the
(env, condition, seed, episode) atoms for each experiment, deduplicates
against a SQLite registry (`outputs/paper_runs/registry.db`), and launches
one Hydra subprocess per atom. Outputs land under
`outputs/paper_runs/runs/<exp>__<env>__<cond>__s<seed>__<hp_hash>/`.

Three subcommands are exposed:

| Subcommand | Use it to |
| --- | --- |
| `run <exp>` | Launch atoms for `<exp>` (skipping any already `done` per the registry) |
| `status [--exp <exp>]` | Count atoms by status (`pending`, `done`, `failed`, `rate_limited`, `skipped_dedup`) |
| `resume [<exp>]` | Re-launch only the atoms that are `pending`, `failed`, or `rate_limited` |

### 5.1. Paper experiment table

| Paper experiment | Command | Default budget |
| --- | --- | --- |
| **E1** main reward | `python scripts/paper/paper_runner.py run E1` | 4 envs × 5 conds × 10 seeds × 3 ep |
| **E2 offline** sample efficiency | `python scripts/paper/paper_runner.py run E2_offline` | 2 envs × 3 conds × N ∈ {2,4,6,8,10,12} × 10 seeds |
| **E2 online** learning curves | `python scripts/paper/paper_runner.py run E2_online` | 2 envs × {ours, curtis_their} × 10 seeds × K=10 ep |
| **E2b** stochastic robustness | `python scripts/paper/paper_runner.py run E2b` | 3 stochastic envs × {ours, curtis_their} × 10 seeds × 3 ep |
| **E4** LLM ablation | `python scripts/paper/paper_runner.py run E4` | 2 envs × {Qwen3-14B, Claude Opus 4.7} × 10 seeds × 3 ep |

### 5.2. Useful flags

```bash
# Plan only — print the atoms that would be launched, and their hp_hash.
python scripts/paper/paper_runner.py run E1 --dry-run

# Restrict the sweep (these flags accept comma-separated lists).
python scripts/paper/paper_runner.py run E1 \
    --envs lava,unlock --conditions ours,tabular --seeds 0,1

# Time-bounded run — stops launching new atoms after the given wall budget.
python scripts/paper/paper_runner.py run E1 --max-hours 4

# Parallel atoms (one Hydra subprocess each — watch RAM and rate limits).
python scripts/paper/paper_runner.py run E1 --workers 2

# Re-attempt only the unfinished atoms.
python scripts/paper/paper_runner.py status
python scripts/paper/paper_runner.py resume E1
```

### 5.3. Single-atom manual launch

The runner takes care of switching to the correct working directory
(`ours` runs from the repository root, while `curtis_*`, `random` and
`tabular` run from `curtis_baseline/`), so you almost never need to invoke
`main.py` directly. If you want to:

```bash
# Prefer the runner:
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions ours --seeds 0

# Direct invocation (advanced — see §7 for the cwd model):
python main.py --config-path=$(pwd)/scripts/paper/configs/ours \
    --config-name=lava seed=0 num_episodes=1
```

Available **conditions** (subdirectories of `scripts/paper/configs/`):
`ours`, `curtis_their`, `curtis_hardcoded`, `tabular`, `random`. Available
**envs** (file names within each subdirectory): `lava`, `lava_stoch`,
`unlock`, `unlock_stoch`, `four_rooms`, `four_rooms_stoch`, `corners`.
Use `--dry-run` to inspect the planned atoms without launching
subprocesses.

The plotting scripts under `scripts/paper/plot_*.py` consume the SQLite
registry produced by these runs to regenerate the paper figures; see
[`scripts/paper/README.md`](scripts/paper/README.md) for the figure
mapping.

---

## 6. Datasets

Offline demonstrations live under
`uncertain_worms/environments/minigrid/trajectory_data/*_paper_N*.pkl`.
They mix successful, failed, and truncated trajectories per environment
(cf. Appendix E of the paper). The same pickles are mirrored under
`curtis_baseline/uncertain_worms/environments/minigrid/trajectory_data/`
so that the POMDP Coder baseline reads them from its own working directory.

The natural-language environment descriptions injected at prompt level L3
(paper App. F.4) live in `env_descriptions.txt` at the repository root.
The other prompt levels (L0 = none, L1 = grid size, L2 = layout hints)
are encoded directly in
`uncertain_worms/policies/prompts/po_inserts.json`.

To regenerate the sliced `_paper_N*.pkl` buffers from the underlying
`_pure_mixed.pkl` sources (kept in the same directory):

```bash
python scripts/paper/regenerate_datasets.py
```

This script writes both subtrees in one pass and is idempotent — it skips
slices whose checksum already matches. See
[`uncertain_worms/environments/minigrid/trajectory_data/README.md`](uncertain_worms/environments/minigrid/trajectory_data/README.md)
for the pickle schema.

---

## 7. How the dual-cwd model works

The bundle ships **two** packages that both call themselves
`uncertain_worms`:

1. The top-level `uncertain_worms/` is the Pinductor implementation.
2. `curtis_baseline/uncertain_worms/` is the reused POMDP Coder baseline.

Each Hydra YAML pins `_target_` paths to classes in **its own**
`uncertain_worms` namespace. To make this work without conflicting
installs, `scripts/paper/runner_backend.py` maps each condition to the
right working directory:

| Conditions | Working directory of the subprocess |
| --- | --- |
| `ours` | repository root (`.`) |
| `curtis_their`, `curtis_hardcoded`, `random`, `tabular` | `curtis_baseline/` |

Python's normal CWD-on-path rule then ensures `import uncertain_worms`
resolves to the right package, *despite* `pip install -e .` having
exposed only the top-level one. The runner also propagates
`OPEN_ROUTER_KEY` explicitly into the subprocess environment so that
`curtis_baseline`'s own `dotenv` lookup does not silently fall back to
`None` when the `.env` file lives only at the repository root.

If you want to launch `curtis_baseline/main.py` by hand, `cd` into
`curtis_baseline/` first (or pass an absolute `--config-path`); see §8.1
for the failure mode you get otherwise.

---

## 8. Troubleshooting

### 8.1. `Error locating target 'uncertain_worms.policies.tabular_learners.…'` (wrong working directory)

Hydra is trying to import a class that lives in `curtis_baseline/`, but
your subprocess is running with the top-level repository as its working
directory. Use the runner (which sets `cwd` correctly), or `cd
curtis_baseline/` before invoking `main.py` for `tabular`, `random`,
`curtis_their`, or `curtis_hardcoded`. Symmetric advice for `ours`: run
from the repository root.

### 8.2. `Could not override 'tabular/hydra/job_logging'. No match in the defaults list.` (Hydra slash trap)

You invoked `main.py` with `--config-name=<cond>/<env>` (a slash). Hydra
treats the slash as a config-group separator and breaks the
`hydra/job_logging` override embedded in the YAML. Use the runner or
split the path:

```bash
# Wrong
python main.py --config-dir=scripts/paper/configs --config-name=tabular/lava

# Right
python scripts/paper/paper_runner.py run E1 \
    --envs lava --conditions tabular --seeds 0
```

### 8.3. `Response status code: 401` (OpenRouter auth)

The API key is unset or wrong. Check that `.env` exists at the repository
root, contains `OPEN_ROUTER_KEY=…`, and is being loaded — the runner
reads it with `python-dotenv` and propagates the variable into the Hydra
subprocess. As a bypass, you can export the variable in the parent
shell:

```bash
read -s OPEN_ROUTER_KEY
export OPEN_ROUTER_KEY
python scripts/paper/paper_runner.py run E1 --envs lava --conditions ours --seeds 0
```

---

## 9. Repository layout

```
.
├── main.py                        # Hydra entrypoint shared by all conditions
├── pyproject.toml                 # Pinductor (top-level) deps
├── env_descriptions.txt           # L3 natural-language env descriptions (cf. App. E)
├── .env.example                   # Copy to .env and set OPEN_ROUTER_KEY
├── LICENSE                        # MIT
├── README.md                      # This file
├── uncertain_worms/               # Pinductor pipeline
│   ├── policies/                  #   joint-model agent + base classes
│   ├── planners/                  #   PO_DAStar belief-space planner
│   ├── environments/minigrid/     #   custom MiniGrid envs + datasets
│   ├── structs.py                 #   ReplayBuffer, Episode, Observation
│   └── utils.py                   #   OpenRouter client, log dirs, RNG seeding
├── particle_filtering/            # Belief scoring helpers
│   ├── get_score_metrics.py       #   ELBOEvaluator (kernel pseudo-likelihood)
│   ├── model_disagreement.py      #   QBC vote-entropy disagreement
│   └── belief_quality_scorer.py   #   Optional oracle scorer (not used in paper)
├── curtis_baseline/               # Reused POMDP Coder baseline
│   ├── main.py                    #   Hydra entrypoint, same shape as the top-level
│   ├── pyproject.toml             #   Standalone dep manifest (not required to install)
│   └── uncertain_worms/           #   Per-component agent + tabular + random policies
└── scripts/paper/
    ├── paper_runner.py            # CLI: run / status / resume
    ├── runner_backend.py          # (env, cond) → main.py + cwd
    ├── experiments.py             # E1 / E2_offline / E2_online / E2b / E4 atoms
    ├── hyperparams.py             # hp_hash, override resolution, N_D validation
    ├── registry.py                # SQLite dedup + audit log
    ├── cost.py                    # Token-cost accounting
    ├── log_parser.py              # Parse Hydra stdout/stderr → structured metrics
    ├── build_configs.py           # Generator for configs/*.yaml (templates not shipped; see scripts/paper/configs/README.md)
    ├── regenerate_datasets.py     # Slice _pure_mixed.pkl → _paper_N{1..12}.pkl
    ├── configs/                   # Self-contained Hydra YAMLs (one per cond × env)
    ├── tests/                     # Unit tests (no API needed)
    ├── plot_pretty.py             # main bars + stochastic robustness figures
    ├── plot_e2_full_sweep.py      # E2 offline sweep figure
    ├── plot_progression.py        # E2 online learning curves
    └── plot_e4_3llms.py           # E4 LLM ablation figure
```

Every important subdirectory ships its own `README.md`. The ones worth
reading first:

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — guided 5-minute install + smoke
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — paper-to-code mapping + glossary
- [`uncertain_worms/README.md`](uncertain_worms/README.md)
- [`uncertain_worms/policies/README.md`](uncertain_worms/policies/README.md)
- [`particle_filtering/README.md`](particle_filtering/README.md)
- [`scripts/paper/README.md`](scripts/paper/README.md)

---

## 10. How to extend

The pieces below cover the common "I want to plug X into Pinductor"
questions. Each subsection points at the file(s) you need to touch and at
the per-directory README that has the long-form recipe.

### 10.1. Add a new MiniGrid environment

1. Subclass `minigrid.MiniGridEnv` in
   `uncertain_worms/environments/minigrid/custom_environments/<env>.py`
   (use `four_rooms.py` as a template).
2. Register it with `gym.register(...)` in
   `uncertain_worms/environments/minigrid/custom_environments/__init__.py`.
3. Add an L3 description block to `env_descriptions.txt`.
4. Collect demonstrations with
   `python -m uncertain_worms.environments.minigrid.collect_demos
   --env <new_id>` (pygame UI) and slice them via
   `python scripts/paper/regenerate_datasets.py`.
5. Hand-write the matching YAMLs under
   `scripts/paper/configs/<cond>/<new_env>.yaml` (use an existing env as
   a template). `build_configs.py` is the original generator pipeline
   but its upstream templates are not shipped — see
   [`scripts/paper/configs/README.md`](scripts/paper/configs/README.md).

Full recipe in
[`uncertain_worms/environments/minigrid/README.md`](uncertain_worms/environments/minigrid/README.md).

### 10.2. Add a new condition (policy variant)

1. Implement the policy as a subclass of
   `uncertain_worms.policies.base_policy.Policy` (Pinductor side) **or**
   `curtis_baseline.uncertain_worms.policies.base_policy.Policy`
   (baseline side).
2. Map the new condition to its repository in
   `scripts/paper/runner_backend.py::_CONDITION_TO_REPO`.
3. Add the YAML group under `scripts/paper/configs/<new_cond>/` (use an
   existing condition as a template — see
   [`scripts/paper/configs/README.md`](scripts/paper/configs/README.md)).
4. Update `scripts/paper/hyperparams.py::CONDITIONS` and, if the policy
   consumes offline demos or LLM calls, `_DATASET_CONSUMERS` /
   `_LLM_CONDITIONS`.
5. Smoke with `paper_runner.py run E1 --envs lava --conditions <new_cond>
   --seeds 0`.

### 10.3. Switch the LLM provider

The single LLM client lives in `uncertain_worms/utils.py::query_llm`. To
add or swap a backend:

1. Add the model and price entry in `scripts/paper/cost.py::PRICING`.
2. Implement the new request path in `query_llm` (mirror the OpenRouter
   branch).
3. Expose the model id via the `PAPER_LLM_MODEL` env var or the per-atom
   `llm_model` field in `scripts/paper/experiments.py` (see
   `E4_llm_variation` for how the runner threads it through Hydra).

### 10.4. Add a new planner

See [`uncertain_worms/planners/README.md`](uncertain_worms/planners/README.md).
Short version: subclass `PartiallyObservablePlanner`, point a YAML at the
new class via `agent.planner._target_`, smoke with the tabular condition
first to skip LLM costs.

### 10.5. Add a new experiment

1. Define a new enumerator in `scripts/paper/experiments.py` returning a
   list of `Atom` tuples.
2. Register it in `all_experiments()`.
3. Run via `paper_runner.py run <name>` and plot via a new
   `scripts/paper/plot_<name>.py` reading `outputs/paper_runs/registry.db`.

---

## 11. License and citation

Released under the **MIT License** (see `LICENSE`).

A citation block will be added here once the paper is publicly available.
