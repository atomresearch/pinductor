# `scripts/paper/` — Paper experiment driver

Everything needed to schedule, deduplicate, run, resume and plot the
paper experiments. `paper_runner.py` is the CLI entrypoint; the other
modules are its dependencies.

## Files

```
scripts/paper/
├── paper_runner.py       # CLI: run / status / resume subcommands
├── experiments.py        # E1 / E2_offline / E2_online / E2b / E4 atom enumerators
├── hyperparams.py        # hp_hash, override resolution, dataset N validation
├── registry.py           # SQLite dedup + audit log (outputs/paper_runs/registry.db)
├── runner_backend.py     # (env, cond) → main.py + cwd + absolute --config-path
├── log_parser.py         # Parse Hydra stdout/stderr → structured metrics
├── cost.py               # Per-model price table + estimate_cost_usd
├── build_configs.py      # Pipeline that generated configs/*.yaml from upstream templates (templates not shipped — see configs/README.md)
├── regenerate_datasets.py# Slice _pure_mixed.pkl → _paper_N{1..12}.pkl mirrored to curtis_baseline/
├── configs/              # Self-contained Hydra YAMLs (6 conditions × 7 envs = 42 files)
├── tests/                # Unit suites (no API key, < 30s total)
├── plot_pretty.py        # E1 main bars (Fig. 2) + E2b stochastic robustness
├── plot_e2_full_sweep.py # E2 offline sweep (Fig. 4)
├── plot_progression.py   # E2 online learning curves
└── plot_e4_3llms.py      # E4 LLM ablation (Tab. 1)
```

## CLI reference

```bash
# Launch all atoms for an experiment (skips already-done atoms).
python scripts/paper/paper_runner.py run <exp> [--envs env,env] [--conditions cond,cond] [--seeds 0,1,…] [--dry-run] [--max-hours H] [--workers W]

# Count atoms by status across the registry.
python scripts/paper/paper_runner.py status [--exp <exp>]

# Relaunch only pending / failed / rate_limited atoms.
python scripts/paper/paper_runner.py resume [<exp>] [same flags as run]
```

`<exp>` is one of `E1`, `E2_offline`, `E2_online`, `E2b`, `E4`. See the
top-level `README.md` §5 for the per-experiment budgets.

## How atoms are built

`experiments.py` exposes one function per experiment, each returning a
list of `Atom` tuples:

```
Atom(exp_id, env, condition, seed, episode_idx, llm_model, extra)
```

`paper_runner` then groups atoms by `(env, condition, seed)` (a "group"
maps to a single Hydra subprocess that runs `num_episodes` episodes),
computes the deterministic `hp_hash` of the resolved overrides, and
queries the SQLite registry for prior completions. Only fresh
`(env, cond, seed, episode_idx, hp_hash)` tuples spawn a subprocess.

## SQLite schema

`outputs/paper_runs/registry.db` has two tables:

| Table | Purpose |
| --- | --- |
| `runs` | One row per episode. Unique on `(exp_id, env, condition, seed, episode_idx, hp_hash)`. Stores reward, wall time, tokens, cost, status, hp_hash, llm_model. |
| `events` | Append-only audit trail of status transitions (`pending → running → done / failed / rate_limited / skipped_dedup / superseded`). Useful for forensic debugging. |

## Outputs

Each subprocess writes to
`outputs/paper_runs/runs/<exp_id>__<env>__<condition>__s<seed>__<hp_hash>/`:

```
result.json     # rewards, terminations, wall_seconds, token_usage, …
stdout.log      # captured Hydra stdout (includes [REWARDS] / [TOKENS] markers)
stderr.log      # captured stderr
metadata.json   # the resolved Hydra config that was actually launched
overrides.json  # the CLI overrides passed to main.py
tensorboard/    # per-episode TB events (optional, agent-side)
episode_*_iter_*_step_*_..._llm_input.txt   # full LLM transcripts (one per call)
```

The plots read either the SQLite registry (fast path) or the per-atom
`result.json` files (fallback when the registry was wiped).
