# `scripts/paper/tests/` — Unit test suites

Seven test suites that verify the experiment infrastructure without
requiring an API key or a GPU. Total wall time < 30 s on a CPU box.

## Running

```bash
python scripts/paper/tests/run_all.py
```

Expected output ends with:

```
OVERALL: PASS (7 suites)
```

`run_all.py` spawns each suite as its own subprocess (using
`sys.executable`), so a single failing assertion does not cascade.

## Suites

| File | What it checks |
| --- | --- |
| `test_hp_hash.py` | The `hp_hash` SHA256 over a (cond, env, overrides) tuple is stable across runs and across machines. Guards against silent dedup collisions when the runner deduplicates across experiments. |
| `test_registry.py` | SQLite schema, `BEGIN IMMEDIATE` locking, transition rules (no demotion of `done`, append-only events table). |
| `test_log_parser.py` | Regex parsers for `[REWARDS]`, `[TOKENS]`, `[OPENROUTER_TOKENS]`, `[OPENROUTER_RATE_LIMIT]` markers. |
| `test_experiments.py` | Atom enumerators for E1 / E2_offline / E2_online / E2b / E4 produce the expected cardinalities and condition sets. |
| `test_multi_exp_group.py` | Grouping atoms by `(env, cond, seed)` correctly batches per-episode atoms into one subprocess. |
| `test_mutable_defaults.py` | Catches the classic "shared mutable default" footgun in dataclasses used by `ReplayBuffer` and friends. |
| `test_seed_reproducibility.py` | Both `uncertain_worms.environments.minigrid.MinigridEnvironment` instances (Pinductor and POMDP Coder baseline) produce the **same initial state** for a given seed. Critical for the fairness of the benchmark. |

## Adding a new test

1. Create `test_<your_topic>.py` in this directory; make it runnable as
   `python test_<your_topic>.py` (no pytest required).
2. Append the filename to the `SUITES` list at the top of `run_all.py`.
3. Use `Path(__file__).resolve().parents[3]` to compute the repository
   root if you need to import modules from `uncertain_worms/`.

The runner intentionally avoids `pytest` to keep the test infrastructure
trivial. If you want IDE-style discovery, `pytest scripts/paper/tests`
also works because each suite uses plain `assert`s.
