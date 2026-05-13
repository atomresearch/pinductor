# Quickstart

This is the **shortest path** from a fresh checkout to a working
Pinductor smoke run. Targets a single CPU box (Ubuntu 24.04, Python 3.11
or 3.12). Should take under five minutes if the network is healthy.

## Checklist

- [ ] **1. Create a venv** with Python ≥ 3.11.

  ```bash
  conda create -n pinductor python=3.11 -y && conda activate pinductor
  ```

- [ ] **2. Install the package + every dep.** A single command installs
  Pinductor and everything the POMDP Coder baseline needs (the baseline's
  own `pyproject.toml` is **not** installed).

  ```bash
  pip install -e .
  ```

  Optional extras:

  ```bash
  pip install -e '.[fast,plots]'   # numba JIT + tueplots paper-style bundle
  ```

- [ ] **3. Smoke 1 — imports.** Should print `uncertain_worms OK`.

  ```bash
  python -c "import uncertain_worms; print('uncertain_worms OK')"
  ```

- [ ] **4. Smoke 2 — unit tests.** Should end with `OVERALL: PASS (7
  suites)`.

  ```bash
  python scripts/paper/tests/run_all.py
  ```

- [ ] **5. Smoke 3 — tabular run (no API key, ≈ 1 min).** Should end with
  `[done] {'done': 3}`. Outputs land under
  `outputs/paper_runs/runs/E1__lava__tabular__s0__*/`.

  ```bash
  python scripts/paper/paper_runner.py run E1 \
      --envs lava --conditions tabular --seeds 0
  ```

- [ ] **6. Configure the LLM API key** (only needed for the Pinductor /
  POMDP Coder baseline conditions).

  ```bash
  cp .env.example .env
  $EDITOR .env   # set OPEN_ROUTER_KEY=sk-or-v1-...
  ```

- [ ] **7. Smoke 4 — LLM end-to-end (≈ 5–10 min).** Watch for at least
  one `[OPENROUTER_TOKENS]` line in the per-atom `stdout.log` to confirm
  the API path is wired.

  ```bash
  timeout 600 python scripts/paper/paper_runner.py run E1 \
      --envs lava --conditions ours --seeds 0
  ```

- [ ] **8. Plot — sanity-check the figure generation pipeline.** Should
  drop PDFs/PNGs under `outputs/paper_runs/plots/pretty/`.

  ```bash
  python scripts/paper/plot_pretty.py
  ```

## If something fails

1. Re-read the troubleshooting section of the main `README.md` (§9).
2. Inspect the per-atom `stderr.log` in the failing
   `outputs/paper_runs/runs/<atom_dir>/`.
3. Check the SQLite registry with `sqlite3
   outputs/paper_runs/registry.db 'SELECT exp_id, env, condition, seed,
   episode_idx, status FROM runs LIMIT 20'`.

## Next steps

- Reproduce a full experiment: `python scripts/paper/paper_runner.py run
  E1` (or E2_offline, E2_online, E2b, E4). See `README.md` §5 for the
  full table of budgets.
- Read `docs/ARCHITECTURE.md` for the paper-to-code mapping.
- Browse the per-directory READMEs:
  - `uncertain_worms/README.md`
  - `uncertain_worms/policies/README.md`
  - `uncertain_worms/planners/README.md`
  - `uncertain_worms/environments/minigrid/README.md`
  - `particle_filtering/README.md`
  - `scripts/paper/README.md`
  - `scripts/paper/configs/README.md`
