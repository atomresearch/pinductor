# `scripts/paper/configs/` — Hydra configs (auto-generated)

82 self-contained YAML files, one per `(condition, environment)` cell.
The 46 MiniGrid cells whose header starts with `# Auto-generated` were
produced by `scripts/paper/build_configs.py`; do not edit them by hand
unless you are sure you understand the implications (your edits will be
wiped on the next regeneration). The others (Blocks, MineSweeper and
Direct LLM) are written by hand, and their header says what they are.

## Layout

```
scripts/paper/configs/
├── ours/                # Pinductor (joint TROI proposal)
├── ours_hardcoded/      # True models in Pinductor's filter and planner (reference)
├── curtis_their/        # POMDP Coder baseline (per-component TROI)
├── curtis_hardcoded/    # POMDP Coder baseline with hand-coded ground-truth models
├── tabular/             # Tabular frequency-table baseline
├── random/              # Uniform random-action baseline
├── direct_llm_fo/       # Direct LLM, fully observed (no model, no planner)
└── direct_llm_po/       # Direct LLM, partially observed
```

The environments are the file names inside each subdirectory, and not
every condition carries all of them:

| Condition | Environments |
| --- | --- |
| `ours`, `ours_hardcoded`, `random` | all 12 |
| `curtis_their`, `tabular` | 11 — all but `empty` |
| `curtis_hardcoded` | 8 — MiniGrid only |
| `direct_llm_po`, `direct_llm_fo` | 8 — `blocks`, `corners`, `four_rooms`, `lava`, `unlock` and the three `minesweeper_*` |

The twelve are `corners`, `empty`, `four_rooms`, `four_rooms_stoch`,
`lava`, `lava_stoch`, `unlock`, `unlock_stoch`, `blocks`,
`minesweeper_easy`, `minesweeper_medium` and `minesweeper_hard`.

## Condition → working directory

The Hydra `_target_` paths in each YAML resolve into a specific
`uncertain_worms` namespace. `runner_backend.spec_for` translates the
condition into the matching subprocess CWD:

| Condition | Subprocess CWD | `uncertain_worms` resolved from |
| --- | --- | --- |
| `ours`, `ours_hardcoded` | repository root | `uncertain_worms/` (Pinductor) |
| `curtis_their`, `curtis_hardcoded`, `random`, `tabular`, `direct_llm_po`, `direct_llm_fo` | `curtis_baseline/` | `curtis_baseline/uncertain_worms/` (POMDP Coder baseline) |
| `curtis_their` on `blocks` only | repository root | `uncertain_worms/` (restored baseline agent, `curtis_original_agent.py`) |

Run a single condition manually (this is what `paper_runner.py` does
under the hood):

```bash
# Pinductor on Lava, seed 0
python main.py --config-path=$(pwd)/scripts/paper/configs/ours \
    --config-name=lava seed=0

# Tabular baseline on Lava, seed 0
cd curtis_baseline
python main.py --config-path=$(pwd)/../scripts/paper/configs/tabular \
    --config-name=lava seed=0
```

## How they were generated

`build_configs.py` reads per-condition source YAMLs from
`uncertain_worms/config/approaches/` and
`curtis_baseline/uncertain_worms/config/approaches/`, normalises them
through a deterministic transformer (`yaml.dump(..., sort_keys=True,
default_flow_style=False)`) and writes the result here.

The source `approaches/` trees are intentionally **not shipped** in this
bundle — they contain hundreds of unused legacy variants from the
original POMDP Coder codebase. The 46 generated YAMLs here are the frozen,
audited output of that pipeline; together with the hand-written files they
are the configurations the paper's experiments use. Re-running
`build_configs.py` on this bundle will skip every cell with
`[skip] …: source not found` and emit `0 YAMLs generated`,
which is the expected behaviour.

If you do want to regenerate from sources (e.g., after editing a
hyperparameter in an upstream template), check out the full research
codebase from the authors, then run:

```bash
python scripts/paper/build_configs.py
```

For the common case — tweaking a single hyperparameter — just edit the
relevant YAML(s) under `scripts/paper/configs/<cond>/<env>.yaml` by
hand and rerun the experiment. The header `# Auto-generated …` becomes a
historical marker once you do, which is fine.

## Hyperparameter audit

If you need to inspect a hyperparameter (e.g., `kernel_bandwidth`,
`ucb1_c`, `entropy_coeff`, `num_particles`, `num_model_attempts`) without
opening every YAML, grep:

```bash
grep -nH "kernel_bandwidth:" scripts/paper/configs/*/*.yaml
```

The paper's hyperparameters (App. A.5) are stored in the YAMLs. Two are
left to the agent's constructor default, which is the value the paper
states: the kernel bandwidth on Blocks (κ = 1) and the final-selection
temperature on MineSweeper (T = 0); `docs/ARCHITECTURE.md` §3 lists every
value per suite.
