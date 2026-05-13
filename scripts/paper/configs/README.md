# `scripts/paper/configs/` — Hydra configs (auto-generated)

42 self-contained YAML files, one per `(condition, environment)` cell.
They are **auto-generated** by `scripts/paper/build_configs.py`; do not
edit them by hand unless you are sure you understand the implications
(your edits will be wiped on the next regeneration).

## Layout

```
scripts/paper/configs/
├── ours/                # Pinductor (joint TROI proposal, paper §4)
├── curtis_their/        # POMDP Coder baseline (per-component TROI, App. F.1)
├── curtis_hardcoded/    # POMDP Coder baseline with hand-coded ground-truth models
├── tabular/             # Tabular frequency-table baseline (App. F.2)
└── random/              # Uniform random-action baseline (App. F.3)
```

Each subdirectory contains seven environment YAMLs:
`corners.yaml`, `lava.yaml`, `lava_stoch.yaml`, `four_rooms.yaml`,
`four_rooms_stoch.yaml`, `unlock.yaml`, `unlock_stoch.yaml`.

## Condition → working directory

The Hydra `_target_` paths in each YAML resolve into a specific
`uncertain_worms` namespace. `runner_backend.spec_for` translates the
condition into the matching subprocess CWD:

| Condition | Subprocess CWD | `uncertain_worms` resolved from |
| --- | --- | --- |
| `ours` | repository root | `uncertain_worms/` (Pinductor) |
| `curtis_their`, `curtis_hardcoded`, `random`, `tabular` | `curtis_baseline/` | `curtis_baseline/uncertain_worms/` (POMDP Coder baseline) |

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
original POMDP Coder codebase. The 42 YAMLs you see here are the frozen,
audited output of that pipeline and the only ones the paper experiments
ever consume. Re-running `build_configs.py` on this bundle will skip
every cell with `[skip] … source manquante` and emit `0 YAMLs generated`,
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

All six paper hyperparameters from Table 1 / App. D are explicitly stored
in the YAMLs — there is no hidden default scattered across the codebase.
