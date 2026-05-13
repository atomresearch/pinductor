# `uncertain_worms/policies/prompts/` — Pinductor LLM prompt templates

Plain-text prompt templates used by `base_policy.requery_joint` to drive
the LLM (paper §4.2).

## Files

| File | Purpose |
| --- | --- |
| `po_inserts.json` | Per-environment inserts (env description, demo snippets, schema hints). Keys match the gym ids registered under `uncertain_worms/environments/minigrid/`. |
| `po_joint_model_prompt_nostate.txt` | Initial-proposal prompt for the joint TROI model. Loaded once at the first REx step. The `nostate` suffix marks the obs-only formulation used in the paper (no hidden-state labels are ever shown to the LLM). |
| `po_joint_model_refining.txt` | Refinement prompt fed at each subsequent REx step, mixing: the previous candidate code, its particle-filter score and diagnostics, QBC disagreement contexts, and a request to edit specific components. |

## How the templates are assembled

```
po_joint_model_prompt_nostate.txt        (skeleton)
        +
po_inserts.json[<env_id>][...]           (env_description, demo block, …)
        =
final prompt → OpenRouter chat completion
```

The actual loader lives in `base_policy.requery_joint`. The function looks
up `po_inserts.json[<env_id>][<insert_key>]` for each `{{insert}}` token
in the skeleton.

## How to add a new template

1. Drop a new `*.txt` skeleton in this directory.
2. Reference it from a Hydra YAML via `agent.prompt_path`.
3. If you need new insert keys, add them to `po_inserts.json` under each
   env id you care about.
4. Smoke-test with `paper_runner.py run E1 --envs lava --conditions ours
   --seeds 0` and check the LLM transcript dropped in the run's output
   directory (`episode_*_iter_*_step_*_joint_models_*_attempt_*_llm_input.txt`).
