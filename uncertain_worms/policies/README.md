# `uncertain_worms/policies/` — Agents and LLM proposal helpers

Implements the Pinductor agent and the abstract `Policy` interface shared
by every condition (random, tabular, hardcoded, Pinductor).

## Files

| File | Role |
| --- | --- |
| `base_policy.py` | Abstract `Policy` class. Houses `requery` / `requery_joint` (single-shot or joint LLM proposal) and the `*_translator` wrappers that turn LLM-generated source into callable Python closures. |
| `partially_obs_planning_agent.py` | The Pinductor agent (`LLMPartiallyObsPlanningAgent`) — REx loop, particle-filter scoring, QBC disagreement, near-best final selection. Top-of-file docstring maps Algorithm 1 of the paper to specific methods. |
| `rex_helpers.py` | Tree node (`RexNode`), UCB1 selector (paper App. B.3, Eq. 10), candidate-pool bookkeeping. |
| `llm_feedback.py` | Pure helpers shared between offline and online refinement: per-step diagnostic summaries, error parsing, prompt formatting. |
| `sfs_directions.py` | "Search-for-state" direction sampler used by the planner for belief-state expansion. |
| `prompts/` | Plain-text prompt templates loaded by `requery_joint`. See its README. |

## LLM proposal flow (Pinductor, joint mode)

```
LLMPartiallyObsPlanningAgent
   ├─ joint_update_models_rex (REx round)
   │     └─ base_policy.requery_joint           # prompts the LLM once
   │           ├─ reads prompts/po_joint_model_prompt_nostate.txt
   │           ├─ injects environment description + a few demo rollouts
   │           ├─ parses the LLM response as Python source
   │           ├─ exec()s it into a sandbox and grabs four callables
   │           │   (initial_func, observation_func, transition_func, reward_func)
   │           └─ wraps them via initial_model_translator, …
   ├─ LikelihoodEvaluator.evaluate_likelihood               # particle-filter score, Eq. 7
   ├─ committee_prediction_entropy              # disagreement signal, Eq. 9
   └─ inline near-best softmax (search          # Eq. 11–12
      `likelihood_softmax_temperature` /
      `near_best` in the agent)
```

## Adding a new condition

Two ways to ship a different policy in this codebase:

1. **Direct subclass** — inherit from `Policy` (see
   `curtis_baseline/uncertain_worms/policies/random_policy.py` for the
   simplest example) and add a new `scripts/paper/configs/<cond>/` group.
2. **Override the prompt template** — keep the same class but copy
   `prompts/po_joint_model_prompt_nostate.txt` to a new path and point
   `agent.prompt_path` at it via Hydra. Useful for ablating the
   description level (paper App. F.4 / Fig. 7).
