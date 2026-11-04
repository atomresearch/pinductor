# Architecture and paper-to-code mapping

This document is the bridge between the **paper** and the **code**. Read
it after the main `README.md` quickstart and before opening any 4000-line
module.

## 1. End-to-end data flow

```
                        ┌──────────────────────────────────────────────┐
                        │  Offline replay buffer                       │
                        │  (uncertain_worms/.../trajectory_data/*.pkl) │
                        │  observations a_t, o_{t+1}, r_{t+1}, d_{t+1} │
                        └────────────────────┬─────────────────────────┘
                                             │  N demonstrations
                                             ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                  REx loop  (paper Algorithm 1)                   │
   │                                                                  │
   │   ┌──────────────────────┐    ┌──────────────────────────────┐   │
   │   │  UCB1 parent select  │    │  LLM proposes candidate m_jk │   │
   │   │  rex_helpers.py      │───▶│  base_policy.requery_joint   │   │
   │   └──────────────────────┘    └──────────────┬───────────────┘   │
   │            ▲                                  │                  │
   │            │     score S_jk, diagnostics D_jk │                  │
   │            │                                  ▼                  │
   │   ┌────────┴──────────────┐   ┌──────────────────────────────┐   │
   │   │  Near-best selector   │   │  Particle filter score       │   │
   │   │  Eq. 17–18            │◀──│  LikelihoodEvaluator.evaluate_likelihood │   │
   │   └───────────────────────┘   │  (Eq. 5 / Eq. 7)             │   │
   │            │                  └──────────────┬───────────────┘   │
   │            │                                  │                  │
   │            │      QBC vote entropy Eq. 8      ▼                  │
   │            │             ┌──────────────────────────────────┐    │
   │            │             │  DisagreementDetector            │    │
   │            │             │  model_disagreement.py           │    │
   │            │             └──────────────────────────────────┘    │
   │            ▼                                                     │
   └───────  m*  (final model) ────────────────────────────────────  ─┘
              │
              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │   Online deployment loop                                         │
   │                                                                  │
   │     belief b_t (K particles)                                     │
   │       │                                                          │
   │       │  PO_DAStar.plan(b_t)                                     │
   │       ▼                                                          │
   │     action a_t   ─▶  env.step(a_t)  ─▶  o_{t+1}, r_{t+1}, d_{t+1}│
   │       │                                                          │
   │       │  agent.update_belief(o_{t+1})  (PF + rejuvenation)       │
   │       ▼                                                          │
   │     b_{t+1}                                                      │
   └──────────────────────────────────────────────────────────────────┘
```

After every episode the new trajectory is appended to the replay buffer
and a fresh REx round (top of the diagram) is triggered. This is the
"online" half of paper §4.5.

## 2. Paper-to-code reference table

| Paper element | Code location |
| --- | --- |
| **Algorithm 1** (Belief-based Pinductor refinement) | `uncertain_worms/policies/partially_obs_planning_agent.py::LLMPartiallyObsPlanningAgent.joint_update_models_rex` |
| Distance kernel `d(ô, o)` (§4.3, Eqs. 4–5) | `uncertain_worms/structs.py::MinigridObservation.distance_soft` |
| **Eq. 5** Distance-kernel observation likelihood per step | `particle_filtering/get_score_metrics.py::LikelihoodEvaluator._step_score` |
| **Eq. 7** Particle-filtered kernel score | `particle_filtering/get_score_metrics.py::LikelihoodEvaluator.evaluate_likelihood` (and `evaluate_score` for the public wrapper) |
| **Eq. 8** QBC vote entropy across the committee | `particle_filtering/model_disagreement.py::committee_prediction_entropy` (+ `DisagreementDetector` for per-context aggregation) |
| **Eq. 16** UCB1 parent selection (App. A.3) | `uncertain_worms/policies/rex_helpers.py::ucb1_select` |
| **Eq. 17 / 18** Near-best set + softmax final selector | Inline in `partially_obs_planning_agent.py::joint_update_models_rex` (search for `softmax_T`, `near_best`); driven by `likelihood_softmax_temperature` |
| **§4.5** PO_DAStar belief-space planner | `uncertain_worms/planners/PO_DAStar.py` |
| **App. A.2** Particle filter + rejuvenation | `partially_obs_planning_agent.py::LLMPartiallyObsPlanningAgent.update_belief` + `LikelihoodEvaluator._rejuvenate` / `_rejuvenate_step` |
| **App. A.3** UCB1 tree expansion | `rex_helpers.py::ucb1_select` + agent's `_select_node_to_refine` |
| **App. A.5** Hyperparameters | Frozen per-condition in `scripts/paper/configs/<cond>/<env>.yaml` |
| **App. D.2** Demonstration buffers | `uncertain_worms/environments/*/trajectory_data/` |
| **App. E.1** Block and MineSweeper | `uncertain_worms/environments/blocksworld/`, `uncertain_worms/environments/popgym/` |
| **App. F** POMDP Coder baseline | `curtis_baseline/uncertain_worms/policies/partially_obs_planning_agent.py` |
| **App. F.1** Tabular baseline | `curtis_baseline/uncertain_worms/policies/tabular_learners.py` |
| **App. F** Random baseline | `curtis_baseline/uncertain_worms/policies/random_policy.py` |
| **App. F.2** Direct LLM | `curtis_baseline/uncertain_worms/policies/direct_llm_{fo,po}_agent.py` |
| **App. F.3** PPO-GRU | `uncertain_worms/policies/recurrent_ppo.py`, `scripts/train_recurrent_ppo.py` |
| **App. F.4** DreamerV3 | `baselines/dreamerv3/` (from scratch), `baselines/dreamerv3-torch/` (demo-seeded) |
| **App. D.3** Optimal bound | `scripts/paper/optimal_baseline.py`, `uncertain_worms/planners/greedy_full_obs_planner.py` |
| **App. D.4** Prompt-information sweep | `env_descriptions.txt` (L3) + `uncertain_worms/policies/prompts/po_inserts.json` |
| Fig. 3 (main reward) | Plotting script not included; the numbers are in `results/` (Table 3) |
| Fig. 5 (E2 offline sweep) | Data from `E2_offline`; `scripts/paper/plot_e2_full_sweep.py` plots it |
| E2 online learning curves (not a paper figure) | Generated by `scripts/paper/plot_progression.py` |
| Fig. 10 (E2b stochastic) | Plotting script not included; runs via `paper_runner.py run E2b` |
| Table 2 (E4 LLM ablation) | Generated by `scripts/paper/plot_e4_3llms.py` |

## 3. Hyperparameter index

The paper reports the following hyperparameters (App. A.5). They are set
explicitly in `scripts/paper/configs/ours/*.yaml`; where a suite's YAML leaves
one out, the agent's constructor default applies, and that default is the
value listed below.

| Symbol | Config key | MiniGrid | Blocks | MineSweeper | Meaning |
| --- | --- | --- | --- | --- | --- |
| κ | `agent.kernel_bandwidth` | 0.2 | 1 (default) | 1e-3 | Distance-kernel sharpness in Eq. 5 |
| K | `agent.num_particles` | 10 | 100 | 80 | Particle belief size |
| M | `agent.num_model_attempts` / `num_online_model_attempts` | 5 / 5 | 5 / 5 | 2 / 1 | Candidates per offline / online REx round |
| T | `agent.likelihood_softmax_temperature` | 0.1 | 0.1 | 0 (default: argmax) | Final-selection softmax temperature (Eq. 18) |
| c | `agent.ucb1_c` | 1.0 | 1.0 | — (Thompson sampling, `use_ucb1_selection` off) | UCB1 exploration coefficient (Eq. 16) |
| α | `agent.planner.entropy_coeff` | 1.0 | 0.0 | 0.05 | Planner entropy bonus |
| λ | `agent.planner.lambda_coeff` | 0.1 | 0.0 | 0.01 | Planner-side cost coefficient |
| — | `agent.planner.max_expansions` | 5000 | 20000 | 30 | Planner budget |
| N_D | dataset / runtime override (E2_offline sweep) | 10 | 15 | 50 | Number of offline demos |
| H | `max_steps` | 10–60 | 120 | 14 / 30 / 54 | Episode horizon |

## 4. Adding pieces — quick pointers

* **New MiniGrid environment** → `uncertain_worms/environments/minigrid/README.md`
* **New condition (policy variant)** → `uncertain_worms/policies/README.md` (last section)
* **New planner** → `uncertain_worms/planners/README.md` (last section)
* **New prompt template** → `uncertain_worms/policies/prompts/README.md`
* **New LLM provider** → patch `uncertain_worms/utils.py::query_llm` and
  expose the provider via the `PAPER_LLM_MODEL` env var (see
  `scripts/paper/experiments.py::E4_llm_variation` for an example of
  how the runner threads the model id through Hydra).
* **New experiment** → add an enumerator to
  `scripts/paper/experiments.py::all_experiments` and reference it from
  `paper_runner.py run <name>`.

## 5. Glossary

| Term | Meaning |
| --- | --- |
| **TROI** | Transition / Reward / Observation / Initial — the four POMDP components proposed jointly (Pinductor) or one-by-one (POMDP Coder baseline). |
| **REx (Refinement EXploration)** | The iterative LLM-proposal + scoring + diagnostic loop (Algorithm 1). |
| **hp_hash** | Deterministic SHA256 over the resolved Hydra overrides. Used by the runner to deduplicate atoms across experiments. |
| **Atom** | A `(exp_id, env, condition, seed, episode_idx, llm_model, extra)` tuple. The unit of work for `paper_runner.py`. |
| **Group** | A bundle of atoms sharing `(env, condition, seed)`; one Hydra subprocess handles them together. |
| **QBC** | Query-By-Committee — using the disagreement across the candidate-model committee to identify uncertain transition contexts (Eq. 8). |
| **Rejuvenation** | Replenishing the particle population by sampling fresh particles from `ρ_0^m` and replaying the action history (App. A.2). |
| **Near-best set** | Set of candidate models within one standard deviation of the top score, from which the final model is softmax-sampled (Eq. 17–18). |
| **CWD-on-path** | Python's default behaviour where the working directory is the first entry of `sys.path`. We exploit this to load two `uncertain_worms` packages without `pip install` collisions. |

## 6. Where to read the actual algorithm

1. `uncertain_worms/policies/partially_obs_planning_agent.py` — start at
   the module docstring (Algorithm 1 pseudocode).
2. `particle_filtering/get_score_metrics.py` — `LikelihoodEvaluator.evaluate_likelihood`
   is the scoring loop.
3. `particle_filtering/model_disagreement.py` — `DisagreementDetector`
   (per-context committee aggregation) plus the global
   `committee_prediction_entropy` helper used as the QBC vote-entropy
   signal in Eq. 8.
4. `uncertain_worms/policies/rex_helpers.py` — UCB1 selector and tree
   node bookkeeping.

If you have one hour, read in this order. The rest of the code is glue.
