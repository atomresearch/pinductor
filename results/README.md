# Results

The numbers reported in the paper, one file per table, as plain CSV, together
with the results of the OpenVLA-7B baseline on Blocks that ships with the code
(`blocks_vision_language_action_baseline.csv`).

These are the values reported in the paper. The induction step calls an LLM,
and the provider is not deterministic even at a fixed seed (paper App. H.3):
replaying one identical request fifty times returned a model exactly equal to
the ground truth 36% of the time (± 13pp), and the programs that came back
were not the same program. A rerun therefore varies within that run-to-run
variance, which is why the paper reports seed-level confidence intervals.

## Layout

Most files are long-format, one observation per row:

| column | meaning |
|---|---|
| `env` | environment: `Blocks`, `Corners`, `Empty`, `FourRooms`, `Lava`, `Unlock`, `MineSweeper-Easy`, `MineSweeper-Medium`, `MineSweeper-Hard` |
| `method` | the arm, e.g. `Pinductor`, `POMDP Coder`, `DreamerV3 (online)` |
| `metric` | `return` for the mean discounted return, `solved_pct` for the percentage of episodes solved |
| `value` | the number |

`return` is the mean discounted return at γ = 0.98, except on MineSweeper,
where the published numbers are normalised to [0, 1]; a MineSweeper return and
a Blocks return are therefore not on the same scale.

`solved_pct` has three different bases depending on the arm, because the
published protocols differ. For the LLM methods it is the held-out evaluation
episodes; for DreamerV3, PPO-GRU and OpenVLA it is the last 100 of their 1000
training episodes; for the MineSweeper exactness row it is 149 audited models.
Each file's own section below says which applies.

Files whose axes are not method-by-environment keep their own named columns:
the per-seed audit, the compute table, the ablation and the exactness split.

Blocks is evaluated as 10 seeds of 3 episodes. The audit file is per seed, and
its two return columns are that seed's mean over its three episodes — the
columns say so in their names. Counting solved episodes from them would give
the wrong rate; the reported 60% is 18 of the 30 individual episodes.

Environment and method names are normalised across files, and every key that
appears in two files carries the same value in both, so joining on `env` and
`method` is safe.

## Files

| File | What it shows |
|---|---|
| `baselines_minigrid.csv` | every method on the five MiniGrid environments |
| `baselines_blocks_and_minesweeper.csv` | every method on the two non-MiniGrid suites |
| `reference_optimal_and_ground_truth_models.csv` | the two references: an oracle that sees the true state and plans exactly, and the true models run through this paper's own filter and planner |
| `induction_one_call_vs_full_pipeline.csv` | one induction call with no learning, against the full loop. Separates what the prior supplies from what the loop adds |
| `no_description_vs_baselines_minigrid.csv` | the method with every semantic cue removed from the prompt, against baselines that never had any |
| `tabular_10_vs_1000_demonstrations.csv` | the counting estimator at a hundred times the data |
| `model_exactness_and_return.csv` | how often an induced program is exactly the ground truth, and what that is worth. The three rows disagree, which is the finding |
| `blocks_transition_match_per_seed.csv` | per seed, how much of the 423,360-transition space the deployed model matched on each of the three episodes, and what that seed returned |
| `blocks_return_by_model_exactness.csv` | return split by whether the model that played was exact |
| `blocks_vision_language_action_baseline.csv` | a fine-tuned OpenVLA-7B given the same description and demonstrations |
| `prompt_scaffold_ablation.csv` | what survives stripping comments, the description, or the class names |
| `compute_cost_per_run.csv` | LLM calls, tokens, dollars and wall-clock per run and per suite. Wall-clock is a min/max pair because one published cell is a range |

## Four things to read carefully

DreamerV3 is reported as in the paper: the demonstration-seeded arm is the
mean over three training seeds, and the from-scratch arm is a single training
run.

`model_exactness_and_return.csv` groups Lava by the final model of each seed,
as the paper's exactness table does, while `baselines_minigrid.csv` reports the
main evaluation episodes. The two are computed over different sets, so the
exactness split is not meant to average back to the main Lava return.

`reference_optimal_and_ground_truth_models.csv` holds two different
references. `Optimal` sees the hidden state and plans exactly; it is an upper
bound nothing partially observable can reach. `GT models` has perfect models
but a hidden state and the same finite planning budget as every other row; it
is the reference a method could in principle match.

In `model_exactness_and_return.csv`, the `return_exact` column means the return
of episodes whose induced model was exactly correct, except on MineSweeper,
where no model ever is: there the column carries the ground-truth-models arm
instead. The `exact_share_basis` column says what each share is a share of,
since Blocks counts episodes and Lava counts seeds.
