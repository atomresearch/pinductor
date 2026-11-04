# OpenVLA-7B on Blocks

A pretrained vision-language-action model given the same two channels this
paper's method gets, and asked to solve the same task.

The point is a fairness objection: our reinforcement-learning baselines consume
no language while our method does, so the comparison could be read as language
access rather than method. This arm removes that reading. **V** is the agent's
own partial observation, rendered as an image — never the board, which would
leak the hidden state. **L** is the environment description, byte-identical to
the one the induction prompt receives (`env_description.txt` here is the same
1587 characters, verified by hash), plus the observation-action history.

Result: 0.000 return, 0 of the last 100 episodes solved. Over the whole run it
solved 5 of 985, at episodes 41, 85, 531, 543 and 797 — no concentration
towards the end, so there is no learning curve to report. This is not an
optimisation failure: the training loss falls from 0.87 to 0.087 over the
window, so the model learns what it is given. `results/` at the repository root
carries the row.

## The two arms, and why there are two

`train_online_1k.py` is the reported one. Its protocol is matched to the
demonstration-seeded DreamerV3 arm rather than to ours: the same 15
demonstrations (780 steps) preload the replay, then 985 interactive episodes,
and the number is the mean over the last 100 training episodes. That makes it
directly comparable to the Dreamer rows with no protocol confound.

`eval_heldout.py` is the check that the first result is not an artefact of that
convention. It evaluates a checkpoint deterministically (argmax) on the 30
held-out instances every other method is evaluated on, `ep_seed 0..29`. It also
reports the action histogram and the policy entropy, which is what separates
"the task is out of reach" from "the policy collapsed onto one action" — both
produce 120-step episodes and zero reward. Measured: 1.178 nats against 1.386
for uniform, so the policy had not collapsed.

Both return 0.000. The conclusion does not depend on the evaluation convention.

## Memory

The policy is π(a | image_t, description, o_0..o_{t-1}). History is the matched
condition: our own belief is updated with the whole observation history, so our
deployed policy is history-dependent too. A memoryless policy is additionally
capped at 72.9% action-accuracy on this data by observation aliasing — 64 of 127
distinct training observations map to conflicting actions. That cap is a
property of the environment, not of the renderer, which produces 185 distinct
images for 185 distinct observations with no collisions.

`hist_format.py` is the serialisation that makes this fit. OpenVLA's LLaMA-2
backbone has 2048 positions, 256 go to visual patches and 419 to the
description, so a 120-step episode has to cost under 9 tokens per step. Nothing
is dropped; `python hist_format.py` runs the round-trip test over the
demonstrations (780/780 exact).

## Running it

```bash
pip install -e '.[vla]'

python baselines/openvla/train_online_1k.py           # seed phase, then 985 episodes
python baselines/openvla/eval_heldout.py outputs/openvla/lora_seed
```

`openvla/openvla-7b` (≈ 14 GB) is pulled from the Hugging Face hub on first
call; the LoRA adapters this repository's runs produced are not shipped, since
they are useless without it. Checkpoints, the per-episode JSONL and the render
scratch file are written under `outputs/openvla/`; both scripts resume from
whatever is already there, so an interrupted run is not lost.

The paper's run was on one 24 GB AMD GPU under ROCm. Peak measured usage is
22.5 GiB at the 640-text-token history window, so 24 GB is the floor rather
than a comfortable margin; `VLA_BS` and `VLA_ACCUM` set the batch and the
accumulation if yours is smaller. The per-episode log of the paper's run totals
13.7 GPU-hours for the 985 online episodes.

## Update rule

Reward-weighted regression: behaviour cloning on replay samples drawn with
probability proportional to `exp(episode_return / T)`, with failed episodes at
weight exactly zero. The docstring of `train_online_1k.py` gives the derivation
and, more usefully, the earlier version that got this wrong and the measurement
that exposed it — a uniform draw with weights in the loss lets self-generated
failures take over the gradient, and the policy stops manipulating blocks at
all. That run is not reported.
