"""Post-hoc check: is a 0.0000 return a genuine failure or a degenerate policy?

An all-timeout, zero-return run is compatible with two very different causes:
  (a) the task is out of reach for this policy class, or
  (b) the policy collapsed onto one action, which also produces 120-step
      episodes and zero reward.
The per-episode JSONL cannot separate them (it stores no actions). This script
does, from the saved checkpoint, without disturbing the running job — run it
AFTER the GPU frees.

It reports:
  * the action histogram over the 30 held-out instances (uniform-ish or collapsed)
  * the entropy of the policy's action distribution, averaged over steps
  * a deterministic (argmax) held-out evaluation, which is also the
    `vla_bc_mem`-style number for the trained policy

Usage:
  python baselines/openvla/eval_heldout.py [checkpoint_dir]
"""
import collections
import pathlib
import sys

import numpy as np
import torch

import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]                      # baselines/openvla -> repo root
_sys.path.insert(0, str(_REPO)); _sys.path.insert(0, str(_HERE))
from uncertain_worms.environments.blocksworld.blocksworld_env import (
    BlocksEnvironment, blocks_apply_action)
from hist_format import step_str, tail
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from train_online_1k import (render_obs, HIST_STEPS, ACTION_TOKENS, EMPTY_TOK,
                             DESC, GAMMA, HORIZON)

ROOT = _REPO / "outputs" / "openvla"
ROOT.mkdir(parents=True, exist_ok=True)
CKPT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lora_online"


def main():
    dev = "cuda"
    proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    tk = proc.tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa").to(dev)
    if CKPT.exists():
        model = PeftModel.from_pretrained(model, str(CKPT))
        print(f"loaded {CKPT.name}", flush=True)
    else:
        print(f"WARNING: {CKPT} missing — evaluating raw pretrained weights", flush=True)
    model.eval()

    cache = {}

    def img_of(stack, held):
        if (stack, held) not in cache:
            cache[(stack, held)] = render_obs(stack, held)
        return cache[(stack, held)]

    env = BlocksEnvironment(instance="rand6")
    hist_actions = collections.Counter()
    entropies, returns, wins, lengths = [], [], 0, []

    for ep_seed in range(30):
        state = env.reset(seed=ep_seed)
        hist, disc, t = "", 0.0, 0
        for t in range(HORIZON):
            obs = env._observe(state)
            p_txt = (f"In: What action should the robot take to {DESC}\n"
                     f"History: {tail(hist, HIST_STEPS)}?\nOut:")
            ids = tk(p_txt, return_tensors="pt").input_ids[0]
            ids = torch.cat([ids, torch.tensor([EMPTY_TOK])]).unsqueeze(0).to(dev)
            px = proc.image_processor([img_of(tuple(obs.current_stack), int(obs.held))],
                                     return_tensors="pt")["pixel_values"].to(dev, dtype=torch.bfloat16)
            with torch.no_grad():
                out = model(input_ids=ids, pixel_values=px)
            logits = out.logits[0, -1, ACTION_TOKENS].float()
            probs = torch.softmax(logits, -1).cpu().numpy()
            entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))
            a = int(probs.argmax())          # deterministic, unlike the training rollouts
            hist_actions[a] += 1
            hist += step_str(a, tuple(obs.current_stack), int(obs.held))
            nxt = blocks_apply_action(state, a)
            if nxt.is_solved():
                disc += GAMMA ** t
                wins += 1
                state = nxt
                break
            state = nxt
        returns.append(disc)
        lengths.append(t + 1)

    r = np.array(returns)
    tot = sum(hist_actions.values())
    print("\naction histogram (0=left 1=right 2=pick 3=drop):")
    for a in range(4):
        n = hist_actions.get(a, 0)
        print(f"  action {a}: {n:6d}  {100.0*n/tot:5.1f}%")
    print(f"mean policy entropy: {np.mean(entropies):.3f} nats "
          f"(uniform over 4 = {np.log(4):.3f}, collapsed = 0.000)")
    collapsed = max(hist_actions.values()) / tot > 0.9
    print(f"VERDICT: {'DEGENERATE — one action >90% of steps' if collapsed else 'not collapsed — the failure is behavioural, not a stuck policy'}")
    print(f"\nheld-out (argmax, 30 instances): return {r.mean():.4f} "
          f"± {1.96*r.std(ddof=1)/np.sqrt(len(r)):.4f} | solved {wins}/30 "
          f"| mean length {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()
