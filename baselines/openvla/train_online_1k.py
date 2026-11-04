"""VLA arm WITH OBSERVATION MEMORY — CORRECTED update rule.

An earlier version of this script drew the batch UNIFORMLY from the replay and
then applied RWR weights in the loss. That is degenerate here, and the run it
produced is not reported: the replay gains 120 failure steps per episode while
the 780 demonstration steps stay fixed, so the demonstrations' share of the
weighted gradient mass fell from 90.7% (25 episodes) to 24.9% (325). The policy
converged to cloning its own failures — measured action distribution left 84.2%
/ right 15.3% / pick 0.4% / drop 0.0%, i.e. it stopped manipulating blocks at
all. The reported number comes from the rule below instead.

The rule, in two parts:

1. Sample the batch WITH PROBABILITY PROPORTIONAL TO THE WEIGHT and drop the
   weight from the loss. Importance-sampled RWR has the same expected gradient as
   uniform-sampling-plus-weights, with lower variance; keeping both would
   double-count.

2. Give failed episodes weight EXACTLY ZERO (filtered behaviour cloning /
   self-imitation). Part 1 alone is not enough — simulated on the real weights,
   the demonstrations' share of the batch would still fall to 22.9% by episode
   985, because failures have weight exp(0/T) = 1 and accumulate without bound.
   Any scheme that leaves failures a positive weight is eventually dominated by
   them. Lowering the temperature would also work but is a knob; a hard zero is
   the principled version, and in a sparse-reward task a failed episode carries
   no reward signal to learn from — cloning it just reinstates the degeneracy.

Protocol, mirroring the demonstration-seeded DreamerV3 arm
(`baselines/dreamerv3-torch/envs/uw_seed_demos.py` + `campaign_offline.sh`):
  * seed phase: behavior cloning on ALL 15 demonstrations / 780 steps — the same
    file and the same count demo-seeded Dreamer preloads (its own log reports
    "wrote 15 episodes, 780 env-steps"). NOT the
    8-episode induction split: that one is matched to Pinductor, this one to
    Dreamer.
  * online phase: 985 interactive episodes on Blocks-rand6, continual updates
    from a replay buffer seeded with the demos -> 1,000 episodes total.
  * metric: discounted return (gamma 0.98) and solve rate over the LAST 100
    training episodes — exactly what the Dreamer rows report.

Update rule: Reward-Weighted Regression — BC on replay samples weighted by
exp(episode_return / T). [RWR: Peters & Schaal; cited from pre-training memory,
NOT verified against the original source.] Chosen because it degrades
gracefully: with no online success it reduces to cloning the demos, and it
strengthens smoothly as successful online episodes appear. A plain policy
gradient would carry no signal at 0% success.

Expectation recorded before the run: PPO-GRU scores 0.000 on this environment
with 5,000,000 steps, and this arm gets 118,200. A near-zero result was the
likely outcome and is itself the answer to the fairness objection — the baseline
now HAS the description and the demonstrations.

Checkpoints + a resumable JSONL of per-episode returns are written every 25
episodes, so a partial overnight run is still reportable at whatever episode
count it reaches.
"""
import json, os, pathlib, pickle, random, sys, time
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]                      # baselines/openvla -> repo root
_sys.path.insert(0, str(_REPO)); _sys.path.insert(0, str(_HERE))
from uncertain_worms.environments.blocksworld.blocksworld_env import (
    BlocksEnvironment, blocks_apply_action)
from transformers import AutoModelForVision2Seq, AutoProcessor
from hist_format import step_str, tail
from peft import LoraConfig, get_peft_model

HERE = _HERE
ROOT = _REPO / "outputs" / "openvla"
ROOT.mkdir(parents=True, exist_ok=True)
DESC = (HERE / "env_description.txt").read_text().strip()
VOCAB = 32000
ACTION_TOKENS = [VOCAB - (b + 1) for b in [32, 96, 159, 223]]
EMPTY_TOK, PAD_TOK = 29871, 32000
COL = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
GAMMA, HORIZON = 0.98, 120
# Hardware-bound history window. Measured sweep (diag_seqlen.py, one model load):
# 419 text tokens -> peak 19.69 GiB | 512 -> 20.98 | 640 (=24 steps) -> 22.54 OK
# | 768 -> OOM. 16 steps is used rather than the maximum 24: 22.54/23.98 is 94%
# utilisation and a 6% margin does not survive 985 episodes of fragmentation.
# Not a tuned hyperparameter — the largest window with a safe margin. Separately
# measured: demo-distribution action ambiguity is already 0 at a 2-step window
# (64 ambiguous contexts at 0 steps, 11 at 1, 0 at 2), so 16 is well above the
# structural requirement seen in the data.
HIST_STEPS = int(os.environ.get("VLA_HIST_STEPS", 16))
REPLAY_CAP = 1_000_000     # DreamerV3's dataset_size; never reached here
MEM_STOP_GB = float(os.environ.get("VLA_MEM_STOP_GB", "19"))
# Floor of free VRAM. The display driver needs headroom: at 0.2 GB free it
# could no longer submit commands and the desktop froze.
VRAM_STOP_GB = float(os.environ.get("VLA_VRAM_STOP_GB", "2.5"))
# 419 description tokens + HIST_STEPS x 9 + the action token, rounded up. Every
# training sample is padded to exactly this. See the note in update().
TRAIN_SEQ_LEN = int(os.environ.get("VLA_TRAIN_SEQ_LEN", "576"))


def _free_vram_gb():
    """Driver-level free VRAM. torch's memory_reserved() only sees its own caching
    allocator: on 2026-07-30 it reported 14.3 GB reserved while the process held
    23.5 GB on the card, leaving 0.2 GB for the display driver. Three desktop
    freezes came from that gap, so the guard reads the driver, not torch."""
    try:
        return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return 999.0


def _rss_gb():
    """Resident set size of this process, in GiB, read from /proc."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0
SEED = int(os.environ.get("VLA_SEED", 0))   # action sampling is stochastic; pin it
N_ONLINE = int(os.environ.get("VLA_N_ONLINE", 985))
# 2, not 8. Measured: one seed epoch is 780 update steps at ~1.9 s/step = >25 min,
# so 8 epochs would burn 3h20 before the first online episode. Shortening it does
# NOT weaken the Dreamer match: Dreamer has no separate seed phase at all — it puts
# the demonstrations in the replay and trains continuously — so this phase only has
# to establish the 4-token action mapping. Read the per-epoch loss in the log: if it
# is still falling at epoch 2, this budget is what limited the arm, not the method.
SEED_EPOCHS = int(os.environ.get("VLA_SEED_EPOCHS", 2))
TEMP = 0.1
BS = int(os.environ.get("VLA_BS", 1))         # measured: fwd+bwd OOMs past 640 text tokens
ACCUM = int(os.environ.get("VLA_ACCUM", 4))   # keeps the effective batch at 4
DEMOS = str(_REPO / "uncertain_worms" / "environments" / "blocksworld" /
            "trajectory_data" / "rand6_demos.pkl")
TMP = ROOT / "_render.png"
OUT = ROOT / "episodes.jsonl"


def render_obs(stack, held):
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    for h, b in enumerate(stack):
        ax.add_patch(Rectangle((-0.4, h), 0.8, 0.9, facecolor=COL[(b - 1) % 6], edgecolor="black", lw=2))
        ax.text(0, h + 0.45, str(b), ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.plot([-0.45, 0.45], [-0.45, -0.45], color="#333", lw=3)
    ax.add_patch(Rectangle((0.6, 5.5), 0.8, 0.9, facecolor=(COL[(held - 1) % 6] if held else "#eee"), edgecolor="black", lw=2))
    ax.text(1.0, 5.95, str(held) if held else "-", ha="center", va="center", fontsize=12)
    ax.text(1.0, 6.6, "held", ha="center", fontsize=8)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.6, 7.2); ax.axis("off")
    fig.savefig(TMP, bbox_inches="tight"); plt.close(fig)
    return Image.open(TMP).convert("RGB").resize((224, 224))


def main():
    # Action selection samples from the policy, so the run is stochastic.
    # Pin every RNG or the arm is not reproducible.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda"
    proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    tk = proc.tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa").to(dev)
    # Abort before the online loop if the card is already too full: the display
    # driver shares it, and at 0.2 GB free amdgpu stops submitting and the desktop
    # freezes. Checked here as well as per-episode, because saturation happens at
    # load time, before the first logging interval.
    _fv = _free_vram_gb()
    print(f"[mem] after model load: {_fv:.2f}G VRAM free "
          f"(floor {VRAM_STOP_GB}G)", flush=True)
    if _fv < VRAM_STOP_GB:
        print(f"[mem] ABORT: {_fv:.2f}G free is under the floor. Nothing started.",
              flush=True)
        sys.exit(3)

    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0,
        # Language-model attention projections only. "all-linear" also adapts the
        # 685 vision-backbone tensors, forcing both ViTs' activations to be kept
        # for the backward pass -> OOM at 22.9/24 GiB even at batch 1. q_proj and
        # v_proj exist ONLY in language_model (verified on the weight index: 64
        # modules, 0 in vision_backbone), so the vision tower stays frozen.
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    # Gradient checkpointing. The earlier note here said it was "measured useless"
    # (23.09 vs 23.14 GiB). That measurement was taken with use_cache=True, which
    # makes HuggingFace silently disable checkpointing, so it measured nothing.
    # Re-measured on 2026-07-30 with use_cache=False (diag_gc.py): OFF exceeds a
    # 19.7 GiB cap, ON peaks at 14.82 GiB. It is what makes this run fit on a
    # 24 GiB card shared with a display driver. Default is now ON.
    # Measured on this model and GPU (diag_mem.py): peak 23.09 GiB without it vs
    # 23.14 GiB with it, i.e. it buys zero memory here — most plausibly because
    # flash and memory-efficient attention are both unavailable on this Navi31
    # GPU, so the math fallback's peak occurs inside a layer rather than in stored
    # activations. It does cost compute (the forward is recomputed during the
    # backward), so leaving it on would slow every update step for nothing.
    def _find_lm(m):
        for name, child in m.named_children():
            if name == "language_model":
                return child
            found = _find_lm(child)
            if found is not None:
                return found
        return None

    _lm = _find_lm(model)
    assert _lm is not None, "language_model not found"
    if os.environ.get("VLA_GC", "1") == "1":
        _lm.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    print(f"[mem] gradient checkpointing: "
          f"{getattr(_lm, 'is_gradient_checkpointing', False)} (peak 14.82 GiB with, OOM without)",
          flush=True)
    # foreach=False: the multi-tensor AdamW path calls torch._foreach_div_, which
    # has no compiled HIP kernel on this gfx1100 build ("no kernel image is
    # available for execution on the device"). The per-tensor loop is slower but
    # is the only path with kernels here. Not a tuning choice — a hardware gap.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-4, foreach=False)

    def ids_for(hist):
        """Prompt = fixed description + observation-action history so far.
        Truncated to HIST_STEPS: 419 desc + 9/step + 256 visual patches. The
        2,048-position context is NOT the binding limit — VRAM is (fwd+bwd OOMs
        past 640 text tokens)."""
        p = f"In: What action should the robot take to {DESC}\nHistory: {tail(hist, HIST_STEPS)}?\nOut:"
        return torch.cat([tk(p, return_tensors="pt").input_ids[0], torch.tensor([EMPTY_TOK])])

    def px_of(imgs):
        return proc.image_processor(imgs, return_tensors="pt")["pixel_values"].to(dev, dtype=torch.bfloat16)

    def update(samples, weights):
        """RWR step on (image, history, action) triples. Weights are divided by a
        GLOBAL reference (mean weight of the data being trained on), never by the
        batch mean: at BS=1 a batch-mean normalisation yields 1.0 for every sample
        and silently deletes the RWR signal."""
        seqs = [torch.cat([ids_for(h), torch.tensor([ACTION_TOKENS[a]])])
                for _, h, a in samples]
        # FIXED training length, not the batch max. Measured 2026-07-30: free VRAM
        # during the backward fell 2.17 -> 2.18 -> 1.82 -> 0.46 GB across episodes
        # 3/6/9/15, monotonically, while the rollout stayed flat. That is a leak,
        # not sequence growth. The backward's GEMM workspaces are cached per shape
        # outside the PyTorch allocator and never freed, and max(len(x)) produced a
        # fresh shape almost every batch. One constant length means one set of
        # shapes and a bounded cache. The attention mask already ignores padding,
        # so this changes no gradient, only the allocation pattern.
        L = TRAIN_SEQ_LEN
        seqs = [x[-L:] for x in seqs]          # never exceed the fixed budget
        # LEFT-pad: keeps the action token at position -1 for every row
        ids = torch.stack([torch.cat([torch.full((L - len(x),), PAD_TOK), x]) for x in seqs]).to(dev)
        attn = torch.stack([torch.cat([torch.zeros(L - len(x)), torch.ones(len(x))])
                            for x in seqs]).to(dev)
        px = px_of([im for im, _, _ in samples])
        out = model(input_ids=ids, attention_mask=attn, pixel_values=px, use_cache=False)
        logp = torch.log_softmax(out.logits[:, -2, ACTION_TOKENS].float(), dim=-1)
        gold = torch.tensor([a for _, _, a in samples], device=dev)
        w = torch.tensor(weights, device=dev, dtype=torch.float32)
        # Normalise by a GLOBAL reference, never by the batch mean: at BS=1 a
        # batch-mean normalisation gives w/w = 1.0 for every sample and deletes
        # the RWR weighting entirely.
        w = w / max(W_REF[0], 1e-8)
        loss = -(w * logp.gather(1, gold[:, None]).squeeze(1)).mean() / ACCUM
        loss.backward()
        STEP[0] += 1
        if STEP[0] % ACCUM == 0:
            opt.step(); opt.zero_grad()
        return float(loss) * ACCUM

    W_REF = [1.0]   # mean RWR weight of the data currently being trained on
    STEP = [0]
    _render_cache = {}

    def render_cached(stack, held):
        """Rendering is matplotlib and costs ~as much as a forward pass; the
        observation space is small enough to memoise exactly."""
        k = (stack, held)
        if k not in _render_cache:
            _render_cache[k] = render_obs(stack, held)
        return _render_cache[k]

    # ---------- seed phase: BC on all 15 demonstrations ----------
    eps = list(pickle.load(open(DEMOS, "rb")).episodes)
    seed_pool, demo_w = [], []
    for i, ep in enumerate(eps):
        # each demo's OWN discounted return, so demonstrations and online episodes
        # are weighted on the same scale (hardcoding 1.0 would have given a failed
        # online episode, exp(0)=1, the same weight as a solved demonstration)
        dr = GAMMA ** (ep.length - 1) if (ep.rewards and ep.rewards[-1] > 0) else 0.0
        w = float(np.exp(dr / TEMP))
        hist = ""
        for t in range(ep.length):
            o = ep.previous_observations[t]
            a = int(ep.actions[t])
            # Store the observation KEY, not the rendered image: rendering is
            # memoised anyway, and a key-only replay is picklable, which is what
            # lets a restarted process resume with the same training distribution.
            seed_pool.append(((tuple(o.current_stack), int(o.held)), hist, a))
            demo_w.append(w)
            hist += step_str(a, tuple(o.current_stack), int(o.held))
    print(f"[seed] 15 demos -> {len(seed_pool)} steps | demo return "
          f"{GAMMA**(eps[0].length-1):.3f}..{GAMMA**(min(e.length for e in eps)-1):.3f} "
          f"-> RWR weight {min(demo_w):.1f}..{max(demo_w):.1f}", flush=True)
    rng = random.Random(0)
    t0 = time.time()
    W_REF[0] = float(np.mean(demo_w))
    pool = list(zip(seed_pool, demo_w))
    # Resume: the seed phase is deterministic and costs ~28 min. If its checkpoint
    # is already on disk (e.g. the run died later, in the rollout), load it rather
    # than recompute it. VLA_RESUME_SEED=0 forces a fresh seed phase.
    SEED_CKPT = ROOT / "lora_seed"
    if SEED_CKPT.exists() and os.environ.get("VLA_RESUME_SEED", "1") == "1":
        # Load the saved ADAPTER weights into the model already on the GPU.
        # An earlier version called PeftModel.from_pretrained on a freshly loaded
        # base, which put a SECOND 7B model in VRAM — visible in the log as a
        # duplicated "Loading checkpoint shards" — and is the likely cause of the
        # rollout OOMs that followed it. The base is loaded exactly once, above.
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        _sd = load_file(str(SEED_CKPT / "adapter_model.safetensors"))
        _res = set_peft_model_state_dict(model, _sd)
        # A key mismatch here would load NOTHING and silently restart from an
        # untrained adapter, discarding the whole seed phase with no error.
        _unexpected = list(getattr(_res, "unexpected_keys", []) or [])
        assert not _unexpected, f"adapter keys not matched: {_unexpected[:3]}"
        _nz = sum(1 for n, q in model.named_parameters()
                  if "lora_B" in n and q.abs().max().item() > 0)
        assert _nz > 0, "every lora_B is zero after load — the adapter did not load"
        print(f"[seed] RESUMED from {SEED_CKPT.name} — {len(_sd)} tensors, "
              f"{_nz} non-zero lora_B, seed phase skipped", flush=True)
    else:
        for e in range(SEED_EPOCHS):
            rng.shuffle(pool)
            tot = nb = 0
            for i in range(0, len(pool), BS):
                b = pool[i:i + BS]
                tot += update([(render_cached(*k), h, a) for (k, h, a), _ in b],
                              [w for _, w in b]); nb += 1
            print(f"[seed] epoch {e}: loss {tot/nb:.4f} | {time.time()-t0:.0f}s", flush=True)
        model.save_pretrained(str(SEED_CKPT))

    # ---------- online phase: 985 interactive episodes ----------
    env = BlocksEnvironment(instance="rand6")
    replay = [(k_, h, a, w) for (k_, h, a), w in zip(seed_pool, demo_w)]
    hist_rows = []
    # The training loop leaks ~13 GB/h of host memory (measured: RSS climbs during
    # the seed phase, which only calls update()), so a single process cannot reach
    # 985 episodes on a 30 GB machine. Rather than guess at the C-level cause, the
    # run checkpoints replay + episode counter and a supervisor restarts it. Resume
    # restores the exact training distribution, so the protocol is unaffected.
    ONLINE_STATE = ROOT / "online_state.pkl"
    k_start = 0
    if ONLINE_STATE.exists() and os.environ.get("VLA_RESUME_ONLINE", "1") == "1":
        _st = pickle.load(open(ONLINE_STATE, "rb"))
        replay, k_start, hist_rows = _st["replay"], _st["k"], _st["hist_rows"]
        print(f"[online-fixed] RESUMED at episode {k_start} — replay {len(replay)} "
              f"entries, {len(hist_rows)} episodes on record", flush=True)
    fout = open(OUT, "a")
    # ---- VECTORISED rollout -------------------------------------------------
    # The rollout is ~95% of wall-clock: 120 sequential forwards per episode at
    # batch 1, on a GPU measured at 46-74% utilisation. Running ROLL_B episodes in
    # lockstep through one batched forward multiplies throughput by roughly ROLL_B
    # without changing the protocol: the episodes are independent, each keeps its
    # own env/state/history, and each still takes exactly one action per step.
    # Episodes that finish early are masked out; almost all run the full horizon,
    # so lockstep wastes very little.
    # ROLL_B is a THROUGHPUT parameter, not a learning one — it cannot change what
    # is measured. It auto-halves on OOM because inference memory was never swept.
    ROLL_B = int(os.environ.get("VLA_ROLL_B", 8))
    envs = [BlocksEnvironment(instance="rand6") for _ in range(ROLL_B)]
    k = k_start
    # Per-phase host-memory accounting. The run leaks at a constant rate; these two
    # accumulators attribute it to the rollout forwards or to the training updates,
    # which is what decides whether the fix is allocator-side or code-side.
    leak_roll = leak_upd = 0.0
    while k < N_ONLINE:
        rss_a = _rss_gb()
        vram_min = [999.0]      # minimum observed INSIDE the episode, not after it
        vram_roll = [999.0]     # split by phase: the fix differs for each
        vram_upd = [999.0]
        b = min(ROLL_B, N_ONLINE - k)
        seeds = [100000 + k + i for i in range(b)]
        states = [envs[i].reset(seed=seeds[i]) for i in range(b)]
        trajs = [[] for _ in range(b)]
        hists = ["" for _ in range(b)]
        discs = [0.0] * b
        solved = [False] * b
        alive = [True] * b
        for t in range(HORIZON):
            live = [i for i in range(b) if alive[i]]
            if not live:
                break
            obs_l = [envs[i]._observe(states[i]) for i in live]
            imgs = [render_cached(tuple(o.current_stack), int(o.held)) for o in obs_l]
            seqs = [ids_for(hists[i]) for i in live]
            L = max(len(x) for x in seqs)
            ids = torch.stack([torch.cat([torch.full((L - len(x),), PAD_TOK), x])
                               for x in seqs]).to(dev)
            attn = torch.stack([torch.cat([torch.zeros(L - len(x)), torch.ones(len(x))])
                                for x in seqs]).to(dev)
            while True:
                try:
                    with torch.no_grad():
                        # use_cache=False: this loop re-encodes the whole history at
                        # every step and never reads past_key_values, so the default
                        # KV cache is pure allocation churn (32 layers x K,V per
                        # forward, 120 forwards per episode).
                        out = model(input_ids=ids, attention_mask=attn,
                                    pixel_values=px_of(imgs), use_cache=False)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(live) == 1:
                        raise
                    raise RuntimeError(
                        f"rollout OOM at batch {len(live)} — relaunch with "
                        f"VLA_ROLL_B={max(1, ROLL_B // 2)}")
            _fv = _free_vram_gb()
            vram_min[0] = min(vram_min[0], _fv)
            vram_roll[0] = min(vram_roll[0], _fv)
            probs = torch.softmax(out.logits[:, -1, ACTION_TOKENS].float(), -1).cpu().numpy()
            for j, i in enumerate(live):
                pj = probs[j] / probs[j].sum()
                a = int(np.random.choice(4, p=pj))
                o = obs_l[j]
                trajs[i].append(((tuple(o.current_stack), int(o.held)), hists[i], a))
                hists[i] += step_str(a, tuple(o.current_stack), int(o.held))
                nxt = blocks_apply_action(states[i], a)
                states[i] = nxt
                if nxt.is_solved():
                    discs[i] += GAMMA ** t
                    solved[i] = True
                    alive[i] = False
        # The rollout and the update phases alternate; without an explicit release
        # their allocations fragment and the peaks add up (measured: OOM at 23.56
        # of 23.98 GiB with ROLL_B=4). Clearing between phases is not tuning — it
        # returns memory that is provably dead.
        torch.cuda.empty_cache()
        rss_b = _rss_gb()
        leak_roll += rss_b - rss_a
        # ---- replay + updates, one episode at a time so the accounting is
        # identical to the sequential version
        for i in range(b):
            w = float(np.exp(discs[i] / TEMP)) if discs[i] > 0 else 0.0
            replay.extend([(im, h, a, w) for im, h, a in trajs[i]])
            assert len(replay) <= REPLAY_CAP, f"replay exceeded {REPLAY_CAP}"
            wts = np.asarray([wt for _, _, _, wt in replay], dtype=np.float64)
            if wts.sum() <= 0:
                raise RuntimeError("all replay weights are zero")
            idx = np.random.choice(len(replay), size=4 * ACCUM, p=wts / wts.sum())
            batch = [replay[x] for x in idx]
            W_REF[0] = 1.0
            n_demo = int((idx < len(seed_pool)).sum())
            ep_losses = []
            _t_upd = time.time()
            for j in range(0, len(batch), BS):
                bb = batch[j:j + BS]
                ep_losses.append(
                    update([(render_cached(*k), h, a) for k, h, a, _ in bb],
                           [1.0] * len(bb)))
                _fv = _free_vram_gb()
                vram_min[0] = min(vram_min[0], _fv)
                vram_upd[0] = min(vram_upd[0], _fv)
            upd_s = (time.time() - _t_upd) / max(len(ep_losses), 1)
            ep_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            row = {"episode": k + i, "ep_seed": seeds[i], "return": discs[i],
                   "loss": round(ep_loss, 5), "s_per_update": round(upd_s, 3),
                   "solved": int(solved[i]), "length": len(trajs[i]),
                   "elapsed_s": round(time.time() - t0, 1),
                   "demo_frac_in_batch": round(n_demo / len(batch), 3)}
            hist_rows.append(row); fout.write(json.dumps(row) + "\n")
        fout.flush()
        torch.cuda.empty_cache()
        leak_upd += _rss_gb() - rss_b
        k += b
        if (k // ROLL_B) % 3 == 0 or k >= N_ONLINE:
            last = hist_rows[-100:]
            r = np.array([x["return"] for x in last])
            _l = np.array([x["loss"] for x in last if x["loss"] == x["loss"]])
            _u = np.array([x["s_per_update"] for x in last])
            print(f"[online-fixed] ep {k}/{N_ONLINE} | last-{len(last)} return {r.mean():.4f} "
                  f"| loss {ep_loss:.4f} (last-{len(_l)} mean {_l.mean():.4f}) "
                  f"| {_u.mean():.2f}s/update "
                  f"| solved {sum(x['solved'] for x in last)}/{len(last)} "
                  f"| roll_b={ROLL_B} | {(time.time()-t0)/60:.0f} min "
                  f"| rss {_rss_gb():.2f}G resv {torch.cuda.memory_reserved()/2**30:.2f}G "
                  f"| vram min {vram_min[0]:.2f}G (rollout {vram_roll[0]:.2f} upd {vram_upd[0]:.2f} repos {_free_vram_gb():.2f}) "
                  f"| leak rollout {leak_roll:+.2f}G updates {leak_upd:+.2f}G "
                  f"| replay {len(replay)} cache {len(_render_cache)}",
                  flush=True)
            model.save_pretrained(str(ROOT / "lora_online"))
            tmp = ONLINE_STATE.with_suffix(".tmp")
            with open(tmp, "wb") as _f:
                pickle.dump({"replay": replay, "k": k, "hist_rows": hist_rows}, _f)
            os.replace(tmp, ONLINE_STATE)   # atomic: never a half-written state
            # Two independent stop conditions, both checked every logging interval.
            # The VRAM one is the important one: three desktop freezes came from the
            # card filling up under the display driver, and a check performed only at
            # launch cannot see a footprint that grows.
            if vram_min[0] < VRAM_STOP_GB:
                print(f"[online-fixed] STOP: VRAM fell to {vram_min[0]:.2f}G during "
                      f"episode {k}, below the {VRAM_STOP_GB}G floor. State saved, "
                      f"exiting with code 17.", flush=True)
                fout.close()
                sys.exit(17)
            if _rss_gb() > MEM_STOP_GB:
                print(f"[online-fixed] STOP: rss {_rss_gb():.1f}G exceeded "
                      f"{MEM_STOP_GB}G at episode {k}; state saved, exiting with "
                      f"code 17 for the supervisor to restart.", flush=True)
                fout.close()
                sys.exit(17)
    fout.close()
    last = hist_rows[-100:]
    r = np.array([x["return"] for x in last])
    print(f"\nFINAL (memory=obs_history, 15 demos + {len(hist_rows)} online, last-100 window): "
          f"return {r.mean():.4f} ± {1.96*r.std(ddof=1)/np.sqrt(len(r)):.4f} | "
          f"solved {sum(x['solved'] for x in last)}/{len(last)}", flush=True)


if __name__ == "__main__":
    main()
