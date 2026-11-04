"""Seed OUR expert demos into a DreamerV3-torch replay (train_eps) directory.

Reads an uncertain_worms ReplayBuffer pickle and writes one .npz per episode in
the EXACT on-disk format tools.simulate/save_episodes produce, so that
tools.load_episodes picks them up as pre-existing training episodes.

Episode -> npz layout (mirrors tools.simulate's cache, T = len(actions)):
  row 0        = reset      : vector=flat(prev_obs[0]),   action=0, reward=0,   discount=1,
                              is_first=True,  is_last=False,           is_terminal=False
  row i (1..T) = step i-1   : vector=flat(next_obs[i-1]), action=onehot(a[i-1]),
                              reward=r[i-1], discount=1-term[i-1],
                              is_first=False, is_last=(i==T), is_terminal=term[i-1]
Stored keys: vector, image(dummy 8x8x3 uint8), action(onehot), reward, discount,
             is_first, is_last, is_terminal.  Filename: {id}-{T+1}.npz  (so
             count_steps = (T+1)-1 = T env steps, matching simulate's accounting).

Observation flattening is IDENTICAL to envs/uw.py (11-d minigrid / 7-d blocks).

Usage:
  python -m envs.uw_seed_demos --name uw_blocks_rand6 --outdir <logdir>/train_eps
  # name may be the task string (uw_<x>) or the short key (<x>).
"""

import argparse
import datetime
import os
import pathlib
import pickle
import sys
import uuid

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import envs.uw as uw  # noqa: E402  (dummy image, flatten helpers, demo paths)


def _short(name):
    return name[3:] if name.startswith("uw_") else name


def _minigrid_flat(o):
    return uw._minigrid_flat(o)


def _blocks_flat(o):
    return np.asarray(o.image, np.float32)


def _popgym_flat_factory(n_obs):
    """Project our demo observation onto the vanilla one-hot space.

    Our MineSweeper demos are recorded through the antialias+markovian+flatten
    wrappers, so the observation vector is n_obs one-hot channels followed by
    extra wrapper channels.  Verified empirically (4/4 probes incl. reset):
    argmax(first n_obs dims) == the raw POPGym Discrete(n) observation the
    vanilla DreamerV3 wrapper consumes.  Taking the first n_obs dims therefore
    reproduces the vanilla observation exactly.
    """

    def flat(o):
        v = o.value
        arr = np.asarray(v[2] if isinstance(v, tuple) and len(v) == 4 else v, np.float32)
        return arr.reshape(-1)[:n_obs]

    return flat


def episode_to_npz_dict(ep, kind, n_actions, flat=None):
    if flat is None:
        flat = _minigrid_flat if kind == "minigrid" else _blocks_flat
    T = len(ep.actions)
    vectors, actions, rewards, discounts = [], [], [], []
    is_first, is_last, is_terminal = [], [], []

    # row 0: reset
    vectors.append(flat(ep.previous_observations[0]))
    actions.append(np.zeros(n_actions, np.float32))
    rewards.append(np.float32(0.0))
    discounts.append(np.float32(1.0))
    is_first.append(True)
    is_last.append(False)
    is_terminal.append(False)

    for i in range(T):
        term = bool(ep.terminated[i]) if getattr(ep, "terminated", None) is not None else (i == T - 1)
        vectors.append(flat(ep.next_observations[i]))
        oh = np.zeros(n_actions, np.float32)
        oh[int(ep.actions[i])] = 1.0
        actions.append(oh)
        rewards.append(np.float32(ep.rewards[i]))
        discounts.append(np.float32(1.0 - float(term)))
        is_first.append(False)
        is_last.append(i == T - 1)
        is_terminal.append(term)

    n = T + 1
    image = np.zeros((n, 8, 8, 3), np.uint8)
    # 'logprob' is added by the agent policy_output during tools.simulate; demo
    # episodes must carry it too so sample_episodes can stitch demo+online
    # episodes (all episodes must share the same key set).  Zeros are fine: the
    # world-model / actor-critic training never reads stored logprob.
    return {
        "vector": np.asarray(vectors, np.float32),
        "image": image,
        "action": np.asarray(actions, np.float32),
        "logprob": np.zeros(n, np.float32),
        "reward": np.asarray(rewards, np.float32),
        "discount": np.asarray(discounts, np.float32),
        "is_first": np.asarray(is_first, bool),
        "is_last": np.asarray(is_last, bool),
        "is_terminal": np.asarray(is_terminal, bool),
    }


def seed(name, outdir, verbose=True):
    key = _short(name)
    assert key in uw.UW_ENVS, f"unknown env {name!r}"
    kind, _cfg = uw.UW_ENVS[key]
    if kind == "popgym":
        n_actions = _cfg["n_actions"]
        flat_fn = _popgym_flat_factory(_cfg["n_obs"])
    else:
        n_actions = 4 if kind == "blocks" else 6
        flat_fn = None

    demo_rel = uw.UW_DEMO_FILES[key]
    demo_path = pathlib.Path(uw.UW_REPO) / "uncertain_worms" / demo_rel
    if not demo_path.exists():
        raise FileNotFoundError(demo_path)

    import uncertain_worms  # noqa: F401  (needed to unpickle)

    with open(demo_path, "rb") as f:
        buf = pickle.load(f)
    episodes = buf.episodes

    outdir = pathlib.Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    for j, ep in enumerate(episodes):
        d = episode_to_npz_dict(ep, kind, n_actions, flat_fn)
        length = len(d["reward"])  # T+1
        total_steps += length - 1
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eid = f"{ts}-demo{j:03d}-{uuid.uuid4().hex}"
        fn = outdir / f"{eid}-{length}.npz"
        with open(fn, "wb") as fo:
            np.savez_compressed(fo, **d)
    if verbose:
        print(
            f"[seed] {name}: wrote {len(episodes)} episodes, "
            f"{total_steps} env-steps -> {outdir}"
        )
    return len(episodes), total_steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="uw_<env> or <env> key")
    ap.add_argument("--outdir", required=True, help="target train_eps directory")
    args = ap.parse_args()
    seed(args.name, args.outdir)
