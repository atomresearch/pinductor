"""Régénération des datasets paper — dual dump ours/curtis.

Chaque codebase définit ``uncertain_worms.structs.ReplayBuffer`` → conflit
pickle si les 2 sont importées dans le même process. Solution : le script
s'auto-relance en 2 passes subprocess (``--side ours``, ``--side curtis``),
chacune avec son ``sys.path`` dédié. Le mode ``--orchestrate`` pilote les 2
passes et fait le sanity check final.

Entrée : ``uncertain_worms/environments/minigrid/trajectory_data/{env}_pure_mixed.pkl``
Sortie, par env et par N ∈ {1, 3, 5, 10} :
  - ``uncertain_worms/environments/minigrid/trajectory_data/{env}_paper_N{N}.pkl``
  - ``curtis_baseline/uncertain_worms/environments/minigrid/trajectory_data/{env}_paper_N{N}.pkl``

Garanties :
  - Mêmes (previous_states, next_states, actions, rewards, terminated) ✓
  - Seule divergence = observations (ours: 3 champs, curtis: 4 champs +
    agent_pos absolu)
  - Sélection déterministe des indices avec diversité succès/échec/truncation
  - Sanity check post-écriture : 3e subprocess compare step-by-step

Usage :
  python scripts/paper/regenerate_datasets.py                # mode orchestrator
  python scripts/paper/regenerate_datasets.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CURTIS_ROOT = REPO_ROOT / "curtis_baseline"

ENV_NAMES = {
    "lava":              "MyMiniGrid-LavaWall-v0",
    "corners":           "CornerGoalRandom-Empty-10x10-v0",
    "four_rooms":        "MyMiniGrid-FourRooms-v0",
    "unlock":            "MyUnlockEnv-v0",
    "lava_stoch":        "MyMiniGrid-StochasticLavaWall-v0",
    "four_rooms_stoch":  "MyMiniGrid-StochasticFourRooms-v0",
    "unlock_stoch":      "MyMiniGrid-StochasticUnlock-v0",
}
N_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12)

# Diversité succès/échec/truncation par (env, N). Source 20 trajs :
#   lava       : 10 succ + 9 fail + 1 trunc
#   corners    : 17 succ + 0 fail + 3 trunc
#   four_rooms : 15 succ + 0 fail + 5 trunc
#   unlock     : 15 succ + 0 fail + 5 trunc
#
# N=12 extends N=10 by construction: select_indices uses pool[:n], so
# N=12 contains N=10 plus two additional trajectories.
SELECTION: Dict[Tuple[str, int], List[Tuple[str, int]]] = {
    # Lava : 10 succ + 9 fail + 1 trunc
    ("lava", 1):  [("succ", 1)],
    ("lava", 2):  [("succ", 1), ("fail", 1)],
    ("lava", 3):  [("succ", 1), ("fail", 1), ("trunc", 1)],
    ("lava", 4):  [("succ", 2), ("fail", 1), ("trunc", 1)],
    ("lava", 5):  [("succ", 2), ("fail", 2), ("trunc", 1)],
    ("lava", 6):  [("succ", 3), ("fail", 2), ("trunc", 1)],
    ("lava", 8):  [("succ", 4), ("fail", 3), ("trunc", 1)],
    ("lava", 10): [("succ", 5), ("fail", 4), ("trunc", 1)],
    ("lava", 12): [("succ", 6), ("fail", 5), ("trunc", 1)],
    # Corners : 17 succ + 0 fail + 3 trunc
    ("corners", 1):  [("succ", 1)],
    ("corners", 2):  [("succ", 2)],
    ("corners", 3):  [("succ", 2), ("trunc", 1)],
    ("corners", 4):  [("succ", 3), ("trunc", 1)],
    ("corners", 5):  [("succ", 4), ("trunc", 1)],
    ("corners", 6):  [("succ", 5), ("trunc", 1)],
    ("corners", 8):  [("succ", 7), ("trunc", 1)],
    ("corners", 10): [("succ", 9), ("trunc", 1)],
    ("corners", 12): [("succ", 10), ("trunc", 2)],
    # Four Rooms : 15 succ + 0 fail + 5 trunc
    ("four_rooms", 1):  [("succ", 1)],
    ("four_rooms", 2):  [("succ", 2)],
    ("four_rooms", 3):  [("succ", 2), ("trunc", 1)],
    ("four_rooms", 4):  [("succ", 3), ("trunc", 1)],
    ("four_rooms", 5):  [("succ", 4), ("trunc", 1)],
    ("four_rooms", 6):  [("succ", 4), ("trunc", 2)],
    ("four_rooms", 8):  [("succ", 6), ("trunc", 2)],
    ("four_rooms", 10): [("succ", 7), ("trunc", 3)],
    ("four_rooms", 12): [("succ", 9), ("trunc", 3)],
    # Unlock : 15 succ + 0 fail + 5 trunc
    ("unlock", 1):  [("succ", 1)],
    ("unlock", 2):  [("succ", 2)],
    ("unlock", 3):  [("succ", 2), ("trunc", 1)],
    ("unlock", 4):  [("succ", 3), ("trunc", 1)],
    ("unlock", 5):  [("succ", 4), ("trunc", 1)],
    ("unlock", 6):  [("succ", 4), ("trunc", 2)],
    ("unlock", 8):  [("succ", 6), ("trunc", 2)],
    ("unlock", 10): [("succ", 7), ("trunc", 3)],
    ("unlock", 12): [("succ", 9), ("trunc", 3)],
    # Stochastic Lava : 5 succ + 5 fail + 0 trunc
    ("lava_stoch", 1):  [("succ", 1)],
    ("lava_stoch", 2):  [("succ", 1), ("fail", 1)],
    ("lava_stoch", 3):  [("succ", 2), ("fail", 1)],
    ("lava_stoch", 4):  [("succ", 2), ("fail", 2)],
    ("lava_stoch", 5):  [("succ", 3), ("fail", 2)],
    ("lava_stoch", 6):  [("succ", 3), ("fail", 3)],
    ("lava_stoch", 8):  [("succ", 4), ("fail", 4)],
    ("lava_stoch", 10): [("succ", 5), ("fail", 5)],
    # Stochastic FourRooms : 10 succ + 0 fail + 0 trunc
    ("four_rooms_stoch", 1):  [("succ", 1)],
    ("four_rooms_stoch", 2):  [("succ", 2)],
    ("four_rooms_stoch", 3):  [("succ", 3)],
    ("four_rooms_stoch", 4):  [("succ", 4)],
    ("four_rooms_stoch", 5):  [("succ", 5)],
    ("four_rooms_stoch", 6):  [("succ", 6)],
    ("four_rooms_stoch", 8):  [("succ", 8)],
    ("four_rooms_stoch", 10): [("succ", 10)],
    # Stochastic Unlock : 10 succ + 0 fail + 0 trunc
    ("unlock_stoch", 1):  [("succ", 1)],
    ("unlock_stoch", 2):  [("succ", 2)],
    ("unlock_stoch", 3):  [("succ", 3)],
    ("unlock_stoch", 4):  [("succ", 4)],
    ("unlock_stoch", 5):  [("succ", 5)],
    ("unlock_stoch", 6):  [("succ", 6)],
    ("unlock_stoch", 8):  [("succ", 8)],
    ("unlock_stoch", 10): [("succ", 10)],
}


def classify(ep) -> str:
    last_r = ep.rewards[-1] if ep.rewards else 0.0
    last_t = ep.terminated[-1] if ep.terminated else False
    if last_r > 0:
        return "succ"
    if last_t:
        return "fail"
    return "trunc"


def select_indices(kinds_list: Sequence[str], plan: Sequence[Tuple[str, int]]) -> List[int]:
    by_kind: Dict[str, List[int]] = {"succ": [], "fail": [], "trunc": []}
    for i, k in enumerate(kinds_list):
        by_kind[k].append(i)
    out: List[int] = []
    for kind, n in plan:
        pool = by_kind.get(kind, [])
        if len(pool) < n:
            raise RuntimeError(
                f"Plan demande {n}×{kind} mais seulement {len(pool)} dispo")
        out.extend(pool[:n])
    return out


# ---------------------------------------------------------------------------
# WORKER : tourne dans un subprocess isolé, PYTHONPATH pointant vers un seul
# codebase. Produit les pkl + un manifest JSON des indices sélectionnés.
# ---------------------------------------------------------------------------
def worker_main(side: str, envs: Sequence[str], dry_run: bool) -> int:
    # sys.path[0] est déjà positionné par l'orchestrateur via PYTHONPATH.
    # L'import `uncertain_worms.*` résout donc vers le bon codebase.
    from uncertain_worms.structs import ReplayBuffer
    from uncertain_worms.environments.minigrid.minigrid_env import (
        minigrid_hardcoded_observation_model_gen,
        minigrid_empty_observation_gen,
        MinigridObservation,
    )
    pomdp_obs_fn = minigrid_hardcoded_observation_model_gen()   # FOV 3×3
    empty_obs_gen_fn = minigrid_empty_observation_gen
    try:
        empty_obs = empty_obs_gen_fn()
    except TypeError:
        empty_obs = empty_obs_gen_fn

    # Baseline side = format NATIF de leur env avec fully_obs=False :
    #   - image = FOV 3x3
    #   - agent_pos = pos reelle (leur MinigridObservation l'inclut by design)
    #   - agent_dir, carrying = proprioception
    # Verifie empiriquement : state.get_field_of_view(3) == env.step(fully_obs=False).image
    # Precedemment on mettait state.grid.copy() (FO 10x10) -> mismatch avec leur
    # env runtime en fully_obs=False qui renvoie 3x3 -> PF collapse.
    def po_obs_fn_curtis(state):
        return MinigridObservation(
            image=state.get_field_of_view(3),
            agent_pos=tuple(state.agent_pos),
            agent_dir=int(state.agent_dir),
            carrying=state.carrying,
        )

    # Le buffer source est dans le codebase ours quel que soit le side
    # (les classes Episode/MinigridState sont structurellement compatibles).
    source_dir = REPO_ROOT / "uncertain_worms" / "environments" / "minigrid" / "trajectory_data"
    if side == "ours":
        out_dir = source_dir
    elif side == "curtis":
        out_dir = CURTIS_ROOT / "uncertain_worms" / "environments" / "minigrid" / "trajectory_data"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise SystemExit(f"Invalid side {side!r}")

    manifest: Dict[str, Dict] = {}

    for env_key in envs:
        env_full = ENV_NAMES[env_key]
        src_path = source_dir / f"{env_full}_pure_mixed.pkl"
        if not src_path.exists():
            print(f"[{side}][SKIP] {env_key}: {src_path} missing")
            continue

        buf_src = ReplayBuffer.load_from_file(str(src_path))
        kinds = [classify(ep) for ep in buf_src.episodes]
        counts = {k: kinds.count(k) for k in ("succ", "fail", "trunc")}
        print(f"[{side}] {env_key}: source n={len(kinds)} counts={counts}")

        env_manifest = {"source_counts": counts, "selections": {}}

        for N in N_VALUES:
            # Skip (env, N) pairs without an explicit SELECTION entry
            # (e.g. stoch envs cap at N=10 since their source_mixed
            # files only contain 10 trajectories).
            if (env_key, N) not in SELECTION:
                print(
                    f"[{side}]   N={N}: SKIP — no SELECTION for "
                    f"({env_key}, {N}) (source insufficient)"
                )
                continue
            plan = SELECTION[(env_key, N)]
            indices = select_indices(kinds, plan)
            sel_kinds = [kinds[i] for i in indices]
            env_manifest["selections"][str(N)] = {
                "indices": indices, "kinds": sel_kinds,
            }
            print(f"[{side}]   N={N}: indices={indices} kinds={sel_kinds}")

            buf_new = ReplayBuffer()
            for sel_idx in indices:
                ep = buf_src.episodes[sel_idx]
                for t in range(len(ep.actions)):
                    prev_state = ep.previous_states[t]
                    next_state = ep.next_states[t]
                    action = ep.actions[t]
                    reward = ep.rewards[t]
                    term = ep.terminated[t]

                    if side == "ours":
                        # PO : FOV 3×3 via la obs_fn POMDP-clean
                        prev_obs = pomdp_obs_fn(
                            state=prev_state, action=int(action),
                            empty_obs=empty_obs, agent_view_size=3,
                        )
                        next_obs = pomdp_obs_fn(
                            state=next_state, action=int(action),
                            empty_obs=empty_obs, agent_view_size=3,
                        )
                    else:
                        # curtis : format natif PO de leur env (fully_obs=False)
                        # FOV 3x3 + agent_pos (revele, leur design) + dir + carrying
                        prev_obs = po_obs_fn_curtis(prev_state)
                        next_obs = po_obs_fn_curtis(next_state)

                    # La signature d'append_episode_step diffère entre les 2
                    # codebases : ours prend prev_obs+next_obs, curtis prend
                    # seulement next_obs. On détecte par introspection.
                    import inspect
                    sig = inspect.signature(buf_new.append_episode_step)
                    if "previous_obs" in sig.parameters or len(sig.parameters) >= 7:
                        buf_new.append_episode_step(
                            prev_state, next_state, action,
                            prev_obs, next_obs, reward, term,
                        )
                    else:
                        buf_new.append_episode_step(
                            prev_state, next_state, action,
                            next_obs, reward, term,
                        )
                buf_new.wrap_up_episode()

            out_path = out_dir / f"{env_full}_paper_N{N}.pkl"
            if dry_run:
                print(f"[{side}]   [DRY] would write {out_path}")
                continue
            buf_new.save_to_file(str(out_path))
            print(f"[{side}]   wrote {out_path}")

        manifest[env_key] = env_manifest

    # Dump manifest côté side (pour orchestrateur)
    manifest_path = Path(f"/tmp/regen_manifest_{side}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


# ---------------------------------------------------------------------------
# SANITY CHECK : 3e subprocess qui charge les 2 pkl et compare step-par-step.
# On le fait côté curtis (les 2 pkl y sont compatibles structurellement).
# ---------------------------------------------------------------------------
SANITY_WORKER = r"""
import json, sys, numpy as np
from uncertain_worms.structs import ReplayBuffer

def cmp(path_ours, path_curtis):
    bo = ReplayBuffer.load_from_file(path_ours)
    bc = ReplayBuffer.load_from_file(path_curtis)
    if len(bo.episodes) != len(bc.episodes):
        return f"n_episodes mismatch {len(bo.episodes)} vs {len(bc.episodes)}"
    for ei, (eo, ec) in enumerate(zip(bo.episodes, bc.episodes)):
        if len(eo.actions) != len(ec.actions):
            return f"ep{ei} length mismatch {len(eo.actions)} vs {len(ec.actions)}"
        for t in range(len(eo.actions)):
            if int(eo.actions[t]) != int(ec.actions[t]):
                return f"ep{ei}.actions[{t}] differ: {eo.actions[t]} vs {ec.actions[t]}"
            if float(eo.rewards[t]) != float(ec.rewards[t]):
                return f"ep{ei}.rewards[{t}] differ"
            if bool(eo.terminated[t]) != bool(ec.terminated[t]):
                return f"ep{ei}.terminated[{t}] differ"
            so, sc = eo.previous_states[t], ec.previous_states[t]
            if list(so.agent_pos) != list(sc.agent_pos):
                return f"ep{ei}.prev.agent_pos[{t}] differ"
            if int(so.agent_dir) != int(sc.agent_dir):
                return f"ep{ei}.prev.agent_dir[{t}] differ"
            if not np.array_equal(so.grid, sc.grid):
                return f"ep{ei}.prev.grid[{t}] differ"
            # check next_state aussi
            sno, snc = eo.next_states[t], ec.next_states[t]
            if list(sno.agent_pos) != list(snc.agent_pos):
                return f"ep{ei}.next.agent_pos[{t}] differ"
            if int(sno.agent_dir) != int(snc.agent_dir):
                return f"ep{ei}.next.agent_dir[{t}] differ"
            if not np.array_equal(sno.grid, snc.grid):
                return f"ep{ei}.next.grid[{t}] differ"
    return "ok"

pairs = json.loads(sys.argv[1])
results = {}
for p in pairs:
    results[p["label"]] = cmp(p["ours"], p["curtis"])
print(json.dumps(results))
"""


def orchestrate(envs: Sequence[str], dry_run: bool) -> int:
    py = sys.executable
    script = str(Path(__file__).resolve())

    def _run(side: str, pp: str) -> int:
        env = {**os.environ, "PYTHONPATH": pp + os.pathsep + os.environ.get("PYTHONPATH", "")}
        cmd = [py, script, "--side", side, "--envs", ",".join(envs)]
        if dry_run:
            cmd.append("--dry-run")
        return subprocess.call(cmd, env=env)

    print("=== PASS 1 : side=ours ===")
    rc1 = _run("ours", str(REPO_ROOT))
    print(f"[orch] pass1 rc={rc1}")
    print("\n=== PASS 2 : side=curtis ===")
    rc2 = _run("curtis", str(CURTIS_ROOT))
    print(f"[orch] pass2 rc={rc2}")
    if rc1 != 0 or rc2 != 0:
        return max(rc1, rc2)
    if dry_run:
        print("[orch] dry-run — skip sanity check")
        return 0

    print("\n=== SANITY CHECK ===")
    pairs = []
    for env_key in envs:
        env_full = ENV_NAMES[env_key]
        for N in N_VALUES:
            # Skip pairs that were not generated (no SELECTION entry —
            # e.g. stoch envs cap at N=10 since their source has only
            # 10 trajectories).
            if (env_key, N) not in SELECTION:
                continue
            pairs.append({
                "label": f"{env_key}_N{N}",
                "ours": str(REPO_ROOT / "uncertain_worms" / "environments" / "minigrid"
                            / "trajectory_data" / f"{env_full}_paper_N{N}.pkl"),
                "curtis": str(CURTIS_ROOT / "uncertain_worms" / "environments" / "minigrid"
                              / "trajectory_data" / f"{env_full}_paper_N{N}.pkl"),
            })
    # Tourne dans le contexte curtis (les 2 pkl y sont loadables)
    env = {**os.environ, "PYTHONPATH": str(CURTIS_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.check_output(
        [py, "-c", SANITY_WORKER, json.dumps(pairs)], env=env,
    ).decode()
    # Extract last line that is pure JSON
    results = None
    for ln in reversed(out.strip().splitlines()):
        try:
            results = json.loads(ln)
            break
        except json.JSONDecodeError:
            continue
    if results is None:
        print("[orch] sanity parse failed, raw output:")
        print(out)
        return 2
    all_ok = True
    print(f"{'label':<20} | status")
    print("-" * 50)
    for lbl, status in results.items():
        flag = "PASS" if status == "ok" else f"FAIL ({status})"
        print(f"{lbl:<20} | {flag}")
        all_ok = all_ok and (status == "ok")
    return 0 if all_ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["ours", "curtis"], default=None,
                    help="(interne) exécute une passe worker")
    ap.add_argument("--envs", default=",".join(ENV_NAMES.keys()))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    envs = [e.strip() for e in args.envs.split(",")]
    if args.side:
        return worker_main(args.side, envs, args.dry_run)
    return orchestrate(envs, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
