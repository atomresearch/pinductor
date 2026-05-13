"""Paper experiment driver — sequential atom launcher with SQLite dedup.

Subcommands
-----------
* ``run <exp>``           launch all atoms for ``<exp>`` (E1, E2_offline,
                          E2_online, E2b, E4); skips atoms already marked
                          ``done`` in the registry.
* ``status [--exp <exp>]`` count atoms by status (pending / done / failed
                          / rate_limited / skipped_dedup) across all or
                          one experiment.
* ``resume [<exp>]``      relaunch only ``pending`` / ``failed`` /
                          ``rate_limited`` atoms.

Per-atom flow
-------------
1. ``runner_backend.spec_for(env, condition)`` resolves the subprocess
   working directory (repo root for ``ours*`` / ``curtis_baseline/`` for
   ``curtis_*``, ``random``, ``tabular``) and the absolute Hydra
   ``--config-path``.
2. ``hyperparams.resolve_overrides`` produces a deterministic list of
   Hydra overrides (seed, dataset path for E2 offline, LLM model for E4,
   …) and ``hyperparams.hp_hash`` derives a SHA256-keyed dedup hash from
   them.
3. The registry is queried for matching ``(exp_id, env, condition, seed,
   episode_idx, hp_hash)`` tuples; matching ``done`` rows are short-
   circuited as ``skipped_dedup``.
4. A subprocess ``python main.py --config-path … --config-name <env>
   <overrides>`` is spawned. Its stdout/stderr is streamed to per-atom
   log files under ``outputs/paper_runs/runs/<atom_dir>/``.
5. On completion, ``log_parser`` extracts per-episode rewards and per-call
   token counts, ``cost.estimate_cost_usd`` converts tokens to USD, and
   the registry rows are updated.

The runner is idempotent: re-running with the same hp_hash never reruns
``done`` atoms; ``resume`` only fires on the non-terminal statuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost import estimate_cost_usd
from experiments import Atom, build_atoms, EPISODES_PER_SEED
from hyperparams import as_hydra_args, hp_hash, resolve_overrides, DEFAULT_LLM
from log_parser import parse as parse_log
from registry import Registry
from runner_backend import REPO_ROOT, spec_for

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUNS_DIR = REPO_ROOT / "outputs" / "paper_runs"
DB_PATH = RUNS_DIR / "registry.db"
ATOM_DIR = RUNS_DIR / "runs"
PLOT_DIR = RUNS_DIR / "plots"
COST_PATH = RUNS_DIR / "cost_report.tsv"


def _git_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Grouping : (env, cond, seed, hp_hash) → subprocess unique
# ---------------------------------------------------------------------------
@dataclass
class Group:
    env: str
    condition: str
    seed: int
    hp_hash: str
    llm_model: Optional[str]
    overrides: Dict[str, object]
    atoms: List[Atom]  # tous les épisodes de ce seed


def _yaml_content_hash(env: str, condition: str) -> str:
    """Stable SHA256 of the YAML paper config for ``(condition, env)``.

    Inclus dans ``hp_hash`` pour invalider le cache dedup quand le YAML
    change SÉMANTIQUEMENT — pas quand on touche aux commentaires ou
    au formatage. La normalisation via ``yaml.safe_load`` + redump trié
    empêche les rotations de hp_hash spurieuses causées par de simples
    changements de format.
    """
    from runner_backend import PAPER_CONFIGS_DIR
    yaml_path = PAPER_CONFIGS_DIR / condition / f"{env}.yaml"
    if not yaml_path.exists():
        return "missing_yaml"
    try:
        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # default_flow_style=False + sort_keys=True → byte-identical
        # output regardless of original formatting / key ordering /
        # comments / trailing whitespace.
        canonical = _yaml.dump(
            data, default_flow_style=False, sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    except Exception:
        # Fallback: raw byte hash (legacy behaviour). Tagged so we can
        # tell at a glance which rows came from the legacy path.
        return "L_" + hashlib.sha256(yaml_path.read_bytes()).hexdigest()[:10]


def group_atoms(atoms: Sequence[Atom]) -> List[Group]:
    """Groupe les atomes par (env, cond, seed, hp_hash, llm_model).

    Assume que tous les atomes d'un même seed partagent le même hp_hash
    (construit une fois). Le hp_hash inclut maintenant le hash du YAML
    paper pour que toute modification du YAML (pas juste des overrides
    runtime) invalide le cache dedup.

    ``llm_model`` est ajouté dans la clé de groupement pour empêcher la
    collision E4 qwen3-14b vs claude-opus-4.7
    (le ``hp_hash`` ne contient pas le LLM, donc 2 atoms avec différents
    LLMs auraient été mergés dans le même group, et le ``llm_model`` du
    Group n'aurait été que celui du premier atom vu).
    """
    buckets: Dict[Tuple[str, str, int, str, Optional[str]], Group] = {}
    for atom in atoms:
        overrides = resolve_overrides(
            atom.env, atom.condition,
            llm_model=atom.llm_model or DEFAULT_LLM,
            extra=atom.extra,
        )
        # Combine runtime overrides hash + YAML content hash
        yaml_hash = _yaml_content_hash(atom.env, atom.condition)
        hh_overrides = hp_hash(overrides)
        hh = hashlib.sha256(
            f"{hh_overrides}::{yaml_hash}".encode("utf-8")
        ).hexdigest()[:16]
        key = (atom.env, atom.condition, atom.seed, hh, atom.llm_model)
        if key not in buckets:
            buckets[key] = Group(
                env=atom.env,
                condition=atom.condition,
                seed=atom.seed,
                hp_hash=hh,
                llm_model=atom.llm_model,
                overrides=overrides,
                atoms=[],
            )
        buckets[key].atoms.append(atom)
    return list(buckets.values())


# ---------------------------------------------------------------------------
# Launch un group
# ---------------------------------------------------------------------------
def launch_group(
    group: Group, *, registry: Registry, dry_run: bool, verbose: bool,
) -> Dict[str, int]:
    """Lance le subprocess pour un group et écrit les lignes dans le registre.

    Retourne un compteur ``{status: n}`` sur les atomes du group.
    """
    counts: Dict[str, int] = defaultdict(int)
    # 1) Upsert les N lignes pending (1 par atome ; plusieurs exp_ids peuvent
    # partager un group si leurs overrides produisent le même hp_hash — dans
    # ce cas chaque exp_id a sa propre ligne registry).
    run_ids: Dict[Tuple[str, int], int] = {}
    for atom in group.atoms:
        rid = registry.upsert_pending(
            exp_id=atom.exp_id,
            env=group.env,
            condition=group.condition,
            seed=group.seed,
            episode_idx=atom.episode_idx,
            hp_hash=group.hp_hash,
            llm_model=group.llm_model,
            overrides=group.overrides,
        )
        run_ids[(atom.exp_id, atom.episode_idx)] = rid

    # 1.5) Skip uniquement si TOUS les atomes du group sont déjà `done`.
    # Auparavant on skippait dès qu'AU MOINS un atome était done (anti-
    # écrasement des row done par mark_failure). Mais depuis que
    # `Registry.mark_failure` refuse explicitement d'écraser un row done
    # avec reward (registry.py:mark_failure guard), on peut retry les
    # groupes partiels en sécurité — le subprocess peut crasher, les
    # done valides sont préservés, et les failed/pending du group sont
    # retentés.
    n_done_existing = 0
    with registry._conn() as _conn:
        for atom in group.atoms:
            rid = run_ids[(atom.exp_id, atom.episode_idx)]
            row = _conn.execute(
                "SELECT status FROM runs WHERE id=?", (rid,)
            ).fetchone()
            if row is not None and row["status"] == "done":
                n_done_existing += 1
    if n_done_existing == len(group.atoms):
        # Tous done : rien à faire.
        if verbose:
            _log(
                f"[all_done] {group.env}/{group.condition}/s{group.seed}: "
                f"{n_done_existing}/{len(group.atoms)} déjà done — skip"
            )
        counts["skipped_partial_done"] = len(group.atoms)
        return counts

    # 2) Dédup : si TOUS les épisodes du group existent `done` ailleurs →
    # skip le subprocess, on marque skipped_dedup.
    all_dedup = True
    dedup_sources: Dict[Tuple[str, int], object] = {}
    for atom in group.atoms:
        src = registry.find_dedup_source(
            env=group.env, condition=group.condition, seed=group.seed,
            episode_idx=atom.episode_idx, hp_hash=group.hp_hash,
            # Pass llm_model to prevent cross-LLM dedup collisions.
            llm_model=group.llm_model,
        )
        if src is None:
            all_dedup = False
            break
        dedup_sources[(atom.exp_id, atom.episode_idx)] = src
    if all_dedup:
        for atom in group.atoms:
            key = (atom.exp_id, atom.episode_idx)
            registry.mark_dedup(run_ids[key], dedup_sources[key])
            counts["skipped_dedup"] += 1
        if verbose:
            _log(f"[dedup] {group.env}/{group.condition}/s{group.seed} "
                 f"→ {len(group.atoms)} ep skipped")
        return counts

    # 3) Claim les lignes → running
    for atom in group.atoms:
        registry.try_claim(run_ids[(atom.exp_id, atom.episode_idx)],
                           hostname=socket.gethostname(),
                           git_sha=_git_sha(REPO_ROOT))

    # 4) Préparer commande Hydra
    spec = spec_for(group.env, group.condition)
    atom_slug = (
        f"{group.atoms[0].exp_id}__{group.env}__{group.condition}"
        f"__s{group.seed}__{group.hp_hash}"
    )
    atom_out_dir = ATOM_DIR / atom_slug
    atom_out_dir.mkdir(parents=True, exist_ok=True)

    overrides_with_seed = dict(group.overrides)
    overrides_with_seed["seed"] = group.seed
    # NB : si le group a forcé un num_episodes (smoke, mini), on le respecte.
    # Sinon on garantit EPISODES_PER_SEED du protocole paper.
    if "num_episodes" not in overrides_with_seed:
        overrides_with_seed["num_episodes"] = EPISODES_PER_SEED
    # Hydra run dir : on redirige dans atom_out_dir pour garder les logs.
    hydra_overrides = as_hydra_args(overrides_with_seed) + [
        f"hydra.run.dir={atom_out_dir.as_posix()}",
    ]

    # Le config-path est déjà absolu (pointant vers scripts/paper/configs/<cond>)
    cmd: List[str] = [
        sys.executable, str(spec.main_py),
        "--config-path", spec.config_path.as_posix(),
        "--config-name", spec.config_name,
    ] + hydra_overrides

    if dry_run:
        _log("[DRY]", " ".join(shlex.quote(c) for c in cmd))
        for atom in group.atoms:
            counts["pending"] += 1
        return counts

    # 5) Écriture metadata + launch
    metadata = {
        "exp_id": group.atoms[0].exp_id,
        "env": group.env,
        "condition": group.condition,
        "seed": group.seed,
        "hp_hash": group.hp_hash,
        "llm_model": group.llm_model,
        "hostname": socket.gethostname(),
        "git_sha": _git_sha(REPO_ROOT),
        "cmd": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (atom_out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (atom_out_dir / "overrides.json").write_text(
        json.dumps(overrides_with_seed, sort_keys=True, indent=2, default=str)
    )

    stdout_path = atom_out_dir / "stdout.log"
    stderr_path = atom_out_dir / "stderr.log"
    t0 = time.time()
    returncode = 0
    # Injecter le modèle LLM via var d'env (utils.py lit PAPER_LLM_MODEL si
    # présent ; sinon fallback sur son default). Voir doc dans hyperparams.py.
    subproc_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",  # stack trace complète dans stderr.log
    }
    # Propager explicitement OPEN_ROUTER_KEY depuis l'environnement parent
    # rend le subprocess indépendant du placement exact du fichier .env.
    if "OPEN_ROUTER_KEY" in os.environ:
        subproc_env["OPEN_ROUTER_KEY"] = os.environ["OPEN_ROUTER_KEY"]
    if group.llm_model:
        subproc_env["PAPER_LLM_MODEL"] = group.llm_model
    # Heartbeat thread — keeps ``last_heartbeat_at`` fresh on every
    # claimed row while the subprocess runs. Lets the registry's
    # ``reset_stale_heartbeat`` reaper distinguish a healthy long
    # subprocess (still heartbeating) from a SIGKILL'd zombie
    # (heartbeat silent for >LEASE_TTL). The thread is daemon so it
    # dies with the parent and never blocks shutdown.
    hb_run_ids = list(run_ids.values())
    hb_stop = threading.Event()
    HEARTBEAT_INTERVAL_S = 30.0

    def _heartbeat_loop() -> None:
        while not hb_stop.wait(HEARTBEAT_INTERVAL_S):
            for hb_rid in hb_run_ids:
                try:
                    registry.heartbeat(hb_rid)
                except Exception:
                    # Heartbeat is best-effort — if the registry is
                    # locked or the row already moved out of running,
                    # we just skip and try again next tick.
                    pass

    hb_thread = threading.Thread(
        target=_heartbeat_loop, daemon=True, name=f"hb-{atom_slug[:32]}",
    )
    hb_thread.start()

    try:
        with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
            proc = subprocess.run(
                cmd, cwd=str(spec.cwd), stdout=so, stderr=se,
                env=subproc_env,
                timeout=3600 * 6,  # safety: 6h max par seed
            )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        returncode = -9
        _log(f"[TIMEOUT] {atom_slug}")
    except Exception as exc:
        returncode = -1
        _log(f"[EXC] {atom_slug} : {exc!r}")
    finally:
        hb_stop.set()
        hb_thread.join(timeout=2.0)
    wall = time.time() - t0

    # 6) Parser le log produit — Hydra crée outputs/paper_runs/runs/<slug>/main.log aussi
    log_candidates = [stdout_path, atom_out_dir / "main.log", atom_out_dir / "output.log"]
    log_text = ""
    for lc in log_candidates:
        if lc.exists():
            log_text += lc.read_text(errors="replace")
    parsed = parse_log(log_text)

    # 7) Écrire result.json (niveau 1 de résilience)
    result = {
        "returncode": returncode,
        "wall_seconds": wall,
        "rewards": parsed.rewards,
        "avg_reward": parsed.avg_reward,
        "steps_per_episode": parsed.steps_per_episode,
        "prompt_tokens": parsed.prompt_tokens,
        "completion_tokens": parsed.completion_tokens,
        "llm_calls": parsed.llm_calls,
        "elbo_final": parsed.elbo_final,
    }
    (atom_out_dir / "result.json").write_text(json.dumps(result, indent=2))

    # 8) Update registry — 1 ligne par épisode
    cost_per_ep = 0.0
    tok_p_per_ep = 0
    tok_c_per_ep = 0
    if len(parsed.rewards) > 0:
        tok_p_per_ep = parsed.prompt_tokens // max(1, len(parsed.rewards))
        tok_c_per_ep = parsed.completion_tokens // max(1, len(parsed.rewards))
        cost_per_ep = estimate_cost_usd(
            group.llm_model or "", tok_p_per_ep, tok_c_per_ep,
        )

    for atom in group.atoms:
        rid = run_ids[(atom.exp_id, atom.episode_idx)]
        if atom.episode_idx < len(parsed.rewards):
            r = parsed.rewards[atom.episode_idx]
            success = 1 if r > 0.05 else 0
            steps = (parsed.steps_per_episode[atom.episode_idx]
                     if atom.episode_idx < len(parsed.steps_per_episode)
                     else None)
            registry.mark_done(
                rid,
                reward=r, success=success, steps_to_goal=steps,
                elbo_final=parsed.elbo_final,
                prompt_tokens=tok_p_per_ep,
                completion_tokens=tok_c_per_ep,
                cost_usd=cost_per_ep,
                wall_seconds=wall / max(1, len(group.atoms)),
                output_dir=atom_out_dir.as_posix(),
            )
            counts["done"] += 1
        else:
            # L'épisode n'a pas été produit. On distingue 3 cas:
            # 1) ``rate_limited`` — OpenRouter rate limit (prioritaire au retry)
            # 2) ``interrupted`` — SIGTERM/SIGKILL externe (returncode -9/-15)
            #    typiquement nous-mêmes (pkill, timeout subprocess.run).
            #    Pas un vrai crash : les autres atomes du group peuvent
            #    avoir terminé legitimement (mark_done). Distinct de
            #    `failed` pour qu'on puisse mesurer le taux de bug
            #    réel du LLM-coded model vs le bruit infrastructure.
            # 3) ``failed`` — vrai crash subprocess (assertion, exception,
            #    return code positif non-zero).
            rate_sig = any(tok in log_text for tok in (
                "Rate limit exceeded", "429 Too Many Requests",
                "\"code\": 429", "rate-limit", "insufficient_quota",
            ))
            if returncode == 42 or rate_sig:
                status = "rate_limited"
            elif returncode in (-9, -15):
                status = "interrupted"
            else:
                status = "failed"
            registry.mark_failure(
                rid, status=status,
                error=f"returncode={returncode}, parsed {len(parsed.rewards)}/{len(group.atoms)} ep",
            )
            counts[status] += 1

    if verbose:
        _log(f"[{group.env}/{group.condition}/s{group.seed}] "
             f"rc={returncode} ep={len(parsed.rewards)}/{len(group.atoms)} "
             f"wall={wall:.1f}s tokens={parsed.prompt_tokens}+{parsed.completion_tokens}")
    return counts


def _log(*args: object) -> None:
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ATOM_DIR.mkdir(parents=True, exist_ok=True)
    registry = Registry(DB_PATH)

    envs = None
    if args.envs:
        envs = tuple(e.strip() for e in args.envs.split(","))

    atoms = build_atoms(args.exp, envs=envs)

    # Filtres optionnels
    if args.conditions:
        conds = set(c.strip() for c in args.conditions.split(","))
        atoms = [a for a in atoms if a.condition in conds]
    if args.seeds:
        seeds = _parse_int_range(args.seeds)
        atoms = [a for a in atoms if a.seed in seeds]

    groups = group_atoms(atoms)
    _log(f"[plan] {args.exp}: {len(atoms)} ep over {len(groups)} groups")
    if args.dry_run:
        for g in groups:
            _log(f"  {g.env}/{g.condition}/s{g.seed} hash={g.hp_hash} "
                 f"({len(g.atoms)} ep)")
        return 0

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    totals: Dict[str, int] = defaultdict(int)
    t_start = time.time()

    def _merge(counts: Dict[str, int]) -> None:
        for k, v in counts.items():
            totals[k] += v

    def _check_stop() -> bool:
        elapsed = time.time() - t_start
        if args.max_hours and elapsed > args.max_hours * 3600:
            _log(f"[stop] --max-hours {args.max_hours} reached")
            return True
        return False

    if workers == 1:
        # Sequential (v1) — comportement historique
        for i, g in enumerate(groups, 1):
            _log(f"[{i}/{len(groups)}] launching {g.env}/{g.condition}/s{g.seed}")
            counts = launch_group(g, registry=registry, dry_run=False, verbose=True)
            _merge(counts)
            elapsed = time.time() - t_start
            _log(f"  cum: {dict(totals)} | elapsed {elapsed:.0f}s")
            if _check_stop():
                break
    else:
        # Parallel (v2) — ProcessPoolExecutor. SQLite WAL + BEGIN IMMEDIATE
        # côté registry garantissent l'absence de double-claim inter-workers.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        _log(f"[parallel] workers={workers}")
        submitted = 0
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for g in groups:
                fut = pool.submit(_worker_launch, g)
                futures[fut] = g
                submitted += 1
            for fut in as_completed(futures):
                done += 1
                g = futures[fut]
                try:
                    counts = fut.result()
                except Exception as e:
                    _log(f"[exc] {g.env}/{g.condition}/s{g.seed}: {e!r}")
                    counts = {"failed": len(g.atoms)}
                _merge(counts)
                elapsed = time.time() - t_start
                _log(f"[{done}/{submitted}] {g.env}/{g.condition}/s{g.seed} "
                     f"{dict(counts)} | cum {dict(totals)} | {elapsed:.0f}s")
                if _check_stop():
                    # Sort du loop, les futures pending vont finir naturellement
                    # quand on sort du `with` (ou sont cancel si pas démarré)
                    break
    _log(f"[done] {dict(totals)}")
    return 0


def _worker_launch(group: "Group") -> Dict[str, int]:
    """Worker-process entrypoint pour ``ProcessPoolExecutor``.

    Chaque worker ouvre sa propre connexion SQLite (WAL mode) — supporte
    pleinement les accès concurrents.
    """
    registry = Registry(DB_PATH)
    return launch_group(group, registry=registry, dry_run=False, verbose=False)


def cmd_status(args: argparse.Namespace) -> int:
    registry = Registry(DB_PATH)
    counts = registry.status_counts(args.exp if args.exp else None)
    total = sum(counts.values())
    _log(f"[status] total={total} {counts}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    # Le resume est implicite : upsert_pending est idempotent, les lignes
    # failed/rate_limited sont relancées dès qu'on ré-exécute run_cmd sur la
    # même exp_id. On se contente d'appeler cmd_run.
    return cmd_run(args)


def _parse_int_range(spec: str) -> set[int]:
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            out.update(range(int(a), int(b) + 1))
        elif chunk:
            out.add(int(chunk))
    return out


def main() -> int:
    p = argparse.ArgumentParser(prog="paper_runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Lancer une expérience")
    pr.add_argument("exp", help="E1 | E2_offline | E2_online | E2b | E4 | all")
    pr.add_argument("--envs", default=None)
    pr.add_argument("--conditions", default=None)
    pr.add_argument("--seeds", default=None)
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--max-hours", type=float, default=None)
    pr.add_argument("--workers", type=int, default=1,
                    help="Nombre de subprocess parallèles (default 1 = sequential)")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", help="Compter les runs par status")
    ps.add_argument("--exp", default=None)
    ps.set_defaults(func=cmd_status)

    prs = sub.add_parser("resume", help="Reprendre pending/failed/rate_limited")
    prs.add_argument("exp", default="all", nargs="?")
    prs.add_argument("--envs", default=None)
    prs.add_argument("--conditions", default=None)
    prs.add_argument("--seeds", default=None)
    prs.add_argument("--dry-run", action="store_true")
    prs.add_argument("--max-hours", type=float, default=None)
    prs.add_argument("--workers", type=int, default=1)
    prs.set_defaults(func=cmd_resume)

    args = p.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
