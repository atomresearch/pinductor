"""Command-line driver for the paper's experiments.

Pipeline, in order:

  1. ``experiments.build_atoms`` enumerates the episodes an experiment
     needs, one atom each;
  2. :func:`group_atoms` merges the atoms that a single subprocess can
     produce — same environment, condition, seed, resolved overrides and
     LLM — since one run evaluates several episodes in a row;
  3. :func:`launch_group` records the group in the registry, skips it if
     the episodes already exist, otherwise runs one Hydra subprocess and
     writes one registry row per episode.

Everything durable lives in the registry (``outputs/paper_runs``), so an
interrupted campaign is resumed by re-running the same command: atoms
already marked done are skipped, and failed or interrupted ones are
retried. Nothing is deleted, so results survive a crash of the driver.

Each group also leaves a self-contained directory on disk holding the
exact command, the resolved overrides, the logs and a ``result.json``.
That is the second line of defence: the numbers can be rebuilt from those
files if the registry is lost.

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

# The sibling modules are imported by bare name, so this directory has to
# be on sys.path when the file is executed as a script.
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
    """Commit the run was launched from, or "unknown" outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Grouping: the atoms one subprocess can produce
# ---------------------------------------------------------------------------
@dataclass
class Group:
    env: str
    condition: str
    seed: int
    hp_hash: str
    llm_model: Optional[str]
    overrides: Dict[str, object]
    atoms: List[Atom]  # the episodes of this seed, possibly across exp_ids


def _yaml_content_hash(env: str, condition: str) -> str:
    """Stable hash of the generated config for ``(condition, env)``.

    Folded into ``hp_hash`` so that editing a config invalidates the
    deduplication cache: an episode run under an older config is no longer
    treated as a valid substitute for the current one.

    The YAML is parsed and re-dumped with sorted keys before hashing, so
    that only semantic changes count. Hashing raw bytes would rotate the
    hash on a comment, a reordering or trailing whitespace, and every
    already-computed episode would silently be re-run under a new hash.
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
        # Unparseable YAML, or PyYAML unavailable: fall back to hashing
        # raw bytes. Tagged with a prefix so registry rows produced by
        # this path are recognisable.
        return "L_" + hashlib.sha256(yaml_path.read_bytes()).hexdigest()[:10]


def group_atoms(atoms: Sequence[Atom]) -> List[Group]:
    """Group atoms by (env, condition, seed, hp_hash, llm_model).

    Atoms that fall in the same bucket are produced by one subprocess, and
    may come from different experiments: an episode computed for E1 also
    satisfies an identically configured E2 atom.

    ``llm_model`` is part of the key even though it is not part of
    ``hp_hash`` — it is passed by environment variable rather than as an
    override, so it leaves no trace in the hash. Without it, two atoms
    differing only by their LLM would merge, and the group would run
    whichever model the first atom happened to name.
    """
    buckets: Dict[Tuple[str, str, int, str, Optional[str]], Group] = {}
    for atom in atoms:
        overrides = resolve_overrides(
            atom.env, atom.condition,
            llm_model=atom.llm_model or DEFAULT_LLM,
            extra=atom.extra,
        )
        # Run identity = runtime overrides + frozen config contents.
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
# Running one group
# ---------------------------------------------------------------------------
def launch_group(
    group: Group, *, registry: Registry, dry_run: bool, verbose: bool,
) -> Dict[str, int]:
    """Run one group's subprocess and record its episodes in the registry.

    Returns a ``{status: count}`` tally over the group's atoms, which the
    caller accumulates into the campaign summary.

    The function is safe to call again on a group that partly succeeded:
    already-completed episodes are kept, the rest are retried.
    """
    counts: Dict[str, int] = defaultdict(int)
    # 1) One pending row per atom. Atoms from different experiments share
    # the subprocess but keep separate rows, so each experiment can be
    # queried on its own.
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

    # 1.5) Skip only when every atom of the group is already done. A
    # partially completed group is re-run: since ``Registry.mark_failure``
    # refuses to demote a done row that holds a reward, a second crash
    # cannot destroy the episodes that already succeeded.
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
        if verbose:
            _log(
                f"[all_done] {group.env}/{group.condition}/s{group.seed}: "
                f"{n_done_existing}/{len(group.atoms)} already done - skipping"
            )
        counts["skipped_partial_done"] = len(group.atoms)
        return counts

    # 2) Cross-experiment dedup. The subprocess is skipped only if every
    # episode of the group already exists elsewhere: a partial match still
    # requires the run, since one subprocess produces all of them at once.
    all_dedup = True
    dedup_sources: Dict[Tuple[str, int], object] = {}
    for atom in group.atoms:
        src = registry.find_dedup_source(
            env=group.env, condition=group.condition, seed=group.seed,
            episode_idx=atom.episode_idx, hp_hash=group.hp_hash,
            # Passed because the LLM is absent from hp_hash: without it,
            # an episode run with one model would satisfy an atom that
            # asked for another.
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

    # 3) Claim the rows so a concurrent worker cannot take the same group.
    for atom in group.atoms:
        registry.try_claim(run_ids[(atom.exp_id, atom.episode_idx)],
                           hostname=socket.gethostname(),
                           git_sha=_git_sha(REPO_ROOT))

    # 4) Build the Hydra command.
    spec = spec_for(group.env, group.condition)
    atom_slug = (
        f"{group.atoms[0].exp_id}__{group.env}__{group.condition}"
        f"__s{group.seed}__{group.hp_hash}"
    )
    atom_out_dir = ATOM_DIR / atom_slug
    atom_out_dir.mkdir(parents=True, exist_ok=True)

    overrides_with_seed = dict(group.overrides)
    overrides_with_seed["seed"] = group.seed
    # An experiment that set its own episode count keeps it; otherwise the
    # protocol default is pinned here rather than left to the YAML, so the
    # number of episodes matches the number of rows the group expects.
    if "num_episodes" not in overrides_with_seed:
        overrides_with_seed["num_episodes"] = EPISODES_PER_SEED
    # Redirect Hydra's output next to the group's own artefacts, instead
    # of its default dated directory under the working directory.
    hydra_overrides = as_hydra_args(overrides_with_seed) + [
        f"hydra.run.dir={atom_out_dir.as_posix()}",
    ]

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

    # 5) Write the metadata before launching, so an aborted run still
    # leaves the exact command and overrides on disk.
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
    subproc_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",  # full traceback in stderr.log
    }
    # Forward the API key explicitly rather than relying on dotenv. Each
    # codebase loads a .env relative to its own directory, and the two run
    # from different working directories, so a key file placed at the
    # repository root is invisible to the baseline. The failure is
    # misleading when it happens: the key reads as None, the request goes
    # out unauthenticated, and the error surfaces as a rate-limit message.
    if "OPEN_ROUTER_KEY" in os.environ:
        subproc_env["OPEN_ROUTER_KEY"] = os.environ["OPEN_ROUTER_KEY"]
    # The LLM travels as an environment variable, not a Hydra override;
    # see hyperparams.py. Left unset for conditions that use no LLM, which
    # then keep their own default.
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
                timeout=3600 * 6,  # generous: some atoms legitimately run for hours
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

    # 6) Read the metrics back out of the logs. All candidates are
    # concatenated because the log file Hydra writes depends on the
    # logging configuration of the codebase that ran.
    log_candidates = [stdout_path, atom_out_dir / "main.log", atom_out_dir / "output.log"]
    log_text = ""
    for lc in log_candidates:
        if lc.exists():
            log_text += lc.read_text(errors="replace")
    parsed = parse_log(log_text)

    # 7) Persist the parsed metrics next to the logs. Several analysis
    # scripts fall back to these files when a registry row is missing.
    result = {
        "returncode": returncode,
        "wall_seconds": wall,
        "rewards": parsed.rewards,
        "avg_reward": parsed.avg_reward,
        "steps_per_episode": parsed.steps_per_episode,
        "prompt_tokens": parsed.prompt_tokens,
        "completion_tokens": parsed.completion_tokens,
        "llm_calls": parsed.llm_calls,
        "likelihood_final": parsed.likelihood_final,
    }
    (atom_out_dir / "result.json").write_text(json.dumps(result, indent=2))

    # 8) One registry row per episode. Tokens and cost are reported per
    # run, so they are divided evenly across the episodes of the group;
    # the per-episode split is an average, not a measurement.
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
                likelihood_final=parsed.likelihood_final,
                prompt_tokens=tok_p_per_ep,
                completion_tokens=tok_c_per_ep,
                cost_usd=cost_per_ep,
                wall_seconds=wall / max(1, len(group.atoms)),
                output_dir=atom_out_dir.as_posix(),
            )
            counts["done"] += 1
        else:
            # The episode was never produced. The three statuses are kept
            # apart because they mean different things when reading the
            # results:
            #   rate_limited — the provider refused the request; retrying
            #     later is expected to work.
            #   interrupted  — the process received SIGTERM or SIGKILL,
            #     usually from the operator or from the timeout above.
            #     Infrastructure noise, not a defect of the run.
            #   failed       — the subprocess genuinely crashed.
            # Collapsing them would make the failure rate of the generated
            # models unmeasurable, since it would also count every process
            # that was killed for unrelated reasons.
            # Match what the two packages actually print. Both log the HTTP
            # status before reading the body, and raise a named error once
            # every key is out of credit; the earlier list looked for provider
            # strings that appear in no log this repository produces.
            rate_sig = any(tok in log_text for tok in (
                "Response status code: 429",
                "Response status code: 402",
                "OpenRouterOutOfCreditsError",
            ))
            if rate_sig:
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
    """Plan an experiment and run it, sequentially or across workers."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ATOM_DIR.mkdir(parents=True, exist_ok=True)
    registry = Registry(DB_PATH)

    envs = None
    if args.envs:
        envs = tuple(e.strip() for e in args.envs.split(","))

    atoms = build_atoms(args.exp, envs=envs)

    # Narrowing filters, applied to the atom list rather than to the
    # experiment definition, so a subset run keeps identical exp_ids and
    # hashes and its results merge with the full campaign.
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
        for i, g in enumerate(groups, 1):
            _log(f"[{i}/{len(groups)}] launching {g.env}/{g.condition}/s{g.seed}")
            counts = launch_group(g, registry=registry, dry_run=False, verbose=True)
            _merge(counts)
            elapsed = time.time() - t_start
            _log(f"  cum: {dict(totals)} | elapsed {elapsed:.0f}s")
            if _check_stop():
                break
    else:
        # Parallel path. Workers are processes, not threads, because each
        # group is itself a subprocess launch; the registry's WAL mode and
        # BEGIN IMMEDIATE transactions are what keep two workers from
        # claiming the same group.
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
                    # Leaving the loop; the pool's context manager waits
                    # for groups already started and drops the rest.
                    break
    _log(f"[done] {dict(totals)}")
    return 0


def _worker_launch(group: "Group") -> Dict[str, int]:
    """Worker-process entry point for ``ProcessPoolExecutor``.

    A ``Registry`` is built here rather than inherited: SQLite connections
    do not survive a fork, so each worker opens its own.
    """
    registry = Registry(DB_PATH)
    return launch_group(group, registry=registry, dry_run=False, verbose=False)


def cmd_status(args: argparse.Namespace) -> int:
    """Print how many registry rows sit in each status."""
    registry = Registry(DB_PATH)
    counts = registry.status_counts(args.exp if args.exp else None)
    total = sum(counts.values())
    _log(f"[status] total={total} {counts}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume a campaign.

    Identical to ``run``: resuming is a property of the registry, not a
    separate code path. ``upsert_pending`` is idempotent, done rows are
    skipped, and failed, rate-limited or interrupted ones are picked up
    again. The subcommand exists to make that intent explicit.
    """
    return cmd_run(args)


def _parse_int_range(spec: str) -> set[int]:
    """Parse a seed selection such as "0,2,5-7" into a set of integers."""
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

    pr = sub.add_parser("run", help="Run an experiment")
    pr.add_argument("exp", help="E1 | E2_offline | E2_online | E2b | E4 | all")
    pr.add_argument("--envs", default=None)
    pr.add_argument("--conditions", default=None)
    pr.add_argument("--seeds", default=None)
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--max-hours", type=float, default=None)
    pr.add_argument("--workers", type=int, default=1,
                    help="Number of parallel subprocesses (default 1 = sequential)")
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
