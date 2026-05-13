"""Registre SQLite central pour les runs papier.

Clé unique : ``(exp_id, env, condition, seed, episode_idx, hp_hash)``.
Dédup inter-exp : si une ligne ``status='done'`` existe avec les 5 derniers
champs identiques (tous ``exp_id`` confondus), un nouvel insert est marqué
``skipped_dedup`` et les métriques sont recopiées. Garantit qu'un épisode
physiquement déjà calculé n'est jamais refait.

Lock : ``BEGIN IMMEDIATE`` → plusieurs workers / gits séparés peuvent partager
le même ``registry.db`` sans double-lancement.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)

# Base schema: new DBs get the full schema; older DBs get the ALTER TABLE
# migrations that follow.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  exp_id           TEXT    NOT NULL,
  env              TEXT    NOT NULL,
  condition        TEXT    NOT NULL,
  seed             INTEGER NOT NULL,
  episode_idx      INTEGER NOT NULL,
  hp_hash          TEXT    NOT NULL,
  llm_model        TEXT,
  status           TEXT    NOT NULL,
  reward           REAL,
  success          INTEGER,
  steps_to_goal    INTEGER,
  elbo_final       REAL,
  prompt_tokens    INTEGER,
  completion_tokens INTEGER,
  cost_usd         REAL,
  wall_seconds     REAL,
  started_at       TEXT,
  last_heartbeat_at TEXT,
  completed_at     TEXT,
  git_sha          TEXT,
  hostname         TEXT,
  output_dir       TEXT,
  error            TEXT,
  overrides_json   TEXT,
  UNIQUE (exp_id, env, condition, seed, episode_idx, hp_hash)
);
CREATE INDEX IF NOT EXISTS idx_dedup     ON runs (env, condition, seed, episode_idx, hp_hash, status);
CREATE INDEX IF NOT EXISTS idx_pending   ON runs (status);
CREATE INDEX IF NOT EXISTS idx_exp       ON runs (exp_id);
"""

# Schema additions that depend on columns introduced by ALTER TABLE
# migrations. Executed AFTER the migrations so they don't fail on a
# older database that does not have last_heartbeat_at yet.
POST_MIGRATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_heartbeat ON runs (status, last_heartbeat_at);

-- TRIGGER prevent_done_overwrite
-- A row that reached `done` with a valid (non-NULL) reward represents
-- a measured experiment result and MUST NEVER be silently demoted. The
-- ``mark_failure`` Python guard already enforces this, but the DB
-- trigger turns the invariant into an unbreakable contract regardless
-- of which client writes (CLI, ad-hoc SQL, future wrappers).  Any
-- demote attempt aborts with sqlite3.IntegrityError.
CREATE TRIGGER IF NOT EXISTS prevent_done_overwrite
BEFORE UPDATE OF status ON runs
WHEN OLD.status='done' AND OLD.reward IS NOT NULL
 AND NEW.status NOT IN ('done', 'skipped_dedup')
BEGIN
  SELECT RAISE(ABORT, 'cannot demote a done row with reward to non-done status');
END;

-- Append-only audit log of run transitions. Forensique
-- pour comprendre comment un atome a évolué sans avoir à scanner les
-- launch.log. NEVER updated/deleted under normal operation. Heartbeat
-- ticks are NOT logged here (too frequent, would pollute the table)
-- — only state transitions are.
CREATE TABLE IF NOT EXISTS events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      INTEGER NOT NULL,
  event_type  TEXT    NOT NULL,
  event_ts    TEXT    NOT NULL,
  actor_pid   INTEGER,
  hostname    TEXT,
  git_sha     TEXT,
  payload_json TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type, event_ts);

-- View `latest_result`: single source of truth for
-- downstream analyses (plot scripts, paper exports, win-rate
-- aggregations). For each (exp_id, env, condition, seed, ep), it
-- exposes exactly ONE row : the most recent ``done`` attempt with a
-- valid reward. Eliminates the "did I remember to dedup?" foot-gun
-- across all analysis code — `SELECT * FROM latest_result` is now
-- the authoritative read path.
CREATE VIEW IF NOT EXISTS latest_result AS
SELECT
    r1.id,
    r1.exp_id, r1.env, r1.condition, r1.seed, r1.episode_idx,
    r1.hp_hash, r1.llm_model,
    r1.reward, r1.success, r1.steps_to_goal, r1.elbo_final,
    r1.prompt_tokens, r1.completion_tokens, r1.cost_usd,
    r1.wall_seconds, r1.output_dir, r1.completed_at, r1.git_sha,
    r1.hostname
FROM runs r1
WHERE r1.status = 'done' AND r1.reward IS NOT NULL
  AND r1.completed_at = (
      SELECT MAX(r2.completed_at) FROM runs r2
      WHERE r2.exp_id = r1.exp_id
        AND r2.env = r1.env
        AND r2.condition = r1.condition
        AND r2.seed = r1.seed
        AND r2.episode_idx = r1.episode_idx
        AND r2.status = 'done' AND r2.reward IS NOT NULL
  );
"""

VALID_STATUSES = {
    "pending", "running", "done", "failed", "rate_limited", "skipped_dedup",
    # Distinct status for SIGTERM/SIGKILL (returncode -9/-15), so code
    # crashes can be distinguished from external interruptions.
    "interrupted",
}


class Registry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            # 1) Base schema (no-op on existing DB)
            c.executescript(SCHEMA_SQL)
            # 2) Additive migrations: ALTER TABLE for columns added
            #    after initial schema. SQLite has no ``IF NOT EXISTS``
            #    for ALTER, so we wrap in try/except.
            for stmt in (
                "ALTER TABLE runs ADD COLUMN last_heartbeat_at TEXT",
            ):
                try:
                    c.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already present
            # 3) Schema additions that depend on the migrated columns
            c.executescript(POST_MIGRATION_SQL)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path, timeout=30.0, isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        """Write transaction : BEGIN IMMEDIATE pour verrou cross-process."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    @staticmethod
    def _log_event(
        conn: sqlite3.Connection,
        run_id: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a row to the audit ``events`` table.

        Best-effort: if the events table is missing in an old DB or the
        insert fails for any reason, we swallow
        the exception silently. The audit log is observability, not
        ground truth — a transition that didn't make it to ``events``
        is still recorded by the ``runs.status`` change itself.
        """
        try:
            conn.execute(
                "INSERT INTO events (run_id, event_type, event_ts, "
                "actor_pid, hostname, git_sha, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, event_type, _now(),
                    os.getpid(), socket.gethostname(),
                    None,  # git_sha not always cheap to compute here
                    json.dumps(payload, sort_keys=True, default=str)
                    if payload else None,
                ),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Insert & claim
    # ------------------------------------------------------------------
    def upsert_pending(
        self,
        *,
        exp_id: str,
        env: str,
        condition: str,
        seed: int,
        episode_idx: int,
        hp_hash: str,
        llm_model: Optional[str],
        overrides: Dict[str, Any],
    ) -> int:
        """Insère une ligne `pending` (ou la retourne si déjà présente)."""
        with self._immediate() as conn:
            row = conn.execute(
                "SELECT id FROM runs WHERE exp_id=? AND env=? AND condition=? "
                "AND seed=? AND episode_idx=? AND hp_hash=?",
                (exp_id, env, condition, seed, episode_idx, hp_hash),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO runs (exp_id, env, condition, seed, episode_idx,"
                " hp_hash, llm_model, status, overrides_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    exp_id, env, condition, seed, episode_idx, hp_hash,
                    llm_model, json.dumps(overrides, sort_keys=True, default=str),
                ),
            )
            new_id = int(cur.lastrowid)
            self._log_event(conn, new_id, "created", payload={
                "exp_id": exp_id, "env": env, "condition": condition,
                "seed": seed, "episode_idx": episode_idx, "hp_hash": hp_hash,
            })
            return new_id

    def try_claim(self, run_id: int, hostname: str, git_sha: str) -> bool:
        """Passe `pending` → `running` atomiquement. Retourne False si déjà pris."""
        with self._immediate() as conn:
            cur = conn.execute(
                "UPDATE runs SET status='running', started_at=?, hostname=?, "
                "git_sha=? WHERE id=? AND status='pending'",
                (_now(), hostname, git_sha, run_id),
            )
            ok = cur.rowcount == 1
            if ok:
                self._log_event(conn, run_id, "claimed", payload={
                    "hostname": hostname, "git_sha": git_sha,
                })
            return ok

    def find_dedup_source(
        self,
        *,
        env: str,
        condition: str,
        seed: int,
        episode_idx: int,
        hp_hash: str,
        llm_model: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        """Cherche une ligne `done` ou `skipped_dedup` avec mêmes champs.

        ``skipped_dedup`` contient les metrics recopiées d'une source `done`
        d'une session antérieure, donc elles sont ré-utilisables pour une
        nouvelle dedup.

        The ``llm_model`` filter prevents E4 atoms with different LLMs
        from being skip-deduped against each other. When ``llm_model`` is
        None (legacy callers), the filter is skipped.
        """
        sql = (
            "SELECT * FROM runs WHERE env=? AND condition=? AND seed=? "
            "AND episode_idx=? AND hp_hash=? "
            "AND status IN ('done','skipped_dedup') AND reward IS NOT NULL "
        )
        params: tuple = (env, condition, seed, episode_idx, hp_hash)
        if llm_model is not None:
            # Match exact llm_model OR legacy NULL rows. Treat NULL as
            # "matches anything" for backward-compat with old runs.
            sql += "AND (llm_model = ? OR llm_model IS NULL) "
            params = params + (llm_model,)
        sql += "LIMIT 1"
        with self._conn() as conn:
            return conn.execute(sql, params).fetchone()

    def mark_dedup(self, run_id: int, source: sqlite3.Row) -> None:
        with self._immediate() as conn:
            conn.execute(
                "UPDATE runs SET status='skipped_dedup', reward=?, success=?, "
                "steps_to_goal=?, elbo_final=?, prompt_tokens=?, "
                "completion_tokens=?, cost_usd=?, wall_seconds=?, "
                "output_dir=?, completed_at=? WHERE id=?",
                (
                    source["reward"], source["success"], source["steps_to_goal"],
                    source["elbo_final"], source["prompt_tokens"],
                    source["completion_tokens"], source["cost_usd"],
                    source["wall_seconds"], source["output_dir"], _now(),
                    run_id,
                ),
            )
            self._log_event(conn, run_id, "dedup_archived", payload={
                "source_id": int(source["id"]),
                "source_reward": source["reward"],
            })

    def mark_done(
        self,
        run_id: int,
        *,
        reward: float,
        success: int,
        steps_to_goal: Optional[int],
        elbo_final: Optional[float],
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        wall_seconds: float,
        output_dir: str,
    ) -> None:
        with self._immediate() as conn:
            # Anti-regression: if the row is already `done` with a strictly
            # higher reward, KEEP the old result. This prevents a retry with
            # a worse outcome (e.g. ép 0 success replaced by ép 0 retry
            # failure that produced a nominal reward of 0) from clobbering
            # the better historical run.
            existing = conn.execute(
                "SELECT status, reward FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if (
                existing is not None
                and existing["status"] == "done"
                and existing["reward"] is not None
                and float(existing["reward"]) > float(reward)
            ):
                log.info(
                    "[REGISTRY] Keep old done (reward=%.3f) over new "
                    "(reward=%.3f) for rid=%d",
                    float(existing["reward"]), float(reward), run_id,
                )
                return
            conn.execute(
                "UPDATE runs SET status='done', reward=?, success=?, "
                "steps_to_goal=?, elbo_final=?, prompt_tokens=?, "
                "completion_tokens=?, cost_usd=?, wall_seconds=?, "
                "output_dir=?, completed_at=? WHERE id=?",
                (
                    reward, success, steps_to_goal, elbo_final, prompt_tokens,
                    completion_tokens, cost_usd, wall_seconds, output_dir,
                    _now(), run_id,
                ),
            )
            self._log_event(conn, run_id, "completed", payload={
                "reward": float(reward),
                "success": int(success),
                "wall_seconds": float(wall_seconds),
            })

    def mark_failure(
        self, run_id: int, *, status: str, error: str,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}")
        with self._immediate() as conn:
            # Critical anti-overwrite: a row that is already `done` with a
            # valid reward must never be demoted to `failed` / `rate_limited`.
            existing = conn.execute(
                "SELECT status, reward FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if (
                existing is not None
                and existing["status"] == "done"
                and existing["reward"] is not None
            ):
                log.warning(
                    "[REGISTRY] Refused to mark_failure rid=%d: already "
                    "done (reward=%.3f). Keeping old result.",
                    run_id, float(existing["reward"]),
                )
                return
            conn.execute(
                "UPDATE runs SET status=?, error=?, completed_at=? WHERE id=?",
                (status, error[:4000], _now(), run_id),
            )
            self._log_event(conn, run_id, status, payload={
                "error": error[:200],
            })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_pending(self, exp_id: Optional[str] = None) -> List[sqlite3.Row]:
        # `interrupted` rows are workers that were SIGTERM/SIGKILL'd
        # (returncode -9/-15) — they had not finished their atom and
        # need a clean retry, identical semantics to `failed` /
        # `rate_limited` for the wrapper but distinguishable in
        # diagnostics.
        retryable = "('pending','failed','rate_limited','interrupted')"
        with self._conn() as conn:
            if exp_id is None:
                return list(conn.execute(
                    f"SELECT * FROM runs WHERE status IN {retryable} ORDER BY id"
                ))
            return list(conn.execute(
                f"SELECT * FROM runs WHERE exp_id=? AND status IN {retryable} ORDER BY id",
                (exp_id,),
            ))

    def status_counts(self, exp_id: Optional[str] = None) -> Dict[str, int]:
        q = (
            "SELECT status, COUNT(*) AS n FROM runs "
            + ("WHERE exp_id=? " if exp_id else "")
            + "GROUP BY status"
        )
        args: Tuple[Any, ...] = (exp_id,) if exp_id else ()
        with self._conn() as conn:
            return {r["status"]: int(r["n"]) for r in conn.execute(q, args)}

    def heartbeat(self, run_id: int) -> bool:
        """Update ``last_heartbeat_at`` for a running row.

        Workers are expected to call this every ~30s while their atom
        runs. The reaper (:meth:`reset_stale_heartbeat`) uses the gap
        between ``now()`` and ``last_heartbeat_at`` to detect SIGKILL'd
        workers without the wide ``max_age_hours`` time-since-claim
        bucket — a short-running atom can finish in 5 min, so a 2h
        cliff would let a true zombie hold its claim for hours.

        Returns False if the row is no longer ``running`` (likely
        because the reaper already re-queued it) — caller should stop
        the worker for this atom in that case.
        """
        with self._immediate() as conn:
            cur = conn.execute(
                "UPDATE runs SET last_heartbeat_at=? "
                "WHERE id=? AND status='running'",
                (_now(), run_id),
            )
            return cur.rowcount == 1

    def reset_stale_heartbeat(
        self, exp_id_prefix: Optional[str] = None,
        max_silence_minutes: float = 10.0,
        target_status: str = "interrupted",
    ) -> int:
        """Reaper for ``running`` rows whose worker stopped heartbeating.

        Sharper than ``reset_stale_running`` because it tracks the
        worker's last sign of life rather than the wall time since
        claim. A 10-min silence is the default — a healthy worker
        heartbeats every 30s so 10 min of silence = either crash,
        SIGKILL, OOM, network partition, or process suspension.

        Marks reaped rows as ``interrupted`` (default) so we can tell
        them apart from rows that were genuinely re-queued for retry.
        Set ``target_status='pending'`` to re-queue silently in the
        legacy way (no diagnostic).
        """
        if target_status not in VALID_STATUSES:
            raise ValueError(f"Invalid target_status {target_status!r}")
        q = (
            "UPDATE runs SET status=?, error=COALESCE(error,'')"
            "||CASE WHEN error IS NULL OR error='' THEN '' ELSE '; ' END"
            "||'reaped: heartbeat silent for >'||?||' min', "
            "completed_at=? "
            "WHERE status='running' AND last_heartbeat_at IS NOT NULL "
            "AND (julianday('now') - julianday(last_heartbeat_at))*1440 > ?"
        )
        args: Tuple[Any, ...] = (target_status, max_silence_minutes, _now(), max_silence_minutes)
        if exp_id_prefix:
            q += " AND exp_id LIKE ?"
            args = args + (exp_id_prefix + "%",)
        with self._immediate() as conn:
            return conn.execute(q, args).rowcount

    def reset_stale_running(
        self, exp_id_prefix: Optional[str] = None,
        max_age_hours: float = 2.0,
    ) -> int:
        """Resync les rows 'running' anciennes (zombies de workers tués) → 'pending'.

        Robuste à un crash/SIGKILL : si un worker est tué SIGKILL, sa row reste
        en 'running' indéfiniment. Cette fonction remet en 'pending' les rows
        running plus vieilles que ``max_age_hours``.
        """
        q = (
            "UPDATE runs SET status='pending', started_at=NULL "
            "WHERE status='running' AND started_at IS NOT NULL "
            "AND (julianday('now') - julianday(started_at))*24 > ?"
        )
        args: Tuple[Any, ...] = (max_age_hours,)
        if exp_id_prefix:
            q += " AND exp_id LIKE ?"
            args = (max_age_hours, exp_id_prefix + "%")
        with self._immediate() as conn:
            return conn.execute(q, args).rowcount

    def dedup_pending_against_done(
        self, exp_id_prefix: Optional[str] = None,
    ) -> int:
        """Marque les rows ``pending``/``failed``/``rate_limited`` en
        ``skipped_dedup`` quand un sibling ``done`` (avec ``reward IS NOT NULL``)
        existe pour les MÊMES ``(exp_id, env, condition, seed, episode_idx)``,
        **quel que soit le ``hp_hash``**.

        Pourquoi c'est nécessaire alors que le schéma a déjà ``UNIQUE
        (exp_id, env, condition, seed, episode_idx, hp_hash)`` :
            la contrainte UNIQUE inclut ``hp_hash``. Quand le yaml config
            évolue entre deux sessions (kernel_bandwidth, sigma, etc.), le
            ``hp_hash`` change et SQLite autorise un nouvel ``upsert_pending``
            pour le même ep_seed (même `(env, condition, seed, ep)`). Sans ce
            cleanup, la nouvelle row ``pending`` reste pickable par
            ``list_pending`` → un worker la claim, exécute le run en entier,
            puis le ``find_dedup_source`` de fin recopie les métriques ↦
            CPU gaspillé pour rien.

        Exécutée au démarrage du wrapper, elle ramasse aussi les pending
        orphelins laissés par un ancien `reset_stale_running` qui aurait
        repassé en ``pending`` une row déjà couverte par un done sibling
        d'un autre ``hp_hash``.

        Returns:
            Nombre de rows mises à jour. Idempotente (rerun = 0 row).
        """
        with self._immediate() as conn:
            # Build LIKE filter for exp_id_prefix (no SQL injection: only
            # internal call sites pass a literal string).
            if exp_id_prefix:
                exp_filter = " AND exp_id LIKE ?"
                exp_args: Tuple[Any, ...] = (exp_id_prefix + "%",)
            else:
                exp_filter = ""
                exp_args = ()

            ghosts = conn.execute(
                f"""
                SELECT r1.id AS gid,
                       (SELECT id FROM runs r2
                        WHERE r2.status='done' AND r2.reward IS NOT NULL
                        AND r2.exp_id=r1.exp_id
                        AND r2.env=r1.env
                        AND r2.condition=r1.condition
                        AND r2.seed=r1.seed
                        AND r2.episode_idx=r1.episode_idx
                        ORDER BY r2.completed_at DESC LIMIT 1) AS source_id
                FROM runs r1
                WHERE r1.status IN ('pending', 'failed', 'rate_limited')
                {exp_filter}
                """,
                exp_args,
            ).fetchall()

            n = 0
            for ghost in ghosts:
                if ghost["source_id"] is None:
                    continue
                src = conn.execute(
                    "SELECT reward, success, steps_to_goal, elbo_final, "
                    "prompt_tokens, completion_tokens, cost_usd, "
                    "wall_seconds, output_dir FROM runs WHERE id=?",
                    (ghost["source_id"],),
                ).fetchone()
                if src is None:
                    continue
                cur = conn.execute(
                    "UPDATE runs SET status='skipped_dedup', reward=?, "
                    "success=?, steps_to_goal=?, elbo_final=?, "
                    "prompt_tokens=?, completion_tokens=?, cost_usd=?, "
                    "wall_seconds=?, output_dir=?, completed_at=? "
                    "WHERE id=? AND status IN ('pending','failed','rate_limited')",
                    (
                        src["reward"], src["success"], src["steps_to_goal"],
                        src["elbo_final"], src["prompt_tokens"],
                        src["completion_tokens"], src["cost_usd"],
                        src["wall_seconds"], src["output_dir"], _now(),
                        ghost["gid"],
                    ),
                )
                n += cur.rowcount
            if n > 0:
                log.info(
                    "[REGISTRY] dedup_pending_against_done: %d ghost rows "
                    "(pending/failed with sibling done) → skipped_dedup",
                    n,
                )
            return n

    def fetch_metrics(
        self, exp_id: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        q = (
            "SELECT * FROM runs WHERE status IN ('done','skipped_dedup') "
            + ("AND exp_id=? " if exp_id else "")
            + "ORDER BY env, condition, seed, episode_idx"
        )
        args: Tuple[Any, ...] = (exp_id,) if exp_id else ()
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def fetch_latest_results(
        self,
        exp_id: Optional[str] = None,
        exp_id_prefix: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """Return the deduplicated latest-done row for each ep_seed.

        Reads from the ``latest_result`` view, which guarantees ≤ 1 row
        per (exp_id, env, condition, seed, episode_idx). Use this for
        ALL downstream analyses (paper plots, win-rate stats, latex
        export) — it eliminates the foot-gun of double-counting rows
        that have multiple ``done`` siblings under different hp_hashes.

        Filter by exact ``exp_id`` or ``exp_id_prefix`` (LIKE 'X%').
        """
        if exp_id and exp_id_prefix:
            raise ValueError("Use either exp_id or exp_id_prefix, not both")
        q = "SELECT * FROM latest_result"
        args: Tuple[Any, ...] = ()
        if exp_id:
            q += " WHERE exp_id=?"
            args = (exp_id,)
        elif exp_id_prefix:
            q += " WHERE exp_id LIKE ?"
            args = (exp_id_prefix + "%",)
        q += " ORDER BY exp_id, env, condition, seed, episode_idx"
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def fetch_events(
        self,
        run_id: Optional[int] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        """Read the audit log. Useful for forensique post-incident.

        ``fetch_events(run_id=42)``      → all events for that run
        ``fetch_events(event_type='failed', limit=10)`` → 10 last failures
        """
        clauses: List[str] = []
        args: List[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            args.append(run_id)
        if event_type is not None:
            clauses.append("event_type=?")
            args.append(event_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        q = f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            return list(conn.execute(q, args))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
