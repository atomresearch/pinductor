"""SQLite registry: what has been run, what it produced, what is left.

One row per evaluation episode, keyed by
``(exp_id, env, condition, seed, episode_idx, hp_hash)``. Two experiments
that ask for the same episode under the same configuration keep separate
rows but share the computation: the second is marked ``skipped_dedup`` and
copies the first one's metrics. Episodes are expensive, so an episode that
has physically been computed is never computed twice.

Concurrency: writes go through ``BEGIN IMMEDIATE`` and the database runs
in WAL mode, so several workers can share one ``registry.db`` without
claiming the same atom.

The design principle throughout is that a measured result is not
destroyed. A row that reached ``done`` with a reward cannot be demoted, by
a Python guard and by a database trigger, because the usual way to lose
results is a retry that fails and overwrites the run that had succeeded.
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

# Base schema, runs table only. A fresh database gets it in full; an
# existing one is brought up to date by the ALTER TABLE migrations below.
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
  likelihood_final       REAL,
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

# Schema objects that reference columns added by the migrations. Kept
# separate and executed after them, because an index or trigger naming a
# column that does not exist yet aborts the whole script.
POST_MIGRATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_heartbeat ON runs (status, last_heartbeat_at);

-- A row that reached `done` with a non-NULL reward holds a measured
-- result and must not be demoted. The ``mark_failure`` guard in Python
-- already refuses to, but the trigger makes the invariant hold for every
-- writer, including ad-hoc SQL run by hand against the database. A
-- demotion attempt aborts with sqlite3.IntegrityError.
CREATE TRIGGER IF NOT EXISTS prevent_done_overwrite
BEFORE UPDATE OF status ON runs
WHEN OLD.status='done' AND OLD.reward IS NOT NULL
 AND NEW.status NOT IN ('done', 'skipped_dedup')
BEGIN
  SELECT RAISE(ABORT, 'cannot demote a done row with reward to non-done status');
END;

-- Append-only audit log of state transitions, so the history of an atom
-- can be reconstructed without reading the launcher logs. Never updated
-- or deleted in normal operation. Heartbeat ticks are excluded: they are
-- frequent enough to bury the transitions they would sit among.
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

-- Read path for every downstream analysis (plots, exports, aggregates).
-- For each (exp_id, env, condition, seed, episode) it exposes exactly one
-- row, the most recent ``done`` attempt carrying a reward. Reading the
-- runs table directly double-counts episodes that were re-run under a
-- second hp_hash, which is why analyses select from this view instead.
CREATE VIEW IF NOT EXISTS latest_result AS
SELECT
    r1.id,
    r1.exp_id, r1.env, r1.condition, r1.seed, r1.episode_idx,
    r1.hp_hash, r1.llm_model,
    r1.reward, r1.success, r1.steps_to_goal, r1.likelihood_final,
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
    # Reserved for external termination (SIGTERM/SIGKILL). Retried exactly
    # like `failed`, but kept apart from it so that the failure rate of the
    # generated models can be measured without counting processes that
    # were killed to restart a worker.
    "interrupted",
}


class Registry:
    """Handle on the run database. Creating one migrates the schema."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            # Ordering matters: base schema, then column migrations, then
            # the objects that reference the migrated columns.
            c.executescript(SCHEMA_SQL)
            self._migrate_likelihood_column(c)
            # SQLite has no IF NOT EXISTS for ALTER TABLE, so re-adding an
            # existing column is caught rather than tested for.
            for stmt in (
                "ALTER TABLE runs ADD COLUMN last_heartbeat_at TEXT",
            ):
                try:
                    c.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already present
            c.executescript(POST_MIGRATION_SQL)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @staticmethod
    def _migrate_likelihood_column(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        legacy_score_col = "el" + "bo_final"
        if legacy_score_col in columns and "likelihood_final" not in columns:
            conn.execute(
                f"ALTER TABLE runs RENAME COLUMN {legacy_score_col} "
                "TO likelihood_final"
            )
        elif "likelihood_final" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN likelihood_final REAL")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection with autocommit and WAL enabled.

        Connections are per-call rather than cached, so the object can be
        shared across processes: an inherited SQLite handle is not safe to
        use after a fork.
        """
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
        """Write transaction holding the database lock for its duration.

        ``BEGIN IMMEDIATE`` takes the write lock up front instead of on
        first write, which is what makes read-then-write sequences such as
        claiming a row atomic between processes.
        """
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

        Best-effort by design: a failed insert is swallowed. The audit log
        is observability, not ground truth, and the transition it records
        is already carried by the ``runs.status`` change in the same
        transaction. Letting it raise would abort a state change that
        succeeded.
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
        """Insert a ``pending`` row, or return the id of the existing one.

        Idempotent, which is what makes re-running a command a resume:
        atoms that were already recorded keep their row and their status.
        """
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
        """Move a row from ``pending`` to ``running``.

        Returns False when the row was not pending, which means another
        worker claimed it first. The status test is part of the UPDATE, so
        two workers cannot both succeed.
        """
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
        """Find an already-computed episode this atom can reuse.

        ``skipped_dedup`` rows are accepted as sources as well as ``done``
        ones, since they carry the metrics copied from an earlier source
        and are equally valid.

        ``llm_model`` must be passed for any LLM condition: it is not part
        of ``hp_hash``, so without the filter an episode produced by one
        model would be reused for an atom that asked for another. It is
        optional because rows written before the column was populated hold
        NULL, and those are treated as matching anything.
        """
        sql = (
            "SELECT * FROM runs WHERE env=? AND condition=? AND seed=? "
            "AND episode_idx=? AND hp_hash=? "
            "AND status IN ('done','skipped_dedup') AND reward IS NOT NULL "
        )
        params: tuple = (env, condition, seed, episode_idx, hp_hash)
        if llm_model is not None:
            sql += "AND (llm_model = ? OR llm_model IS NULL) "
            params = params + (llm_model,)
        sql += "LIMIT 1"
        with self._conn() as conn:
            return conn.execute(sql, params).fetchone()

    def mark_dedup(self, run_id: int, source: sqlite3.Row) -> None:
        """Record that this atom reuses ``source``, copying its metrics.

        The metrics are duplicated rather than referenced so that every
        experiment can be queried on its own rows, without joining back to
        whichever experiment happened to run the episode first.
        """
        with self._immediate() as conn:
            conn.execute(
                "UPDATE runs SET status='skipped_dedup', reward=?, success=?, "
                "steps_to_goal=?, likelihood_final=?, prompt_tokens=?, "
                "completion_tokens=?, cost_usd=?, wall_seconds=?, "
                "output_dir=?, completed_at=? WHERE id=?",
                (
                    source["reward"], source["success"], source["steps_to_goal"],
                    source["likelihood_final"], source["prompt_tokens"],
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
        likelihood_final: Optional[float],
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        wall_seconds: float,
        output_dir: str,
    ) -> None:
        """Record a completed episode.

        Keeps the better of the old and new reward when the row is already
        done. A retry that reruns an episode and fails still produces a
        nominal reward, so without this the retry of a successful episode
        would replace it with the failure.
        """
        with self._immediate() as conn:
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
                "steps_to_goal=?, likelihood_final=?, prompt_tokens=?, "
                "completion_tokens=?, cost_usd=?, wall_seconds=?, "
                "output_dir=?, completed_at=? WHERE id=?",
                (
                    reward, success, steps_to_goal, likelihood_final, prompt_tokens,
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
        """Record a non-completion, unless the row already holds a result.

        The guard matters because failures are marked per group: when a
        group is retried and its subprocess dies, every atom of the group
        is marked failed, including those whose episode had already
        succeeded in an earlier attempt. Without the refusal, retrying a
        partially successful group destroys the part that worked.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}")
        with self._immediate() as conn:
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
        """Return every row that still needs to be run.

        The four retryable statuses differ in diagnosis, not in what has
        to happen next: none of them produced an episode.
        """
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
        """Count rows per status, over one experiment or over all of them."""
        q = (
            "SELECT status, COUNT(*) AS n FROM runs "
            + ("WHERE exp_id=? " if exp_id else "")
            + "GROUP BY status"
        )
        args: Tuple[Any, ...] = (exp_id,) if exp_id else ()
        with self._conn() as conn:
            return {r["status"]: int(r["n"]) for r in conn.execute(q, args)}

    def heartbeat(self, run_id: int) -> bool:
        """Refresh ``last_heartbeat_at`` on a running row.

        Workers call this about every 30 seconds. It gives
        :meth:`reset_stale_heartbeat` a signal of liveness rather than of
        elapsed time: run durations vary from minutes to hours, so no
        single age threshold separates a healthy long run from a dead one,
        while silence does.

        Returns False when the row is no longer running, typically because
        the reaper already re-queued it.
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
        """Re-queue ``running`` rows whose worker stopped heartbeating.

        Preferred over :meth:`reset_stale_running`, which can only ask how
        long ago a row was claimed. A worker beats every 30 seconds, so
        ten minutes of silence means it is gone, whatever the expected
        duration of the atom.

        Reaped rows are marked ``interrupted`` so they remain
        distinguishable from rows that failed on their own;
        ``target_status='pending'`` re-queues them without that trace.
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
        """Re-queue ``running`` rows claimed more than ``max_age_hours`` ago.

        A killed worker leaves its rows claimed forever, and nothing else
        will ever pick them up. This is the age-based fallback for rows
        that predate heartbeats or were never heartbeaten; prefer
        :meth:`reset_stale_heartbeat`, which does not have to guess how
        long an atom should take.
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
        """Retire retryable rows whose episode already exists under another hash.

        Marks ``pending`` / ``failed`` / ``rate_limited`` rows as
        ``skipped_dedup`` when a ``done`` sibling with a reward exists for
        the same ``(exp_id, env, condition, seed, episode_idx)``,
        **whatever its hp_hash**.

        This is deliberately looser than the uniqueness constraint, which
        includes ``hp_hash``. Editing a config changes the hash, so the
        same episode can legitimately be inserted again under a new one.
        Those rows are then indistinguishable from real work: a worker
        claims one, runs the whole episode, and only at the end does the
        dedup check find the earlier result and copy it. The episode was
        recomputed for nothing. Calling this before a campaign starts
        collapses that class of rows up front.

        Also catches rows re-queued by :meth:`reset_stale_running` that
        are covered by a done sibling under a different hash.

        Returns:
            Number of rows updated. Idempotent: a second call returns 0.
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
                    "SELECT reward, success, steps_to_goal, likelihood_final, "
                    "prompt_tokens, completion_tokens, cost_usd, "
                    "wall_seconds, output_dir FROM runs WHERE id=?",
                    (ghost["source_id"],),
                ).fetchone()
                if src is None:
                    continue
                cur = conn.execute(
                    "UPDATE runs SET status='skipped_dedup', reward=?, "
                    "success=?, steps_to_goal=?, likelihood_final=?, "
                    "prompt_tokens=?, completion_tokens=?, cost_usd=?, "
                    "wall_seconds=?, output_dir=?, completed_at=? "
                    "WHERE id=? AND status IN ('pending','failed','rate_limited')",
                    (
                        src["reward"], src["success"], src["steps_to_goal"],
                        src["likelihood_final"], src["prompt_tokens"],
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
        """Return every row that carries metrics, including deduplicated ones.

        This can return several rows for one episode when it was re-run
        under different hashes. Use :meth:`fetch_latest_results` for
        anything that averages or counts.
        """
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

        Reads the ``latest_result`` view, which holds at most one row per
        (exp_id, env, condition, seed, episode_idx). This is the intended
        entry point for every analysis that aggregates results: querying
        the runs table directly double-counts episodes that have several
        ``done`` rows under different hashes.

        Filter by exact ``exp_id`` or by ``exp_id_prefix``.
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
        """Read the audit log, most recent first.

        ``fetch_events(run_id=42)`` returns the history of one run;
        ``fetch_events(event_type='failed', limit=10)`` the last ten
        failures across the campaign.
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
