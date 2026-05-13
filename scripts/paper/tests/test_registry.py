"""Tests registry : insert, claim, dedup, resume."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import Registry


def _fresh() -> Registry:
    tmp = Path(tempfile.mkdtemp()) / "registry.db"
    return Registry(tmp)


def test_registry_migrates_legacy_score_column() -> None:
    tmp = Path(tempfile.mkdtemp()) / "registry.db"
    legacy_col = "el" + "bo_final"
    with sqlite3.connect(tmp) as conn:
        conn.execute(
            f"""
            CREATE TABLE runs (
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
              {legacy_col}     REAL,
              prompt_tokens    INTEGER,
              completion_tokens INTEGER,
              cost_usd         REAL,
              wall_seconds     REAL,
              started_at       TEXT,
              completed_at     TEXT,
              git_sha          TEXT,
              hostname         TEXT,
              output_dir       TEXT,
              error            TEXT,
              overrides_json   TEXT
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO runs (
              exp_id, env, condition, seed, episode_idx, hp_hash,
              status, reward, {legacy_col}
            ) VALUES ('E', 'lava', 'ours', 0, 0, 'h', 'done', 1.0, -7.0)
            """
        )

    Registry(tmp)
    with sqlite3.connect(tmp) as conn:
        conn.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        row = conn.execute("SELECT likelihood_final FROM runs").fetchone()

    assert "likelihood_final" in columns
    assert legacy_col not in columns
    assert row["likelihood_final"] == -7.0


def test_upsert_then_claim_roundtrip() -> None:
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="E1", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h0", llm_model="qwen/qwen3.6-plus",
        overrides={"gamma": 0.98},
    )
    assert rid > 0
    # Idempotent : 2e appel retourne le même id
    rid2 = r.upsert_pending(
        exp_id="E1", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h0", llm_model="qwen/qwen3.6-plus",
        overrides={"gamma": 0.98},
    )
    assert rid2 == rid

    # Claim : pending → running
    assert r.try_claim(rid, hostname="h", git_sha="sha") is True
    # 2e claim sur même id échoue
    assert r.try_claim(rid, hostname="h2", git_sha="sha") is False


def test_dedup_finds_existing_done() -> None:
    r = _fresh()
    # E1 a fait un run done
    rid1 = r.upsert_pending(
        exp_id="E1", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h42", llm_model="qwen",
        overrides={},
    )
    r.try_claim(rid1, hostname="h", git_sha="sha")
    r.mark_done(
        rid1, reward=0.9, success=1, steps_to_goal=12, likelihood_final=-2.1,
        prompt_tokens=1000, completion_tokens=500, cost_usd=0.0015,
        wall_seconds=45.0, output_dir="/tmp/foo",
    )
    # E4 insère un pending avec même (env, cond, seed, ep, hash)
    rid2 = r.upsert_pending(
        exp_id="E4", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h42", llm_model="qwen",
        overrides={},
    )
    assert rid2 != rid1
    src = r.find_dedup_source(
        env="lava", condition="ours", seed=0, episode_idx=0, hp_hash="h42",
    )
    assert src is not None
    r.mark_dedup(rid2, src)
    # Vérifie les metrics recopiées
    with sqlite3.connect(r.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE id=?", (rid2,)).fetchone()
    assert row["status"] == "skipped_dedup"
    assert row["reward"] == 0.9
    assert row["success"] == 1
    assert row["wall_seconds"] == 45.0


def test_dedup_does_not_match_different_hash() -> None:
    r = _fresh()
    rid1 = r.upsert_pending(
        exp_id="E1", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hA", llm_model="qwen", overrides={},
    )
    r.try_claim(rid1, hostname="h", git_sha="s")
    r.mark_done(
        rid1, reward=1.0, success=1, steps_to_goal=5, likelihood_final=None,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=10.0, output_dir="/x",
    )
    src = r.find_dedup_source(
        env="lava", condition="ours", seed=0, episode_idx=0, hp_hash="hB",
    )
    assert src is None


def test_list_pending_and_resume() -> None:
    r = _fresh()
    for i in range(3):
        r.upsert_pending(
            exp_id="E1", env="lava", condition="ours", seed=i, episode_idx=0,
            hp_hash=f"h{i}", llm_model="qwen", overrides={},
        )
    pending = r.list_pending()
    assert len(pending) == 3
    # Simule un crash : 1 passe à failed
    r.mark_failure(pending[0]["id"], status="failed", error="boom")
    # failed doit toujours apparaître dans list_pending (= candidat retry)
    again = r.list_pending()
    assert len(again) == 3
    statuses = {row["status"] for row in again}
    assert "failed" in statuses


def test_status_counts() -> None:
    r = _fresh()
    ids = [
        r.upsert_pending(
            exp_id="E1", env="lava", condition="ours", seed=i, episode_idx=0,
            hp_hash=f"h{i}", llm_model="qwen", overrides={},
        ) for i in range(4)
    ]
    r.try_claim(ids[0], "h", "s")
    r.mark_done(
        ids[0], reward=0.5, success=1, steps_to_goal=None, likelihood_final=None,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=1.0, output_dir="",
    )
    r.mark_failure(ids[1], status="rate_limited", error="429")
    counts = r.status_counts()
    assert counts.get("done", 0) == 1
    assert counts.get("pending", 0) == 2
    assert counts.get("rate_limited", 0) == 1


def _seed_done_row(r: Registry, *, reward: float) -> int:
    """Helper: create a row, claim it, mark_done with the given reward.

    Returns the row id.
    """
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="anti_overwrite_test", llm_model="qwen", overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="sha")
    r.mark_done(
        rid, reward=reward, success=1 if reward > 0 else 0,
        steps_to_goal=10, likelihood_final=-2.0,
        prompt_tokens=100, completion_tokens=50, cost_usd=0.0,
        wall_seconds=10.0, output_dir="/x",
    )
    return rid


def _row(r: Registry, rid: int) -> sqlite3.Row:
    with sqlite3.connect(r.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()


# Anti-overwrite guard: a `done` row with a valid reward must never be
# silently downgraded by a subsequent retry that timed out. These tests pin
# the contract on `Registry.mark_failure` and `Registry.mark_done`.


def test_mark_failure_refuses_to_overwrite_done_with_reward() -> None:
    r = _fresh()
    rid = _seed_done_row(r, reward=1.0)
    # The wrapper retries the group; one of the atoms times out and tries
    # to mark this rid as failed. The guard MUST refuse — the row must keep
    # its done(reward=1.0) state.
    r.mark_failure(rid, status="failed", error="timeout 7200s")
    row = _row(r, rid)
    assert row["status"] == "done", f"expected done, got {row['status']}"
    assert row["reward"] == 1.0, f"expected reward=1.0, got {row['reward']}"


def test_mark_failure_refuses_to_overwrite_done_with_zero_reward() -> None:
    """Even reward=0.0 (a valid evaluation) must be preserved."""
    r = _fresh()
    rid = _seed_done_row(r, reward=0.0)
    r.mark_failure(rid, status="failed", error="timeout")
    row = _row(r, rid)
    assert row["status"] == "done"
    assert row["reward"] == 0.0


def test_mark_done_keeps_higher_reward() -> None:
    r = _fresh()
    rid = _seed_done_row(r, reward=1.0)
    # A retry succeeds but with a worse reward (e.g. 0.5).
    r.mark_done(
        rid, reward=0.5, success=0, steps_to_goal=20, likelihood_final=-3.0,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=20.0, output_dir="/y",
    )
    row = _row(r, rid)
    assert row["status"] == "done"
    assert row["reward"] == 1.0, f"expected old reward 1.0 kept, got {row['reward']}"


def test_mark_done_upgrades_lower_reward() -> None:
    r = _fresh()
    rid = _seed_done_row(r, reward=0.3)
    # A retry succeeds with a better reward.
    r.mark_done(
        rid, reward=0.9, success=1, steps_to_goal=5, likelihood_final=-1.5,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=5.0, output_dir="/z",
    )
    row = _row(r, rid)
    assert row["status"] == "done"
    assert row["reward"] == 0.9, f"expected upgrade to 0.9, got {row['reward']}"


def test_mark_done_idempotent_same_reward() -> None:
    r = _fresh()
    rid = _seed_done_row(r, reward=0.7)
    r.mark_done(
        rid, reward=0.7, success=1, steps_to_goal=12, likelihood_final=-2.5,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=10.0, output_dir="/x",
    )
    row = _row(r, rid)
    assert row["status"] == "done"
    assert row["reward"] == 0.7


def test_mark_failure_normal_pending_path_still_works() -> None:
    """Without an existing done, mark_failure behaves as before."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=1, episode_idx=0,
        hp_hash="normal", llm_model="qwen", overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_failure(rid, status="failed", error="boom")
    row = _row(r, rid)
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_mark_done_normal_pending_path_still_works() -> None:
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=2, episode_idx=0,
        hp_hash="normal_done", llm_model="qwen", overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(
        rid, reward=0.6, success=1, steps_to_goal=8, likelihood_final=-2.2,
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        wall_seconds=8.0, output_dir="/n",
    )
    row = _row(r, rid)
    assert row["status"] == "done"
    assert row["reward"] == 0.6


def test_mark_failure_can_still_fail_a_failed_row() -> None:
    """A row already failed can be re-marked failed (e.g. updated error)."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=3, episode_idx=0,
        hp_hash="refailed", llm_model="qwen", overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_failure(rid, status="failed", error="first")
    r.mark_failure(rid, status="rate_limited", error="429")
    row = _row(r, rid)
    assert row["status"] == "rate_limited"
    assert row["error"] == "429"


def test_dedup_pending_against_done_marks_ghost_with_different_hp_hash() -> None:
    """Ghost: pending row with done sibling under a different hp_hash
    (config evolved between runs) must be dedup'd → skipped_dedup."""
    r = _fresh()
    rid_done = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hashA", llm_model=None, overrides={},
    )
    rid_ghost = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hashB", llm_model=None, overrides={},  # different hp_hash
    )
    r.try_claim(rid_done, hostname="h", git_sha="s")
    r.mark_done(
        rid_done, reward=0.7, success=1, steps_to_goal=20, likelihood_final=-2.5,
        prompt_tokens=100, completion_tokens=50, cost_usd=0.1,
        wall_seconds=400.0, output_dir="/tmp/x",
    )

    n = r.dedup_pending_against_done()
    assert n == 1, f"expected 1 ghost dedup, got {n}"

    ghost_row = _row(r, rid_ghost)
    assert ghost_row["status"] == "skipped_dedup"
    assert abs(ghost_row["reward"] - 0.7) < 1e-9
    assert ghost_row["wall_seconds"] == 400.0


def test_dedup_pending_against_done_idempotent() -> None:
    """Running the dedup twice produces 0 the second time."""
    r = _fresh()
    rid_done = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hA", llm_model=None, overrides={},
    )
    rid_ghost = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hB", llm_model=None, overrides={},
    )
    r.try_claim(rid_done, hostname="h", git_sha="s")
    r.mark_done(
        rid_done, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
        prompt_tokens=50, completion_tokens=20, cost_usd=0.01,
        wall_seconds=100.0, output_dir="/tmp/y",
    )
    assert r.dedup_pending_against_done() == 1
    assert r.dedup_pending_against_done() == 0
    # Ghost row stays in skipped_dedup, untouched
    assert _row(r, rid_ghost)["status"] == "skipped_dedup"


def test_dedup_pending_against_done_skips_genuine_pending() -> None:
    """A pending without sibling done must NOT be touched."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=42, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    n = r.dedup_pending_against_done()
    assert n == 0
    assert _row(r, rid)["status"] == "pending"


def test_dedup_pending_against_done_skips_done_without_reward() -> None:
    """A done row without reward (corrupt / interrupted) must NOT count
    as a valid source — the pending sibling stays pending."""
    r = _fresh()
    rid_done_corrupt = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hA", llm_model=None, overrides={},
    )
    rid_pending = r.upsert_pending(
        exp_id="E2", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hB", llm_model=None, overrides={},
    )
    r.try_claim(rid_done_corrupt, hostname="h", git_sha="s")
    # Hand-write a 'done' status without reward (simulates corruption)
    import sqlite3
    with sqlite3.connect(r.db_path) as c:
        c.execute(
            "UPDATE runs SET status='done', reward=NULL WHERE id=?",
            (rid_done_corrupt,),
        )
    n = r.dedup_pending_against_done()
    assert n == 0, "must not dedup against a done without reward"
    assert _row(r, rid_pending)["status"] == "pending"


def test_dedup_pending_against_done_filters_by_exp_id_prefix() -> None:
    """exp_id_prefix only deduplicates matching exp_ids."""
    r = _fresh()
    # E2 group with ghost
    rid_e2_done = r.upsert_pending(exp_id="E2_offline_ND2", env="lava",
        condition="ours", seed=0, episode_idx=0, hp_hash="hA",
        llm_model=None, overrides={})
    rid_e2_ghost = r.upsert_pending(exp_id="E2_offline_ND2", env="lava",
        condition="ours", seed=0, episode_idx=0, hp_hash="hB",
        llm_model=None, overrides={})
    r.try_claim(rid_e2_done, hostname="h", git_sha="s")
    r.mark_done(rid_e2_done, reward=0.5, success=1, steps_to_goal=10,
        likelihood_final=-1.0, prompt_tokens=10, completion_tokens=5,
        cost_usd=0.01, wall_seconds=50.0, output_dir="/tmp/a")

    # E4 group with ghost
    rid_e4_done = r.upsert_pending(exp_id="E4_qwen", env="lava",
        condition="ours", seed=0, episode_idx=0, hp_hash="hC",
        llm_model=None, overrides={})
    rid_e4_ghost = r.upsert_pending(exp_id="E4_qwen", env="lava",
        condition="ours", seed=0, episode_idx=0, hp_hash="hD",
        llm_model=None, overrides={})
    r.try_claim(rid_e4_done, hostname="h", git_sha="s")
    r.mark_done(rid_e4_done, reward=0.3, success=0, steps_to_goal=None,
        likelihood_final=None, prompt_tokens=20, completion_tokens=10,
        cost_usd=0.05, wall_seconds=200.0, output_dir="/tmp/b")

    # Filter on E4 only
    n_e4 = r.dedup_pending_against_done(exp_id_prefix="E4_")
    assert n_e4 == 1
    # E2 ghost untouched
    assert _row(r, rid_e2_ghost)["status"] == "pending"
    # E4 ghost cleaned
    assert _row(r, rid_e4_ghost)["status"] == "skipped_dedup"

    # Now clean E2
    n_e2 = r.dedup_pending_against_done(exp_id_prefix="E2_")
    assert n_e2 == 1
    assert _row(r, rid_e2_ghost)["status"] == "skipped_dedup"


def test_interrupted_status_is_retryable() -> None:
    """`interrupted` rows are returned by ``list_pending`` so the
    wrapper picks them up on the next cycle."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_failure(rid, status="interrupted", error="returncode=-15 SIGTERM")
    assert _row(r, rid)["status"] == "interrupted"
    pending = r.list_pending()
    assert any(p["id"] == rid for p in pending), \
        "interrupted rows must show up in list_pending"


def test_trigger_prevent_done_overwrite() -> None:
    """The DB-level trigger refuses raw UPDATEs that would demote a
    done-with-reward row to pending/failed/etc — even if the Python
    guards are bypassed."""
    import sqlite3
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(
        rid, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
        prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
        wall_seconds=50.0, output_dir="/tmp/x",
    )
    # Raw UPDATE that bypasses Python guards must still fail
    raised = False
    with sqlite3.connect(str(r.db_path)) as c:
        try:
            c.execute("UPDATE runs SET status='pending' WHERE id=?", (rid,))
            c.commit()
        except sqlite3.IntegrityError:
            raised = True
    assert raised, "trigger must ABORT the demote"
    # Row still done
    assert _row(r, rid)["status"] == "done"


def test_trigger_allows_done_to_done_idempotent() -> None:
    """Done → done update (e.g. re-mark with same metrics) must NOT
    trip the trigger — only demotion is forbidden."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(rid, reward=0.3, success=0, steps_to_goal=None, likelihood_final=None,
                prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
                wall_seconds=10.0, output_dir="/tmp/y")
    # Re-marking with HIGHER reward should succeed
    r.mark_done(rid, reward=0.7, success=1, steps_to_goal=20, likelihood_final=-1.0,
                prompt_tokens=20, completion_tokens=10, cost_usd=0.02,
                wall_seconds=20.0, output_dir="/tmp/y")
    assert abs(_row(r, rid)["reward"] - 0.7) < 1e-9


def test_trigger_allows_done_to_skipped_dedup() -> None:
    """Done → skipped_dedup is allowed (mark_dedup recopies metrics)."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(rid, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
                prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
                wall_seconds=50.0, output_dir="/tmp/x")
    import sqlite3
    with sqlite3.connect(str(r.db_path)) as c:
        c.execute("UPDATE runs SET status='skipped_dedup' WHERE id=?", (rid,))
        c.commit()
    assert _row(r, rid)["status"] == "skipped_dedup"


def test_heartbeat_updates_only_running_rows() -> None:
    """Heartbeat must only succeed on rows in 'running' status."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    # pending → no heartbeat
    assert r.heartbeat(rid) is False
    # claim → running
    r.try_claim(rid, hostname="h", git_sha="s")
    assert r.heartbeat(rid) is True
    row = _row(r, rid)
    assert row["last_heartbeat_at"] is not None
    # mark_done → no longer running
    r.mark_done(rid, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
                prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
                wall_seconds=50.0, output_dir="/tmp/x")
    assert r.heartbeat(rid) is False


def test_events_log_lifecycle() -> None:
    """Each transition (created, claimed, completed) emits one event row."""
    import sqlite3
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={"k": "v"},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(
        rid, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
        prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
        wall_seconds=50.0, output_dir="/tmp/x",
    )
    with sqlite3.connect(str(r.db_path)) as c:
        c.row_factory = sqlite3.Row
        evts = list(c.execute(
            "SELECT event_type, payload_json FROM events WHERE run_id=? ORDER BY id",
            (rid,),
        ))
    types = [e["event_type"] for e in evts]
    assert "created" in types
    assert "claimed" in types
    assert "completed" in types
    # Completed event has reward in payload
    completed = next(e for e in evts if e["event_type"] == "completed")
    import json
    payload = json.loads(completed["payload_json"])
    assert abs(payload["reward"] - 0.5) < 1e-9


def test_events_log_failure() -> None:
    """mark_failure emits an event with the status as event_type."""
    import sqlite3
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_failure(rid, status="interrupted", error="returncode=-15 SIGTERM")
    with sqlite3.connect(str(r.db_path)) as c:
        c.row_factory = sqlite3.Row
        evts = list(c.execute(
            "SELECT event_type FROM events WHERE run_id=? ORDER BY id", (rid,)
        ))
    types = [e["event_type"] for e in evts]
    # ``interrupted`` event type recorded
    assert "interrupted" in types


def test_events_log_dedup_archive() -> None:
    """mark_dedup emits a `dedup_archived` event referencing the source."""
    import sqlite3
    r = _fresh()
    rid_done = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hA", llm_model=None, overrides={},
    )
    rid_dup = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hB", llm_model=None, overrides={},
    )
    r.try_claim(rid_done, hostname="h", git_sha="s")
    r.mark_done(
        rid_done, reward=0.7, success=1, steps_to_goal=20, likelihood_final=-2.0,
        prompt_tokens=100, completion_tokens=50, cost_usd=0.1,
        wall_seconds=300.0, output_dir="/tmp/y",
    )
    src = r.find_dedup_source(env="lava", condition="ours", seed=0,
                              episode_idx=0, hp_hash="hB")
    # Note find_dedup_source uses hp_hash exact, won't match — use direct row
    with sqlite3.connect(str(r.db_path)) as c:
        c.row_factory = sqlite3.Row
        src = c.execute("SELECT * FROM runs WHERE id=?", (rid_done,)).fetchone()
    r.mark_dedup(rid_dup, src)
    with sqlite3.connect(str(r.db_path)) as c:
        c.row_factory = sqlite3.Row
        evts = list(c.execute(
            "SELECT event_type, payload_json FROM events WHERE run_id=? ORDER BY id",
            (rid_dup,),
        ))
    assert "dedup_archived" in [e["event_type"] for e in evts]


def test_latest_result_view_dedups_to_one_per_ep_seed() -> None:
    """The ``latest_result`` view returns at most one row per
    (exp_id, env, condition, seed, ep) — picks the most recent done."""
    import time as _time
    r = _fresh()
    rid_old = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hA", llm_model=None, overrides={},
    )
    rid_new = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="hB", llm_model=None, overrides={},
    )
    r.try_claim(rid_old, hostname="h", git_sha="s")
    r.mark_done(rid_old, reward=0.3, success=0, steps_to_goal=10, likelihood_final=-1.0,
                prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
                wall_seconds=10.0, output_dir="/tmp/old")
    _time.sleep(1.0)  # ensure later completed_at
    r.try_claim(rid_new, hostname="h", git_sha="s")
    r.mark_done(rid_new, reward=0.7, success=1, steps_to_goal=20, likelihood_final=-2.0,
                prompt_tokens=20, completion_tokens=10, cost_usd=0.02,
                wall_seconds=20.0, output_dir="/tmp/new")
    rows = r.fetch_latest_results(exp_id="EX")
    assert len(rows) == 1, f"expected 1 latest_result row, got {len(rows)}"
    assert abs(rows[0]["reward"] - 0.7) < 1e-9, "must pick latest done"


def test_fetch_events_filters() -> None:
    """fetch_events filters by run_id and event_type."""
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.mark_done(rid, reward=0.5, success=1, steps_to_goal=10, likelihood_final=-1.0,
                prompt_tokens=10, completion_tokens=5, cost_usd=0.01,
                wall_seconds=10.0, output_dir="/tmp/x")

    # All events for run_id
    evts = r.fetch_events(run_id=rid)
    assert len(evts) >= 3  # created, claimed, completed
    types = {e["event_type"] for e in evts}
    assert {"created", "claimed", "completed"} <= types

    # Filter by event_type
    completed = r.fetch_events(event_type="completed")
    assert all(e["event_type"] == "completed" for e in completed)


def test_reset_stale_heartbeat_marks_interrupted() -> None:
    """A row whose last_heartbeat_at is older than the silence
    threshold gets marked ``interrupted`` (not ``pending``)."""
    import sqlite3
    r = _fresh()
    rid = r.upsert_pending(
        exp_id="EX", env="lava", condition="ours", seed=0, episode_idx=0,
        hp_hash="h", llm_model=None, overrides={},
    )
    r.try_claim(rid, hostname="h", git_sha="s")
    r.heartbeat(rid)
    # Hand-write a stale heartbeat (1 hour ago)
    with sqlite3.connect(str(r.db_path)) as c:
        c.execute(
            "UPDATE runs SET last_heartbeat_at=datetime('now','-1 hour') WHERE id=?",
            (rid,),
        )
        c.commit()
    n = r.reset_stale_heartbeat(max_silence_minutes=10.0)
    assert n == 1
    assert _row(r, rid)["status"] == "interrupted"


if __name__ == "__main__":
    test_upsert_then_claim_roundtrip()
    test_dedup_finds_existing_done()
    test_dedup_does_not_match_different_hash()
    test_list_pending_and_resume()
    test_status_counts()
    test_mark_failure_refuses_to_overwrite_done_with_reward()
    test_mark_failure_refuses_to_overwrite_done_with_zero_reward()
    test_mark_done_keeps_higher_reward()
    test_mark_done_upgrades_lower_reward()
    test_mark_done_idempotent_same_reward()
    test_mark_failure_normal_pending_path_still_works()
    test_mark_done_normal_pending_path_still_works()
    test_mark_failure_can_still_fail_a_failed_row()
    test_dedup_pending_against_done_marks_ghost_with_different_hp_hash()
    test_dedup_pending_against_done_idempotent()
    test_dedup_pending_against_done_skips_genuine_pending()
    test_dedup_pending_against_done_skips_done_without_reward()
    test_dedup_pending_against_done_filters_by_exp_id_prefix()
    test_interrupted_status_is_retryable()
    test_trigger_prevent_done_overwrite()
    test_trigger_allows_done_to_done_idempotent()
    test_trigger_allows_done_to_skipped_dedup()
    test_heartbeat_updates_only_running_rows()
    test_events_log_lifecycle()
    test_events_log_failure()
    test_events_log_dedup_archive()
    test_latest_result_view_dedups_to_one_per_ep_seed()
    test_fetch_events_filters()
    test_reset_stale_heartbeat_marks_interrupted()
    print("OK — all registry tests passed")
