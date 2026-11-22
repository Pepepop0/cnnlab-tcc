from __future__ import annotations

import sqlite3

import pytest

from cnnlab.tcc.contracts import EvaluationResult
from cnnlab.tcc.storage import ConsistencyError, NotFoundError, SQLiteRunStore
from cnnlab.tcc.storage.schema import APPEND_ONLY_TABLES
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, run_candidate

TS = "2026-09-27T00:00:00Z"
A = make_job("run-a")
B = make_job("run-b", seed_block=1)


@pytest.fixture
def two_runs(seeded: SQLiteRunStore) -> SQLiteRunStore:
    seeded.create_run(A)
    seeded.create_run(B)
    run_candidate(seeded, A, "a1", "a1.t1", make_genome(1), FakeEvaluator(gpu_seconds=6.0))
    run_candidate(seeded, A, "a2", "a2.t1", make_genome(2), FakeEvaluator(gpu_seconds=6.0))
    seeded.register_candidate(A.run_id, "a3", make_genome(3), {})
    seeded.commit_evaluation(
        EvaluationResult(A.run_id, "a3", "a3.t1", "invalid", None, 2.0, failure_stage="train", failure_code="nan_loss")
    )
    seeded.register_candidate(B.run_id, "b1", make_genome(4), {})
    return seeded


def test_foreign_keys_are_enforced(two_runs: SQLiteRunStore) -> None:
    conn = two_runs._conn
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(
            "INSERT INTO candidates (candidate_id, run_id, seq, genome_json, phenotype_hash, "
            "cache_context_hash, space_hash, metadata_json, record_sha256, created_at) "
            "VALUES ('z', 'no-run', 1, '{}', 'p', 'c', 's', '{}', ?, ?)",
            ("0" * 64, TS),
        )
    # Attempt of run A for a candidate of run B: composite FK (run_id, candidate_id).
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(
            "INSERT INTO evaluation_attempts (attempt_id, run_id, candidate_id, attempt_number, status, started_at) "
            "VALUES ('cross', ?, 'b1', 1, 'running', ?)",
            (A.run_id, TS),
        )
    two_runs.begin_attempt(B.run_id, "b1", "b1.t1")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(
            "INSERT INTO gpu_segments (run_id, attempt_id, gpu_seconds, source, recorded_at) "
            "VALUES (?, 'b1.t1', 1.0, 'runner', ?)",
            (A.run_id, TS),
        )
    with pytest.raises(sqlite3.IntegrityError, match="running attempt"):
        conn.execute(
            "INSERT INTO gpu_segments (run_id, attempt_id, gpu_seconds, source, recorded_at) "
            "VALUES (?, 'a1.t1', 1.0, 'runner', ?)",
            (A.run_id, TS),
        )
    with pytest.raises(sqlite3.IntegrityError, match="created as running"):
        conn.execute(
            "INSERT INTO evaluation_attempts (attempt_id, run_id, candidate_id, attempt_number, status, "
            "gpu_seconds, started_at) VALUES ('direct', ?, 'b1', 1, 'valid', 0, ?)",
            (B.run_id, TS),
        )
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_append_only_tables_reject_update_and_delete(two_runs: SQLiteRunStore) -> None:
    populated = [t for t in APPEND_ONLY_TABLES if two_runs._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]]
    assert {"runs", "candidates", "measurements", "artifacts", "gpu_segments", "datasets"} <= set(populated)
    for table in populated:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            two_runs._conn.execute(f"UPDATE {table} SET rowid = rowid")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            two_runs._conn.execute(f"DELETE FROM {table}")


def test_front_accepts_only_valid_candidates(two_runs: SQLiteRunStore) -> None:
    two_runs.record_front(A.run_id, ("a1", "a2"), 12.0, 0.42)
    with pytest.raises(ConsistencyError, match="no valid attempt"):
        two_runs.record_front(A.run_id, ("a1", "a3"), 14.0, 0.40)
    with pytest.raises(NotFoundError):
        two_runs.record_front(A.run_id, ("b1",), 14.0, None)
    with pytest.raises(ConsistencyError, match="increasing"):
        two_runs.record_front(A.run_id, ("a1",), 5.0, None)
    with pytest.raises(ConsistencyError, match="ledger"):
        two_runs.record_front(A.run_id, ("a1",), 15.0, None)
    with pytest.raises(ValueError):
        two_runs.record_front(A.run_id, ("a1", "a1"), 14.0, None)
    two_runs.record_front(A.run_id, (), 14.0, None)  # empty front is a valid observation
    snap = two_runs._conn.execute(
        "SELECT unique_valid_evaluations, hypervolume FROM front_snapshots ORDER BY snapshot_id"
    ).fetchall()
    assert [tuple(r) for r in snap] == [(2, 0.42), (2, None)]
    with pytest.raises(sqlite3.IntegrityError, match="valid attempt"):
        two_runs._conn.execute("INSERT INTO front_members VALUES (1, ?, 'a3')", (A.run_id,))


def test_full_workflow_passes_integrity_check(two_runs: SQLiteRunStore) -> None:
    report = two_runs.check_integrity()
    assert report.ok, report
    assert report.sqlite_messages == ("ok",) and report.running_attempts == ()
