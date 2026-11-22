"""`ReadModel` (tcc-read/1): read-only queries for TUI/Streamlit, synthetic data only."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.storage import (
    LATEST_VERSION,
    JobQueue,
    NotFoundError,
    ReadModel,
    ReadModelError,
    SQLiteRunStore,
    open_read_model,
)
from cnnlab.tcc.storage.schema import LATEST_VERSION
from cnnlab.tcc.storage.testing import FakeEvaluator, cache_hit_result, make_genome, make_job, run_candidate

JOB = make_job("run-read", method="mono", seed_block=5)


@pytest.fixture
def populated(seeded: SQLiteRunStore) -> SQLiteRunStore:
    s = seeded
    s.create_run(JOB)
    bad = make_genome(1).cache_context_hash
    evaluator = FakeEvaluator(gpu_seconds=10.0, invalid_contexts=frozenset({bad}))
    run_candidate(s, JOB, "c0", "c0.a1", make_genome(0), evaluator, generation=0)
    run_candidate(s, JOB, "c1", "c1.a1", make_genome(1), evaluator, generation=0)
    source = s.find_cached(JOB.run_id, make_genome(0).cache_context_hash)
    s.register_candidate(JOB.run_id, "c2", make_genome(0), {"generation": 1})
    s.commit_evaluation(cache_hit_result(source, candidate_id="c2", attempt_id="c2.a1"))
    s.register_candidate(JOB.run_id, "c3", make_genome(3), {"generation": 1})
    s.begin_attempt(JOB.run_id, "c3", "c3.a1")
    s.record_gpu_segment(JOB.run_id, "c3.a1", 1.5)
    s.recover_interrupted_attempts(JOB.run_id)
    s.record_front(JOB.run_id, ("c0",), s.active_gpu_seconds(JOB.run_id), None)
    JobQueue(s).enqueue("job-read", JOB.run_id)
    return s


def test_evaluations_expose_measured_fitness_only(populated: SQLiteRunStore) -> None:
    with ReadModel(populated.db_path) as rm:
        rows = {r["attempt_id"]: r for r in rm.evaluations(JOB.run_id)}
    valid = rows["c0.a1"]
    assert all(math.isfinite(valid[k]) for k in ("e", "l_ms", "v_mb", "s_mb"))
    assert 0 <= valid["e"] <= 1 and valid["l_ms"] > 0 and valid["v_mb"] >= 0 and valid["s_mb"] > 0
    assert valid["synthetic"] is True and valid["gpu_seconds"] == 10.0
    hit = rows["c2.a1"]
    assert hit["cache_hit"] and hit["source_attempt_id"] == "c0.a1" and hit["gpu_seconds"] == 0.0
    assert (hit["e"], hit["l_ms"], hit["v_mb"], hit["s_mb"]) == (
        valid["e"],
        valid["l_ms"],
        valid["v_mb"],
        valid["s_mb"],
    )
    for failed in ("c1.a1", "c3.a1"):  # never a zero or a guess
        assert (rows[failed]["e"], rows[failed]["l_ms"], rows[failed]["v_mb"], rows[failed]["s_mb"]) == (None,) * 4
    assert rows["c1.a1"]["failure_code"] == "synthetic_build_failure"
    assert rows["c3.a1"]["recovered"] and rows["c3.a1"]["gpu_seconds"] == 1.5


def test_progress_runs_fronts_and_jobs(populated: SQLiteRunStore) -> None:
    with ReadModel(populated.db_path) as rm:
        assert rm.version == "tcc-read/1" and rm.schema_version() == LATEST_VERSION
        progress = rm.run_progress(JOB.run_id)
        assert progress["active_gpu_s"] == pytest.approx(21.5)
        assert progress["attempts"] == {"valid": 2, "invalid": 1, "interrupted": 1}
        assert progress["unique_valid_evaluations"] == 1 and progress["cache_hits"] == 1
        assert progress["synthetic_results"] == 3  # valid, invalid, hit; the recovered one is not a result
        [run] = rm.runs()
        assert run["run_id"] == JOB.run_id and run["stage"] == "confirmatory" and run["system_version"] is None
        assert rm.fronts(JOB.run_id)[0]["candidate_ids"] == ["c0"]
        [job] = rm.jobs()
        assert job["state"] == "queued" and job["lease_live"] is None
        assert [e["kind"] for e in rm.job_events("job-read")] == ["enqueued"]
        with pytest.raises(NotFoundError):
            rm.run_progress("nope")


def test_read_model_cannot_write_and_does_not_block_the_worker(populated: SQLiteRunStore) -> None:
    queue = JobQueue(populated)
    lease = queue.claim_next("w", lease_s=60)
    with ReadModel(populated.db_path) as rm:
        with pytest.raises(sqlite3.OperationalError):
            rm._conn.execute("UPDATE jobs SET priority = 1")
        assert rm.jobs()[0]["lease_live"] is True
        queue.heartbeat(lease)  # writer proceeds while the viewer is open
        assert rm.jobs()[0]["heartbeat_count"] == 1
    assert queue.get("job-read").lease_owner == "w"


def test_read_model_needs_an_existing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ReadModel(tmp_path / "missing.sqlite3")


# -- open_read_model: operator-supplied paths ---------------------------------


def _code(path: object) -> str:
    with pytest.raises(ReadModelError) as info:
        open_read_model(path)  # type: ignore[arg-type]
    return info.value.code


def test_open_read_model_explains_bad_paths_and_creates_nothing(tmp_path: Path) -> None:
    assert _code(None) == "empty_path" and _code("   ") == "empty_path"
    missing = tmp_path / "nope" / "tcc.sqlite3"
    assert _code(missing) == "not_found"
    assert not missing.parent.exists()  # a viewer never creates a database
    assert _code(tmp_path) == "not_a_file"
    text = tmp_path / "notes.txt"
    text.write_text("this is not a database, just some text " * 40, encoding="utf-8")
    assert _code(text) == "not_sqlite"
    assert text.read_text(encoding="utf-8").startswith("this is not a database")


def test_open_read_model_refuses_foreign_old_and_newer_schemas(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.sqlite3"
    conn = sqlite3.connect(foreign)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE something (a INTEGER)")
    conn.commit()
    conn.close()
    assert _code(foreign) == "not_tcc"

    unmigrated = tmp_path / "unmigrated.sqlite3"  # the bookkeeping table exists, no migration was applied
    conn = sqlite3.connect(unmigrated)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, sql_sha256 TEXT, applied_at TEXT)"
    )
    conn.commit()
    conn.close()
    assert _code(unmigrated) == "schema_too_old"

    newer = tmp_path / "db" / "newer.sqlite3"
    with SQLiteRunStore(newer, tmp_path / "art") as s:
        s._conn.execute(
            "INSERT INTO schema_migrations VALUES (?, 'from_the_future', ?, '2026-09-27T00:00:00.000000Z')",
            (LATEST_VERSION + 1, "0" * 64),
        )
    assert _code(newer) == "schema_too_new"


def test_open_read_model_opens_a_valid_database_read_only(populated: SQLiteRunStore) -> None:
    before = populated.db_path.read_bytes()
    with open_read_model(str(populated.db_path)) as rm:
        assert rm.schema_version() == LATEST_VERSION and len(rm.jobs()) == 1
        assert rm._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        for statement in (
            "INSERT INTO jobs (job_id) VALUES ('x')",
            "DELETE FROM evaluation_attempts",
            "CREATE TABLE t (a)",
            "PRAGMA query_only = OFF; DELETE FROM jobs",
        ):
            with pytest.raises((sqlite3.OperationalError, sqlite3.ProgrammingError, sqlite3.Warning)):
                rm._conn.execute(statement)
    assert populated.db_path.read_bytes() == before  # main file untouched by the viewer
