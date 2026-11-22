"""Migration 3: job events take the sentinel kinds (task A9). CPU only.

A database of schema v2 is evidence. The new code must lift it to v3 only when a writer is opened on purpose, keep every
event it had, and no interface that only looks may do it as a side effect.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.operation import launcher
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.schema import (
    JOB_EVENT_KINDS,
    JOB_EVENT_KINDS_V2,
    MIGRATIONS,
    SENTINEL_EVENT_KINDS,
    current_version,
)
from cnnlab.tcc.storage.testing import make_job, make_v2_copy, seed_reference_rows

JOB = make_job("run-m3", method="joint", seed_block=1)
MIGRATION_1_SHA256 = "18032baa71f3bff23f31f67a635297daedf2b1deef7134026557e05f4d1d4bb0"
MIGRATION_2_SHA256 = "13811d51ee10c5a9de709d099cf4c7c7aad122744be1e95b8b660ed99a9edf6b"
STAMP = "2026-10-06T00:00:00.000000Z"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def v3_db(tmp_path: Path) -> Path:
    """A database with one finished job (enqueued, claimed, done), at the latest schema."""
    db = tmp_path / "v3" / "tcc.sqlite3"
    with SQLiteRunStore(db, tmp_path / "artifacts") as store:
        seed_reference_rows(store)
        store.create_run(JOB)
        queue = JobQueue(store)
        queue.enqueue("job-1", JOB.run_id)
        lease = queue.claim_next("w", lease_s=60.0)
        assert lease is not None
        queue.finish(lease, "done", detail="ok")
    return db


@pytest.fixture
def v2_db(v3_db: Path, tmp_path: Path) -> Path:
    return make_v2_copy(v3_db, tmp_path / "v2" / "tcc.sqlite3")


def _insert(conn: sqlite3.Connection, kind: str) -> None:
    conn.execute(
        "INSERT INTO job_events (job_id, kind, from_state, to_state, detail_json, recorded_at) "
        f"VALUES ('job-1', '{kind}', 'done', 'done', '{{}}', '{STAMP}')"
    )


def test_the_kinds_are_the_old_ones_plus_the_sentinel_ones_and_the_old_migrations_do_not_move() -> None:
    assert JOB_EVENT_KINDS == JOB_EVENT_KINDS_V2 + SENTINEL_EVENT_KINDS
    assert SENTINEL_EVENT_KINDS == ("sentinel_measured", "sentinel_drift", "sentinel_failed")
    assert [m.version for m in MIGRATIONS] == [1, 2, 3] and MIGRATIONS[2].name == "job_events_sentinel_kinds"
    assert MIGRATIONS[0].sha256 == MIGRATION_1_SHA256 and MIGRATIONS[1].sha256 == MIGRATION_2_SHA256


def test_a_v2_database_cannot_hold_a_sentinel_event_and_a_v3_one_can(v3_db: Path, v2_db: Path) -> None:
    conn = sqlite3.connect(v2_db)
    try:
        assert current_version(conn) == 2
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert(conn, "sentinel_measured")
    finally:
        conn.close()
    with SQLiteRunStore(v3_db, v3_db.parent / "a") as store:  # a v3 writer accepts them
        assert current_version(store._conn) == 3
        store._conn.execute("BEGIN IMMEDIATE")
        _insert(store._conn, "sentinel_drift")
        store._conn.execute("ROLLBACK")


def test_a_writer_lifts_v2_to_v3_keeping_every_event_the_index_and_the_triggers(v2_db: Path) -> None:
    conn = sqlite3.connect(v2_db)
    before = conn.execute("SELECT * FROM job_events ORDER BY event_id").fetchall()
    conn.close()
    assert [r[2] for r in before] == ["enqueued", "claimed", "done"]

    with SQLiteRunStore(v2_db, v2_db.parent / "artifacts") as store:  # the deliberate step: a writer
        assert current_version(store._conn) == 3
        assert [tuple(r) for r in store._conn.execute("SELECT * FROM job_events ORDER BY event_id")] == [
            tuple(r) for r in before
        ]
        applied = [
            tuple(r) for r in store._conn.execute("SELECT version, sql_sha256 FROM schema_migrations ORDER BY version")
        ]
        assert applied == [(1, MIGRATION_1_SHA256), (2, MIGRATION_2_SHA256), (3, MIGRATIONS[2].sha256)]
        assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert store.check_integrity().ok
        names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE tbl_name = 'job_events'")}
        assert {"job_events_by_job", "job_events_no_update", "job_events_no_delete"} <= names
        assert not any("job_events_v" in n for n in names)  # the renamed table is gone
        assert JobQueue(store).get("job-1").state == "done"
        for statement in ("UPDATE job_events SET kind = 'done'", "DELETE FROM job_events"):
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                store._conn.execute(statement)
        store._conn.execute("BEGIN IMMEDIATE")
        _insert(store._conn, "sentinel_measured")
        store._conn.execute("COMMIT")
        # the counter continues after the last event it had
        assert (
            store._conn.execute("SELECT MAX(event_id) FROM job_events").fetchone()[0] == max(r[0] for r in before) + 1
        )


def test_nothing_that_only_looks_migrates_a_v2_database(
    v2_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cnnlab.tcc.operation import cli
    from cnnlab.tcc.operation.demo import EXIT_REFUSED

    before = digest(v2_db)
    with ReadModel(v2_db) as rm:  # the read model reads v2 as it is
        assert rm.schema_version() == 2 and rm.has_job_queue
        assert [e["kind"] for e in rm.job_events("job-1")] == ["enqueued", "claimed", "done"]
        assert rm.sentinel_events(JOB.run_id) == []  # no sentinel before v3, and no error
    art = tmp_path / "artifacts"
    assert cli.main(["status", "--db", str(v2_db)]) == 0
    assert cli.main(["verify", "--db", str(v2_db), "--artifacts", str(art)]) in (0, 2)
    cli.main(["export", "--db", str(v2_db), "--run", JOB.run_id, "--dest", str(tmp_path / "csv"), "--allow-synthetic"])
    cli.main(
        [
            "campaign",
            "curves",
            "--db",
            str(v2_db),
            "--dest",
            str(tmp_path / "hv"),
            "--reference",
            "1,2,3,4",
            "--allow-synthetic",
        ]
    )
    capsys.readouterr()
    refused = cli.main(["stop", "--db", str(v2_db), "--artifacts", str(art), "--job-id", "job-1", "--reason", "x"])
    assert refused == EXIT_REFUSED and "MigrationNeededError" in capsys.readouterr().err
    with pytest.raises(launcher.MigrationNeededError, match="would migrate"):
        launcher.ensure_current_schema(v2_db)
    assert digest(v2_db) == before  # the evidence is what it was
