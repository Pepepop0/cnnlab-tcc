from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.storage import (
    LATEST_VERSION,
    MIGRATIONS,
    SchemaVersionError,
    SQLiteRunStore,
    open_reader,
    open_writer,
    schema_dump,
)
from cnnlab.tcc.storage.connection import pragma_snapshot
from cnnlab.tcc.storage.schema import Migration, current_version, migrate
from cnnlab.tcc.storage.testing import PROTOCOL_DOC_SHA256


def test_migration_is_versioned_and_rerunnable(db_path: Path, artifact_root: Path) -> None:
    with SQLiteRunStore(db_path, artifact_root):
        pass
    conn = open_writer(db_path)
    try:
        assert current_version(conn) == LATEST_VERSION
        assert migrate(conn) == LATEST_VERSION  # second run is a verified no-op
        rows = conn.execute("SELECT version, name, sql_sha256 FROM schema_migrations").fetchall()
        assert rows == [(m.version, m.name, m.sha256) for m in MIGRATIONS]
        assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION
    finally:
        conn.close()
    with SQLiteRunStore(db_path, artifact_root):  # reopen after migration
        pass


def test_writer_pragmas_fk_wal_full_sync(store: SQLiteRunStore) -> None:
    snap = pragma_snapshot(store._conn)
    assert snap["foreign_keys"] == 1
    assert snap["journal_mode"] == "wal"
    assert snap["synchronous"] == 2  # FULL
    assert Path(str(store.db_path) + "-wal").exists()


def test_reader_is_read_only_and_not_blocked_by_writer(seeded: SQLiteRunStore) -> None:
    reader = open_reader(seeded.db_path)
    try:
        assert pragma_snapshot(reader)["foreign_keys"] == 1
        writer = seeded._conn
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO protocol_versions (protocol_id, document_sha256, status, created_at) "
            "VALUES ('p-uncommitted', ?, 'pilot', '2026-09-27T00:00:00.000000Z')",
            (PROTOCOL_DOC_SHA256,),
        )
        # WAL: the reader sees the last committed state while the writer holds its lock.
        ids = {r[0] for r in reader.execute("SELECT protocol_id FROM protocol_versions")}
        assert "p-uncommitted" not in ids and len(ids) == 1
        writer.execute("ROLLBACK")
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("DELETE FROM environments")
    finally:
        reader.close()


def test_reader_opens_after_writer_closed(seeded: SQLiteRunStore) -> None:
    seeded.close()  # e.g. Streamlit opened while the worker is stopped
    reader = open_reader(seeded.db_path)
    try:
        assert reader.execute("SELECT COUNT(*) FROM experiment_plans").fetchone()[0] == 1
    finally:
        reader.close()


def test_newer_database_is_refused(db_path: Path, artifact_root: Path) -> None:
    with SQLiteRunStore(db_path, artifact_root) as s:
        s._conn.execute(
            "INSERT INTO schema_migrations VALUES (?, 'from_the_future', ?, '2026-09-27T00:00:00.000000Z')",
            (LATEST_VERSION + 1, "0" * 64),
        )
    with pytest.raises(SchemaVersionError, match="newer"):
        SQLiteRunStore(db_path, artifact_root)


def test_edited_migration_is_refused(db_path: Path, artifact_root: Path) -> None:
    with SQLiteRunStore(db_path, artifact_root) as s:
        s._conn.execute("UPDATE schema_migrations SET sql_sha256 = ? WHERE version = 1", ("f" * 64,))
    with pytest.raises(SchemaVersionError, match="differs"):
        SQLiteRunStore(db_path, artifact_root)


def test_failed_migration_rolls_back_atomically(db_path: Path) -> None:
    conn = open_writer(db_path)
    try:
        bad = Migration(
            version=LATEST_VERSION + 1,
            name="broken",
            statements=("CREATE TABLE half_done (a INTEGER) STRICT", "INSERT INTO missing_table VALUES (1)"),
        )
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn, (*MIGRATIONS, bad))
        assert current_version(conn) == LATEST_VERSION
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'half_done'").fetchone() is None
    finally:
        conn.close()


def test_migrations_must_be_contiguous(db_path: Path) -> None:
    conn = open_writer(db_path)
    try:
        gap = Migration(version=LATEST_VERSION + 2, name="gap", statements=("SELECT 1",))
        with pytest.raises(SchemaVersionError):
            migrate(conn, (*MIGRATIONS, gap))
    finally:
        conn.close()


def test_in_memory_database_is_rejected() -> None:
    with pytest.raises(ValueError):
        open_writer(Path(":memory:"))


def test_database_inside_artifact_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SQLiteRunStore(tmp_path / "art" / "tcc.sqlite3", tmp_path / "art")


def test_schema_dump_is_deterministic_and_lists_migration(store: SQLiteRunStore) -> None:
    dump = schema_dump(store._conn)
    assert dump == schema_dump(store._conn)
    assert f"sha256={MIGRATIONS[0].sha256}" in dump
    assert "CREATE TABLE evaluation_attempts" in dump
