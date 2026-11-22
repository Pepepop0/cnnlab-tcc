"""Backup of the live WAL database + artifact catalog, and restore rehearsal.

Only synthetic data (`storage.testing`). The sources are checked to be
byte-identical after every backup: backup and restore never repair,
delete or overwrite evidence.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import SearchCheckpoint
from cnnlab.tcc.storage import (
    BackupError,
    JobQueue,
    SQLiteRunStore,
    create_backup,
    open_reader,
    publish_bytes,
    rehearse_restore,
    restore_backup,
    verify_backup,
)
from cnnlab.tcc.storage.backup import DB_FILENAME, MANIFEST_FILENAME, load_manifest, table_digests
from cnnlab.tcc.storage.schema import LATEST_VERSION
from cnnlab.tcc.storage.testing import (
    FakeEvaluator,
    cache_hit_result,
    make_genome,
    make_job,
    run_candidate,
)

JOB = make_job("run-backup", method="sequential", seed_block=4)


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _live_digests(db_path: Path) -> dict[str, dict[str, object]]:
    reader = open_reader(db_path)
    try:
        return table_digests(reader)
    finally:
        reader.close()


@pytest.fixture
def populated(seeded: SQLiteRunStore) -> SQLiteRunStore:
    """Valid, invalid, cache-hit and interrupted attempts, checkpoint, front, a job."""
    store = seeded
    store.create_run(JOB)
    bad = make_genome(2).cache_context_hash
    evaluator = FakeEvaluator(gpu_seconds=10.0, invalid_contexts=frozenset({bad}))
    for i in range(3):
        run_candidate(store, JOB, f"c{i}", f"c{i}.a1", make_genome(i), evaluator)
    source = store.find_cached(JOB.run_id, make_genome(0).cache_context_hash)
    store.register_candidate(JOB.run_id, "c3", make_genome(0), {"generation": 0})
    store.commit_evaluation(cache_hit_result(source, candidate_id="c3", attempt_id="c3.a1"))
    store.register_candidate(JOB.run_id, "c4", make_genome(4), {"generation": 0})
    store.begin_attempt(JOB.run_id, "c4", "c4.a1")
    store.record_gpu_segment(JOB.run_id, "c4.a1", 2.5)
    store.recover_interrupted_attempts(JOB.run_id)
    hashes = store.run_hashes(JOB.run_id)
    store.save_checkpoint(
        SearchCheckpoint(
            run_id=JOB.run_id,
            generation=0,
            population=("c0", "c1", "c2", "c3", "c4"),
            pareto_archive=("c0", "c1"),
            pending_candidates=("c4",),
            rng_states={"synthetic": [1, 2]},
            unique_evaluations=2,
            active_gpu_seconds=store.active_gpu_seconds(JOB.run_id),
            protocol_hash=hashes.protocol_hash,
            plan_hash=hashes.plan_hash,
            space_hash=hashes.space_hash,
        )
    )
    store.record_front(JOB.run_id, ("c0", "c1"), store.active_gpu_seconds(JOB.run_id), None)
    JobQueue(store).enqueue("job-backup", JOB.run_id)
    return store


def test_backup_of_live_wal_database_restores_identically(
    populated: SQLiteRunStore, db_path: Path, artifact_root: Path, tmp_path: Path
) -> None:
    wal = Path(str(db_path) + "-wal")
    assert wal.exists() and wal.stat().st_size > 0  # committed data still lives in the WAL
    before_db = _live_digests(db_path)
    before_files = _tree(artifact_root)

    backup = create_backup(db_path, artifact_root, tmp_path / "backups" / "b1", label="synthetic-test")
    assert backup.complete
    assert backup.manifest["database"]["tables"] == before_db  # snapshot == committed state
    assert backup.manifest["summary"] == {
        "verified": 2,
        "mismatched": 0,
        "orphan": 0,
        "temporary": 0,
        "missing": 0,
        "cataloged": 2,
    }
    assert backup.manifest["database"]["schema_versions"][-1][0] == LATEST_VERSION
    assert str(tmp_path) not in (backup.path / MANIFEST_FILENAME).read_text(encoding="utf-8")  # no absolute paths
    assert verify_backup(backup.path).ok

    report = rehearse_restore(backup.path, tmp_path / "rehearsal")
    assert report.ok and report.backup_complete, report
    assert report.orphans == () and report.temporaries == ()
    with SQLiteRunStore(report.db_path, report.artifact_root) as restored:
        assert restored.active_gpu_seconds(JOB.run_id) == pytest.approx(32.5)  # failed work kept
        assert restored.unique_valid_evaluations(JOB.run_id) == 2
        assert restored.get_attempt("c4.a1").status == "interrupted"
        assert restored.load_checkpoint(JOB.run_id) == populated.load_checkpoint(JOB.run_id)
        assert JobQueue(restored).get("job-backup").state == "queued"

    # The sources were only read.
    assert _live_digests(db_path) == before_db
    assert _tree(artifact_root) == before_files


def test_backup_is_consistent_while_the_writer_keeps_committing(
    populated: SQLiteRunStore, db_path: Path, artifact_root: Path, tmp_path: Path
) -> None:
    populated.close()
    writer = SQLiteRunStore(db_path, artifact_root)
    stop = threading.Event()
    committed: list[str] = []

    def keep_writing() -> None:
        evaluator = FakeEvaluator(gpu_seconds=1.0)
        i = 10
        while not stop.is_set() and i < 200:
            run_candidate(writer, JOB, f"c{i}", f"c{i}.a1", make_genome(i), evaluator)
            committed.append(f"c{i}")
            i += 1

    thread = threading.Thread(target=keep_writing)
    thread.start()
    try:
        reports = []
        for n in range(3):
            backup = create_backup(db_path, artifact_root, tmp_path / f"b{n}", label=f"concurrent-{n}")
            assert backup.manifest["database"]["integrity_check"] == ["ok"]
            assert verify_backup(backup.path).ok
            report = rehearse_restore(backup.path, tmp_path / f"r{n}")
            reports.append(report)
    finally:
        stop.set()
        thread.join()
        writer.close()
    assert len(committed) > 0
    for report in reports:
        # Every snapshot is a committed state: invariants and catalog hold; files published
        # after the snapshot are carried over as orphans, never as missing.
        assert report.ok, report
        assert report.missing == () and report.mismatched == ()


def test_existing_destinations_are_never_overwritten(
    populated: SQLiteRunStore, db_path: Path, artifact_root: Path, tmp_path: Path
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("previous backup", encoding="utf-8")
    with pytest.raises(BackupError, match="already exists"):
        create_backup(db_path, artifact_root, occupied, label="x")
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "previous backup"
    with pytest.raises(BackupError, match="outside the artifact root"):
        create_backup(db_path, artifact_root, artifact_root / "bk", label="x")

    backup = create_backup(db_path, artifact_root, tmp_path / "b", label="x")
    with pytest.raises(BackupError, match="refusing to overwrite"):
        restore_backup(backup.path, db_path, tmp_path / "fresh-artifacts")  # the live database
    with pytest.raises(BackupError, match="not empty"):
        restore_backup(backup.path, tmp_path / "new" / DB_FILENAME, artifact_root)
    assert not (tmp_path / "new").exists()
    assert list(tmp_path.glob(".*partial*")) == []


def test_evidence_problems_are_recorded_not_repaired(
    populated: SQLiteRunStore, db_path: Path, artifact_root: Path, tmp_path: Path
) -> None:
    engines = sorted(artifact_root.rglob("engine.plan"))
    tampered, lost = engines
    tampered.write_bytes(b"bytes changed after commit")  # simulated corruption
    lost_bytes = lost.read_bytes()
    lost.unlink()  # simulated disk loss
    publish_bytes(artifact_root, "runs/run-backup/attempts/c9.a1/engine.plan", b"published, never committed")
    (artifact_root / "runs" / ".half.plan.deadbeef.tmp").write_bytes(b"partial")
    before_files = _tree(artifact_root)

    backup = create_backup(db_path, artifact_root, tmp_path / "b", label="damaged-source")
    assert not backup.complete
    status = {e.rel_path: e.status for e in backup.artifacts}
    assert sorted(status.values()) == ["mismatched", "missing", "orphan", "temporary"]
    assert _tree(artifact_root) == before_files  # nothing fixed, deleted or rewritten
    assert verify_backup(backup.path).ok  # the backup faithfully holds what was there

    report = rehearse_restore(backup.path, tmp_path / "r")
    assert not report.ok and not report.backup_complete
    assert len(report.mismatched) == 1 and len(report.missing) == 1
    assert len(report.orphans) == 1
    assert report.temporaries == ()  # temporaries are listed in the manifest, not copied
    assert [e.size_bytes for e in backup.artifacts if e.status == "temporary"] == [len(b"partial")]
    assert report.table_mismatches == () and report.catalog_matches
    assert lost_bytes  # the original content exists only if some earlier backup kept it


def test_tampered_backup_is_detected_and_not_restored(
    populated: SQLiteRunStore, db_path: Path, artifact_root: Path, tmp_path: Path
) -> None:
    backup = create_backup(db_path, artifact_root, tmp_path / "b", label="x")
    copy = next((backup.path / "artifacts").rglob("engine.plan"))
    copy.write_bytes(b"tampered backup")
    problems = verify_backup(backup.path).problems
    assert any("artifact copy hash differs" in p for p in problems)
    with pytest.raises(BackupError, match="does not verify"):
        rehearse_restore(backup.path, tmp_path / "r")
    assert not (tmp_path / "r").exists()

    other = create_backup(db_path, artifact_root, tmp_path / "b2", label="x")
    manifest = other.path / MANIFEST_FILENAME
    manifest.write_bytes(manifest.read_bytes().replace(b'"label":"x"', b'"label":"y"'))
    with pytest.raises(BackupError, match="manifest hash"):
        load_manifest(other.path)
    assert not verify_backup(other.path).ok

    third = create_backup(db_path, artifact_root, tmp_path / "b3", label="x")
    with open(third.path / DB_FILENAME, "r+b") as fh:
        fh.seek(-10, 2)
        fh.write(b"\x00" * 10)
    assert "database copy hash/size differs from manifest" in verify_backup(third.path).problems
