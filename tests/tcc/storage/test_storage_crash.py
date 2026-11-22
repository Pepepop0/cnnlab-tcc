"""Crash simulation: a child process dies with `os._exit` at chosen points.

`os._exit` skips `finally`, `atexit` and connection close, which is the
closest CPU-only stand-in for a killed worker or a power cut of the process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import cnnlab
from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, run_candidate

JOB = make_job("run-crash")

_PRELUDE = """
import os, sys
from pathlib import Path
from cnnlab.tcc.storage import SQLiteRunStore, publish_bytes
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, make_request
store = SQLiteRunStore(Path(sys.argv[1]), Path(sys.argv[2]))
job = make_job("run-crash")
"""


def _run_child(body: str, db_path: Path, artifact_root: Path) -> int:
    src = str(Path(cnnlab.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = _PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(db_path), str(artifact_root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 1, proc.stderr  # 1 would be an unexpected Python error
    return proc.returncode


@pytest.fixture
def prepared(seeded: SQLiteRunStore) -> SQLiteRunStore:
    seeded.create_run(JOB)
    run_candidate(seeded, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator(gpu_seconds=10.0))
    seeded.register_candidate(JOB.run_id, "c2", make_genome(2), {"generation": 0})
    seeded.close()  # the child becomes the only writer
    return seeded


def test_crash_inside_commit_transaction_rolls_back(
    prepared: SQLiteRunStore, db_path: Path, artifact_root: Path
) -> None:
    code = _run_child(
        """
        store.begin_attempt(job.run_id, "c2", "c2.a1")
        store.record_gpu_segment(job.run_id, "c2.a1", 4.0)
        result = FakeEvaluator(gpu_seconds=10.0).evaluate(
            make_request(job, "c2", "c2.a1", make_genome(2), store.artifact_root))
        def die(*args, **kwargs):
            os._exit(3)  # attempt row already updated to 'valid' inside the open transaction
        store._write_measurement = die
        store.commit_evaluation(result)
        """,
        db_path,
        artifact_root,
    )
    assert code == 3
    with SQLiteRunStore(db_path, artifact_root) as store:
        confirmed = store.get_attempt("c1.a1")
        assert confirmed is not None and confirmed.status == "valid"  # earlier individual survives
        crashed = store.get_attempt("c2.a1")
        assert crashed is not None and crashed.status == "running"  # the terminal update was rolled back
        assert crashed.ledger_gpu_seconds == pytest.approx(4.0)  # time recorded before the crash survives
        assert store._conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1
        report = store.check_integrity()
        assert report.running_attempts == ("c2.a1",)
        assert len(report.artifacts.orphans) == 1  # engine published, never cataloged

        assert store.recover_interrupted_attempts(JOB.run_id) == ("c2.a1",)
        interrupted = store.get_attempt("c2.a1")
        assert interrupted.status == "interrupted" and interrupted.recovered
        assert interrupted.result.gpu_seconds == pytest.approx(4.0)
        retry = run_candidate(store, JOB, "c2", "c2.a2", make_genome(2), FakeEvaluator(gpu_seconds=10.0))
        assert retry.status == "valid"
        assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0 + 4.0 + 10.0)
        assert store.unique_valid_evaluations(JOB.run_id) == 2


def test_crash_right_after_commit_keeps_the_individual(
    prepared: SQLiteRunStore, db_path: Path, artifact_root: Path
) -> None:
    code = _run_child(
        """
        store.begin_attempt(job.run_id, "c2", "c2.a1")
        result = FakeEvaluator(gpu_seconds=7.0).evaluate(
            make_request(job, "c2", "c2.a1", make_genome(2), store.artifact_root))
        store.commit_evaluation(result)
        os._exit(4)  # no close(), no WAL checkpoint
        """,
        db_path,
        artifact_root,
    )
    assert code == 4
    with SQLiteRunStore(db_path, artifact_root) as store:
        attempt = store.get_attempt("c2.a1")
        assert attempt is not None and attempt.status == "valid"
        assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(17.0)
        assert store.check_integrity().ok
        # A replay of the same commit by a restarted runner is a no-op.
        store.commit_evaluation(attempt.result)
        assert store.recover_interrupted_attempts(JOB.run_id) == ()


def test_crash_during_artifact_publication_leaves_detectable_temp(
    prepared: SQLiteRunStore, db_path: Path, artifact_root: Path
) -> None:
    code = _run_child(
        """
        import cnnlab.tcc.storage.artifacts as artifacts
        def die(src, dst):
            os._exit(5)  # temp written and fsynced, never renamed
        artifacts.os.replace = die
        publish_bytes(store.artifact_root, "runs/run-crash/attempts/c2.a1/engine.plan", b"partial")
        """,
        db_path,
        artifact_root,
    )
    assert code == 5
    with SQLiteRunStore(db_path, artifact_root) as store:
        scan = store.check_integrity().artifacts
        assert len(scan.temporaries) == 1 and scan.orphans == () and scan.missing == ()
        assert not (artifact_root / "runs/run-crash/attempts/c2.a1/engine.plan").exists()
