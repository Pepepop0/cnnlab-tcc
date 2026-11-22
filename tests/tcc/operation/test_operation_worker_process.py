"""Real worker processes with the heartbeat thread and real clocks (CPU only).

`os._exit` kills the worker without `finally`, `atexit` or connection close:
it simulates a killed process, **not** a power cut (the OS still flushes
what SQLite handed to it). Durability against power loss is not claimed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from datetime import timedelta
from pathlib import Path

import pytest

import cnnlab
from cnnlab.tcc.operation import OperationService
from cnnlab.tcc.operation.testing import ClockedFakeEvaluator, FakeMonotonic, SyntheticListRunner
from cnnlab.tcc.storage import JobQueue, LeaseHeldError, SQLiteRunStore
from cnnlab.tcc.storage.canonical import utc_now
from cnnlab.tcc.storage.testing import make_job, seed_reference_rows

JOB = make_job("run-proc", method="random", seed_block=3)

_WORKER = """
import sys
from pathlib import Path
from cnnlab.tcc.operation import OperationService
from cnnlab.tcc.operation.testing import SleepingFakeEvaluator, SyntheticListRunner
from cnnlab.tcc.storage import SQLiteRunStore
db, art, sleep_s, crash_on, crash_after, n, lease_s, beat_s = sys.argv[1:9]
store = SQLiteRunStore(Path(db), Path(art))
evaluator = SleepingFakeEvaluator(
    gpu_seconds=10.0, sleep_s=float(sleep_s), crash_on_attempt=crash_on or None, crash_after_s=float(crash_after)
)
service = OperationService(
    store, SyntheticListRunner(n_candidates=int(n)), evaluator,
    worker_id="worker-proc", lease_s=float(lease_s), heartbeat_s=float(beat_s),
)
outcome = service.run_next()
print(outcome.end_state, len(service.heartbeat_errors))
store.close()
"""


def _spawn(db_path: Path, artifact_root: Path, *args: object) -> subprocess.Popen[str]:
    src = str(Path(cnnlab.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_WORKER), str(db_path), str(artifact_root), *map(str, args)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def queued(db_path: Path, artifact_root: Path) -> None:
    with SQLiteRunStore(db_path, artifact_root) as store:
        seed_reference_rows(store)
        store.create_run(JOB)
        JobQueue(store).enqueue("job-proc", JOB.run_id)


def test_live_worker_is_never_declared_dead_by_viewers_or_rivals(
    queued: None, db_path: Path, artifact_root: Path
) -> None:
    # 8 candidates x 0.3 s ~ 2.4 s of work under a 0.8 s lease: only heartbeats keep it alive.
    proc = _spawn(db_path, artifact_root, 0.3, "", 0, 8, 0.8, 0.1)
    checks = 0
    try:
        deadline = time.monotonic() + 60
        while proc.poll() is None and time.monotonic() < deadline:
            with SQLiteRunStore(db_path, artifact_root) as viewer:  # a TUI opens, looks, closes
                queue = JobQueue(viewer)
                job = queue.get("job-proc")
                if job.state == "running":
                    assert queue.claim_next("rival", lease_s=0.8) is None
                    assert queue.reap_expired() == ()
                    with pytest.raises(LeaseHeldError):
                        viewer.recover_interrupted_attempts(JOB.run_id)
                    checks += 1
            time.sleep(0.15)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert out.split() == ["done", "0"]
    assert checks >= 3
    with SQLiteRunStore(db_path, artifact_root) as store:
        queue = JobQueue(store)
        assert [e.kind for e in queue.events("job-proc")] == ["enqueued", "claimed", "done"]
        assert queue.get("job-proc").heartbeat_count >= 10
        assert all(a.status == "valid" for a in store.list_attempts(JOB.run_id))
        assert store.check_integrity().ok


def test_crash_mid_attempt_keeps_confirmed_work_and_time_then_resumes(
    queued: None, db_path: Path, artifact_root: Path
) -> None:
    n = 5
    proc = _spawn(db_path, artifact_root, 0.05, "c2.a1", 0.6, n, 30.0, 0.05)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 9, (out, err)  # killed inside c2.a1

    with SQLiteRunStore(db_path, artifact_root) as store:
        confirmed = {
            r[0]: r[1]
            for r in store._conn.execute(
                "SELECT attempt_id, result_sha256 FROM evaluation_attempts WHERE status = 'valid'"
            )
        }
        assert set(confirmed) == {"c0.a1", "c1.a1"}
        crashed = store.get_attempt("c2.a1")
        assert crashed.status == "running"  # nobody may close it while the lease could be alive
        flushed = crashed.ledger_gpu_seconds
        assert 0.2 <= flushed <= 0.6  # heartbeats flushed most of the 0.6 s before the kill

    # A restarted worker observes the lease expiry (clock 31 s later) and resumes.
    later = lambda: utc_now() + timedelta(seconds=31)  # noqa: E731
    mono = FakeMonotonic()
    with SQLiteRunStore(db_path, artifact_root, clock=later) as store:
        service = OperationService(
            store,
            SyntheticListRunner(n_candidates=n),
            ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono),
            worker_id="worker-restarted",
            lease_s=30.0,
            heartbeat_thread=False,
            monotonic=mono,
        )
        outcome = service.run_next()
        assert outcome.end_state == "done"
        assert outcome.attempt_ids == ("c2.a2", "c3.a1", "c4.a1")

        interrupted = store.get_attempt("c2.a1")
        assert interrupted.status == "interrupted" and interrupted.result.failure_code == "lease_expired"
        assert interrupted.result.gpu_seconds == pytest.approx(flushed)  # failed work counts
        for attempt_id, digest in confirmed.items():  # confirmed individuals untouched
            assert (
                store._conn.execute(
                    "SELECT result_sha256 FROM evaluation_attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()[0]
                == digest
            )
        valid = [a.candidate_id for a in store.list_attempts(JOB.run_id) if a.status == "valid"]
        assert sorted(valid) == [f"c{i}" for i in range(n)]  # no duplicated individual
        assert store._conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == n
        assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0 * n + flushed)
        kinds = [e.kind for e in JobQueue(store).events("job-proc")]
        assert kinds == ["enqueued", "claimed", "lease_expired", "claimed", "done"]
        assert store.check_integrity().ok
