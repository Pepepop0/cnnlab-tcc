"""Migration 2: durable queue, single-writer lease, heartbeat, expiry and cancel.

All data is synthetic (`storage.testing`); no GPU, no real run.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from cnnlab.tcc.operation import prepare_resume
from cnnlab.tcc.storage import (
    MIGRATIONS,
    ConsistencyError,
    IdempotencyConflictError,
    JobQueue,
    JobStateError,
    LeaseHeldError,
    LeaseLostError,
    SQLiteRunStore,
)
from cnnlab.tcc.storage.connection import open_writer
from cnnlab.tcc.storage.schema import migrate
from cnnlab.tcc.storage.testing import (
    FakeEvaluator,
    ManualClock,
    make_genome,
    make_job,
    make_request,
    run_candidate,
    seed_reference_rows,
)

JOB = make_job("run-jobs", method="joint", seed_block=0)
OTHER = make_job("run-jobs-2", method="mono", seed_block=0)
LEASE_S = 60.0

# SHA-256 of migration 1 as published on main (e648532). Must never change.
MIGRATION_1_SHA256 = "18032baa71f3bff23f31f67a635297daedf2b1deef7134026557e05f4d1d4bb0"
MIGRATION_2_SHA256 = "13811d51ee10c5a9de709d099cf4c7c7aad122744be1e95b8b660ed99a9edf6b"  # applied to real databases


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def worker_a(db_path: Path, artifact_root: Path, clock: ManualClock) -> Iterator[SQLiteRunStore]:
    store = SQLiteRunStore(db_path, artifact_root, clock=clock)
    seed_reference_rows(store)
    store.create_run(JOB)
    store.create_run(OTHER)
    yield store
    store.close()


@pytest.fixture
def worker_b(
    worker_a: SQLiteRunStore, db_path: Path, artifact_root: Path, clock: ManualClock
) -> Iterator[SQLiteRunStore]:
    """Second process-like writer: its own connection on the same file and clock."""
    store = SQLiteRunStore(db_path, artifact_root, clock=clock)
    yield store
    store.close()


def _count(store: SQLiteRunStore, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# -- migration -----------------------------------------------------------------


def test_migration_1_is_unchanged_and_2_is_added() -> None:
    assert MIGRATIONS[0].sha256 == MIGRATION_1_SHA256
    assert MIGRATIONS[1].sha256 == MIGRATION_2_SHA256  # applied to real databases: it must never change
    assert [m.version for m in MIGRATIONS] == [1, 2, 3]


def test_v1_database_upgrades_to_v2_without_touching_data(db_path: Path, artifact_root: Path) -> None:
    conn = open_writer(db_path)
    try:
        assert migrate(conn, MIGRATIONS[:1]) == 1
    finally:
        conn.close()
    with SQLiteRunStore(db_path, artifact_root) as store:  # applies migration 2
        seed_reference_rows(store)
        store.create_run(JOB)
        run_candidate(store, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator())
        before = store._conn.execute("SELECT result_sha256 FROM evaluation_attempts").fetchall()
        applied = store._conn.execute("SELECT version, sql_sha256 FROM schema_migrations ORDER BY version").fetchall()
        assert [tuple(r) for r in applied] == [
            (1, MIGRATION_1_SHA256),
            (2, MIGRATION_2_SHA256),
            (3, MIGRATIONS[2].sha256),
        ]
    with SQLiteRunStore(db_path, artifact_root) as store:
        after = store._conn.execute("SELECT result_sha256 FROM evaluation_attempts").fetchall()
        assert [tuple(r) for r in after] == [tuple(r) for r in before]
        assert store.check_integrity().ok


# -- enqueue -------------------------------------------------------------------


def test_enqueue_is_idempotent_and_one_active_job_per_run(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    job = queue.enqueue("job-1", JOB.run_id, options={"checkpoint_every": 2})
    assert job.state == "queued" and job.queue_seq == 1 and job.options["checkpoint_every"] == 2
    assert queue.enqueue("job-1", JOB.run_id, options={"checkpoint_every": 2}) == job
    with pytest.raises(IdempotencyConflictError):
        queue.enqueue("job-1", JOB.run_id, options={"checkpoint_every": 3})
    with pytest.raises(JobStateError):
        queue.enqueue("job-1b", JOB.run_id)
    assert [e.kind for e in queue.events("job-1")] == ["enqueued"]


def test_claim_order_is_priority_then_fifo_and_single_running_job(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-low", JOB.run_id, priority=0)
    queue.enqueue("job-high", OTHER.run_id, priority=5)
    lease = queue.claim_next("w1", lease_s=LEASE_S)
    assert lease is not None and lease.job_id == "job-high" and lease.claim_number == 1
    # One GPU job at a time: nothing else is claimable while a live lease exists.
    assert queue.claim_next("w2", lease_s=LEASE_S) is None
    with pytest.raises(sqlite3.IntegrityError):
        worker_a._conn.execute(
            "UPDATE jobs SET state = 'running', lease_owner = 'x', lease_token = 'y', "
            "lease_expires_at = '2026-09-28T13:00:00.000000Z', claim_count = 1 WHERE job_id = 'job-low'"
        )
    queue.finish(lease, "done", detail="synthetic")
    nxt = queue.claim_next("w2", lease_s=LEASE_S)
    assert nxt is not None and nxt.job_id == "job-low"


# -- two workers ---------------------------------------------------------------


def test_two_workers_racing_get_exactly_one_lease(worker_a: SQLiteRunStore, worker_b: SQLiteRunStore) -> None:
    JobQueue(worker_a).enqueue("job-1", JOB.run_id)
    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def claim(name: str, store: SQLiteRunStore) -> None:
        barrier.wait()
        results[name] = JobQueue(store).claim_next(name, lease_s=LEASE_S)

    threads = [threading.Thread(target=claim, args=(n, s)) for n, s in (("w-a", worker_a), ("w-b", worker_b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [lease for lease in results.values() if lease is not None]
    assert len(winners) == 1
    job = JobQueue(worker_a).get("job-1")
    assert job.state == "running" and job.claim_count == 1 and job.lease_owner == winners[0].worker_id
    assert [e.kind for e in JobQueue(worker_a).events("job-1")] == ["enqueued", "claimed"]


def test_second_worker_cannot_write_to_a_leased_run(worker_a: SQLiteRunStore, worker_b: SQLiteRunStore) -> None:
    queue_a = JobQueue(worker_a)
    queue_a.enqueue("job-1", JOB.run_id)
    lease = queue_a.claim_next("w-a", lease_s=LEASE_S)
    with worker_a.lease_fence(lease.fence):
        worker_a.register_candidate(JOB.run_id, "c1", make_genome(1), {"generation": 0})
        worker_a.begin_attempt(JOB.run_id, "c1", "c1.a1")
    before = _count(worker_a, "gpu_segments")
    with pytest.raises(LeaseHeldError):
        worker_b.register_candidate(JOB.run_id, "c2", make_genome(2), {"generation": 0})
    with pytest.raises(LeaseHeldError):
        worker_b.record_gpu_segment(JOB.run_id, "c1.a1", 1.0)
    with pytest.raises(LeaseHeldError):
        worker_b.recover_interrupted_attempts(JOB.run_id)  # a live worker is never declared dead
    with pytest.raises(LeaseHeldError):
        prepare_resume(worker_b, JOB)
    assert _count(worker_a, "gpu_segments") == before
    assert worker_a.get_attempt("c1.a1").status == "running"
    assert JobQueue(worker_a).job_attempt_ids("job-1") == ("c1.a1",)
    # Other runs are not affected by this lease.
    worker_b.register_candidate(OTHER.run_id, "o1", make_genome(1), {"generation": 0})


# -- heartbeat and expiry ------------------------------------------------------


def test_heartbeat_keeps_a_live_worker_alive(
    worker_a: SQLiteRunStore, worker_b: SQLiteRunStore, clock: ManualClock
) -> None:
    queue_a, queue_b = JobQueue(worker_a), JobQueue(worker_b)
    queue_a.enqueue("job-1", JOB.run_id)
    lease = queue_a.claim_next("w-a", lease_s=LEASE_S)
    for _ in range(5):  # 5 x 50 s = 250 s, far beyond one lease, but always renewed in time
        clock.advance(50)
        job = queue_a.heartbeat(lease)
        assert queue_b.claim_next("w-b", lease_s=LEASE_S) is None
        assert queue_b.reap_expired() == ()
    assert job.heartbeat_count == 5 and job.state == "running" and job.lease_owner == "w-a"


def test_viewer_closing_does_not_touch_the_lease(
    worker_a: SQLiteRunStore, db_path: Path, artifact_root: Path, clock: ManualClock
) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    for _ in range(3):  # a TUI opens the database, looks at the queue and closes
        with SQLiteRunStore(db_path, artifact_root, clock=clock) as viewer:
            assert JobQueue(viewer).get("job-1").state == "running"
        clock.advance(10)
    assert JobQueue(worker_a).heartbeat(lease).state == "running"


def test_expired_lease_is_interrupted_only_by_a_claimer_and_keeps_time(
    worker_a: SQLiteRunStore, worker_b: SQLiteRunStore, clock: ManualClock
) -> None:
    queue_a, queue_b = JobQueue(worker_a), JobQueue(worker_b)
    queue_a.enqueue("job-1", JOB.run_id)
    lease_a = queue_a.claim_next("w-a", lease_s=LEASE_S)
    with worker_a.lease_fence(lease_a.fence):
        run_candidate(worker_a, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator(gpu_seconds=10.0))
        worker_a.register_candidate(JOB.run_id, "c2", make_genome(2), {"generation": 0})
        worker_a.begin_attempt(JOB.run_id, "c2", "c2.a1")
        assert worker_a.record_job_gpu_segment(JOB.run_id, "c2.a1", 100.0, 103.5) == pytest.approx(3.5)
    confirmed = worker_a._conn.execute(
        "SELECT result_sha256 FROM evaluation_attempts WHERE attempt_id = 'c1.a1'"
    ).fetchone()[0]

    clock.advance(LEASE_S - 1)
    assert queue_b.claim_next("w-b", lease_s=LEASE_S) is None  # not yet expired
    clock.advance(2)  # worker A went silent (crash, freeze...)
    lease_b = queue_b.claim_next("w-b", lease_s=LEASE_S)
    assert lease_b is not None and lease_b.job_id == "job-1" and lease_b.claim_number == 2

    interrupted = worker_b.get_attempt("c2.a1")
    assert interrupted.status == "interrupted" and interrupted.recovered
    assert interrupted.result.failure_code == "lease_expired"
    assert interrupted.result.gpu_seconds == pytest.approx(3.5)  # failed work still counts
    assert worker_b.active_gpu_seconds(JOB.run_id) == pytest.approx(13.5)
    assert (
        worker_b._conn.execute("SELECT result_sha256 FROM evaluation_attempts WHERE attempt_id = 'c1.a1'").fetchone()[0]
        == confirmed
    )  # the confirmed individual is untouched
    kinds = [e.kind for e in queue_b.events("job-1")]
    assert kinds == ["enqueued", "claimed", "lease_expired", "claimed"]
    expired = queue_b.events("job-1")[2]
    assert expired.worker_id == "w-a" and expired.detail["interrupted_attempt_ids"] == ["c2.a1"]


def test_worker_whose_lease_was_taken_over_is_fenced(
    worker_a: SQLiteRunStore, worker_b: SQLiteRunStore, clock: ManualClock
) -> None:
    queue_a, queue_b = JobQueue(worker_a), JobQueue(worker_b)
    queue_a.enqueue("job-1", JOB.run_id)
    lease_a = queue_a.claim_next("w-a", lease_s=LEASE_S)
    with worker_a.lease_fence(lease_a.fence):
        worker_a.register_candidate(JOB.run_id, "c1", make_genome(1), {"generation": 0})
        worker_a.begin_attempt(JOB.run_id, "c1", "c1.a1")
    clock.advance(LEASE_S + 1)
    queue_b.claim_next("w-b", lease_s=LEASE_S)

    # Worker A wakes up with a finished (synthetic) evaluation in hand.
    late = FakeEvaluator().evaluate(make_request(JOB, "c1", "c1.a1", make_genome(1), worker_a.artifact_root))
    with pytest.raises(LeaseLostError):
        queue_a.heartbeat(lease_a)
    with worker_a.lease_fence(lease_a.fence):
        with pytest.raises(LeaseLostError):
            worker_a.commit_evaluation(late)
        with pytest.raises(LeaseLostError):
            worker_a.record_job_gpu_segment(JOB.run_id, "c1.a1", 0.0, 1.0)
        with pytest.raises(LeaseLostError):
            worker_a.register_candidate(JOB.run_id, "c9", make_genome(9), {})
    with pytest.raises(LeaseLostError):
        queue_a.finish(lease_a, "done", detail="too late")
    assert worker_b.get_attempt("c1.a1").status == "interrupted"
    assert _count(worker_b, "measurements") == 0


def test_expired_lease_nobody_took_over_can_still_be_renewed(worker_a: SQLiteRunStore, clock: ManualClock) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    clock.advance(LEASE_S * 3)  # e.g. the machine slept; no other claimer observed it
    assert queue.heartbeat(lease).state == "running"
    assert queue.reap_expired() == ()


# -- cancel, release, finish ---------------------------------------------------


def test_cancel_queued_job_is_immediate(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    job = queue.request_cancel("job-1", reason="operator")
    assert job.state == "cancelled" and job.finished_at is not None
    assert queue.claim_next("w-a", lease_s=LEASE_S) is None
    assert queue.request_cancel("job-1", reason="again") == job  # idempotent
    queue.enqueue("job-2", JOB.run_id)  # a new job may continue the run later


def test_cancel_running_job_is_cooperative(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    with pytest.raises(JobStateError):
        queue.finish(lease, "cancelled", detail="no request yet")
    job = queue.request_cancel("job-1", reason="operator")
    assert job.state == "running" and job.cancel_requested  # the worker decides when to stop
    assert queue.is_cancel_requested(lease)
    assert queue.finish(lease, "cancelled", detail="stopped between attempts").state == "cancelled"
    assert queue.request_cancel("job-1", reason="again").state == "cancelled"  # idempotent
    with pytest.raises(LeaseLostError):
        queue.finish(lease, "done", detail="terminal jobs never change")


def test_finish_refuses_open_attempts_and_release_requeues(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    with worker_a.lease_fence(lease.fence):
        worker_a.register_candidate(JOB.run_id, "c1", make_genome(1), {})
        worker_a.begin_attempt(JOB.run_id, "c1", "c1.a1")
    with pytest.raises(JobStateError):
        queue.finish(lease, "done", detail="attempt still open")
    with pytest.raises(JobStateError):
        queue.release(lease, reason="attempt still open")
    with worker_a.lease_fence(lease.fence):
        worker_a.commit_evaluation(
            FakeEvaluator().evaluate(make_request(JOB, "c1", "c1.a1", make_genome(1), worker_a.artifact_root))
        )
    assert queue.release(lease, reason="planned stop").state == "queued"
    again = queue.claim_next("w-b", lease_s=LEASE_S)
    assert again.claim_number == 2 and again.lease_token != lease.lease_token


# -- raw SQL backstops ---------------------------------------------------------


def test_triggers_guard_job_rows(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    conn = worker_a._conn
    with pytest.raises(sqlite3.IntegrityError, match="lease token"):
        conn.execute("UPDATE jobs SET lease_token = 'forged' WHERE job_id = 'job-1'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM jobs")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM job_events")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE jobs SET priority = 9 WHERE job_id = 'job-1'")
    queue.request_cancel("job-1", reason="x")
    queue.finish(lease, "cancelled", detail="x")
    with pytest.raises(sqlite3.IntegrityError, match="terminal job|invalid job state"):
        conn.execute("UPDATE jobs SET state = 'queued', finished_at = NULL WHERE job_id = 'job-1'")
    with pytest.raises(sqlite3.IntegrityError, match="created queued"):
        conn.execute(
            "INSERT INTO jobs (job_id, run_id, queue_seq, priority, options_json, options_sha256, state, "
            "lease_owner, lease_token, lease_expires_at, created_at, updated_at) VALUES "
            "('j', 'run-jobs-2', 99, 0, '{}', ?, 'running', 'x', 'y', "
            "'2026-09-28T13:00:00.000000Z', '2026-09-28T12:00:00.000000Z', '2026-09-28T12:00:00.000000Z')",
            ("0" * 64,),
        )


def test_job_segment_must_match_its_ledger_segment(worker_a: SQLiteRunStore) -> None:
    queue = JobQueue(worker_a)
    queue.enqueue("job-1", JOB.run_id)
    lease = queue.claim_next("w-a", lease_s=LEASE_S)
    with pytest.raises(LeaseLostError):
        worker_a.record_job_gpu_segment(JOB.run_id, "c1.a1", 0.0, 1.0)  # no fence
    with worker_a.lease_fence(lease.fence):
        worker_a.register_candidate(JOB.run_id, "c1", make_genome(1), {})
        worker_a.begin_attempt(JOB.run_id, "c1", "c1.a1")
        with pytest.raises(ValueError):
            worker_a.record_job_gpu_segment(JOB.run_id, "c1.a1", 5.0, 4.0)
        worker_a.record_job_gpu_segment(JOB.run_id, "c1.a1", 10.0, 12.25)
    seg = worker_a._conn.execute("SELECT MAX(segment_id) FROM gpu_segments").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        worker_a._conn.execute("UPDATE job_gpu_segments SET mono_end = 99 WHERE segment_id = ?", (seg,))
    assert queue.job_gpu_seconds("job-1") == pytest.approx(2.25)
    with pytest.raises(ConsistencyError):
        with worker_a.lease_fence(lease.fence):
            worker_a.register_candidate(OTHER.run_id, "o1", make_genome(1), {})  # fenced to another run
