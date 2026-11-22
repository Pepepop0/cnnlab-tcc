"""Headless operation service with synthetic runner/evaluator (CPU only, no GPU).

Deterministic tests drive a `ManualClock` (wall clock, lease expiry) and a
`FakeMonotonic` (GPU meter) and call `heartbeat_now()` explicitly; the
heartbeat thread is exercised in `test_operation_worker_process.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import TCCEvent
from cnnlab.tcc.operation import EVALUATOR_EXCEPTION_CODE, OperationService
from cnnlab.tcc.operation.testing import ClockedFakeEvaluator, FakeMonotonic, SyntheticListRunner
from cnnlab.tcc.storage import JobQueue, SQLiteRunStore
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.testing import ManualClock, make_job, seed_reference_rows

JOB = make_job("run-svc", method="joint", seed_block=1)
LEASE_S = 60.0
N = 5


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def mono() -> FakeMonotonic:
    return FakeMonotonic()


@pytest.fixture
def store(db_path: Path, artifact_root: Path, clock: ManualClock) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(db_path, artifact_root, clock=clock)
    seed_reference_rows(s)
    s.create_run(JOB)
    JobQueue(s).enqueue("job-1", JOB.run_id, options={"synthetic": True})
    yield s
    s.close()


@pytest.fixture
def viewer(store: SQLiteRunStore, db_path: Path, artifact_root: Path, clock: ManualClock) -> Iterator[SQLiteRunStore]:
    """Another connection, as a TUI/CLI or a second worker process would have."""
    s = SQLiteRunStore(db_path, artifact_root, clock=clock)
    yield s
    s.close()


def _service(
    store: SQLiteRunStore, evaluator: ClockedFakeEvaluator, mono: FakeMonotonic, **kw: object
) -> OperationService:
    events: list[TCCEvent] = kw.pop("events", [])  # type: ignore[assignment]
    runner = kw.pop("runner", None) or SyntheticListRunner(n_candidates=N)
    return OperationService(
        store,
        runner,  # type: ignore[arg-type]
        evaluator,
        worker_id=str(kw.pop("worker_id", "worker-1")),
        lease_s=LEASE_S,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=events.append,
    )


def _valid_attempts_per_candidate(store: SQLiteRunStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in store.list_attempts(JOB.run_id):
        if a.status == "valid":
            counts[a.candidate_id] = counts.get(a.candidate_id, 0) + 1
    return counts


def test_job_runs_to_done_with_metered_gpu_time(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    bad = sha256_hex("ctx-2")
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono, invalid_contexts=frozenset({bad}))
    events: list[TCCEvent] = []
    runner = SyntheticListRunner(n_candidates=N, contexts=(None, "shared", None, "shared", None))
    service = _service(store, evaluator, mono, events=events, runner=runner)

    outcome = service.run_next()
    assert outcome is not None and outcome.end_state == "done"
    assert runner.log == ["valid:c0.a1", "valid:c1.a1", "invalid:c2.a1", "hit:c3.a1", "valid:c4.a1"]
    queue = service.queue
    job = queue.get("job-1")
    assert job.state == "done" and job.finished_at is not None and job.lease_token is None
    # 4 evaluated attempts x 10 synthetic GPU seconds, cache hit free, invalid attempt counted.
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(40.0)
    assert queue.job_gpu_seconds("job-1") == pytest.approx(40.0)
    assert outcome.session_gpu_seconds == pytest.approx(40.0) and outcome.unrecorded_gpu_seconds == 0
    assert store._conn.execute("SELECT COUNT(*) FROM gpu_segments WHERE source = 'commit_reconcile'").fetchone()[0] == 0
    assert store.unique_valid_evaluations(JOB.run_id) == 3
    assert queue.job_attempt_ids("job-1") == ("c0.a1", "c1.a1", "c2.a1", "c3.a1", "c4.a1")  # hit included
    for attempt in store.list_attempts(JOB.run_id):
        assert attempt.result.quality["synthetic"] is True  # every fake result, cache hit included
        if attempt.result.source_attempt_id is None:
            assert attempt.result.quality["operation"]["metered_gpu_s"] == pytest.approx(10.0)
    assert store.load_checkpoint(JOB.run_id).population == tuple(f"c{i}" for i in range(N))
    assert [e.kind for e in events] == ["job_started", "job_ended"]
    assert service.run_next() is None  # queue drained
    assert store.check_integrity().ok


def test_evaluator_report_above_meter_is_reconciled(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=None)  # meter sees no time
    outcome = _service(store, evaluator, mono).run_next()
    assert outcome.end_state == "done"
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0 * N)
    first = store.get_attempt("c0.a1").result
    assert first.gpu_seconds == 10.0 and first.quality["operation"] == {
        "metered_gpu_s": 0.0,
        "evaluator_reported_gpu_s": 10.0,
    }


def test_heartbeat_flushes_gpu_time_during_an_attempt(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono)
    service = _service(store, evaluator, mono)
    seen: list[float] = []

    def beat(_request: object) -> None:
        service.heartbeat_now()
        seen.append(store.get_attempt("c1.a1").ledger_gpu_seconds)

    evaluator.hooks["c1.a1"] = beat
    assert service.run_next().end_state == "done"
    assert seen == [pytest.approx(5.0)]  # half of the attempt was already in the ledger
    rows = store._conn.execute(
        "SELECT j.mono_start, j.mono_end FROM job_gpu_segments j JOIN gpu_segments s USING (segment_id) "
        "WHERE s.attempt_id = 'c1.a1' ORDER BY j.segment_id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [(1010.0, 1015.0), (1015.0, 1020.0)]
    assert service.queue.get("job-1").heartbeat_count == 1


def test_cooperative_cancel_finishes_current_attempt(
    store: SQLiteRunStore, viewer: SQLiteRunStore, mono: FakeMonotonic
) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono)
    evaluator.hooks["c1.a1"] = lambda _r: JobQueue(viewer).request_cancel("job-1", reason="operator via TUI")
    runner = SyntheticListRunner(n_candidates=N)
    outcome = _service(store, evaluator, mono, runner=runner).run_next()

    assert outcome.end_state == "cancelled"
    assert runner.log == ["valid:c0.a1", "valid:c1.a1", "stop@c2"]  # c1 was completed, not abandoned
    assert store.get_attempt("c1.a1").status == "valid"
    assert [c.candidate_id for c in store.list_candidates(JOB.run_id)] == ["c0", "c1"]
    job = JobQueue(viewer).get("job-1")
    assert job.state == "cancelled" and job.cancel_reason == "operator via TUI"
    assert store.load_checkpoint(JOB.run_id).rng_states == {"synthetic_list_runner": {"next_index": 2}}

    # The run continues later under a new job, without redoing confirmed work.
    JobQueue(store).enqueue("job-2", JOB.run_id)
    assert _service(store, ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono), mono).run_next().end_state == "done"
    assert _valid_attempts_per_candidate(store) == {f"c{i}": 1 for i in range(N)}
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0 * N)


def test_planned_stop_requeues_and_resumes(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono)
    service = _service(store, evaluator, mono)
    evaluator.hooks["c2.a1"] = lambda _r: service.request_stop()
    outcome = service.run_next()
    assert outcome.end_state == "queued"
    assert service.queue.get("job-1").state == "queued"
    assert service.run_until_idle() == ()  # a stopped service claims nothing more

    resumed = _service(store, ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono), mono, worker_id="worker-2")
    again = resumed.run_next()
    assert again.end_state == "done" and again.attempt_ids == ("c3.a1", "c4.a1")
    assert resumed.queue.get("job-1").claim_count == 2
    assert _valid_attempts_per_candidate(store) == {f"c{i}": 1 for i in range(N)}


def test_evaluator_exception_fails_job_and_keeps_the_time(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono, raise_on_contexts=frozenset({sha256_hex("ctx-2")}))
    outcome = _service(store, evaluator, mono).run_next()
    assert outcome.end_state == "failed" and "synthetic evaluator crash" in outcome.detail
    crashed = store.get_attempt("c2.a1")
    assert crashed.status == "interrupted"
    assert crashed.result.failure_code == EVALUATOR_EXCEPTION_CODE
    assert crashed.result.gpu_seconds == pytest.approx(5.0)  # half done before the crash, still counted
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(25.0)
    assert JobQueue(store).get("job-1").state == "failed"

    JobQueue(store).enqueue("job-2", JOB.run_id)
    fixed = _service(store, ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono), mono)
    assert fixed.run_next().attempt_ids == ("c2.a2", "c3.a1", "c4.a1")
    assert _valid_attempts_per_candidate(store) == {f"c{i}": 1 for i in range(N)}
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(55.0)
    assert store.check_integrity().ok


def test_lease_taken_over_mid_attempt_fences_the_old_worker(
    store: SQLiteRunStore, viewer: SQLiteRunStore, clock: ManualClock, mono: FakeMonotonic
) -> None:
    evaluator = ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono)
    rival: dict[str, object] = {}

    def freeze_then_takeover(_request: object) -> None:
        clock.advance(LEASE_S + 1)  # worker 1 stalls past its lease without heartbeat
        rival["lease"] = JobQueue(viewer).claim_next("worker-2", lease_s=LEASE_S)

    evaluator.hooks["c1.a1"] = freeze_then_takeover
    outcome = _service(store, evaluator, mono).run_next()

    assert outcome.end_state == "lease_lost"
    assert outcome.unrecorded_gpu_seconds == pytest.approx(10.0)  # reported, cannot enter the ledger
    lost = viewer.get_attempt("c1.a1")
    assert lost.status == "interrupted" and lost.result.failure_code == "lease_expired"
    assert viewer._conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1  # only c0
    job = JobQueue(viewer).get("job-1")
    assert job.state == "running" and job.lease_owner == "worker-2"  # worker 1 did not end it

    JobQueue(viewer).release(rival["lease"], reason="hand over to a service")  # type: ignore[arg-type]
    second = _service(viewer, ClockedFakeEvaluator(gpu_seconds=10.0, clock=mono), mono, worker_id="worker-2")
    assert second.run_next().attempt_ids == ("c1.a2", "c2.a1", "c3.a1", "c4.a1")
    assert _valid_attempts_per_candidate(viewer) == {f"c{i}": 1 for i in range(N)}
    kinds = [e.kind for e in JobQueue(viewer).events("job-1")]
    assert kinds == ["enqueued", "claimed", "lease_expired", "claimed", "released", "claimed", "done"]


def test_budget_cut_between_attempts_counts_the_crossing_attempt(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    # 19,800 s budget; 5,000 synthetic s per attempt: the 4th attempt crosses it, finishes and counts.
    evaluator = ClockedFakeEvaluator(gpu_seconds=5000.0, clock=mono)
    runner = SyntheticListRunner(n_candidates=N)
    outcome = _service(store, evaluator, mono, runner=runner).run_next()
    assert outcome.end_state == "done" and "budget reached" in outcome.detail
    assert runner.log == ["valid:c0.a1", "valid:c1.a1", "valid:c2.a1", "valid:c3.a1", "stop@c4"]
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(20000.0)  # excess is auditable, not hidden
    assert JobQueue(store).get("job-1").state == "done"


def test_a_search_that_ran_out_of_space_before_the_budget_says_so_in_the_job_detail(
    store: SQLiteRunStore, mono: FakeMonotonic
) -> None:
    class ExhaustedRunner:
        """Evaluates nothing new and reports, as the search runner does, that the space ran out."""

        def __call__(self, ctx: object) -> str:
            ctx.emit(  # type: ignore[attr-defined]
                TCCEvent(
                    event_id="e1",
                    ts_utc="2026-10-05T00:00:00Z",
                    run_id=ctx.job.run_id,  # type: ignore[attr-defined]
                    kind="search_exhausted",
                    payload={"generation": 3, "stalled_generations": 3},
                )
            )
            return "completed"

    events: list[TCCEvent] = []
    outcome = _service(
        store, ClockedFakeEvaluator(gpu_seconds=1.0, clock=mono), mono, runner=ExhaustedRunner(), events=events
    ).run_next()
    assert outcome.end_state == "done"
    assert outcome.detail.startswith(
        "espaço esgotado antes do orçamento: 3 gerações seguidas sem nenhuma avaliação nova"
    )
    assert "de 19800.0s de GPU ativa" in outcome.detail  # the budget it did not reach
    assert [e.kind for e in events if e.kind == "search_exhausted"] == ["search_exhausted"]  # still forwarded
    [done] = [e for e in JobQueue(store).events("job-1") if e.kind == "done"]
    assert done.detail["detail"].startswith("espaço esgotado antes do orçamento")  # what the TUI shows


def test_run_next_can_target_one_job_and_records_session_detail(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    service = OperationService(
        store,
        SyntheticListRunner(n_candidates=1),
        ClockedFakeEvaluator(gpu_seconds=1.0, clock=mono),
        worker_id="w",
        heartbeat_thread=False,
        monotonic=mono,
        session_detail={"code_version": {"label": "t", "git_sha": "abc"}},
    )
    assert service.run_next(job_id="no-such-job") is None
    assert service.run_next(job_id="job-1").end_state == "done"
    claimed = [e for e in JobQueue(store).events("job-1") if e.kind == "claimed"][0]
    assert claimed.detail["code_version"] == {"label": "t", "git_sha": "abc"} and "pid" in claimed.detail
