"""Headless operation service: consume the durable queue with one GPU job at a time.

One `OperationService` is one worker process. For each claimed job it:

1. takes the lease (`JobQueue.claim_next`) and starts a heartbeat thread in
   *this* process, so liveness never depends on a TUI or CLI being open;
2. runs `prepare_resume` under the lease: attempts left `running` by a dead
   predecessor become `interrupted` (their GPU time stays), the latest
   checkpoint is verified, and the unresolved candidates are listed;
3. hands the search runner a `RunStore` that writes only under the lease and
   an evaluator wrapper that opens the attempt before any GPU work, meters
   active GPU time with a monotonic clock in ledger segments (flushed at
   every heartbeat, so a crash loses at most one heartbeat period), and
   closes the attempt as `interrupted` if the evaluator raises;
4. ends the job: `done` when the runner completes, `cancelled` after a
   cooperative cancel, back to `queued` after a planned stop, `failed` on an
   unexpected exception. If the lease was taken over, it writes nothing more.

The service never decides genetics nor alters measurements: the runner (front
A) and the evaluator are injected. Tests use synthetic fakes only.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from cnnlab.tcc.contracts import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    EventSink,
    GenomeRecord,
    RunStore,
    SearchCheckpoint,
    TCCEvent,
    TCCJobSpec,
)
from cnnlab.tcc.operation import sentinel_control
from cnnlab.tcc.operation.recovery import ResumePlan, prepare_resume
from cnnlab.tcc.operation.sentinel_control import SentinelHooks, SentinelState
from cnnlab.tcc.storage.canonical import format_utc
from cnnlab.tcc.storage.errors import ConsistencyError, LeaseLostError
from cnnlab.tcc.storage.jobs import JobQueue, Lease
from cnnlab.tcc.storage.sqlite_store import LeaseFence, RunHashes, SQLiteRunStore

EVALUATOR_EXCEPTION_CODE = "evaluator_exception"
RunnerStatus = Literal["completed", "stopped"]
JobEndState = Literal["done", "failed", "cancelled", "queued", "lease_lost"]
Monotonic = Callable[[], float]


@dataclass(frozen=True, slots=True)
class RunnerContext:
    """Everything a search runner receives for one job session."""

    job: TCCJobSpec
    job_id: str
    options: Mapping[str, Any]
    store: RunStore
    evaluator: Evaluator
    resume: ResumePlan
    hashes: RunHashes
    artifact_root: Path
    emit: EventSink
    should_stop: Callable[[], bool]
    active_gpu_seconds: Callable[[], float]
    unique_valid_evaluations: Callable[[], int]


class SearchRunner(Protocol):
    """Search algorithm entry point (implemented by the science layer; fakes in tests).

    Must call `ctx.should_stop()` between attempts and return "stopped" after
    saving a checkpoint, or "completed" when the run's search is over.
    """

    def __call__(self, ctx: RunnerContext) -> RunnerStatus: ...


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    run_id: str
    end_state: JobEndState
    detail: str
    attempt_ids: tuple[str, ...]
    session_gpu_seconds: float
    unrecorded_gpu_seconds: float
    interrupted_on_resume: tuple[str, ...]


class GpuMeter:
    """Monotonic stopwatch of the attempt in progress, written as ledger segments.

    `flush` may be called from the heartbeat thread while the evaluator runs;
    `stop` is called by the evaluator wrapper when the attempt's GPU work
    ends. Both write under the caller thread's lease fence.
    """

    def __init__(self, store: SQLiteRunStore, monotonic: Monotonic) -> None:
        self._store = store
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._open: tuple[str, str] | None = None
        self._mark = 0.0
        self.recorded_s = 0.0
        self.unrecorded_s = 0.0

    def start(self, run_id: str, attempt_id: str) -> None:
        with self._lock:
            if self._open is not None:
                raise ConsistencyError(f"attempt {self._open[1]!r} is still being metered")
            self._open = (run_id, attempt_id)
            self._mark = self._monotonic()

    def _flush_locked(self) -> None:
        if self._open is None:
            return
        now = self._monotonic()
        if now <= self._mark:
            return
        run_id, attempt_id = self._open
        try:
            self.recorded_s += self._store.record_job_gpu_segment(run_id, attempt_id, self._mark, now)
        except LeaseLostError:
            # The lease holder changed: this time cannot enter the ledger any more.
            self.unrecorded_s += now - self._mark
            self._open = None
            raise
        self._mark = now

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def stop(self) -> None:
        with self._lock:
            try:
                self._flush_locked()
            finally:
                self._open = None

    @property
    def open_attempt(self) -> str | None:
        with self._lock:
            return None if self._open is None else self._open[1]


class FencedRunStore:
    """`RunStore` port view that writes only while the lease is held."""

    def __init__(self, store: SQLiteRunStore, fence: LeaseFence) -> None:
        self._store = store
        self._fence = fence

    def create_run(self, job: TCCJobSpec) -> None:
        with self._store.lease_fence(self._fence):
            self._store.create_run(job)

    def register_candidate(
        self, run_id: str, candidate_id: str, genome: GenomeRecord, metadata: Mapping[str, Any]
    ) -> None:
        with self._store.lease_fence(self._fence):
            self._store.register_candidate(run_id, candidate_id, genome, metadata)

    def find_cached(self, run_id: str, cache_context_hash: str) -> EvaluationResult | None:
        return self._store.find_cached(run_id, cache_context_hash)

    def commit_evaluation(self, result: EvaluationResult) -> None:
        with self._store.lease_fence(self._fence):
            self._store.commit_evaluation(result)

    def save_checkpoint(self, checkpoint: SearchCheckpoint) -> None:
        with self._store.lease_fence(self._fence):
            self._store.save_checkpoint(checkpoint)

    def load_checkpoint(self, run_id: str) -> SearchCheckpoint | None:
        return self._store.load_checkpoint(run_id)

    def record_gpu_segment(self, run_id: str, attempt_id: str, gpu_seconds: float) -> None:
        with self._store.lease_fence(self._fence):
            self._store.record_gpu_segment(run_id, attempt_id, gpu_seconds)

    def record_front(
        self, run_id: str, candidate_ids: tuple[str, ...], active_gpu_seconds: float, hypervolume: float | None
    ) -> None:
        with self._store.lease_fence(self._fence):
            self._store.record_front(run_id, candidate_ids, active_gpu_seconds, hypervolume)

    def dataset_splits_for_run(self, run_id: str) -> Any:
        """The splits the run may read; the official test appears only for a finalist run (the store decides)."""
        return self._store.dataset_splits_for_run(run_id)

    def record_test_evaluation(
        self,
        test_id: str,
        *,
        run_id: str,
        selection_id: str,
        attempt_id: str,
        engine_sha256: str,
        metrics: Mapping[str, Any],
    ) -> None:
        """The single official-test opening of an engine, written only while the lease is held."""
        with self._store.lease_fence(self._fence):
            self._store.record_test_evaluation(
                test_id,
                run_id=run_id,
                selection_id=selection_id,
                attempt_id=attempt_id,
                engine_sha256=engine_sha256,
                metrics=metrics,
            )


class MeteredEvaluator:
    """Wraps the scientific evaluator: attempt row first, metered GPU time, safe failure.

    The committed `gpu_seconds` is `max(evaluator report, metered ledger)`;
    both values are kept in `quality` for audit. If the evaluator raises, the
    attempt is committed as `interrupted` with the metered time and the
    exception propagates.
    """

    def __init__(
        self,
        inner: Evaluator,
        store: SQLiteRunStore,
        fence: LeaseFence,
        meter: GpuMeter,
        on_attempt: Callable[[str], None],
    ) -> None:
        self._inner = inner
        self._store = store
        self._fence = fence
        self._meter = meter
        self._on_attempt = on_attempt

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        with self._store.lease_fence(self._fence):
            self._store.begin_attempt(request.run_id, request.candidate_id, request.attempt_id)
            self._on_attempt(request.attempt_id)
            self._meter.start(request.run_id, request.attempt_id)
            try:
                result = self._inner.evaluate(request)
            except LeaseLostError:
                self._meter.stop()
                raise
            except BaseException as exc:
                self._meter.stop()
                self._close_failed(request, exc)
                raise
            self._meter.stop()
            ids = (result.run_id, result.candidate_id, result.attempt_id)
            if ids != (request.run_id, request.candidate_id, request.attempt_id):
                self._close_failed(request, ConsistencyError(f"evaluator answered for {ids}"))
                raise ConsistencyError(f"evaluator result {ids} does not match its request")
            attempt = self._store.get_attempt(request.attempt_id)
            metered = 0.0 if attempt is None else attempt.ledger_gpu_seconds
            quality = {
                **dict(result.quality),
                "operation": {"metered_gpu_s": metered, "evaluator_reported_gpu_s": float(result.gpu_seconds)},
            }
            return dataclasses.replace(result, gpu_seconds=max(float(result.gpu_seconds), metered), quality=quality)

    def _close_failed(self, request: EvaluationRequest, exc: BaseException) -> None:
        attempt = self._store.get_attempt(request.attempt_id)
        metered = 0.0 if attempt is None else attempt.ledger_gpu_seconds
        self._store.commit_evaluation(
            EvaluationResult(
                run_id=request.run_id,
                candidate_id=request.candidate_id,
                attempt_id=request.attempt_id,
                status="interrupted",
                fitness=None,
                gpu_seconds=metered,
                quality={"operation": {"metered_gpu_s": metered}},
                failure_stage="operation",
                failure_code=EVALUATOR_EXCEPTION_CODE,
                failure_detail=f"{type(exc).__name__}: {exc}"[:500],
            )
        )


class OperationService:
    """One worker: claims jobs, keeps the lease alive, runs the injected search runner."""

    def __init__(
        self,
        store: SQLiteRunStore,
        runner: SearchRunner,
        evaluator: Evaluator,
        *,
        worker_id: str,
        lease_s: float = 120.0,
        heartbeat_s: float | None = None,
        heartbeat_thread: bool = True,
        enforce_budget: bool = True,
        monotonic: Monotonic = time.monotonic,
        on_event: EventSink | None = None,
        session_detail: Mapping[str, Any] | None = None,
        sentinel: SentinelHooks | None = None,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        self._store = store
        self._queue = JobQueue(store)
        self._runner = runner
        self._evaluator = evaluator
        self._worker_id = worker_id
        self._lease_s = float(lease_s)
        self._heartbeat_s = float(heartbeat_s) if heartbeat_s is not None else self._lease_s / 8
        if not 0 < self._heartbeat_s < self._lease_s:
            raise ValueError("heartbeat period must be positive and shorter than the lease")
        self._heartbeat_thread = heartbeat_thread
        self._enforce_budget = enforce_budget
        self._monotonic = monotonic
        self._on_event = on_event
        # Recorded in every claim event (e.g. code version of this worker session).
        self._session_detail = dict(session_detail or {})
        self._sentinel = sentinel  # measured between attempts, see sentinel_control
        self._stop = threading.Event()
        self._lease_lost = threading.Event()
        self._lease: Lease | None = None
        self._meter: GpuMeter | None = None
        self._hb_errors: list[str] = []

    @property
    def queue(self) -> JobQueue:
        return self._queue

    @property
    def heartbeat_errors(self) -> tuple[str, ...]:
        return tuple(self._hb_errors)

    def request_stop(self) -> None:
        """Planned shutdown: finish the current attempt, checkpoint, requeue the job."""
        self._stop.set()

    # -- events ----------------------------------------------------------

    def _emit(self, run_id: str, kind: str, payload: Mapping[str, Any], **ids: str | None) -> None:
        if self._on_event is None:
            return
        self._on_event(
            TCCEvent(
                event_id=uuid.uuid4().hex,
                ts_utc=format_utc(self._store.now()),
                run_id=run_id,
                kind=kind,
                payload=dict(payload),
                candidate_id=ids.get("candidate_id"),
                attempt_id=ids.get("attempt_id"),
            )
        )

    # -- heartbeat -------------------------------------------------------

    def heartbeat_now(self) -> None:
        """Flush metered GPU time and renew the lease (what the thread does periodically)."""
        lease, meter = self._lease, self._meter
        if lease is None or meter is None or self._lease_lost.is_set():
            return
        try:
            with self._store.lease_fence(lease.fence):
                meter.flush()
                self._queue.heartbeat(lease)
        except LeaseLostError:
            self._lease_lost.set()
            self._emit(lease.run_id, "lease_lost", {"job_id": lease.job_id, "worker_id": self._worker_id})

    def _heartbeat_loop(self, done: threading.Event) -> None:
        while not done.wait(self._heartbeat_s):
            try:
                self.heartbeat_now()
            except Exception as exc:  # transient (e.g. busy database): retry next period
                self._hb_errors.append(f"{type(exc).__name__}: {exc}")
            if self._lease_lost.is_set():
                return

    # -- main loop -------------------------------------------------------

    def run_until_idle(self, *, max_jobs: int | None = None) -> tuple[JobOutcome, ...]:
        outcomes: list[JobOutcome] = []
        while not self._stop.is_set() and (max_jobs is None or len(outcomes) < max_jobs):
            outcome = self.run_next()
            if outcome is None:
                break
            outcomes.append(outcome)
        return tuple(outcomes)

    def run_next(self, *, job_id: str | None = None) -> JobOutcome | None:
        """Claim and execute one job (or only `job_id`); None when nothing is runnable."""
        lease = self._queue.claim_next(
            self._worker_id,
            lease_s=self._lease_s,
            job_id=job_id,
            detail={**self._session_detail, "pid": os.getpid(), "host": platform.node()},
        )
        if lease is None:
            return None
        meter = GpuMeter(self._store, self._monotonic)
        self._lease, self._meter = lease, meter
        self._lease_lost.clear()
        attempts: list[str] = []
        done = threading.Event()
        beat = None
        if self._heartbeat_thread:
            beat = threading.Thread(
                target=self._heartbeat_loop, args=(done,), name=f"heartbeat-{lease.job_id}", daemon=True
            )
            beat.start()
        try:
            return self._execute(lease, meter, attempts)
        finally:
            done.set()
            if beat is not None:
                beat.join()
            self._lease, self._meter = None, None

    def _sentinel_tick(self, lease: Lease, state: SentinelState | None) -> None:
        """Measure the sentinel if the run reached the next point. Never raises into the run."""
        if self._sentinel is None or state is None or self._lease_lost.is_set():
            return
        count = len(self._store.evaluated_attempt_ids(lease.run_id))
        if count < state.next_at:
            return

        def record(kind: str, detail: Mapping[str, Any]) -> None:
            self._queue.record_sentinel(lease, kind, detail)
            self._emit(lease.run_id, kind, detail)

        try:
            sentinel_control.tick(
                self._sentinel,
                state,
                count,
                lambda: self._store.evaluated_attempt_ids(lease.run_id),
                record,
                clock=self._monotonic,
            )
        except LeaseLostError:
            self._lease_lost.set()

    def _cancel_requested(self, lease: Lease) -> bool:
        try:
            return self._queue.is_cancel_requested(lease)
        except LeaseLostError:
            self._lease_lost.set()
            return True

    def _execute(self, lease: Lease, meter: GpuMeter, attempts: list[str]) -> JobOutcome:
        store = self._store
        job_record = self._queue.get(lease.job_id)
        interrupted: tuple[str, ...] = ()

        def outcome(state: JobEndState, detail: str) -> JobOutcome:
            self._emit(lease.run_id, "job_ended", {"job_id": lease.job_id, "state": state, "detail": detail})
            return JobOutcome(
                job_id=lease.job_id,
                run_id=lease.run_id,
                end_state=state,
                detail=detail,
                attempt_ids=tuple(attempts),
                session_gpu_seconds=meter.recorded_s,
                unrecorded_gpu_seconds=meter.unrecorded_s,
                interrupted_on_resume=interrupted,
            )

        budget_reached = threading.Event()
        sentinel_state = (
            None
            if self._sentinel is None
            else SentinelState.from_events(self._queue.sentinel_events(lease.run_id), self._sentinel.every)
        )
        exhausted: list[dict[str, Any]] = []  # payloads of `search_exhausted` events of this session
        forward = self._on_event or (lambda _event: None)

        def emit(event: TCCEvent) -> None:
            if event.kind == "search_exhausted":
                exhausted.append(dict(event.payload))
            forward(event)

        def should_stop() -> bool:
            if self._stop.is_set() or self._lease_lost.is_set() or self._cancel_requested(lease):
                return True
            # Cut between attempts: the attempt that crosses the budget finishes and counts.
            if self._enforce_budget:
                budget = store.get_run(lease.run_id).budget_gpu_s  # type: ignore[union-attr]
                if store.active_gpu_seconds(lease.run_id) >= budget:
                    budget_reached.set()
                    return True
            self._sentinel_tick(lease, sentinel_state)  # between attempts, and only when the run goes on
            return False

        try:
            with store.lease_fence(lease.fence):
                job = store.get_run(lease.run_id)
                if job is None:  # FK makes this impossible; kept for a clear message
                    raise ConsistencyError(f"job {lease.job_id!r} points to a missing run")
                resume = prepare_resume(store, job)
                interrupted = resume.interrupted_attempt_ids
                self._emit(
                    lease.run_id,
                    "job_started",
                    {
                        "job_id": lease.job_id,
                        "claim": lease.claim_number,
                        "interrupted_on_resume": list(interrupted),
                        "retry_candidates": list(resume.retry_candidate_ids),
                        "active_gpu_seconds": resume.active_gpu_seconds,
                    },
                )
                ctx = RunnerContext(
                    job=job,
                    job_id=lease.job_id,
                    options=job_record.options,
                    store=FencedRunStore(store, lease.fence),
                    evaluator=MeteredEvaluator(self._evaluator, store, lease.fence, meter, attempts.append),
                    resume=resume,
                    hashes=store.run_hashes(lease.run_id),
                    artifact_root=store.artifact_root,
                    emit=emit,
                    should_stop=should_stop,
                    active_gpu_seconds=lambda: store.active_gpu_seconds(lease.run_id),
                    unique_valid_evaluations=lambda: store.unique_valid_evaluations(lease.run_id),
                )
                status = self._runner(ctx)
        except LeaseLostError:
            self._lease_lost.set()
            return outcome("lease_lost", "lease taken over by another worker; nothing more was written")
        except KeyboardInterrupt:
            return self._end(lease, meter, outcome, "queued", "stopped by operator interrupt", close_open=True)
        except Exception as exc:
            if self._lease_lost.is_set():
                return outcome("lease_lost", f"lease lost; last error {type(exc).__name__}: {exc}")
            return self._end(lease, meter, outcome, "failed", f"{type(exc).__name__}: {exc}"[:500], close_open=True)

        if self._lease_lost.is_set():
            return outcome("lease_lost", "lease taken over by another worker; nothing more was written")
        if status == "completed" and exhausted:
            spent = store.active_gpu_seconds(lease.run_id)
            budget = store.get_run(lease.run_id).budget_gpu_s  # type: ignore[union-attr]
            detail = (
                f"espaço esgotado antes do orçamento: {exhausted[-1].get('stalled_generations')} gerações seguidas "
                f"sem nenhuma avaliação nova ({spent:.1f}s de {budget:.1f}s de GPU ativa gastos)"
            )
            return self._end(lease, meter, outcome, "done", detail)
        if status == "completed":
            return self._end(lease, meter, outcome, "done", "runner completed")
        if budget_reached.is_set() and not self._stop.is_set() and not self._cancel_requested(lease):
            spent = store.active_gpu_seconds(lease.run_id)
            return self._end(lease, meter, outcome, "done", f"GPU budget reached ({spent:.1f}s active)")
        if self._cancel_requested(lease) and not self._lease_lost.is_set():
            reason = self._queue.get(lease.job_id).cancel_reason or "cancelled"
            return self._end(lease, meter, outcome, "cancelled", f"cancelled cooperatively: {reason}")
        return self._end(lease, meter, outcome, "queued", "planned stop; job requeued")

    def _end(
        self,
        lease: Lease,
        meter: GpuMeter,
        outcome: Callable[[JobEndState, str], JobOutcome],
        state: Literal["done", "failed", "cancelled", "queued"],
        detail: str,
        *,
        close_open: bool = False,
    ) -> JobOutcome:
        try:
            with self._store.lease_fence(lease.fence):
                meter.stop()
                if close_open:
                    # A runner that died between evaluate() and commit leaves a running
                    # attempt; close it (time kept) before the job can end.
                    self._store.recover_interrupted_attempts(lease.run_id)
            if state == "queued":
                self._queue.release(lease, reason=detail)
            else:
                self._queue.finish(lease, state, detail=detail)
        except LeaseLostError:
            self._lease_lost.set()
            return outcome("lease_lost", f"lease lost before ending the job ({detail})")
        return outcome(state, detail)
