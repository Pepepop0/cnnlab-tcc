"""The science SearchRunner injected into the operation service, CPU and synthetic only.

Skipped until ``cnnlab.tcc.operation.service`` is present on the branch
(it lives on the operation front). With both fronts merged it checks that the
runner satisfies the service's ``SearchRunner`` protocol without either side
importing the other: the job ends ``done``, every candidate has exactly one
valid attempt, GPU time is metered, and a cooperative stop followed by a new
service session completes the run without repeating a valid evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

service_mod = pytest.importorskip("cnnlab.tcc.operation.service")
testing_mod = pytest.importorskip("cnnlab.tcc.operation.testing")

from cnnlab.tcc.contracts import TCCEvent, TCCJobSpec  # noqa: E402
from cnnlab.tcc.science.operators import FP32_SLICE  # noqa: E402
from cnnlab.tcc.science.record import CacheContext  # noqa: E402
from cnnlab.tcc.science.search import SearchConfig, SearchRunner  # noqa: E402
from cnnlab.tcc.storage import JobQueue, SQLiteRunStore  # noqa: E402
from cnnlab.tcc.storage.testing import ManualClock, make_job, seed_reference_rows  # noqa: E402

CONFIG = SearchConfig(population=4, generations=2)
JOB = make_job("run-ga-svc", method="joint", stage="pilot", seed_block=2, space_hash=FP32_SLICE.space_hash())


def context_for(job: TCCJobSpec) -> CacheContext:
    return CacheContext(
        split_fingerprint="synthetic-split",
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        train_policy={"synthetic": True},
        pruning_policy={},
        quantization_policy={},
        code_version="test",
        backend={"synthetic": True},
        builder_config={},
        space_hash=FP32_SLICE.space_hash(),
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(tmp_path / "db.sqlite", tmp_path / "artifacts", clock=ManualClock())
    seed_reference_rows(s)
    s.create_run(JOB)
    JobQueue(s).enqueue("job-ga", JOB.run_id, options={"synthetic": True})
    yield s
    s.close()


def service(store: SQLiteRunStore, events: list[TCCEvent], stop_after: int | None = None) -> object:
    mono = testing_mod.FakeMonotonic()
    evaluator = testing_mod.ClockedFakeEvaluator(gpu_seconds=3.0, clock=mono)
    svc = service_mod.OperationService(
        store,
        SearchRunner(CONFIG, FP32_SLICE, context_for),
        evaluator,
        worker_id="worker-ga",
        lease_s=60.0,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=events.append,
    )
    if stop_after is not None:

        def on_event(event: TCCEvent) -> None:
            events.append(event)
            if sum(e.kind == "candidate_evaluated" for e in events) >= stop_after:
                svc.request_stop()

        svc._on_event = on_event  # type: ignore[attr-defined]
    return svc


def valid_per_candidate(store: SQLiteRunStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in store.list_attempts(JOB.run_id):
        if attempt.status == "valid":
            counts[attempt.candidate_id] = counts.get(attempt.candidate_id, 0) + 1
    return counts


def test_search_runner_runs_under_the_operation_service(store: SQLiteRunStore) -> None:
    events: list[TCCEvent] = []
    outcome = service(store, events).run_next()  # type: ignore[attr-defined]
    assert outcome is not None and outcome.end_state == "done", outcome
    counts = valid_per_candidate(store)
    assert len(counts) == CONFIG.population * (1 + CONFIG.generations)
    assert set(counts.values()) == {1}
    assert store.active_gpu_seconds(JOB.run_id) > 0
    assert store.load_checkpoint(JOB.run_id) is not None
    assert any(e.kind == "search_completed" for e in events)


def test_stop_then_new_session_completes_without_repeating_valid_work(store: SQLiteRunStore) -> None:
    events: list[TCCEvent] = []
    first = service(store, events, stop_after=3).run_next()  # type: ignore[attr-defined]
    assert first is not None and first.end_state == "queued", first
    done_before = set(valid_per_candidate(store))
    second = service(store, []).run_next()  # type: ignore[attr-defined]
    assert second is not None and second.end_state == "done", second
    counts = valid_per_candidate(store)
    assert done_before <= set(counts)
    assert len(counts) == CONFIG.population * (1 + CONFIG.generations)
    assert set(counts.values()) == {1}
