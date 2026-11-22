"""The sentinel network under the operation service (task A9). CPU only, with a fake hook.

The hook stands for ``model_pipeline.sentinel``: ``measure`` returns what ``measure_sentinel`` returns (the fields the
operation looks at are ``plan_sha256``, ``l_ms`` and ``v_mb``) and ``drift`` has the signature of ``sentinel.drift``.
The evaluator is the synthetic one: nothing here is measured.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import TCCEvent
from cnnlab.tcc.operation import OperationService
from cnnlab.tcc.operation.sentinel_control import SentinelHooks, SentinelState, tick
from cnnlab.tcc.operation.testing import ClockedFakeEvaluator, FakeMonotonic, SyntheticListRunner
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.errors import LeaseLostError
from cnnlab.tcc.storage.testing import ManualClock, make_job, seed_reference_rows

JOB = make_job("run-sent", method="joint", seed_block=1)
GPU_S = 10.0
SENTINEL_S = 7.0
EVERY = 5


class FakeSentinel:
    """What ``measure_sentinel`` and ``drift`` look like to the operation, driven by a script."""

    def __init__(
        self,
        mono: FakeMonotonic,
        l_ms: dict[int, float] | None = None,
        fail_at: set[int] | None = None,
        other_engine_at: set[int] | None = None,
    ) -> None:
        self.mono, self.calls = mono, 0
        self.l_ms = l_ms or {}  # n-th measurement (0 is the baseline) -> latency; default 1.0
        self.fail_at, self.other_engine_at = fail_at or set(), other_engine_at or set()

    def measure(self) -> dict[str, Any]:
        n = self.calls
        self.calls += 1
        self.mono.advance(SENTINEL_S)  # the measurement takes time, outside the budget
        if n in self.fail_at:
            raise RuntimeError("bench process died")
        return {
            "sentinel_version": "fake/1",
            "plan_sha256": "e2" * 32 if n in self.other_engine_at else "e1" * 32,
            "l_ms": self.l_ms.get(n, 1.0),
            "v_mb": 2.0,
            "s_mb": 3.0,
        }

    @staticmethod
    def drift(reference: Any, current: Any, *, limit: float = 0.05) -> dict[str, Any]:
        if reference["plan_sha256"] != current["plan_sha256"]:
            raise ValueError("sentinel measurements of different engines cannot be compared")
        l_rel = abs(current["l_ms"] - reference["l_ms"]) / reference["l_ms"]
        v_rel = abs(current["v_mb"] - reference["v_mb"]) / reference["v_mb"]
        return {"l_rel": l_rel, "v_rel": v_rel, "limit": limit, "drifted": l_rel > limit or v_rel > limit}

    def hooks(self, every: int = EVERY) -> SentinelHooks:
        return SentinelHooks(measure=self.measure, drift=self.drift, every=every)


@pytest.fixture
def mono() -> FakeMonotonic:
    return FakeMonotonic()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts", clock=ManualClock())
    seed_reference_rows(s)
    s.create_run(JOB)
    JobQueue(s).enqueue("job-1", JOB.run_id, options={"synthetic": True})
    yield s
    s.close()


def service(
    store: SQLiteRunStore,
    mono: FakeMonotonic,
    sentinel: FakeSentinel | None,
    *,
    n: int = 12,
    events: list[TCCEvent] | None = None,
    contexts: tuple[str | None, ...] = (),
    every: int = EVERY,
    evaluator: ClockedFakeEvaluator | None = None,
    worker: str = "w1",
) -> OperationService:
    return OperationService(
        store,
        SyntheticListRunner(n_candidates=n, contexts=contexts, checkpoint_every=100),
        evaluator or ClockedFakeEvaluator(gpu_seconds=GPU_S, clock=mono),
        worker_id=worker,
        lease_s=60.0,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=(events.append if events is not None else None),
        sentinel=None if sentinel is None else sentinel.hooks(every),
    )


def sentinel_events(store: SQLiteRunStore) -> list[dict[str, Any]]:
    return JobQueue(store).sentinel_events(JOB.run_id)


def test_without_hooks_nothing_changes(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    assert service(store, mono, None).run_next().end_state == "done"  # type: ignore[union-attr]
    assert sentinel_events(store) == [] and store.evaluated_attempt_ids(JOB.run_id) == [f"c{i}.a1" for i in range(12)]


def test_a_baseline_then_a_measurement_every_n_evaluated_attempts_between_attempts(
    store: SQLiteRunStore, mono: FakeMonotonic
) -> None:
    fake = FakeSentinel(mono)
    events: list[TCCEvent] = []
    outcome = service(store, mono, fake, events=events).run_next()
    assert outcome is not None and outcome.end_state == "done"
    measured = [e for e in sentinel_events(store) if e["kind"] == "sentinel_measured"]
    assert [m["detail"]["evaluated_attempts"] for m in measured] == [0, 5, 10]  # the baseline, then every 5
    assert [m["detail"]["reference"] for m in measured] == [True, False, False]  # the first one is the reference
    assert [m["detail"]["index"] for m in measured] == [1, 2, 3]
    assert all(
        m["detail"]["seconds"] == SENTINEL_S and m["detail"]["measurement"]["plan_sha256"] == "e1" * 32
        for m in measured
    )
    assert not [e for e in sentinel_events(store) if e["kind"] != "sentinel_measured"]  # no drift, no failure
    assert [e.kind for e in events if e.kind.startswith("sentinel")] == [
        "sentinel_measured"
    ] * 3  # the log shows them too
    assert fake.calls == 3

    # between attempts, never during one: every measurement fell where no attempt was running
    with ReadModel(store.db_path) as rm:
        assert [a["status"] for a in rm.evaluations(JOB.run_id)] == ["valid"] * 12
        assert [e["kind"] for e in rm.sentinel_events(JOB.run_id)] == ["sentinel_measured"] * 3
        kinds = [e["kind"] for e in rm.job_events("job-1")]
    assert kinds[0] == "enqueued" and kinds[-1] == "done" and kinds.count("sentinel_measured") == 3

    # the time of the sentinel is not part of the budget of the run
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(12 * GPU_S)
    assert sum(m["detail"]["seconds"] for m in measured) == 3 * SENTINEL_S


def test_only_attempts_that_went_through_the_evaluator_are_counted(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    # c0..c3 share two cache contexts: c1 and c3 are cache hits, which never reach the evaluator
    ctx = ("a", "a", "b", "b", None, None, None, None, None, None, None, None)
    fake = FakeSentinel(mono)
    service(store, mono, fake, contexts=ctx).run_next()
    assert len(store.evaluated_attempt_ids(JOB.run_id)) == 10  # 12 attempts, 2 cache hits
    assert "c1.a1" not in store.evaluated_attempt_ids(JOB.run_id) and "c3.a1" not in store.evaluated_attempt_ids(
        JOB.run_id
    )
    # the point at 10 is never reached: the run has nothing left to evaluate, and a control after the end is useless
    assert [e["detail"]["evaluated_attempts"] for e in sentinel_events(store)] == [0, 5]


def test_a_drift_is_recorded_with_the_attempts_it_marks_and_nothing_is_replaced(
    store: SQLiteRunStore, mono: FakeMonotonic
) -> None:
    fake = FakeSentinel(mono, l_ms={1: 1.10, 2: 1.0})  # drifted at the 5th attempt, back at the 10th
    service(store, mono, fake, n=16).run_next()
    events = sentinel_events(store)
    drifts = [e for e in events if e["kind"] == "sentinel_drift"]
    assert len(drifts) == 1
    detail = drifts[0]["detail"]
    assert detail["evaluated_attempts_from"] == 1 and detail["evaluated_attempts_to"] == 5  # since the baseline
    assert detail["attempt_ids"] == [f"c{i}.a1" for i in range(5)]
    assert detail["drift"]["drifted"] is True and detail["drift"]["l_rel"] == pytest.approx(0.10)
    assert detail["drift"]["limit"] == 0.05 and detail["drift"]["v_rel"] == 0.0
    assert [e["kind"] for e in events] == [
        "sentinel_measured",
        "sentinel_measured",
        "sentinel_drift",
        "sentinel_measured",
        "sentinel_measured",
    ]  # measured, then drift, in order
    with ReadModel(store.db_path) as rm:  # nothing was touched: every attempt is as it was committed
        assert [a["status"] for a in rm.evaluations(JOB.run_id)] == ["valid"] * 16


def test_a_drift_that_goes_on_marks_each_window_it_was_seen_in(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    fake = FakeSentinel(mono, l_ms={1: 1.2, 2: 1.2, 3: 1.01})  # against the first measurement, not the previous
    service(store, mono, fake, n=16).run_next()
    drifts = [e["detail"] for e in sentinel_events(store) if e["kind"] == "sentinel_drift"]
    assert [(d["evaluated_attempts_from"], d["evaluated_attempts_to"]) for d in drifts] == [(1, 5), (6, 10)]
    assert drifts[1]["attempt_ids"] == [f"c{i}.a1" for i in range(5, 10)]  # the window after the previous measurement


def test_a_failed_measurement_is_recorded_and_the_job_goes_on(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    fake = FakeSentinel(mono, fail_at={0})  # the baseline fails: the first one that works becomes the reference
    outcome = service(store, mono, fake).run_next()
    assert outcome is not None and outcome.end_state == "done"
    events = sentinel_events(store)
    assert [e["kind"] for e in events] == ["sentinel_failed", "sentinel_measured", "sentinel_measured"]
    failed = events[0]["detail"]
    assert failed["stage"] == "measure" and "RuntimeError: bench process died" in failed["error"]
    assert failed["evaluated_attempts"] == 0 and failed["seconds"] == SENTINEL_S
    assert [e["detail"]["reference"] for e in events[1:]] == [True, False]  # the reference is the first that worked
    assert events[1]["detail"]["evaluated_attempts"] == 5


def test_measurements_of_another_engine_are_not_compared_and_say_so(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    fake = FakeSentinel(mono, other_engine_at={1}, l_ms={1: 5.0})
    service(store, mono, fake).run_next()
    events = sentinel_events(store)
    assert [e["kind"] for e in events] == [
        "sentinel_measured",
        "sentinel_measured",
        "sentinel_failed",
        "sentinel_measured",
    ]
    failed = events[2]["detail"]
    assert failed["stage"] == "compare" and "different engines cannot be compared" in failed["error"]
    assert not any(e["kind"] == "sentinel_drift" for e in events)  # it is not a drift: it is no comparison at all


def test_a_resumed_job_continues_the_schedule_and_keeps_the_reference(
    store: SQLiteRunStore, mono: FakeMonotonic
) -> None:
    holder: dict[str, OperationService] = {}

    class StopAfter(ClockedFakeEvaluator):
        def evaluate(self, request: Any) -> Any:
            result = super().evaluate(request)
            if len(self.calls) == 7:
                holder["svc"].request_stop()  # a planned stop after the 7th evaluated attempt
            return result

    fake = FakeSentinel(mono, l_ms={1: 1.5})  # the second measurement (at 5) drifts
    first = service(store, mono, fake, evaluator=StopAfter(gpu_seconds=GPU_S, clock=mono))
    holder["svc"] = first
    outcome = first.run_next()
    assert outcome is not None and outcome.end_state == "queued"
    assert [e["detail"]["evaluated_attempts"] for e in sentinel_events(store)] == [0, 5, 5]  # measured twice, one drift

    second = service(store, mono, fake, worker="w2")  # a new session of the same run, with the same hooks
    assert second.run_next().end_state == "done"  # type: ignore[union-attr]
    events = sentinel_events(store)
    measured = [e["detail"] for e in events if e["kind"] == "sentinel_measured"]
    assert [m["evaluated_attempts"] for m in measured] == [0, 5, 10]  # no second baseline, 10 is the next point
    assert [m["reference"] for m in measured] == [True, False, False]
    assert len({e["job_id"] for e in events}) == 1  # one job here; the reference would survive a new job as well
    assert fake.calls == 3


def test_the_hook_runs_only_when_the_run_goes_on(store: SQLiteRunStore, mono: FakeMonotonic) -> None:
    fake = FakeSentinel(mono)
    evaluator = ClockedFakeEvaluator(gpu_seconds=5000.0, clock=mono)  # the 4th attempt crosses the 19,800 s budget
    outcome = service(store, mono, fake, evaluator=evaluator, n=12, every=2).run_next()
    assert outcome is not None and "budget reached" in outcome.detail
    counts = [e["detail"]["evaluated_attempts"] for e in sentinel_events(store)]
    assert counts == [0, 2]  # at 4 attempts the budget stops the run first: no measurement after the end
    assert store.active_gpu_seconds(JOB.run_id) == pytest.approx(20000.0)  # and none of its time is in the ledger


# -- the pure schedule and the storage ----------------------------------------------------------------------------


def test_the_schedule_is_rebuilt_from_the_events() -> None:
    events = [
        {"kind": "sentinel_measured", "detail": {"evaluated_attempts": 0, "measurement": {"l_ms": 1.0}}},
        {"kind": "sentinel_failed", "detail": {"evaluated_attempts": 50}},
        {"kind": "sentinel_drift", "detail": {"evaluated_attempts": 50}},  # a drift event does not move the schedule
        {"kind": "sentinel_measured", "detail": {"evaluated_attempts": 101, "measurement": {"l_ms": 1.2}}},
    ]
    state = SentinelState.from_events(events, 50)
    assert (
        state.reference == {"l_ms": 1.0}
        and state.last_count == 101
        and state.next_at == 150
        and state.measurements == 2
    )
    assert SentinelState.from_events([], 50).next_at == 0  # a run with no events measures at once


def test_tick_does_nothing_before_its_time_and_asks_for_the_attempts_only_on_a_drift() -> None:
    recorded: list[tuple[str, dict[str, Any]]] = []
    asked: list[int] = []
    mono = FakeMonotonic()
    fake = FakeSentinel(mono, l_ms={1: 2.0})
    state = SentinelState()

    def ids() -> list[str]:
        asked.append(1)
        return [f"a{i}" for i in range(20)]

    def record(kind: str, detail: Any) -> None:
        recorded.append((kind, dict(detail)))

    assert tick(fake.hooks(), state, 0, ids, record, clock=mono) is True  # baseline
    assert tick(fake.hooks(), state, 3, ids, record, clock=mono) is False and fake.calls == 1  # not yet
    assert asked == []
    assert tick(fake.hooks(), state, 6, ids, record, clock=mono) is True  # at 5 or later: measured, and it drifted
    assert (
        asked == [1]
        and recorded[-1][0] == "sentinel_drift"
        and recorded[-1][1]["attempt_ids"] == [f"a{i}" for i in range(6)]
    )
    assert state.next_at == 10  # the next point is the next multiple after what was reached


def test_the_hooks_refuse_a_period_that_is_not_a_whole_number_of_attempts() -> None:
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="period"):
            SentinelHooks(measure=lambda: {}, drift=lambda a, b: {}, every=bad)  # type: ignore[arg-type]
    assert SentinelHooks(measure=lambda: {}, drift=lambda a, b: {}).every == 50  # the protocol's


def test_only_the_lease_holder_records_and_only_sentinel_kinds(store: SQLiteRunStore) -> None:
    queue = JobQueue(store)
    lease = queue.claim_next("w", lease_s=60.0)
    assert lease is not None
    queue.record_sentinel(lease, "sentinel_measured", {"evaluated_attempts": 0})
    with pytest.raises(ValueError, match="not a sentinel event kind"):
        queue.record_sentinel(lease, "done", {})
    queue.release(lease, reason="test")
    with pytest.raises(LeaseLostError):
        queue.record_sentinel(
            lease, "sentinel_measured", {"evaluated_attempts": 1}
        )  # the lease is gone: nothing written
    assert [e["detail"] for e in queue.sentinel_events(JOB.run_id)] == [{"evaluated_attempts": 0}]


# -- what the analysis does with the events -------------------------------------------------------------------------


REFERENCE = (0.99, 5.2, 52.0, 20.2)  # beyond everything the fake evaluator can return


def test_the_analysis_labels_the_marked_attempts_and_declares_the_cost_of_the_sentinel(
    store: SQLiteRunStore, mono: FakeMonotonic, tmp_path: Path
) -> None:
    import csv
    import json

    from cnnlab.tcc.operation import export, hv_analysis

    fake = FakeSentinel(mono, l_ms={1: 1.10, 2: 1.0}, fail_at={3})
    service(store, mono, fake, n=16).run_next()  # measured at 0, 5 (drift), 10, failed at 15
    db = store.db_path

    curves, _, _ = hv_analysis.read_curves(db, reference=REFERENCE)
    [curve] = curves
    assert curve.drift_attempts == frozenset(f"c{i}.a1" for i in range(5))
    assert [p["sentinel_drift"] for p in curve.points] == [True] * 5 + [False] * 11  # labeled, nothing removed
    assert (curve.sentinel_measurements, curve.sentinel_drifts, curve.sentinel_failures) == (3, 1, 1)
    assert curve.sentinel_gpu_s == 4 * SENTINEL_S  # the failed one cost its time too
    assert curve.unique == 16 and curve.gpu_total_s == 16 * GPU_S  # the curve and the budget are as they were

    done = hv_analysis.export_hv(db, tmp_path / "hv", reference=REFERENCE, allow_synthetic=True)
    with open(tmp_path / "hv" / "hv_final.csv", encoding="utf-8", newline="") as fh:
        [final] = list(csv.DictReader(fh))
    assert final["sentinel_gpu_s"] == repr(28.0) and final["sentinel_measurements"] == "3"
    assert (final["sentinel_failures"], final["sentinel_drifts"], final["points_in_drift"]) == ("1", "1", "5")
    assert float(final["gpu_total_s"]) == 160.0  # the sentinel's 28 s are not in it
    with open(tmp_path / "hv" / "hv_curvas.csv", encoding="utf-8", newline="") as fh:
        flags = [r["sentinel_drift"] for r in csv.DictReader(fh)]
    assert flags == ["true"] * 5 + ["false"] * 11
    assert (
        done.manifest["sentinel_gpu_s_total"] == 28.0 and "not part of the GPU budget" in done.manifest["sentinel_note"]
    )
    assert done.manifest["runs"][0]["sentinel_drifts"] == 1

    export.export_run(db, JOB.run_id, tmp_path / "csv", allow_synthetic=True)  # the CSV of the run says the same
    with open(tmp_path / "csv" / "tentativas.csv", encoding="utf-8", newline="") as fh:
        attempts = list(csv.DictReader(fh))
    assert [a["sentinel_drift"] for a in attempts] == ["true"] * 5 + ["false"] * 11
    with open(tmp_path / "csv" / "medidas.csv", encoding="utf-8", newline="") as fh:
        assert [m["sentinel_drift"] for m in csv.DictReader(fh)] == ["true"] * 5 + ["false"] * 11
    with open(tmp_path / "csv" / "tempo_gpu.csv", encoding="utf-8", newline="") as fh:
        gpu = {r["escopo"]: r for r in csv.DictReader(fh)}
    assert float(next(v for k, v in gpu.items() if k.startswith("sentinela"))["gpu_ativa_s_SINTETICO"]) == 28.0
    manifest = json.loads((tmp_path / "csv" / "manifesto.json").read_text(encoding="utf-8"))
    assert manifest["sentinel"] == {
        "measured": 3,
        "drift": 1,
        "failed": 1,
        "seconds_outside_budget": 28.0,
        "attempts_marked_by_drift": 5,
    }


def test_a_run_without_a_sentinel_exports_the_same_columns_with_no_flag(
    store: SQLiteRunStore, mono: FakeMonotonic, tmp_path: Path
) -> None:
    import csv

    from cnnlab.tcc.operation import export, hv_analysis

    service(store, mono, None, n=6).run_next()
    [curve], _, _ = hv_analysis.read_curves(store.db_path, reference=REFERENCE)
    assert (
        curve.sentinel_gpu_s == 0.0
        and curve.drift_attempts == frozenset()
        and not any(p["sentinel_drift"] for p in curve.points)
    )
    export.export_run(store.db_path, JOB.run_id, tmp_path / "csv", allow_synthetic=True)
    with open(tmp_path / "csv" / "tempo_gpu.csv", encoding="utf-8", newline="") as fh:
        assert not any(r["escopo"].startswith("sentinela") for r in csv.DictReader(fh))  # nothing to declare
