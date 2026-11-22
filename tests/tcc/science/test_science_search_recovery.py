"""Search runner under infrastructure failures and crash recovery (CPU, synthetic evaluator).

Every fitness here comes from ``FakeEvaluator`` (hash-derived, labelled
synthetic). The tests check bookkeeping only: which attempts exist, that an
unobserved candidate is never ranked as invalid, and that a resumed run ends
with the population of an uninterrupted one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from test_science_search import context_for, make_direct_context, new_run, store_population  # same directory

from cnnlab.tcc.contracts import EvaluationRequest, EvaluationResult, TCCEvent
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.search import DirectResume, SearchBlockedError, SearchConfig, SearchRunner
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator

CONFIG = SearchConfig(population=4, generations=2)
TOTAL = CONFIG.population * (1 + CONFIG.generations)


@dataclass
class FlakyEvaluator:
    """``FakeEvaluator`` whose infrastructure fails for chosen attempts (no candidate is observed)."""

    interrupt: dict[str, int] = field(default_factory=dict)  # candidate_id -> how many attempts fail
    crash_on_call: int | None = None  # raise on the n-th call (1-based), as a killed process would
    inner: FakeEvaluator = field(default_factory=lambda: FakeEvaluator(gpu_seconds=2.0))
    calls: list[str] = field(default_factory=list)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request.attempt_id)
        if self.crash_on_call is not None and len(self.calls) == self.crash_on_call:
            raise RuntimeError("synthetic crash of the evaluator process")
        if self.interrupt.get(request.candidate_id, 0) > 0:
            self.interrupt[request.candidate_id] -= 1
            return EvaluationResult(
                run_id=request.run_id,
                candidate_id=request.candidate_id,
                attempt_id=request.attempt_id,
                status="interrupted",
                fitness=None,
                gpu_seconds=1.5,
                quality={"synthetic": True},
                failure_stage="benchmark",
                failure_code="infrastructure_spawn_failed",
                failure_detail="synthetic: child never started",
            )
        return self.inner.evaluate(request)


def attempts(store: SQLiteRunStore, run_id: str) -> list[tuple[str, str]]:
    return [(a.attempt_id, a.status) for a in store.list_attempts(run_id)]


def resume_from_store(store: SQLiteRunStore, run_id: str) -> DirectResume:
    """What ``operation.recovery.prepare_resume`` hands to the runner, built from the same store calls."""
    store.recover_interrupted_attempts(run_id)
    history = store.run_history(run_id)
    return DirectResume(
        checkpoint=history.latest_checkpoint,
        attempts_per_candidate=dict(Counter(a.candidate_id for a in history.attempts)),
        retry_candidate_ids=history.unresolved_candidate_ids,
    )


def reference_population(tmp_path: Path) -> set[str]:
    store, job = new_run(tmp_path, name="reference")
    assert (
        SearchRunner(CONFIG, FP32_SLICE, context_for)(make_direct_context(store, job, FakeEvaluator())) == "completed"
    )
    population = set(store_population(store, job.run_id))
    store.close()
    return population


def test_interrupted_attempt_is_retried_with_a_new_attempt_id(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="flaky")
    victim = f"{job.run_id}.g000.c0001"
    evaluator = FlakyEvaluator(interrupt={victim: 1})
    events: list[TCCEvent] = []
    status = SearchRunner(CONFIG, FP32_SLICE, context_for)(make_direct_context(store, job, evaluator, events=events))
    assert status == "completed"
    by_candidate = [a for a in attempts(store, job.run_id) if a[0].startswith(victim)]
    assert by_candidate == [(f"{victim}.a1", "interrupted"), (f"{victim}.a2", "valid")]
    assert len(attempts(store, job.run_id)) == TOTAL + 1
    assert [e.kind for e in events].count("candidate_interrupted") == 1
    # The interrupted attempt consumed GPU time, and that time stays in the ledger.
    assert store.active_gpu_seconds(job.run_id) == pytest.approx(1.5 + 2.0 * TOTAL)
    # The retried candidate was measured, so the search is the same as without the failure.
    assert set(store_population(store, job.run_id)) == reference_population(tmp_path)
    store.close()


def test_persistent_interruption_blocks_the_search_instead_of_ranking_it_invalid(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="blocked")
    victim = f"{job.run_id}.g000.c0002"
    evaluator = FlakyEvaluator(interrupt={victim: 2})  # first try and its single retry both fail
    events: list[TCCEvent] = []
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    with pytest.raises(SearchBlockedError, match="interrupted 2 times"):
        runner(make_direct_context(store, job, evaluator, events=events))
    recorded = dict(attempts(store, job.run_id))
    assert recorded[f"{victim}.a1"] == recorded[f"{victim}.a2"] == "interrupted"
    assert "invalid" not in recorded.values()
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None and checkpoint.pending_candidates[0] == victim
    assert events[-1].kind == "search_blocked"

    # Infrastructure fixed: the next session retries with a3 and finishes like an undisturbed run.
    resume = resume_from_store(store, job.run_id)
    assert resume.attempts_per_candidate[victim] == 2
    healthy = FlakyEvaluator()
    assert runner(make_direct_context(store, job, healthy, resume=resume)) == "completed"
    assert healthy.calls[0] == f"{victim}.a3"
    assert dict(attempts(store, job.run_id))[f"{victim}.a3"] == "valid"
    assert set(store_population(store, job.run_id)) == reference_population(tmp_path)
    store.close()


@pytest.mark.parametrize("crash_on_call", [1, 3, 4, 6, 9, 12])
def test_crash_recovery_adopts_committed_results_and_matches_an_undisturbed_run(
    tmp_path: Path, crash_on_call: int
) -> None:
    """Kill the evaluator at several points: the checkpoint is then older than the store."""
    store, job = new_run(tmp_path, name="crash")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    crashing = FlakyEvaluator(crash_on_call=crash_on_call)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        runner(make_direct_context(store, job, crashing))
    committed_before = {a[0] for a in attempts(store, job.run_id)}
    assert len(committed_before) == crash_on_call - 1

    resume = resume_from_store(store, job.run_id)
    events: list[TCCEvent] = []
    healthy = FlakyEvaluator()
    assert runner(make_direct_context(store, job, healthy, events=events, resume=resume)) == "completed"

    # Nothing committed before the crash is evaluated again, and nothing is missing.
    final = attempts(store, job.run_id)
    assert {a[0] for a in final} == committed_before | set(healthy.calls)
    assert committed_before.isdisjoint(healthy.calls)
    assert len(final) == TOTAL and {status for _, status in final} == {"valid"}
    per_candidate = Counter(a.candidate_id for a in store.list_attempts(job.run_id))
    assert set(per_candidate.values()) == {1}
    assert set(store_population(store, job.run_id)) == reference_population(tmp_path)
    store.close()


def test_adoption_keeps_invalid_results_invalid(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="adopt-invalid")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    # Learn the cache context of the first candidate from a dry registration run.
    probe_store, probe_job = new_run(tmp_path, name="probe")
    runner(make_direct_context(probe_store, probe_job, FakeEvaluator(), should_stop=lambda: True))
    first = probe_store.run_history(probe_job.run_id).candidates[0]
    probe_store.close()

    crashing = FlakyEvaluator(
        crash_on_call=3,
        inner=FakeEvaluator(gpu_seconds=2.0, invalid_contexts=frozenset({first.genome.cache_context_hash})),
    )
    with pytest.raises(RuntimeError):
        runner(make_direct_context(store, job, crashing))
    assert attempts(store, job.run_id)[0][1] == "invalid"

    events: list[TCCEvent] = []
    healthy = FlakyEvaluator(
        inner=FakeEvaluator(gpu_seconds=2.0, invalid_contexts=frozenset({first.genome.cache_context_hash}))
    )
    resume = resume_from_store(store, job.run_id)
    assert runner(make_direct_context(store, job, healthy, events=events, resume=resume)) == "completed"
    adopted = {e.candidate_id: e.payload["status"] for e in events if e.kind == "candidate_adopted"}
    assert adopted == {f"{job.run_id}.g000.c0000": "invalid", f"{job.run_id}.g000.c0001": "valid"}
    first_attempts = [a for a in attempts(store, job.run_id) if a[0].startswith(f"{job.run_id}.g000.c0000.")]
    assert first_attempts == [(f"{job.run_id}.g000.c0000.a1", "invalid")]  # never re-evaluated, never valid
    store.close()


def test_resume_without_resolution_info_adopts_nothing(tmp_path: Path) -> None:
    """A resume plan that cannot tell resolved candidates apart keeps the old, safe behaviour."""
    store, job = new_run(tmp_path, name="no-info")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    events: list[TCCEvent] = []
    stop_after = 2
    runner(
        make_direct_context(
            store,
            job,
            FakeEvaluator(),
            events=events,
            should_stop=lambda: sum(e.kind == "candidate_evaluated" for e in events) >= stop_after,
        )
    )
    resume = DirectResume(checkpoint=store.load_checkpoint(job.run_id), attempts_per_candidate={})
    later: list[TCCEvent] = []
    assert runner(make_direct_context(store, job, FakeEvaluator(), events=later, resume=resume)) == "completed"
    assert not [e for e in later if e.kind == "candidate_adopted"]
    assert len(attempts(store, job.run_id)) == TOTAL
    store.close()


class CrashingStore:
    """Delegates to the real store and dies at the n-th candidate registration."""

    def __init__(self, store: SQLiteRunStore, crash_on_registration: int) -> None:
        self._store = store
        self._left = crash_on_registration

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)

    def register_candidate(self, *args: object, **kwargs: object) -> None:
        self._left -= 1
        if self._left == 0:
            raise RuntimeError("synthetic crash while registering")
        self._store.register_candidate(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("crash_on_registration", [2, 6, 7, 11])
def test_crash_while_registering_candidates_is_replayed_identically(tmp_path: Path, crash_on_registration: int) -> None:
    """Candidates registered after the last checkpoint are proposed again, byte-identical."""
    store, job = new_run(tmp_path, name="reg-crash")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    ctx = make_direct_context(store, job, FakeEvaluator())
    broken = type(ctx)(
        **{
            **{f: getattr(ctx, f) for f in ctx.__dataclass_fields__},
            "store": CrashingStore(store, crash_on_registration),
        }
    )
    with pytest.raises(RuntimeError, match="while registering"):
        runner(broken)
    registered = len(store.run_history(job.run_id).candidates)
    assert registered == crash_on_registration - 1

    resume = resume_from_store(store, job.run_id)
    assert runner(make_direct_context(store, job, FakeEvaluator(), resume=resume)) == "completed"
    history = store.run_history(job.run_id)
    assert len(history.candidates) == TOTAL
    assert Counter(a.candidate_id for a in history.attempts) == Counter(c.candidate_id for c in history.candidates)
    assert set(store_population(store, job.run_id)) == reference_population(tmp_path)
    store.close()


# --- stop exactly at a generation boundary (cross-review finding C1) -----------------------------------


def survival_records(store: SQLiteRunStore, run_id: str) -> tuple[list[tuple[int, str]], int]:
    import json as _json

    rows = store._conn.execute(
        "SELECT generation, payload_json FROM checkpoints WHERE run_id = ? ORDER BY checkpoint_id", (run_id,)
    ).fetchall()
    phases = [(r[0], _json.loads(r[1])["rng_states"]["nsga2"]["phase"]) for r in rows]
    fronts = store._conn.execute("SELECT COUNT(*) FROM front_snapshots WHERE run_id = ?", (run_id,)).fetchone()[0]
    return [p for p in phases if p[1] == "survived"], fronts


@pytest.mark.parametrize("boundary", [4, 8])
def test_stop_at_a_generation_boundary_stores_each_survival_once(tmp_path: Path, boundary: int) -> None:
    store, job = new_run(tmp_path, name="boundary")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    events: list[TCCEvent] = []
    status = runner(
        make_direct_context(
            store,
            job,
            FakeEvaluator(),
            events=events,
            should_stop=lambda: sum(e.kind == "candidate_evaluated" for e in events) >= boundary,
        )
    )
    assert status == "stopped"
    assert len(attempts(store, job.run_id)) == boundary  # stopped right after the survival of that generation

    resume = resume_from_store(store, job.run_id)
    assert runner(make_direct_context(store, job, FakeEvaluator(), resume=resume)) == "completed"
    survived, fronts = survival_records(store, job.run_id)
    assert [generation for generation, _ in survived] == [0, 1, 2]
    assert fronts == 3
    assert len(attempts(store, job.run_id)) == TOTAL
    assert set(store_population(store, job.run_id)) == reference_population(tmp_path)
    store.close()


def test_run_resumed_after_completion_stores_nothing_again(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="done-twice")
    runner = SearchRunner(CONFIG, FP32_SLICE, context_for)
    assert runner(make_direct_context(store, job, FakeEvaluator())) == "completed"
    before = survival_records(store, job.run_id)
    evaluator = FakeEvaluator()
    resume = resume_from_store(store, job.run_id)
    assert runner(make_direct_context(store, job, evaluator, resume=resume)) == "completed"
    assert survival_records(store, job.run_id) == before and evaluator.calls == []
    store.close()


# --- empty invalid reason (cross-review finding C4) ------------------------------------------------------


def test_empty_invalid_reason_still_gets_a_failure_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from cnnlab.tcc.science import search as search_module

    real = search_module.prepare_in_slice
    first: list[str] = []

    def broken(genome: object, space: object, context: object) -> object:
        prepared = real(genome, space, context)  # type: ignore[arg-type]
        if not first:
            first.append(prepared.record.phenotype_hash)
        if prepared.record.phenotype_hash == first[0]:
            return dataclasses.replace(prepared, expression=dataclasses.replace(prepared.expression, invalid_reason=""))
        return prepared

    monkeypatch.setattr(search_module, "prepare_in_slice", broken)
    store, job = new_run(tmp_path, name="empty-reason")
    evaluator = FakeEvaluator()
    assert SearchRunner(CONFIG, FP32_SLICE, context_for)(make_direct_context(store, job, evaluator)) == "completed"
    invalid = [a for a in store.list_attempts(job.run_id) if a.status == "invalid"]
    assert len(invalid) == 1 and invalid[0].result is not None
    assert (invalid[0].result.failure_stage, invalid[0].result.failure_code) == ("repair", "invalid_expression")
    assert invalid[0].attempt_id not in evaluator.calls
    store.close()
