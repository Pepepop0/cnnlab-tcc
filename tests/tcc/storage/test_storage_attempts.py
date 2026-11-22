from __future__ import annotations

import dataclasses
import math
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import EvaluationResult, Fitness
from cnnlab.tcc.storage import (
    ArtifactIntegrityError,
    AttemptStateError,
    ConsistencyError,
    IdempotencyConflictError,
    NotFoundError,
    SerializationError,
    SQLiteRunStore,
)
from cnnlab.tcc.storage.testing import (
    FakeEvaluator,
    cache_hit_result,
    make_genome,
    make_job,
    make_request,
    run_candidate,
)

JOB = make_job("run-joint-b0")


@pytest.fixture
def run(seeded: SQLiteRunStore) -> SQLiteRunStore:
    seeded.create_run(JOB)
    return seeded


def _count(store: SQLiteRunStore, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _invalid(attempt_id: str = "c1.a1", candidate_id: str = "c1", **kw: object) -> EvaluationResult:
    fields = dict(
        run_id=JOB.run_id,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        status="invalid",
        fitness=None,
        gpu_seconds=3.5,
        failure_stage="build",
        failure_code="trt_build_failed",
        failure_detail="synthetic",
    )
    fields.update(kw)
    return EvaluationResult(**fields)  # type: ignore[arg-type]


def test_candidate_insert_and_restore_across_reopen(run: SQLiteRunStore, db_path: Path, artifact_root: Path) -> None:
    genome = make_genome(1)
    run.register_candidate(
        JOB.run_id, "c1", genome, {"generation": 3, "parents": ["p1", "p2"], "operator": "crossover"}
    )
    run.register_candidate(
        JOB.run_id, "c1", genome, {"generation": 3, "parents": ["p1", "p2"], "operator": "crossover"}
    )
    result = FakeEvaluator().evaluate(make_request(JOB, "c1", "c1.a1", genome, artifact_root))
    run.commit_evaluation(result)
    run.close()

    with SQLiteRunStore(db_path, artifact_root) as reopened:
        assert reopened.get_run(JOB.run_id) == JOB
        (cand,) = reopened.list_candidates(JOB.run_id)
        assert cand.genome == genome
        assert cand.generation == 3 and cand.metadata["parents"] == ["p1", "p2"] and cand.seq == 1
        attempt = reopened.get_attempt("c1.a1")
        assert attempt is not None and attempt.status == "valid" and attempt.result == result


def test_candidate_conflicts(run: SQLiteRunStore) -> None:
    run.register_candidate(JOB.run_id, "c1", make_genome(1), {})
    with pytest.raises(IdempotencyConflictError):
        run.register_candidate(JOB.run_id, "c1", make_genome(2), {})
    with pytest.raises(ConsistencyError, match="space_hash"):
        run.register_candidate(JOB.run_id, "c2", make_genome(2, space_hash="other-space"), {})
    with pytest.raises(NotFoundError):
        run.register_candidate("no-such-run", "c3", make_genome(3), {})
    with pytest.raises(ValueError):
        run.register_candidate(JOB.run_id, "c4", make_genome(4), {"generation": -1})


def test_invalid_attempt_has_no_fitness_but_keeps_time(run: SQLiteRunStore) -> None:
    run.register_candidate(JOB.run_id, "c1", make_genome(1), {})
    run.commit_evaluation(_invalid())
    attempt = run.get_attempt("c1.a1")
    assert attempt is not None and attempt.status == "invalid"
    assert attempt.result is not None and attempt.result.fitness is None
    assert _count(run, "measurements") == 0
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(3.5)
    assert run.unique_valid_evaluations(JOB.run_id) == 0
    assert run.find_cached(JOB.run_id, make_genome(1).cache_context_hash) is None
    with pytest.raises(sqlite3.IntegrityError, match="valid, non-cached"):
        run._conn.execute("INSERT INTO measurements VALUES ('m', 'c1.a1', 0.5, 1.0, 1.0, 1.0, '2026-09-27T00:00:00Z')")


def test_invalid_attempt_needs_reason(run: SQLiteRunStore) -> None:
    run.register_candidate(JOB.run_id, "c1", make_genome(1), {})
    with pytest.raises(ValueError, match="failure_stage"):
        run.commit_evaluation(_invalid(failure_code=None))
    with pytest.raises(ValueError, match="valid attempts"):
        run.commit_evaluation(
            EvaluationResult(JOB.run_id, "c1", "c1.a1", "valid", Fitness(0.3, 1, 1, 1), 1.0, failure_code="x")
        )
    assert run.get_attempt("c1.a1") is None


def test_valid_measurement_and_artifacts(run: SQLiteRunStore, artifact_root: Path) -> None:
    result = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator(gpu_seconds=12.0))
    row = run._conn.execute("SELECT e_fraction, l_ms, v_mb, s_mb FROM measurements").fetchone()
    assert tuple(row) == (result.fitness.e, result.fitness.l_ms, result.fitness.v_mb, result.fitness.s_mb)
    art = run._conn.execute("SELECT kind, rel_path, sha256, size_bytes FROM artifacts").fetchone()
    assert art["rel_path"] == result.artifact_paths["engine"] and not Path(art["rel_path"]).is_absolute()
    assert art["sha256"] == result.artifact_sha256["engine"]
    assert art["size_bytes"] == (artifact_root / art["rel_path"]).stat().st_size
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(12.0)
    assert run.unique_valid_evaluations(JOB.run_id) == 1


def test_duplicate_commit_is_idempotent(run: SQLiteRunStore) -> None:
    result = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    run.commit_evaluation(result)
    run.commit_evaluation(
        dataclasses.replace(
            result, fitness=Fitness(result.fitness.e, result.fitness.l_ms, result.fitness.v_mb, result.fitness.s_mb)
        )
    )
    assert _count(run, "evaluation_attempts") == 1
    assert _count(run, "measurements") == 1
    assert _count(run, "artifacts") == 1
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0)


def test_divergent_duplicate_is_rejected(run: SQLiteRunStore) -> None:
    result = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    before = run.get_attempt("c1.a1")
    for changed in (
        dataclasses.replace(result, gpu_seconds=11.0),
        dataclasses.replace(result, fitness=dataclasses.replace(result.fitness, e=0.5)),
        dataclasses.replace(result, quality={"synthetic": False}),
        _invalid(),
    ):
        with pytest.raises(IdempotencyConflictError):
            run.commit_evaluation(changed)
    assert run.get_attempt("c1.a1") == before
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0)


def test_terminal_attempt_is_immutable_in_raw_sql(run: SQLiteRunStore) -> None:
    run_candidate(run, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    with pytest.raises(sqlite3.IntegrityError, match="terminal attempt is immutable"):
        run._conn.execute("UPDATE evaluation_attempts SET status = 'invalid' WHERE attempt_id = 'c1.a1'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        run._conn.execute("DELETE FROM evaluation_attempts")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        run._conn.execute("UPDATE measurements SET e_fraction = 0.0")


def test_gpu_ledger_segments_and_reconciliation(run: SQLiteRunStore, artifact_root: Path) -> None:
    genome = make_genome(1)
    run.register_candidate(JOB.run_id, "c1", genome, {})
    assert run.begin_attempt(JOB.run_id, "c1", "c1.a1") == 1
    assert run.begin_attempt(JOB.run_id, "c1", "c1.a1") == 1  # idempotent while running
    with pytest.raises(AttemptStateError, match="running attempt"):
        run.begin_attempt(JOB.run_id, "c1", "c1.other")
    run.record_gpu_segment(JOB.run_id, "c1.a1", 4.0)
    run.record_gpu_segment(JOB.run_id, "c1.a1", 3.0)
    with pytest.raises(ValueError):
        run.record_gpu_segment(JOB.run_id, "c1.a1", math.inf)
    with pytest.raises(NotFoundError):
        run.record_gpu_segment(JOB.run_id, "never-begun", 1.0)

    result = FakeEvaluator(gpu_seconds=10.0).evaluate(make_request(JOB, "c1", "c1.a1", genome, artifact_root))
    with pytest.raises(ConsistencyError, match="ledger"):
        run.commit_evaluation(dataclasses.replace(result, gpu_seconds=5.0))
    run.commit_evaluation(result)
    sources = run._conn.execute("SELECT source, gpu_seconds FROM gpu_segments ORDER BY segment_id").fetchall()
    assert [tuple(r) for r in sources] == [("runner", 4.0), ("runner", 3.0), ("commit_reconcile", 3.0)]
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0)
    with pytest.raises(AttemptStateError):
        run.record_gpu_segment(JOB.run_id, "c1.a1", 1.0)
    with pytest.raises(AttemptStateError):
        run.begin_attempt(JOB.run_id, "c1", "c1.a1")
    assert run.begin_attempt(JOB.run_id, "c1", "c1.a2") == 2  # remeasurement is a new attempt
    assert run.check_integrity().ledger_mismatches == ()


def test_cache_hit_within_same_run(run: SQLiteRunStore) -> None:
    source = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1, context="shared"), FakeEvaluator())
    run.register_candidate(JOB.run_id, "c2", make_genome(2, context="shared"), {"generation": 1})
    cached = run.find_cached(JOB.run_id, make_genome(2, context="shared").cache_context_hash)
    assert cached == source
    run.commit_evaluation(cache_hit_result(cached, candidate_id="c2", attempt_id="c2.a1"))
    hit = run.get_attempt("c2.a1")
    assert hit is not None and hit.status == "valid" and hit.result.source_attempt_id == "c1.a1"
    assert _count(run, "measurements") == 1  # the hit reuses the source measurement
    assert run.unique_valid_evaluations(JOB.run_id) == 1
    assert run.active_gpu_seconds(JOB.run_id) == pytest.approx(10.0)
    assert run.find_cached(JOB.run_id, make_genome(2, context="shared").cache_context_hash) == source


def test_cache_is_restricted_to_the_same_run(run: SQLiteRunStore) -> None:
    other = make_job("run-joint-b1", seed_block=1)
    run.create_run(other)
    source = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1, context="same-arch"), FakeEvaluator())
    run.register_candidate(other.run_id, "d1", make_genome(1, context="same-arch"), {})
    assert run.find_cached(other.run_id, make_genome(1, context="same-arch").cache_context_hash) is None
    cross = dataclasses.replace(cache_hit_result(source, candidate_id="d1", attempt_id="d1.a1"), run_id=other.run_id)
    with pytest.raises(ConsistencyError, match="same run"):
        run.commit_evaluation(cross)
    assert run.get_attempt("d1.a1") is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: dataclasses.replace(r, gpu_seconds=1.0), "no GPU"),
        (lambda r: dataclasses.replace(r, fitness=dataclasses.replace(r.fitness, e=0.99)), "fitness"),
    ],
)
def test_cache_hit_must_mirror_source(run: SQLiteRunStore, mutate, message: str) -> None:
    source = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1, context="k"), FakeEvaluator())
    run.register_candidate(JOB.run_id, "c2", make_genome(2, context="k"), {})
    with pytest.raises(ConsistencyError, match=message):
        run.commit_evaluation(mutate(cache_hit_result(source, candidate_id="c2", attempt_id="c2.a1")))


def test_cache_hit_needs_same_context_and_valid_source(run: SQLiteRunStore) -> None:
    source = run_candidate(run, JOB, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    run.register_candidate(JOB.run_id, "c2", make_genome(2), {})
    with pytest.raises(ConsistencyError, match="identical cache_context_hash"):
        run.commit_evaluation(cache_hit_result(source, candidate_id="c2", attempt_id="c2.a1"))
    run.register_candidate(JOB.run_id, "c3", make_genome(3, context="bad"), {})
    run.commit_evaluation(_invalid("c3.a1", "c3"))
    run.register_candidate(JOB.run_id, "c4", make_genome(4, context="bad"), {})
    fake_source = dataclasses.replace(source, attempt_id="c3.a1")
    with pytest.raises(ConsistencyError, match="valid, original"):
        run.commit_evaluation(cache_hit_result(fake_source, candidate_id="c4", attempt_id="c4.a1"))


def test_artifact_problems_abort_commit(run: SQLiteRunStore, artifact_root: Path) -> None:
    genome = make_genome(1)
    run.register_candidate(JOB.run_id, "c1", genome, {})
    good = FakeEvaluator().evaluate(make_request(JOB, "c1", "c1.a1", genome, artifact_root))
    with pytest.raises(ArtifactIntegrityError, match="relative"):
        run.commit_evaluation(dataclasses.replace(good, artifact_paths={"engine": str(artifact_root / "x.plan")}))
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        run.commit_evaluation(dataclasses.replace(good, artifact_paths={"engine": "runs/nope/engine.plan"}))
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        run.commit_evaluation(dataclasses.replace(good, artifact_sha256={"engine": "0" * 64}))
    with pytest.raises(ArtifactIntegrityError, match="same kinds"):
        run.commit_evaluation(dataclasses.replace(good, artifact_sha256={}))
    assert run.get_attempt("c1.a1") is None and _count(run, "artifacts") == 0
    run.commit_evaluation(good)


def test_unregistered_candidate_and_bad_payload(run: SQLiteRunStore) -> None:
    with pytest.raises(NotFoundError):
        run.commit_evaluation(_invalid())
    run.register_candidate(JOB.run_id, "c1", make_genome(1), {})
    with pytest.raises(SerializationError):
        run.commit_evaluation(_invalid(quality={"curve": [0.1, math.nan]}))
    with pytest.raises(ValueError):
        run.commit_evaluation(_invalid(durations_s={"train": -1.0}))
    assert run.get_attempt("c1.a1") is None


def test_find_cached_ignores_invalid_and_interrupted(run: SQLiteRunStore) -> None:
    run.register_candidate(JOB.run_id, "c1", make_genome(1, context="x"), {})
    run.commit_evaluation(_invalid())
    run.commit_evaluation(dataclasses.replace(_invalid("c1.a2"), status="interrupted"))
    assert run.find_cached(JOB.run_id, make_genome(1, context="x").cache_context_hash) is None
    assert [a.attempt_number for a in run.list_attempts(JOB.run_id)] == [1, 2]
