from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import SearchCheckpoint
from cnnlab.tcc.operation import prepare_resume, retry_attempt_id
from cnnlab.tcc.storage import CheckpointMismatchError, NotFoundError, SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, run_candidate

JOB = make_job("run-resume", method="random", seed_block=2)


def _generation_zero(store: SQLiteRunStore) -> SearchCheckpoint:
    store.create_run(JOB)
    bad_context = make_genome(2).cache_context_hash
    evaluator = FakeEvaluator(gpu_seconds=10.0, invalid_contexts=frozenset({bad_context}))
    for i in range(3):
        run_candidate(store, JOB, f"c{i}", f"c{i}.a1", make_genome(i), evaluator)
    store.register_candidate(JOB.run_id, "c3", make_genome(3), {"generation": 0})
    store.register_candidate(JOB.run_id, "c4", make_genome(4), {"generation": 0})
    hashes = store.run_hashes(JOB.run_id)
    checkpoint = SearchCheckpoint(
        run_id=JOB.run_id,
        generation=0,
        population=("c0", "c1", "c2", "c3", "c4"),
        pareto_archive=("c0", "c1"),
        pending_candidates=("c3", "c4"),
        rng_states={"search": [3, [1, 2, 3], None]},
        unique_evaluations=2,
        active_gpu_seconds=store.active_gpu_seconds(JOB.run_id),
        protocol_hash=hashes.protocol_hash,
        plan_hash=hashes.plan_hash,
        space_hash=hashes.space_hash,
    )
    store.save_checkpoint(checkpoint)
    store.record_front(JOB.run_id, ("c0", "c1"), checkpoint.active_gpu_seconds, None)
    return checkpoint


def test_resume_after_crash_retries_only_unconfirmed_work(
    seeded: SQLiteRunStore, db_path: Path, artifact_root: Path
) -> None:
    checkpoint = _generation_zero(seeded)
    seeded.begin_attempt(JOB.run_id, "c3", "c3.a1")
    seeded.record_gpu_segment(JOB.run_id, "c3.a1", 2.5)
    seeded.close()  # crash: c3 was mid-evaluation, c4 was proposed but never started

    with SQLiteRunStore(db_path, artifact_root) as store:
        plan = prepare_resume(store, JOB)
        assert plan.checkpoint == checkpoint
        assert plan.interrupted_attempt_ids == ("c3.a1",)
        assert plan.retry_candidate_ids == ("c3", "c4")  # valid c0/c1 and invalid c2 are final
        assert plan.active_gpu_seconds == pytest.approx(32.5)  # interrupted time is kept
        assert plan.unique_valid_evaluations == 2
        assert dict(plan.attempts_per_candidate) == {"c0": 1, "c1": 1, "c2": 1, "c3": 1}

        evaluator = FakeEvaluator(gpu_seconds=10.0)
        for cid in plan.retry_candidate_ids:
            attempt_id = retry_attempt_id(cid, plan.attempts_per_candidate.get(cid, 0))
            run_candidate(store, JOB, cid, attempt_id, make_genome(int(cid[1:])), evaluator)
        assert [a.attempt_id for a in store.list_attempts(JOB.run_id) if a.candidate_id in ("c3", "c4")] == [
            "c3.a1",
            "c3.a2",
            "c4.a1",
        ]

        again = prepare_resume(store, JOB)
        assert again.interrupted_attempt_ids == () and again.retry_candidate_ids == ()
        assert again.unique_valid_evaluations == 4  # no confirmed point was duplicated
        assert again.active_gpu_seconds == pytest.approx(52.5)
        assert store.check_integrity().ok


def test_resume_refuses_a_different_job_spec(seeded: SQLiteRunStore) -> None:
    _generation_zero(seeded)
    with pytest.raises(CheckpointMismatchError):
        prepare_resume(seeded, dataclasses.replace(JOB, space_hash="space-v2"))
    with pytest.raises(NotFoundError):
        prepare_resume(seeded, dataclasses.replace(JOB, run_id="never-created"))


def test_retry_attempt_ids_are_fresh() -> None:
    assert retry_attempt_id("c7", 0) == "c7.a1"
    assert retry_attempt_id("c7", 2) == "c7.a3"
