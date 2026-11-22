from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from cnnlab.tcc.contracts import EvaluationResult
from cnnlab.tcc.storage import (
    ConsistencyError,
    FinalistSelection,
    IdempotencyConflictError,
    NotFoundError,
    OfficialTestAccessError,
    SQLiteRunStore,
)
from cnnlab.tcc.storage.testing import (
    PLAN_ID,
    PROTOCOL_ID,
    FakeEvaluator,
    make_genome,
    make_job,
    run_candidate,
)

TS = "2026-09-27T00:00:00Z"
PILOT = make_job("pilot-joint-b0", stage="pilot")
JOINT = make_job("conf-joint-b0", method="joint")
MONO = make_job("conf-mono-b0", method="mono")
FINAL = make_job("final-joint-b0", stage="finalist")


@pytest.fixture
def searched(seeded: SQLiteRunStore) -> SQLiteRunStore:
    ev = FakeEvaluator()
    for job in (PILOT, JOINT, MONO):
        seeded.create_run(job)
    run_candidate(seeded, PILOT, "p1", "p1.a1", make_genome(1), ev)
    run_candidate(seeded, JOINT, "j1", "j1.a1", make_genome(2), ev)
    run_candidate(seeded, JOINT, "j2", "j2.a1", make_genome(3), ev)
    run_candidate(seeded, MONO, "m1", "m1.a1", make_genome(4), ev)
    seeded.register_candidate(JOINT.run_id, "j3", make_genome(5), {})
    seeded.commit_evaluation(
        EvaluationResult(JOINT.run_id, "j3", "j3.a1", "invalid", None, 1.0, failure_stage="build", failure_code="x")
    )
    return seeded


SELECTIONS = (
    FinalistSelection("s-j-e", JOINT.run_id, "j1", "menor_E", "min E on validation"),
    FinalistSelection("s-j-h1", JOINT.run_id, "j2", "H1", "E <= E_mono + 0.01, min L/rL+V/rV+S/rS"),
    FinalistSelection("s-m-e", MONO.run_id, "m1", "menor_E", "mono champion"),
)


def _freeze(store: SQLiteRunStore, selections=SELECTIONS, list_id: str = "final-list") -> str:
    return store.freeze_finalist_list(
        list_id, protocol_id=PROTOCOL_ID, plan_id=PLAN_ID, selections=selections, author="test-suite"
    )


def test_search_and_pilot_runs_never_see_the_test_split(searched: SQLiteRunStore) -> None:
    for job in (PILOT, JOINT, MONO):
        assert set(searched.dataset_splits_for_run(job.run_id)) == {"train", "val", "calibration"}
        with pytest.raises(OfficialTestAccessError):
            searched.official_test_split(job.run_id)


def test_finalist_run_requires_a_frozen_list(searched: SQLiteRunStore) -> None:
    with pytest.raises(OfficialTestAccessError):
        searched.create_run(FINAL)
    with pytest.raises(NotFoundError):
        searched.create_run(FINAL, finalist_list_id="not-frozen-yet")
    with pytest.raises(ConsistencyError):
        searched.create_run(make_job("conf-random-b0", method="random"), finalist_list_id="x")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        searched._conn.execute(
            "INSERT INTO runs SELECT 'raw-final', protocol_id, protocol_sha256, plan_id, plan_sha256, dataset_id, "
            "environment_id, space_hash, method, 'finalist', 7, search_seed, train_seed, calibration_seed, "
            "budget_gpu_s, NULL, job_spec_json, job_spec_sha256, created_at FROM runs WHERE run_id = ?",
            (JOINT.run_id,),
        )


def test_finalist_selection_rules(searched: SQLiteRunStore) -> None:
    with pytest.raises(ConsistencyError, match="confirmatory"):
        _freeze(searched, (FinalistSelection("s-p", PILOT.run_id, "p1", "menor_E", "pilot"),))
    with pytest.raises(ConsistencyError, match="H1"):
        _freeze(searched, (FinalistSelection("s-m-h1", MONO.run_id, "m1", "H1", "wrong run"),))
    with pytest.raises(ConsistencyError):  # candidate without a valid attempt
        _freeze(searched, (FinalistSelection("s-j3", JOINT.run_id, "j3", "equilibrio", "invalid one"),))
    assert searched._conn.execute("SELECT COUNT(*) FROM finalist_lists").fetchone()[0] == 0

    digest = _freeze(searched)
    assert _freeze(searched, tuple(reversed(SELECTIONS))) == digest  # idempotent, order-free
    with pytest.raises(IdempotencyConflictError):
        _freeze(searched, SELECTIONS[:2])
    with pytest.raises(sqlite3.IntegrityError, match="invalid finalist selection"):
        searched._conn.execute(
            "INSERT INTO finalist_selections VALUES ('late', 'final-list', ?, 'j2', 'equilibrio', 'added later')",
            (JOINT.run_id,),
        )


def test_official_test_opens_once_per_engine_in_finalist_run(searched: SQLiteRunStore) -> None:
    _freeze(searched)
    searched.create_run(FINAL, finalist_list_id="final-list")
    assert searched.official_test_split(FINAL.run_id).n_examples == 10_000
    retrained = run_candidate(searched, FINAL, "f1", "f1.a1", make_genome(2), FakeEvaluator(gpu_seconds=30.0))
    engine = retrained.artifact_sha256["engine"]
    metrics = {"top1": 0.61, "top5": 0.86, "synthetic": True}

    with pytest.raises(OfficialTestAccessError):
        searched.record_test_evaluation(
            "t0", run_id=JOINT.run_id, selection_id="s-j-e", attempt_id="j1.a1", engine_sha256=engine, metrics=metrics
        )
    searched.record_test_evaluation(
        "t1", run_id=FINAL.run_id, selection_id="s-j-e", attempt_id="f1.a1", engine_sha256=engine, metrics=metrics
    )
    searched.record_test_evaluation(
        "t1", run_id=FINAL.run_id, selection_id="s-j-e", attempt_id="f1.a1", engine_sha256=engine, metrics=metrics
    )  # idempotent
    with pytest.raises(OfficialTestAccessError, match="already opened"):
        searched.record_test_evaluation(
            "t2",
            run_id=FINAL.run_id,
            selection_id="s-j-e",
            attempt_id="f1.a1",
            engine_sha256=engine,
            metrics={"top1": 0.7},
        )
    with pytest.raises(sqlite3.IntegrityError, match="official test only"):
        searched._conn.execute(
            "INSERT INTO test_evaluations VALUES ('raw', ?, 's-j-e', 'j1.a1', ?, '{}', ?)",
            (JOINT.run_id, "a" * 64, TS),
        )
    assert searched._conn.execute("SELECT COUNT(*) FROM test_evaluations").fetchone()[0] == 1


def test_finalist_list_must_match_run_plan(searched: SQLiteRunStore) -> None:
    _freeze(searched)
    other_plan = dataclasses.replace(FINAL, plan_id="plan-other")
    searched.create_plan("plan-other", PROTOCOL_ID, {"methods": ["joint"]}, author="t")
    searched.freeze_plan("plan-other")
    with pytest.raises(ConsistencyError, match="another plan"):
        searched.create_run(other_plan, finalist_list_id="final-list")
