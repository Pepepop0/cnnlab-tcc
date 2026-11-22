"""Cross-front contract smoke: a real genome record with a fake evaluator."""

from __future__ import annotations

from pathlib import Path

from cnnlab.tcc.science import SPACE_V1, CacheContext, GeneRng, prepare_genome, sample_genome
from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import (
    FakeEvaluator,
    cache_hit_result,
    make_job,
    run_candidate,
    seed_reference_rows,
)


def test_real_genome_record_roundtrips_through_sqlite_with_fake_measurement(tmp_path: Path) -> None:
    """The two wave-one branches agree on identity, cache and fitness boundaries."""
    space_hash = SPACE_V1.space_hash()
    context = CacheContext(
        split_fingerprint="synthetic-split-fingerprint",
        train_seed=2000,
        calibration_seed=3000,
        train_policy={"epochs": 5, "synthetic": True},
        pruning_policy={"synthetic": True},
        quantization_policy={"synthetic": True},
        code_version="integration-test",
        backend={"name": "fake-evaluator"},
        builder_config={"synthetic": True},
        space_hash=space_hash,
    )
    prepared = prepare_genome(sample_genome(SPACE_V1, GeneRng(17)), SPACE_V1, context)
    assert prepared.expression.invalid_reason is None

    db_path = tmp_path / "db" / "tcc.sqlite3"
    with SQLiteRunStore(db_path, tmp_path / "artifacts") as store:
        seed_reference_rows(store)
        job = make_job("integration-joint-0", space_hash=space_hash)
        store.create_run(job)
        source = run_candidate(store, job, "candidate-1", "attempt-1", prepared.record, FakeEvaluator())
        assert source.status == "valid" and source.fitness is not None
        assert store.find_cached(job.run_id, prepared.record.cache_context_hash) == source

        store.register_candidate(job.run_id, "candidate-2", prepared.record, {"generation": 0})
        hit = cache_hit_result(source, candidate_id="candidate-2", attempt_id="attempt-2")
        store.commit_evaluation(hit)
        assert store.unique_valid_evaluations(job.run_id) == 1

    with SQLiteRunStore(db_path, tmp_path / "artifacts") as reopened:
        assert reopened.find_cached(job.run_id, prepared.record.cache_context_hash) == source
        assert reopened.unique_valid_evaluations(job.run_id) == 1
