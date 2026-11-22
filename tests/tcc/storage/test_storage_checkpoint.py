from __future__ import annotations

import dataclasses
import random
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import SearchCheckpoint
from cnnlab.tcc.storage import (
    CheckpointMismatchError,
    ConsistencyError,
    NotFoundError,
    SQLiteRunStore,
)
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, run_candidate

JOB = make_job("run-ckpt")


@pytest.fixture
def run(seeded: SQLiteRunStore) -> SQLiteRunStore:
    seeded.create_run(JOB)
    evaluator = FakeEvaluator(gpu_seconds=5.0)
    for i in range(1, 5):
        run_candidate(seeded, JOB, f"c{i}", f"c{i}.a1", make_genome(i), evaluator)
    seeded.register_candidate(JOB.run_id, "c5", make_genome(5), {"generation": 1})
    return seeded


def _checkpoint(store: SQLiteRunStore, **kw: object) -> SearchCheckpoint:
    hashes = store.run_hashes(JOB.run_id)
    rng = random.Random(1234)
    rng.random()
    fields = dict(
        run_id=JOB.run_id,
        generation=1,
        population=("c1", "c2", "c3", "c4"),
        pareto_archive=("c2", "c4"),
        pending_candidates=("c5",),
        rng_states={"search": rng.getstate(), "numpy": {"bit_generator": "PCG64", "state": {"state": 2**100}}},
        unique_evaluations=4,
        active_gpu_seconds=20.0,
        protocol_hash=hashes.protocol_hash,
        plan_hash=hashes.plan_hash,
        space_hash=hashes.space_hash,
    )
    fields.update(kw)
    return SearchCheckpoint(**fields)  # type: ignore[arg-type]


def test_checkpoint_round_trip_restores_rng(run: SQLiteRunStore, db_path: Path, artifact_root: Path) -> None:
    assert run.load_checkpoint(JOB.run_id) is None
    original = _checkpoint(run)
    run.save_checkpoint(original)
    run.close()
    with SQLiteRunStore(db_path, artifact_root) as reopened:
        loaded = reopened.load_checkpoint(JOB.run_id)
    assert loaded is not None
    assert (loaded.population, loaded.pareto_archive, loaded.pending_candidates) == (
        original.population,
        original.pareto_archive,
        original.pending_candidates,
    )
    assert (loaded.generation, loaded.unique_evaluations, loaded.active_gpu_seconds) == (1, 4, 20.0)
    # JSON turns tuples into lists; the runner converts back before setstate.
    version, internal, gauss = loaded.rng_states["search"]
    restored = random.Random()
    restored.setstate((version, tuple(internal), gauss))
    expected = random.Random(1234)
    expected.random()
    assert restored.random() == expected.random()
    assert loaded.rng_states["numpy"]["state"]["state"] == 2**100


def test_latest_checkpoint_wins_and_history_is_kept(run: SQLiteRunStore) -> None:
    run.save_checkpoint(_checkpoint(run))
    run.save_checkpoint(_checkpoint(run, generation=2, population=("c4", "c3", "c2", "c1")))
    loaded = run.load_checkpoint(JOB.run_id)
    assert loaded is not None and loaded.generation == 2 and loaded.population[0] == "c4"
    assert run._conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        run._conn.execute("UPDATE checkpoints SET generation = 0")


def test_checkpoint_with_divergent_hash_is_refused(run: SQLiteRunStore) -> None:
    for field in ("protocol_hash", "plan_hash", "space_hash"):
        with pytest.raises(CheckpointMismatchError):
            run.save_checkpoint(_checkpoint(run, **{field: "0" * 64}))
    cp = _checkpoint(run)
    with pytest.raises(sqlite3.IntegrityError, match="differ from run"):
        run._conn.execute(
            "INSERT INTO checkpoints (run_id, generation, unique_evaluations, active_gpu_seconds, "
            "protocol_hash, plan_hash, space_hash, payload_json, payload_sha256, created_at) "
            "VALUES (?, 0, 0, 0, ?, 'bad', ?, '{}', ?, '2026-09-27T00:00:00Z')",
            (JOB.run_id, cp.protocol_hash, cp.space_hash, "0" * 64),
        )


def test_checkpoint_counters_are_monotonic_and_bounded_by_ledger(run: SQLiteRunStore) -> None:
    run.save_checkpoint(_checkpoint(run, generation=2))
    with pytest.raises(ConsistencyError, match="backwards"):
        run.save_checkpoint(_checkpoint(run, generation=1))
    with pytest.raises(ConsistencyError, match="backwards"):
        run.save_checkpoint(_checkpoint(run, generation=3, unique_evaluations=3))
    with pytest.raises(ConsistencyError, match="ledger"):
        run.save_checkpoint(_checkpoint(run, generation=3, active_gpu_seconds=20.5))
    with pytest.raises(ValueError):
        run.save_checkpoint(_checkpoint(run, generation=3, active_gpu_seconds=float("nan")))


def test_checkpoint_members_must_belong_to_the_run(run: SQLiteRunStore) -> None:
    other = make_job("run-other", seed_block=1)
    run.create_run(other)
    run.register_candidate(other.run_id, "x1", make_genome(9), {})
    with pytest.raises(NotFoundError):
        run.save_checkpoint(_checkpoint(run, pending_candidates=("x1",)))
    with pytest.raises(NotFoundError):
        run.save_checkpoint(_checkpoint(run, population=("ghost",)))
    assert run.load_checkpoint(JOB.run_id) is None


def test_unknown_run(run: SQLiteRunStore) -> None:
    with pytest.raises(NotFoundError):
        run.load_checkpoint("nope")
    with pytest.raises(NotFoundError):
        run.save_checkpoint(dataclasses.replace(_checkpoint(run), run_id="nope"))
