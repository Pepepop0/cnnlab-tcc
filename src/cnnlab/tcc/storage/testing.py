"""Deterministic, CPU-only fixtures for storage/operation tests and integration fakes.

Nothing here is a scientific result. Every fitness value is derived from a
hash, every engine is a few synthetic bytes, and every result carries
`quality["synthetic"] = True` so it can never be mistaken for a measurement.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.contracts import (
    EvaluationRequest,
    EvaluationResult,
    Fitness,
    GenomeRecord,
    Method,
    Stage,
    TCCJobSpec,
)
from cnnlab.tcc.storage.artifacts import attempt_artifact_dir, publish_bytes
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.sqlite_store import SplitDescriptor, SQLiteRunStore

PROTOCOL_ID = "protocol-synthetic"
PROTOCOL_DOC_SHA256 = sha256_hex("synthetic protocol document, not TCC-01")
DATASET_ID = "cifar100-synthetic-splits"
ENVIRONMENT_ID = "env-synthetic-cpu"
SPACE_HASH = "space-synthetic-v1"
PLAN_ID = "plan-synthetic-v1"
CONFIRMATORY_BUDGET_S = 19800.0


def synthetic_plan_content() -> dict[str, Any]:
    """Eight paired blocks with distinct synthetic seeds (format `tcc-plan/1`)."""
    return {
        "blocks": [
            {
                "seed_block": b,
                "search_seed": 1000 + b,
                "train_seed": 2000 + b,
                "calibration_seed": 3000 + b,
                "retrain_seed": 4000 + b,
            }
            for b in range(8)
        ],
        "methods": ["joint", "mono", "random", "sequential"],
        "budget_gpu_s": CONFIRMATORY_BUDGET_S,
        "cut_rule": "synthetic: finish the running attempt, exclude points after the cut",
        "interruption_policy": "synthetic: new attempt_id, keep time of the interrupted one",
        "tolerances": {"fp32": 1e-5},
        "benchmark_policy_version": "synthetic-0",
    }


def synthetic_splits() -> dict[str, SplitDescriptor]:
    sizes = {"train": 45_000, "val": 5_000, "test": 10_000, "calibration": 1_024}
    return {
        name: SplitDescriptor(sha256=sha256_hex(f"synthetic-{name}"), n_examples=n, ref=f"splits/{name}.npy")
        for name, n in sizes.items()
    }


def seed_reference_rows(store: SQLiteRunStore, *, plan_id: str = PLAN_ID, freeze: bool = True) -> str:
    """Register protocol, environment, dataset and a plan; returns the plan id."""
    store.register_protocol(PROTOCOL_ID, PROTOCOL_DOC_SHA256, "pilot")
    store.register_environment(ENVIRONMENT_ID, {"gpu": "synthetic", "note": "CPU test fixture"})
    store.register_dataset(DATASET_ID, {"name": "CIFAR-100", "synthetic": True}, synthetic_splits())
    store.create_plan(plan_id, PROTOCOL_ID, synthetic_plan_content(), author="test-suite")
    if freeze:
        store.freeze_plan(plan_id)
    return plan_id


def make_job(
    run_id: str,
    *,
    method: Method = "joint",
    stage: Stage = "confirmatory",
    seed_block: int = 0,
    plan_id: str = PLAN_ID,
    space_hash: str = SPACE_HASH,
) -> TCCJobSpec:
    """Job spec whose seeds match `synthetic_plan_content()` for the given block."""
    train_seed = (4000 if stage == "finalist" else 2000) + seed_block
    return TCCJobSpec(
        run_id=run_id,
        protocol_id=PROTOCOL_ID,
        plan_id=plan_id,
        dataset_id=DATASET_ID,
        environment_id=ENVIRONMENT_ID,
        space_hash=space_hash,
        method=method,
        stage=stage,
        seed_block=seed_block,
        search_seed=1000 + seed_block,
        train_seed=train_seed,
        calibration_seed=3000 + seed_block,
        budget_gpu_s=CONFIRMATORY_BUDGET_S,
    )


def make_genome(index: int, *, context: str | None = None, space_hash: str = SPACE_HASH) -> GenomeRecord:
    """Synthetic genome; `context` lets two candidates share a cache context."""
    expressed = {"n_blocks": 2 + index % 5, "stem": 16 * (1 + index % 3)}
    ctx = context if context is not None else f"ctx-{index}"
    return GenomeRecord(
        raw_genome={**expressed, "latent": [index, index + 1]},
        expressed_genome=expressed,
        repair_log=("synthetic: no repair",),
        phenotype_hash=sha256_hex(f"phenotype-{index}"),
        cache_context_hash=sha256_hex(ctx),
        space_hash=space_hash,
    )


def make_request(
    job: TCCJobSpec, candidate_id: str, attempt_id: str, genome: GenomeRecord, artifact_root: Path
) -> EvaluationRequest:
    return EvaluationRequest(
        run_id=job.run_id,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        stage=job.stage,
        genome=genome,
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        dataset_id=job.dataset_id,
        environment_id=job.environment_id,
        protocol_id=job.protocol_id,
        plan_id=job.plan_id,
        artifact_root=artifact_root,
    )


@dataclass
class FakeEvaluator:
    """Deterministic `Evaluator`: same context and seed give the same synthetic fitness."""

    gpu_seconds: float = 10.0
    invalid_contexts: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request.attempt_id)
        ctx = request.genome.cache_context_hash
        durations = {"train": self.gpu_seconds * 0.7, "build": self.gpu_seconds * 0.2, "bench": self.gpu_seconds * 0.1}
        if ctx in self.invalid_contexts:
            return EvaluationResult(
                run_id=request.run_id,
                candidate_id=request.candidate_id,
                attempt_id=request.attempt_id,
                status="invalid",
                fitness=None,
                gpu_seconds=self.gpu_seconds,
                durations_s=durations,
                quality={"synthetic": True},
                failure_stage="build",
                failure_code="synthetic_build_failure",
                failure_detail="FakeEvaluator marks this context invalid",
            )
        digest = sha256_hex(f"{ctx}:{request.train_seed}")
        fitness = Fitness(
            e=0.05 + 0.9 * int(digest[0:8], 16) / 0xFFFFFFFF,
            l_ms=0.1 + 5.0 * int(digest[8:16], 16) / 0xFFFFFFFF,
            v_mb=1.0 + 50.0 * int(digest[16:24], 16) / 0xFFFFFFFF,
            s_mb=0.1 + 20.0 * int(digest[24:32], 16) / 0xFFFFFFFF,
        )
        rel = f"{attempt_artifact_dir(request.run_id, request.attempt_id)}/engine.plan"
        published = publish_bytes(request.artifact_root, rel, f"synthetic-engine:{ctx}:{request.attempt_id}".encode())
        return EvaluationResult(
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            attempt_id=request.attempt_id,
            status="valid",
            fitness=fitness,
            gpu_seconds=self.gpu_seconds,
            durations_s=durations,
            artifact_paths={"engine": published.rel_path},
            artifact_sha256={"engine": published.sha256},
            quality={"synthetic": True},
        )


def run_candidate(
    store: SQLiteRunStore,
    job: TCCJobSpec,
    candidate_id: str,
    attempt_id: str,
    genome: GenomeRecord,
    evaluator: FakeEvaluator,
    *,
    generation: int = 0,
) -> EvaluationResult:
    """Register (idempotently), open, evaluate and commit one attempt."""
    store.register_candidate(job.run_id, candidate_id, genome, {"generation": generation})
    store.begin_attempt(job.run_id, candidate_id, attempt_id)
    result = evaluator.evaluate(make_request(job, candidate_id, attempt_id, genome, store.artifact_root))
    store.commit_evaluation(result)
    return result


class ManualClock:
    """Wall clock for tests: shared by several stores to simulate one machine."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)


def cache_hit_result(source: EvaluationResult, *, candidate_id: str, attempt_id: str) -> EvaluationResult:
    """Result a runner commits when `find_cached` returned `source`."""
    return EvaluationResult(
        run_id=source.run_id,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        status="valid",
        fitness=source.fitness,
        gpu_seconds=0.0,
        quality={
            "cache_hit": True,
            "saved_gpu_seconds": source.gpu_seconds,
            "synthetic": bool(source.quality.get("synthetic", False)),
        },
        source_attempt_id=source.attempt_id,
    )


SCIENCE_CODE_SHA = "0123456789abcdef0123456789abcdef01234567"


def science_style_manifest() -> dict[str, Any]:
    """Environment manifest in the shape the science wiring records (no `system` section)."""
    return {
        "probe": {"platform": "synthetic", "python": "synthetic"},
        "science": {
            "code_version": SCIENCE_CODE_SHA,
            "wiring": "synthetic-wiring/v0",
            "evaluator": {"evaluator": "synthetic-evaluator/v0", "representation": "none"},
            "search": {"runner": "synthetic-runner/v0", "population": 2},
        },
    }


def make_v2_copy(source_db: Path, dest_db: Path) -> Path:
    """Copy a database and give the *copy* the shape of schema v2 (job events without the sentinel kinds).

    Stands for a database written by code older than migration 3, evidence that must never be migrated by a side
    effect. Every row of the job queue is kept. The source is only read; every change happens in `dest_db`.
    """
    from cnnlab.tcc.storage.schema import MIGRATIONS

    create_v2 = next(s for s in MIGRATIONS[1].statements if " ".join(s.split()).startswith("CREATE TABLE job_events"))
    dest_db = Path(dest_db)
    if dest_db.exists():
        raise FileExistsError(dest_db)
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{Path(source_db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(str(dest_db), isolation_level=None)
        try:
            source.backup(copy)
            if copy.execute("SELECT 1 FROM job_events WHERE kind LIKE 'sentinel_%'").fetchone():
                raise ValueError("the source has sentinel events: they cannot exist in a v2 database")
            copy.execute("ALTER TABLE job_events RENAME TO job_events_v3")
            copy.execute(create_v2)
            copy.execute("INSERT INTO job_events SELECT * FROM job_events_v3 ORDER BY event_id")
            copy.execute("DROP TABLE job_events_v3")
            copy.execute("CREATE INDEX job_events_by_job ON job_events(job_id, event_id)")
            for suffix, verb in (("no_update", "UPDATE"), ("no_delete", "DELETE")):
                copy.execute(
                    f"CREATE TRIGGER job_events_{suffix} BEFORE {verb} ON job_events "
                    "BEGIN SELECT RAISE(ABORT, 'job_events is append-only'); END"
                )
            copy.execute("DELETE FROM schema_migrations WHERE version >= 3")
            copy.execute("PRAGMA user_version = 2")
            copy.execute("PRAGMA journal_mode = WAL")
        finally:
            copy.close()
    finally:
        source.close()
    return dest_db


def make_v1_copy(source_db: Path, dest_db: Path) -> Path:
    """Copy a database and give the *copy* the shape of schema v1 (no job queue).

    Stands for a run written by code older than migration 2. The source is
    only read; every change happens in `dest_db`, which must not exist.
    """
    dest_db = Path(dest_db)
    if dest_db.exists():
        raise FileExistsError(dest_db)
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{Path(source_db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(str(dest_db), isolation_level=None)
        try:
            source.backup(copy)
            for table in ("job_gpu_segments", "job_attempts", "job_events", "jobs"):
                copy.execute(f"DROP TABLE {table}")
            copy.execute("DELETE FROM schema_migrations WHERE version >= 2")
            copy.execute("PRAGMA user_version = 1")
            copy.execute("PRAGMA journal_mode = WAL")
        finally:
            copy.close()
    finally:
        source.close()
    return dest_db
