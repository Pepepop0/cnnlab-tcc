"""Shared, UI- and storage-independent boundaries for TCC v2."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

Stage = Literal["pilot", "confirmatory", "finalist"]
Method = Literal["joint", "mono", "random", "sequential"]
AttemptStatus = Literal["valid", "invalid", "interrupted"]
JobStatus = Literal["queued", "running", "interrupted", "done", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class Fitness:
    """Four measured objectives, all minimized in their published units."""

    e: float  # top-1 error fraction
    l_ms: float  # median GPU latency, batch 1
    v_mb: float  # incremental peak, decimal MB
    s_mb: float  # serialized TensorRT plan, decimal MB

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.e, self.l_ms, self.v_mb, self.s_mb)):
            raise ValueError("fitness values must be finite")
        if not (0 <= self.e <= 1 and self.l_ms > 0 and self.v_mb >= 0 and self.s_mb > 0):
            raise ValueError("fitness values are outside their valid domains")


@dataclass(frozen=True, slots=True)
class TCCJobSpec:
    run_id: str
    protocol_id: str
    plan_id: str
    dataset_id: str
    environment_id: str
    space_hash: str
    method: Method
    stage: Stage
    seed_block: int
    search_seed: int
    train_seed: int
    calibration_seed: int
    budget_gpu_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.budget_gpu_s) or self.budget_gpu_s <= 0:
            raise ValueError("budget_gpu_s must be positive and finite")
        if self.stage == "confirmatory" and self.budget_gpu_s != 19800:
            raise ValueError("confirmatory budget must be 19,800 GPU seconds")


@dataclass(frozen=True, slots=True)
class GenomeRecord:
    raw_genome: Mapping[str, Any]
    expressed_genome: Mapping[str, Any]
    repair_log: tuple[str, ...]
    phenotype_hash: str
    cache_context_hash: str
    space_hash: str


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    run_id: str
    candidate_id: str
    attempt_id: str
    stage: Stage
    genome: GenomeRecord
    train_seed: int
    calibration_seed: int
    dataset_id: str
    environment_id: str
    protocol_id: str
    plan_id: str
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_id: str
    candidate_id: str
    attempt_id: str
    status: AttemptStatus
    fitness: Fitness | None
    gpu_seconds: float
    durations_s: Mapping[str, float] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    source_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if (self.status == "valid") != (self.fitness is not None):
            raise ValueError("only valid attempts have fitness")
        if not isfinite(self.gpu_seconds) or self.gpu_seconds < 0:
            raise ValueError("gpu_seconds must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class TCCEvent:
    event_id: str
    ts_utc: str
    run_id: str
    kind: str
    payload: Mapping[str, Any]
    candidate_id: str | None = None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class SearchCheckpoint:
    run_id: str
    generation: int
    population: tuple[str, ...]
    pareto_archive: tuple[str, ...]
    pending_candidates: tuple[str, ...]
    rng_states: Mapping[str, Any]
    unique_evaluations: int
    active_gpu_seconds: float
    protocol_hash: str
    plan_hash: str
    space_hash: str


class Evaluator(Protocol):
    """Scientific evaluator injected into the search engine."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class RunStore(Protocol):
    """Persistence port; the core never imports a concrete database."""

    def create_run(self, job: TCCJobSpec) -> None: ...
    def register_candidate(
        self, run_id: str, candidate_id: str, genome: GenomeRecord, metadata: Mapping[str, Any]
    ) -> None: ...
    def find_cached(self, run_id: str, cache_context_hash: str) -> EvaluationResult | None: ...
    def commit_evaluation(self, result: EvaluationResult) -> None: ...
    def save_checkpoint(self, checkpoint: SearchCheckpoint) -> None: ...
    def load_checkpoint(self, run_id: str) -> SearchCheckpoint | None: ...
    def record_gpu_segment(self, run_id: str, attempt_id: str, gpu_seconds: float) -> None: ...
    def record_front(
        self, run_id: str, candidate_ids: tuple[str, ...], active_gpu_seconds: float, hypervolume: float | None
    ) -> None: ...


EventSink = Callable[[TCCEvent], None]
