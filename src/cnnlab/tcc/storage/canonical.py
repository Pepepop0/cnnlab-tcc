"""Canonical JSON, hashing, UTC timestamps and (de)serialization of v2 contracts.

Idempotency compares bytes, so every contract object has exactly one JSON
form: sorted keys, no whitespace, UTF-8, finite floats only, tuples as lists.
Numeric fields whose contract type is `float` are coerced with `float()` so
`Fitness(0.2, 1, 1, 1)` and `Fitness(0.2, 1.0, 1.0, 1.0)` hash the same.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timezone
from typing import Any

from cnnlab.tcc.contracts import (
    EvaluationResult,
    Fitness,
    GenomeRecord,
    SearchCheckpoint,
)
from cnnlab.tcc.storage.errors import SerializationError

RESULT_FORMAT = "tcc-evaluation-result/1"
GENOME_FORMAT = "tcc-genome-record/1"
CHECKPOINT_FORMAT = "tcc-search-checkpoint/1"


def to_jsonable(value: Any, *, _path: str = "$") -> Any:
    """Return a plain JSON tree or raise `SerializationError` naming the bad path."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError(f"{_path}: non-finite float {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(f"{_path}: mapping key {key!r} is not a str")
            out[key] = to_jsonable(item, _path=f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, _path=f"{_path}[{i}]") for i, item in enumerate(value)]
    raise SerializationError(f"{_path}: {type(value).__name__} is not JSON-serializable")


def canonical_json(value: Any) -> str:
    """Single, byte-stable JSON representation used for hashes and idempotency."""
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def format_utc(moment: datetime) -> str:
    """ISO 8601 in UTC with `Z`; naive datetimes are rejected, never assumed local."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    text = moment.astimezone(UTC).isoformat(timespec="microseconds")
    return text.replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(UTC)


# --- Fitness -----------------------------------------------------------------


def fitness_to_payload(fitness: Fitness | None) -> dict[str, float] | None:
    if fitness is None:
        return None
    return {
        "e": float(fitness.e),
        "l_ms": float(fitness.l_ms),
        "v_mb": float(fitness.v_mb),
        "s_mb": float(fitness.s_mb),
    }


def fitness_from_payload(payload: Mapping[str, Any] | None) -> Fitness | None:
    if payload is None:
        return None
    return Fitness(
        e=float(payload["e"]),
        l_ms=float(payload["l_ms"]),
        v_mb=float(payload["v_mb"]),
        s_mb=float(payload["s_mb"]),
    )


# --- EvaluationResult --------------------------------------------------------


def result_to_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "format": RESULT_FORMAT,
        "run_id": result.run_id,
        "candidate_id": result.candidate_id,
        "attempt_id": result.attempt_id,
        "status": result.status,
        "fitness": fitness_to_payload(result.fitness),
        "gpu_seconds": float(result.gpu_seconds),
        "durations_s": {k: float(v) for k, v in result.durations_s.items()},
        "artifact_paths": dict(result.artifact_paths),
        "artifact_sha256": dict(result.artifact_sha256),
        "quality": to_jsonable(result.quality, _path="$.quality"),
        "failure_stage": result.failure_stage,
        "failure_code": result.failure_code,
        "failure_detail": result.failure_detail,
        "source_attempt_id": result.source_attempt_id,
    }


def result_from_payload(payload: Mapping[str, Any]) -> EvaluationResult:
    if payload.get("format") != RESULT_FORMAT:
        raise SerializationError(f"unsupported result format {payload.get('format')!r}")
    return EvaluationResult(
        run_id=payload["run_id"],
        candidate_id=payload["candidate_id"],
        attempt_id=payload["attempt_id"],
        status=payload["status"],
        fitness=fitness_from_payload(payload["fitness"]),
        gpu_seconds=float(payload["gpu_seconds"]),
        durations_s=dict(payload["durations_s"]),
        artifact_paths=dict(payload["artifact_paths"]),
        artifact_sha256=dict(payload["artifact_sha256"]),
        quality=dict(payload["quality"]),
        failure_stage=payload["failure_stage"],
        failure_code=payload["failure_code"],
        failure_detail=payload["failure_detail"],
        source_attempt_id=payload["source_attempt_id"],
    )


# --- GenomeRecord ------------------------------------------------------------


def genome_to_payload(genome: GenomeRecord) -> dict[str, Any]:
    return {
        "format": GENOME_FORMAT,
        "raw_genome": to_jsonable(genome.raw_genome, _path="$.raw_genome"),
        "expressed_genome": to_jsonable(genome.expressed_genome, _path="$.expressed_genome"),
        "repair_log": list(genome.repair_log),
        "phenotype_hash": genome.phenotype_hash,
        "cache_context_hash": genome.cache_context_hash,
        "space_hash": genome.space_hash,
    }


def genome_from_payload(payload: Mapping[str, Any]) -> GenomeRecord:
    if payload.get("format") != GENOME_FORMAT:
        raise SerializationError(f"unsupported genome format {payload.get('format')!r}")
    return GenomeRecord(
        raw_genome=payload["raw_genome"],
        expressed_genome=payload["expressed_genome"],
        repair_log=tuple(payload["repair_log"]),
        phenotype_hash=payload["phenotype_hash"],
        cache_context_hash=payload["cache_context_hash"],
        space_hash=payload["space_hash"],
    )


# --- SearchCheckpoint --------------------------------------------------------


def checkpoint_to_payload(checkpoint: SearchCheckpoint) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "run_id": checkpoint.run_id,
        "generation": checkpoint.generation,
        "population": list(checkpoint.population),
        "pareto_archive": list(checkpoint.pareto_archive),
        "pending_candidates": list(checkpoint.pending_candidates),
        "rng_states": to_jsonable(checkpoint.rng_states, _path="$.rng_states"),
        "unique_evaluations": checkpoint.unique_evaluations,
        "active_gpu_seconds": float(checkpoint.active_gpu_seconds),
        "protocol_hash": checkpoint.protocol_hash,
        "plan_hash": checkpoint.plan_hash,
        "space_hash": checkpoint.space_hash,
    }


def checkpoint_from_payload(payload: Mapping[str, Any]) -> SearchCheckpoint:
    """Rebuild a checkpoint; JSON arrays inside `rng_states` come back as lists."""
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SerializationError(f"unsupported checkpoint format {payload.get('format')!r}")
    return SearchCheckpoint(
        run_id=payload["run_id"],
        generation=int(payload["generation"]),
        population=tuple(payload["population"]),
        pareto_archive=tuple(payload["pareto_archive"]),
        pending_candidates=tuple(payload["pending_candidates"]),
        rng_states=dict(payload["rng_states"]),
        unique_evaluations=int(payload["unique_evaluations"]),
        active_gpu_seconds=float(payload["active_gpu_seconds"]),
        protocol_hash=payload["protocol_hash"],
        plan_hash=payload["plan_hash"],
        space_hash=payload["space_hash"],
    )
