"""Resume procedure after a crash or a planned stop (TCC-03, "Modelo de execução").

Resuming means: refuse a job spec that differs from the stored run, close
attempts left `running` as `interrupted` (their GPU time stays in the ledger),
load the latest checkpoint verified against protocol/plan/space hashes, and
list the candidates that still need a new attempt. The search runner then
restores population/RNG from the checkpoint and retries only those
candidates, each with a fresh `attempt_id`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from cnnlab.tcc.contracts import SearchCheckpoint, TCCJobSpec
from cnnlab.tcc.storage.errors import CheckpointMismatchError, NotFoundError
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore


@dataclass(frozen=True, slots=True)
class ResumePlan:
    run_id: str
    checkpoint: SearchCheckpoint | None
    interrupted_attempt_ids: tuple[str, ...]
    retry_candidate_ids: tuple[str, ...]
    attempts_per_candidate: Mapping[str, int]
    active_gpu_seconds: float
    unique_valid_evaluations: int


def prepare_resume(store: SQLiteRunStore, job: TCCJobSpec) -> ResumePlan:
    """Bring a run to a consistent state and describe what is left to do.

    Must run with no evaluator alive for this run: it closes `running` attempts.
    """
    stored = store.get_run(job.run_id)
    if stored is None:
        raise NotFoundError(f"run {job.run_id!r} does not exist")
    if stored != job:
        raise CheckpointMismatchError(f"job spec for {job.run_id!r} differs from the stored run")
    interrupted = store.recover_interrupted_attempts(job.run_id)
    history = store.run_history(job.run_id)
    return ResumePlan(
        run_id=job.run_id,
        checkpoint=history.latest_checkpoint,
        interrupted_attempt_ids=interrupted,
        retry_candidate_ids=history.unresolved_candidate_ids,
        attempts_per_candidate=dict(Counter(a.candidate_id for a in history.attempts)),
        active_gpu_seconds=history.active_gpu_seconds,
        unique_valid_evaluations=history.unique_valid_evaluations,
    )


def retry_attempt_id(candidate_id: str, attempts_so_far: int) -> str:
    """Deterministic, never-reused id for the next attempt of a candidate."""
    return f"{candidate_id}.a{attempts_so_far + 1}"
