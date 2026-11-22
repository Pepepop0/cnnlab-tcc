"""Synthetic runner, clocks and evaluators for CPU-only operation tests.

Nothing here is a search algorithm or a measurement. `SyntheticListRunner`
walks a fixed list of synthetic genomes (no selection, no genetics) so the
service can be exercised end to end; every result it commits carries
`quality["synthetic"] = True`, including cache hits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cnnlab.tcc.contracts import EvaluationRequest, EvaluationResult, GenomeRecord, SearchCheckpoint
from cnnlab.tcc.operation.demo import DatasetRegistration, DemoRunSpec, DemoWiring
from cnnlab.tcc.operation.recovery import retry_attempt_id
from cnnlab.tcc.operation.service import RunnerContext, RunnerStatus
from cnnlab.tcc.storage.testing import (
    DATASET_ID,
    SPACE_HASH,
    FakeEvaluator,
    cache_hit_result,
    make_genome,
    make_request,
    synthetic_splits,
)


class FakeMonotonic:
    """Monotonic clock advanced explicitly (by the fake evaluator or the test)."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a monotonic clock never goes back")
        with self._lock:
            self._now += seconds


@dataclass
class ClockedFakeEvaluator(FakeEvaluator):
    """`FakeEvaluator` whose synthetic GPU work advances a monotonic clock.

    `hooks[attempt_id]` runs midway through that attempt (after half of its
    synthetic time), e.g. to heartbeat, cancel, take the lease over or raise.
    """

    clock: FakeMonotonic | None = None
    hooks: dict[str, Callable[[EvaluationRequest], None]] = field(default_factory=dict)
    raise_on_contexts: frozenset[str] = frozenset()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        half = self.gpu_seconds / 2
        if self.clock is not None:
            self.clock.advance(half)
        hook = self.hooks.get(request.attempt_id)
        if hook is not None:
            hook(request)
        if request.genome.cache_context_hash in self.raise_on_contexts:
            self.calls.append(request.attempt_id)
            raise RuntimeError("synthetic evaluator crash")
        if self.clock is not None:
            self.clock.advance(self.gpu_seconds - half)
        return super().evaluate(request)


@dataclass
class SleepingFakeEvaluator(FakeEvaluator):
    """Real-time fake for subprocess crash tests: sleeps instead of using a GPU."""

    sleep_s: float = 0.2
    crash_on_attempt: str | None = None
    crash_after_s: float = 0.0
    crash_code: int = 9

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        import os

        if request.attempt_id == self.crash_on_attempt:
            time.sleep(self.crash_after_s)
            os._exit(self.crash_code)  # killed mid-attempt: no commit, no close
        time.sleep(self.sleep_s)
        return super().evaluate(request)


@dataclass
class SyntheticListRunner:
    """Evaluates candidates `c0..c{n-1}` in order, resuming from the store.

    `contexts[i]` lets two candidates share a cache context (a cache hit on the
    second one). A checkpoint is saved every `checkpoint_every` resolved
    candidates and whenever the runner stops.
    """

    n_candidates: int
    contexts: Sequence[str | None] = ()
    checkpoint_every: int = 2
    prefix: str = ""  # candidate ids are global in a database: runs that share one need distinct prefixes
    log: list[str] = field(default_factory=list)

    def genome(self, index: int) -> GenomeRecord:
        context = self.contexts[index] if index < len(self.contexts) else None
        return make_genome(index, context=context)

    def _checkpoint(self, ctx: RunnerContext, registered: list[str], next_index: int) -> None:
        latest = ctx.store.load_checkpoint(ctx.job.run_id)
        generation = 0 if latest is None else latest.generation
        ctx.store.save_checkpoint(
            SearchCheckpoint(
                run_id=ctx.job.run_id,
                generation=generation,
                population=tuple(registered),
                pareto_archive=(),
                pending_candidates=(),
                rng_states={"synthetic_list_runner": {"next_index": next_index}},
                unique_evaluations=ctx.unique_valid_evaluations(),
                active_gpu_seconds=ctx.active_gpu_seconds(),
                protocol_hash=ctx.hashes.protocol_hash,
                plan_hash=ctx.hashes.plan_hash,
                space_hash=ctx.hashes.space_hash,
            )
        )

    def __call__(self, ctx: RunnerContext) -> RunnerStatus:
        run_id = ctx.job.run_id
        resume = ctx.resume
        retry = set(resume.retry_candidate_ids)
        resolved = {cid for cid in resume.attempts_per_candidate if cid not in retry}
        registered: list[str] = []
        since_checkpoint = 0
        for index in range(self.n_candidates):
            cid = f"{self.prefix}c{index}"
            if cid in resolved:
                registered.append(cid)
                continue
            if ctx.should_stop():
                self._checkpoint(ctx, registered, index)
                self.log.append(f"stop@{cid}")
                return "stopped"
            genome = self.genome(index)
            ctx.store.register_candidate(run_id, cid, genome, {"generation": 0, "synthetic": True})
            registered.append(cid)
            attempt_id = retry_attempt_id(cid, resume.attempts_per_candidate.get(cid, 0))
            cached = ctx.store.find_cached(run_id, genome.cache_context_hash)
            if cached is not None:
                ctx.store.commit_evaluation(cache_hit_result(cached, candidate_id=cid, attempt_id=attempt_id))
                self.log.append(f"hit:{attempt_id}")
            else:
                result = ctx.evaluator.evaluate(make_request(ctx.job, cid, attempt_id, genome, ctx.artifact_root))
                ctx.store.commit_evaluation(result)
                self.log.append(f"{result.status}:{attempt_id}")
            since_checkpoint += 1
            if since_checkpoint >= self.checkpoint_every:
                self._checkpoint(ctx, registered, index + 1)
                since_checkpoint = 0
        self._checkpoint(ctx, registered, self.n_candidates)
        return "completed"


def synthetic_demo_wiring(spec: DemoRunSpec) -> DemoWiring:
    """Wiring factory for CPU tests of `operation.demo`; everything is labeled synthetic.

    `spec.factory_args`: `n_candidates` (default 4), `sleep_s` (0.05),
    `gpu_seconds` reported per attempt (10.0), `crash_on_attempt`, `crash_after_s`.
    A spec that names its run (as every queue item does) gets candidate ids prefixed with that run id,
    like the real runner's, so several runs can share one database.
    """
    args = dict(spec.factory_args)
    return DemoWiring(
        runner=SyntheticListRunner(
            n_candidates=int(args.get("n_candidates", 4)), prefix=f"{spec.run_id}." if spec.run_id else ""
        ),
        evaluator=SleepingFakeEvaluator(
            gpu_seconds=float(args.get("gpu_seconds", 10.0)),
            sleep_s=float(args.get("sleep_s", 0.05)),
            crash_on_attempt=args.get("crash_on_attempt"),
            crash_after_s=float(args.get("crash_after_s", 0.0)),
        ),
        dataset=DatasetRegistration(DATASET_ID, {"name": "CIFAR-100", "synthetic": True}, synthetic_splits()),
        space_hash=SPACE_HASH,
        runner_label="synthetic-list-runner",
        evaluator_label="sleeping-fake-evaluator",
        synthetic=True,
        environment={"gpu": "none (CPU test)", "note": "synthetic wiring for operation tests"},
    )


# -- the final phase (A12) ----------------------------------------------------------------------------------------


class SyntheticFinalEvaluator:
    """A final evaluator without a network: three scripted builds per retraining, the engine as a synthetic file.

    ``fail_runs`` names runs whose first attempt ends invalid (a technical failure); every result is synthetic and
    carries the bench version ``tcc-bench/synthetic-final``.
    """

    BENCH = "tcc-bench/synthetic-final"

    def __init__(self, fail_runs: Sequence[str] = ()) -> None:
        self.fail_runs = set(fail_runs)
        self.calls: list[str] = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        from cnnlab.tcc.contracts import Fitness
        from cnnlab.tcc.storage.artifacts import attempt_artifact_dir, publish_bytes
        from cnnlab.tcc.storage.canonical import sha256_hex

        self.calls.append(request.attempt_id)
        if request.run_id in self.fail_runs and request.attempt_id.endswith(".a1"):
            return EvaluationResult(
                request.run_id,
                request.candidate_id,
                request.attempt_id,
                "invalid",
                None,
                1.0,
                quality={"synthetic": True},
                failure_stage="build",
                failure_code="synthetic_build_failure",
                failure_detail="scripted technical failure",
            )
        digest = sha256_hex(f"{request.genome.phenotype_hash}:{request.train_seed}")
        unit = lambda i, lo, span: lo + span * int(digest[i : i + 8], 16) / 0xFFFFFFFF  # noqa: E731
        builds = [
            {
                "l_ms": unit(8, 1.0, 1.0) * (1 + 0.01 * k),
                "v_mb": unit(16, 10.0, 5.0),
                "s_mb": unit(24, 4.0, 2.0) * (1 + 0.001 * k),
            }
            for k in (-1, 0, 1)
        ]
        median = {key: sorted(b[key] for b in builds)[1] for key in ("l_ms", "v_mb", "s_mb")}
        rel = f"{attempt_artifact_dir(request.run_id, request.attempt_id)}/engine.plan"
        published = publish_bytes(
            request.artifact_root, rel, f"synthetic-final-engine:{digest}:{request.attempt_id}".encode()
        )
        return EvaluationResult(
            request.run_id,
            request.candidate_id,
            request.attempt_id,
            "valid",
            Fitness(e=unit(0, 0.2, 0.3), l_ms=median["l_ms"], v_mb=median["v_mb"], s_mb=median["s_mb"]),
            1.0,
            durations_s={"train": 0.7, "build": 0.2, "bench": 0.1},
            artifact_paths={"engine": published.rel_path},
            artifact_sha256={"engine": published.sha256},
            quality={
                "synthetic": True,
                "measurement": {"bench_version": self.BENCH},
                "final": {
                    "builds": builds,
                    "epochs": 30,
                    "manifest": {"engine_sha256": published.sha256, "label": request.attempt_id},
                },
            },
        )


class SyntheticTestEvaluator:
    """Opens nothing: answers an accuracy scripted from the engine's bytes. ``wrong_hash`` makes it report another engine."""

    def __init__(
        self, *, wrong_hash: bool = False, accuracy: Callable[[str], float] | None = None, claims: bool = True
    ) -> None:
        self.calls: list[str] = []
        self.wrong_hash = wrong_hash
        self.accuracy = accuracy
        self.claims = claims  # False: an evaluator that forgets to claim the engine in the ledger

    def evaluate_test(self, request: object) -> dict[str, object]:
        from cnnlab.tcc.storage.canonical import sha256_hex

        digest = sha256_hex(request.engine_path.read_bytes())  # type: ignore[attr-defined]
        if self.claims:
            request.ledger.claim(digest)  # type: ignore[attr-defined]  # right before "reading the test"
        self.calls.append(request.run_id)  # type: ignore[attr-defined]  # only an opened test counts as a call
        accuracy = self.accuracy(request.run_id) if self.accuracy else 0.3 + 0.4 * int(digest[:8], 16) / 0xFFFFFFFF  # type: ignore[attr-defined]
        return {
            "top1_accuracy": accuracy,
            "top5_accuracy": min(1.0, accuracy + 0.2),
            "n_examples": request.splits["test"].n_examples,  # type: ignore[attr-defined]
            "engine_sha256": "0" * 64 if self.wrong_hash else digest,
        }


def synthetic_final_wiring(spec: DemoRunSpec) -> object:
    """Factory for CPU tests of ``operation.final_stage``; everything is labeled synthetic."""
    from cnnlab.tcc.operation.final_stage import FinalWiring
    from cnnlab.tcc.science.record import CacheContext

    def context_for(job: object) -> CacheContext:
        return CacheContext(
            split_fingerprint="synthetic-split",
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,  # type: ignore[attr-defined]
            train_policy={"synthetic": True, "epochs": 30, "stage": "final"},
            pruning_policy={},
            quantization_policy={},
            code_version="test",
            backend={"synthetic": True},
            builder_config={},
            space_hash="a" * 64,  # a CacheContext wants a hash
        )

    return FinalWiring(
        evaluator=SyntheticFinalEvaluator(),
        test_evaluator=SyntheticTestEvaluator(),
        context_for=context_for,
        dataset=DatasetRegistration(DATASET_ID, {"name": "CIFAR-100", "synthetic": True}, synthetic_splits()),
        space_hash=SPACE_HASH,
        evaluator_label="synthetic-final-evaluator",
        synthetic=True,
        environment={"gpu": "none (CPU test)", "note": "synthetic final wiring for operation tests"},
    )
