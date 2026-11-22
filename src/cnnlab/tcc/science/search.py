"""Search runner v2: NSGA-II (joint), mono-objective and random search over a genome slice.

``SearchRunner`` is a callable ``runner(ctx) -> "completed" | "stopped"``
matching the ``SearchRunner`` protocol of the operation service.
``ctx`` is typed here structurally (:class:`RunnerContextLike`) so science
never imports ``operation`` or SQLite; :class:`DirectContext` builds an
equivalent context around any ``RunStore`` for runs without the service.

Per candidate: sample/vary -> repair/express/hash -> register -> cache lookup
(same run only) -> evaluate -> commit. Invalid attempts are committed as
such and ranked behind every valid one; no objective value is invented.

An ``interrupted`` result means the infrastructure failed and the candidate
was not observed. It is committed (its GPU time counts) and the candidate is
tried again with a new ``attempt_id``; after ``max_interrupted_retries`` the
runner saves a checkpoint and raises :class:`SearchBlockedError` instead of
ranking unobserved candidates as invalid.

Resuming from a checkpoint that is older than the last committed evaluation
(a crash, not a planned stop) adopts what the store already holds: a pending
candidate that the resume plan lists as resolved gets its committed result
(valid fitness from the same-run cache lookup, otherwise invalid) and is not
evaluated again.

Generations. ``SearchConfig.generations`` is a limit or ``None``. With ``None``
the only normal stop is the GPU budget, which the operation service enforces
between attempts through ``ctx.should_stop``; the runner then returns
``"stopped"``. A search that can no longer produce a new evaluation (every
offspring is a duplicate served by the cache, which costs no GPU time and so
never reaches the budget) ends as ``"completed"`` after
``max_stalled_generations`` such generations in a row, with a
``search_exhausted`` event; the protocol lets a run end before its cut.

Front and hypervolume. The external archive is the set of non-dominated valid candidates
over **all four objectives** for every method (TCC-01 section 4): ``mono`` selects on E
only, but its archive, like the others', is judged on (E, L, V, S). With
``SearchConfig.reference_point`` set (read from the frozen plan by the wiring), every
front snapshot also stores the exact hypervolume of the archive (``hypervolume.py``)
normalized by the reference point, in [0, 1], and a ``front_recorded`` event carries the
raw volume too, labeled, and lists the evaluated candidates beyond the reference point.

Checkpoints are appended after the initial population, whenever a stop is
requested, and after each generation. ``SearchCheckpoint`` carries candidate
ids only and the ``RunStore`` port has no call to read genomes back, so the
runner keeps its own resumable state (raw genomes and objective vectors of
the candidates it still needs) under ``rng_states["nsga2"]``, next to the
gene RNG state. This is an interim encoding.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from cnnlab.tcc.contracts import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    EventSink,
    GenomeRecord,
    RunStore,
    SearchCheckpoint,
    TCCEvent,
    TCCJobSpec,
)
from cnnlab.tcc.science.genome import Genome
from cnnlab.tcc.science.hypervolume import hypervolume_of
from cnnlab.tcc.science.nsga2 import (
    Objectives,
    environmental_selection,
    pareto_archive,
    rank_and_crowding,
    tournament,
)
from cnnlab.tcc.science.operators import (
    DEFAULT_CROSSOVER_P,
    DEFAULT_MUTATION_P,
    DEFAULT_SIGMA_REL,
    OPERATORS_VERSION,
    SearchSlice,
    make_offspring,
    prepare_in_slice,
)
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng

RunnerStatus = Literal["completed", "stopped"]
RUNNER_VERSION = "tcc-search-runner/v2"
STATE_KEY = "nsga2"


class SearchBlockedError(RuntimeError):
    """The evaluator kept returning ``interrupted``: the infrastructure needs attention.

    A checkpoint was saved before raising; the pending candidates are retried
    by the next session with new attempt ids.
    """


class HashesLike(Protocol):
    protocol_hash: str
    plan_hash: str
    space_hash: str


class ResumeLike(Protocol):
    """``retry_candidate_ids`` (candidates without a valid/invalid attempt) is optional."""

    checkpoint: SearchCheckpoint | None
    attempts_per_candidate: Mapping[str, int]


class RunnerContextLike(Protocol):
    """Structural twin of ``operation.service.RunnerContext`` (fields the runner uses)."""

    job: TCCJobSpec
    store: RunStore
    evaluator: Evaluator
    resume: ResumeLike
    hashes: HashesLike
    artifact_root: Path
    emit: EventSink
    should_stop: Callable[[], bool]
    active_gpu_seconds: Callable[[], float]
    unique_valid_evaluations: Callable[[], int]


@dataclass(frozen=True, slots=True)
class DirectResume:
    checkpoint: SearchCheckpoint | None = None
    attempts_per_candidate: Mapping[str, int] = field(default_factory=dict)
    retry_candidate_ids: tuple[str, ...] | None = None  # None: unknown, nothing is adopted


@dataclass(frozen=True, slots=True)
class DirectHashes:
    protocol_hash: str
    plan_hash: str
    space_hash: str


@dataclass(frozen=True)
class DirectContext:
    """A runner context without the operation service (no lease, no metering)."""

    job: TCCJobSpec
    store: RunStore
    evaluator: Evaluator
    resume: ResumeLike
    hashes: HashesLike
    artifact_root: Path
    emit: EventSink
    should_stop: Callable[[], bool]
    active_gpu_seconds: Callable[[], float]
    unique_valid_evaluations: Callable[[], int]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Operator probabilities default to TCC-02 section 4; population 30 in the protocol.

    ``generations=None`` means no limit: the run ends by its GPU budget (campaign setting).
    ``max_stalled_generations`` only matters then; see the module docstring.
    ``reference_point`` is the hypervolume reference ``(r_E, r_L, r_V, r_S)``; None records no hypervolume.
    """

    population: int = 30
    generations: int | None = 10
    crossover_p: float = DEFAULT_CROSSOVER_P
    mutation_p: float = DEFAULT_MUTATION_P
    sigma_rel: float = DEFAULT_SIGMA_REL
    max_resamples: int = 10
    max_interrupted_retries: int = 1
    max_stalled_generations: int = 3
    reference_point: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.reference_point is not None:
            point = tuple(float(x) for x in self.reference_point)
            if len(point) != 4 or not all(math.isfinite(x) and x > 0 for x in point):
                raise ValueError("reference_point must be four finite positive numbers (E, L, V, S)")
            object.__setattr__(self, "reference_point", point)  # a list from JSON becomes a tuple
        if self.population < 2 or self.population % 2:
            raise ValueError("population must be an even number >= 2")
        if self.generations is not None and (
            isinstance(self.generations, bool) or not isinstance(self.generations, int) or self.generations < 0
        ):
            raise ValueError("generations must be an int >= 0, or None for no limit")
        if self.max_stalled_generations < 1:
            raise ValueError("max_stalled_generations must be >= 1")
        if self.max_interrupted_retries < 0:
            raise ValueError("max_interrupted_retries must be >= 0")

    def definition(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "generations": self.generations,
            "crossover_p": self.crossover_p,
            "mutation_p": self.mutation_p,
            "sigma_rel": self.sigma_rel,
            "max_resamples": self.max_resamples,
            "max_interrupted_retries": self.max_interrupted_retries,
            "max_stalled_generations": self.max_stalled_generations,
            "reference_point": None if self.reference_point is None else list(self.reference_point),
            "operators": OPERATORS_VERSION,
            "runner": RUNNER_VERSION,
        }


def objectives_for(method: str, fitness: Any) -> Objectives | None:
    """Objective vector used for selection: all four for joint/random, E only for mono."""
    if fitness is None:
        return None
    if method == "mono":
        return (fitness.e,)
    return (fitness.e, fitness.l_ms, fitness.v_mb, fitness.s_mb)


@dataclass
class _State:
    generation: int = 0
    phase: str = "initial"  # "initial" | "offspring" (children proposed, survival pending) | "survived"
    next_index: int = 0
    population: list[str] = field(default_factory=list)
    offspring: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    genomes: dict[str, Genome] = field(default_factory=dict)
    objectives: dict[str, Objectives | None] = field(default_factory=dict)
    fitness4: dict[str, tuple[float, float, float, float] | None] = field(default_factory=dict)
    seen_phenotypes: dict[str, str] = field(default_factory=dict)  # phenotype -> first candidate id
    # Used by the sequential method only (see ``sequential.py``); the defaults are the other methods'.
    stage: int = 1
    frozen: list[str] = field(default_factory=list)  # candidate ids of the architectures frozen after stage 1
    rotation: int = 0  # how many proposals stage 2 has handed out, for the rotating allocation
    stage_start_gpu_s: float | None = None  # active GPU seconds when the current stage began

    def to_json(self) -> dict[str, Any]:
        keep = (
            set(self.population)
            | set(self.pending)
            | set(self.offspring)
            | {k for k, v in self.fitness4.items() if v is not None}
        )
        return {
            "generation": self.generation,
            "phase": self.phase,
            "next_index": self.next_index,
            "population": list(self.population),
            "offspring": list(self.offspring),
            "pending": list(self.pending),
            "genomes": {k: self.genomes[k].to_dict() for k in sorted(keep) if k in self.genomes},
            "fitness4": {
                k: (list(v) if v is not None else None) for k, v in sorted(self.fitness4.items()) if k in keep
            },
            "seen_phenotypes": dict(sorted(self.seen_phenotypes.items())),
            "stage": self.stage,
            "frozen": list(self.frozen),
            "rotation": self.rotation,
            "stage_start_gpu_s": self.stage_start_gpu_s,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], method: str) -> _State:
        fitness4 = {k: (tuple(v) if v is not None else None) for k, v in data["fitness4"].items()}
        state = cls(
            generation=int(data["generation"]),
            phase=str(data["phase"]),
            next_index=int(data["next_index"]),
            population=list(data["population"]),
            offspring=list(data["offspring"]),
            pending=list(data["pending"]),
            genomes={k: Genome.from_dict(v) for k, v in data["genomes"].items()},
            fitness4=fitness4,  # type: ignore[arg-type]
            seen_phenotypes=dict(data["seen_phenotypes"]),
            stage=int(data.get("stage", 1)),
            frozen=list(data.get("frozen", [])),
            rotation=int(data.get("rotation", 0)),
            stage_start_gpu_s=data.get("stage_start_gpu_s"),
        )
        state.objectives = {k: _project(method, v) for k, v in fitness4.items()}
        return state


def _project(method: str, f4: tuple[float, float, float, float] | None) -> Objectives | None:
    if f4 is None:
        return None
    return (f4[0],) if method == "mono" else tuple(f4)


class SearchRunner:
    """One run of ``joint`` (NSGA-II), ``mono`` (NSGA-II on E only) or ``random`` search.

    ``context_for(job)`` supplies the cache context of the evaluator (split
    fingerprint, seeds, policies, code, backend, builder config); it must be
    built for ``space`` so candidate hashes match the run's ``space_hash``.

    ``operator_space`` (default: ``space``) is where sampling and the genetic operators work. It
    must be a restriction of ``space`` (every genome it produces lies in ``space``): the sequential
    method searches architectures in the FP32 slice while the run is identified by the full slice.
    """

    def __init__(
        self,
        config: SearchConfig,
        space: SearchSlice,
        context_for: Callable[[TCCJobSpec], CacheContext],
        operator_space: SearchSlice | None = None,
    ) -> None:
        self.config = config
        self.space = space
        self.ops = operator_space if operator_space is not None else space
        self.context_for = context_for

    # -- helpers ----------------------------------------------------------------

    def _emit(
        self,
        ctx: RunnerContextLike,
        kind: str,
        payload: Mapping[str, Any],
        candidate_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        ctx.emit(
            TCCEvent(
                event_id=uuid.uuid4().hex,
                ts_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                run_id=ctx.job.run_id,
                kind=kind,
                payload=dict(payload),
                candidate_id=candidate_id,
                attempt_id=attempt_id,
            )
        )

    @staticmethod
    def _archive(state: _State) -> list[str]:
        """Non-dominated valid candidates on all four objectives, whatever the method selects on."""
        return pareto_archive(dict(state.fitness4))

    def _record_front(self, ctx: RunnerContextLike, state: _State) -> None:
        archive = self._archive(state)
        reference = self.config.reference_point
        volume: float | None = None
        if reference is not None:
            # every valid evaluated candidate, not only the archive: the volume is the same and the
            # list of candidates beyond the reference point is then complete
            result = hypervolume_of(state.fitness4, reference)
            volume = result.value  # the official one, normalized by the reference point: in [0, 1]
            self._emit(
                ctx,
                "front_recorded",
                {
                    "generation": state.generation,
                    "hypervolume": volume,
                    "hypervolume_raw": result.raw,
                    "normalization": "hypervolume = raw volume / prod(reference point)",
                    "reference_point": list(reference),
                    "front_size": len(archive),
                    "outside_reference": list(result.outside),
                },
            )
        ctx.store.record_front(ctx.job.run_id, tuple(archive), ctx.active_gpu_seconds(), volume)

    def _checkpoint(self, ctx: RunnerContextLike, state: _State, rng: GeneRng) -> None:
        archive = self._archive(state)
        ctx.store.save_checkpoint(
            SearchCheckpoint(
                run_id=ctx.job.run_id,
                generation=state.generation,
                population=tuple(state.population),
                pareto_archive=tuple(archive),
                pending_candidates=tuple(state.pending),
                rng_states={
                    "gene_rng": rng.get_state(),
                    STATE_KEY: state.to_json(),
                    "config": self.config.definition(),
                },
                unique_evaluations=ctx.unique_valid_evaluations(),
                active_gpu_seconds=ctx.active_gpu_seconds(),
                protocol_hash=ctx.hashes.protocol_hash,
                plan_hash=ctx.hashes.plan_hash,
                space_hash=ctx.hashes.space_hash,
            )
        )
        self._emit(
            ctx,
            "checkpoint",
            {
                "generation": state.generation,
                "population": list(state.population),
                "pending": list(state.pending),
                "pareto_archive": archive,
            },
        )

    def _new_candidate(
        self, ctx: RunnerContextLike, state: _State, genome: Genome, cache: CacheContext, metadata: dict[str, Any]
    ) -> str:
        candidate_id = f"{ctx.job.run_id}.g{state.generation:03d}.c{state.next_index:04d}"
        state.next_index += 1
        prepared = prepare_in_slice(genome, self.space, cache)
        record = prepared.record
        meta = {
            **metadata,
            "generation": state.generation,
            "repairs": len(record.repair_log),
            "invalid_reason": prepared.expression.invalid_reason,
        }
        ctx.store.register_candidate(ctx.job.run_id, candidate_id, record, meta)
        state.genomes[candidate_id] = genome
        state.pending.append(candidate_id)
        self._records[candidate_id] = (record, prepared.expression.invalid_reason)
        return candidate_id

    def _evaluate(self, ctx: RunnerContextLike, state: _State, candidate_id: str) -> bool:
        """Evaluate and commit one attempt; False if it was interrupted (candidate still pending)."""
        record, invalid_reason = self._records.get(candidate_id) or self._rebuild_record(ctx, state, candidate_id)
        run_id = ctx.job.run_id
        earlier = int(ctx.resume.attempts_per_candidate.get(candidate_id, 0))
        attempt_no = earlier + self._session_attempts.get(candidate_id, 0) + 1
        self._session_attempts[candidate_id] = self._session_attempts.get(candidate_id, 0) + 1
        attempt_id = f"{candidate_id}.a{attempt_no}"
        cached = ctx.store.find_cached(run_id, record.cache_context_hash) if invalid_reason is None else None
        if invalid_reason is not None:
            result = EvaluationResult(
                run_id,
                candidate_id,
                attempt_id,
                "invalid",
                None,
                0.0,
                failure_stage="repair",
                failure_code=invalid_reason.split(":")[0].strip() or "invalid_expression",
                failure_detail=invalid_reason,
            )
            ctx.store.commit_evaluation(result)
        elif cached is not None and cached.fitness is not None:
            source = cached.source_attempt_id or cached.attempt_id
            result = EvaluationResult(
                run_id,
                candidate_id,
                attempt_id,
                "valid",
                cached.fitness,
                0.0,
                quality={"cache_hit": True, "source_candidate_id": cached.candidate_id},
                source_attempt_id=source,
            )
            ctx.store.commit_evaluation(result)
        else:
            request = EvaluationRequest(
                run_id=run_id,
                candidate_id=candidate_id,
                attempt_id=attempt_id,
                stage=ctx.job.stage,
                genome=record,
                train_seed=ctx.job.train_seed,
                calibration_seed=ctx.job.calibration_seed,
                dataset_id=ctx.job.dataset_id,
                environment_id=ctx.job.environment_id,
                protocol_id=ctx.job.protocol_id,
                plan_id=ctx.job.plan_id,
                artifact_root=ctx.artifact_root,
            )
            self._fresh_evaluations += 1
            result = ctx.evaluator.evaluate(request)
            ctx.store.commit_evaluation(result)
        if result.status == "interrupted":
            self._emit(
                ctx,
                "candidate_interrupted",
                {
                    "generation": state.generation,
                    "failure_stage": result.failure_stage,
                    "failure_code": result.failure_code,
                    "gpu_seconds": result.gpu_seconds,
                },
                candidate_id,
                attempt_id,
            )
            return False
        f = result.fitness
        state.fitness4[candidate_id] = None if f is None else (f.e, f.l_ms, f.v_mb, f.s_mb)
        state.objectives[candidate_id] = objectives_for(ctx.job.method, f)
        state.pending.remove(candidate_id)
        self._emit(
            ctx,
            "candidate_evaluated",
            {
                "generation": state.generation,
                "status": result.status,
                "cache_hit": result.source_attempt_id is not None,
                "fitness": None if f is None else {"e": f.e, "l_ms": f.l_ms, "v_mb": f.v_mb, "s_mb": f.s_mb},
                "failure_stage": result.failure_stage,
                "failure_code": result.failure_code,
                "gpu_seconds": result.gpu_seconds,
            },
            candidate_id,
            attempt_id,
        )
        return True

    def _adopt_committed(self, ctx: RunnerContextLike, state: _State) -> None:
        """After a crash: take from the store the pending candidates it already resolved."""
        unresolved = getattr(ctx.resume, "retry_candidate_ids", None)
        if unresolved is None:
            return
        unresolved = set(unresolved)
        for candidate_id in list(state.pending):
            seen = int(ctx.resume.attempts_per_candidate.get(candidate_id, 0))
            if candidate_id in unresolved or seen == 0:
                continue
            record, _ = self._records.get(candidate_id) or self._rebuild_record(ctx, state, candidate_id)
            cached = ctx.store.find_cached(ctx.job.run_id, record.cache_context_hash)
            f = None if cached is None else cached.fitness
            state.fitness4[candidate_id] = None if f is None else (f.e, f.l_ms, f.v_mb, f.s_mb)
            state.objectives[candidate_id] = objectives_for(ctx.job.method, f)
            state.pending.remove(candidate_id)
            self._emit(
                ctx,
                "candidate_adopted",
                {
                    "generation": state.generation,
                    "status": "valid" if f is not None else "invalid",
                    "attempts_in_store": seen,
                },
                candidate_id,
            )

    def _rebuild_record(
        self, ctx: RunnerContextLike, state: _State, candidate_id: str
    ) -> tuple[GenomeRecord, str | None]:
        prepared = prepare_in_slice(state.genomes[candidate_id], self.space, self.context_for(ctx.job))
        pair = (prepared.record, prepared.expression.invalid_reason)
        self._records[candidate_id] = pair
        return pair

    def _propose(
        self, ctx: RunnerContextLike, state: _State, rng: GeneRng, make: Callable[[], Genome], cache: CacheContext
    ) -> tuple[Genome, dict[str, Any]]:
        """A genome whose phenotype is new in this run, after up to ``max_resamples`` retries."""
        genome = make()
        for tries in range(self.config.max_resamples):
            phenotype = prepare_in_slice(genome, self.space, cache).record.phenotype_hash
            if phenotype not in state.seen_phenotypes:
                return genome, {"resamples": tries}
            genome = make()
        return genome, {"resamples": self.config.max_resamples, "duplicate_after_resamples": True}

    def _register(
        self, ctx: RunnerContextLike, state: _State, genome: Genome, cache: CacheContext, metadata: dict[str, Any]
    ) -> str:
        candidate_id = self._new_candidate(ctx, state, genome, cache, metadata)
        state.seen_phenotypes.setdefault(self._records[candidate_id][0].phenotype_hash, candidate_id)
        return candidate_id

    def _evaluate_pending(self, ctx: RunnerContextLike, state: _State, rng: GeneRng) -> bool:
        """Evaluate pending candidates in order; False when a stop was requested (after checkpoint)."""
        interrupted: dict[str, int] = {}
        while state.pending:
            if self._stopped(ctx, state):
                self._checkpoint(ctx, state, rng)
                return False
            candidate_id = state.pending[0]
            if self._evaluate(ctx, state, candidate_id):
                continue
            interrupted[candidate_id] = interrupted.get(candidate_id, 0) + 1
            if interrupted[candidate_id] > self.config.max_interrupted_retries:
                self._checkpoint(ctx, state, rng)
                self._emit(
                    ctx,
                    "search_blocked",
                    {"generation": state.generation, "interrupted_attempts": interrupted[candidate_id]},
                    candidate_id,
                )
                raise SearchBlockedError(
                    f"candidate {candidate_id!r} was interrupted {interrupted[candidate_id]} times in a row; "
                    "checkpoint saved, fix the infrastructure and resume"
                )
        return True

    # -- main entry -------------------------------------------------------------------

    def __call__(self, ctx: RunnerContextLike) -> RunnerStatus:
        job = ctx.job
        if job.space_hash != self.space.space_hash():
            raise ValueError("job space_hash differs from the runner's search slice")
        cache = self.context_for(job)
        self._records: dict[str, tuple[GenomeRecord, str | None]] = {}
        self._session_attempts: dict[str, int] = {}
        self._fresh_evaluations = 0  # calls of the evaluator in this session (cache hits do not count)
        checkpoint = ctx.resume.checkpoint
        # A run stopped exactly at a generation boundary resumes from a checkpoint that already
        # holds this survival and its front; storing them again would repeat a point of the curve.
        recorded = False
        if checkpoint is not None and STATE_KEY in checkpoint.rng_states:
            rng = GeneRng.from_state(checkpoint.rng_states["gene_rng"])
            state = _State.from_json(checkpoint.rng_states[STATE_KEY], job.method)
            recorded = state.phase == "survived" and not state.pending
            self._emit(ctx, "search_resumed", {"generation": state.generation, "pending": list(state.pending)})
            self._adopt_committed(ctx, state)
        else:
            rng = GeneRng(job.search_seed)
            state = _State()
            self._emit(
                ctx,
                "search_started",
                {
                    "method": job.method,
                    "config": self.config.definition(),
                    "space_hash": job.space_hash,
                    "slice": self.space.reason,
                    **self._start_info(ctx),
                },
            )
            for _ in range(self.config.population):
                genome, info = self._propose(ctx, state, rng, lambda: self.ops.sample(rng), cache)
                self._register(ctx, state, genome, cache, {"origin": "initial", **info})
            state.population = list(state.pending)
            self._checkpoint(ctx, state, rng)
        self._rng, self._state, self._cache = rng, state, cache  # for the subclasses that run more than one loop
        return self._loop(ctx, state, rng, cache, recorded)

    # -- hooks (the defaults are the behavior of joint, mono and random) ------------

    def _start_info(self, ctx: RunnerContextLike) -> dict[str, Any]:
        """Extra fields of the ``search_started`` event."""
        return {}

    def _stopped(self, ctx: RunnerContextLike, state: _State) -> bool:
        """Whether to stop before the next attempt: the operation service's signal (budget, cancel, stop)."""
        return ctx.should_stop()

    def _on_exhausted(
        self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext, stalled: int
    ) -> RunnerStatus | None:
        """No generation produced a new evaluation: end the run (None would mean 'go on')."""
        self._emit(
            ctx,
            "search_exhausted",
            {
                "generation": state.generation,
                "stalled_generations": stalled,
                "reason": "no new evaluation: every offspring was a duplicate served by the cache",
                "pareto_archive": self._archive(state),
            },
        )
        return "completed"

    def _loop(
        self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext, recorded: bool
    ) -> RunnerStatus:
        stalled = 0
        while True:
            fresh_before = self._fresh_evaluations
            had_pending = bool(state.pending)  # a generation with nothing to evaluate is no sign of a stall
            if not self._evaluate_pending(ctx, state, rng):
                return "stopped"
            if state.phase == "offspring" and self.config.generations is None and had_pending:
                stalled = stalled + 1 if self._fresh_evaluations == fresh_before else 0
            if state.phase == "offspring":
                self._survive(ctx, state)
            state.phase = "survived"
            if not recorded:
                self._checkpoint(ctx, state, rng)
                self._record_front(ctx, state)
            recorded = False
            if self.config.generations is not None and state.generation >= self.config.generations:
                self._emit(
                    ctx, "search_completed", {"generation": state.generation, "pareto_archive": self._archive(state)}
                )
                return "completed"
            if stalled >= self.config.max_stalled_generations:
                status = self._on_exhausted(ctx, state, rng, cache, stalled)
                if status is not None:
                    return status
                stalled = 0
                continue  # the hook started something new (the next stage): evaluate what it proposed
            if self._stopped(ctx, state):
                return "stopped"
            state.generation += 1
            state.offspring = []
            self._make_offspring(ctx, state, rng, cache)
            state.phase = "offspring"
            self._checkpoint(ctx, state, rng)

    def _make_offspring(self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext) -> None:
        n = self.config.population
        if ctx.job.method == "random":
            for _ in range(n):
                genome, info = self._propose(ctx, state, rng, lambda: self.ops.sample(rng), cache)
                state.offspring.append(self._register(ctx, state, genome, cache, {"origin": "random", **info}))
            return
        ranks = rank_and_crowding({k: state.objectives.get(k) for k in state.population})
        parents = tournament(state.population, ranks, rng, n)
        children: list[tuple[Genome, dict[str, Any]]] = []
        for a, b in zip(parents[0::2], parents[1::2]):
            c1, c2, info = make_offspring(
                state.genomes[a],
                state.genomes[b],
                rng,
                self.ops,
                self.config.crossover_p,
                self.config.mutation_p,
                self.config.sigma_rel,
            )
            children += [(c1, {**info, "parents": [a, b]}), (c2, {**info, "parents": [a, b]})]
        for genome, info in children:

            def remutate(g: Genome = genome, i: dict[str, Any] = info) -> Genome:
                pa, pb = i["parents"]
                return make_offspring(
                    state.genomes[pa],
                    state.genomes[pb],
                    rng,
                    self.ops,
                    self.config.crossover_p,
                    1.0,
                    self.config.sigma_rel,
                )[0]

            first = [genome]
            chosen, extra = self._propose(ctx, state, rng, lambda: first.pop() if first else remutate(), cache)
            state.offspring.append(
                self._register(
                    ctx,
                    state,
                    chosen,
                    cache,
                    {"origin": "offspring", **info, **extra, "mutated": list(info["mutated"])},
                )
            )

    def _survive(self, ctx: RunnerContextLike, state: _State) -> None:
        if ctx.job.method == "random":
            state.population = list(state.offspring)
            return
        pool = list(dict.fromkeys(state.population + state.offspring))
        state.population = environmental_selection({k: state.objectives.get(k) for k in pool}, self.config.population)
