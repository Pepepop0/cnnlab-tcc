"""End-to-end, CPU-only tests of ``SearchRunner`` against a real ``SQLiteRunStore``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import TCCEvent, TCCJobSpec
from cnnlab.tcc.science.hypervolume import hypervolume_of
from cnnlab.tcc.science.nsga2 import pareto_archive
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.search import (
    DirectContext,
    DirectHashes,
    DirectResume,
    SearchConfig,
    SearchRunner,
)
from cnnlab.tcc.storage import ReadModel
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_job, seed_reference_rows

SLICE_HASH = FP32_SLICE.space_hash()


def make_store(tmp_path: Path, *, name: str = "run") -> SQLiteRunStore:
    """A fresh DB and artifact root under ``tmp_path``, the DB kept outside the root."""
    return SQLiteRunStore(tmp_path / name / "db" / "tcc.sqlite3", tmp_path / name / "artifacts")


def context_for(job: TCCJobSpec) -> CacheContext:
    """Evaluator recipe built for the FP32 slice (required by ``prepare_in_slice``)."""
    return CacheContext(
        split_fingerprint="split-45k-5k-cal1024-v1",
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        train_policy={"epochs": 1, "batch": 128, "optimizer": "adamw", "schedule": "cosine"},
        pruning_policy={},
        quantization_policy={},
        code_version="test-fixture",
        backend={"torch": "cpu-test"},
        builder_config={},
        space_hash=SLICE_HASH,
    )


def make_direct_context(
    store: SQLiteRunStore,
    job: TCCJobSpec,
    evaluator: FakeEvaluator,
    *,
    events: list[TCCEvent] | None = None,
    should_stop: Any = lambda: False,
    resume: DirectResume | None = None,
) -> DirectContext:
    hashes = store.run_hashes(job.run_id)
    return DirectContext(
        job=job,
        store=store,
        evaluator=evaluator,
        resume=resume if resume is not None else DirectResume(),
        hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
        artifact_root=store.artifact_root,
        emit=(events.append if events is not None else (lambda e: None)),
        should_stop=should_stop,
        active_gpu_seconds=lambda: store.active_gpu_seconds(job.run_id),
        unique_valid_evaluations=lambda: store.unique_valid_evaluations(job.run_id),
    )


def new_run(
    tmp_path: Path, *, name: str, method: str = "joint", seed_block: int = 0
) -> tuple[SQLiteRunStore, TCCJobSpec]:
    """A store with a frozen plan and one created run, ready for a ``SearchRunner``."""
    store = make_store(tmp_path, name=name)
    seed_reference_rows(store)
    job = make_job(f"{name}-run", method=method, stage="pilot", seed_block=seed_block, space_hash=SLICE_HASH)
    store.create_run(job)
    return store, job


def candidate_count(store: SQLiteRunStore, run_id: str) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = ?", (run_id,)).fetchone()[0]


def attempt_count(store: SQLiteRunStore, run_id: str) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM evaluation_attempts WHERE run_id = ?", (run_id,)).fetchone()[0]


def raw_genomes_by_seq(store: SQLiteRunStore, run_id: str) -> list[str]:
    """Raw genomes of every candidate, ordered by registration sequence."""
    rows = store._conn.execute("SELECT genome_json FROM candidates WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
    return [r[0] for r in rows]


def expected_fitness_tuple(cache_context_hash: str, train_seed: int) -> tuple[float, float, float, float]:
    """Re-derives the fitness ``FakeEvaluator`` computes, independent of the runner."""
    digest = sha256_hex(f"{cache_context_hash}:{train_seed}")
    return (
        0.05 + 0.9 * int(digest[0:8], 16) / 0xFFFFFFFF,
        0.1 + 5.0 * int(digest[8:16], 16) / 0xFFFFFFFF,
        1.0 + 50.0 * int(digest[16:24], 16) / 0xFFFFFFFF,
        0.1 + 20.0 * int(digest[24:32], 16) / 0xFFFFFFFF,
    )


# --- end-to-end run --------------------------------------------------------------


@pytest.mark.parametrize("method", ["joint", "mono", "random"])
def test_search_runner_completes_and_evaluates_every_candidate(tmp_path: Path, method: str) -> None:
    store, job = new_run(tmp_path, name=method, method=method)
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    events: list[TCCEvent] = []
    config = SearchConfig(population=4, generations=2)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    status = runner(make_direct_context(store, job, evaluator, events=events))

    assert status == "completed"
    expected_n = config.population * (1 + config.generations)
    assert candidate_count(store, job.run_id) == expected_n
    assert attempt_count(store, job.run_id) == expected_n
    assert store.unique_valid_evaluations(job.run_id) == expected_n  # no duplicate phenotypes expected
    assert any(e.kind == "search_completed" for e in events)
    assert store.load_checkpoint(job.run_id) is not None

    # Every committed, non-cached fitness matches FakeEvaluator's own formula.
    rows = store._conn.execute(
        "SELECT c.cache_context_hash, m.e_fraction, m.l_ms, m.v_mb, m.s_mb "
        "FROM candidates c JOIN evaluation_attempts a ON a.candidate_id = c.candidate_id "
        "JOIN measurements m ON m.attempt_id = a.attempt_id "
        "WHERE c.run_id = ? AND a.source_attempt_id IS NULL",
        (job.run_id,),
    ).fetchall()
    assert rows, "expected at least one measured (non-cached) attempt"
    for cache_context_hash, e, l_ms, v_mb, s_mb in rows:
        assert (e, l_ms, v_mb, s_mb) == pytest.approx(expected_fitness_tuple(cache_context_hash, job.train_seed))
    store.close()


def test_checkpoint_final_state_matches_the_config(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="ckpt")
    config = SearchConfig(population=4, generations=2)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    assert runner(make_direct_context(store, job, FakeEvaluator())) == "completed"
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None
    assert checkpoint.generation == config.generations
    assert len(checkpoint.population) == config.population
    assert checkpoint.unique_evaluations == config.population * (1 + config.generations)
    store.close()


def test_search_runner_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    config = SearchConfig(population=4, generations=1)

    store_a, job_a = new_run(tmp_path, name="det-a")
    runner_a = SearchRunner(config, FP32_SLICE, context_for)
    assert runner_a(make_direct_context(store_a, job_a, FakeEvaluator())) == "completed"
    genomes_a = raw_genomes_by_seq(store_a, job_a.run_id)
    store_a.close()

    store_b, job_b = new_run(tmp_path, name="det-b")
    assert job_b.search_seed == job_a.search_seed  # same seed_block -> same seed, by construction
    runner_b = SearchRunner(config, FP32_SLICE, context_for)
    assert runner_b(make_direct_context(store_b, job_b, FakeEvaluator())) == "completed"
    genomes_b = raw_genomes_by_seq(store_b, job_b.run_id)
    store_b.close()

    assert genomes_a == genomes_b


def test_mono_method_selects_on_error_only_but_still_tracks_all_four_fitnesses(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="mono", method="mono")
    config = SearchConfig(population=4, generations=1)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    assert runner(make_direct_context(store, job, FakeEvaluator())) == "completed"
    # l_ms/v_mb/s_mb are still measured and stored even though selection uses e only.
    rows = store._conn.execute(
        "SELECT m.l_ms, m.v_mb, m.s_mb FROM measurements m "
        "JOIN evaluation_attempts a ON a.attempt_id = m.attempt_id WHERE a.run_id = ?",
        (job.run_id,),
    ).fetchall()
    assert rows and all(l_ms > 0 and v_mb >= 0 and s_mb > 0 for l_ms, v_mb, s_mb in rows)
    store.close()


# --- invalid candidates ------------------------------------------------------------


def test_invalid_candidate_is_ranked_behind_valid_and_never_gets_fitness(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="invalid")
    # Mirror the runner's first draw (no resampling happens for a fresh phenotype)
    # to know, ahead of time, which cache context the first candidate will hash to.
    rng = GeneRng(job.search_seed)
    first_genome = FP32_SLICE.sample(rng)
    cache = context_for(job)
    from cnnlab.tcc.science.operators import prepare_in_slice

    first_hash = prepare_in_slice(first_genome, FP32_SLICE, cache).record.cache_context_hash

    evaluator = FakeEvaluator(invalid_contexts=frozenset({first_hash}))
    events: list[TCCEvent] = []
    config = SearchConfig(population=2, generations=0)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    status = runner(make_direct_context(store, job, evaluator, events=events))
    assert status == "completed"

    evaluated = [e for e in events if e.kind == "candidate_evaluated"]
    assert len(evaluated) == 2
    first, second = evaluated
    assert first.payload["status"] == "invalid" and first.payload["fitness"] is None
    assert second.payload["status"] == "valid" and second.payload["fitness"] is not None

    completed = next(e for e in events if e.kind == "search_completed")
    assert first.candidate_id not in completed.payload["pareto_archive"]
    assert second.candidate_id in completed.payload["pareto_archive"]
    store.close()


# --- stop and resume ---------------------------------------------------------------


def test_stop_and_resume_evaluates_every_candidate_exactly_once(tmp_path: Path) -> None:
    config = SearchConfig(population=4, generations=2)

    # Reference: an uninterrupted run with the same seed, for the "no design
    # gap" comparison below.
    ref_store, ref_job = new_run(tmp_path, name="ref")
    SearchRunner(config, FP32_SLICE, context_for)(make_direct_context(ref_store, ref_job, FakeEvaluator()))
    ref_population = set(store_population(ref_store, ref_job.run_id))
    ref_store.close()

    store, job = new_run(tmp_path, name="resumed")
    stop_after = 2
    events: list[TCCEvent] = []

    def should_stop() -> bool:
        return sum(1 for e in events if e.kind == "candidate_evaluated") >= stop_after

    evaluator_1 = FakeEvaluator(gpu_seconds=1.0)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    status = runner(make_direct_context(store, job, evaluator_1, events=events, should_stop=should_stop))
    assert status == "stopped"
    assert len(evaluator_1.calls) == stop_after
    assert candidate_count(store, job.run_id) == config.population  # only the initial population was proposed

    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None
    assert len(checkpoint.pending_candidates) == config.population - stop_after

    evaluator_2 = FakeEvaluator(gpu_seconds=1.0)
    resume = DirectResume(checkpoint=checkpoint, attempts_per_candidate={})
    status2 = runner(make_direct_context(store, job, evaluator_2, events=events, resume=resume))
    assert status2 == "completed"

    # No candidate was ever evaluated twice as valid: one committed attempt each.
    expected_n = config.population * (1 + config.generations)
    assert candidate_count(store, job.run_id) == expected_n
    assert attempt_count(store, job.run_id) == expected_n
    assert set(evaluator_1.calls).isdisjoint(evaluator_2.calls)

    resumed_population = set(store_population(store, job.run_id))
    # The design keeps the gene RNG and the run's proposal state fully in the
    # checkpoint, so a stop that only pauses between evaluations (no RNG draw
    # is skipped or repeated) reproduces the same final population as an
    # uninterrupted run with the same seed.
    assert resumed_population == ref_population
    store.close()


def store_population(store: SQLiteRunStore, run_id: str) -> tuple[str, ...]:
    checkpoint = store.load_checkpoint(run_id)
    assert checkpoint is not None
    genomes = []
    for candidate_id in checkpoint.population:
        row = store._conn.execute(
            "SELECT genome_json FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        genomes.append(row[0])
    return tuple(genomes)


def test_should_stop_true_from_the_start_stops_before_any_generation(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="stop-early")
    config = SearchConfig(population=2, generations=1)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    status = runner(make_direct_context(store, job, FakeEvaluator(), should_stop=lambda: True))
    assert status == "stopped"
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None
    assert len(checkpoint.pending_candidates) == config.population
    store.close()


# --- no generation limit: the budget is the only normal stop (A1) ---------------------


def test_search_config_accepts_no_generation_limit_and_validates_it() -> None:
    config = SearchConfig(population=30, generations=None)
    assert config.generations is None and config.population == 30
    definition = config.definition()
    assert definition["generations"] is None and definition["max_stalled_generations"] == 3
    assert SearchConfig(population=4).generations == 10  # the default of the pilot did not change
    for bad in (
        {"generations": -1},
        {"generations": True},
        {"generations": 2.5},
        {"max_stalled_generations": 0},
        {"population": 3},
    ):
        with pytest.raises(ValueError):
            SearchConfig(**{"population": 4, **bad})


@pytest.mark.parametrize("method", ["joint", "mono", "random"])
def test_without_a_generation_limit_only_the_stop_signal_ends_the_run(tmp_path: Path, method: str) -> None:
    store, job = new_run(tmp_path, name=f"unlimited-{method}", method=method)
    config = SearchConfig(population=4, generations=None)
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    budget_calls = 4 * 6  # the initial population and five generations

    def budget_reached() -> bool:  # what the operation service answers once the GPU budget is spent
        return len(evaluator.calls) >= budget_calls

    status = SearchRunner(config, FP32_SLICE, context_for)(
        make_direct_context(store, job, evaluator, events=events, should_stop=budget_reached)
    )
    assert status == "stopped"  # not "completed": nothing but the budget ended this search
    assert not any(e.kind in ("search_completed", "search_exhausted") for e in events)
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None and checkpoint.generation >= 5  # far past the pilot's 2 generations
    assert attempt_count(store, job.run_id) == budget_calls
    store.close()


def test_run_without_a_generation_limit_resumes_and_keeps_going(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="unlimited-resume")
    config = SearchConfig(population=4, generations=None)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    first = FakeEvaluator(gpu_seconds=1.0)
    assert runner(make_direct_context(store, job, first, should_stop=lambda: len(first.calls) >= 10)) == "stopped"
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None and checkpoint.pending_candidates  # stopped inside a generation

    second = FakeEvaluator(gpu_seconds=1.0)
    resume = DirectResume(checkpoint=checkpoint, attempts_per_candidate={})
    assert (
        runner(make_direct_context(store, job, second, resume=resume, should_stop=lambda: len(second.calls) >= 14))
        == "stopped"
    )
    assert set(first.calls).isdisjoint(second.calls)  # nothing evaluated twice
    assert attempt_count(store, job.run_id) == 10 + 14
    assert store.load_checkpoint(job.run_id).generation > checkpoint.generation  # type: ignore[union-attr]
    store.close()


def _same_genome_runner(config: SearchConfig) -> SearchRunner:
    """A runner whose every proposal is one and the same genome: after the first evaluation
    everything is a duplicate served by the cache, which costs no GPU time."""
    runner = SearchRunner(config, FP32_SLICE, context_for)
    genome = FP32_SLICE.sample(GeneRng(1))
    runner._propose = lambda ctx, state, rng, make, cache: (genome, {})  # type: ignore[method-assign]
    return runner


def test_search_that_can_no_longer_evaluate_anything_new_ends_instead_of_looping(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="exhausted")
    config = SearchConfig(population=4, generations=None, max_stalled_generations=3)
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    events: list[TCCEvent] = []
    status = _same_genome_runner(config)(make_direct_context(store, job, evaluator, events=events))

    assert status == "completed"  # ended on its own although no limit and no stop was given
    assert len(evaluator.calls) == 1  # the first genome; every later candidate was a cache hit
    [exhausted] = [e for e in events if e.kind == "search_exhausted"]
    assert exhausted.payload["stalled_generations"] == 3 and exhausted.payload["generation"] == 3
    assert "duplicate" in exhausted.payload["reason"]
    assert not any(e.kind == "search_completed" for e in events)
    assert store.active_gpu_seconds(job.run_id) == 1.0  # the budget could never have been reached
    assert candidate_count(store, job.run_id) == 4 * 4  # the initial population and three generations
    store.close()


def test_the_exhaustion_guard_does_not_change_runs_with_a_generation_limit(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="limited-duplicates")
    config = SearchConfig(population=4, generations=5, max_stalled_generations=1)
    events: list[TCCEvent] = []
    status = _same_genome_runner(config)(make_direct_context(store, job, FakeEvaluator(), events=events))
    assert status == "completed"
    assert any(e.kind == "search_completed" for e in events) and not any(e.kind == "search_exhausted" for e in events)
    assert candidate_count(store, job.run_id) == 4 * 6  # all five generations ran
    store.close()


def test_a_generation_with_a_new_evaluation_resets_the_stall_count(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="stall-reset")
    config = SearchConfig(population=4, generations=None, max_stalled_generations=2)
    runner = SearchRunner(config, FP32_SLICE, context_for)
    fixed = FP32_SLICE.sample(GeneRng(1))
    fresh = [FP32_SLICE.sample(GeneRng(seed)) for seed in (11, 12, 13, 14)]
    proposals = {"generation": 0}

    def propose(ctx: Any, state: Any, rng: Any, make: Any, cache: Any) -> tuple[Any, dict[str, Any]]:
        proposals["generation"] = state.generation
        if state.generation == 2:  # one generation of genuinely new genomes between two stalled ones
            return fresh[len(state.pending) % 4], {}
        return fixed, {}

    runner._propose = propose  # type: ignore[method-assign]
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator()
    status = runner(make_direct_context(store, job, evaluator, events=events))
    assert status == "completed"
    [exhausted] = [e for e in events if e.kind == "search_exhausted"]
    assert exhausted.payload["generation"] == 4 and exhausted.payload["stalled_generations"] == 2
    assert len(evaluator.calls) > 1  # generation 2 did evaluate something new
    store.close()


# --- hypervolume in every front snapshot (A4) -----------------------------------------

# FakeEvaluator's ranges: E in [0.05, 0.95], L in [0.1, 5.1], V in [1, 51], S in [0.1, 20.1]. This
# reference point leaves a part of the candidates beyond it, so the listing has something to list.
REFERENCE = (0.9, 3.0, 30.0, 12.0)


def run_with_reference(
    tmp_path: Path, method: str, *, generations: int = 4
) -> tuple[SQLiteRunStore, TCCJobSpec, list[TCCEvent]]:
    store, job = new_run(tmp_path, name=f"hv-{method}", method=method)
    events: list[TCCEvent] = []
    config = SearchConfig(population=6, generations=generations, reference_point=REFERENCE)
    status = SearchRunner(config, FP32_SLICE, context_for)(
        make_direct_context(store, job, FakeEvaluator(), events=events)
    )
    assert status == "completed"
    return store, job, events


def measured(store: SQLiteRunStore, run_id: str) -> dict[str, tuple[float, float, float, float]]:
    with ReadModel(store.db_path) as rm:
        return {
            r["candidate_id"]: (r["e"], r["l_ms"], r["v_mb"], r["s_mb"])
            for r in rm.evaluations(run_id)
            if r["status"] == "valid"
        }


def test_reference_point_is_validated_normalized_and_recorded_in_the_definition() -> None:
    assert SearchConfig(population=4).reference_point is None
    assert SearchConfig(population=4).definition()["reference_point"] is None
    config = SearchConfig(population=4, reference_point=[1, 2.5, 12.5, 5])  # type: ignore[arg-type]  # JSON gives a list
    assert config.reference_point == (1.0, 2.5, 12.5, 5.0) and config.definition()["reference_point"] == [
        1.0,
        2.5,
        12.5,
        5.0,
    ]
    for bad in (
        (1, 2, 3),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 0),
        (1, 2, -3, 4),
        (1, 2, 3, float("nan")),
        (1, 2, 3, float("inf")),
    ):
        with pytest.raises(ValueError, match="reference_point"):
            SearchConfig(population=4, reference_point=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["joint", "mono", "random"])
def test_every_front_snapshot_stores_the_exact_hypervolume_of_its_archive(tmp_path: Path, method: str) -> None:
    store, job, events = run_with_reference(tmp_path, method)
    points = measured(store, job.run_id)
    with ReadModel(store.db_path) as rm:
        fronts = rm.fronts(job.run_id)
    assert len(fronts) == 5  # the initial population and four generations
    for snapshot in fronts:
        assert snapshot["hypervolume"] is not None
        # what had been evaluated when the snapshot was taken: every candidate up to its generation
        seen = {cid: v for cid, v in points.items() if int(cid.split(".g")[1][:3]) <= fronts.index(snapshot)}
        assert sorted(snapshot["candidate_ids"]) == sorted(pareto_archive(seen))  # 4 objectives, whatever the method
        assert snapshot["hypervolume"] == pytest.approx(hypervolume_of(seen, REFERENCE).value, rel=1e-12)
    volumes = [f["hypervolume"] for f in fronts]
    assert volumes == sorted(volumes) and volumes[-1] > 0  # the archive only ever gets better
    store.close()


def test_the_mono_archive_is_judged_on_all_four_objectives_not_on_error_alone(tmp_path: Path) -> None:
    store, job, events = run_with_reference(tmp_path, "mono")
    points = measured(store, job.run_id)
    with ReadModel(store.db_path) as rm:
        last = rm.fronts(job.run_id)[-1]
    assert len(last["candidate_ids"]) > 1  # an error-only archive would hold the single best candidate
    assert sorted(last["candidate_ids"]) == sorted(pareto_archive(points))
    best_error = min(points, key=lambda k: points[k][0])
    assert best_error in last["candidate_ids"]  # the best error is still there, it is just not alone
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None and sorted(checkpoint.pareto_archive) == sorted(pareto_archive(points))
    store.close()


def test_candidates_beyond_the_reference_point_are_listed_in_the_front_event(tmp_path: Path) -> None:
    store, job, events = run_with_reference(tmp_path, "joint")
    points = measured(store, job.run_id)
    beyond = sorted(k for k, v in points.items() if not all(x < r for x, r in zip(v, REFERENCE)))
    assert beyond and len(beyond) < len(points)  # the fixture is meaningful: some beyond, some inside
    recorded = [e for e in events if e.kind == "front_recorded"]
    assert len(recorded) == 5 and recorded[-1].payload["reference_point"] == list(REFERENCE)
    assert recorded[-1].payload["outside_reference"] == beyond  # every one, sorted, not only the archive's
    result = hypervolume_of(points, REFERENCE)
    assert recorded[-1].payload["hypervolume"] == pytest.approx(result.value)
    assert 0.0 < recorded[-1].payload["hypervolume"] < 1.0  # normalized: the official value
    # the raw volume travels with its label, and is the normalized one times the product of the point
    assert recorded[-1].payload["hypervolume_raw"] == pytest.approx(result.raw)
    assert result.raw == pytest.approx(recorded[-1].payload["hypervolume"] * math.prod(REFERENCE))
    assert "prod(reference point)" in recorded[-1].payload["normalization"]
    first = recorded[0].payload["outside_reference"]
    assert set(first) <= set(beyond) and len(first) <= len(beyond)  # the list only grows
    store.close()


def test_without_a_reference_point_no_hypervolume_is_recorded(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, name="hv-none")
    events: list[TCCEvent] = []
    SearchRunner(SearchConfig(population=4, generations=2), FP32_SLICE, context_for)(
        make_direct_context(store, job, FakeEvaluator(), events=events)
    )
    with ReadModel(store.db_path) as rm:
        fronts = rm.fronts(job.run_id)
    assert len(fronts) == 3 and all(f["hypervolume"] is None for f in fronts)
    assert not any(e.kind == "front_recorded" for e in events)
    store.close()


# --- job/space consistency ----------------------------------------------------------


def test_runner_rejects_a_job_built_for_another_space(tmp_path: Path) -> None:
    store = make_store(tmp_path, name="wrong-space")
    seed_reference_rows(store)
    job = make_job("wrong-space-run", stage="pilot", space_hash="0" * 64)
    store.create_run(job)
    runner = SearchRunner(SearchConfig(population=2, generations=0), FP32_SLICE, context_for)
    with pytest.raises(ValueError):
        runner(make_direct_context(store, job, FakeEvaluator()))
    store.close()
