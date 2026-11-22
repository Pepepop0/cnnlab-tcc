"""The sequential method (task A5): architecture search, freeze, then representation and pruning search.

CPU only, with the deterministic ``FakeEvaluator`` (synthetic values, one GPU second per fresh evaluation).
The admitted slice is the FP16-per-block one (the largest the pipeline materializes today) and the
architecture slice is the FP32 one, which lies inside it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import TCCEvent, TCCJobSpec
from cnnlab.tcc.operation.service import OperationService
from cnnlab.tcc.operation.testing import ClockedFakeEvaluator, FakeMonotonic
from cnnlab.tcc.science.genome import Genome
from cnnlab.tcc.science.operators import FP16_BLOCK_SLICE, FP32_SLICE, SearchSlice
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.search import (
    DirectContext,
    DirectHashes,
    DirectResume,
    SearchConfig,
    SearchRunner,
)
from cnnlab.tcc.science.sequential import (
    ARCHITECTURE_BLOCK,
    ARCHITECTURE_GLOBAL,
    SequentialConfig,
    SequentialRunner,
    architecture_signature,
    graft,
    representation_is_searchable,
)
from cnnlab.tcc.science.space import SPACE_V1
from cnnlab.tcc.storage import JobQueue, ReadModel
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, ManualClock, make_job, seed_reference_rows

ADMITTED = FP16_BLOCK_SLICE
SPACE_HASH = ADMITTED.space_hash()


def context_for(job: TCCJobSpec) -> CacheContext:
    return CacheContext(
        split_fingerprint="split-45k-5k-cal1024-v1",
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        train_policy={"epochs": 1},
        pruning_policy={},
        quantization_policy={},
        code_version="test-fixture",
        backend={"torch": "cpu-test"},
        builder_config={},
        space_hash=SPACE_HASH,
    )


def make_runner(
    population: int = 6, n_frozen: int = 3, stages: tuple[float, float] | None = (20.0, 20.0), **config: Any
) -> SequentialRunner:
    return SequentialRunner(
        SearchConfig(population=population, generations=None, **config),
        SequentialConfig(n_frozen, stages),
        ADMITTED,
        FP32_SLICE,
        context_for,
    )


def new_run(tmp_path: Path, name: str) -> tuple[SQLiteRunStore, TCCJobSpec]:
    store = SQLiteRunStore(tmp_path / name / "db" / "tcc.sqlite3", tmp_path / name / "artifacts")
    seed_reference_rows(store)
    job = make_job(f"{name}-run", method="sequential", stage="pilot", space_hash=SPACE_HASH)
    store.create_run(job)
    return store, job


def context(
    store: SQLiteRunStore,
    job: TCCJobSpec,
    evaluator: Any,
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
        resume=resume or DirectResume(),
        hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
        artifact_root=store.artifact_root,
        emit=(events.append if events is not None else (lambda e: None)),
        should_stop=should_stop,
        active_gpu_seconds=lambda: store.active_gpu_seconds(job.run_id),
        unique_valid_evaluations=lambda: store.unique_valid_evaluations(job.run_id),
    )


def candidates(store: SQLiteRunStore, run_id: str) -> list[dict[str, Any]]:
    """Every registered candidate in registration order: genome, metadata and phenotype."""
    rows = store._conn.execute(
        "SELECT candidate_id, generation, genome_json, metadata_json, phenotype_hash FROM candidates "
        "WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "generation": r[1],
            "genome": Genome.from_dict(json.loads(r[2])["raw_genome"]),
            "meta": json.loads(r[3]),
            "phenotype": r[4],
        }
        for r in rows
    ]


def measured(store: SQLiteRunStore, run_id: str) -> dict[str, tuple[float, float, float, float]]:
    with ReadModel(store.db_path) as rm:
        return {
            r["candidate_id"]: (r["e"], r["l_ms"], r["v_mb"], r["s_mb"])
            for r in rm.evaluations(run_id)
            if r["status"] == "valid"
        }


def all_fp32(genome: Genome) -> bool:
    return genome.stem_precision == genome.head_precision == "fp32" and all(
        b.precision == "fp32" and b.prune_pct == 0 for b in genome.blocks
    )


def run_to_the_end(
    tmp_path: Path, name: str = "seq", **kw: Any
) -> tuple[SQLiteRunStore, TCCJobSpec, list[TCCEvent], FakeEvaluator]:
    store, job = new_run(tmp_path, name)
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    status = make_runner(**kw)(context(store, job, evaluator, events=events))
    assert status == "completed"
    return store, job, events, evaluator


# --- the two stages ------------------------------------------------------------------------


def test_stage_one_searches_fp32_architectures_and_freezes_the_lowest_error_ones(tmp_path: Path) -> None:
    store, job, events, evaluator = run_to_the_end(tmp_path)
    [changed] = [e for e in events if e.kind == "stage_changed"]
    assert (changed.payload["from"], changed.payload["to"]) == (1, 2)
    assert changed.payload["active_gpu_seconds"] == 20.0 and changed.payload["stage1_unused_gpu_s"] == 0.0

    rows = candidates(store, job.run_id)
    stage1 = [c for c in rows if c["generation"] < changed.payload["generation"]]
    assert stage1 and all(all_fp32(c["genome"]) for c in stage1)  # representation fixed, whatever was proposed
    assert {c["meta"]["origin"] for c in stage1} == {"initial", "offspring"}

    error = {cid: v[0] for cid, v in measured(store, job.run_id).items()}
    phenotype = {c["id"]: c["phenotype"] for c in rows}
    expected: list[str] = []
    seen: set[str] = set()
    for cid in sorted((c["id"] for c in stage1 if c["id"] in error), key=lambda k: (error[k], phenotype[k], k)):
        if phenotype[cid] not in seen:
            seen.add(phenotype[cid])
            expected.append(cid)
    frozen = [f["candidate_id"] for f in changed.payload["frozen"]]
    assert frozen == expected[:3] and len(set(phenotype[c] for c in frozen)) == 3  # the 3 lowest E, all distinct
    assert [f["e"] for f in changed.payload["frozen"]] == sorted(f["e"] for f in changed.payload["frozen"])
    assert changed.payload["dropped_unevaluated"]  # proposals of stage 1 that never got GPU time
    for dropped in changed.payload["dropped_unevaluated"]:
        assert dropped not in error  # never evaluated late
    store.close()


def test_stage_two_keeps_the_architecture_and_moves_only_representation_and_pruning(tmp_path: Path) -> None:
    store, job, events, evaluator = run_to_the_end(tmp_path)
    [changed] = [e for e in events if e.kind == "stage_changed"]
    rows = {c["id"]: c for c in candidates(store, job.run_id)}
    frozen = [f["candidate_id"] for f in changed.payload["frozen"]]
    signatures = {architecture_signature(rows[f]["genome"]): f for f in frozen}
    stage2 = [c for c in rows.values() if c["generation"] >= changed.payload["generation"]]
    assert {c["meta"]["origin"] for c in stage2} == {"initial_stage2", "offspring_stage2"}

    for c in stage2:
        owner = signatures[architecture_signature(c["genome"])]  # exactly one of the frozen architectures
        assert c["meta"]["architecture"] == owner
        assert ADMITTED.contains(c["genome"])  # the same admitted set as the joint method
        if c["meta"]["origin"] == "offspring_stage2":  # parents belong to the architecture they breed
            assert all(
                architecture_signature(rows[p]["genome"]) == architecture_signature(c["genome"])
                for p in c["meta"]["parents"]
            )
    # representation did move: some blocks run in FP16, and no individual is just a frozen copy
    assert any(b.precision == "fp16" for c in stage2 for b in c["genome"].active_blocks)
    for c in stage2:
        assert c["phenotype"] not in {rows[f]["phenotype"] for f in frozen}  # a new evaluation, not a baseline copy
    store.close()


def test_the_rotating_allocation_gives_every_frozen_architecture_the_same_share(tmp_path: Path) -> None:
    store, job, events, _ = run_to_the_end(tmp_path)  # population 6 over 3 architectures: exactly even
    per_architecture = Counter(
        c["meta"]["architecture"] for c in candidates(store, job.run_id) if "architecture" in c["meta"]
    )
    assert len(per_architecture) == 3 and len(set(per_architecture.values())) == 1
    store.close()

    # 4 architectures for a population of 6 does not divide: within one pair of each other, all the same
    store2, job2, _, _ = run_to_the_end(tmp_path, name="seq4", population=6, n_frozen=4)
    counts = Counter(c["meta"]["architecture"] for c in candidates(store2, job2.run_id) if "architecture" in c["meta"])
    assert len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 2
    store2.close()


def test_each_stage_gets_its_own_time_and_the_total_is_the_sum(tmp_path: Path) -> None:
    store, job, events, evaluator = run_to_the_end(tmp_path)
    assert len(evaluator.calls) == 40 and store.active_gpu_seconds(job.run_id) == 40.0  # 20 s + 20 s
    [finished] = [e for e in events if e.kind == "sequential_finished"]
    assert finished.payload["stage2_gpu_seconds"] == 20.0 and finished.payload["active_gpu_seconds"] == 40.0
    assert finished.payload["reason"] == "stage 2 used its GPU time"
    assert finished.payload["unused_gpu_s"] == job.budget_gpu_s - 40.0  # declared, not converted into evaluations
    assert not any(e.kind in ("search_completed", "search_exhausted") for e in events)
    store.close()


def test_default_stage_budgets_are_half_and_half_of_the_job_budget(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, "halves")  # 19,800 s: 9,900 s + 9,900 s
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator(gpu_seconds=1000.0)
    status = make_runner(stages=None)(context(store, job, evaluator, events=events))
    assert status == "completed"
    [changed] = [e for e in events if e.kind == "stage_changed"]
    assert (changed.payload["stage1_budget_gpu_s"], changed.payload["stage2_budget_gpu_s"]) == (9900.0, 9900.0)
    assert changed.payload["active_gpu_seconds"] == 10_000.0  # the attempt that crosses 9,900 s finishes and counts
    started = next(e for e in events if e.kind == "search_started")
    assert started.payload["sequential"]["stage_budgets_gpu_s"] == [9900.0, 9900.0]
    assert started.payload["sequential"]["n_frozen"] == 3
    # stage 2 starts at 10,000 s and has its own 9,900 s: the run ends at 19,900 s here (no service to cut at 19,800)
    assert store.active_gpu_seconds(job.run_id) == 20_000.0
    store.close()


def test_a_stage_one_that_cannot_find_anything_new_hands_over_and_its_time_is_not_given_to_stage_two(
    tmp_path: Path,
) -> None:
    store, job = new_run(tmp_path, "exhausted")
    runner = make_runner(stages=(20.0, 20.0), max_stalled_generations=2)
    fixed = FP32_SLICE.sample(GeneRng(1))
    original = SearchRunner._propose

    def propose(ctx: Any, state: Any, rng: Any, make: Any, cache: Any) -> Any:
        if state.stage == 1:  # every proposal of stage 1 is the same architecture: a duplicate after the first
            return fixed, {}
        return original(runner, ctx, state, rng, make, cache)

    runner._propose = propose  # type: ignore[method-assign]
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    assert runner(context(store, job, evaluator, events=events)) == "completed"
    [changed] = [e for e in events if e.kind == "stage_changed"]
    assert changed.payload["reason"] == "stage 1 can no longer produce a new evaluation"
    assert changed.payload["active_gpu_seconds"] == 1.0 and changed.payload["stage1_unused_gpu_s"] == 19.0
    assert len(changed.payload["frozen"]) == 1  # only one architecture existed
    # Stage 1 left 19 s unused. They are not handed to stage 2: it has its own 20 s and the run never
    # goes beyond 1 s + 20 s (it may end sooner: one architecture has few representations to try).
    assert 1.0 < store.active_gpu_seconds(job.run_id) <= 21.0
    [ended] = [e for e in events if e.kind in ("sequential_finished", "search_exhausted")]
    if ended.kind == "sequential_finished":
        assert ended.payload["stage2_gpu_seconds"] == 20.0
    store.close()


def test_without_any_valid_architecture_the_run_ends_instead_of_compressing_nothing(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, "nothing")
    events: list[TCCEvent] = []
    evaluator = FakeEvaluator(gpu_seconds=1.0)
    status = make_runner()(context(store, job, _AllInvalid(evaluator), events=events))
    assert status == "completed"
    assert not any(e.kind == "stage_changed" for e in events)
    [finished] = [e for e in events if e.kind == "sequential_finished"]
    assert "no valid architecture" in finished.payload["reason"]
    store.close()


class _AllInvalid:
    """An evaluator for which every candidate is invalid (so there is nothing to freeze)."""

    def __init__(self, inner: FakeEvaluator) -> None:
        self.inner = inner

    def evaluate(self, request: Any) -> Any:
        self.inner.invalid_contexts = frozenset({request.genome.cache_context_hash})
        return self.inner.evaluate(request)


def test_the_richest_admitted_slice_also_lets_stage_two_compress_the_stem_and_the_head(tmp_path: Path) -> None:
    from cnnlab.tcc.science.operators import FP16_REGIONS_SLICE

    regions = FP16_REGIONS_SLICE

    def regions_context(job: TCCJobSpec) -> CacheContext:
        return CacheContext(
            split_fingerprint="split-45k-5k-cal1024-v1",
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,
            train_policy={"epochs": 1},
            pruning_policy={},
            quantization_policy={},
            code_version="test-fixture",
            backend={"torch": "cpu-test"},
            builder_config={},
            space_hash=regions.space_hash(),
        )

    store = SQLiteRunStore(tmp_path / "regions" / "db" / "tcc.sqlite3", tmp_path / "regions" / "artifacts")
    seed_reference_rows(store)
    job = make_job("regions-run", method="sequential", stage="pilot", space_hash=regions.space_hash())
    store.create_run(job)
    runner = SequentialRunner(
        SearchConfig(population=6, generations=None),
        SequentialConfig(3, (20.0, 20.0)),
        regions,
        FP32_SLICE,
        regions_context,
    )
    assert runner(context(store, job, FakeEvaluator(gpu_seconds=1.0))) == "completed"
    rows = candidates(store, job.run_id)
    stage1 = [c for c in rows if c["meta"].get("origin") in ("initial", "offspring")]
    stage2 = [c for c in rows if "architecture" in c["meta"]]
    assert stage1 and stage2 and all(all_fp32(c["genome"]) for c in stage1)
    assert all(regions.contains(c["genome"]) for c in rows)
    assert any(c["genome"].stem_precision == "fp16" or c["genome"].head_precision == "fp16" for c in stage2)
    store.close()


def test_the_admitted_space_and_its_architecture_slice_run_the_two_stages_and_stay_inside(tmp_path: Path) -> None:
    from cnnlab.tcc.science.operators import SLICES

    admitted, architecture = SLICES["admitted"], SLICES["architecture"]

    def admitted_context(job: TCCJobSpec) -> CacheContext:
        return CacheContext(
            split_fingerprint="split-45k-5k-cal1024-v1",
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,
            train_policy={"epochs": 1},
            pruning_policy={},
            quantization_policy={},
            code_version="test-fixture",
            backend={"torch": "cpu-test"},
            builder_config={},
            space_hash=admitted.space_hash(),
        )

    store = SQLiteRunStore(tmp_path / "admitted" / "db" / "tcc.sqlite3", tmp_path / "admitted" / "artifacts")
    seed_reference_rows(store)
    job = make_job("admitted-run", method="sequential", stage="pilot", space_hash=admitted.space_hash())
    store.create_run(job)
    runner = SequentialRunner(
        SearchConfig(population=6, generations=None),
        SequentialConfig(3, (30.0, 30.0)),
        admitted,
        architecture,
        admitted_context,
    )
    assert runner(context(store, job, FakeEvaluator(gpu_seconds=1.0))) == "completed"
    rows = candidates(store, job.run_id)
    stage1 = [c for c in rows if c["meta"].get("origin") in ("initial", "offspring")]
    stage2 = [c for c in rows if "architecture" in c["meta"]]
    assert stage1 and stage2 and all(architecture.contains(c["genome"]) for c in stage1)
    assert all(admitted.contains(c["genome"]) for c in rows)  # BF16 outside its scope was projected, not drawn
    store.close()


def test_a_frozen_architecture_grafted_onto_a_sample_is_projected_into_the_admitted_slice() -> None:
    from cnnlab.tcc.science.operators import SLICES

    admitted, architecture = SLICES["admitted"], SLICES["architecture"]
    for seed in range(300):
        frozen = architecture.sample(GeneRng(seed))
        grafted = graft(frozen, admitted.sample(GeneRng(10_000 + seed)))
        assert admitted.contains(admitted.pin(grafted)), seed  # what _initial_individual hands to the evaluator


# --- the front, the archive and the curve ------------------------------------------------------


def test_the_archive_and_the_hypervolume_cover_both_stages(tmp_path: Path) -> None:
    reference = (0.99, 5.2, 52.0, 20.2)
    store, job, events, _ = run_to_the_end(tmp_path, reference_point=reference)
    with ReadModel(store.db_path) as rm:
        fronts = rm.fronts(job.run_id)
    volumes = [f["hypervolume"] for f in fronts]
    assert len(fronts) >= 6 and volumes == sorted(volumes) and 0.0 < volumes[-1] <= 1.0
    [changed] = [e for e in events if e.kind == "stage_changed"]
    stage1_ids = {c["id"] for c in candidates(store, job.run_id) if c["generation"] < changed.payload["generation"]}
    # the stage-1 results are on the curve before stage 2 starts: a snapshot at exactly the switch time
    assert any(f["active_gpu_seconds"] == 20.0 for f in fronts)
    last = fronts[-1]
    assert last["active_gpu_seconds"] == 40.0  # the final snapshot closes the partial generation at the stop
    assert set(last["candidate_ids"]) & stage1_ids  # an FP32 architecture is still on the final front
    store.close()


# --- stop and resume ---------------------------------------------------------------------------


def resumed_run(tmp_path: Path, name: str, stops: list[int]) -> tuple[SQLiteRunStore, TCCJobSpec]:
    """Run to the end in sessions that each stop after ``stops[i]`` fresh evaluations (a planned stop)."""
    store, job = new_run(tmp_path, name)
    runner = make_runner()
    resume = DirectResume()
    for limit in [*stops, None]:
        evaluator = FakeEvaluator(gpu_seconds=1.0)
        status = runner(
            context(
                store,
                job,
                evaluator,
                resume=resume,
                should_stop=(lambda e=evaluator, n=limit: n is not None and len(e.calls) >= n),
            )
        )
        if limit is None:
            assert status == "completed"
            break
        assert status == "stopped"
        checkpoint = store.load_checkpoint(job.run_id)
        assert checkpoint is not None
        resume = DirectResume(checkpoint=checkpoint, attempts_per_candidate={})
    return store, job


def fingerprint(store: SQLiteRunStore, job: TCCJobSpec) -> dict[str, Any]:
    strip = lambda cid: cid.split(".", 1)[1]  # noqa: E731 - the run id prefix differs between the two runs
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None
    return {
        "genomes": [
            (
                strip(c["id"]),
                c["genome"].to_dict(),
                c["meta"].get("origin"),
                c["meta"].get("architecture") and strip(c["meta"]["architecture"]),
            )
            for c in candidates(store, job.run_id)
        ],
        "population": sorted(strip(c) for c in checkpoint.population),
        "archive": sorted(strip(c) for c in checkpoint.pareto_archive),
        "gpu": store.active_gpu_seconds(job.run_id),
        "measured": sorted((strip(k), v) for k, v in measured(store, job.run_id).items()),
    }


# Fresh evaluations per session: 20 + 20 in all, so every stop must fall before the work runs out.
@pytest.mark.parametrize("stops", [[7], [12, 15], [20], [25], [7, 19, 10]])
def test_stopping_and_resuming_anywhere_reproduces_the_uninterrupted_run(tmp_path: Path, stops: list[int]) -> None:
    reference_store, reference_job, _, _ = run_to_the_end(tmp_path, name="whole")
    expected = fingerprint(reference_store, reference_job)
    reference_store.close()

    store, job = resumed_run(tmp_path, "pieces", stops)
    assert fingerprint(store, job) == expected  # same candidates, same stage switch, same final archive
    store.close()


# --- construction and the operators ------------------------------------------------------------


def test_the_runner_refuses_what_would_make_the_method_meaningless() -> None:
    with pytest.raises(ValueError, match="GPU time per stage"):
        SequentialRunner(
            SearchConfig(population=6, generations=3), SequentialConfig(), ADMITTED, FP32_SLICE, context_for
        )
    with pytest.raises(ValueError, match="nothing to search"):  # an admitted slice that pins all representation
        SequentialRunner(
            SearchConfig(population=6, generations=None), SequentialConfig(), FP32_SLICE, FP32_SLICE, context_for
        )
    outside = SearchSlice(
        base=SPACE_V1,
        pinned_global=(("head_precision", "fp32"), ("stem_precision", "fp32")),
        pinned_block=(("precision", "fp32"), ("prune_pct", 25)),
        reason="prunes 25 %: not inside the admitted slice",
    )
    with pytest.raises(ValueError, match="inside the admitted slice"):
        SequentialRunner(
            SearchConfig(population=6, generations=None), SequentialConfig(), ADMITTED, outside, context_for
        )
    assert representation_is_searchable(ADMITTED) and not representation_is_searchable(FP32_SLICE)
    for bad in (
        {"n_frozen": 0},
        {"n_frozen": True},
        {"stage_gpu_s": (0.0, 5.0)},
        {"stage_gpu_s": (1.0,)},
        {"stage_gpu_s": (1.0, float("nan"))},
    ):
        with pytest.raises(ValueError):
            SequentialConfig(**bad)
    assert SequentialConfig().n_frozen == 7 and SequentialConfig(stage_gpu_s=[3, 4]).stage_gpu_s == (3.0, 4.0)  # type: ignore[arg-type]
    assert SequentialConfig().budgets(19_800.0) == (9_900.0, 9_900.0)


def test_the_runner_runs_only_the_sequential_method(tmp_path: Path) -> None:
    store, job = new_run(tmp_path, "wrong")
    other = make_job("wrong-joint", method="joint", stage="pilot", space_hash=SPACE_HASH)
    store.create_run(other)
    with pytest.raises(ValueError, match="sequential method"):
        make_runner()(context(store, other, FakeEvaluator()))
    store.close()


def test_graft_takes_architecture_from_one_genome_and_representation_from_the_other() -> None:
    rng = GeneRng(5)
    architecture, donor = FP32_SLICE.sample(rng), ADMITTED.sample(rng)
    grafted = graft(architecture, donor)
    assert architecture_signature(grafted) == architecture_signature(architecture)
    assert graft(architecture, architecture) == architecture and graft(architecture, grafted) == grafted
    for name in ("stem_precision", "head_precision", "int8_mode", "weight_granularity", "calibration"):
        assert getattr(grafted, name) == getattr(donor, name)
    for slot, (g, d) in enumerate(zip(grafted.blocks, donor.blocks)):
        assert (g.precision, g.prune_pct, g.sparsity) == (d.precision, d.prune_pct, d.sparsity), slot
    assert ADMITTED.contains(grafted)
    assert architecture_signature(grafted) != architecture_signature(donor) or architecture == donor
    assert set(ARCHITECTURE_GLOBAL) | {"stem_precision", "head_precision"} <= set(Genome.__dataclass_fields__)
    assert set(ARCHITECTURE_BLOCK) <= set(grafted.blocks[0].__dataclass_fields__)


def test_variation_in_stage_two_never_touches_the_architecture_and_always_changes_something(tmp_path: Path) -> None:
    runner = make_runner()
    rng = GeneRng(11)
    architecture = FP32_SLICE.sample(rng)
    copies = 0
    for trial in range(60):
        a = graft(architecture, ADMITTED.sample(rng))
        b = graft(architecture, ADMITTED.sample(rng))
        c1, c2, info = runner._vary(a, b, rng, architecture, 0.3)
        for child in (c1, c2):
            assert architecture_signature(child) == architecture_signature(architecture), trial  # latent slots too
            assert ADMITTED.contains(child)
            copies += child in (a, b)
        assert len(info["mutated"]) == 2
    # A child that equals a parent before mutation is always mutated, so an exact copy of a parent is rare:
    # it can only be the mutation of one parent landing on the other (few representations to tell them apart).
    assert copies <= 3  # measured: 1 of 120 children with this seed
