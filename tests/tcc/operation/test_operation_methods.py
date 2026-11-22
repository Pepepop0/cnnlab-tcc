"""The three searchable methods end to end, CPU and synthetic only.

Part 1 runs the real ``SearchRunner`` under the real operation service with no generation limit: each
method must end by the GPU budget alone. Part 2 puts items of each method in the run queue and lets the
real executor start real workers (the synthetic wiring, opted into explicitly) and checks what reaches
the database. Fitness values here are stand-ins labeled synthetic; nothing is measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cnnlab.tcc.contracts import TCCEvent, TCCJobSpec
from cnnlab.tcc.operation import launcher, schedule
from cnnlab.tcc.operation import scheduler as sch
from cnnlab.tcc.operation.service import OperationService
from cnnlab.tcc.operation.testing import ClockedFakeEvaluator, FakeMonotonic
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.search import SearchConfig, SearchRunner
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.testing import ManualClock, make_job, seed_reference_rows

NO_GPU = {"CUDA_VISIBLE_DEVICES": "-1"}
POPULATION = 4
ATTEMPT_GPU_S = 1000.0  # 19,800 s of budget: the 20th attempt crosses it


def context_for(job: TCCJobSpec) -> CacheContext:
    return CacheContext(
        split_fingerprint="synthetic-split",
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        train_policy={"synthetic": True},
        pruning_policy={},
        quantization_policy={},
        code_version="test",
        backend={"synthetic": True},
        builder_config={},
        space_hash=FP32_SLICE.space_hash(),
    )


@pytest.mark.parametrize("method", ["joint", "mono", "random"])
def test_each_method_runs_without_a_generation_limit_until_the_gpu_budget(tmp_path: Path, method: str) -> None:
    store = SQLiteRunStore(tmp_path / "db.sqlite", tmp_path / "artifacts", clock=ManualClock())
    seed_reference_rows(store)
    job = make_job(f"run-{method}", method=method, stage="pilot", space_hash=FP32_SLICE.space_hash())
    store.create_run(job)
    JobQueue(store).enqueue(f"job-{method}", job.run_id, options={"synthetic": True})
    mono = FakeMonotonic()
    events: list[TCCEvent] = []
    service = OperationService(
        store,
        SearchRunner(SearchConfig(population=POPULATION, generations=None), FP32_SLICE, context_for),
        ClockedFakeEvaluator(gpu_seconds=ATTEMPT_GPU_S, clock=mono),
        worker_id="w",
        lease_s=60.0,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=events.append,
    )
    outcome = service.run_next()
    assert outcome is not None and outcome.end_state == "done" and "budget reached" in outcome.detail, outcome

    assert store.get_run(job.run_id).method == method  # type: ignore[union-attr]
    attempts = store.list_attempts(job.run_id)
    assert len(attempts) == 20 and {a.status for a in attempts} == {"valid"}  # the budget was the only stop
    assert store.active_gpu_seconds(job.run_id) == pytest.approx(20_000.0)
    checkpoint = store.load_checkpoint(job.run_id)
    assert checkpoint is not None and checkpoint.generation >= 4  # many more generations than the pilot's 2
    assert not any(e.kind in ("search_completed", "search_exhausted") for e in events)
    started = next(e for e in events if e.kind == "search_started")
    assert started.payload["method"] == method and started.payload["config"]["generations"] is None
    store.close()


def test_the_sequential_method_runs_both_stages_under_the_operation_service(tmp_path: Path) -> None:
    from cnnlab.tcc.science.operators import FP16_BLOCK_SLICE
    from cnnlab.tcc.science.sequential import SequentialConfig, SequentialRunner

    admitted = FP16_BLOCK_SLICE

    def admitted_context(job: TCCJobSpec) -> CacheContext:
        return CacheContext(
            split_fingerprint="synthetic-split",
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,
            train_policy={"synthetic": True},
            pruning_policy={},
            quantization_policy={},
            code_version="test",
            backend={"synthetic": True},
            builder_config={},
            space_hash=admitted.space_hash(),
        )

    store = SQLiteRunStore(tmp_path / "db.sqlite", tmp_path / "artifacts", clock=ManualClock())
    seed_reference_rows(store)
    job = make_job("run-seq", method="sequential", stage="pilot", space_hash=admitted.space_hash())
    store.create_run(job)  # the budget is 19,800 s: 9,900 s for each stage
    JobQueue(store).enqueue("job-seq", job.run_id, options={"synthetic": True})
    mono = FakeMonotonic()
    events: list[TCCEvent] = []
    service = OperationService(
        store,
        SequentialRunner(
            SearchConfig(population=6, generations=None),
            SequentialConfig(n_frozen=3),
            admitted,
            FP32_SLICE,
            admitted_context,
        ),
        ClockedFakeEvaluator(gpu_seconds=ATTEMPT_GPU_S, clock=mono),
        worker_id="w",
        lease_s=60.0,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=events.append,
    )
    outcome = service.run_next()
    assert outcome is not None and outcome.end_state == "done" and "budget reached" in outcome.detail, outcome
    [changed] = [e for e in events if e.kind == "stage_changed"]
    assert changed.payload["active_gpu_seconds"] == 10_000.0  # stage 1: the attempt that crosses 9,900 s counts
    assert changed.payload["stage2_budget_gpu_s"] == 9_900.0 and len(changed.payload["frozen"]) == 3
    # stage 2 starts at 10,000 s; the run's own budget (19,800 s) is reached first and ends the job
    assert store.active_gpu_seconds(job.run_id) >= 19_800.0
    assert not any(e.kind == "sequential_finished" for e in events)  # ended by the budget, not by a stage limit
    assert store.get_run(job.run_id).method == "sequential"  # type: ignore[union-attr]
    store.close()


def test_queue_items_of_each_method_run_through_the_executor_and_real_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qdir = tmp_path / "fila"
    monkeypatch.setattr(schedule, "FACTORY", "cnnlab.tcc.operation.testing:synthetic_demo_wiring")
    for method in schedule.METHODS:
        settings = schedule.RunSettings(
            method=method,
            population=30,
            generations=None,
            train_per_class=None,
            budget_gpu_s=19_800,
            slice="fp16_block" if method == "sequential" else "fp32",
        )
        schedule.mutate(qdir, lambda st, m=method, v=settings: schedule.add_item(st, f"campanha-{m}", v))
    paths = schedule.queue_dir_paths(qdir)

    def spawn(spec_path: Path, item: schedule.QueueItem, work_dir: Path) -> launcher.WorkerHandle:
        return launcher.spawn_worker(
            paths["db"],
            paths["artifacts"],
            spec_path=spec_path,
            allow_synthetic=True,  # tests of the operation only
            extra_env={**NO_GPU, "CNNLAB_TCC_WORK_DIR": str(work_dir)},
        )

    assert sch.Scheduler(qdir, spawn=spawn, poll_s=0.2, heartbeat_s=0.5).run() == 0

    items = schedule.load_state(qdir).items
    assert [i.state for i in items] == [schedule.DONE] * 4, [i.detail for i in items]
    with ReadModel(paths["db"]) as rm:
        runs = {r["run_id"]: r for r in rm.runs()}
        jobs = {j["run_id"]: j for j in rm.jobs()}
        assert {runs[i.run_id]["method"] for i in items} == set(schedule.METHODS)
        for item in items:
            job = jobs[item.run_id]
            assert job["state"] == "done" and job["options"]["demo_spec"]["method"] == item.settings.method
            assert job["options"]["demo_spec"]["factory_args"]["generations"] is None
            assert job["options"]["synthetic_evaluator"] is True  # nothing here was measured
            assert all(e["synthetic"] for e in rm.evaluations(item.run_id))


@pytest.mark.parametrize("method", ["joint", "mono", "random", "sequential"])
def test_the_four_methods_run_in_the_admitted_slice_with_the_g7_pair(tmp_path: Path, method: str) -> None:
    """The campaign's slices end to end under the operation service, synthetic evaluator, equal budget for all."""
    import json

    from cnnlab.tcc import demo_wiring
    from cnnlab.tcc.science import operators
    from cnnlab.tcc.science.genome import Genome
    from cnnlab.tcc.science.sequential import SequentialConfig, SequentialRunner

    admitted, architecture = operators.SLICES["admitted"], operators.SLICES["architecture"]
    assert demo_wiring.ARCHITECTURE_SLICE is architecture is operators.ARCHITECTURE_SLICE  # the pair of G7, not a copy

    def admitted_context(job: TCCJobSpec) -> CacheContext:
        return CacheContext(
            split_fingerprint="synthetic-split",
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,
            train_policy={"synthetic": True},
            pruning_policy={},
            quantization_policy={},
            code_version="test",
            backend={"synthetic": True},
            builder_config={},
            space_hash=admitted.space_hash(),
        )

    store = SQLiteRunStore(tmp_path / "db.sqlite", tmp_path / "artifacts", clock=ManualClock())
    seed_reference_rows(store)
    job = make_job(f"adm-{method}", method=method, stage="pilot", space_hash=admitted.space_hash())  # type: ignore[arg-type]
    store.create_run(job)
    JobQueue(store).enqueue(f"job-{method}", job.run_id, options={"synthetic": True})
    config = SearchConfig(population=6, generations=None)
    runner = (
        SequentialRunner(
            config, SequentialConfig(n_frozen=3), admitted, demo_wiring.ARCHITECTURE_SLICE, admitted_context
        )
        if method == "sequential"
        else SearchRunner(config, admitted, admitted_context)
    )
    mono = FakeMonotonic()
    events: list[TCCEvent] = []
    service = OperationService(
        store,
        runner,
        ClockedFakeEvaluator(gpu_seconds=ATTEMPT_GPU_S, clock=mono),
        worker_id="w",
        lease_s=60.0,
        heartbeat_thread=False,
        monotonic=mono,
        on_event=events.append,
    )
    outcome = service.run_next()
    assert outcome is not None and outcome.end_state == "done" and "budget reached" in outcome.detail, outcome
    attempts = store.list_attempts(job.run_id)
    assert len(attempts) == 20 and {a.status for a in attempts} == {
        "valid"
    }  # the same budget, the same count, for every method
    rows = store._conn.execute(
        "SELECT genome_json, metadata_json FROM candidates WHERE run_id = ?", (job.run_id,)
    ).fetchall()
    genomes = [(Genome.from_dict(json.loads(g)["raw_genome"]), json.loads(m)) for g, m in rows]
    assert genomes and all(admitted.contains(g) for g, _ in genomes)  # a BF16 outside its scope is never drawn
    if method == "sequential":
        [changed] = [e for e in events if e.kind == "stage_changed"]
        assert len(changed.payload["frozen"]) == 3
        stage1 = [g for g, meta in genomes if meta.get("origin") in ("initial", "offspring")]
        stage2 = [g for g, meta in genomes if "architecture" in meta]
        assert stage1 and stage2 and all(architecture.contains(g) for g in stage1)
    store.close()
