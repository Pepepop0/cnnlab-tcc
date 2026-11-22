"""The glue between the final phase and the two-pass final evaluation of the science front (``cnnlab.tcc.final_wiring``). CPU only.

The science front is replaced by a small fake that does what its interface promises: ``final_build`` writes and hashes
its files under ``artifact_dir/label`` and returns the manifest; ``final_test`` checks the engine against the manifest,
claims it in the ledger it is given and answers a scripted accuracy. The test then runs the whole final phase through the
operation with the glue, which proves the layout of the artifacts is the one the catalog accepts and the ledger contract
holds on both sides. Nothing trains, builds an engine or reads an image.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # tests/tcc/operation has no package: borrow the campaign builder
from test_operation_final_stage import (  # noqa: E402
    LIST_ID,
    World,
    build_all,
    make_world,
    open_all,
    run,
)

from cnnlab.tcc import final_wiring as gw
from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.operation import final_stage as fs
from cnnlab.tcc.storage import ReadModel
from cnnlab.tcc.storage.testing import make_genome


class TestAlreadyEvaluated(RuntimeError):
    __test__ = False


@dataclass
class FakeBuildResult:
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.manifest


@dataclass
class FakeTestResult:
    body: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.body


@dataclass(frozen=True)
class FakeConfig:
    train_time_limit_s: float | None = None


@dataclass
class FakeTestAccess:
    grant: Any
    source: Any
    ledger: Any


class FakeFinalEval:
    """The module ``model_pipeline.final_eval`` of the science front, in miniature."""

    FinalConfig = FakeConfig
    TestAccess = FakeTestAccess
    TestAlreadyEvaluated = TestAlreadyEvaluated

    def __init__(self) -> None:
        self.grants: list[tuple[str, str]] = []
        self.builds: list[dict[str, Any]] = []
        self.prepared = 0
        self.tested: list[str] = []

    def final_grant(self, source: Any, run_id: str, finalist_list_id: str) -> object:
        self.grants.append((run_id, finalist_list_id))
        return object()

    def prepare_final_data(self, grant: Any, source: Any, device: Any) -> object:
        self.prepared += 1
        return object()

    def final_build(
        self,
        genome: Any,
        retrain_seed: int,
        config: Any,
        data: Any,
        work_dir: Path,
        artifact_dir: Path,
        *,
        label: str,
        device: str = "cuda",
    ) -> FakeBuildResult:
        target = Path(artifact_dir) / label
        target.mkdir(parents=True, exist_ok=True)
        body = f"engine:{label}:{json.dumps(genome, sort_keys=True)}:{retrain_seed}".encode()  # one engine per attempt
        (target / "engine.plan").write_bytes(body)
        (target / "bench.json").write_bytes(b"{}")
        digests = {"engine": hashlib.sha256(body).hexdigest(), "bench": hashlib.sha256(b"{}").hexdigest()}
        self.builds.append(
            {"genome": genome, "seed": retrain_seed, "label": label, "dir": Path(artifact_dir), "device": device}
        )
        builds = [
            {
                "index": i,
                "l_ms": 1.0 + 0.1 * i,
                "v_mb": 12.0,
                "s_mb": 4.0,
                "e_val": 0.3,
                "bench_version": "tcc-bench/fake-v4",
            }
            for i in range(3)
        ]
        return FakeBuildResult(
            {
                "status": "valid",
                "retrain_seed": retrain_seed,
                "label": label,
                "builds": builds,
                "chosen_build": 1,
                "l_ms": 1.1,
                "v_mb": 12.0,
                "s_mb": 4.0,
                "e_val": 0.3,
                "engine_sha256": digests["engine"],
                "artifact_paths": {"engine": f"{label}/engine.plan", "bench": f"{label}/bench.json"},
                "artifact_sha256": digests,
                "quality": {"protocol_stage": "final"},
                "durations_s": {"train": 1.0},
                "gpu_seconds": 1.5,
            }
        )

    def final_test(
        self,
        manifest: dict[str, Any],
        artifact_dir: Path,
        config: Any,
        access: FakeTestAccess,
        work_dir: Path,
        *,
        device: str = "cuda",
    ) -> FakeTestResult:
        engine = Path(artifact_dir) / manifest["artifact_paths"]["engine"]
        digest = hashlib.sha256(engine.read_bytes()).hexdigest()
        if digest != manifest["engine_sha256"]:
            return FakeTestResult(
                {
                    "status": "invalid",
                    "engine_sha256": digest,
                    "test": None,
                    "failure_code": "artifact_changed",
                    "failure_detail": "engine differs from the manifest",
                }
            )
        access.ledger.claim(digest)  # TestAlreadyEvaluated on a repeat
        self.tested.append(digest)
        return FakeTestResult(
            {
                "status": "valid",
                "engine_sha256": digest,
                "test": {
                    "n": 10_000,
                    "e_fraction": 0.4,
                    "errors": 4000,
                    "top1_accuracy": 0.6,
                    "top5_accuracy": 0.85,
                    "per_class": [],
                    "bench_version": "tcc-bench/fake-v4",
                },
            }
        )


def wiring(world: World, fake: FakeFinalEval, root: Path) -> fs.FinalWiring:
    science = type(
        "Science", (), {"final_eval": fake, "TorchvisionCifar100": staticmethod(lambda path: ("cifar", path))}
    )()
    synthetic = world.final
    return dataclasses.replace(
        synthetic,
        evaluator=gw.FinalBuildEvaluator(science, root / "cifar", root / "work", LIST_ID, 1800.0, device="cpu"),
        test_evaluator=gw.FinalTestEvaluator(science, root / "cifar", root / "work", 1800.0, device="cpu"),
    )


def request(**changes: Any) -> EvaluationRequest:
    genome = make_genome(1)
    base = dict(
        run_id="final-rt-b0-abc",
        candidate_id="final-rt-b0-abc.g",
        attempt_id="final-rt-b0-abc.g.a1",
        stage="finalist",
        genome=genome,
        train_seed=777,
        calibration_seed=3,
        dataset_id="d",
        environment_id="e",
        protocol_id="p",
        plan_id="plan",
        artifact_root=Path("ROOT"),
    )
    return EvaluationRequest(**{**base, **changes})  # type: ignore[arg-type]


# -- the conversion ---------------------------------------------------------------------------------------------------------


def test_the_artifact_directory_puts_the_label_where_the_catalog_expects_the_attempt() -> None:
    directory, label = gw.artifact_dir_of(Path("ROOT"), "final-rt-b0-abc", "final-rt-b0-abc.g.a1")
    assert directory == Path("ROOT") / "runs" / "final-rt-b0-abc" / "attempts" and label == "final-rt-b0-abc.g.a1"


def test_a_valid_build_manifest_becomes_a_result_with_medians_artifacts_by_kind_and_the_bench_version() -> None:
    fake = FakeFinalEval()
    manifest = fake.final_build(
        {"n_blocks": 2},
        777,
        FakeConfig(),
        object(),
        Path("w"),
        Path("ROOT/runs/final-rt-b0-abc/attempts"),
        label="final-rt-b0-abc.g.a1",
    ).to_dict()
    result = gw.evaluation_result_of(request(), manifest)
    assert result.status == "valid" and (
        result.fitness.e,
        result.fitness.l_ms,
        result.fitness.v_mb,
        result.fitness.s_mb,
    ) == (0.3, 1.1, 12.0, 4.0)  # type: ignore[union-attr]
    assert result.artifact_paths["engine"] == "runs/final-rt-b0-abc/attempts/final-rt-b0-abc.g.a1/engine.plan"
    assert result.artifact_sha256["engine"] == manifest["engine_sha256"]
    assert (
        result.quality["measurement"]["bench_version"] == "tcc-bench/fake-v4" and result.quality["synthetic"] is False
    )
    assert (
        result.quality["final"]["manifest"]["engine_sha256"] == manifest["engine_sha256"]
        and len(result.quality["final"]["builds"]) == 3
    )


@pytest.mark.parametrize("status", ["invalid", "interrupted"])
def test_a_failed_build_has_no_fitness_and_no_artifacts(status: str) -> None:
    result = gw.evaluation_result_of(
        request(),
        {
            "status": status,
            "failure_stage": "train",
            "failure_code": "oom",
            "failure_detail": "out of memory",
            "gpu_seconds": 3.0,
            "artifact_paths": {"engine": "x"},
        },
    )
    assert (
        result.status == status
        and result.fitness is None
        and result.artifact_paths == {}
        and result.failure_code == "oom"
    )


def test_the_test_metrics_get_the_keys_the_barrier_checks_and_a_failed_pass_records_nothing() -> None:
    ok = gw.metrics_of_test_result(
        {"status": "valid", "engine_sha256": "e" * 64, "test": {"n": 10_000, "top1_accuracy": 0.6}}
    )
    assert ok["n_examples"] == 10_000 and ok["engine_sha256"] == "e" * 64 and ok["top1_accuracy"] == 0.6
    with pytest.raises(fs.FinalStageError, match="no result was recorded"):
        gw.metrics_of_test_result(
            {"status": "interrupted", "engine_sha256": "e" * 64, "test": None, "failure_code": "oom"}
        )


def test_the_build_evaluator_prepares_the_data_once_and_never_takes_a_search_request(tmp_path: Path) -> None:
    fake = FakeFinalEval()
    science = type("Science", (), {"final_eval": fake, "TorchvisionCifar100": staticmethod(lambda path: path)})()
    evaluator = gw.FinalBuildEvaluator(science, tmp_path, tmp_path / "w", LIST_ID, None, device="cpu")
    evaluator.evaluate(request(artifact_root=tmp_path))
    evaluator.evaluate(request(artifact_root=tmp_path, attempt_id="final-rt-b0-abc.g.a2"))
    assert fake.prepared == 1 and fake.grants == [("final-rt-b0-abc", LIST_ID)]
    assert [b["seed"] for b in fake.builds] == [777, 777] and fake.builds[0]["genome"] == make_genome(
        1
    ).expressed_genome
    with pytest.raises(fs.FinalStageError, match="finalist requests"):
        evaluator.evaluate(request(stage="confirmatory"))


# -- the whole phase through the glue -----------------------------------------------------------------------------------------


def test_the_final_phase_runs_through_the_glue_and_the_ledger_contract_holds(tmp_path: Path) -> None:
    fake = FakeFinalEval()
    world = make_world(tmp_path)
    world = dataclasses.replace(world, final=wiring(world, fake, tmp_path))
    build_all(world)
    assert fake.prepared == 1 and len(fake.builds) == 9 and not fake.tested  # nine retrainings, no test touched
    with ReadModel(world.db) as rm:
        unit = fs.units_of(rm.finalist_list(LIST_ID))[0]
        [engine] = [a for a in rm.artifacts(unit.run_id) if a["kind"] == "engine"]
        assert engine["rel_path"].startswith(f"runs/{unit.run_id}/attempts/") and rm.test_evaluations() == []
    open_all(world)
    assert len(fake.tested) == 9 and len(set(fake.tested)) == 9
    with ReadModel(world.db) as rm:
        tests = rm.test_evaluations()
    assert (
        len(tests) == 9
        and {t["metrics"]["top1_accuracy"] for t in tests} == {0.6}
        and tests[0]["metrics"]["n_examples"] == 10_000
    )
    # the ledger the science front sees is the operation's: a second claim of any engine is its TestAlreadyEvaluated
    ledger = fs.MarkLedger(world.db.parent / fs.MARKS_DIR, "x", world.list_sha, lambda: set())
    adapter = gw._LedgerAdapter(ledger, TestAlreadyEvaluated)
    with pytest.raises(TestAlreadyEvaluated):
        adapter.claim(tests[0]["engine_sha256"])
    assert run(world, fs.TEST) == []  # nothing left to open


# -- against the real interface of the science front (no GPU: signatures and the manifest it produces) ----------------------


def test_the_glue_matches_the_real_final_eval_interface() -> None:
    import inspect

    final_eval = pytest.importorskip("cnnlab.tcc.model_pipeline.final_eval")
    build = inspect.signature(final_eval.final_build).parameters
    assert list(build)[:6] == ["genome", "retrain_seed", "config", "data", "work_dir", "artifact_dir"]
    assert {"label", "device"} <= set(build) and build["label"].kind is inspect.Parameter.KEYWORD_ONLY
    test = inspect.signature(final_eval.final_test).parameters
    assert list(test)[:5] == ["manifest", "artifact_dir", "config", "access", "work_dir"] and "device" in test
    assert list(inspect.signature(final_eval.final_grant).parameters)[:3] == ["source", "run_id", "finalist_list_id"]
    assert list(inspect.signature(final_eval.prepare_final_data).parameters) == ["grant", "source", "device"]
    assert {"train_time_limit_s"} <= set(final_eval.FinalConfig.__dataclass_fields__)
    assert list(final_eval.TestAccess.__dataclass_fields__) == ["grant", "source", "ledger"]
    assert issubclass(final_eval.TestAlreadyEvaluated, Exception)
    # the ledger contract: claim raises its TestAlreadyEvaluated, release gives back; the adapter speaks it
    ledger = fs.MarkLedger(Path("unused"), "r", "l" * 64, lambda: {"a" * 64})
    with pytest.raises(final_eval.TestAlreadyEvaluated):
        gw._LedgerAdapter(ledger, final_eval.TestAlreadyEvaluated).claim("a" * 64)


def test_a_real_build_manifest_is_converted_without_loss() -> None:
    final_eval = pytest.importorskip("cnnlab.tcc.model_pipeline.final_eval")
    builds = tuple(
        final_eval.FinalBuild(
            index=i,
            plan_sha256="b" * 64,
            plan_bytes=1000 + i,
            s_mb=4.0,
            l_ms=1.0 + 0.1 * i,
            v_mb=12.0,
            e_val=0.3,
            top5_val=0.1,
            precision_ok=True,
            build_s=2.0,
            bench_version="tcc-bench/preliminary-v4",
        )
        for i in range(3)
    )
    built = final_eval.FinalBuildResult(
        status="valid",
        retrain_seed=777,
        label="final-rt-b0-abc.g.a1",
        builds=builds,
        chosen_build=1,
        l_ms=1.1,
        v_mb=12.0,
        s_mb=4.0,
        e_val=0.3,
        engine_sha256="e" * 64,
        artifact_paths={"engine": "final-rt-b0-abc.g.a1/engine.plan"},
        artifact_sha256={"engine": "e" * 64},
        quality={"protocol_stage": "final"},
        durations_s={"train": 1.0},
        gpu_seconds=2.0,
    )
    result = gw.evaluation_result_of(request(), built.to_dict())
    assert result.status == "valid" and result.fitness.l_ms == 1.1 and result.fitness.e == 0.3  # type: ignore[union-attr]
    assert result.artifact_paths["engine"] == "runs/final-rt-b0-abc/attempts/final-rt-b0-abc.g.a1/engine.plan"
    assert result.quality["measurement"]["bench_version"] == "tcc-bench/preliminary-v4"
    assert (
        result.quality["final"]["manifest"]["engine_sha256"] == "e" * 64 and len(result.quality["final"]["builds"]) == 3
    )
    failed = final_eval.FinalBuildResult(
        status="invalid",
        retrain_seed=1,
        label="x",
        failure_stage="train",
        failure_code="oom",
        failure_detail="out of memory",
    )
    assert gw.evaluation_result_of(request(), failed.to_dict()).fitness is None
    done = final_eval.FinalTestResult(
        status="valid", engine_sha256="e" * 64, test={"n": 10_000, "top1_accuracy": 0.6, "e_fraction": 0.4}
    )
    assert gw.metrics_of_test_result(done.to_dict())["n_examples"] == 10_000


def test_the_retraining_limit_is_the_plans_limit_scaled_by_the_epochs() -> None:
    # Amendment of 2026-10-24: the pilot's 4140 s were for a 5-epoch search training; a finalist trains 30 epochs.
    assert gw.final_train_limit(4140.0, 5, 30) == pytest.approx(24840.0)
    assert gw.final_train_limit(None, 5, 30) is None  # no limit in the plan, none in the retraining
    with pytest.raises(fs.FinalStageError):
        gw.final_train_limit(4140.0, None, 30)  # never guess the ratio
