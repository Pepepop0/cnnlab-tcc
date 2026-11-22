"""The science wiring run by the operation front's demo service, on CPU with stand-in engine and benchmark.

Skipped until ``cnnlab.tcc.operation.demo`` is on the branch (it belongs
to the operation front). The factory below is the adapter:
it is the only place where operation and science types meet.
Training and ONNX export are real (CPU, synthetic images); TensorRT build and
benchmark are stand-ins, so the numbers are not measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

demo = pytest.importorskip("cnnlab.tcc.operation.demo")

from test_pipeline_training import FakeTrainFile  # noqa: E402  (same directory)
from test_pipeline_verify_run import standin_bench, standin_build  # noqa: E402

from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod  # noqa: E402
from cnnlab.tcc.model_pipeline import pilot_wiring  # noqa: E402
from cnnlab.tcc.model_pipeline.verify_run import verify  # noqa: E402
from cnnlab.tcc.storage.sqlite_store import SplitDescriptor, SQLiteRunStore  # noqa: E402


def demo_wiring(spec: Any) -> Any:
    """Adapter between the operation and science layers."""
    science = pilot_wiring.build_pilot_science(spec.factory_args)
    return demo.DemoWiring(
        runner=science.runner,
        evaluator=science.evaluator,
        dataset=demo.DatasetRegistration(
            science.dataset_id,
            science.dataset_manifest,
            {name: SplitDescriptor(**descriptor) for name, descriptor in science.splits.items()},
        ),
        space_hash=science.space_hash,
        runner_label=science.runner_label,
        evaluator_label=science.evaluator_label,
        synthetic=False,
        environment=science.environment,
    )


@pytest.fixture
def cpu_science(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.setattr(pilot_wiring, "TorchvisionCifar100", lambda root: FakeTrainFile())
    monkeypatch.setattr(evaluator_mod, "build_engine", standin_build)
    monkeypatch.setattr(evaluator_mod, "run_isolated", standin_bench)
    monkeypatch.setenv("CNNLAB_TCC_CIFAR_ROOT", str(tmp_path / "unused-dataset-dir"))
    monkeypatch.setenv("CNNLAB_TCC_WORK_DIR", str(tmp_path / "work"))
    return {
        "train_per_class": 1,
        "epochs": 1,
        "population": 2,
        "generations": 1,
        "device": "cpu",
        "gpu_guard": False,
        "code_version": "test-code",
    }


def spec_for(factory_args: dict[str, Any], **over: Any) -> Any:
    base = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 1001,
        "train_seed": 2001,
        "calibration_seed": 0,
        "budget_gpu_s": 600.0,
        "code_label": "wiring-test",
        "author": "gauss",
        "factory_args": factory_args,
    }
    return demo.DemoRunSpec(**{**base, **over})


def test_operation_service_runs_the_science_wiring(cpu_science: dict[str, Any], tmp_path: Path) -> None:
    db, root = tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"
    db.parent.mkdir()
    result = demo.start_demo(db, root, spec_for(cpu_science), demo_wiring, worker_id="gauss-test", heartbeat_s=0.5)
    assert result.outcome is not None and result.outcome.end_state == "done", result.message

    with SQLiteRunStore(db, root) as store:
        attempts = store.list_attempts(result.run_id)
        assert [a.status for a in attempts] == ["valid"] * 4
        for attempt in attempts:
            quality = attempt.result.quality  # type: ignore[union-attr]
            assert quality["preliminary"] and quality["not_confirmatory"] and "synthetic" not in quality
            assert set(quality["operation"]) == {"metered_gpu_s", "evaluator_reported_gpu_s"}
        assert "test" not in store.dataset_splits_for_run(result.run_id)
        plan = store.get_plan(store.get_run(result.run_id).plan_id)  # type: ignore[union-attr]
        assert plan.content["confirmatory"] is False and plan.content["synthetic_evaluator"] is False
        assert str(tmp_path) not in str(plan.content)  # machine paths came from the environment
        assert store.check_integrity(hash_files=True).ok

    report = verify(db, root, result.run_id)
    assert report["ok"], report["problems"]
    assert report["counts"]["valid"] == 4 and report["dataset"]["official_test_read"] is False
    assert (
        report["system"]["operation_system"] and "not the confirmatory protocol" in report["system"]["operation_system"]
    )
    assert report["system"]["scope"] == list(pilot_wiring.SCOPE_LABELS)


def test_only_the_joint_method_is_wired(cpu_science: dict[str, Any], tmp_path: Path) -> None:
    """mono/random exist in the runner, but the pilot labels and plan describe the joint search."""
    science = pilot_wiring.build_pilot_science(cpu_science)
    assert "nsga2-joint" in science.runner_label


def test_a_busy_gpu_fails_the_job_without_inventing_results(
    cpu_science: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot_wiring, "other_gpu_python_jobs", lambda: [{"pid": "1", "process_name": "python.exe"}])
    db, root = tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"
    db.parent.mkdir()
    result = demo.start_demo(
        db,
        root,
        spec_for({**cpu_science, "gpu_guard": True, "gpu_wait_s": 0}),
        demo_wiring,
        worker_id="gauss-test",
        heartbeat_s=0.5,
    )
    assert result.outcome is not None and result.outcome.end_state == "failed"
    assert "SearchBlockedError" in result.outcome.detail
    with SQLiteRunStore(db, root) as store:
        attempts = store.list_attempts(result.run_id)
        assert [a.status for a in attempts] == ["interrupted", "interrupted"]
        assert {a.result.failure_code for a in attempts} == {"infrastructure_gpu_busy"}  # type: ignore[union-attr]
        assert store.unique_valid_evaluations(result.run_id) == 0
        assert store.load_checkpoint(result.run_id) is not None
