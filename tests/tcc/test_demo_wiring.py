"""The wiring adapter must preserve scientific identity and the fake guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cnnlab.tcc.demo_wiring import demo_wiring
from cnnlab.tcc.operation.demo import DemoRunSpec
from cnnlab.tcc.storage.sqlite_store import SplitDescriptor


def _spec(method: str = "joint") -> DemoRunSpec:
    return DemoRunSpec(
        method=method,
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 4},
    )


def test_factory_maps_science_without_inventing_measurements() -> None:
    runner, evaluator = object(), object()
    science = SimpleNamespace(
        runner=runner,
        evaluator=evaluator,
        dataset_id="data-1",
        dataset_manifest={"scope": "pilot"},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "subset"}},
        space_hash="space-1",
        runner_label="nsga2",
        evaluator_label="engine-fp32",
        environment={"backend": "TensorRT"},
        synthetic=False,
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science) as build:
        wiring = demo_wiring(_spec())
    build.assert_called_once_with({"population": 4})
    assert wiring.runner is runner
    assert wiring.evaluator is evaluator
    assert wiring.dataset.splits["train"] == SplitDescriptor("a" * 64, 10, "subset")
    assert wiring.synthetic is False
    assert wiring.environment == {"backend": "TensorRT"}


@pytest.mark.parametrize(
    ("method", "label"), [("joint", "nsga2-joint"), ("mono", "nsga2-mono"), ("random", "random-search")]
)
def test_factory_accepts_joint_mono_and_random_and_labels_the_runner(method: str, label: str) -> None:
    science = SimpleNamespace(
        runner=object(),
        evaluator=object(),
        dataset_id="data-1",
        dataset_manifest={},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "subset"}},
        space_hash="space-1",
        runner_label="nsga2-joint/tcc-search-runner/v2 pop=4 gen=2 slice=fp32",
        evaluator_label="engine-fp32",
        environment={},
        synthetic=False,
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science):
        wiring = demo_wiring(_spec(method))
    assert wiring.runner_label == f"{label}/tcc-search-runner/v2 pop=4 gen=2 slice=fp32"
    assert wiring.runner is science.runner and wiring.synthetic is False  # same runner: it reads the job's method


def test_reference_point_in_the_spec_reaches_the_search_runner_and_not_the_science_arguments() -> None:
    from cnnlab.tcc.science.operators import FP32_SLICE
    from cnnlab.tcc.science.search import SearchConfig, SearchRunner

    def context_for(job: object) -> object:
        return object()

    config = SearchConfig(population=30, generations=None)
    science = SimpleNamespace(
        runner=object(),
        evaluator=object(),
        dataset_id="data-1",
        dataset_manifest={},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "full"}},
        space_hash="space-1",
        runner_label="nsga2-joint/x",
        evaluator_label="engine-fp32",
        environment={},
        synthetic=False,
        search_config=config,
        slice_name="fp32",
        context_for=context_for,
    )
    spec = DemoRunSpec(
        method="mono",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 30, "generations": None, "reference_point": [1.0, 2.5, 12.5, 5.0]},
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science) as build:
        wiring = demo_wiring(spec)
    build.assert_called_once_with({"population": 30, "generations": None})  # the science front never sees it
    assert isinstance(wiring.runner, SearchRunner) and wiring.runner is not science.runner
    assert wiring.runner.config.reference_point == (1.0, 2.5, 12.5, 5.0)
    assert (wiring.runner.config.population, wiring.runner.config.generations) == (30, None)  # the rest is kept
    assert wiring.runner.space is FP32_SLICE and wiring.runner.context_for is context_for
    assert dict(spec.factory_args)["reference_point"] == [1.0, 2.5, 12.5, 5.0]  # the plan keeps it, with the rest


def _sequential_spec(**args: object) -> DemoRunSpec:
    return DemoRunSpec(
        method="sequential",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=19_800.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 30, "slice": "fp16_block", **args},
    )


def test_factory_builds_the_two_stage_runner_for_the_sequential_method() -> None:
    from cnnlab.tcc.science.operators import FP16_BLOCK_SLICE, FP32_SLICE, SLICES
    from cnnlab.tcc.science.search import SearchConfig
    from cnnlab.tcc.science.sequential import SequentialRunner

    def context_for(job: object) -> object:
        return object()

    science = SimpleNamespace(
        runner=object(),
        evaluator=object(),
        dataset_id="data-1",
        dataset_manifest={},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "full"}},
        space_hash="space-1",
        runner_label="nsga2-joint/tcc-search-runner/v2 pop=30 gen=unlimited slice=fp16_block",
        evaluator_label="e",
        environment={},
        synthetic=False,
        search_config=SearchConfig(population=30, generations=None),
        slice_name="fp16_block",
        context_for=context_for,
    )
    spec = _sequential_spec(n_frozen=5, stage_gpu_s=[9000, 9900], reference_point=[1.0, 2.5, 12.5, 5.0])
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science) as build:
        wiring = demo_wiring(spec)
    # the science front sees neither the sequential settings nor the reference point; generations is null
    build.assert_called_once_with({"population": 30, "slice": "fp16_block", "generations": None})
    runner = wiring.runner
    assert isinstance(runner, SequentialRunner)
    assert (runner.seq.n_frozen, runner.seq.stage_gpu_s) == (5, (9000.0, 9900.0))
    assert runner.space is FP16_BLOCK_SLICE and runner.ops is SLICES["architecture"]  # slice, G7 architecture slice
    assert runner.config.reference_point == (1.0, 2.5, 12.5, 5.0) and runner.config.generations is None
    assert wiring.runner_label.startswith("sequential-2stage/") and wiring.synthetic is False

    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science):
        defaults = demo_wiring(_sequential_spec()).runner
    assert (defaults.seq.n_frozen, defaults.seq.stage_gpu_s) == (7, None)  # 7 architectures; half of the budget each


def _science() -> SimpleNamespace:
    return SimpleNamespace(
        runner=object(),
        evaluator=object(),
        dataset_id="d",
        dataset_manifest={},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "full"}},
        space_hash="s",
        runner_label="nsga2-joint/x",
        evaluator_label="e",
        environment={},
        synthetic=False,
    )


def test_the_plan_arguments_of_the_sentinel_become_hooks_and_never_reach_the_science_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    calls: list[tuple[str, tuple, dict]] = []
    fake = SimpleNamespace(
        measure_sentinel=lambda work_dir: calls.append(("measure", (work_dir,), {})) or {"l_ms": 1.0},
        drift=lambda ref, cur, *, limit: calls.append(("drift", (ref, cur), {"limit": limit})) or {"drifted": False},
    )
    monkeypatch.setitem(sys.modules, "cnnlab.tcc.model_pipeline.sentinel", fake)
    import cnnlab.tcc.model_pipeline as pipeline

    monkeypatch.setattr(pipeline, "sentinel", fake, raising=False)  # `from pkg import mod` looks at the package first
    spec = DemoRunSpec(
        method="joint",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 4, "sentinel_every": 50, "sentinel_drift_limit": 0.08, "work_dir": str(tmp_path)},
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=_science()) as build:
        wiring = demo_wiring(spec)
    build.assert_called_once_with({"population": 4, "work_dir": str(tmp_path)})  # the science front never sees them
    assert wiring.sentinel is not None and wiring.sentinel.every == 50
    assert wiring.sentinel.measure() == {"l_ms": 1.0} and wiring.sentinel.drift({"l_ms": 1.0}, {"l_ms": 2.0}) == {
        "drifted": False
    }
    assert calls == [("measure", (tmp_path,), {}), ("drift", ({"l_ms": 1.0}, {"l_ms": 2.0}), {"limit": 0.08})]

    plain = DemoRunSpec(
        method="joint",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 4},
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=_science()):
        assert demo_wiring(plain).sentinel is None  # a pilot run asks for none

    monkeypatch.delenv("CNNLAB_TCC_WORK_DIR", raising=False)
    no_dir = DemoRunSpec(
        method="joint",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 4, "sentinel_every": 50},
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=_science()):
        with pytest.raises(ValueError, match="work directory"):
            demo_wiring(no_dir)
    monkeypatch.setenv("CNNLAB_TCC_WORK_DIR", str(tmp_path / "from-env"))
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=_science()):
        assert demo_wiring(no_dir).sentinel.measure()  # type: ignore[union-attr]  # the environment is enough
    assert calls[-1] == ("measure", (tmp_path / "from-env",), {})


def test_the_training_time_limit_reaches_the_science_wiring_as_an_argument() -> None:
    science = SimpleNamespace(
        runner=object(),
        evaluator=object(),
        dataset_id="d",
        dataset_manifest={},
        splits={"train": {"sha256": "a" * 64, "n_examples": 10, "ref": "full"}},
        space_hash="s",
        runner_label="nsga2-joint/x",
        evaluator_label="e",
        environment={},
        synthetic=False,
    )
    spec = DemoRunSpec(
        method="joint",
        seed_block=0,
        search_seed=1,
        train_seed=2,
        calibration_seed=3,
        budget_gpu_s=60.0,
        code_label="pilot",
        author="test",
        factory_args={"population": 4, "train_time_limit_s": 1800.0},
    )
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science", return_value=science) as build:
        demo_wiring(spec)
    build.assert_called_once_with({"population": 4, "train_time_limit_s": 1800.0})  # the evaluator's key, untouched


def test_factory_refuses_a_sequential_run_that_could_not_work_before_loading_the_dataset() -> None:
    with patch("cnnlab.tcc.demo_wiring.build_pilot_science") as build:
        with pytest.raises(ValueError, match="free representation"):
            demo_wiring(_spec("sequential"))  # the default slice, fp32, pins every representation gene
        with pytest.raises(ValueError, match="generations must be null"):
            demo_wiring(_sequential_spec(generations=3))
    build.assert_not_called()
