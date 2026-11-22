"""No exception leaves the evaluator: every failure is an invalid or an interrupted attempt (CPU).

Cross-review finding A2 (29/09/2026): an exception outside the anticipated
ones ended the whole job and, when deterministic, made every resume repeat
the same candidate. Engine build and benchmark are stand-ins here; the
numbers are not measurements.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_training import cpu_data, fake_bench, fake_build, request_for  # same directory

from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.bench import BenchProcessError
from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator, EvaluatorConfig
from cnnlab.tcc.model_pipeline.trainer import TrainConfig


def make(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kw: Any) -> tuple[EngineEvaluator, Any]:
    monkeypatch.setattr(evaluator_mod, "build_engine", fake_build)
    monkeypatch.setattr(evaluator_mod, "run_isolated", fake_bench)
    config = EvaluatorConfig(
        train=TrainConfig(epochs=1, batch_size=16), bench_retry_wait_s=0.0, gpu_poll_s=0.01, gpu_wait_s=0.2
    )
    data = cpu_data()
    evaluator = EngineEvaluator(data, tmp_path / "val.npz", config, device="cpu", work_root=tmp_path / "work", **kw)
    return evaluator, request_for(3, tmp_path, config, data)


def raiser(exc: BaseException) -> Any:
    def call(*_: Any, **__: Any) -> Any:
        raise exc

    return call


STAGES = [
    ("build_model", "build"),
    ("train_model", "train"),
    ("evaluate_accuracy", "reference"),
    ("fold_batchnorm", "reference"),
    ("typed_network", "export"),
    ("export_network", "export"),
    ("build_engine", "trt_build"),
    ("check_engine_precision", "precision_check"),
    ("run_isolated", "benchmark"),
]


@pytest.mark.parametrize(("function", "stage"), STAGES)
def test_unexpected_error_on_a_healthy_device_is_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, function: str, stage: str
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, function, raiser(RuntimeError("CUDA error: device-side assert triggered")))
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", stage, "unexpected_error")
    assert result.fitness is None and not result.artifact_paths
    assert "RuntimeError: CUDA error" in (result.failure_detail or "")
    assert result.quality["unexpected_exception"]["device_health"] == "cpu device"
    assert not (tmp_path / "work" / request.run_id / request.attempt_id).exists()


@pytest.mark.parametrize(("function", "stage"), STAGES)
def test_out_of_memory_anywhere_is_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, function: str, stage: str
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, function, raiser(torch.cuda.OutOfMemoryError("CUDA out of memory")))
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", stage, "out_of_memory")


@pytest.mark.parametrize(("function", "stage"), STAGES[:3] + STAGES[-1:])
def test_unexpected_error_on_a_broken_device_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, function: str, stage: str
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, function, raiser(RuntimeError("CUDA error: unknown error")))
    monkeypatch.setattr(evaluator, "_gpu_healthy", lambda: (False, "RuntimeError: GPU is lost"))
    result = evaluator.evaluate(request)
    assert result.status == "interrupted" and result.failure_stage == stage
    assert result.failure_code == "infrastructure_device_unhealthy"
    assert "GPU is lost" in (result.failure_detail or "")


def test_model_moved_to_the_device_out_of_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The case named in the review: ``model.to(device)`` right after the build."""
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.nn.Module, "to", raiser(torch.cuda.OutOfMemoryError("CUDA out of memory")))
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "build", "out_of_memory")


def test_disk_error_while_publishing_is_infrastructure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, "write_atomic", raiser(OSError(28, "No space left on device")))
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "publish",
        "infrastructure_host_error",
    )
    assert result.fitness is None and not result.artifact_paths


def test_artifact_that_reads_back_differently_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, "sha256_file", lambda path: "0" * 64)
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "publish",
        "infrastructure_artifact_write_mismatch",
    )


def test_scratch_directory_that_cannot_be_created_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    (tmp_path / "work").write_text("a file where the scratch directory should be", encoding="utf-8")
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "guard",
        "infrastructure_host_error",
    )


def test_operator_interrupt_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator_mod, "train_model", raiser(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        evaluator.evaluate(request)


# --- benchmark failures (finding C3) ---------------------------------------------------------------


def bench_sequence(monkeypatch: pytest.MonkeyPatch, failures: list[BenchProcessError]) -> list[int]:
    calls: list[int] = []

    def flaky(plan_path: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        calls.append(len(calls))
        if failures:
            raise failures.pop(0)
        return fake_bench(plan_path, out_path, **kw)

    monkeypatch.setattr(evaluator_mod, "run_isolated", flaky)
    return calls


def measurement(detail: str = "RuntimeError | engine deserialization failed | tb") -> BenchProcessError:
    return BenchProcessError("measurement", 1, detail)


def test_measurement_error_is_retried_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    calls = bench_sequence(monkeypatch, [measurement()])
    result = evaluator.evaluate(request)
    assert result.status == "valid" and len(calls) == 2
    assert [t["kind"] for t in result.quality["benchmark_tries"]] == ["measurement", "ok"]


def test_repeated_measurement_error_on_a_healthy_device_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    calls = bench_sequence(monkeypatch, [measurement(), measurement()])
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_code) == ("invalid", "benchmark_failed") and len(calls) == 2
    assert result.quality["benchmark_tries"][-1]["device_health"] == "cpu device"


@pytest.mark.parametrize("kind", ["measurement", "timeout", "crash"])
def test_benchmark_failure_on_a_broken_device_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    bench_sequence(monkeypatch, [BenchProcessError(kind, 1, "detail") for _ in range(3)])  # type: ignore[arg-type]
    monkeypatch.setattr(evaluator, "_gpu_healthy", lambda: (False, "device lost"))
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "benchmark",
        "infrastructure_device_unhealthy",
    )


def test_out_of_memory_in_the_benchmark_child_is_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    oom = "OutOfMemoryError | CUDA out of memory. Tried to allocate 2.00 GiB | tb"
    bench_sequence(monkeypatch, [measurement(oom), measurement(oom)])
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "benchmark", "out_of_memory")


# --- what counts as time of the attempt (finding C2) ---------------------------------------------------


def test_waiting_for_the_gpu_is_not_time_of_the_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answers = [[{"pid": "1", "process_name": "python.exe"}]] * 8

    def guard() -> list[dict[str, str]]:
        time.sleep(0.05)
        return answers.pop(0) if answers else []

    evaluator, request = make(monkeypatch, tmp_path, gpu_guard=guard)
    evaluator.config = dataclasses.replace(evaluator.config, gpu_wait_s=30.0)
    started = time.monotonic()
    result = evaluator.evaluate(request)
    elapsed = time.monotonic() - started
    excluded = result.quality["excluded_from_gpu_seconds"]
    assert result.status == "valid"
    assert excluded["gpu_wait"] >= 0.4 and excluded["session_prepare"] > 0
    assert result.gpu_seconds == pytest.approx(elapsed - excluded["gpu_wait"] - excluded["session_prepare"], abs=0.05)
    assert result.quality["gpu_waits"][0]["before"] == "guard"


def test_session_setup_is_charged_to_no_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    first = evaluator.evaluate(request)
    second = evaluator.evaluate(request)
    assert first.quality["excluded_from_gpu_seconds"]["session_prepare"] > 0
    assert second.quality["excluded_from_gpu_seconds"]["session_prepare"] < 0.01


def test_gpu_that_stays_busy_is_infrastructure_and_costs_the_attempt_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator, request = make(monkeypatch, tmp_path, gpu_guard=lambda: [{"pid": "1", "process_name": "python.exe"}])
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "guard",
        "infrastructure_gpu_busy",
    )
    assert result.gpu_seconds < 0.1 and result.quality["excluded_from_gpu_seconds"]["gpu_wait"] >= 0.2
