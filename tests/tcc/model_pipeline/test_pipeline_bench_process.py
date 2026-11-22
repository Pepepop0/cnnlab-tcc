"""Isolated-benchmark process handling on CPU: failure kinds and what they mean for a candidate.

No GPU is used. One test starts the real child with ``CUDA_VISIBLE_DEVICES``
empty, so CUDA initialization fails inside the child and the error path of
the process protocol is exercised end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_pipeline_training import cpu_data, fake_build, request_for  # same directory

from cnnlab.tcc.model_pipeline import bench
from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.bench import BenchProcessError, child_process_kwargs, run_isolated, started_marker
from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator, EvaluatorConfig
from cnnlab.tcc.model_pipeline.trainer import TrainConfig

DLL_INIT_FAILED = 0xC0000142  # observed on 28/09/2026 when the parent's console host had died
ACCESS_VIOLATION = 0xC0000005


class Completed:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode, self.stderr, self.stdout = returncode, stderr, ""


def fake_run(returncode: int, *, marker: bool, report: dict[str, Any] | None, stderr: str = "") -> Any:
    def run(cmd: list[str], **_: Any) -> Completed:
        out = Path(cmd[cmd.index("--out") + 1])
        if marker:
            started_marker(out).write_text("started", encoding="ascii")
        if report is not None:
            out.write_text(json.dumps(report), encoding="utf-8")
        return Completed(returncode, stderr)

    return run


def test_child_gets_its_own_console_on_windows() -> None:
    kwargs = child_process_kwargs()
    assert kwargs["stdin"] is subprocess.DEVNULL
    if sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW


@pytest.mark.parametrize(
    ("returncode", "marker", "report", "kind"),
    [
        (DLL_INIT_FAILED, False, None, "spawn"),
        (ACCESS_VIOLATION, True, None, "crash"),
        (1, True, {"error": {"type": "RuntimeError", "message": "CUDA out of memory", "trace": "tb"}}, "measurement"),
        (0, True, None, "crash"),  # exit 0 without a report is not a measurement
    ],
)
def test_failure_kinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    marker: bool,
    report: dict[str, Any] | None,
    kind: str,
) -> None:
    monkeypatch.setattr(bench.subprocess, "run", fake_run(returncode, marker=marker, report=report))
    with pytest.raises(BenchProcessError) as err:
        run_isolated(tmp_path / "engine.plan", tmp_path / "bench.json")
    assert err.value.kind == kind and err.value.exit_code == returncode
    if kind == "spawn":
        assert "0xC0000142" in str(err.value)
    if kind == "measurement":
        assert "CUDA out of memory" in err.value.detail


def test_timeout_is_spawn_or_timeout_depending_on_the_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def hang(started: bool) -> Any:
        def run(cmd: list[str], **_: Any) -> Completed:
            if started:
                started_marker(Path(cmd[cmd.index("--out") + 1])).write_text("started", encoding="ascii")
            raise subprocess.TimeoutExpired(cmd, 1.0)

        return run

    for started, kind in ((False, "spawn"), (True, "timeout")):
        monkeypatch.setattr(bench.subprocess, "run", hang(started))
        with pytest.raises(BenchProcessError) as err:
            run_isolated(tmp_path / "engine.plan", tmp_path / "bench.json", timeout_s=1.0)
        assert err.value.kind == kind


def test_stale_report_of_an_earlier_try_is_never_returned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    out.write_text(json.dumps({"latency": {"median_ms": 0.1}}), encoding="utf-8")
    started_marker(out).write_text("started", encoding="ascii")
    monkeypatch.setattr(bench.subprocess, "run", fake_run(DLL_INIT_FAILED, marker=False, report=None))
    with pytest.raises(BenchProcessError) as err:
        run_isolated(tmp_path / "engine.plan", out)
    assert err.value.kind == "spawn"


def test_real_child_reports_a_measurement_error_without_a_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real child process starts (own console), fails to reach CUDA and says so in its report."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    plan = tmp_path / "engine.plan"
    plan.write_bytes(b"not an engine")
    with pytest.raises(BenchProcessError) as err:
        run_isolated(plan, tmp_path / "bench.json", warmup=1, iters=1, timeout_s=300.0)
    assert err.value.kind == "measurement", str(err.value)
    assert err.value.exit_code == 1
    assert started_marker(tmp_path / "bench.json").exists()


# --- what the evaluator makes of each kind ---------------------------------------------------------


def evaluator_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failures: list[BenchProcessError], gpu_guard: Any = None
) -> tuple[EngineEvaluator, Any, list[int]]:
    """CPU evaluator whose benchmark fails with ``failures`` in order, then succeeds (synthetic report)."""
    calls: list[int] = []

    def flaky(plan_path: Path, out_path: Path, **_: Any) -> dict[str, Any]:
        calls.append(len(calls))
        if failures:
            raise failures.pop(0)
        return {
            "bench_version": "synthetic",
            "latency": {"median_ms": 0.25},
            "memory": {"v_bytes_allocator_peak": 2_500_000},
            "e": {"e_fraction": 0.75, "errors": 12, "n": 16, "all_finite": True},
            "quality": {},
        }

    monkeypatch.setattr(evaluator_mod, "build_engine", fake_build)
    monkeypatch.setattr(evaluator_mod, "run_isolated", flaky)
    config = EvaluatorConfig(
        train=TrainConfig(epochs=1, batch_size=16), bench_retry_wait_s=0.0, gpu_wait_s=0.3, gpu_poll_s=0.05
    )
    data = cpu_data()
    evaluator = EngineEvaluator(
        data, tmp_path / "val.npz", config, device="cpu", work_root=tmp_path / "work", gpu_guard=gpu_guard
    )
    return evaluator, request_for(3, tmp_path, config, data), calls


class ScriptedGuard:
    """Answers of the GPU guard, one per call; the last answer repeats."""

    def __init__(self, *answers: int) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self) -> list[dict[str, str]]:
        n = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return [{"pid": str(100 + i), "process_name": "python.exe"} for i in range(n)]


def test_gpu_guard_ignores_processes_that_already_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import subprocess

    from cnnlab.tcc.model_pipeline import environment

    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    assert evaluator_mod.process_is_alive(os.getpid())
    assert not evaluator_mod.process_is_alive(ended.pid)
    apps = [
        {"pid": str(ended.pid), "process_name": "python.exe"},
        {"pid": str(os.getpid()), "process_name": "python.exe"},
        {"pid": str(os.getppid()), "process_name": "python.exe"},
        {"pid": "4", "process_name": "explorer.exe"},
    ]
    monkeypatch.setattr(environment, "nvidia_smi_snapshot", lambda: {"available": True, "compute_apps": apps})
    assert [job["pid"] for job in evaluator_mod.other_gpu_python_jobs()] == [str(os.getppid())]


def test_evaluation_waits_until_the_gpu_is_exclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    guard = ScriptedGuard(1, 1, 0)  # busy twice before training, free afterwards
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, [], gpu_guard=guard)
    result = evaluator.evaluate(request)
    assert result.status == "valid" and len(calls) == 1
    assert result.quality["gpu_waits"][0]["before"] == "guard"
    assert result.quality["gpu_exclusive_at_benchmark"] is True


def test_measurement_with_an_intruder_is_discarded_and_repeated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # guard before training, before benchmark 1, after benchmark 1 (intruder), before benchmark 2, after it
    guard = ScriptedGuard(0, 0, 1, 0, 0)
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, [], gpu_guard=guard)
    result = evaluator.evaluate(request)
    assert result.status == "valid" and len(calls) == 2
    assert [t["kind"] for t in result.quality["benchmark_tries"]] == ["shared_gpu", "ok"]


def test_gpu_shared_during_every_measurement_is_interrupted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class AfterOnly:
        """Free whenever asked before a measurement, occupied right after each one."""

        def __init__(self, calls: list[int]) -> None:
            self.calls, self.seen = calls, 0

        def __call__(self) -> list[dict[str, str]]:
            intruder = len(self.calls) > self.seen
            self.seen = len(self.calls)
            return [{"pid": "9", "process_name": "python.exe"}] if intruder else []

    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, [])
    evaluator.gpu_guard = AfterOnly(calls)
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "benchmark",
        "infrastructure_gpu_shared",
    )
    assert len(calls) == 3 and result.fitness is None


def spawn_error() -> BenchProcessError:
    return BenchProcessError("spawn", DLL_INIT_FAILED, "child never reached Python code")


def test_spawn_failure_is_retried_and_then_measured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, [spawn_error(), spawn_error()])
    result = evaluator.evaluate(request)
    assert result.status == "valid" and len(calls) == 3
    assert [t["kind"] for t in result.quality["benchmark_tries"]] == ["spawn", "spawn", "ok"]


def test_persistent_spawn_failure_is_interrupted_not_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, [spawn_error() for _ in range(3)])
    result = evaluator.evaluate(request)
    assert result.status == "interrupted" and result.fitness is None
    assert (result.failure_stage, result.failure_code) == ("benchmark", "infrastructure_spawn_failed")
    assert "0xC0000142" in (result.failure_detail or "")
    assert len(calls) == 3 and result.gpu_seconds > 0
    assert not result.artifact_paths
    assert "train" in result.durations_s and "benchmark" in result.durations_s


def test_repeated_crash_while_measuring_is_a_candidate_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    crash = [BenchProcessError("crash", ACCESS_VIOLATION, "child died without a report") for _ in range(2)]
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, crash)
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_code) == ("invalid", "benchmark_process_crashed")
    assert len(calls) == 2 and "0xC0000005" in (result.failure_detail or "")


@pytest.mark.parametrize(
    ("kind", "code", "tries"),
    [("timeout", "benchmark_timeout", 1), ("measurement", "benchmark_failed", 2)],  # a measurement error gets one retry
)
def test_timeout_and_repeated_measurement_errors_are_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, code: str, tries: int
) -> None:
    failures = [BenchProcessError(kind, 1, "detail") for _ in range(tries)]  # type: ignore[arg-type]
    evaluator, request, calls = evaluator_with(monkeypatch, tmp_path, failures)
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "benchmark", code)
    assert len(calls) == tries


def test_classification_summary_on_a_known_case() -> None:
    """3 classes, 6 images: hand-computed accuracy, top-5 and macro precision, recall, F1."""
    import torch

    from cnnlab.tcc.model_pipeline.bench import classification_summary

    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = torch.tensor([0, 1, 1, 1, 0, 2])
    in_top5 = torch.tensor([True, True, True, True, False, True])
    out = classification_summary(preds, labels, in_top5, 3)
    assert out["top1_accuracy"] == 4 / 6
    assert out["top5_accuracy"] == 5 / 6
    precision = [1 / 2, 2 / 3, 1.0]  # class 0: 1 of 2 predicted; class 1: 2 of 3; class 2: 1 of 1
    recall = [1 / 2, 1.0, 1 / 2]
    f1 = [2 * p * r / (p + r) for p, r in zip(precision, recall)]
    assert abs(out["macro_precision"] - sum(precision) / 3) < 1e-12
    assert abs(out["macro_recall"] - sum(recall) / 3) < 1e-12
    assert abs(out["macro_f1"] - sum(f1) / 3) < 1e-12
    assert out["per_class"] == {"support": [2, 2, 2], "predicted": [2, 3, 1], "true_positive": [1, 2, 1]}


def test_classification_summary_counts_a_never_predicted_class_as_zero() -> None:
    import torch

    from cnnlab.tcc.model_pipeline.bench import classification_summary

    labels = torch.tensor([0, 1])
    preds = torch.tensor([0, 0])
    out = classification_summary(preds, labels, torch.tensor([True, True]), 2)
    assert out["macro_precision"] == (0.5 + 0.0) / 2
    assert out["macro_recall"] == (1.0 + 0.0) / 2
