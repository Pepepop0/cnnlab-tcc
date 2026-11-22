"""GPU budget accounting (CPU): training pauses are excluded and the VRAM cap is applied once.

Origin: an exploratory run of 01/10/2026 charged 17.4 h to one candidate whose training
did not fit in VRAM; the Windows driver backed it with system memory instead of raising
out-of-memory. Engine build and benchmark are stand-ins; the numbers are not measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_evaluator_failures import make  # same directory

from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline import trainer as trainer_mod
from cnnlab.tcc.model_pipeline.evaluator import cap_process_memory
from cnnlab.tcc.model_pipeline.trainer import TrainConfig, train_model


def _tiny() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    return model, torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))


def test_a_long_gap_between_steps_is_reported_as_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0] + [0.01 * i for i in range(1, 3)] + [0.02 + 500.0 + 0.01 * i for i in range(1, 50)])
    monkeypatch.setattr(trainer_mod, "_clock", lambda: next(ticks))
    model, x, y = _tiny()
    report = train_model(model, x, y, log10_lr=-3, log10_wd=-4, config=TrainConfig(epochs=1, batch_size=16), seed=0)
    assert report.extra["pauses"] == 1
    assert report.extra["paused_s"] == pytest.approx(500.01)
    assert report.duration_s > 500


def test_normal_steps_have_no_pause() -> None:
    model, x, y = _tiny()
    report = train_model(model, x, y, log10_lr=-3, log10_wd=-4, config=TrainConfig(epochs=1, batch_size=16), seed=0)
    assert report.extra == {"paused_s": 0.0, "pauses": 0, "pause_gap_s": trainer_mod.PAUSE_GAP_S}


def test_paused_time_leaves_the_gpu_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, request = make(monkeypatch, tmp_path)
    real = evaluator_mod.train_model

    def paused_train(*args: Any, **kw: Any) -> Any:
        report = real(*args, **kw)
        report.extra["paused_s"] = 3600.0
        return report

    monkeypatch.setattr(evaluator_mod, "train_model", paused_train)
    result = evaluator.evaluate(request)
    assert result.status == "valid"
    assert result.quality["excluded_from_gpu_seconds"]["paused"] == 3600.0
    assert result.gpu_seconds == 0.0  # the whole attempt took far less than the hour excluded


def test_vram_cap_leaves_the_margin_free(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, Any]] = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (8_000_000_000, 12_000_000_000))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction, device=None: calls.append((fraction, device))
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cap = cap_process_memory(torch.device("cuda"), margin_mb=1024)  # no index, as the evaluator passes it
    assert calls == [((8_000_000_000 - 1_024_000_000) / 12_000_000_000, 0)]
    assert cap["cap_mb"] == pytest.approx(6976.0) and cap["margin_mb"] == 1024


def test_vram_cap_never_drops_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (500_000_000, 12_000_000_000))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device=None: None)
    assert cap_process_memory(torch.device("cuda", 0), margin_mb=1024)["fraction"] == 0.05


def test_definition_records_the_accounting_policy() -> None:
    definition = evaluator_mod.EvaluatorConfig().definition()
    assert definition["memory_cap"]["margin_mb"] == 1024
    assert "training pauses" in definition["gpu_seconds"]
