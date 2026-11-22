"""Longest training per attempt (``train_time_limit_s``): the evaluator and the wiring (CPU, fake clock)."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import pytest
from test_pipeline_precision import evaluator_for  # same directory
from test_pipeline_training import request_for  # same directory

from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.evaluator import EvaluatorConfig
from cnnlab.tcc.model_pipeline.pilot_wiring import DEFAULTS, resolve_arguments


def fake_clock(monkeypatch: pytest.MonkeyPatch, step: float) -> list[float]:
    """Replace the limit's clock by one that advances ``step`` seconds per reading; returns the readings."""
    readings: list[float] = []
    ticks = itertools.count()

    def now() -> float:
        readings.append(next(ticks) * step)
        return readings[-1]

    monkeypatch.setattr(evaluator_mod, "_now", now)
    return readings


def evaluate_with_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: float | None) -> Any:
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path)
    evaluator.config = dataclasses.replace(config, train_time_limit_s=limit)
    return evaluator.evaluate(request_for(3, tmp_path, evaluator.config, data))


def test_training_that_passes_the_limit_is_an_invalid_candidate_with_the_limit_in_the_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = fake_clock(monkeypatch, step=10.0)
    result = evaluate_with_limit(monkeypatch, tmp_path, 15.0)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "train", "train_time_limit")
    assert "limit of 15.0 s" in (result.failure_detail or "") and result.fitness is None
    assert "benchmark" not in result.durations_s and not result.artifact_paths
    assert readings[:3] == [0.0, 10.0, 20.0]  # started, one check inside the limit, the next one outside


def test_no_limit_never_stops_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_clock(monkeypatch, step=1e9)  # a clock that would break any limit
    result = evaluate_with_limit(monkeypatch, tmp_path, None)
    assert result.status == "valid", (result.failure_code, result.failure_detail)


def test_a_limit_the_training_stays_under_does_not_interfere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_clock(monkeypatch, step=0.0)  # time stands still
    result = evaluate_with_limit(monkeypatch, tmp_path, 1.0)
    assert result.status == "valid", (result.failure_code, result.failure_detail)


def test_the_limit_is_part_of_the_evaluator_definition() -> None:
    none, ten = EvaluatorConfig(), EvaluatorConfig(train_time_limit_s=10.0)
    assert none.definition()["train_time_limit_s"] is None and ten.definition()["train_time_limit_s"] == 10.0
    assert none.definition() != ten.definition()


def test_wiring_accepts_none_or_a_positive_number_and_hands_it_on() -> None:
    assert DEFAULTS["train_time_limit_s"] is None
    base = {"cifar_root": "d", "work_dir": "w"}
    assert resolve_arguments(base)["train_time_limit_s"] is None
    assert resolve_arguments({**base, "train_time_limit_s": 120})["train_time_limit_s"] == 120
    assert resolve_arguments({**base, "train_time_limit_s": 0.5})["train_time_limit_s"] == 0.5
    for bad in (0, -1, "60", True, float("nan")):
        with pytest.raises(ValueError, match="train_time_limit_s"):
            resolve_arguments({**base, "train_time_limit_s": bad})
