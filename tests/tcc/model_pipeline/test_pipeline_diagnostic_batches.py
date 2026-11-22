"""Diagnostic passes (reference logits, PyTorch accuracy) back off on out-of-memory instead of invalidating (CPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_precision import evaluator_for  # same directory
from test_pipeline_training import request_for  # same directory
from torch import nn

from cnnlab.tcc.model_pipeline.trainer import DIAGNOSTIC_BATCH, evaluate_accuracy, run_in_batches


def oom(message: str = "CUDA out of memory (simulated)") -> torch.cuda.OutOfMemoryError:
    return torch.cuda.OutOfMemoryError(message)


def test_default_diagnostic_batch_is_small() -> None:
    assert DIAGNOSTIC_BATCH == 250


def test_a_batch_that_does_not_fit_is_halved_and_the_same_slice_retried() -> None:
    seen: list[tuple[int, int]] = []

    def run(start: int, stop: int) -> tuple[int, int]:
        if stop - start > 40:
            raise oom()
        seen.append((start, stop))
        return start, stop

    info: dict[str, Any] = {}
    out = run_in_batches(run, 200, batch_size=250, info=info)
    assert [r for chunk in out for r in range(*chunk)] == list(range(200))  # every image exactly once, in order
    assert info == {"start_batch": 250, "final_batch": 31, "oom_halvings": 3}  # 250 -> 125 -> 62 -> 31
    assert seen[0] == (0, 31)


def test_only_a_batch_of_one_that_still_fails_raises() -> None:
    def never(_start: int, _stop: int) -> None:
        raise oom("does not fit at all")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_in_batches(never, 10, batch_size=8)

    sizes: list[int] = []

    def only_singles(start: int, stop: int) -> int:
        sizes.append(stop - start)
        if stop - start > 1:
            raise oom()
        return start

    assert run_in_batches(only_singles, 5, batch_size=8) == [0, 1, 2, 3, 4]
    assert sizes[:4] == [5, 4, 2, 1]  # the first slice is clipped to the 5 images, then 8 -> 4 -> 2 -> 1


def test_errors_other_than_out_of_memory_are_not_swallowed() -> None:
    def broken(_start: int, _stop: int) -> None:
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_in_batches(broken, 10)


def test_nothing_to_run_is_an_empty_pass() -> None:
    info: dict[str, Any] = {}
    assert run_in_batches(lambda a, b: (a, b), 0, info=info) == [] and info["oom_halvings"] == 0


class FlakyModel(nn.Module):
    """Predicts the label stored in the first value of each image; raises out-of-memory above ``fits`` images."""

    def __init__(self, fits: int) -> None:
        super().__init__()
        self.fits = fits
        self.batches: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batches.append(x.shape[0])
        if x.shape[0] > self.fits:
            raise oom()
        return torch.nn.functional.one_hot(x[:, 0, 0, 0].long(), 10).float()


def test_accuracy_is_the_same_whatever_the_batch_that_fits() -> None:
    x = torch.zeros(37, 3, 4, 4)
    x[:, 0, 0, 0] = torch.arange(37) % 10
    y = (torch.arange(37) % 10).clone()
    y[:5] = (y[:5] + 1) % 10  # 5 wrong
    info: dict[str, Any] = {}
    roomy = evaluate_accuracy(FlakyModel(fits=1000), x, y)
    tight = FlakyModel(fits=6)
    assert evaluate_accuracy(tight, x, y, batch_size=32, info=info) == roomy == pytest.approx(32 / 37)
    assert info["final_batch"] <= 6 and info["oom_halvings"] >= 3 and max(tight.batches[len(tight.batches) // 2 :]) <= 6


def test_evaluator_keeps_a_candidate_that_only_failed_in_the_diagnostic_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The network trains and fits; its diagnostic passes do not fit at 5+ images and must back off, not invalidate."""
    real_forward = nn.Conv2d.forward

    def conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        if not self.training and x.shape[0] > 5:
            raise oom()
        return real_forward(self, x)

    monkeypatch.setattr(nn.Conv2d, "forward", conv_forward)
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path)
    result = evaluator.evaluate(request_for(3, tmp_path, config, data))
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    batches = result.quality["diagnostic_batches"]
    for name in ("reference_logits", "torch_accuracy"):
        assert batches[name]["final_batch"] <= 5 and batches[name]["oom_halvings"] >= 1, name


def test_evaluator_still_calls_a_network_invalid_when_not_even_one_image_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_forward = nn.Conv2d.forward

    def conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            raise oom()
        return real_forward(self, x)

    monkeypatch.setattr(nn.Conv2d, "forward", conv_forward)
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path)
    result = evaluator.evaluate(request_for(3, tmp_path, config, data))
    assert result.status == "invalid" and result.failure_code == "out_of_memory" and result.fitness is None
