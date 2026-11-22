"""Keep the CPU tests of the model pipeline off the GPU.

With PyTorch 2.13 a process that only uses CPU tensors still creates a CUDA
context when a GPU is visible:

* ``optimizer.step()`` asks for ``torch.accelerator.current_stream()`` in its
  graph-capture health check;
* the dynamo ONNX exporter saves the CUDA RNG state (``preserve_rng_state``).

A test run would then hold a GPU context next to a measurement job, which
breaks the rule of one GPU job at a time. Observed on 28/09/2026 with a test
run of another worktree during a pilot run.

On this machine an **empty** ``CUDA_VISIBLE_DEVICES`` does not hide the GPU
(``torch.cuda.is_available()`` stays true); ``-1`` does. The variable only
works if it is set before CUDA is first queried in the process, which a
``conftest.py`` of a sub-directory cannot guarantee, so the fixture below
also makes PyTorch report that no accelerator is available. Both are skipped
when the GPU tests are asked for with ``CNNLAB_TCC_GPU_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest
import torch

GPU_TESTS = os.environ.get("CNNLAB_TCC_GPU_TESTS") == "1"

if not GPU_TESTS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture(autouse=True)
def cpu_tests_never_touch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    if GPU_TESTS:
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.optim.Optimizer, "_accelerator_graph_capture_health_check", lambda self: None, raising=False
    )
