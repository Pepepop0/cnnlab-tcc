"""Short supervised training of a builder network (TCC-02 section 2, fixed items).

AdamW, batch 128, cosine schedule with a linear warm-up epoch, random crop
32x32 with padding 4 and horizontal flip, learning rate / weight decay from
the genome, dropout already inside the head. Data are tensors resident on
the device (no ``DataLoader``); every random choice (order, crop, flip) comes
from generators seeded by ``train_seed`` so a retrain reproduces the order.

Training runs in FP32 (TF32 matmuls allowed by the PyTorch default); AMP BF16
is not used because the protocol only allows it after a local equivalence
test, which has not been done.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)
TRAINER_VERSION = "tcc-trainer/v1"
# A gap this long between two optimizer steps is not training: the machine slept or the
# process was frozen. It is reported as paused time and left out of the GPU budget.
PAUSE_GAP_S = 120.0
_clock = time.perf_counter  # replaced in tests


class TrainingDiverged(RuntimeError):
    """Loss or logits stopped being finite; the candidate is invalid at the train stage."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    warmup_epochs: int = 1
    crop_padding: int = 4
    deterministic: bool = False  # cuDNN deterministic algorithms; slower, same seed gives same weights

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.warmup_epochs < 0:
            raise ValueError("epochs and batch_size must be >= 1, warmup_epochs >= 0")

    def definition(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "optimizer": "AdamW",
            "schedule": "linear-warmup+cosine",
            "augment": "crop32-pad4+hflip",
            "precision": "fp32",
            "trainer": TRAINER_VERSION,
        }


@dataclass(frozen=True, slots=True)
class TrainReport:
    epochs: int
    steps: int
    epoch_loss: tuple[float, ...]
    epoch_train_acc: tuple[float, ...]
    duration_s: float
    lr: float
    weight_decay: float
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_images(images_u8: torch.Tensor) -> torch.Tensor:
    """NHWC uint8 -> NCHW float32 normalized with the CIFAR-100 channel statistics."""
    x = images_u8.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, padding: int, generator: torch.Generator) -> torch.Tensor:
    """Random crop (zero padding) and horizontal flip on a batch, with a device generator."""
    n, _, h, w = x.shape
    padded = F.pad(x, (padding, padding, padding, padding))
    dy = torch.randint(0, 2 * padding + 1, (n,), device=x.device, generator=generator)
    dx = torch.randint(0, 2 * padding + 1, (n,), device=x.device, generator=generator)
    rows = dy.view(n, 1) + torch.arange(h, device=x.device).view(1, h)  # n x h
    cols = dx.view(n, 1) + torch.arange(w, device=x.device).view(1, w)  # n x w
    batch = torch.arange(n, device=x.device).view(n, 1, 1)
    out = padded.permute(0, 2, 3, 1)[batch, rows.view(n, h, 1), cols.view(n, 1, w)].permute(0, 3, 1, 2)
    flip = torch.rand(n, device=x.device, generator=generator) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), out.flip(3), out)


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warm-up to 1, then cosine decay to 0 at ``total_steps``."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool) -> Iterator[None]:
    """Deterministic cuDNN/torch algorithms for the duration, restoring the previous settings.

    ``warn_only`` keeps an operation without a deterministic implementation
    from aborting the training; PyTorch then warns and the run is not
    guaranteed reproducible. cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG`` set
    before the first CUDA call, so it is set only if still absent.
    """
    if not enabled:
        yield
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[0], before[1]
        torch.use_deterministic_algorithms(before[2], warn_only=before[3])


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    log10_lr: float,
    log10_wd: float,
    config: TrainConfig,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> TrainReport:
    """Train ``model`` in place on normalized ``x`` (N,3,32,32) and labels ``y``, both on its device.

    Raises :class:`TrainingDiverged` on a non-finite loss. Dropout draws from
    the global torch RNG, so the global state is forked and seeded with
    ``seed`` for the duration: equal seeds give equal weights, and the
    caller's RNG stream is left untouched.
    """
    devices = [x.device.index or 0] if x.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), deterministic_algorithms(config.deterministic):
        torch.manual_seed(seed + 2)
        return _train(
            model, x, y, log10_lr=log10_lr, log10_wd=log10_wd, config=config, seed=seed, should_stop=should_stop
        )


def _train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    log10_lr: float,
    log10_wd: float,
    config: TrainConfig,
    seed: int,
    should_stop: Callable[[], bool] | None,
) -> TrainReport:
    device = x.device
    lr, wd = 10.0**log10_lr, 10.0**log10_wd
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = x.shape[0]
    steps_per_epoch = math.ceil(n / config.batch_size)
    total = steps_per_epoch * config.epochs
    warmup = steps_per_epoch * config.warmup_epochs
    order_gen = torch.Generator(device=device).manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    losses: list[float] = []
    accs: list[float] = []
    step = 0
    started = last = _clock()
    paused = 0.0
    pauses = 0
    model.train()
    for _epoch in range(config.epochs):
        perm = torch.randperm(n, device=device, generator=order_gen)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.int64)
        for start in range(0, n, config.batch_size):
            if should_stop is not None and should_stop():
                raise InterruptedError("training stopped cooperatively")
            idx = perm[start : start + config.batch_size]
            xb = augment(x[idx], config.crop_padding, aug_gen)
            yb = y[idx]
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_factor(step, warmup, total)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * idx.numel()
            correct += (logits.detach().argmax(1) == yb).sum()
            step += 1
            now = _clock()
            if now - last > PAUSE_GAP_S:
                paused += now - last
                pauses += 1
            last = now
        epoch_loss = float(loss_sum) / n  # one sync per epoch
        if not math.isfinite(epoch_loss):
            raise TrainingDiverged(f"non-finite training loss in epoch {_epoch}")
        losses.append(epoch_loss)
        accs.append(float(correct) / n)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.eval()
    return TrainReport(
        config.epochs,
        step,
        tuple(losses),
        tuple(accs),
        _clock() - started,
        lr,
        wd,
        extra={"paused_s": paused, "pauses": pauses, "pause_gap_s": PAUSE_GAP_S},
    )


DIAGNOSTIC_BATCH = 250  # images per step of a diagnostic pass (reference logits, accuracy)


def run_in_batches(
    run: Callable[[int, int], Any], n: int, batch_size: int = DIAGNOSTIC_BATCH, info: dict[str, Any] | None = None
) -> list[Any]:
    """``run(start, stop)`` over ``[0, n)``; on CUDA out-of-memory the batch is halved and the same slice retried.

    A diagnostic pass must not turn a network that trained and fits in memory into an invalid candidate because
    of its own batch size: only a batch of 1 that still does not fit raises. ``info`` receives the batch size the
    pass ended with and the number of halvings.
    """
    results: list[Any] = []
    size, backoffs, start = max(1, batch_size), 0, 0
    while start < n:
        stop = min(start + size, n)
        try:
            results.append(run(start, stop))
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            size, backoffs = max(1, size // 2), backoffs + 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        start = stop
    if info is not None:
        info.update(start_batch=max(1, batch_size), final_batch=size, oom_halvings=backoffs)
    return results


def evaluate_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = DIAGNOSTIC_BATCH,
    info: dict[str, Any] | None = None,
) -> float:
    """Top-1 accuracy of the PyTorch model (diagnostic; the protocol's E comes from the engine)."""
    model.eval()

    def hits(start: int, stop: int) -> int:
        with torch.no_grad():
            return int((model(x[start:stop]).argmax(1) == y[start:stop]).sum())

    return sum(run_in_batches(hits, x.shape[0], batch_size, info)) / x.shape[0]
