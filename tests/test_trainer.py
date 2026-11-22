"""Testes de core/trainer.py.

Os testes rapidos rodam na CPU sobre um `DataBundle` sintetico minusculo: nao
baixam dataset e nao exigem GPU. O que depende de CUDA (AMP, ganho de tempo)
esta marcado como `slow`.
"""

from __future__ import annotations

import threading

import pytest
import torch

from cnnlab.config import ModelConfig, OptimConfig, TrainConfig
from cnnlab.core import trainer as trainer_mod
from cnnlab.core.data import DataBundle, iter_batches
from cnnlab.core.trainer import TrainResult, UnitContext, train
from cnnlab.events import EpochEnd

NUM_CLASSES = 4


def make_bundle(n_train: int = 64, n_val: int = 32, device: torch.device | None = None) -> DataBundle:
    """Bundle sintetico com a mesma forma do CIFAR-100 ja normalizado."""
    device = device or torch.device("cpu")
    gen = torch.Generator().manual_seed(0)

    def block(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, 3, 32, 32, generator=gen)
        y = torch.arange(n, dtype=torch.int64) % NUM_CLASSES
        # Sinal aprendivel: desloca cada classe por um offset constante.
        x += y.view(-1, 1, 1, 1).float()
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        return x, y.to(device)

    x_train, y_train = block(n_train)
    x_val, y_val = block(n_val)
    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        class_names=[f"c{i}" for i in range(NUM_CLASSES)],
        num_classes=NUM_CLASSES,
    )


def tiny_cfg(**kwargs) -> TrainConfig:
    """Config pequena o bastante para 2 epocas rodarem em fracao de segundo."""
    base = {
        "model": ModelConfig(base_filters=8, n_blocks=2, dropout=0.1),
        "optim": OptimConfig(optimizer="adam", lr=1e-3, batch_size=16),
        "epochs": 2,
        "amp": False,
        "channels_last": True,
        "seed": 42,
    }
    base.update(kwargs)
    return TrainConfig(**base)


def ctx(**kwargs) -> UnitContext:
    base = {"run_id": "20260901-2314_ga_a3f1", "unit_index": 0, "unit_total": 1}
    base.update(kwargs)
    return UnitContext(**base)


class Collector:
    """Callback de eventos que so acumula. A UI real faria post_message."""

    def __init__(self) -> None:
        self.events: list[EpochEnd] = []

    def __call__(self, event) -> None:
        self.events.append(event)


# --- contrato de eventos -----------------------------------------------------


def test_two_epochs_emit_exactly_two_epochend() -> None:
    events = Collector()
    result = train(
        tiny_cfg(epochs=2),
        make_bundle(),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )

    assert len(events.events) == 2
    assert all(isinstance(e, EpochEnd) for e in events.events)
    assert [e.epoch for e in events.events] == [1, 2]
    assert result.epochs_run == 2


def test_one_event_per_epoch_not_per_batch() -> None:
    # 64 amostras / batch 8 = 8 batches por epoca. Se houvesse emissao por
    # batch, viriam 8 eventos em vez de 1.
    events = Collector()
    train(
        tiny_cfg(epochs=1, optim=OptimConfig(batch_size=16)),
        make_bundle(n_train=64),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )
    assert len(events.events) == 1


def test_epochend_carries_unit_context() -> None:
    events = Collector()
    train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(run_id="run-x", unit_index=19, unit_total=43, gen=2, individual=5),
        events,
        threading.Event(),
        torch.device("cpu"),
    )

    evt = events.events[0]
    assert evt.run_id == "run-x"
    assert (evt.unit_index, evt.unit_total) == (19, 43)
    assert (evt.gen, evt.individual) == (2, 5)
    assert evt.epochs_total == 1


def test_classic_run_leaves_gen_and_individual_none() -> None:
    events = Collector()
    train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )
    assert events.events[0].gen is None
    assert events.events[0].individual is None


def test_metrics_are_plain_floats_and_finite() -> None:
    events = Collector()
    result = train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )

    evt = events.events[0]
    for value in (evt.train_loss, evt.val_loss, evt.val_acc, evt.elapsed_s):
        assert isinstance(value, float)
        assert value == value  # nao e NaN
    assert 0.0 <= evt.val_acc <= 1.0
    assert result.elapsed_s > 0.0


# --- determinismo ------------------------------------------------------------


def test_same_seed_gives_same_final_val_acc() -> None:
    def run() -> TrainResult:
        return train(
            tiny_cfg(epochs=2),
            make_bundle(),
            ctx(),
            Collector(),
            threading.Event(),
            torch.device("cpu"),
        )

    a, b = run(), run()
    assert a.final_val_acc == b.final_val_acc
    assert a.final_train_loss == b.final_train_loss
    assert a.best_val_acc == b.best_val_acc


def test_different_seed_changes_the_result() -> None:
    def run(seed: int) -> float:
        return train(
            tiny_cfg(epochs=2, seed=seed),
            make_bundle(),
            ctx(),
            Collector(),
            threading.Event(),
            torch.device("cpu"),
        ).final_train_loss

    assert run(42) != run(7)


def test_best_val_acc_is_the_maximum_seen() -> None:
    events = Collector()
    result = train(
        tiny_cfg(epochs=3),
        make_bundle(),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )

    assert result.best_val_acc == max(e.val_acc for e in events.events)
    assert result.final_val_acc == events.events[-1].val_acc


# --- cancelamento ------------------------------------------------------------


def test_cancel_during_first_epoch_stops_after_it() -> None:
    cancel = threading.Event()
    events = Collector()

    def on_event(event) -> None:
        events(event)
        cancel.set()  # cancelado enquanto a epoca 1 rodava

    result = train(
        tiny_cfg(epochs=5),
        make_bundle(),
        ctx(),
        on_event,
        cancel,
        torch.device("cpu"),
    )

    assert result.cancelled is True
    assert result.epochs_run == 1
    assert len(events.events) == 1


def test_cancel_does_not_raise_and_keeps_partial_metrics() -> None:
    cancel = threading.Event()

    def on_event(event) -> None:
        cancel.set()

    result = train(
        tiny_cfg(epochs=4),
        make_bundle(),
        ctx(),
        on_event,
        cancel,
        torch.device("cpu"),
    )

    assert isinstance(result, TrainResult)
    assert result.cancelled is True
    assert 0.0 <= result.final_val_acc <= 1.0


def test_cancel_set_before_start_runs_nothing() -> None:
    cancel = threading.Event()
    cancel.set()
    events = Collector()

    result = train(
        tiny_cfg(epochs=3),
        make_bundle(),
        ctx(),
        events,
        cancel,
        torch.device("cpu"),
    )

    assert result.cancelled is True
    assert result.epochs_run == 0
    assert events.events == []


def test_full_run_is_not_marked_cancelled() -> None:
    result = train(
        tiny_cfg(epochs=2),
        make_bundle(),
        ctx(),
        Collector(),
        threading.Event(),
        torch.device("cpu"),
    )
    assert result.cancelled is False


# --- pesos -------------------------------------------------------------------


def test_keep_weights_false_returns_no_state_dict() -> None:
    result = train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(),
        Collector(),
        threading.Event(),
        torch.device("cpu"),
    )
    assert result.state_dict is None


def test_keep_weights_true_returns_cpu_state_dict() -> None:
    result = train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(),
        Collector(),
        threading.Event(),
        torch.device("cpu"),
        keep_weights=True,
    )

    assert result.state_dict is not None
    assert len(result.state_dict) > 0
    assert all(t.device.type == "cpu" for t in result.state_dict.values())


# --- otimizadores ------------------------------------------------------------


@pytest.mark.parametrize("optimizer", ["adam", "sgd_momentum"])
def test_both_optimizers_train(optimizer: str) -> None:
    result = train(
        tiny_cfg(epochs=1, optim=OptimConfig(optimizer=optimizer, lr=1e-2)),
        make_bundle(),
        ctx(),
        Collector(),
        threading.Event(),
        torch.device("cpu"),
    )
    assert result.epochs_run == 1


def test_sgd_momentum_uses_nesterov_with_momentum_09() -> None:
    model = torch.nn.Linear(2, 2)
    opt = trainer_mod._build_optimizer(model, OptimConfig(optimizer="sgd_momentum", lr=0.1, weight_decay=1e-4))

    group = opt.param_groups[0]
    assert isinstance(opt, torch.optim.SGD)
    assert group["momentum"] == 0.9
    assert group["nesterov"] is True
    assert group["weight_decay"] == 1e-4


def test_adam_receives_lr_and_weight_decay() -> None:
    model = torch.nn.Linear(2, 2)
    opt = trainer_mod._build_optimizer(model, OptimConfig(optimizer="adam", lr=3e-3, weight_decay=5e-5))

    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 3e-3
    assert opt.param_groups[0]["weight_decay"] == 5e-5


# --- OOM ---------------------------------------------------------------------


def test_oom_retry_halves_the_batch_once() -> None:
    seen: list[int] = []

    def fn(batch_size: int) -> str:
        seen.append(batch_size)
        if len(seen) == 1:
            raise torch.cuda.OutOfMemoryError("simulado")
        return "ok"

    value, batch_size = trainer_mod._with_oom_retry(fn, 128)

    assert value == "ok"
    assert batch_size == 64
    assert seen == [128, 64]


def test_second_oom_propagates() -> None:
    def fn(batch_size: int) -> str:
        raise torch.cuda.OutOfMemoryError("simulado")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer_mod._with_oom_retry(fn, 128)


def test_oom_retry_does_not_go_below_the_batchnorm_floor() -> None:
    def fn(batch_size: int) -> str:
        raise torch.cuda.OutOfMemoryError("simulado")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer_mod._with_oom_retry(fn, 2)


def test_training_recovers_from_one_oom_and_keeps_the_smaller_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    exploded = False

    def flaky_iter_batches(x, y, batch_size, shuffle, generator, augment_flip=False):
        nonlocal exploded
        calls.append(batch_size)
        if not exploded and shuffle:
            exploded = True
            raise torch.cuda.OutOfMemoryError("simulado")
        return iter_batches(x, y, batch_size, shuffle, generator, augment_flip)

    monkeypatch.setattr(trainer_mod, "iter_batches", flaky_iter_batches)

    events = Collector()
    result = train(
        tiny_cfg(epochs=2, optim=OptimConfig(batch_size=32)),
        make_bundle(),
        ctx(),
        events,
        threading.Event(),
        torch.device("cpu"),
    )

    assert result.epochs_run == 2
    assert len(events.events) == 2
    assert calls[0] == 32  # tentativa que estourou
    assert calls[1] == 16  # retry com metade
    assert calls[-1] == 16  # o batch reduzido persiste nas epocas seguintes


# --- fallback com dataset na CPU ---------------------------------------------


def test_works_when_bundle_and_device_are_the_same_cpu() -> None:
    result = train(
        tiny_cfg(epochs=1),
        make_bundle(),
        ctx(),
        Collector(),
        threading.Event(),
        torch.device("cpu"),
    )
    assert result.epochs_run == 1


# --- GPU / AMP ---------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="exige GPU CUDA")


@pytest.mark.slow
@requires_cuda
def test_amp_runs_on_cuda_and_emits_the_same_event_count() -> None:
    bundle = make_bundle(n_train=256, n_val=128, device=torch.device("cuda"))
    events = Collector()

    result = train(
        tiny_cfg(epochs=2, amp=True),
        bundle,
        ctx(),
        events,
        threading.Event(),
        torch.device("cuda"),
        keep_weights=True,
    )

    assert result.epochs_run == 2
    assert len(events.events) == 2
    assert all(t.device.type == "cpu" for t in result.state_dict.values())


@pytest.mark.slow
@requires_cuda
def test_amp_is_not_slower_than_fp32() -> None:
    """Aceite 4 da spec: AMP tem que reduzir o tempo por epoca."""
    bundle = make_bundle(n_train=4096, n_val=1024, device=torch.device("cuda"))
    cfg = {"model": ModelConfig(base_filters=32, n_blocks=3, dropout=0.25)}

    def epoch_times(amp: bool) -> float:
        events = Collector()
        train(
            tiny_cfg(epochs=3, amp=amp, optim=OptimConfig(batch_size=128), **cfg),
            bundle,
            ctx(),
            events,
            threading.Event(),
            torch.device("cuda"),
        )
        # Descarta a epoca 1: cudnn.benchmark ainda esta calibrando.
        return min(e.elapsed_s for e in events.events[1:])

    fp32 = epoch_times(amp=False)
    amp = epoch_times(amp=True)
    assert amp <= fp32
