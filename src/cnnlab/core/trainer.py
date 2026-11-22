"""Treino de uma CNN por N epocas.

Nao sabe o que e um AG nem o que e a TUI: reporta por `on_event` e devolve um
`TrainResult`. O AG chama `train` dezenas de vezes por sessao, entao o caminho
quente evita sincronizacoes desnecessarias — as metricas ficam acumuladas em
tensor no device e so viram float uma vez por epoca.
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
from torch import nn

from cnnlab.config import OptimConfig, TrainConfig
from cnnlab.core.data import DataBundle, iter_batches
from cnnlab.core.model import build_cnn
from cnnlab.events import EpochEnd, Event

# Regra 2 da spec: ligado uma vez, no import do modulo.
torch.backends.cudnn.benchmark = True

# Nem `TrainConfig` nem `DataBundle` carregam a flag de augmentation — ela vive
# em `DatasetConfig`, que o trainer nao recebe. Assumimos o default do contrato
# (`DatasetConfig.augment_flip = True`). Ver nota no relatorio do modulo.
_AUGMENT_FLIP = True

# Piso do batch no retry de OOM: BatchNorm em modo treino quebra com 1 amostra.
_MIN_BATCH_SIZE = 2

_T = TypeVar("_T")


@dataclass
class TrainResult:
    """Resultado de um treino. Sem tensores vivos, exceto `state_dict` na CPU."""

    best_val_acc: float
    final_val_acc: float
    final_train_loss: float
    epochs_run: int
    elapsed_s: float
    cancelled: bool
    state_dict: dict | None = None


@dataclass
class UnitContext:
    """Onde este treino se encaixa na run. O AG preenche gen/individual."""

    run_id: str
    unit_index: int
    unit_total: int
    gen: int | None = None
    individual: int | None = None


def train(
    train_cfg: TrainConfig,
    data: DataBundle,
    ctx: UnitContext,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
    keep_weights: bool = False,
) -> TrainResult:
    """Treina por `train_cfg.epochs` epocas, emitindo um `EpochEnd` por epoca.

    Cancelamento e checado no fim de cada epoca e retorna o que ja foi obtido,
    sem levantar excecao.
    """
    device = torch.device(device)
    _seed_everything(train_cfg.seed)

    use_amp = train_cfg.amp and device.type == "cuda"
    channels_last = train_cfg.channels_last

    model = build_cnn(train_cfg.model, data.num_classes).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, train_cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator(device=data.x_train.device).manual_seed(train_cfg.seed)

    batch_size = train_cfg.optim.batch_size
    started = time.perf_counter()

    best_val_acc = 0.0
    final_val_acc = 0.0
    final_train_loss = 0.0
    epochs_run = 0
    cancelled = False

    if cancel.is_set():
        return TrainResult(0.0, 0.0, 0.0, 0, 0.0, True, None)

    for epoch in range(1, train_cfg.epochs + 1):
        epoch_started = time.perf_counter()

        train_loss, batch_size = _with_oom_retry(
            lambda bs: _train_one_epoch(
                model,
                data,
                optimizer,
                criterion,
                scaler,
                generator,
                bs,
                device,
                use_amp,
                channels_last,
            ),
            batch_size,
        )
        (val_loss, val_acc), batch_size = _with_oom_retry(
            lambda bs: _evaluate(model, data, criterion, generator, bs, device, use_amp, channels_last),
            batch_size,
        )

        epochs_run = epoch
        final_train_loss = train_loss
        final_val_acc = val_acc
        best_val_acc = max(best_val_acc, val_acc)

        on_event(
            EpochEnd(
                run_id=ctx.run_id,
                epoch=epoch,
                epochs_total=train_cfg.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                elapsed_s=time.perf_counter() - epoch_started,
                unit_index=ctx.unit_index,
                unit_total=ctx.unit_total,
                gen=ctx.gen,
                individual=ctx.individual,
            )
        )

        if cancel.is_set():
            cancelled = True
            break

    state_dict = _cpu_state_dict(model) if keep_weights else None
    return TrainResult(
        best_val_acc=best_val_acc,
        final_val_acc=final_val_acc,
        final_train_loss=final_train_loss,
        epochs_run=epochs_run,
        elapsed_s=time.perf_counter() - started,
        cancelled=cancelled,
        state_dict=state_dict,
    )


def _train_one_epoch(
    model: nn.Module,
    data: DataBundle,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> float:
    """Uma passada de treino. Devolve a loss media ponderada por amostra."""
    model.train()
    amp_ctx = _amp_context(use_amp)
    loss_sum = torch.zeros((), device=device)
    seen = 0

    for xb, yb in iter_batches(
        data.x_train,
        data.y_train,
        batch_size,
        shuffle=True,
        generator=generator,
        augment_flip=_AUGMENT_FLIP,
    ):
        xb, yb = _to_device(xb, yb, device, channels_last)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = yb.shape[0]
        loss_sum += loss.detach().float() * n
        seen += n

    return float(loss_sum.item()) / max(seen, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    data: DataBundle,
    criterion: nn.Module,
    generator: torch.Generator,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> tuple[float, float]:
    """Avalia em validacao. Sem augmentation, sem consumir o generator."""
    model.eval()
    amp_ctx = _amp_context(use_amp)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0

    for xb, yb in iter_batches(
        data.x_val,
        data.y_val,
        batch_size,
        shuffle=False,
        generator=generator,
        augment_flip=False,
    ):
        xb, yb = _to_device(xb, yb, device, channels_last)
        with amp_ctx:
            logits = model(xb)
            loss = criterion(logits, yb)

        n = yb.shape[0]
        loss_sum += loss.detach().float() * n
        correct += (logits.argmax(dim=1) == yb).sum()
        seen += n

    model.train()
    denom = max(seen, 1)
    return float(loss_sum.item()) / denom, float(correct.item()) / denom


def _with_oom_retry(fn: Callable[[int], _T], batch_size: int) -> tuple[_T, int]:
    """Roda `fn(batch_size)`; em OOM limpa o cache e tenta uma vez com metade.

    O batch reduzido persiste nas epocas seguintes: se nao coube uma vez, nao
    vai caber depois. Um segundo OOM propaga.
    """
    try:
        return fn(batch_size), batch_size
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        smaller = max(_MIN_BATCH_SIZE, batch_size // 2)
        if smaller >= batch_size:
            raise
        return fn(smaller), smaller


def _build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )


def _amp_context(use_amp: bool) -> Any:
    """Autocast fp16 quando AMP esta ligado; caso contrario, contexto vazio.

    Um `nullcontext` evita o aviso do autocast em maquina sem CUDA.
    """
    if not use_amp:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16)


def _to_device(
    xb: torch.Tensor, yb: torch.Tensor, device: torch.device, channels_last: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """No-op quando o dataset ja mora no device (o caso normal, decisao D5)."""
    if xb.device != device:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
    if channels_last and not xb.is_contiguous(memory_format=torch.channels_last):
        xb = xb.contiguous(memory_format=torch.channels_last)
    return xb, yb


def _cpu_state_dict(model: nn.Module) -> dict:
    """Copia os pesos para a CPU: o AG so guarda isso do melhor individuo."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
