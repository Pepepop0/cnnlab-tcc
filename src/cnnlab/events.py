"""Eventos emitidos pelo core.

Todos sao imutaveis e serializaveis. Nenhum carrega tensor, modulo torch ou
referencia a widget: eles cruzam a fronteira entre a worker thread e a UI.

Este modulo nao importa nada do projeto nem torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

RunKind = Literal["classic", "ga"]
RunStatus = Literal["queued", "running", "done", "failed", "cancelled"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    kind: RunKind
    unit_total: int
    """Total de treinos da run. 1 no modo classico, N no AG."""
    epochs_per_unit: int
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class EpochEnd:
    run_id: str
    epoch: int
    epochs_total: int
    train_loss: float
    val_loss: float
    val_acc: float
    elapsed_s: float
    unit_index: int
    """Indice do treino atual dentro da run, base 0."""
    unit_total: int
    gen: int | None = None
    individual: int | None = None
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class IndividualDone:
    run_id: str
    gen: int
    individual: int
    genome: dict[str, Any]
    fitness: float
    epochs: int
    elapsed_s: float
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class GenerationDone:
    run_id: str
    gen: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_genome: dict[str, Any]
    elapsed_s: float
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class RunFinished:
    run_id: str
    kind: RunKind
    best_fitness: float
    best_genome: dict[str, Any] | None
    total_trainings: int
    total_epochs: int
    elapsed_s: float
    artifact: str | None
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class RunFailed:
    run_id: str
    error: str
    traceback: str = ""
    ts: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class RunCancelled:
    run_id: str
    at_unit: int
    ts: str = field(default_factory=_now)


Event: TypeAlias = (
    RunStarted
    | EpochEnd
    | IndividualDone
    | GenerationDone
    | RunFinished
    | RunFailed
    | RunCancelled
)


def event_to_dict(evt: Event) -> dict[str, Any]:
    """Serializa um evento, incluindo o nome da classe em `event`."""
    payload = asdict(evt)
    payload["event"] = type(evt).__name__
    return payload
