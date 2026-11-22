"""Escrita do estado de uma run em disco.

`RunWriter.handle` e o callback `on_event` que o core recebe. O core nao sabe
que existe disco; quem traduz evento em arquivo e este modulo.

torch so e importado dentro de `save_artifact`: o resto do storage precisa
rodar em maquina sem GPU e em teste sem torch instalado.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from cnnlab.config import JobSpec, ModelConfig
from cnnlab.events import (
    EpochEnd,
    Event,
    GenerationDone,
    IndividualDone,
    RunCancelled,
    RunFailed,
    RunFinished,
    RunStarted,
    RunStatus,
    event_to_dict,
)
from cnnlab.storage import layout

_TERMINAL_STATUS: frozenset[str] = frozenset({"done", "failed", "cancelled"})


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _torch_info() -> tuple[str | None, str | None]:
    """(device, torch_version) para o `summary.json`. Nunca levanta."""
    try:
        import torch
    except Exception:
        return None, None
    device: str | None = "cpu"
    try:
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:
        device = None
    return device, getattr(torch, "__version__", None)


class RunWriter:
    """Dono da pasta de uma run. Uma instancia por run."""

    def __init__(self, run_id: str, spec: JobSpec) -> None:
        self.run_id = run_id
        self.spec = spec
        self.dir: Path = layout.run_dir(run_id)
        self._lock = threading.Lock()
        self._created_at = _now()
        self._started_at: str | None = None
        self._status: RunStatus = "queued"

        self.dir.mkdir(parents=True, exist_ok=True)
        # config.json vai para o disco antes de qualquer treino: se o processo
        # morrer no segundo seguinte, a run ja existe e e reproduzivel.
        layout.atomic_write_json(self.dir / layout.CONFIG_FILE, spec.model_dump(mode="json"))
        layout.mark_run_active(run_id)
        self._write_status("queued")

    # -- ciclo de vida ---------------------------------------------------

    def start(self) -> None:
        """Marca a run como `running`. Idempotente.

        Existe para quem inicia a run sem emitir `RunStarted`; quando o evento
        vem, `handle` chega no mesmo estado.
        """
        with self._lock:
            if self._started_at is None:
                self._started_at = _now()
            self._write_status("running")

    def fail(self, exc: BaseException) -> None:
        """Marca a run como `failed`. Chamavel junto com `handle(RunFailed)`."""
        with self._lock:
            self._write_status("failed", error=f"{type(exc).__name__}: {exc}")
        layout.mark_run_inactive(self.run_id)

    # -- callback do core ------------------------------------------------

    def handle(self, evt: Event) -> None:
        """Roteia um evento para o arquivo correspondente."""
        if isinstance(evt, RunStarted):
            with self._lock:
                self._started_at = evt.ts
                self._write_status("running")
            return

        if isinstance(evt, EpochEnd):
            layout.append_jsonl(self.dir / layout.PROGRESS_FILE, event_to_dict(evt))
            return

        if isinstance(evt, IndividualDone):
            layout.append_jsonl(self.dir / layout.INDIVIDUALS_FILE, event_to_dict(evt))
            return

        if isinstance(evt, GenerationDone):
            layout.append_jsonl(self.dir / layout.GENERATIONS_FILE, event_to_dict(evt))
            return

        if isinstance(evt, RunFinished):
            self._write_summary(evt)
            with self._lock:
                self._write_status("done")
            layout.mark_run_inactive(self.run_id)
            return

        if isinstance(evt, RunFailed):
            with self._lock:
                self._write_status("failed", error=evt.error)
            layout.mark_run_inactive(self.run_id)
            return

        if isinstance(evt, RunCancelled):
            with self._lock:
                self._write_status("cancelled")
            layout.mark_run_inactive(self.run_id)
            return

    # -- artefato --------------------------------------------------------

    def save_artifact(
        self,
        state_dict: dict,
        model_cfg: ModelConfig,
        class_names: list[str],
        val_acc: float,
    ) -> Path:
        """Salva `best.pt` com o suficiente para reconstruir via `build_cnn`.

        Nunca serializa o objeto do modelo, so o `state_dict`.
        """
        import torch

        path = self.dir / layout.ARTIFACT_FILE
        tmp = path.with_suffix(".tmp")
        payload = {
            "state_dict": state_dict,
            "model_config": model_cfg.model_dump(),
            "class_names": list(class_names),
            "val_acc": float(val_acc),
            "run_id": self.run_id,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    # -- internos --------------------------------------------------------

    def _write_status(self, status: RunStatus, error: str | None = None) -> None:
        finished = _now() if status in _TERMINAL_STATUS else None
        self._status = status
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": self.spec.kind,
            "status": status,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "finished_at": finished,
            "error": error,
            # `note` mora aqui porque `list_runs` nao pode abrir config.json.
            "note": self.spec.note,
        }
        layout.atomic_write_json(self.dir / layout.STATUS_FILE, payload)

    def _write_summary(self, evt: RunFinished) -> None:
        device, torch_version = _torch_info()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": evt.kind,
            "status": "done",
            "best_fitness": evt.best_fitness,
            "best_genome": evt.best_genome,
            "total_trainings": evt.total_trainings,
            "total_epochs": evt.total_epochs,
            "elapsed_s": evt.elapsed_s,
            "artifact": evt.artifact,
            "device": device,
            "torch_version": torch_version,
            "note": self.spec.note,
            "ts": evt.ts,
        }
        layout.atomic_write_json(self.dir / layout.SUMMARY_FILE, payload)
