"""Caminhos, nomes de run e primitivas de escrita em disco.

Este modulo nao importa torch nem nada de `core`/`tui`. Todas as escritas de
JSON de estado passam por `atomic_write_json`: escrever direto no destino final
deixa arquivo truncado se o processo morrer no meio (decisao D3).
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from cnnlab.events import RunKind

RUNS_DIR = Path("runs")
UPLOADS_DIR = Path("uploads")
QUEUE_FILE = Path("queue.json")
PREDICTIONS_FILE = Path("predictions.jsonl")

# Nomes dos arquivos dentro da pasta de uma run. Writer e reader compartilham.
CONFIG_FILE = "config.json"
STATUS_FILE = "status.json"
SUMMARY_FILE = "summary.json"
PROGRESS_FILE = "progress.jsonl"
INDIVIDUALS_FILE = "individuals.jsonl"
GENERATIONS_FILE = "generations.jsonl"
ARTIFACT_FILE = "best.pt"

RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M"

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_RUNS: set[str] = set()
_ATOMIC_REPLACE_LOCK = threading.Lock()


def new_run_id(kind: RunKind, now: datetime | None = None) -> str:
    """Gera `YYYYMMDD-HHMM_{kind}_{hash4}`.

    Os 4 hex aleatorios existem para duas runs iniciadas no mesmo minuto nao
    colidirem de nome de pasta.
    """
    stamp = (now or datetime.now()).strftime(RUN_ID_TIME_FORMAT)
    return f"{stamp}_{kind}_{secrets.token_hex(2)}"


def run_dir(run_id: str) -> Path:
    """Pasta de uma run. Le `RUNS_DIR` na chamada, entao e redirecionavel."""
    return RUNS_DIR / run_id


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escreve JSON via temporario unico + replace atomico."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fd = -1  # o context manager passa a ser o dono do descritor
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        # No Windows, replaces simultaneos do mesmo destino podem disputar o
        # handle do arquivo mesmo quando os temporarios sao distintos.
        with _ATOMIC_REPLACE_LOCK:
            os.replace(tmp, path)
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Acrescenta uma linha JSON e fecha o handle.

    Handle fechado sobrevive a crash: o que ja foi escrito continua legivel.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()


def mark_run_active(run_id: str) -> None:
    """Registra que esta run pertence a este processo (uso interno do storage)."""
    with _ACTIVE_LOCK:
        _ACTIVE_RUNS.add(run_id)


def mark_run_inactive(run_id: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_RUNS.discard(run_id)


def is_run_active(run_id: str) -> bool:
    """True se um `RunWriter` vivo neste processo esta escrevendo nesta run.

    E o que permite ao reader distinguir uma run realmente em andamento de uma
    orfa deixada por crash, sem tocar no disco.
    """
    with _ACTIVE_LOCK:
        return run_id in _ACTIVE_RUNS
