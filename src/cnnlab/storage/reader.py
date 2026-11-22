"""Leitura das runs em disco para a tela de historico.

Duas regras que definem este modulo:

1. `list_runs` le apenas `status.json` e `summary.json`. Abrir os `.jsonl` de
   dezenas de runs para montar uma lista tornaria a tela lenta sem motivo.
2. Toda leitura de `.jsonl` tolera a ultima linha truncada: crash no meio de um
   append deixa exatamente isso.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cnnlab.config import JobSpec, JobSpecEnvelope
from cnnlab.events import RunKind, RunStatus
from cnnlab.storage import layout

_KINDS: frozenset[str] = frozenset({"classic", "ga"})
_STATUSES: frozenset[str] = frozenset({"queued", "running", "done", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Uma linha da tela de historico."""

    run_id: str
    kind: RunKind
    status: RunStatus
    created_at: str
    best_fitness: float | None
    elapsed_s: float | None
    note: str


def list_runs(kind: RunKind | None = None) -> list[RunSummary]:
    """Lista as runs, mais recentes primeiro.

    Uma run marcada `running` que nao pertence a nenhum writer vivo deste
    processo e orfa (crash) e aparece como `failed`. O disco nao e reescrito:
    corrigir estado nao e trabalho de quem le.
    """
    root = layout.RUNS_DIR
    if not root.is_dir():
        return []

    out: list[RunSummary] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        summary = _run_summary(entry)
        if summary is None:
            continue
        if kind is not None and summary.kind != kind:
            continue
        out.append(summary)

    out.sort(key=lambda r: r.created_at, reverse=True)
    return out


def read_progress(run_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(layout.run_dir(run_id) / layout.PROGRESS_FILE)


def read_generations(run_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(layout.run_dir(run_id) / layout.GENERATIONS_FILE)


def read_individuals(run_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(layout.run_dir(run_id) / layout.INDIVIDUALS_FILE)


def read_config(run_id: str) -> JobSpec:
    """Reconstroi o `JobSpec` da run. Levanta se o arquivo nao existir."""
    path = layout.run_dir(run_id) / layout.CONFIG_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobSpecEnvelope.model_validate({"spec": data}).spec


def artifact_path(run_id: str) -> Path | None:
    path = layout.run_dir(run_id) / layout.ARTIFACT_FILE
    return path if path.is_file() else None


# -- internos ------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    """Le um JSON de estado. Arquivo ausente ou corrompido vira None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Le um `.jsonl` ignorando linhas invalidas (a ultima pode estar truncada)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _kind_from_run_id(run_id: str) -> RunKind | None:
    parts = run_id.split("_")
    if len(parts) >= 2 and parts[1] in _KINDS:
        return parts[1]  # type: ignore[return-value]
    return None


def _created_at_from_run_id(run_id: str) -> str:
    """`20260901-2314_ga_a3f1` -> `2026-09-01T23:14:00`. "" se nao casar."""
    stamp = run_id.split("_", 1)[0]
    if len(stamp) != 13 or stamp[8] != "-":
        return ""
    date, time = stamp[:8], stamp[9:]
    if not (date.isdigit() and time.isdigit()):
        return ""
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:]}:00"


def _run_summary(entry: Path) -> RunSummary | None:
    run_id = entry.name
    status_data = _read_json(entry / layout.STATUS_FILE) or {}
    summary_data = _read_json(entry / layout.SUMMARY_FILE) or {}
    if not status_data and not summary_data:
        return None  # pasta sem metadados: nao e uma run

    kind = status_data.get("kind") or summary_data.get("kind")
    if kind not in _KINDS:
        kind = _kind_from_run_id(run_id)
        if kind is None:
            return None

    status = status_data.get("status") or summary_data.get("status")
    if status not in _STATUSES:
        status = "failed" if not summary_data else "done"
    if status == "running" and not layout.is_run_active(run_id):
        status = "failed"  # orfa: processo morreu antes de fechar a run

    created_at = status_data.get("created_at") or _created_at_from_run_id(run_id)

    return RunSummary(
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        created_at=created_at or "",
        best_fitness=_as_float(summary_data.get("best_fitness")),
        elapsed_s=_as_float(summary_data.get("elapsed_s")),
        note=status_data.get("note") or summary_data.get("note") or "",
    )
