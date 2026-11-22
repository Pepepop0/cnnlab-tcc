"""Persistencia em sistema de arquivos: caminhos, escrita e leitura de runs."""

from __future__ import annotations

from cnnlab.storage.layout import (
    PREDICTIONS_FILE,
    QUEUE_FILE,
    RUNS_DIR,
    UPLOADS_DIR,
    append_jsonl,
    atomic_write_json,
    new_run_id,
    run_dir,
)
from cnnlab.storage.reader import (
    RunSummary,
    artifact_path,
    list_runs,
    read_config,
    read_generations,
    read_individuals,
    read_progress,
)
from cnnlab.storage.writer import RunWriter

__all__ = [
    "PREDICTIONS_FILE",
    "QUEUE_FILE",
    "RUNS_DIR",
    "UPLOADS_DIR",
    "RunSummary",
    "RunWriter",
    "append_jsonl",
    "artifact_path",
    "atomic_write_json",
    "list_runs",
    "new_run_id",
    "read_config",
    "read_generations",
    "read_individuals",
    "read_progress",
    "run_dir",
]
