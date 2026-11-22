"""Export one run to CSV files for analysis, reading the database only.

`export_run` writes, into a directory that must not exist yet::

    medidas.csv      one row per valid attempt with the four objectives, next to
                     the summary of the candidate's genome (precision per block...)
    tentativas.csv   every attempt with its status, failure and GPU time
    geracoes.csv     per-generation summary
    frentes.csv      one row per (front snapshot, member candidate)
    checkpoints.csv  every checkpoint
    tempo_gpu.csv    GPU ledger of the run and GPU time by attempt status
    manifesto.json   SHA-256 of each file, read-model version, run id, space
                     hash and the commit recorded in the run

The database is opened through `ReadModel` (read-only, `query_only`), so an
export never migrates, never holds a lease and can run while a worker writes.
A drift of the sentinel network marks a stretch of attempts: they carry ``sentinel_drift = true`` in
``medidas.csv`` and ``tentativas.csv``, and nothing is replaced. The seconds the sentinel took are declared in
``tempo_gpu.csv`` and in the manifest, apart from the budget.

Values are written in full, never rounded; what was not measured stays empty,
never zero.

Synthetic results (fake evaluator of the operation tests) are not
measurements. A run that has them is refused unless `allow_synthetic`, and
then every number derived from them goes in a column whose name ends in
`_SINTETICO`.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.errors import NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import (
    BLOCK_FIELDS,
    GENOME_FIELDS,
    generation_summary,
    open_read_model,
    per_block,
)

EXPORT_FORMAT = "tcc-export/1"
MANIFEST_NAME = "manifesto.json"
SYNTHETIC_SUFFIX = "_SINTETICO"
NOT_IN_BLOCK = "NA"  # a block that does not record the field, inside a "per block" cell
OBJECTIVES = (("e", "E_erro_top1"), ("l_ms", "L_ms"), ("v_mb", "V_MB"), ("s_mb", "S_MB"))
UNITS = {
    "E_erro_top1": "top-1 error fraction (0 to 1)",
    "L_ms": "milliseconds",
    "V_MB": "decimal megabytes",
    "S_MB": "decimal megabytes",
    "gpu_s": "seconds of active GPU",
}


class ExportError(StorageError):
    """The export was refused; nothing was written."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    manifest: Mapping[str, Any]
    files: tuple[str, ...]


def _cell(value: Any) -> Any:
    if value is None:
        return ""  # not measured or not recorded: empty, never zero
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)  # full value, no rounding
    return value


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in columns])
    return buffer.getvalue().encode("utf-8")


def _labeled(name: str, synthetic: bool) -> str:
    return f"{name}{SYNTHETIC_SUFFIX}" if synthetic else name


def _value_columns(names: Sequence[str], has_real: bool, has_synthetic: bool) -> list[str]:
    """Real columns (also when there is no row at all), then the labeled synthetic ones."""
    out = list(names) if has_real or not has_synthetic else []
    return out + ([_labeled(n, True) for n in names] if has_synthetic else [])


def _measures(
    rows: Sequence[Mapping[str, Any]],
    genomes: Mapping[str, Mapping[str, Any]],
    front_ids: frozenset[str],
    has_real: bool,
    has_synthetic: bool,
    drift: frozenset[str] = frozenset(),
) -> tuple[list[str], list[dict[str, Any]]]:
    values = _value_columns([name for _, name in OBJECTIVES] + ["gpu_s"], has_real, has_synthetic)
    genome_columns = [*GENOME_FIELDS, *(f"{field}_por_bloco" for field in BLOCK_FIELDS)]
    columns = [
        "attempt_id",
        "candidate_id",
        "generation",
        *values,
        "sintetico",
        "cache_hit",
        "source_attempt_id",
        "na_frente_mais_recente",
        "preliminar",
        "nao_confirmatoria",
        "sentinel_drift",
        *genome_columns,
        "phenotype_hash",
        "cache_context_hash",
        "committed_at",
    ]
    out = []
    for r in rows:
        if r["status"] != "valid" or any(r[key] is None for key, _ in OBJECTIVES):
            continue
        synthetic = bool(r["synthetic"])
        genome = genomes.get(r["candidate_id"], {})
        row: dict[str, Any] = {
            "attempt_id": r["attempt_id"],
            "candidate_id": r["candidate_id"],
            "generation": r["generation"],
            "sintetico": synthetic,
            "cache_hit": bool(r["cache_hit"]),
            "source_attempt_id": r["source_attempt_id"],
            "na_frente_mais_recente": r["candidate_id"] in front_ids,
            "preliminar": bool(r.get("preliminary")),
            "nao_confirmatoria": bool(r.get("not_confirmatory")),
            "phenotype_hash": r["phenotype_hash"],
            "cache_context_hash": r["cache_context_hash"],
            "committed_at": r["committed_at"],
            "sentinel_drift": r["attempt_id"] in drift,
            _labeled("gpu_s", synthetic): r["gpu_seconds"],
        }
        for key, name in OBJECTIVES:
            row[_labeled(name, synthetic)] = r[key]
        for field in GENOME_FIELDS:
            row[field] = genome.get(field)
        for field in BLOCK_FIELDS:
            row[f"{field}_por_bloco"] = per_block(genome, field, missing=NOT_IN_BLOCK) if genome.get("blocks") else None
        out.append(row)
    return columns, out


def _attempts(
    rows: Sequence[Mapping[str, Any]], has_real: bool, has_synthetic: bool, drift: frozenset[str] = frozenset()
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [
        "attempt_id",
        "candidate_id",
        "generation",
        "attempt_number",
        "status",
        "failure_stage",
        "failure_code",
        *_value_columns(["gpu_s"], has_real, has_synthetic),
        "sintetico",
        "cache_hit",
        "source_attempt_id",
        "recuperada_apos_queda",
        "sentinel_drift",
        "started_at",
        "committed_at",
        "commit_seq",
    ]
    out = []
    for r in rows:
        synthetic = bool(r["synthetic"])
        out.append(
            {
                "attempt_id": r["attempt_id"],
                "candidate_id": r["candidate_id"],
                "generation": r["generation"],
                "attempt_number": r["attempt_number"],
                "status": r["status"],
                "failure_stage": r["failure_stage"],
                "failure_code": r["failure_code"],
                _labeled("gpu_s", synthetic): r["gpu_seconds"],
                "sintetico": synthetic,
                "cache_hit": bool(r["cache_hit"]),
                "source_attempt_id": r["source_attempt_id"],
                "recuperada_apos_queda": bool(r.get("recovered")),
                "started_at": r["started_at"],
                "committed_at": r["committed_at"],
                "commit_seq": r["commit_seq"],
                "sentinel_drift": r["attempt_id"] in drift,
            }
        )
    return columns, out


def _generations(
    rows: Sequence[Mapping[str, Any]], has_real: bool, has_synthetic: bool
) -> tuple[list[str], list[dict[str, Any]]]:
    names = ["gpu_s", *(f"menor_{name}" for _, name in OBJECTIVES)]
    columns = [
        "generation",
        "sintetico",
        "candidatas",
        "tentativas",
        "validas",
        "invalidas",
        "interrompidas",
        "em_curso",
        "cache_hits",
        *_value_columns(names, has_real, has_synthetic),
    ]
    out = []
    for synthetic in (False, True):
        group = [r for r in rows if bool(r["synthetic"]) == synthetic]
        for g in generation_summary(group):
            row: dict[str, Any] = {
                "generation": g["generation"],
                "sintetico": synthetic,
                "candidatas": g["candidates"],
                "tentativas": g["attempts"],
                "validas": g["valid"],
                "invalidas": g["invalid"],
                "interrompidas": g["interrupted"],
                "em_curso": g["running"],
                "cache_hits": g["cache_hits"],
                _labeled("gpu_s", synthetic): g["gpu_seconds"],
            }
            for key, name in OBJECTIVES:
                row[_labeled(f"menor_{name}", synthetic)] = g[f"min_{key}"]
            out.append(row)
    return columns, out


def export_run(db_path: Path, run_id: str, dest: Path, *, allow_synthetic: bool = False) -> ExportResult:
    """Write the CSV files and the manifest of one run into the new directory `dest`."""
    dest = Path(dest)
    if dest.exists():
        raise ExportError(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    if not dest.parent.is_dir():
        raise ExportError(f"parent of the destination does not exist: {dest.parent}")
    with open_read_model(Path(db_path)) as rm:
        run = next((r for r in rm.runs() if r["run_id"] == run_id), None)
        if run is None:
            raise NotFoundError(f"run {run_id!r} does not exist in this database")
        before = rm.run_progress(run_id)
        rows = rm.evaluations(run_id)
        genomes = {g["candidate_id"]: g for g in rm.genomes(run_id)}
        fronts = rm.fronts(run_id)
        sentinel = rm.sentinel_events(run_id)
        checkpoints = rm.checkpoints(run_id)
        after = rm.run_progress(run_id)
        read_model, schema_version = rm.version, rm.schema_version()
    drift = frozenset(a for e in sentinel if e["kind"] == "sentinel_drift" for a in e["detail"].get("attempt_ids", ()))
    sentinel_seconds = sum(
        float(e["detail"].get("seconds", 0.0))
        for e in sentinel
        if e["kind"] in ("sentinel_measured", "sentinel_failed")
    )
    sentinel_counts = {
        k: sum(1 for e in sentinel if e["kind"] == f"sentinel_{k}") for k in ("measured", "drift", "failed")
    }

    synthetic_rows = [r for r in rows if r["synthetic"]]
    if synthetic_rows and not allow_synthetic:
        raise ExportError(
            f"run {run_id!r} has {len(synthetic_rows)} synthetic result(s) (fake evaluator of the operation tests); "
            "they are not measurements. Pass allow_synthetic (--allow-synthetic) to export them in labeled columns"
        )
    has_synthetic = bool(synthetic_rows)
    has_real = any(not r["synthetic"] for r in rows)
    derived = has_synthetic  # run-level numbers (ledger, checkpoints, fronts) come from synthetic attempts
    front_ids = frozenset(fronts[-1]["candidate_ids"]) if fronts else frozenset()

    gpu = _labeled("gpu_ativa_s", derived)
    front_rows = [
        {
            "snapshot_id": f["snapshot_id"],
            gpu: f["active_gpu_seconds"],
            "unicas_validas": f["unique_valid_evaluations"],
            "hipervolume": f["hypervolume"],
            "membros": len(f["candidate_ids"]),
            "candidate_id": candidate_id,
            "created_at": f["created_at"],
        }
        for f in fronts
        for candidate_id in f["candidate_ids"]
    ]
    checkpoint_rows = [
        {
            "checkpoint_id": c["checkpoint_id"],
            "generation": c["generation"],
            "avaliacoes_unicas": c["unique_evaluations"],
            gpu: c["active_gpu_seconds"],
            "populacao": c["population"],
            "arquivo_pareto": c["pareto_archive"],
            "pendentes": c["pending"],
            "created_at": c["created_at"],
        }
        for c in checkpoints
    ]
    by_status: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)
    gpu_rows: list[dict[str, Any]] = [
        {
            "escopo": "run (livro de GPU)",
            "tentativas": len(rows),
            gpu: after["active_gpu_s"],
            "orcamento_gpu_s": run["budget_gpu_s"],
        }
    ]
    for status in sorted(by_status):
        measured = [r["gpu_seconds"] for r in by_status[status] if r["gpu_seconds"] is not None]
        gpu_rows.append(
            {
                "escopo": f"tentativas {status}",
                "tentativas": len(by_status[status]),
                gpu: sum(measured) if measured else None,
                "orcamento_gpu_s": None,
            }
        )

    if sentinel:  # the control is declared next to the budget, and apart from it
        gpu_rows.append(
            {
                "escopo": "sentinela (fora do orcamento)",
                "tentativas": sentinel_counts["measured"],
                gpu: sentinel_seconds,
                "orcamento_gpu_s": None,
            }
        )

    tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "medidas.csv": _measures(rows, genomes, front_ids, has_real, has_synthetic, drift),
        "tentativas.csv": _attempts(rows, has_real, has_synthetic, drift),
        "geracoes.csv": _generations(rows, has_real, has_synthetic),
        "frentes.csv": (
            ["snapshot_id", gpu, "unicas_validas", "hipervolume", "membros", "candidate_id", "created_at"],
            front_rows,
        ),
        "checkpoints.csv": (
            [
                "checkpoint_id",
                "generation",
                "avaliacoes_unicas",
                gpu,
                "populacao",
                "arquivo_pareto",
                "pendentes",
                "created_at",
            ],
            checkpoint_rows,
        ),
        "tempo_gpu.csv": (["escopo", "tentativas", gpu, "orcamento_gpu_s"], gpu_rows),
    }
    contents = {name: _csv_bytes(columns, table) for name, (columns, table) in tables.items()}
    manifest: dict[str, Any] = {
        "format": EXPORT_FORMAT,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "read_model": read_model,
        "schema_version": schema_version,
        "database_file": Path(db_path).name,
        "run": {
            key: run.get(key)
            for key in (
                "run_id",
                "stage",
                "method",
                "seed_block",
                "search_seed",
                "train_seed",
                "calibration_seed",
                "budget_gpu_s",
                "protocol_id",
                "plan_id",
                "plan_sha256",
                "plan_purpose",
                "plan_confirmatory",
                "dataset_id",
                "environment_id",
                "space_hash",
                "created_at",
                "git_sha",
                "code_dirty",
                "code_label",
                "system_version",
                "version_source",
            )
        },
        "confirmatory": bool(run.get("plan_confirmatory")),
        "synthetic": has_synthetic,
        "synthetic_note": (
            f"columns ending in {SYNTHETIC_SUFFIX} hold numbers of a fake evaluator; they are not measurements"
            if has_synthetic
            else None
        ),
        "attempts_by_status": after["attempts"],
        "sentinel": {
            **sentinel_counts,
            "seconds_outside_budget": sentinel_seconds,
            "attempts_marked_by_drift": len(drift),
        },
        "changed_during_export": before != after,  # a worker was writing: the files may not be one instant
        "encoding": "utf-8",
        "empty_cell": "not measured or not recorded (never zero)",
        "per_block_cell": f"active blocks in order, separated by one space; {NOT_IN_BLOCK} = not recorded",
        "units": UNITS,
        "files": [
            {
                "name": name,
                "sha256": sha256_hex(contents[name]),
                "bytes": len(contents[name]),
                "rows": len(tables[name][1]),
                "columns": tables[name][0],
            }
            for name in tables
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

    # Built aside and renamed at the end: `dest` either holds a complete export or does not exist.
    building = dest.parent / f".{dest.name}.partial-{uuid.uuid4().hex[:8]}"
    building.mkdir()
    try:
        for name, data in {**contents, MANIFEST_NAME: manifest_bytes}.items():
            with open(building / name, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        os.rename(building, dest)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return ExportResult(dest, manifest, (*tables, MANIFEST_NAME))


def verify_export(path: Path) -> list[str]:
    """Problems of an export directory against its manifest (empty list: it is intact)."""
    path = Path(path)
    try:
        manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"manifest unreadable: {exc}"]
    problems = []
    for entry in manifest.get("files", []):
        file = path / entry["name"]
        if not file.is_file():
            problems.append(f"missing: {entry['name']}")
        elif sha256_hex(file.read_bytes()) != entry["sha256"]:
            problems.append(f"mismatched: {entry['name']}")
    return problems
