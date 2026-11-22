"""Versioned, read-only queries for TUI/Streamlit (`tcc-read/1`).

Interfaces never open `SQLiteRunStore` (a writer) nor read write tables with
their own assumptions: they use this module. Every method runs on a
`mode=ro` + `query_only` connection inside one read transaction, returns
plain JSON-friendly dicts/lists (ready for `pandas.DataFrame`) and never
touches a lease, so watching a run cannot interfere with the worker.

Fitness values come only from `measurements` (or, for cache hits, from the
source attempt's measurement, flagged `cache_hit`). Missing or invalid
results are `None`, never zero. `synthetic` exposes `quality.synthetic` so a
screen can label fake results; `preliminary` and `not_confirmatory` expose
what the evaluator itself declared about its measurements.

Databases of schema v1 are read as they are, never migrated: a run written
straight through the `RunStore` port has no job queue, so `jobs()` is empty
and `has_job_queue` is False. The code version of a run is read from the
environment manifest in either recorded shape (`system` written by
`operation.demo`, `science` written by the science wiring); what a manifest
does not record stays `None`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.storage.connection import open_reader
from cnnlab.tcc.storage.errors import NotFoundError, StorageError
from cnnlab.tcc.storage.plans import plan_body
from cnnlab.tcc.storage.schema import LATEST_VERSION, SENTINEL_EVENT_KINDS

READ_MODEL_VERSION = "tcc-read/1"
# Scientific tables exist from migration 1; the job queue only from migration 2.
MIN_SCHEMA_VERSION = 1
QUEUE_TABLES = ("jobs", "job_events")
DIRTY_SUFFIX = "+dirty"
_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")
OBJECTIVES = ("e", "l_ms", "v_mb", "s_mb")
GENOME_FIELDS = ("n_blocks", "stem_channels", "stem_precision", "activation", "head", "head_precision")
BLOCK_FIELDS = ("channels", "op", "kernel", "n_convs", "residual", "downsample", "precision", "prune_pct")


class ReadModelError(StorageError):
    """The path cannot be read as a TCC v2 database; `code` names the reason for the UI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def system_of(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Code/system identity recorded in an environment manifest; unrecorded fields are None."""
    system = manifest.get("system") or {}
    science = manifest.get("science") or {}
    out: dict[str, Any] = {
        "version_source": None,
        "system_version": None,
        "code_label": None,
        "git_sha": None,
        "code_dirty": None,
        "wiring_label": None,
        "evaluator_label": None,
        "runner_label": None,
    }
    if isinstance(system, Mapping) and system:
        code = system.get("code_version") or {}
        out.update(
            version_source="operation-manifest",
            system_version=system.get("readable"),
            code_label=code.get("label"),
            git_sha=code.get("git_sha"),
            code_dirty=code.get("dirty"),
            evaluator_label=manifest.get("evaluator"),
            runner_label=manifest.get("runner"),
        )
    elif isinstance(science, Mapping) and science:
        raw = science.get("code_version")
        dirty = None
        if isinstance(raw, str) and raw.endswith(DIRTY_SUFFIX) and _GIT_SHA.match(raw[: -len(DIRTY_SUFFIX)]):
            raw, dirty = raw[: -len(DIRTY_SUFFIX)], True  # a clean checkout is recorded without any mark
        is_sha = isinstance(raw, str) and bool(_GIT_SHA.match(raw))
        evaluator = science.get("evaluator")
        search = science.get("search")
        out.update(
            version_source="science-manifest",
            git_sha=raw if is_sha else None,
            code_dirty=dirty,
            code_label=None if is_sha or raw is None else str(raw),
            wiring_label=science.get("wiring"),
            evaluator_label=evaluator.get("evaluator") if isinstance(evaluator, Mapping) else None,
            runner_label=search.get("runner") if isinstance(search, Mapping) else None,
        )
    return out


def version_text(run: Mapping[str, Any], *, missing: str = "(not recorded)") -> str:
    """One line with what the database records about the code of a run; nothing is invented."""
    if run.get("system_version"):
        return str(run["system_version"])
    sha = run.get("git_sha")
    parts = [f"git {sha[:12]}" if sha else f"git {missing}"]
    if run.get("code_dirty") is True:
        parts[0] += "+dirty"
    parts += [str(run[k]) for k in ("code_label", "wiring_label", "evaluator_label", "runner_label") if run.get(k)]
    return " | ".join(parts)


def genome_summary(expressed: Any) -> dict[str, Any]:
    """What a recorded expressed genome says the candidate is; unrecorded fields are None.

    Reads the stored JSON only. It does not know the search space, so it
    neither validates nor completes a genome: a key that is absent stays
    absent. `blocks` holds the active blocks (the first `n_blocks`).
    """
    genome = expressed if isinstance(expressed, Mapping) else {}
    out: dict[str, Any] = {field: genome.get(field) for field in GENOME_FIELDS}
    raw_blocks = genome.get("blocks")
    blocks = (
        [b for b in raw_blocks if isinstance(b, Mapping)]
        if isinstance(raw_blocks, Sequence) and not isinstance(raw_blocks, str)
        else []
    )
    n_blocks = out["n_blocks"]
    if isinstance(n_blocks, int) and not isinstance(n_blocks, bool) and 0 <= n_blocks <= len(blocks):
        blocks = blocks[:n_blocks]
    out["blocks"] = [{field: block.get(field) for field in BLOCK_FIELDS} for block in blocks]
    out["genome_schema"] = genome.get("schema")
    out["blocks_recorded"] = bool(out["blocks"])
    return out


def per_block(genome: Mapping[str, Any], field: str, *, missing: str = "(not recorded)") -> str:
    """One field of every active block on a single line, e.g. "fp32 fp16 fp16 fp32"."""
    blocks = genome.get("blocks") or []
    if not blocks:
        return missing
    return " ".join(missing if block.get(field) is None else str(block[field]) for block in blocks)


def generation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per generation: attempts by status and the best (lowest) measured value of each objective.

    The four minima are independent (usually from different candidates); they
    summarize the generation and are not a point of the front.
    """
    by_generation: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_generation.setdefault(row["generation"], []).append(row)
    out = []
    for generation in sorted(by_generation, key=lambda g: (g is None, g)):
        group = by_generation[generation]
        valid = [r for r in group if r["status"] == "valid"]
        summary: dict[str, Any] = {
            "generation": generation,
            "candidates": len({r["candidate_id"] for r in group}),
            "attempts": len(group),
            "valid": len(valid),
            "invalid": sum(1 for r in group if r["status"] == "invalid"),
            "interrupted": sum(1 for r in group if r["status"] == "interrupted"),
            "running": sum(1 for r in group if r["status"] == "running"),
            "cache_hits": sum(1 for r in group if r["cache_hit"]),
            "gpu_seconds": sum(float(r["gpu_seconds"] or 0.0) for r in group),
        }
        for key in OBJECTIVES:
            values = [r[key] for r in valid if r[key] is not None]
            summary[f"min_{key}"] = min(values) if values else None
        out.append(summary)
    return out


class ReadModel:
    """Read-only view of a TCC v2 database. Use as a context manager."""

    version = READ_MODEL_VERSION

    def __init__(self, db_path: Path) -> None:
        self._conn = open_reader(Path(db_path))
        self._conn.row_factory = sqlite3.Row
        self._queue: bool | None = None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ReadModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        finally:
            self._conn.execute("ROLLBACK")

    # -- schema and references -------------------------------------------

    def schema_version(self) -> int:
        with self._read() as conn:
            return int(conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0])

    @property
    def has_job_queue(self) -> bool:
        """False for schema v1: the run was written without the operation queue (no job, no lease)."""
        if self._queue is None:
            with self._read() as conn:
                found = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)", QUEUE_TABLES
                    )
                }
            self._queue = found == set(QUEUE_TABLES)
        return self._queue

    def environment(self, environment_id: str) -> dict[str, Any]:
        with self._read() as conn:
            row = conn.execute(
                "SELECT manifest_json, manifest_sha256 FROM environments WHERE environment_id = ?", (environment_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"environment {environment_id!r} does not exist")
        return {"environment_id": environment_id, "manifest_sha256": row[1], "manifest": json.loads(row[0])}

    def plan(self, plan_id: str) -> dict[str, Any]:
        """One experiment plan as stored, with its canonical ``body`` (the bytes the hash is taken over).

        ``hash_ok`` says whether the stored hash is the hash of that body; a frozen plan whose ``hash_ok``
        is false has been tampered with.
        """
        with self._read() as conn:
            row = conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"plan {plan_id!r} does not exist")
        content = json.loads(row["content_json"])
        body = plan_body(
            plan_id=row["plan_id"],
            protocol_id=row["protocol_id"],
            version=row["version"],
            author=row["author"],
            amends_plan_id=row["amends_plan_id"],
            amendment_reason=row["amendment_reason"],
            content=content,
        )
        from cnnlab.tcc.storage.canonical import sha256_hex

        return {
            "plan_id": row["plan_id"],
            "protocol_id": row["protocol_id"],
            "version": row["version"],
            "status": row["status"],
            "author": row["author"],
            "content": content,
            "plan_sha256": row["plan_sha256"],
            "amends_plan_id": row["amends_plan_id"],
            "amendment_reason": row["amendment_reason"],
            "created_at": row["created_at"],
            "frozen_at": row["frozen_at"],
            "body": body,
            "hash_ok": sha256_hex(body) == row["plan_sha256"],
        }

    # -- runs --------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        """Every run with its stage, cell, budget, plan and the code version it was created with."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT r.*, e.manifest_json, p.content_json AS plan_content_json "
                "FROM runs r JOIN environments e USING (environment_id) "
                "JOIN experiment_plans p ON p.plan_id = r.plan_id "
                "ORDER BY r.created_at, r.run_id"
            ).fetchall()
        out = []
        for r in rows:
            plan = json.loads(r["plan_content_json"])
            out.append(
                {
                    "run_id": r["run_id"],
                    "stage": r["stage"],
                    "method": r["method"],
                    "seed_block": r["seed_block"],
                    "search_seed": r["search_seed"],
                    "train_seed": r["train_seed"],
                    "calibration_seed": r["calibration_seed"],
                    "budget_gpu_s": r["budget_gpu_s"],
                    "protocol_id": r["protocol_id"],
                    "plan_id": r["plan_id"],
                    "plan_sha256": r["plan_sha256"],
                    "plan_purpose": plan.get("purpose"),
                    "plan_confirmatory": plan.get("confirmatory"),
                    "dataset_id": r["dataset_id"],
                    "environment_id": r["environment_id"],
                    "space_hash": r["space_hash"],
                    "finalist_list_id": r["finalist_list_id"],
                    "created_at": r["created_at"],
                    **system_of(json.loads(r["manifest_json"])),
                }
            )
        return out

    def run_progress(self, run_id: str) -> dict[str, Any]:
        """GPU ledger, attempt counts, unique valid evaluations and latest checkpoint."""
        with self._read() as conn:
            run = conn.execute("SELECT budget_gpu_s FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            attempts = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT status, COUNT(*) FROM evaluation_attempts WHERE run_id = ? GROUP BY status", (run_id,)
                )
            }
            ledger = conn.execute(
                "SELECT COALESCE(SUM(gpu_seconds), 0.0) FROM gpu_segments WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            unique = conn.execute(
                "SELECT COUNT(DISTINCT c.cache_context_hash) FROM evaluation_attempts a "
                "JOIN candidates c ON c.candidate_id = a.candidate_id "
                "WHERE a.run_id = ? AND a.status = 'valid' AND a.source_attempt_id IS NULL",
                (run_id,),
            ).fetchone()[0]
            hits = conn.execute(
                "SELECT COUNT(*) FROM evaluation_attempts WHERE run_id = ? AND source_attempt_id IS NOT NULL",
                (run_id,),
            ).fetchone()[0]
            synthetic = conn.execute(
                "SELECT COUNT(*) FROM evaluation_attempts WHERE run_id = ? AND status <> 'running' "
                "AND json_extract(quality_json, '$.synthetic') = 1",
                (run_id,),
            ).fetchone()[0]
            candidates = conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = ?", (run_id,)).fetchone()[0]
            generations = conn.execute(
                "SELECT COUNT(DISTINCT generation) FROM candidates WHERE run_id = ? AND generation IS NOT NULL",
                (run_id,),
            ).fetchone()[0]
            counts = conn.execute(
                "SELECT (SELECT COUNT(*) FROM checkpoints WHERE run_id = ?), "
                "(SELECT COUNT(*) FROM front_snapshots WHERE run_id = ?)",
                (run_id, run_id),
            ).fetchone()
            checkpoint = conn.execute(
                "SELECT generation, unique_evaluations, active_gpu_seconds, created_at FROM checkpoints "
                "WHERE run_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return {
            "run_id": run_id,
            "budget_gpu_s": run["budget_gpu_s"],
            "active_gpu_s": float(ledger),
            "attempts": attempts,
            "unique_valid_evaluations": unique,
            "cache_hits": hits,
            "synthetic_results": synthetic,
            "candidates": candidates,
            "generations": generations,
            "checkpoints": counts[0],
            "front_snapshots": counts[1],
            "latest_checkpoint": None if checkpoint is None else dict(checkpoint),
        }

    def evaluations(self, run_id: str) -> list[dict[str, Any]]:
        """One row per attempt in commit order, with E/L/V/S only where measured."""
        with self._read() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            rows = conn.execute(
                "SELECT a.attempt_id, a.candidate_id, a.attempt_number, a.status, a.source_attempt_id, "
                "a.gpu_seconds, a.failure_stage, a.failure_code, a.quality_json, a.started_at, a.committed_at, "
                "a.commit_seq, a.recovered, c.generation, c.phenotype_hash, c.cache_context_hash, "
                "COALESCE(m.e_fraction, ms.e_fraction) AS e, COALESCE(m.l_ms, ms.l_ms) AS l_ms, "
                "COALESCE(m.v_mb, ms.v_mb) AS v_mb, COALESCE(m.s_mb, ms.s_mb) AS s_mb "
                "FROM evaluation_attempts a JOIN candidates c ON c.candidate_id = a.candidate_id "
                "LEFT JOIN measurements m ON m.attempt_id = a.attempt_id "
                "LEFT JOIN measurements ms ON ms.attempt_id = a.source_attempt_id "
                "WHERE a.run_id = ? ORDER BY a.commit_seq IS NULL, a.commit_seq, a.rowid",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            quality = json.loads(r["quality_json"]) if r["quality_json"] else {}
            valid = r["status"] == "valid"
            out.append(
                {
                    "attempt_id": r["attempt_id"],
                    "candidate_id": r["candidate_id"],
                    "generation": r["generation"],
                    "attempt_number": r["attempt_number"],
                    "status": r["status"],
                    "cache_hit": r["source_attempt_id"] is not None,
                    "source_attempt_id": r["source_attempt_id"],
                    "e": r["e"] if valid else None,
                    "l_ms": r["l_ms"] if valid else None,
                    "v_mb": r["v_mb"] if valid else None,
                    "s_mb": r["s_mb"] if valid else None,
                    "gpu_seconds": r["gpu_seconds"],
                    "failure_stage": r["failure_stage"],
                    "failure_code": r["failure_code"],
                    "recovered": bool(r["recovered"]),
                    "synthetic": bool(quality.get("synthetic", False)),
                    "preliminary": bool(quality.get("preliminary", False)),
                    "not_confirmatory": bool(quality.get("not_confirmatory", False)),
                    "bench_version": (quality.get("measurement") or {}).get("bench_version"),
                    "phenotype_hash": r["phenotype_hash"],
                    "cache_context_hash": r["cache_context_hash"],
                    "started_at": r["started_at"],
                    "committed_at": r["committed_at"],
                    "commit_seq": r["commit_seq"],
                }
            )
        return out

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """Every checkpoint of a run, oldest first, with the size of its member lists."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT c.checkpoint_id, c.generation, c.unique_evaluations, c.active_gpu_seconds, c.created_at, "
                "(SELECT COUNT(*) FROM checkpoint_members m WHERE m.checkpoint_id = c.checkpoint_id "
                " AND m.role = 'population') AS population, "
                "(SELECT COUNT(*) FROM checkpoint_members m WHERE m.checkpoint_id = c.checkpoint_id "
                " AND m.role = 'pareto_archive') AS pareto_archive, "
                "(SELECT COUNT(*) FROM checkpoint_members m WHERE m.checkpoint_id = c.checkpoint_id "
                " AND m.role = 'pending') AS pending "
                "FROM checkpoints c WHERE c.run_id = ? ORDER BY c.checkpoint_id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def fronts(self, run_id: str) -> list[dict[str, Any]]:
        """Recorded front snapshots (external archive) with their member candidates."""
        with self._read() as conn:
            snaps = conn.execute(
                "SELECT snapshot_id, active_gpu_seconds, unique_valid_evaluations, hypervolume, created_at "
                "FROM front_snapshots WHERE run_id = ? ORDER BY snapshot_id",
                (run_id,),
            ).fetchall()
            out = []
            for s in snaps:
                members = [
                    r[0]
                    for r in conn.execute(
                        "SELECT candidate_id FROM front_members WHERE snapshot_id = ? ORDER BY candidate_id",
                        (s["snapshot_id"],),
                    )
                ]
                out.append({**dict(s), "candidate_ids": members})
        return out

    def evaluation_details(self, run_id: str) -> list[dict[str, Any]]:
        """What the evaluator recorded next to E, L, V, S for each valid original attempt.

        Accuracy comes from E for every run; top-5 and the macro metrics exist
        only for attempts measured by benchmark v3 or later and are None before.
        """
        with self._read() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            rows = conn.execute(
                "SELECT a.attempt_id, a.candidate_id, c.generation, a.gpu_seconds, a.durations_json, a.quality_json, "
                "m.e_fraction, m.l_ms, m.v_mb, m.s_mb "
                "FROM evaluation_attempts a JOIN candidates c ON c.candidate_id = a.candidate_id "
                "JOIN measurements m ON m.attempt_id = a.attempt_id "
                "WHERE a.run_id = ? AND a.status = 'valid' ORDER BY a.commit_seq, a.rowid",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            quality = json.loads(r["quality_json"]) if r["quality_json"] else {}
            measurement = quality.get("measurement") or {}
            e_info = measurement.get("e") or {}
            latency = measurement.get("l") or {}
            train = quality.get("train") or {}
            parity = (measurement.get("parity") or {}).get("parity_vs_torch_fp32") or {}
            out.append(
                {
                    "attempt_id": r["attempt_id"],
                    "candidate_id": r["candidate_id"],
                    "generation": r["generation"],
                    "e": r["e_fraction"],
                    "l_ms": r["l_ms"],
                    "v_mb": r["v_mb"],
                    "s_mb": r["s_mb"],
                    "gpu_seconds": r["gpu_seconds"],
                    "synthetic": bool(quality.get("synthetic", False)),
                    "bench_version": measurement.get("bench_version"),
                    "accuracy": 1.0 - r["e_fraction"],
                    "top5_accuracy": e_info.get("top5_accuracy"),
                    "macro_precision": e_info.get("macro_precision"),
                    "macro_recall": e_info.get("macro_recall"),
                    "macro_f1": e_info.get("macro_f1"),
                    "torch_accuracy": quality.get("torch_val_accuracy_diagnostic"),
                    "parity_disagreements": parity.get("top1_disagreements"),
                    "parameters": (quality.get("model") or {}).get("parameters"),
                    # of the pruned network, the one that becomes the engine; `parameters` above is the unpruned build
                    "macs": (quality.get("model") or {}).get("macs"),
                    "parameters_pruned": (quality.get("model") or {}).get("parameters_pruned"),
                    "l_p5_ms": latency.get("p5_ms"),
                    "l_p95_ms": latency.get("p95_ms"),
                    "l_iqr_ms": latency.get("iqr_ms"),
                    "epoch_loss": list(train.get("epoch_loss") or []),
                    "epoch_train_acc": list(train.get("epoch_train_acc") or []),
                    "durations": json.loads(r["durations_json"]) if r["durations_json"] else {},
                }
            )
        return out

    def lineage(self, run_id: str) -> list[dict[str, Any]]:
        """Origin and parents of every candidate, as the search recorded them."""
        with self._read() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            rows = conn.execute(
                "SELECT candidate_id, generation, metadata_json FROM candidates WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            metadata = json.loads(r["metadata_json"])
            metadata = metadata if isinstance(metadata, Mapping) else {}
            mutated = metadata.get("mutated")
            out.append(
                {
                    "candidate_id": r["candidate_id"],
                    "generation": r["generation"],
                    "origin": metadata.get("origin"),
                    "parents": list(metadata.get("parents") or []),
                    "crossover": metadata.get("crossover"),
                    "mutated": any(mutated) if isinstance(mutated, list) else mutated,
                }
            )
        return out

    def populations(self, run_id: str) -> list[dict[str, Any]]:
        """Population recorded in each checkpoint, oldest first (one row per checkpoint and member)."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT c.checkpoint_id, c.generation, m.position, m.candidate_id "
                "FROM checkpoints c JOIN checkpoint_members m ON m.checkpoint_id = c.checkpoint_id "
                "WHERE c.run_id = ? AND m.role = 'population' ORDER BY c.checkpoint_id, m.position",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def genomes(self, run_id: str) -> list[dict[str, Any]]:
        """Every candidate of a run with the summary of its recorded expressed genome."""
        with self._read() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            rows = conn.execute(
                "SELECT candidate_id, seq, generation, genome_json, metadata_json, phenotype_hash "
                "FROM candidates WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            record = json.loads(r["genome_json"])
            metadata = json.loads(r["metadata_json"])
            out.append(
                {
                    "candidate_id": r["candidate_id"],
                    "seq": r["seq"],
                    "generation": r["generation"],
                    "origin": metadata.get("origin") if isinstance(metadata, Mapping) else None,
                    "phenotype_hash": r["phenotype_hash"],
                    "repairs": len(record.get("repair_log") or []) if isinstance(record, Mapping) else None,
                    **genome_summary(record.get("expressed_genome") if isinstance(record, Mapping) else None),
                }
            )
        return out

    def expressed_genomes(self, run_id: str) -> dict[str, dict[str, Any]]:
        """candidate_id -> the stored expressed genome (raw JSON), for offline analyses that rebuild the network."""
        with self._read() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run {run_id!r} does not exist")
            rows = conn.execute(
                "SELECT candidate_id, genome_json FROM candidates WHERE run_id = ?", (run_id,)
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            record = json.loads(r["genome_json"])
            expressed = record.get("expressed_genome") if isinstance(record, Mapping) else None
            if isinstance(expressed, Mapping):
                out[r["candidate_id"]] = dict(expressed)
        return out

    def finalist_list(self, list_id: str) -> dict[str, Any]:
        """A frozen finalist list with its selections (and what each one is), and whether its stored hash still holds.

        ``hash_ok`` recomputes the hash from the selections as stored: false means the list was altered after it was
        frozen, and nothing may be opened on the official test from it.
        """
        from cnnlab.tcc.storage.sqlite_store import FinalistSelection, finalist_list_digest

        with self._read() as conn:
            row = conn.execute("SELECT * FROM finalist_lists WHERE list_id = ?", (list_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"finalist list {list_id!r} does not exist")
            rows = conn.execute(
                "SELECT s.selection_id, s.source_run_id, s.candidate_id, s.role, s.criterion, r.seed_block, r.method, "
                "c.phenotype_hash FROM finalist_selections s JOIN runs r ON r.run_id = s.source_run_id "
                "JOIN candidates c ON c.run_id = s.source_run_id AND c.candidate_id = s.candidate_id "
                "WHERE s.list_id = ? ORDER BY s.selection_id",
                (list_id,),
            ).fetchall()
        selections = [dict(r) for r in rows]
        digest = finalist_list_digest(
            row["list_id"],
            row["protocol_id"],
            row["plan_id"],
            row["author"],
            [
                FinalistSelection(r["selection_id"], r["source_run_id"], r["candidate_id"], r["role"], r["criterion"])
                for r in selections
            ],
        )
        return {
            "list_id": row["list_id"],
            "protocol_id": row["protocol_id"],
            "plan_id": row["plan_id"],
            "n_selections": row["n_selections"],
            "list_sha256": row["list_sha256"],
            "author": row["author"],
            "frozen_at": row["frozen_at"],
            "selections": selections,
            "hash_ok": digest == row["list_sha256"] and len(selections) == row["n_selections"],
        }

    def test_evaluations(self) -> list[dict[str, Any]]:
        """Every opening of the official test: one row per engine, never two."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT test_id, run_id, selection_id, attempt_id, engine_sha256, metrics_json, opened_at "
                "FROM test_evaluations ORDER BY opened_at, test_id"
            ).fetchall()
        return [{**dict(r), "metrics": json.loads(r["metrics_json"])} for r in rows]

    def genome_record(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """The stored genome record of a candidate (raw and expressed genome, repair log, hashes)."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT genome_json, phenotype_hash, cache_context_hash, space_hash FROM candidates "
                "WHERE run_id = ? AND candidate_id = ?",
                (run_id, candidate_id),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"candidate {candidate_id!r} of run {run_id!r} does not exist")
        return {
            **json.loads(row["genome_json"]),
            "phenotype_hash": row["phenotype_hash"],
            "cache_context_hash": row["cache_context_hash"],
            "space_hash": row["space_hash"],
        }

    def artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Artifact catalog (relative path, hash, size), of one run or of the whole database.

        With `storage.artifacts.scan_artifacts` this checks the files on disk
        against the catalog without opening a writer.
        """
        query = "SELECT artifact_id, run_id, attempt_id, kind, rel_path, sha256, size_bytes, created_at FROM artifacts"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            query, params = f"{query} WHERE run_id = ?", (run_id,)
        with self._read() as conn:
            return [dict(r) for r in conn.execute(f"{query} ORDER BY rowid", params)]

    # -- jobs --------------------------------------------------------------

    def jobs(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Queue state; `lease_live` compares the lease expiry with `now` (UTC). Empty without a queue."""
        if not self.has_job_queue:
            return []
        now = now or datetime.now(UTC)
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY queue_seq").fetchall()
        out = []
        for j in rows:
            expires = j["lease_expires_at"]
            out.append(
                {
                    "job_id": j["job_id"],
                    "run_id": j["run_id"],
                    "state": j["state"],
                    "priority": j["priority"],
                    "queue_seq": j["queue_seq"],
                    "options": json.loads(j["options_json"]),
                    "lease_owner": j["lease_owner"],
                    "lease_expires_at": expires,
                    "lease_live": None if expires is None else _parse_utc(expires) > now,
                    "heartbeat_at": j["heartbeat_at"],
                    "heartbeat_count": j["heartbeat_count"],
                    "claim_count": j["claim_count"],
                    "cancel_requested": j["cancel_requested_at"] is not None,
                    "cancel_reason": j["cancel_reason"],
                    "final_detail": j["final_detail"],
                    "created_at": j["created_at"],
                    "finished_at": j["finished_at"],
                }
            )
        return out

    def job_events(self, job_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Transitions of a job, oldest first (the last `limit` when given). Empty without a queue."""
        if not self.has_job_queue:
            return []
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM job_events WHERE job_id = ? ORDER BY event_id", (job_id,)).fetchall()
        events = [
            {
                "event_id": e["event_id"],
                "kind": e["kind"],
                "from_state": e["from_state"],
                "to_state": e["to_state"],
                "worker_id": e["worker_id"],
                "detail": json.loads(e["detail_json"]),
                "recorded_at": e["recorded_at"],
            }
            for e in rows
        ]
        return events if limit is None else events[-limit:]

    def sentinel_events(self, run_id: str) -> list[dict[str, Any]]:
        """The sentinel events of a run (measured, drift, failed), oldest first, from every job of the run.

        Empty for a database without the job queue or from before the sentinel (schema below 3).
        """
        if not self.has_job_queue or self.schema_version() < 3:
            return []
        marks = ", ".join("?" for _ in SENTINEL_EVENT_KINDS)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT e.event_id, e.job_id, e.kind, e.detail_json, e.recorded_at FROM job_events e "
                f"JOIN jobs j ON j.job_id = e.job_id WHERE j.run_id = ? AND e.kind IN ({marks}) ORDER BY e.event_id",
                (run_id, *SENTINEL_EVENT_KINDS),
            ).fetchall()
        return [
            {
                "event_id": r["event_id"],
                "job_id": r["job_id"],
                "kind": r["kind"],
                "detail": json.loads(r["detail_json"]),
                "recorded_at": r["recorded_at"],
            }
            for r in rows
        ]


def open_read_model(db_path: str | Path | None) -> ReadModel:
    """Validate an operator-supplied path and open it read-only.

    Every failure is a `ReadModelError` whose `code` is one of `empty_path`,
    `not_found`, `not_a_file`, `not_sqlite`, `not_tcc`, `schema_too_old`,
    `schema_too_new`, so a screen can explain it and say what to do next.
    Nothing is created and nothing is migrated: a missing file stays missing
    and a schema v1 database is read as v1.
    """
    text = "" if db_path is None else str(db_path).strip()
    if not text:
        raise ReadModelError("empty_path", "no database path was given")
    path = Path(text)
    if not path.exists():
        raise ReadModelError("not_found", f"database file does not exist: {path}")
    if not path.is_file():
        raise ReadModelError("not_a_file", f"path is not a file: {path}")
    try:
        model = ReadModel(path)
    except (sqlite3.Error, RuntimeError, OSError, ValueError) as exc:
        raise ReadModelError("not_sqlite", f"cannot open {path} as a WAL SQLite database: {exc}") from exc
    try:
        try:
            version = model.schema_version()
        except sqlite3.DatabaseError as exc:
            code = "not_tcc" if "no such table" in str(exc) else "not_sqlite"
            raise ReadModelError(code, f"{path} is not a TCC v2 database: {exc}") from exc
        if version < MIN_SCHEMA_VERSION:
            raise ReadModelError(
                "schema_too_old", f"{path} has schema v{version}; the read model needs v{MIN_SCHEMA_VERSION} or newer"
            )
        if version > LATEST_VERSION:
            raise ReadModelError(
                "schema_too_new", f"{path} has schema v{version}, newer than this code (v{LATEST_VERSION})"
            )
    except BaseException:
        model.close()
        raise
    return model
