"""Reproducible pilot demo: NSGA-II over (E, L, V, S) with real training and TensorRT on one GPU.

Headless smoke tool of the science layer. It wires the SQLite ``RunStore``,
the :class:`SearchRunner` and the :class:`EngineEvaluator` built by
:mod:`pilot_wiring`, then exports an evidence JSON read back from the database
through a read-only connection. Storage is imported only here; runner and
evaluator stay independent of SQLite. The operation service injects the very
same runner and evaluator through its wiring factory.

Scope, stated in the plan, the manifest and the evidence: **pilot** stage,
FP32 unpruned slice, fixed class-balanced subset of the 45k train split, 5k
validation split for E, preliminary L/V (no sentinel or drift control). The
official test file is never read. This is not the confirmatory protocol.

Robustness:

* every attempt is opened in the database before the evaluator runs, so a
  killed process leaves a ``running`` attempt that ``--resume`` closes as
  ``interrupted`` and retries with a new attempt id;
* ``--resume`` continues an existing run from its last checkpoint, adopting
  evaluations committed after it, and refuses to continue with other code;
* ``--stop-file``: creating that file asks for a planned stop between
  attempts (checkpoint saved, exit status ``stopped``);
* child processes get their own console, so the run survives the end of the
  session that launched it.

Run::

    python -m cnnlab.tcc.model_pipeline.pilot_demo --cifar-root <dir with cifar-100-python>
        --db <file.sqlite> --artifacts <dir> --work <dir> --evidence <file.json>
        [--population 4 --generations 2 --epochs 5] [--resume] [--stop-file <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cnnlab.tcc.contracts import EvaluationRequest, EvaluationResult, Evaluator, TCCEvent, TCCJobSpec
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.pilot_wiring import PilotScience, build_pilot_science
from cnnlab.tcc.science.hashing import digest
from cnnlab.tcc.science.operators import SLICES
from cnnlab.tcc.science.search import (
    DirectContext,
    DirectHashes,
    DirectResume,
    SearchBlockedError,
)

PROTOCOL_ID = "tcc-pilot-demo-v1"
DEMO_VERSION = "tcc-pilot-demo/v2"
EXIT_CODES = {"completed": 0, "stopped": 0, "blocked": 3, "refused": 5}


class DemoRefused(RuntimeError):
    """The demo will not start or resume (busy GPU, other code, existing run without --resume)."""


class AttemptOpener:
    """Opens the attempt row before the evaluator runs, as the operation service does."""

    def __init__(self, inner: Evaluator, store: Any) -> None:
        self._inner = inner
        self._store = store

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self._store.begin_attempt(request.run_id, request.candidate_id, request.attempt_id)
        return self._inner.evaluate(request)


def python_processes() -> int | None:
    """How many python.exe processes exist (diagnostic of concurrent host load; no names kept)."""
    import subprocess

    from cnnlab.tcc.model_pipeline.bench import child_process_kwargs

    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=20,
            **child_process_kwargs(),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.splitlines() if line.lower().startswith('"python.exe"'))


def plan_content(science: PilotScience, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "blocks": [
            {"seed_block": 0, "search_seed": args.search_seed, "train_seed": args.train_seed, "calibration_seed": 0}
        ],
        "methods": ["joint"],
        "budget_gpu_s": float(args.budget_gpu_s),
        "purpose": "exploratory-demo",
        "confirmatory": False,
        "scope": list(science.scope),
        "slice": science.slice_name,
        "note": "Pilot diagnostic run; does not enter H1-H4 nor the frozen reference point.",
        "data": science.dataset_manifest,
        "search": science.search_config.definition(),
        "evaluator": science.evaluator_config.definition(),
        "space_hash": science.space_hash,
        "code_version": science.code_version,
        "wiring_arguments": dict(science.arguments),
        "demo": DEMO_VERSION,
    }


def bootstrap(store: Any, science: PilotScience, args: argparse.Namespace, env: dict[str, Any]) -> TCCJobSpec:
    """Register protocol, environment, dataset, a frozen pilot plan and the run (fresh run only)."""
    from cnnlab.tcc.storage.sqlite_store import SplitDescriptor

    run_id = args.run_id
    store.register_protocol(
        PROTOCOL_ID, digest("tcc.protocol-doc/v1", {"doc": "TCC-01 proposal 1.0", "scope": "pilot demo"}), "pilot"
    )
    env_id = f"env-{digest('tcc.env/v1', science.environment)[:16]}"
    store.register_environment(env_id, {"science": science.environment, "probe": env})
    store.register_dataset(
        science.dataset_id, science.dataset_manifest, {name: SplitDescriptor(**d) for name, d in science.splits.items()}
    )
    plan_id = f"{run_id}-plan"
    store.create_plan(plan_id, PROTOCOL_ID, plan_content(science, args), author="gauss/pilot_demo")
    store.freeze_plan(plan_id)
    job = TCCJobSpec(
        run_id=run_id,
        protocol_id=PROTOCOL_ID,
        plan_id=plan_id,
        dataset_id=science.dataset_id,
        environment_id=env_id,
        space_hash=science.space_hash,
        method="joint",
        stage="pilot",
        seed_block=0,
        search_seed=args.search_seed,
        train_seed=args.train_seed,
        calibration_seed=0,
        budget_gpu_s=float(args.budget_gpu_s),
    )
    store.create_run(job)
    return job


def resume_state(store: Any, science: PilotScience, job: TCCJobSpec) -> DirectResume:
    """Close attempts left running, then describe what the store already holds."""
    plan = store.get_plan(job.plan_id).content
    if plan.get("code_version") != science.code_version:
        raise DemoRefused(
            f"run {job.run_id!r} was started with code {plan.get('code_version')!r}; "
            f"this checkout is {science.code_version!r}"
        )
    for key, current in (
        ("data", science.dataset_manifest),
        ("search", science.search_config.definition()),
        ("evaluator", science.evaluator_config.definition()),
    ):
        if plan.get(key) != current:
            raise DemoRefused(f"run {job.run_id!r} was planned with another {key} configuration")
    store.recover_interrupted_attempts(job.run_id)
    history = store.run_history(job.run_id)
    return DirectResume(
        checkpoint=history.latest_checkpoint,
        attempts_per_candidate=dict(Counter(a.candidate_id for a in history.attempts)),
        retry_candidate_ids=history.unresolved_candidate_ids,
    )


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

    env = probe_environment(touch_cuda=False)
    science = build_pilot_science(
        {
            "cifar_root": str(args.cifar_root),
            "work_dir": str(args.work),
            "train_per_class": args.train_per_class,
            "subset_seed": args.subset_seed,
            "epochs": args.epochs,
            "population": args.population,
            "generations": args.generations,
            "slice": args.slice,
        }
    )
    busy = science.evaluator.gpu_guard() if science.evaluator.gpu_guard is not None else []
    if busy:
        raise DemoRefused(f"GPU busy, refusing to start: {busy}")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    host = {"python_processes_at_start": python_processes(), "pid": os.getpid()}
    store = SQLiteRunStore(Path(args.db), Path(args.artifacts))
    try:
        job = store.get_run(run_id)
        if job is None:
            if args.resume:
                raise DemoRefused(f"--resume given but run {run_id!r} does not exist in {args.db}")
            job = bootstrap(store, science, args, env)
            resume = DirectResume()
        elif not args.resume:
            raise DemoRefused(f"run {run_id!r} already exists; pass --resume to continue it")
        else:
            resume = resume_state(store, science, job)
        hashes = store.run_hashes(run_id)
        events_path = work / "events.jsonl"
        started = time.monotonic()
        stop_file = Path(args.stop_file) if args.stop_file else None

        def emit(event: TCCEvent) -> None:
            with open(events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(event), sort_keys=True, default=str) + "\n")
            stamp = f"[{time.monotonic() - started:7.1f}s]"
            if event.kind == "candidate_evaluated":
                f = event.payload.get("fitness")
                shown = (
                    "invalid " + str(event.payload.get("failure_code"))
                    if f is None
                    else (f"E={f['e']:.4f} L={f['l_ms']:.4f}ms V={f['v_mb']:.3f}MB S={f['s_mb']:.3f}MB")
                )
                print(f"{stamp} {event.candidate_id} {shown}", flush=True)
            elif event.kind in ("candidate_interrupted", "candidate_adopted", "search_blocked"):
                print(f"{stamp} {event.kind} {event.candidate_id} {json.dumps(event.payload, default=str)}", flush=True)
            elif event.kind in ("search_started", "search_resumed", "checkpoint", "search_completed"):
                print(f"{stamp} {event.kind} {json.dumps(event.payload, default=str)[:160]}", flush=True)

        ctx = DirectContext(
            job=job,
            store=store,
            evaluator=AttemptOpener(science.evaluator, store),
            resume=resume,
            hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
            artifact_root=Path(args.artifacts),
            emit=emit,
            should_stop=lambda: stop_file is not None and stop_file.exists(),
            active_gpu_seconds=lambda: store.active_gpu_seconds(run_id),
            unique_valid_evaluations=lambda: store.unique_valid_evaluations(run_id),
        )
        detail = ""
        try:
            status = science.runner(ctx)
        except SearchBlockedError as exc:
            status, detail = "blocked", str(exc)
        integrity = store.check_integrity(hash_files=True)
        host["python_processes_at_end"] = python_processes()
        evidence = {
            "demo": DEMO_VERSION,
            "status": status,
            "detail": detail,
            "resumed": bool(args.resume),
            "session_wall_s": time.monotonic() - started,
            "run_id": run_id,
            "labels": list(science.scope),
            "slice": science.slice_name,
            "official_test_read": False,
            "code_version": science.code_version,
            "system": {"environment_probe": env, "science": science.environment, "host": host},
            "val_file_sha256": science.evaluator.val_file_sha256,
            "integrity_ok": integrity.ok,
            "integrity": {
                "sqlite": list(integrity.sqlite_messages),
                "running_attempts": list(integrity.running_attempts),
                "ledger_mismatches": list(integrity.ledger_mismatches),
                "valid_without_measurement": list(integrity.valid_without_measurement),
            },
        }
    finally:
        store.close()
    evidence["database_sha256"] = hashlib.sha256(Path(args.db).read_bytes()).hexdigest()
    evidence["sql"] = dump_run(Path(args.db), run_id)
    return evidence


def dump_run(db_path: Path, run_id: str) -> dict[str, Any]:
    """Read-only snapshot of the run's rows for the evidence file."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:

        def rows(query: str, *params: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

        return {
            "run": rows(
                "SELECT run_id, method, stage, seed_block, search_seed, train_seed, space_hash, "
                "budget_gpu_s, plan_id, plan_sha256, protocol_sha256, dataset_id, environment_id "
                "FROM runs WHERE run_id = ?",
                run_id,
            ),
            "by_generation": rows(
                "SELECT c.generation, a.status, a.failure_stage, a.failure_code, COUNT(*) AS attempts, "
                "SUM(a.gpu_seconds) AS gpu_seconds FROM evaluation_attempts a "
                "JOIN candidates c ON c.candidate_id = a.candidate_id WHERE a.run_id = ? "
                "GROUP BY c.generation, a.status, a.failure_stage, a.failure_code ORDER BY c.generation",
                run_id,
            ),
            "candidates": rows(
                "SELECT candidate_id, generation, phenotype_hash, cache_context_hash, "
                "metadata_json FROM candidates WHERE run_id = ? ORDER BY seq",
                run_id,
            ),
            "attempts": rows(
                "SELECT a.attempt_id, a.candidate_id, a.attempt_number, a.status, a.gpu_seconds, "
                "a.failure_stage, a.failure_code, a.source_attempt_id, a.durations_json, "
                "m.e_fraction, m.l_ms, m.v_mb, m.s_mb "
                "FROM evaluation_attempts a LEFT JOIN measurements m ON m.attempt_id = a.attempt_id "
                "WHERE a.run_id = ? ORDER BY a.commit_seq",
                run_id,
            ),
            "artifacts": rows(
                "SELECT attempt_id, kind, rel_path, sha256, size_bytes FROM artifacts "
                "WHERE run_id = ? ORDER BY attempt_id, kind",
                run_id,
            ),
            "checkpoints": rows(
                "SELECT checkpoint_id, generation, unique_evaluations, active_gpu_seconds, "
                "payload_sha256, created_at FROM checkpoints WHERE run_id = ? "
                "ORDER BY checkpoint_id",
                run_id,
            ),
            "front_snapshots": rows("SELECT * FROM front_snapshots WHERE run_id = ? ORDER BY snapshot_id", run_id),
            "front_members": rows(
                "SELECT m.* FROM front_members m JOIN front_snapshots s "
                "ON s.snapshot_id = m.snapshot_id WHERE s.run_id = ? "
                "ORDER BY m.snapshot_id",
                run_id,
            ),
            "gpu_seconds_total": conn.execute(
                "SELECT COALESCE(SUM(gpu_seconds), 0) FROM gpu_segments WHERE run_id = ?", (run_id,)
            ).fetchone()[0],
            "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="scratch dir outside the artifact root")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-id", default="pilot-demo-joint-b0")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2, help="offspring generations after the initial one")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--search-seed", type=int, default=1001)
    parser.add_argument("--train-seed", type=int, default=2001)
    parser.add_argument("--budget-gpu-s", type=float, default=3600.0)
    parser.add_argument(
        "--slice",
        choices=tuple(SLICES),
        default="fp32",
        help="search slice: all FP32, FP32/FP16 per block, or per region (stem, blocks, head)",
    )
    parser.add_argument("--resume", action="store_true", help="continue an existing run from its checkpoint")
    parser.add_argument("--stop-file", type=Path, help="planned stop between attempts when this file exists")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        evidence = run_demo(args)
    except DemoRefused as exc:
        print(f"refused: {exc}", file=sys.stderr, flush=True)
        return EXIT_CODES["refused"]
    tmp = args.evidence.with_name(args.evidence.name + ".tmp")
    tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, args.evidence)
    print(f"status={evidence['status']} integrity_ok={evidence['integrity_ok']} evidence={args.evidence}", flush=True)
    return EXIT_CODES[evidence["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
