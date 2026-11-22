"""Process control for interfaces: drive the delivered CLI in child processes.

An interface (TUI) never imports the science front and never runs GPU work
in its own process. `prepare` runs as a short child process; `start` spawns
a **detached worker** (own process group, hidden console, output to a log
file), so closing the interface does not kill the worker and a worker crash
does not take the interface down. Whether a worker is alive is decided by
its lease in the database, never by the interface that launched it.

Everything here goes through `python -m cnnlab.tcc.operation.demo`, so
the guards of the CLI (synthetic wiring, code version, single GPU job) apply
unchanged. Status is read with `open_read_model` (read-only).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import cnnlab
from cnnlab.tcc.operation.demo import DemoRunSpec, demo_status
from cnnlab.tcc.storage.errors import StorageError
from cnnlab.tcc.storage.jobs import JobQueue, JobRecord
from cnnlab.tcc.storage.readmodel import open_read_model
from cnnlab.tcc.storage.schema import LATEST_VERSION
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

CLI_MODULE = "cnnlab.tcc.operation.demo"
LOG_DIRNAME = "logs"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


WARNING_PREFIX = "warning:"


class MigrationNeededError(StorageError):
    """Writing would migrate a database of an older schema; interfaces never do that implicitly."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a short CLI command (prepare, status...)."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def message(self) -> str:
        """Last meaningful line: the CLI prints one clean `error: ...` line on refusal."""
        text = self.stderr if not self.ok and self.stderr.strip() else self.stdout
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else f"exit code {self.returncode}"

    @property
    def warnings(self) -> tuple[str, ...]:
        """`warning: ...` lines of the output, which `message` (the last line) would hide."""
        return tuple(
            line.strip()[len(WARNING_PREFIX) :].strip()
            for line in self.stdout.splitlines()
            if line.strip().startswith(WARNING_PREFIX)
        )


@dataclass(slots=True)
class WorkerHandle:
    """A detached worker launched by this session (the lease, not this, proves liveness)."""

    pid: int
    target: str
    log_path: Path
    command: tuple[str, ...]
    started_at: str
    process: subprocess.Popen[bytes] = field(repr=False)

    def poll(self) -> int | None:
        return self.process.poll()

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


def cli_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment in which a child imports the same `cnnlab` as this process."""
    src = str(Path(cnnlab.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env["PYTHONUNBUFFERED"] = "1"  # log lines reach the file as they happen
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env


def cli_command(*args: str | os.PathLike[str]) -> list[str]:
    return [sys.executable, "-m", CLI_MODULE, *(str(a) for a in args)]


def default_worker_id() -> str:
    return f"tui-{platform.node() or 'host'}"


def load_spec(spec_path: str | Path) -> DemoRunSpec:
    """Read and validate a spec file; errors say what is wrong with it."""
    text = str(spec_path).strip()
    if not text:
        raise ValueError("no spec file was given")
    path = Path(text)
    if not path.is_file():
        raise ValueError(f"spec file does not exist: {path}")
    try:
        return DemoRunSpec.load(path)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"spec file {path} is not a valid demo spec: {type(exc).__name__}: {exc}") from exc


def is_testing_factory(factory: str | None) -> bool:
    """Heuristic for the interface only (the CLI checks `wiring.synthetic` itself)."""
    module = (factory or "").partition(":")[0]
    return module.endswith(".testing") or ".testing." in module


def run_prepare(
    db_path: Path,
    artifact_root: Path,
    spec_path: Path,
    *,
    allow_synthetic: bool = False,
    repo_root: Path | None = None,
    timeout_s: float = 900.0,
    extra_env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run `prepare` in a child process (the factory may import heavy science code)."""
    ensure_current_schema(db_path)
    args: list[str | os.PathLike[str]] = ["prepare", "--db", db_path, "--artifacts", artifact_root, "--spec", spec_path]
    if repo_root is not None:
        args += ["--repo-root", repo_root]
    if allow_synthetic:
        args.append("--allow-synthetic")
    command = cli_command(*args)
    proc = subprocess.run(
        command,
        env=cli_environment(extra_env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        stdin=subprocess.DEVNULL,
    )
    return CommandResult(tuple(command), proc.returncode, proc.stdout, proc.stderr)


def _detached_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        # Own hidden console and process group: closing the interface's console or
        # pressing Ctrl+C there does not reach the worker.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def spawn_worker(
    db_path: Path,
    artifact_root: Path,
    *,
    spec_path: Path | None = None,
    job_id: str | None = None,
    run_id: str | None = None,
    worker_id: str | None = None,
    lease_s: float = 120.0,
    heartbeat_s: float | None = None,
    allow_code_change: bool = False,
    allow_synthetic: bool = False,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> WorkerHandle:
    """Start `start --spec`, `--job-id` or `--run-id` as a detached worker writing to a log file."""
    if sum(value is not None for value in (spec_path, job_id, run_id)) != 1:
        raise ValueError("give exactly one of spec_path, job_id or run_id")
    db_path, artifact_root = Path(db_path), Path(artifact_root)
    ensure_current_schema(db_path)
    if job_id is not None:
        target, chosen = job_id, ["--job-id", job_id]
    elif run_id is not None:
        target, chosen = run_id, ["--run-id", run_id]
    else:
        target, chosen = Path(spec_path).stem, ["--spec", spec_path]  # type: ignore[arg-type, list-item]
    args: list[str | os.PathLike[str]] = ["start", "--db", db_path, "--artifacts", artifact_root]
    args += chosen
    args += ["--worker-id", worker_id or default_worker_id(), "--lease-s", str(lease_s)]
    if heartbeat_s is not None:
        args += ["--heartbeat-s", str(heartbeat_s)]
    if repo_root is not None:
        args += ["--repo-root", repo_root]
    if allow_code_change:
        args.append("--allow-code-change")
    if allow_synthetic:
        args.append("--allow-synthetic")
    command = cli_command(*args)

    started = datetime.now(UTC)
    logs = Path(log_dir) if log_dir is not None else db_path.parent / LOG_DIRNAME
    if logs.resolve() == artifact_root.resolve() or artifact_root.resolve() in logs.resolve().parents:
        raise ValueError("worker logs must live outside the artifact root (they are not artifacts)")
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"worker-{started.strftime('%Y%m%dT%H%M%S%f')}-{_UNSAFE.sub('_', target)[:80]}"
    log_path, attempt = logs / f"{stem}.log", 1
    while True:  # one log per launch: never append to the log of another worker
        try:
            log = open(log_path, "xb")
            break
        except FileExistsError:
            attempt += 1
            log_path = logs / f"{stem}.{attempt}.log"
    with log:
        header = f"# {started.isoformat().replace('+00:00', 'Z')} launched: {' '.join(command)}\n"
        log.write(header.encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            command,
            env=cli_environment(extra_env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **_detached_kwargs(),
        )
    return WorkerHandle(
        pid=process.pid,
        target=target,
        log_path=log_path,
        command=tuple(command),
        started_at=started.isoformat().replace("+00:00", "Z"),
        process=process,
    )


def ensure_current_schema(db_path: Path) -> None:
    """Refuse to write through an interface to a database that a writer would first migrate.

    Opening a writer applies pending migrations. On a database that already
    holds evidence that must be a deliberate step (backup, then the CLI), never
    a side effect of pressing a key. A database that does not exist yet is fine.
    """
    if not Path(db_path).exists():
        return
    with open_read_model(db_path) as rm:
        version = rm.schema_version()
    if version < LATEST_VERSION:
        raise MigrationNeededError(
            f"{db_path} has schema v{version}; this code writes v{LATEST_VERSION}. Operating on it would migrate "
            "the database. Back it up and migrate deliberately with the CLI, or use another database"
        )


def request_cancel(db_path: Path, artifact_root: Path, job_id: str, *, reason: str) -> JobRecord:
    """Cooperative cancel: one short write, no lease. Never creates nor migrates a database."""
    open_read_model(db_path).close()  # a missing or foreign file is refused before any writer opens it
    ensure_current_schema(db_path)
    with SQLiteRunStore(Path(db_path), Path(artifact_root)) as store:
        return JobQueue(store).request_cancel(job_id, reason=reason)


def read_status(db_path: Path, *, run_id: str | None = None, events: int = 6) -> dict[str, Any]:
    """Status plus, when asked, evaluations, checkpoints and fronts of one run (read-only)."""
    status = demo_status(db_path, events=events)
    known = {j["run_id"] for j in status["jobs"]} | {r["run"]["run_id"] for r in status["runs_without_job"]}
    evaluations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    genomes: list[dict[str, Any]] = []
    if run_id is not None and run_id in known:
        with open_read_model(db_path) as rm:
            evaluations = rm.evaluations(run_id)
            checkpoints = rm.checkpoints(run_id)
            fronts = rm.fronts(run_id)
            genomes = rm.genomes(run_id)
    return {
        **status,
        "run_id": run_id,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "fronts": fronts,
        "genomes": genomes,
    }


def tail(path: Path, *, max_lines: int = 30, max_bytes: int = 65536) -> list[str]:
    """Last lines of a log file; empty when the file does not exist yet."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line may be cut in the middle
    return lines[-max_lines:]


def latest_log(db_path: Path, *, log_dir: Path | None = None) -> Path | None:
    """Most recent worker log next to the database, if any (e.g. after reopening the TUI)."""
    logs = Path(log_dir) if log_dir is not None else Path(db_path).parent / LOG_DIRNAME
    candidates: Sequence[Path] = (
        sorted(logs.glob("worker-*.log"), key=lambda p: p.stat().st_mtime) if logs.is_dir() else []
    )
    return candidates[-1] if candidates else None
