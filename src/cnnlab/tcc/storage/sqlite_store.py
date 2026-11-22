"""SQLite implementation of the v2 `RunStore` port.

One `SQLiteRunStore` is the single writer of a database file. Every public
write is one `BEGIN IMMEDIATE` transaction: an evaluation commit writes the
terminal attempt, its measurement, the GPU-ledger reconciliation and the
artifact catalog together or not at all. Readers for analysis use
`connection.open_reader` and never this class.

Beyond the shared port (`create_run`, `register_candidate`, `find_cached`,
`commit_evaluation`, `save_checkpoint`, `load_checkpoint`,
`record_gpu_segment`, `record_front`) the class exposes internal APIs needed
by operation code: reference registration, plans, `begin_attempt`, recovery,
history queries, finalist lists and the official-test gate.

Single writer per run (migration 2): while a run has a job in state
`running`, every write touching that run must come from the thread that holds
the job's lease (`lease_fence`). The check runs inside the same `BEGIN
IMMEDIATE` transaction as the write, so a worker whose lease was taken over
cannot commit anything afterwards, and nobody else can close its attempts
while the lease is alive.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

from cnnlab.tcc.contracts import (
    AttemptStatus,
    EvaluationResult,
    GenomeRecord,
    SearchCheckpoint,
    TCCJobSpec,
)
from cnnlab.tcc.storage.artifacts import (
    ArtifactScan,
    normalize_rel_path,
    safe_component,
    scan_artifacts,
    verify_file,
)
from cnnlab.tcc.storage.canonical import (
    canonical_json,
    canonical_sha256,
    checkpoint_from_payload,
    checkpoint_to_payload,
    fitness_to_payload,
    format_utc,
    genome_from_payload,
    genome_to_payload,
    is_sha256_hex,
    result_from_payload,
    result_to_payload,
    sha256_hex,
    to_jsonable,
    utc_now,
)
from cnnlab.tcc.storage.connection import open_writer
from cnnlab.tcc.storage.errors import (
    ArtifactIntegrityError,
    AttemptStateError,
    CheckpointMismatchError,
    ConsistencyError,
    IdempotencyConflictError,
    ImmutableRecordError,
    LeaseHeldError,
    LeaseLostError,
    NotFoundError,
    OfficialTestAccessError,
    StorageError,
)
from cnnlab.tcc.storage.plans import (
    PlanExport,
    PlanRecord,
    check_job_against_plan,
    plan_body,
    validate_plan_content,
)
from cnnlab.tcc.storage.schema import migrate

JOB_SPEC_FORMAT = "tcc-job-spec/1"
FINALIST_LIST_FORMAT = "tcc-finalist-list/1"
# Tolerance for float sums in the GPU ledger (seconds).
GPU_EPSILON_S = 1e-6
SPLITS: tuple[str, ...] = ("train", "val", "test", "calibration")
SEARCH_SPLITS: frozenset[str] = frozenset({"train", "val", "calibration"})
RECOVERED_FAILURE_CODE = "recovered_after_crash"
LEASE_EXPIRED_FAILURE_CODE = "lease_expired"

AttemptState = Literal["running", "valid", "invalid", "interrupted"]
FinalistRole = Literal["menor_E", "equilibrio", "H1"]
_TERMINAL_STATUSES = frozenset(get_args(AttemptStatus))


@dataclass(frozen=True, slots=True)
class LeaseFence:
    """Proof of the lease a writer thread holds: job id + token from the claim."""

    job_id: str
    lease_token: str


@dataclass(frozen=True, slots=True)
class SplitDescriptor:
    """Fingerprint of one split; the indices themselves live outside SQLite."""

    sha256: str
    n_examples: int
    ref: str


@dataclass(frozen=True, slots=True)
class RunHashes:
    protocol_hash: str
    plan_hash: str
    space_hash: str


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    run_id: str
    seq: int
    generation: int | None
    genome: GenomeRecord
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    run_id: str
    candidate_id: str
    attempt_number: int
    status: AttemptState
    ledger_gpu_seconds: float
    recovered: bool
    started_at: str
    committed_at: str | None
    commit_seq: int | None
    result: EvaluationResult | None


@dataclass(frozen=True, slots=True)
class RunHistory:
    """Everything needed to rebuild a run's operational state from SQL alone."""

    job: TCCJobSpec
    hashes: RunHashes
    candidates: tuple[CandidateRecord, ...]
    attempts: tuple[AttemptRecord, ...]
    active_gpu_seconds: float
    unique_valid_evaluations: int
    unresolved_candidate_ids: tuple[str, ...]
    latest_checkpoint: SearchCheckpoint | None


@dataclass(frozen=True, slots=True)
class FinalistSelection:
    selection_id: str
    source_run_id: str
    candidate_id: str
    role: FinalistRole
    criterion: str


def finalist_list_digest(
    list_id: str, protocol_id: str, plan_id: str, author: str, selections: Sequence[FinalistSelection]
) -> str:
    """SHA-256 of a finalist list as frozen: what ``freeze_finalist_list`` stores and a reader can recompute."""
    return canonical_sha256(
        {
            "format": FINALIST_LIST_FORMAT,
            "list_id": list_id,
            "protocol_id": protocol_id,
            "plan_id": plan_id,
            "author": author,
            "selections": [
                {
                    "selection_id": s.selection_id,
                    "source_run_id": s.source_run_id,
                    "candidate_id": s.candidate_id,
                    "role": s.role,
                    "criterion": s.criterion,
                }
                for s in sorted(selections, key=lambda s: s.selection_id)
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    sqlite_messages: tuple[str, ...]
    foreign_key_violations: tuple[tuple[Any, ...], ...]
    valid_without_measurement: tuple[str, ...]
    ledger_mismatches: tuple[str, ...]
    running_attempts: tuple[str, ...]
    artifacts: ArtifactScan

    @property
    def ok(self) -> bool:
        """True when nothing is corrupt; running attempts alone are not corruption."""
        return (
            self.sqlite_messages == ("ok",)
            and not self.foreign_key_violations
            and not self.valid_without_measurement
            and not self.ledger_mismatches
            and self.artifacts.ok
        )


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def job_to_payload(job: TCCJobSpec) -> dict[str, Any]:
    return {
        "format": JOB_SPEC_FORMAT,
        "run_id": job.run_id,
        "protocol_id": job.protocol_id,
        "plan_id": job.plan_id,
        "dataset_id": job.dataset_id,
        "environment_id": job.environment_id,
        "space_hash": job.space_hash,
        "method": job.method,
        "stage": job.stage,
        "seed_block": job.seed_block,
        "search_seed": job.search_seed,
        "train_seed": job.train_seed,
        "calibration_seed": job.calibration_seed,
        "budget_gpu_s": float(job.budget_gpu_s),
    }


def job_from_payload(payload: Mapping[str, Any]) -> TCCJobSpec:
    fields = {k: v for k, v in payload.items() if k != "format"}
    return TCCJobSpec(**fields)


def _validate_result(result: EvaluationResult) -> None:
    for name in ("run_id", "candidate_id", "attempt_id"):
        _nonempty(name, getattr(result, name))
    if result.status not in _TERMINAL_STATUSES:
        raise ValueError(f"status {result.status!r} is not a terminal attempt status")
    for key, value in result.durations_s.items():
        _nonempty("durations_s key", key)
        _real(f"durations_s[{key!r}]", value)
    failure = (result.failure_stage, result.failure_code, result.failure_detail)
    if result.status == "valid" and any(v is not None for v in failure):
        raise ValueError("valid attempts carry no failure fields")
    if result.status == "invalid" and not (result.failure_stage and result.failure_code):
        raise ValueError("invalid attempts need failure_stage and failure_code (TCC-04)")
    if set(result.artifact_paths) != set(result.artifact_sha256):
        raise ArtifactIntegrityError("artifact_paths and artifact_sha256 must have the same kinds")
    seen: set[str] = set()
    for kind, rel_path in result.artifact_paths.items():
        safe_component(kind)
        normalize_rel_path(rel_path)
        if not is_sha256_hex(result.artifact_sha256[kind]):
            raise ArtifactIntegrityError(f"artifact {kind!r} needs a lowercase SHA-256")
        if rel_path.casefold() in seen:
            raise ArtifactIntegrityError(f"artifact path {rel_path!r} used twice")
        seen.add(rel_path.casefold())


class SQLiteRunStore:
    """Durable, single-writer `RunStore` backed by SQLite (WAL, FK on, FULL sync)."""

    def __init__(
        self,
        db_path: Path,
        artifact_root: Path,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._db_path = Path(db_path)
        self._artifact_root = Path(artifact_root)
        # The artifact scan treats every uncataloged file under the root as an orphan.
        if self._artifact_root.resolve() in self._db_path.resolve().parents:
            raise ValueError("the database must live outside the artifact root")
        self._clock = clock
        self._lock = threading.RLock()
        self._fence = threading.local()
        self._conn = open_writer(self._db_path)
        try:
            migrate(self._conn)
        except BaseException:
            self._conn.close()
            raise
        self._conn.row_factory = sqlite3.Row

    # -- lifecycle -------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteRunStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _now(self) -> str:
        return format_utc(self._clock())

    def now(self) -> datetime:
        """Wall clock used for timestamps and lease expiry (injectable for tests)."""
        return self._clock()

    # -- single-writer fence ---------------------------------------------

    @contextmanager
    def lease_fence(self, fence: LeaseFence) -> Iterator[None]:
        """Writes in this thread are made on behalf of `fence` (checked per transaction)."""
        previous = getattr(self._fence, "value", None)
        self._fence.value = fence
        try:
            yield
        finally:
            self._fence.value = previous

    @property
    def current_fence(self) -> LeaseFence | None:
        return getattr(self._fence, "value", None)

    def _check_run_lease(self, conn: sqlite3.Connection, run_id: str) -> str | None:
        """Refuse a write to `run_id` unless this thread holds its live lease.

        Returns the fenced job id (None when writing without a lease to a run
        that no worker holds, e.g. tests or offline administration).
        """
        fence = self.current_fence
        if fence is not None:
            job = conn.execute(
                "SELECT run_id, state, lease_token FROM jobs WHERE job_id = ?", (fence.job_id,)
            ).fetchone()
            if job is None or job["state"] != "running" or job["lease_token"] != fence.lease_token:
                raise LeaseLostError(f"lease of job {fence.job_id!r} is no longer held by this worker")
            if job["run_id"] != run_id:
                raise ConsistencyError(f"job {fence.job_id!r} is fenced to run {job['run_id']!r}, not {run_id!r}")
            return fence.job_id
        holder = conn.execute(
            "SELECT job_id, lease_owner FROM jobs WHERE run_id = ? AND state = 'running'", (run_id,)
        ).fetchone()
        if holder is not None:
            raise LeaseHeldError(
                f"run {run_id!r} is leased by job {holder['job_id']!r} (worker {holder['lease_owner']!r})"
            )
        return None

    def _rollback(self) -> None:
        if self._conn.in_transaction:
            self._conn.execute("ROLLBACK")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction; SQLite constraint failures become `ConsistencyError`."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except sqlite3.IntegrityError as exc:
                self._rollback()
                raise ConsistencyError(str(exc)) from exc
            except BaseException:
                self._rollback()
                raise
            try:
                conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise

    @contextmanager
    def _snapshot(self) -> Iterator[sqlite3.Connection]:
        """Consistent multi-statement read."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            finally:
                self._rollback()

    # -- small helpers ---------------------------------------------------

    @staticmethod
    def _insert(conn: sqlite3.Connection, table: str, row: Mapping[str, Any]) -> None:
        names = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        conn.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(row.values()))

    def _insert_or_verify(
        self,
        conn: sqlite3.Connection,
        table: str,
        key_col: str,
        row: Mapping[str, Any],
        compare: Sequence[str],
    ) -> bool:
        """Insert, or accept an identical existing row; returns True if inserted."""
        cols = ", ".join(compare)
        existing = conn.execute(f"SELECT {cols} FROM {table} WHERE {key_col} = ?", (row[key_col],)).fetchone()
        if existing is not None:
            if tuple(existing) != tuple(row[c] for c in compare):
                raise IdempotencyConflictError(
                    f"{table}.{key_col}={row[key_col]!r} already exists with different content"
                )
            return False
        self._insert(conn, table, row)
        return True

    @staticmethod
    def _require_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"run {run_id!r} does not exist")
        return row

    @staticmethod
    def _attempt_ledger(conn: sqlite3.Connection, attempt_id: str) -> float:
        return float(
            conn.execute(
                "SELECT COALESCE(SUM(gpu_seconds), 0.0) FROM gpu_segments WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
        )

    @staticmethod
    def _run_ledger(conn: sqlite3.Connection, run_id: str) -> float:
        return float(
            conn.execute(
                "SELECT COALESCE(SUM(gpu_seconds), 0.0) FROM gpu_segments WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        )

    @staticmethod
    def _unique_valid(conn: sqlite3.Connection, run_id: str) -> int:
        return int(
            conn.execute(
                "SELECT COUNT(DISTINCT c.cache_context_hash) FROM evaluation_attempts a "
                "JOIN candidates c ON c.candidate_id = a.candidate_id "
                "WHERE a.run_id = ? AND a.status = 'valid' AND a.source_attempt_id IS NULL",
                (run_id,),
            ).fetchone()[0]
        )

    # -- reference rows --------------------------------------------------

    def register_protocol(
        self,
        protocol_id: str,
        document_sha256: str,
        status: Literal["pilot", "confirmed", "amended"],
        *,
        predecessor_id: str | None = None,
        justification: str | None = None,
    ) -> None:
        """Record a protocol version by the hash of its document; idempotent."""
        _nonempty("protocol_id", protocol_id)
        if not is_sha256_hex(document_sha256):
            raise ValueError("document_sha256 must be a lowercase SHA-256")
        row = {
            "protocol_id": protocol_id,
            "document_sha256": document_sha256,
            "status": status,
            "predecessor_id": predecessor_id,
            "justification": justification,
            "created_at": self._now(),
        }
        with self._tx() as conn:
            self._insert_or_verify(
                conn,
                "protocol_versions",
                "protocol_id",
                row,
                ("protocol_id", "document_sha256", "status", "predecessor_id", "justification"),
            )

    def register_environment(self, environment_id: str, manifest: Mapping[str, Any]) -> str:
        """Record the hardware/software manifest; returns its hash."""
        _nonempty("environment_id", environment_id)
        manifest_json = canonical_json(manifest)
        row = {
            "environment_id": environment_id,
            "manifest_json": manifest_json,
            "manifest_sha256": sha256_hex(manifest_json),
            "created_at": self._now(),
        }
        with self._tx() as conn:
            self._insert_or_verify(conn, "environments", "environment_id", row, ("environment_id", "manifest_sha256"))
        return row["manifest_sha256"]

    def get_environment(self, environment_id: str) -> dict[str, Any]:
        """The manifest as recorded (to explain why two runs differ)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT manifest_json FROM environments WHERE environment_id = ?", (environment_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"environment {environment_id!r} does not exist")
        return json.loads(row[0])

    def register_dataset(
        self,
        dataset_id: str,
        manifest: Mapping[str, Any],
        splits: Mapping[str, SplitDescriptor],
    ) -> str:
        """Record split fingerprints (train/val/test/calibration); returns the dataset hash."""
        _nonempty("dataset_id", dataset_id)
        if set(splits) != set(SPLITS):
            raise ValueError(f"splits must be exactly {SPLITS}")
        split_payload: dict[str, dict[str, Any]] = {}
        for name in SPLITS:
            split = splits[name]
            if not is_sha256_hex(split.sha256):
                raise ValueError(f"split {name!r} needs a lowercase SHA-256")
            if isinstance(split.n_examples, bool) or not isinstance(split.n_examples, int) or split.n_examples <= 0:
                raise ValueError(f"split {name!r} needs a positive n_examples")
            _nonempty(f"split {name!r} ref", split.ref)
            split_payload[name] = {"sha256": split.sha256, "n_examples": split.n_examples, "ref": split.ref}
        manifest_json = canonical_json(manifest)
        digest = canonical_sha256({"manifest": manifest, "splits": split_payload})
        row = {
            "dataset_id": dataset_id,
            "manifest_json": manifest_json,
            "manifest_sha256": digest,
            "created_at": self._now(),
        }
        with self._tx() as conn:
            inserted = self._insert_or_verify(conn, "datasets", "dataset_id", row, ("dataset_id", "manifest_sha256"))
            if inserted:
                for name in SPLITS:
                    self._insert(
                        conn, "dataset_splits", {"dataset_id": dataset_id, "split": name, **split_payload[name]}
                    )
        return digest

    # -- plans -----------------------------------------------------------

    @staticmethod
    def _plan_from_row(row: sqlite3.Row) -> PlanRecord:
        return PlanRecord(
            plan_id=row["plan_id"],
            protocol_id=row["protocol_id"],
            version=row["version"],
            status=row["status"],
            content=json.loads(row["content_json"]),
            plan_sha256=row["plan_sha256"],
            author=row["author"],
            amends_plan_id=row["amends_plan_id"],
            amendment_reason=row["amendment_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            frozen_at=row["frozen_at"],
        )

    @staticmethod
    def _body_of(row: sqlite3.Row, content: Mapping[str, Any]) -> str:
        return plan_body(
            plan_id=row["plan_id"],
            protocol_id=row["protocol_id"],
            version=row["version"],
            author=row["author"],
            amends_plan_id=row["amends_plan_id"],
            amendment_reason=row["amendment_reason"],
            content=content,
        )

    def create_plan(
        self,
        plan_id: str,
        protocol_id: str,
        content: Mapping[str, Any],
        *,
        author: str,
        amends_plan_id: str | None = None,
        amendment_reason: str | None = None,
    ) -> PlanRecord:
        """Create a `draft`; an amendment links to the frozen plan it replaces."""
        _nonempty("plan_id", plan_id)
        _nonempty("author", author)
        validate_plan_content(content)
        if (amends_plan_id is None) != (amendment_reason is None):
            raise ValueError("an amendment needs both amends_plan_id and amendment_reason")
        now = self._now()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone():
                raise IdempotencyConflictError(f"plan {plan_id!r} already exists")
            if not conn.execute("SELECT 1 FROM protocol_versions WHERE protocol_id = ?", (protocol_id,)).fetchone():
                raise NotFoundError(f"protocol {protocol_id!r} does not exist")
            if amends_plan_id is not None:
                prior = conn.execute(
                    "SELECT status FROM experiment_plans WHERE plan_id = ?", (amends_plan_id,)
                ).fetchone()
                if prior is None:
                    raise NotFoundError(f"plan {amends_plan_id!r} does not exist")
                if prior["status"] != "frozen":
                    raise ConsistencyError("only a frozen plan can be amended")
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM experiment_plans WHERE protocol_id = ?",
                (protocol_id,),
            ).fetchone()[0]
            body = plan_body(
                plan_id=plan_id,
                protocol_id=protocol_id,
                version=version,
                author=author,
                amends_plan_id=amends_plan_id,
                amendment_reason=amendment_reason,
                content=content,
            )
            self._insert(
                conn,
                "experiment_plans",
                {
                    "plan_id": plan_id,
                    "protocol_id": protocol_id,
                    "version": version,
                    "status": "draft",
                    "content_json": canonical_json(content),
                    "plan_sha256": sha256_hex(body),
                    "author": author,
                    "amends_plan_id": amends_plan_id,
                    "amendment_reason": amendment_reason,
                    "created_at": now,
                    "updated_at": now,
                    "frozen_at": None,
                },
            )
        return self.get_plan(plan_id)

    def update_plan(self, plan_id: str, content: Mapping[str, Any]) -> PlanRecord:
        """Replace the content of a draft; a frozen plan raises `ImmutableRecordError`."""
        validate_plan_content(content)
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"plan {plan_id!r} does not exist")
            if row["status"] == "frozen":
                raise ImmutableRecordError(f"plan {plan_id!r} is frozen; create an amendment")
            conn.execute(
                "UPDATE experiment_plans SET content_json = ?, plan_sha256 = ?, updated_at = ? WHERE plan_id = ?",
                (canonical_json(content), sha256_hex(self._body_of(row, content)), self._now(), plan_id),
            )
        return self.get_plan(plan_id)

    def freeze_plan(self, plan_id: str) -> PlanRecord:
        """Make a plan immutable; freezing again is a no-op. Hash does not change."""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"plan {plan_id!r} does not exist")
            if row["status"] == "draft":
                content = json.loads(row["content_json"])
                validate_plan_content(content)
                if sha256_hex(self._body_of(row, content)) != row["plan_sha256"]:
                    raise StorageError(f"plan {plan_id!r} hash does not match its content")
                now = self._now()
                conn.execute(
                    "UPDATE experiment_plans SET status = 'frozen', frozen_at = ?, updated_at = ? WHERE plan_id = ?",
                    (now, now, plan_id),
                )
        return self.get_plan(plan_id)

    def get_plan(self, plan_id: str) -> PlanRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"plan {plan_id!r} does not exist")
        return self._plan_from_row(row)

    def export_plan(self, plan_id: str) -> PlanExport:
        """Canonical bytes + hash to archive outside the database."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (plan_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"plan {plan_id!r} does not exist")
        body = self._body_of(row, json.loads(row["content_json"])).encode("utf-8")
        if sha256_hex(body) != row["plan_sha256"]:
            raise StorageError(f"plan {plan_id!r} hash does not match its content")
        return PlanExport(
            plan_id=plan_id,
            status=row["status"],
            frozen_at=row["frozen_at"],
            body=body,
            plan_sha256=row["plan_sha256"],
        )

    # -- runs and candidates (RunStore port) -----------------------------

    def create_run(self, job: TCCJobSpec, *, finalist_list_id: str | None = None) -> None:
        """Register a run against a frozen plan; re-creating the same spec is a no-op.

        `finalist_list_id` is required for `stage="finalist"` (the frozen list
        that authorizes the official test) and forbidden otherwise.
        """
        payload = job_to_payload(job)
        spec_json = canonical_json(payload)
        spec_sha = sha256_hex(spec_json)
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT job_spec_sha256, finalist_list_id FROM runs WHERE run_id = ?", (job.run_id,)
            ).fetchone()
            if existing is not None:
                if (existing[0], existing[1]) == (spec_sha, finalist_list_id):
                    return
                raise IdempotencyConflictError(f"run {job.run_id!r} exists with a different spec")
            protocol = conn.execute(
                "SELECT document_sha256 FROM protocol_versions WHERE protocol_id = ?", (job.protocol_id,)
            ).fetchone()
            if protocol is None:
                raise NotFoundError(f"protocol {job.protocol_id!r} does not exist")
            plan = conn.execute("SELECT * FROM experiment_plans WHERE plan_id = ?", (job.plan_id,)).fetchone()
            if plan is None:
                raise NotFoundError(f"plan {job.plan_id!r} does not exist")
            if plan["status"] != "frozen":
                raise ConsistencyError(f"plan {job.plan_id!r} must be frozen before any run")
            if plan["protocol_id"] != job.protocol_id:
                raise ConsistencyError("run protocol differs from plan protocol")
            check_job_against_plan(job, json.loads(plan["content_json"]))
            for table, key, value in (
                ("datasets", "dataset_id", job.dataset_id),
                ("environments", "environment_id", job.environment_id),
            ):
                if not conn.execute(f"SELECT 1 FROM {table} WHERE {key} = ?", (value,)).fetchone():
                    raise NotFoundError(f"{key} {value!r} does not exist")
            if job.stage == "finalist":
                if finalist_list_id is None:
                    raise OfficialTestAccessError("finalist runs need a frozen finalist list")
                flist = conn.execute(
                    "SELECT plan_id FROM finalist_lists WHERE list_id = ?", (finalist_list_id,)
                ).fetchone()
                if flist is None:
                    raise NotFoundError(f"finalist list {finalist_list_id!r} does not exist")
                if flist["plan_id"] != job.plan_id:
                    raise ConsistencyError("finalist list belongs to another plan")
            elif finalist_list_id is not None:
                raise ConsistencyError("only finalist runs reference a finalist list")
            else:
                clash = conn.execute(
                    "SELECT run_id FROM runs WHERE plan_id = ? AND stage = ? AND method = ? AND seed_block = ?",
                    (job.plan_id, job.stage, job.method, job.seed_block),
                ).fetchone()
                if clash is not None:
                    raise ConsistencyError(
                        f"plan cell {job.stage}/{job.method}/block {job.seed_block} already is run {clash[0]!r}"
                    )
            self._insert(
                conn,
                "runs",
                {
                    "run_id": job.run_id,
                    "protocol_id": job.protocol_id,
                    "protocol_sha256": protocol["document_sha256"],
                    "plan_id": job.plan_id,
                    "plan_sha256": plan["plan_sha256"],
                    "dataset_id": job.dataset_id,
                    "environment_id": job.environment_id,
                    "space_hash": job.space_hash,
                    "method": job.method,
                    "stage": job.stage,
                    "seed_block": job.seed_block,
                    "search_seed": job.search_seed,
                    "train_seed": job.train_seed,
                    "calibration_seed": job.calibration_seed,
                    "budget_gpu_s": float(job.budget_gpu_s),
                    "finalist_list_id": finalist_list_id,
                    "job_spec_json": spec_json,
                    "job_spec_sha256": spec_sha,
                    "created_at": self._now(),
                },
            )

    def get_run(self, run_id: str) -> TCCJobSpec | None:
        with self._lock:
            row = self._conn.execute("SELECT job_spec_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else job_from_payload(json.loads(row[0]))

    def run_hashes(self, run_id: str) -> RunHashes:
        """Hashes a checkpoint must carry (the job spec only has ids)."""
        with self._lock:
            row = self._require_run(self._conn, run_id)
        return RunHashes(row["protocol_sha256"], row["plan_sha256"], row["space_hash"])

    def register_candidate(
        self,
        run_id: str,
        candidate_id: str,
        genome: GenomeRecord,
        metadata: Mapping[str, Any],
    ) -> None:
        """Persist a proposed candidate; identical re-registration is a no-op."""
        _nonempty("candidate_id", candidate_id)
        _nonempty("phenotype_hash", genome.phenotype_hash)
        _nonempty("cache_context_hash", genome.cache_context_hash)
        genome_payload = genome_to_payload(genome)
        meta = to_jsonable(metadata, _path="$.metadata")
        generation = meta.get("generation")
        if generation is not None:
            _count("metadata['generation']", generation)
        record_sha = canonical_sha256(
            {"run_id": run_id, "candidate_id": candidate_id, "genome": genome_payload, "metadata": meta}
        )
        with self._tx() as conn:
            run = self._require_run(conn, run_id)
            self._check_run_lease(conn, run_id)
            existing = conn.execute(
                "SELECT record_sha256 FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] == record_sha:
                    return
                raise IdempotencyConflictError(f"candidate {candidate_id!r} exists with different content")
            if genome.space_hash != run["space_hash"]:
                raise ConsistencyError("candidate space_hash differs from the run space_hash")
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM candidates WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            self._insert(
                conn,
                "candidates",
                {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "seq": seq,
                    "generation": generation,
                    "genome_json": canonical_json(genome_payload),
                    "phenotype_hash": genome.phenotype_hash,
                    "cache_context_hash": genome.cache_context_hash,
                    "space_hash": genome.space_hash,
                    "metadata_json": canonical_json(meta),
                    "record_sha256": record_sha,
                    "created_at": self._now(),
                },
            )

    # -- attempts --------------------------------------------------------

    def _insert_running_attempt(self, conn: sqlite3.Connection, run_id: str, candidate_id: str, attempt_id: str) -> int:
        other = conn.execute(
            "SELECT attempt_id FROM evaluation_attempts WHERE candidate_id = ? AND status = 'running'",
            (candidate_id,),
        ).fetchone()
        if other is not None:
            raise AttemptStateError(f"candidate {candidate_id!r} already has running attempt {other[0]!r}")
        number = conn.execute(
            "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM evaluation_attempts WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()[0]
        self._insert(
            conn,
            "evaluation_attempts",
            {
                "attempt_id": attempt_id,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "attempt_number": number,
                "status": "running",
                "started_at": self._now(),
            },
        )
        return int(number)

    @staticmethod
    def _require_candidate(conn: sqlite3.Connection, run_id: str, candidate_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT cache_context_hash FROM candidates WHERE run_id = ? AND candidate_id = ?",
            (run_id, candidate_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"candidate {candidate_id!r} is not registered in run {run_id!r}")
        return row

    def begin_attempt(self, run_id: str, candidate_id: str, attempt_id: str) -> int:
        """Open a `running` attempt before GPU work so a crash leaves a visible trace.

        Returns the attempt number (1, 2, ...). Calling again for the same running
        attempt is a no-op; a terminal attempt cannot be reopened.
        """
        _nonempty("attempt_id", attempt_id)
        with self._tx() as conn:
            self._require_run(conn, run_id)
            job_id = self._check_run_lease(conn, run_id)
            self._require_candidate(conn, run_id, candidate_id)
            row = conn.execute(
                "SELECT run_id, candidate_id, status, attempt_number FROM evaluation_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is not None:
                if (row["run_id"], row["candidate_id"], row["status"]) == (run_id, candidate_id, "running"):
                    return int(row["attempt_number"])
                raise AttemptStateError(f"attempt {attempt_id!r} already exists ({row['status']})")
            number = self._insert_running_attempt(conn, run_id, candidate_id, attempt_id)
            self._link_attempt_to_job(conn, run_id, attempt_id, job_id)
            return number

    def record_gpu_segment(self, run_id: str, attempt_id: str, gpu_seconds: float) -> None:
        """Append active GPU time to a running attempt (ledger is append-only)."""
        seconds = _real("gpu_seconds", gpu_seconds)
        with self._tx() as conn:
            row = conn.execute(
                "SELECT run_id, status FROM evaluation_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist; call begin_attempt first")
            if row["run_id"] != run_id:
                raise ConsistencyError(f"attempt {attempt_id!r} belongs to run {row['run_id']!r}")
            if row["status"] != "running":
                raise AttemptStateError(f"attempt {attempt_id!r} is already {row['status']}")
            self._check_run_lease(conn, run_id)
            self._insert_segment(conn, run_id, attempt_id, seconds, "runner")

    def _insert_segment(
        self, conn: sqlite3.Connection, run_id: str, attempt_id: str, seconds: float, source: str
    ) -> None:
        self._insert(
            conn,
            "gpu_segments",
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "gpu_seconds": seconds,
                "source": source,
                "recorded_at": self._now(),
            },
        )

    def _check_cache_hit(self, conn: sqlite3.Connection, result: EvaluationResult, context_hash: str) -> None:
        src = conn.execute(
            "SELECT a.status, a.source_attempt_id, c.cache_context_hash, "
            "m.e_fraction, m.l_ms, m.v_mb, m.s_mb "
            "FROM evaluation_attempts a "
            "JOIN candidates c ON c.candidate_id = a.candidate_id "
            "LEFT JOIN measurements m ON m.attempt_id = a.attempt_id "
            "WHERE a.attempt_id = ? AND a.run_id = ?",
            (result.source_attempt_id, result.run_id),
        ).fetchone()
        if src is None:
            raise ConsistencyError("cache source must be an attempt of the same run")
        if src["status"] != "valid" or src["source_attempt_id"] is not None:
            raise ConsistencyError("cache source must be a valid, original (non-cached) attempt")
        if src["cache_context_hash"] != context_hash:
            raise ConsistencyError("cache hit requires an identical cache_context_hash")
        if result.status != "valid":
            raise ConsistencyError("a cache hit reuses a valid result")
        stored = {"e": src["e_fraction"], "l_ms": src["l_ms"], "v_mb": src["v_mb"], "s_mb": src["s_mb"]}
        if fitness_to_payload(result.fitness) != stored:
            raise ConsistencyError("cache hit fitness differs from the source measurement")
        if result.gpu_seconds != 0:
            raise ConsistencyError("a cache hit consumes no GPU time")
        if result.artifact_paths:
            raise ConsistencyError("a cache hit references the source artifacts, not new ones")

    def _write_measurement(self, conn: sqlite3.Connection, result: EvaluationResult, now: str) -> None:
        fitness = fitness_to_payload(result.fitness)
        assert fitness is not None  # guaranteed by EvaluationResult for valid
        self._insert(
            conn,
            "measurements",
            {
                "measurement_id": f"{result.attempt_id}#measurement",
                "attempt_id": result.attempt_id,
                "e_fraction": fitness["e"],
                "l_ms": fitness["l_ms"],
                "v_mb": fitness["v_mb"],
                "s_mb": fitness["s_mb"],
                "created_at": now,
            },
        )

    def _write_artifact(
        self, conn: sqlite3.Connection, result: EvaluationResult, kind: str, size: int, now: str
    ) -> None:
        rel_path = result.artifact_paths[kind]
        clash = conn.execute("SELECT attempt_id FROM artifacts WHERE rel_path = ?", (rel_path,)).fetchone()
        if clash is not None:
            raise ArtifactIntegrityError(f"artifact path {rel_path!r} already cataloged by {clash[0]!r}")
        self._insert(
            conn,
            "artifacts",
            {
                "artifact_id": f"{result.attempt_id}#{kind}",
                "run_id": result.run_id,
                "attempt_id": result.attempt_id,
                "kind": kind,
                "rel_path": rel_path,
                "sha256": result.artifact_sha256[kind],
                "size_bytes": size,
                "created_at": now,
            },
        )

    def _prior_terminal(self, attempt_id: str, digest: str) -> bool:
        """True if already committed with the same bytes; raises if bytes differ."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status, result_sha256 FROM evaluation_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        if row is None or row["status"] == "running":
            return False
        if row["result_sha256"] == digest:
            return True
        raise IdempotencyConflictError(
            f"attempt {attempt_id!r} was committed as {row['status']} with different content"
        )

    def commit_evaluation(self, result: EvaluationResult) -> None:
        """Atomically make an attempt terminal with its measurement, ledger and artifacts.

        Same `attempt_id` with byte-identical content is a no-op; different
        content raises `IdempotencyConflictError`. Artifact files must already
        be published under the artifact root with the declared SHA-256.
        """
        _validate_result(result)
        body = canonical_json(result_to_payload(result))
        digest = sha256_hex(body)
        if self.current_fence is not None:
            # A fenced worker learns first that it lost the lease, not about a conflict.
            with self._snapshot() as conn:
                self._check_run_lease(conn, result.run_id)
        if self._prior_terminal(result.attempt_id, digest):
            return
        sizes = {
            kind: verify_file(self._artifact_root, result.artifact_paths[kind], result.artifact_sha256[kind])
            for kind in sorted(result.artifact_paths)
        }
        with self._tx() as conn:
            self._require_run(conn, result.run_id)
            job_id = self._check_run_lease(conn, result.run_id)
            candidate = self._require_candidate(conn, result.run_id, result.candidate_id)
            row = conn.execute(
                "SELECT run_id, candidate_id, status, result_sha256 FROM evaluation_attempts WHERE attempt_id = ?",
                (result.attempt_id,),
            ).fetchone()
            if row is None:
                self._insert_running_attempt(conn, result.run_id, result.candidate_id, result.attempt_id)
                self._link_attempt_to_job(conn, result.run_id, result.attempt_id, job_id)
            elif row["status"] != "running":
                if row["result_sha256"] == digest:
                    return
                raise IdempotencyConflictError(f"attempt {result.attempt_id!r} committed concurrently")
            elif (row["run_id"], row["candidate_id"]) != (result.run_id, result.candidate_id):
                raise AttemptStateError(f"attempt {result.attempt_id!r} belongs to another candidate")
            if result.source_attempt_id is not None:
                self._check_cache_hit(conn, result, candidate["cache_context_hash"])
            gap = float(result.gpu_seconds) - self._attempt_ledger(conn, result.attempt_id)
            if gap < -GPU_EPSILON_S:
                raise ConsistencyError(
                    f"attempt {result.attempt_id!r} reports {result.gpu_seconds}s but the ledger already has more"
                )
            if gap > GPU_EPSILON_S:
                self._insert_segment(conn, result.run_id, result.attempt_id, gap, "commit_reconcile")
            now = self._now()
            seq = conn.execute("SELECT COALESCE(MAX(commit_seq), 0) + 1 FROM evaluation_attempts").fetchone()[0]
            conn.execute(
                "UPDATE evaluation_attempts SET status = ?, source_attempt_id = ?, gpu_seconds = ?, "
                "failure_stage = ?, failure_code = ?, failure_detail = ?, durations_json = ?, "
                "quality_json = ?, result_json = ?, result_sha256 = ?, committed_at = ?, commit_seq = ? "
                "WHERE attempt_id = ?",
                (
                    result.status,
                    result.source_attempt_id,
                    float(result.gpu_seconds),
                    result.failure_stage,
                    result.failure_code,
                    result.failure_detail,
                    canonical_json(result.durations_s),
                    canonical_json(result.quality),
                    body,
                    digest,
                    now,
                    seq,
                    result.attempt_id,
                ),
            )
            if result.status == "valid" and result.source_attempt_id is None:
                self._write_measurement(conn, result, now)
            for kind, size in sizes.items():
                self._write_artifact(conn, result, kind, size, now)

    def find_cached(self, run_id: str, cache_context_hash: str) -> EvaluationResult | None:
        """Earliest valid original evaluation of this context *in this run only*."""
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            row = conn.execute(
                "SELECT a.result_json FROM evaluation_attempts a "
                "JOIN candidates c ON c.candidate_id = a.candidate_id "
                "WHERE a.run_id = ? AND c.cache_context_hash = ? "
                "AND a.status = 'valid' AND a.source_attempt_id IS NULL "
                "ORDER BY a.commit_seq LIMIT 1",
                (run_id, cache_context_hash),
            ).fetchone()
        return None if row is None else result_from_payload(json.loads(row[0]))

    def recover_interrupted_attempts(self, run_id: str) -> tuple[str, ...]:
        """Close every `running` attempt of a run as `interrupted`, keeping its GPU time.

        Call only when no evaluator of this run is alive (after a crash/restart).
        While a worker holds the run's lease this raises `LeaseHeldError`; the
        holder itself may call it under its `lease_fence`. The attempt keeps
        the time already in the ledger; retrying the candidate requires a new
        `attempt_id`.
        """
        with self._tx() as conn:
            self._require_run(conn, run_id)
            self._check_run_lease(conn, run_id)
            return self._close_running_attempts(
                conn,
                run_id,
                failure_code=RECOVERED_FAILURE_CODE,
                failure_detail="attempt was running when the writer restarted",
            )

    def _close_running_attempts(
        self, conn: sqlite3.Connection, run_id: str, *, failure_code: str, failure_detail: str
    ) -> tuple[str, ...]:
        """Terminal `interrupted` for every running attempt of `run_id` (inside a transaction)."""
        closed: list[str] = []
        rows = conn.execute(
            "SELECT attempt_id, candidate_id FROM evaluation_attempts "
            "WHERE run_id = ? AND status = 'running' ORDER BY rowid",
            (run_id,),
        ).fetchall()
        for row in rows:
            ledger = self._attempt_ledger(conn, row["attempt_id"])
            synthetic = EvaluationResult(
                run_id=run_id,
                candidate_id=row["candidate_id"],
                attempt_id=row["attempt_id"],
                status="interrupted",
                fitness=None,
                gpu_seconds=ledger,
                failure_stage="unknown",
                failure_code=failure_code,
                failure_detail=failure_detail,
            )
            body = canonical_json(result_to_payload(synthetic))
            seq = conn.execute("SELECT COALESCE(MAX(commit_seq), 0) + 1 FROM evaluation_attempts").fetchone()[0]
            conn.execute(
                "UPDATE evaluation_attempts SET status = 'interrupted', gpu_seconds = ?, "
                "failure_stage = ?, failure_code = ?, failure_detail = ?, durations_json = ?, "
                "quality_json = ?, result_json = ?, result_sha256 = ?, recovered = 1, "
                "committed_at = ?, commit_seq = ? WHERE attempt_id = ?",
                (
                    ledger,
                    synthetic.failure_stage,
                    synthetic.failure_code,
                    synthetic.failure_detail,
                    canonical_json({}),
                    canonical_json({}),
                    body,
                    sha256_hex(body),
                    self._now(),
                    seq,
                    row["attempt_id"],
                ),
            )
            closed.append(row["attempt_id"])
        return tuple(closed)

    def _link_attempt_to_job(self, conn: sqlite3.Connection, run_id: str, attempt_id: str, job_id: str | None) -> None:
        fence = self.current_fence
        if job_id is None or fence is None:
            return
        self._insert(
            conn,
            "job_attempts",
            {"attempt_id": attempt_id, "run_id": run_id, "job_id": job_id, "lease_token": fence.lease_token},
        )

    def record_job_gpu_segment(self, run_id: str, attempt_id: str, mono_start: float, mono_end: float) -> float:
        """Ledger segment measured by the lease holder with a monotonic clock.

        Same effect as `record_gpu_segment` plus an audit row tying the time to
        the job, the lease (worker session) and the monotonic interval. Must be
        called under `lease_fence`. Returns the seconds recorded.
        """
        start, end = float(mono_start), float(mono_end)
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError("monotonic interval must be finite with end >= start")
        fence = self.current_fence
        if fence is None:
            raise LeaseLostError("record_job_gpu_segment requires a lease_fence")
        seconds = end - start
        with self._tx() as conn:
            row = conn.execute(
                "SELECT run_id, status FROM evaluation_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist; call begin_attempt first")
            if row["run_id"] != run_id:
                raise ConsistencyError(f"attempt {attempt_id!r} belongs to run {row['run_id']!r}")
            job_id = self._check_run_lease(conn, run_id)
            if row["status"] != "running":
                raise AttemptStateError(f"attempt {attempt_id!r} is already {row['status']}")
            self._insert_segment(conn, run_id, attempt_id, seconds, "runner")
            segment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            self._insert(
                conn,
                "job_gpu_segments",
                {
                    "segment_id": segment_id,
                    "job_id": job_id,
                    "lease_token": fence.lease_token,
                    "mono_start": start,
                    "mono_end": end,
                },
            )
        return seconds

    # -- checkpoints and fronts ------------------------------------------

    def save_checkpoint(self, checkpoint: SearchCheckpoint) -> None:
        """Append a checkpoint; hashes must match the run and counters never go back."""
        generation = _count("generation", checkpoint.generation)
        unique = _count("unique_evaluations", checkpoint.unique_evaluations)
        active = _real("active_gpu_seconds", checkpoint.active_gpu_seconds)
        payload_json = canonical_json(checkpoint_to_payload(checkpoint))
        with self._tx() as conn:
            run = self._require_run(conn, checkpoint.run_id)
            self._check_run_lease(conn, checkpoint.run_id)
            expected = (run["protocol_sha256"], run["plan_sha256"], run["space_hash"])
            got = (checkpoint.protocol_hash, checkpoint.plan_hash, checkpoint.space_hash)
            if got != expected:
                raise CheckpointMismatchError(f"checkpoint hashes {got} differ from run {expected}")
            last = conn.execute(
                "SELECT generation, unique_evaluations, active_gpu_seconds FROM checkpoints "
                "WHERE run_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (checkpoint.run_id,),
            ).fetchone()
            if last is not None and (
                generation < last["generation"]
                or unique < last["unique_evaluations"]
                or active < last["active_gpu_seconds"] - GPU_EPSILON_S
            ):
                raise ConsistencyError("checkpoint counters cannot go backwards")
            if active > self._run_ledger(conn, checkpoint.run_id) + GPU_EPSILON_S:
                raise ConsistencyError("checkpoint claims more active GPU time than the ledger")
            members = [
                (role, position, candidate_id)
                for role, ids in (
                    ("population", checkpoint.population),
                    ("pareto_archive", checkpoint.pareto_archive),
                    ("pending", checkpoint.pending_candidates),
                )
                for position, candidate_id in enumerate(ids)
            ]
            for candidate_id in {m[2] for m in members}:
                self._require_candidate(conn, checkpoint.run_id, candidate_id)
            cur = conn.execute(
                "INSERT INTO checkpoints (run_id, generation, unique_evaluations, active_gpu_seconds, "
                "protocol_hash, plan_hash, space_hash, payload_json, payload_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    checkpoint.run_id,
                    generation,
                    unique,
                    active,
                    checkpoint.protocol_hash,
                    checkpoint.plan_hash,
                    checkpoint.space_hash,
                    payload_json,
                    sha256_hex(payload_json),
                    self._now(),
                ),
            )
            checkpoint_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO checkpoint_members (checkpoint_id, run_id, role, position, candidate_id) "
                "VALUES (?, ?, ?, ?, ?)",
                [(checkpoint_id, checkpoint.run_id, role, pos, cid) for role, pos, cid in members],
            )

    def _load_checkpoint(self, conn: sqlite3.Connection, run_id: str) -> SearchCheckpoint | None:
        run = self._require_run(conn, run_id)
        row = conn.execute(
            "SELECT checkpoint_id, payload_json, payload_sha256 FROM checkpoints "
            "WHERE run_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        if sha256_hex(row["payload_json"]) != row["payload_sha256"]:
            raise StorageError(f"checkpoint {row['checkpoint_id']} payload hash mismatch")
        checkpoint = checkpoint_from_payload(json.loads(row["payload_json"]))
        members: dict[str, list[str]] = {"population": [], "pareto_archive": [], "pending": []}
        for m in conn.execute(
            "SELECT role, candidate_id FROM checkpoint_members WHERE checkpoint_id = ? ORDER BY role, position",
            (row["checkpoint_id"],),
        ):
            members[m["role"]].append(m["candidate_id"])
        if (
            tuple(members["population"]) != checkpoint.population
            or tuple(members["pareto_archive"]) != checkpoint.pareto_archive
            or tuple(members["pending"]) != checkpoint.pending_candidates
        ):
            raise StorageError(f"checkpoint {row['checkpoint_id']} members differ from its payload")
        expected = (run["protocol_sha256"], run["plan_sha256"], run["space_hash"])
        if (checkpoint.protocol_hash, checkpoint.plan_hash, checkpoint.space_hash) != expected:
            raise CheckpointMismatchError("stored checkpoint hashes differ from the run")
        return checkpoint

    def load_checkpoint(self, run_id: str) -> SearchCheckpoint | None:
        """Latest checkpoint, verified against its hash, members and the run hashes."""
        with self._snapshot() as conn:
            return self._load_checkpoint(conn, run_id)

    def record_front(
        self,
        run_id: str,
        candidate_ids: tuple[str, ...],
        active_gpu_seconds: float,
        hypervolume: float | None,
    ) -> None:
        """Snapshot the external Pareto archive; only candidates with a valid attempt."""
        active = _real("active_gpu_seconds", active_gpu_seconds)
        hv = None if hypervolume is None else _real("hypervolume", hypervolume)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("front candidate_ids must be distinct")
        with self._tx() as conn:
            self._require_run(conn, run_id)
            self._check_run_lease(conn, run_id)
            last = conn.execute(
                "SELECT active_gpu_seconds FROM front_snapshots WHERE run_id = ? ORDER BY snapshot_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if last is not None and active < last[0] - GPU_EPSILON_S:
                raise ConsistencyError("front snapshots must be recorded in increasing GPU time")
            if active > self._run_ledger(conn, run_id) + GPU_EPSILON_S:
                raise ConsistencyError("front snapshot claims more active GPU time than the ledger")
            for candidate_id in candidate_ids:
                self._require_candidate(conn, run_id, candidate_id)
                if not conn.execute(
                    "SELECT 1 FROM evaluation_attempts WHERE run_id = ? AND candidate_id = ? AND status = 'valid'",
                    (run_id, candidate_id),
                ).fetchone():
                    raise ConsistencyError(f"candidate {candidate_id!r} has no valid attempt")
            cur = conn.execute(
                "INSERT INTO front_snapshots (run_id, active_gpu_seconds, unique_valid_evaluations, "
                "hypervolume, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, active, self._unique_valid(conn, run_id), hv, self._now()),
            )
            conn.executemany(
                "INSERT INTO front_members (snapshot_id, run_id, candidate_id) VALUES (?, ?, ?)",
                [(cur.lastrowid, run_id, cid) for cid in candidate_ids],
            )

    # -- queries ---------------------------------------------------------

    def active_gpu_seconds(self, run_id: str) -> float:
        """Sum of the ledger, including failed and interrupted attempts."""
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            return self._run_ledger(conn, run_id)

    def unique_valid_evaluations(self, run_id: str) -> int:
        """Distinct cache contexts with a valid original evaluation (cache hits excluded)."""
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            return self._unique_valid(conn, run_id)

    def evaluated_attempt_ids(self, run_id: str) -> list[str]:
        """Attempts that went through the evaluator, in commit order.

        They are the ones carrying the ``operation`` block that the metered evaluator adds to every result:
        cache hits and candidates invalidated by repair never reach the evaluator and have none. The list is
        read from the database, so it is the same after any resume.
        """
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            rows = conn.execute(
                "SELECT attempt_id FROM evaluation_attempts WHERE run_id = ? AND status <> 'running' "
                "AND json_extract(quality_json, '$.operation') IS NOT NULL ORDER BY commit_seq, rowid",
                (run_id,),
            ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            seq=row["seq"],
            generation=row["generation"],
            genome=genome_from_payload(json.loads(row["genome_json"])),
            metadata=json.loads(row["metadata_json"]),
        )

    def _attempts(self, conn: sqlite3.Connection, where: str, params: tuple[Any, ...]) -> tuple[AttemptRecord, ...]:
        rows = conn.execute(
            "SELECT a.*, (SELECT COALESCE(SUM(s.gpu_seconds), 0.0) FROM gpu_segments s "
            "WHERE s.attempt_id = a.attempt_id) AS ledger "
            f"FROM evaluation_attempts a WHERE {where} ORDER BY a.rowid",
            params,
        ).fetchall()
        return tuple(
            AttemptRecord(
                attempt_id=r["attempt_id"],
                run_id=r["run_id"],
                candidate_id=r["candidate_id"],
                attempt_number=r["attempt_number"],
                status=r["status"],
                ledger_gpu_seconds=float(r["ledger"]),
                recovered=bool(r["recovered"]),
                started_at=r["started_at"],
                committed_at=r["committed_at"],
                commit_seq=r["commit_seq"],
                result=None if r["result_json"] is None else result_from_payload(json.loads(r["result_json"])),
            )
            for r in rows
        )

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        with self._snapshot() as conn:
            found = self._attempts(conn, "a.attempt_id = ?", (attempt_id,))
        return found[0] if found else None

    def list_attempts(self, run_id: str) -> tuple[AttemptRecord, ...]:
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            return self._attempts(conn, "a.run_id = ?", (run_id,))

    def list_candidates(self, run_id: str) -> tuple[CandidateRecord, ...]:
        with self._snapshot() as conn:
            self._require_run(conn, run_id)
            rows = conn.execute("SELECT * FROM candidates WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return tuple(self._candidate_from_row(r) for r in rows)

    def run_history(self, run_id: str) -> RunHistory:
        """Consistent snapshot of a run for resume and audit."""
        with self._snapshot() as conn:
            run = self._require_run(conn, run_id)
            candidates = tuple(
                self._candidate_from_row(r)
                for r in conn.execute("SELECT * FROM candidates WHERE run_id = ? ORDER BY seq", (run_id,))
            )
            attempts = self._attempts(conn, "a.run_id = ?", (run_id,))
            resolved = {a.candidate_id for a in attempts if a.status in ("valid", "invalid")}
            return RunHistory(
                job=job_from_payload(json.loads(run["job_spec_json"])),
                hashes=RunHashes(run["protocol_sha256"], run["plan_sha256"], run["space_hash"]),
                candidates=candidates,
                attempts=attempts,
                active_gpu_seconds=self._run_ledger(conn, run_id),
                unique_valid_evaluations=self._unique_valid(conn, run_id),
                unresolved_candidate_ids=tuple(c.candidate_id for c in candidates if c.candidate_id not in resolved),
                latest_checkpoint=self._load_checkpoint(conn, run_id),
            )

    # -- official test separation ----------------------------------------

    def dataset_splits_for_run(self, run_id: str) -> dict[str, SplitDescriptor]:
        """Splits a run may read: search/pilot never receive the official test split."""
        with self._snapshot() as conn:
            run = self._require_run(conn, run_id)
            rows = conn.execute(
                "SELECT split, sha256, n_examples, ref FROM dataset_splits WHERE dataset_id = ?",
                (run["dataset_id"],),
            ).fetchall()
        allowed = set(SPLITS) if run["stage"] == "finalist" else SEARCH_SPLITS
        return {
            r["split"]: SplitDescriptor(r["sha256"], r["n_examples"], r["ref"]) for r in rows if r["split"] in allowed
        }

    def official_test_split(self, run_id: str) -> SplitDescriptor:
        """Official test fingerprint, only for finalist runs (which require a frozen list)."""
        splits = self.dataset_splits_for_run(run_id)
        if "test" not in splits:
            raise OfficialTestAccessError(f"run {run_id!r} is not a finalist run; test split is sealed")
        return splits["test"]

    def freeze_finalist_list(
        self,
        list_id: str,
        *,
        protocol_id: str,
        plan_id: str,
        selections: Sequence[FinalistSelection],
        author: str,
    ) -> str:
        """Freeze the finalist list in one transaction; returns its hash. Idempotent."""
        _nonempty("list_id", list_id)
        _nonempty("author", author)
        if not selections:
            raise ValueError("a finalist list needs at least one selection")
        ordered = sorted(selections, key=lambda s: s.selection_id)
        digest = finalist_list_digest(list_id, protocol_id, plan_id, author, ordered)
        with self._tx() as conn:
            existing = conn.execute("SELECT list_sha256 FROM finalist_lists WHERE list_id = ?", (list_id,)).fetchone()
            if existing is not None:
                if existing[0] == digest:
                    return digest
                raise IdempotencyConflictError(f"finalist list {list_id!r} is frozen with other content")
            for s in ordered:
                run = self._require_run(conn, s.source_run_id)
                if run["stage"] != "confirmatory" or run["plan_id"] != plan_id:
                    raise ConsistencyError(
                        f"selection {s.selection_id!r} must come from a confirmatory run of {plan_id!r}"
                    )
                if s.role == "H1" and run["method"] != "joint":
                    raise ConsistencyError("role H1 is selected only from joint runs")
            self._insert(
                conn,
                "finalist_lists",
                {
                    "list_id": list_id,
                    "protocol_id": protocol_id,
                    "plan_id": plan_id,
                    "n_selections": len(ordered),
                    "list_sha256": digest,
                    "author": author,
                    "frozen_at": self._now(),
                },
            )
            for s in ordered:
                self._insert(
                    conn,
                    "finalist_selections",
                    {
                        "selection_id": s.selection_id,
                        "list_id": list_id,
                        "source_run_id": s.source_run_id,
                        "candidate_id": s.candidate_id,
                        "role": s.role,
                        "criterion": s.criterion,
                    },
                )
        return digest

    def record_test_evaluation(
        self,
        test_id: str,
        *,
        run_id: str,
        selection_id: str,
        attempt_id: str,
        engine_sha256: str,
        metrics: Mapping[str, Any],
    ) -> None:
        """Record the single official-test opening of an engine of a finalist run."""
        _nonempty("test_id", test_id)
        if not is_sha256_hex(engine_sha256):
            raise ValueError("engine_sha256 must be a lowercase SHA-256")
        metrics_json = canonical_json(metrics)
        with self._tx() as conn:
            run = self._require_run(conn, run_id)
            if run["stage"] != "finalist":
                raise OfficialTestAccessError(f"run {run_id!r} is {run['stage']}; official test is sealed")
            self._check_run_lease(conn, run_id)
            existing = conn.execute(
                "SELECT test_id, run_id, selection_id, attempt_id, engine_sha256, metrics_json "
                "FROM test_evaluations WHERE test_id = ? OR engine_sha256 = ?",
                (test_id, engine_sha256),
            ).fetchone()
            if existing is not None:
                if tuple(existing) == (test_id, run_id, selection_id, attempt_id, engine_sha256, metrics_json):
                    return
                raise OfficialTestAccessError(
                    f"engine {engine_sha256[:12]} already opened on the official test ({existing['test_id']!r})"
                )
            selection = conn.execute(
                "SELECT list_id FROM finalist_selections WHERE selection_id = ?", (selection_id,)
            ).fetchone()
            if selection is None or selection["list_id"] != run["finalist_list_id"]:
                raise OfficialTestAccessError("selection is not part of this run's frozen finalist list")
            self._insert(
                conn,
                "test_evaluations",
                {
                    "test_id": test_id,
                    "run_id": run_id,
                    "selection_id": selection_id,
                    "attempt_id": attempt_id,
                    "engine_sha256": engine_sha256,
                    "metrics_json": metrics_json,
                    "opened_at": self._now(),
                },
            )

    # -- integrity -------------------------------------------------------

    def artifact_catalog(self) -> dict[str, str]:
        with self._lock:
            return {r["rel_path"]: r["sha256"] for r in self._conn.execute("SELECT rel_path, sha256 FROM artifacts")}

    def check_integrity(self, *, hash_files: bool = True) -> IntegrityReport:
        """SQLite integrity, FKs, attempt/measurement/ledger invariants and artifact scan."""
        with self._snapshot() as conn:
            messages = tuple(r[0] for r in conn.execute("PRAGMA integrity_check"))
            fk = tuple(tuple(r) for r in conn.execute("PRAGMA foreign_key_check"))
            no_measure = tuple(
                r[0]
                for r in conn.execute(
                    "SELECT a.attempt_id FROM evaluation_attempts a "
                    "LEFT JOIN measurements m ON m.attempt_id = a.attempt_id "
                    "WHERE a.status = 'valid' AND a.source_attempt_id IS NULL AND m.attempt_id IS NULL"
                )
            )
            ledger = tuple(
                r[0]
                for r in conn.execute(
                    "SELECT a.attempt_id FROM evaluation_attempts a "
                    "WHERE a.status <> 'running' AND abs(a.gpu_seconds - (SELECT COALESCE(SUM(s.gpu_seconds), 0.0) "
                    "FROM gpu_segments s WHERE s.attempt_id = a.attempt_id)) > ?",
                    (GPU_EPSILON_S,),
                )
            )
            running = tuple(
                r[0] for r in conn.execute("SELECT attempt_id FROM evaluation_attempts WHERE status = 'running'")
            )
            catalog = {r["rel_path"]: r["sha256"] for r in conn.execute("SELECT rel_path, sha256 FROM artifacts")}
        return IntegrityReport(
            sqlite_messages=messages,
            foreign_key_violations=fk,
            valid_without_measurement=no_measure,
            ledger_mismatches=ledger,
            running_attempts=running,
            artifacts=scan_artifacts(self._artifact_root, catalog, hash_files=hash_files),
        )
