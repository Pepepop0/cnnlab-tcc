"""Durable job queue with a single-writer lease (migration 2).

Lifecycle (TCC-03, "Modelo de execução")::

    queued ──claim──► running ──finish──► done | failed | cancelled
      │                 │  ▲
      │                 │  └──claim── interrupted ◄──(lease expired, observed by a claimer)
      │                 └──release──► queued      (planned stop; pauses do not count)
      └──cancel──► cancelled

Liveness belongs to the worker process: it renews its lease with heartbeats
(`heartbeat`), independently of any TUI or CLI that merely watches the queue.
Closing a viewer never touches a lease. A job becomes `interrupted` only when
a claimer, inside a `BEGIN IMMEDIATE` transaction, sees that the lease expiry
is in the past; the same transaction closes that run's running attempts as
`interrupted` (time kept). From then on the old token fails every fenced
write (`LeaseLostError`), so a worker that was merely slow cannot become a
second writer. Renewal of an expired lease that nobody took over is allowed:
the lease is lost only when somebody else actually observed the expiry.

Expiry uses the store's wall clock (UTC), shared by every process on the
machine; the heartbeat period should be a small fraction of the lease (the
service defaults to 1/8). A forward jump of the system clock larger than the
lease can expire a live worker; the fence then stops it cleanly, and the
interrupted attempt keeps its time.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from cnnlab.tcc.contracts import JobStatus
from cnnlab.tcc.storage.canonical import canonical_json, format_utc, sha256_hex, to_jsonable
from cnnlab.tcc.storage.errors import (
    IdempotencyConflictError,
    JobStateError,
    LeaseLostError,
    NotFoundError,
)
from cnnlab.tcc.storage.schema import JOB_ACTIVE_STATES, JOB_TERMINAL_STATES, SENTINEL_EVENT_KINDS
from cnnlab.tcc.storage.sqlite_store import (
    LEASE_EXPIRED_FAILURE_CODE,
    LeaseFence,
    SQLiteRunStore,
)

JOB_OPTIONS_FORMAT = "tcc-queue-job/1"
TerminalJobState = Literal["done", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    run_id: str
    queue_seq: int
    priority: int
    options: Mapping[str, Any]
    state: JobStatus
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    heartbeat_count: int
    claim_count: int
    cancel_requested_at: str | None
    cancel_reason: str | None
    final_detail: str | None
    created_at: str
    updated_at: str
    finished_at: str | None

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_requested_at is not None


@dataclass(frozen=True, slots=True)
class Lease:
    """What a worker receives from a claim; the token is its write permission."""

    job_id: str
    run_id: str
    worker_id: str
    lease_token: str
    lease_s: float
    expires_at: str
    claim_number: int

    @property
    def fence(self) -> LeaseFence:
        return LeaseFence(self.job_id, self.lease_token)


@dataclass(frozen=True, slots=True)
class ReapedJob:
    job_id: str
    run_id: str
    previous_owner: str
    interrupted_attempt_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JobEvent:
    event_id: int
    job_id: str
    kind: str
    from_state: str | None
    to_state: str
    lease_token: str | None
    worker_id: str | None
    detail: Mapping[str, Any]
    recorded_at: str


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _positive_seconds(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite number of seconds")
    return float(value)


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        run_id=row["run_id"],
        queue_seq=row["queue_seq"],
        priority=row["priority"],
        options=json.loads(row["options_json"]),
        state=row["state"],
        lease_owner=row["lease_owner"],
        lease_token=row["lease_token"],
        lease_expires_at=row["lease_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        heartbeat_count=row["heartbeat_count"],
        claim_count=row["claim_count"],
        cancel_requested_at=row["cancel_requested_at"],
        cancel_reason=row["cancel_reason"],
        final_detail=row["final_detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
    )


class JobQueue:
    """Queue operations over a `SQLiteRunStore` connection (same single writer file).

    Every method is one `BEGIN IMMEDIATE` transaction, so two workers (two
    processes, each with its own store) racing for the same job are
    serialized by SQLite: exactly one claim wins.
    """

    def __init__(self, store: SQLiteRunStore) -> None:
        self._store = store

    @property
    def store(self) -> SQLiteRunStore:
        return self._store

    # -- helpers ---------------------------------------------------------

    def _event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        kind: str,
        from_state: str | None,
        to_state: str,
        *,
        lease_token: str | None = None,
        worker_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO job_events (job_id, kind, from_state, to_state, lease_token, worker_id, "
            "detail_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                kind,
                from_state,
                to_state,
                lease_token,
                worker_id,
                canonical_json(detail or {}),
                self._store._now(),
            ),
        )

    @staticmethod
    def _require_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id!r} does not exist")
        return row

    @staticmethod
    def _require_lease(conn: sqlite3.Connection, lease: Lease) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (lease.job_id,)).fetchone()
        if row is None or row["state"] != "running" or row["lease_token"] != lease.lease_token:
            raise LeaseLostError(f"lease {lease.lease_token[:8]} of job {lease.job_id!r} is no longer held")
        return row

    # -- producer side ---------------------------------------------------

    def enqueue(
        self,
        job_id: str,
        run_id: str,
        *,
        priority: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        """Queue a run for execution; identical re-enqueue is a no-op.

        `options` are operational knobs (checkpoint cadence, ...), frozen with
        the job. The scientific configuration lives in the run and its plan.
        Only one queued/running/interrupted job per run exists at a time.
        """
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("priority must be an int (higher runs first)")
        payload = {"format": JOB_OPTIONS_FORMAT, **to_jsonable(dict(options or {}), _path="$.options")}
        options_json = canonical_json(payload)
        options_sha = sha256_hex(options_json)
        with self._store._tx() as conn:
            existing = conn.execute(
                "SELECT run_id, priority, options_sha256 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing is not None:
                if tuple(existing) != (run_id, priority, options_sha):
                    raise IdempotencyConflictError(f"job {job_id!r} exists with different content")
            else:
                self._store._require_run(conn, run_id)
                active = conn.execute(
                    f"SELECT job_id, state FROM jobs WHERE run_id = ? AND state IN {JOB_ACTIVE_STATES!r}",
                    (run_id,),
                ).fetchone()
                if active is not None:
                    raise JobStateError(
                        f"run {run_id!r} already has active job {active['job_id']!r} ({active['state']})"
                    )
                now = self._store._now()
                seq = conn.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM jobs").fetchone()[0]
                conn.execute(
                    "INSERT INTO jobs (job_id, run_id, queue_seq, priority, options_json, options_sha256, "
                    "state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                    (job_id, run_id, seq, priority, options_json, options_sha, now, now),
                )
                self._event(conn, job_id, "enqueued", None, "queued", detail={"priority": priority})
        return self.get(job_id)

    # -- worker side -----------------------------------------------------

    def _reap_expired(self, conn: sqlite3.Connection, now: datetime) -> list[ReapedJob]:
        reaped: list[ReapedJob] = []
        rows = conn.execute("SELECT * FROM jobs WHERE state = 'running' ORDER BY queue_seq").fetchall()
        for row in rows:
            if _parse_utc(row["lease_expires_at"]) > now:
                continue
            closed = self._store._close_running_attempts(
                conn,
                row["run_id"],
                failure_code=LEASE_EXPIRED_FAILURE_CODE,
                failure_detail=(
                    f"lease of worker {row['lease_owner']!r} expired at {row['lease_expires_at']} "
                    f"(last heartbeat {row['heartbeat_at']})"
                ),
            )
            stamp = format_utc(now)
            conn.execute(
                "UPDATE jobs SET state = 'interrupted', lease_owner = NULL, lease_token = NULL, "
                "lease_expires_at = NULL, updated_at = ? WHERE job_id = ?",
                (stamp, row["job_id"]),
            )
            self._event(
                conn,
                row["job_id"],
                "lease_expired",
                "running",
                "interrupted",
                lease_token=row["lease_token"],
                worker_id=row["lease_owner"],
                detail={
                    "lease_expires_at": row["lease_expires_at"],
                    "last_heartbeat_at": row["heartbeat_at"],
                    "heartbeat_count": row["heartbeat_count"],
                    "interrupted_attempt_ids": list(closed),
                },
            )
            reaped.append(ReapedJob(row["job_id"], row["run_id"], row["lease_owner"], closed))
        return reaped

    def reap_expired(self) -> tuple[ReapedJob, ...]:
        """Mark jobs whose lease expired as `interrupted`; live leases are untouched."""
        with self._store._tx() as conn:
            return tuple(self._reap_expired(conn, self._store.now()))

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_s: float,
        job_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> Lease | None:
        """Atomically take the next runnable job (or `job_id`) and return its lease.

        Runnable: `queued` or `interrupted`, without a cancel request, highest
        priority first, then enqueue order. Expired leases are reaped first
        in the same transaction. Returns None when nothing is runnable or
        when another job holds a live lease: one GPU job at a time, enforced
        also by a unique index on the running state.
        """
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        lease_s = _positive_seconds("lease_s", lease_s)
        with self._store._tx() as conn:
            now = self._store.now()
            self._reap_expired(conn, now)
            if conn.execute("SELECT 1 FROM jobs WHERE state = 'running'").fetchone() is not None:
                return None
            query = "SELECT * FROM jobs WHERE state IN ('queued', 'interrupted') AND cancel_requested_at IS NULL"
            params: tuple[Any, ...] = ()
            if job_id is not None:
                query += " AND job_id = ?"
                params = (job_id,)
            row = conn.execute(query + " ORDER BY priority DESC, queue_seq LIMIT 1", params).fetchone()
            if row is None:
                return None
            token = secrets.token_hex(16)
            expires = format_utc(now + timedelta(seconds=lease_s))
            stamp = format_utc(now)
            conn.execute(
                "UPDATE jobs SET state = 'running', lease_owner = ?, lease_token = ?, lease_expires_at = ?, "
                "heartbeat_at = ?, claim_count = claim_count + 1, updated_at = ? WHERE job_id = ?",
                (worker_id, token, expires, stamp, stamp, row["job_id"]),
            )
            self._event(
                conn,
                row["job_id"],
                "claimed",
                row["state"],
                "running",
                lease_token=token,
                worker_id=worker_id,
                detail={"lease_s": lease_s, "expires_at": expires, **dict(detail or {})},
            )
            return Lease(
                job_id=row["job_id"],
                run_id=row["run_id"],
                worker_id=worker_id,
                lease_token=token,
                lease_s=lease_s,
                expires_at=expires,
                claim_number=row["claim_count"] + 1,
            )

    def heartbeat(self, lease: Lease) -> JobRecord:
        """Renew the lease from the worker process; `LeaseLostError` if taken over."""
        with self._store._tx() as conn:
            self._require_lease(conn, lease)
            now = self._store.now()
            stamp = format_utc(now)
            conn.execute(
                "UPDATE jobs SET lease_expires_at = ?, heartbeat_at = ?, heartbeat_count = heartbeat_count + 1, "
                "updated_at = ? WHERE job_id = ?",
                (format_utc(now + timedelta(seconds=lease.lease_s)), stamp, stamp, lease.job_id),
            )
            return _job_from_row(self._require_job(conn, lease.job_id))

    def record_sentinel(self, lease: Lease, kind: str, detail: Mapping[str, Any]) -> None:
        """A sentinel observation (measured, drift, failed) as an event of the job, by the lease holder.

        It changes no state and no counter: the job stays as it is, and the event is append-only.
        """
        if kind not in SENTINEL_EVENT_KINDS:
            raise ValueError(f"{kind!r} is not a sentinel event kind {SENTINEL_EVENT_KINDS}")
        with self._store._tx() as conn:
            row = self._require_lease(conn, lease)
            self._event(
                conn,
                lease.job_id,
                kind,
                row["state"],
                row["state"],
                lease_token=lease.lease_token,
                worker_id=lease.worker_id,
                detail=to_jsonable(dict(detail), _path="$.detail"),
            )

    def sentinel_events(self, run_id: str) -> list[dict[str, Any]]:
        """The sentinel events of every job of a run, oldest first (what a resumed job starts from)."""
        marks = ", ".join("?" for _ in SENTINEL_EVENT_KINDS)
        with self._store._snapshot() as conn:
            rows = conn.execute(
                f"SELECT e.* FROM job_events e JOIN jobs j ON j.job_id = e.job_id WHERE j.run_id = ? "
                f"AND e.kind IN ({marks}) ORDER BY e.event_id",
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

    def is_cancel_requested(self, lease: Lease) -> bool:
        """Cheap read for the cooperative cancel check; raises if the lease is gone."""
        with self._store._snapshot() as conn:
            return self._require_lease(conn, lease)["cancel_requested_at"] is not None

    def release(self, lease: Lease, *, reason: str) -> JobRecord:
        """Planned stop: back to `queued` so another session resumes it later."""
        return self._end(lease, "queued", "released", reason)

    def finish(self, lease: Lease, state: TerminalJobState, *, detail: str) -> JobRecord:
        """Terminal state by the lease holder. `cancelled` requires a prior cancel request."""
        if state not in JOB_TERMINAL_STATES:
            raise ValueError(f"{state!r} is not a terminal job state")
        return self._end(lease, state, state, detail)

    def _end(self, lease: Lease, to_state: str, kind: str, detail: str) -> JobRecord:
        with self._store._tx() as conn:
            row = self._require_lease(conn, lease)
            if to_state == "cancelled" and row["cancel_requested_at"] is None:
                raise JobStateError("a job is cancelled only after request_cancel")
            running_attempts = conn.execute(
                "SELECT attempt_id FROM evaluation_attempts WHERE run_id = ? AND status = 'running'",
                (lease.run_id,),
            ).fetchall()
            if running_attempts:
                raise JobStateError(
                    f"job {lease.job_id!r} still has running attempts {[r[0] for r in running_attempts]}"
                )
            stamp = self._store._now()
            conn.execute(
                "UPDATE jobs SET state = ?, lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL, "
                "final_detail = ?, updated_at = ?, finished_at = ? WHERE job_id = ?",
                (to_state, detail, stamp, None if to_state == "queued" else stamp, lease.job_id),
            )
            self._event(
                conn,
                lease.job_id,
                kind,
                "running",
                to_state,
                lease_token=lease.lease_token,
                worker_id=lease.worker_id,
                detail={"detail": detail},
            )
            return _job_from_row(self._require_job(conn, lease.job_id))

    # -- operator side ---------------------------------------------------

    def request_cancel(self, job_id: str, *, reason: str) -> JobRecord:
        """Cooperative cancel. Queued/interrupted jobs end now; running ones at the next step.

        Never touches a lease or an attempt: the worker finishes the current
        attempt, commits it, and ends the job itself.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("a cancel needs a reason")
        with self._store._tx() as conn:
            row = self._require_job(conn, job_id)
            state = row["state"]
            if state in JOB_TERMINAL_STATES:
                if state == "cancelled":
                    return _job_from_row(row)
                raise JobStateError(f"job {job_id!r} is already {state}")
            if row["cancel_requested_at"] is not None:
                return _job_from_row(row)
            stamp = self._store._now()
            if state == "running":
                conn.execute(
                    "UPDATE jobs SET cancel_requested_at = ?, cancel_reason = ?, updated_at = ? WHERE job_id = ?",
                    (stamp, reason, stamp, job_id),
                )
                self._event(conn, job_id, "cancel_requested", state, state, detail={"reason": reason})
            else:
                conn.execute(
                    "UPDATE jobs SET state = 'cancelled', cancel_requested_at = ?, cancel_reason = ?, "
                    "final_detail = ?, updated_at = ?, finished_at = ? WHERE job_id = ?",
                    (stamp, reason, reason, stamp, stamp, job_id),
                )
                self._event(conn, job_id, "cancelled", state, "cancelled", detail={"reason": reason})
            return _job_from_row(self._require_job(conn, job_id))

    # -- queries ---------------------------------------------------------

    def get(self, job_id: str) -> JobRecord:
        with self._store._snapshot() as conn:
            return _job_from_row(self._require_job(conn, job_id))

    def list_jobs(self, states: Iterable[str] | None = None) -> tuple[JobRecord, ...]:
        with self._store._snapshot() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY queue_seq").fetchall()
        wanted = None if states is None else set(states)
        return tuple(_job_from_row(r) for r in rows if wanted is None or r["state"] in wanted)

    def events(self, job_id: str) -> tuple[JobEvent, ...]:
        with self._store._snapshot() as conn:
            self._require_job(conn, job_id)
            rows = conn.execute("SELECT * FROM job_events WHERE job_id = ? ORDER BY event_id", (job_id,)).fetchall()
        return tuple(
            JobEvent(
                event_id=r["event_id"],
                job_id=r["job_id"],
                kind=r["kind"],
                from_state=r["from_state"],
                to_state=r["to_state"],
                lease_token=r["lease_token"],
                worker_id=r["worker_id"],
                detail=json.loads(r["detail_json"]),
                recorded_at=r["recorded_at"],
            )
            for r in rows
        )

    def job_gpu_seconds(self, job_id: str) -> float:
        """Monotonic GPU time measured by this job's lease holders (all sessions)."""
        with self._store._snapshot() as conn:
            self._require_job(conn, job_id)
            return float(
                conn.execute(
                    "SELECT COALESCE(SUM(s.gpu_seconds), 0.0) FROM job_gpu_segments j "
                    "JOIN gpu_segments s ON s.segment_id = j.segment_id WHERE j.job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )

    def job_attempt_ids(self, job_id: str) -> tuple[str, ...]:
        with self._store._snapshot() as conn:
            self._require_job(conn, job_id)
            return tuple(
                r[0]
                for r in conn.execute(
                    "SELECT a.attempt_id FROM job_attempts a JOIN evaluation_attempts e "
                    "ON e.attempt_id = a.attempt_id WHERE a.job_id = ? ORDER BY e.rowid",
                    (job_id,),
                )
            )
