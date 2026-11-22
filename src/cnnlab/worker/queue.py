"""Fila persistente de jobs executados pelo worker."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from cnnlab.config import JobSpec, JobSpecEnvelope
from cnnlab.storage.layout import QUEUE_FILE, atomic_write_json

_VALID_STATUSES = frozenset({"pending", "running", "done", "failed", "cancelled"})
_FINISHED_STATUSES = frozenset({"done", "failed", "cancelled"})


@dataclass
class Job:
    id: str
    spec: JobSpec
    status: str
    run_id: str | None = None


class JobQueue:
    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._lock = threading.RLock()

    def load(self) -> None:
        """Carrega a fila e recupera jobs interrompidos por um crash."""
        with self._lock:
            if not QUEUE_FILE.exists():
                self._jobs = []
                return

            with QUEUE_FILE.open(encoding="utf-8") as queue_file:
                payload = json.load(queue_file)

            jobs = [self._deserialize_job(item) for item in payload["jobs"]]
            recovered = False
            for job in jobs:
                if job.status == "running":
                    job.status = "pending"
                    recovered = True

            self._jobs = jobs
            if recovered:
                self._save()

    def add(self, spec: JobSpec) -> Job:
        with self._lock:
            job = Job(id=uuid4().hex, spec=spec, status="pending")
            self._jobs.append(job)
            self._save()
            return job

    def add_many(self, specs: list[JobSpec]) -> list[Job]:
        with self._lock:
            jobs = [Job(id=uuid4().hex, spec=spec, status="pending") for spec in specs]
            if jobs:
                self._jobs.extend(jobs)
                self._save()
            return jobs

    def next_pending(self) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs if job.status == "pending"), None)

    def mark(self, job_id: str, status: str, run_id: str | None = None) -> None:
        with self._lock:
            if status not in _VALID_STATUSES:
                raise ValueError(f"invalid job status: {status!r}")

            for job in self._jobs:
                if job.id == job_id:
                    job.status = status
                    job.run_id = run_id
                    self._save()
                    return

            raise KeyError(job_id)

    def pending_count(self) -> int:
        with self._lock:
            return sum(job.status == "pending" for job in self._jobs)

    def clear_finished(self) -> None:
        with self._lock:
            remaining = [job for job in self._jobs if job.status not in _FINISHED_STATUSES]
            if len(remaining) != len(self._jobs):
                self._jobs = remaining
                self._save()

    def _save(self) -> None:
        with self._lock:
            atomic_write_json(
                QUEUE_FILE,
                {"jobs": [self._serialize_job(job) for job in self._jobs]},
            )

    @staticmethod
    def _serialize_job(job: Job) -> dict[str, Any]:
        spec = JobSpecEnvelope(spec=job.spec).model_dump(mode="json")["spec"]
        return {
            "id": job.id,
            "spec": spec,
            "status": job.status,
            "run_id": job.run_id,
        }

    @staticmethod
    def _deserialize_job(payload: dict[str, Any]) -> Job:
        status = payload["status"]
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid job status: {status!r}")

        spec = JobSpecEnvelope.model_validate({"spec": payload["spec"]}).spec
        return Job(
            id=payload["id"],
            spec=spec,
            status=status,
            run_id=payload.get("run_id"),
        )
