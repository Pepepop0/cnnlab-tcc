"""Testes da fila persistente do worker."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cnnlab.config import ClassicJob, GAJob, JobSpec
from cnnlab.worker import queue as queue_module
from cnnlab.worker.queue import JobQueue


@pytest.fixture
def queue_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "queue.json"
    monkeypatch.setattr(queue_module, "QUEUE_FILE", path)
    return path


def test_adds_three_jobs_and_consumes_them_in_order(queue_file: Path) -> None:
    queue = JobQueue()
    specs: list[JobSpec] = [
        ClassicJob(note="first"),
        GAJob(note="second"),
        ClassicJob(note="third"),
    ]

    added = queue.add_many(specs)

    consumed_ids: list[str] = []
    while job := queue.next_pending():
        consumed_ids.append(job.id)
        queue.mark(job.id, "done", run_id=f"run-{len(consumed_ids)}")

    assert consumed_ids == [job.id for job in added]
    assert queue.pending_count() == 0
    assert queue_file.exists()


def test_marking_one_job_failed_does_not_affect_the_others(queue_file: Path) -> None:
    queue = JobQueue()
    first, second, third = queue.add_many([ClassicJob(note="first"), ClassicJob(note="second"), GAJob(note="third")])

    queue.mark(first.id, "failed", run_id="failed-run")

    assert queue.pending_count() == 2
    assert queue.next_pending() == second
    queue.mark(second.id, "done")
    assert queue.next_pending() == third


def test_load_recovers_running_job_as_pending(queue_file: Path) -> None:
    original = JobQueue()
    running = original.add(ClassicJob(note="interrupted"))
    original.mark(running.id, "running", run_id="orphaned-run")

    recovered = JobQueue()
    recovered.load()

    job = recovered.next_pending()
    assert job is not None
    assert job.id == running.id
    assert job.status == "pending"
    assert job.run_id == "orphaned-run"

    persisted = json.loads(queue_file.read_text(encoding="utf-8"))
    assert persisted["jobs"][0]["status"] == "pending"


def test_load_preserves_job_spec_types(queue_file: Path) -> None:
    queue = JobQueue()
    queue.add_many([ClassicJob(note="classic"), GAJob(note="genetic")])

    loaded = JobQueue()
    loaded.load()
    first = loaded.next_pending()
    assert first is not None
    assert isinstance(first.spec, ClassicJob)
    loaded.mark(first.id, "done")
    second = loaded.next_pending()
    assert second is not None
    assert isinstance(second.spec, GAJob)


def test_clear_finished_keeps_pending_and_running_jobs(queue_file: Path) -> None:
    queue = JobQueue()
    done, failed, cancelled, pending, running = queue.add_many([ClassicJob(note=str(index)) for index in range(5)])
    queue.mark(done.id, "done")
    queue.mark(failed.id, "failed")
    queue.mark(cancelled.id, "cancelled")
    queue.mark(running.id, "running")

    queue.clear_finished()

    assert queue.pending_count() == 1
    assert queue.next_pending() == pending
    payload = json.loads(queue_file.read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["jobs"]] == [pending.id, running.id]


def test_mark_rejects_unknown_job_and_status(queue_file: Path) -> None:
    queue = JobQueue()
    job = queue.add(ClassicJob())

    with pytest.raises(ValueError):
        queue.mark(job.id, "unknown")
    with pytest.raises(KeyError):
        queue.mark("missing", "done")


def test_concurrent_adds_preserve_every_job(queue_file: Path) -> None:
    queue = JobQueue()
    workers = 16
    barrier = threading.Barrier(workers)

    def add(index: int) -> None:
        barrier.wait()
        queue.add(ClassicJob(note=f"job-{index}"))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(add, index) for index in range(workers)]
        for future in futures:
            future.result()

    assert queue.pending_count() == workers
    reloaded = JobQueue()
    reloaded.load()
    assert reloaded.pending_count() == workers
